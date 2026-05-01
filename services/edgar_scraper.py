"""EDGAR 8-K backfill scraper — historical biotech catalyst extraction.

User selected: 2015-2025, source priority EDGAR > Finnhub > BPC, one-shot.

ARCHITECTURE NOTE (post-deploy revision):
  Initial design fetched 8-K filing HTML from www.sec.gov/Archives. That
  endpoint returns 403 from Railway IPs (SEC blocks our network range with
  the 'Undeclared Automated Tool' page despite a polite UA). However,
  data.sec.gov works fine (different SEC subdomain, different access policy).

  Revised pipeline uses ONLY data.sec.gov endpoints:
    1. resolve_cik_via_data_sec(ticker) — first lookup hits data.sec.gov
       per-ticker (we don't need the master list).
    2. list_8k_filings(cik, range)      — data.sec.gov/submissions JSON
       gives us {filing_date, accession, items, primary_doc} for every 8-K.
    3. is_catalyst_by_items_code(items) — pre-filter on item codes alone.
       Items 7.01 (Reg FD), 8.01 (Other Events), 1.01 (Material Agreements)
       cover virtually all clinical/regulatory disclosures. Skip 2.02
       (financial results) and 5.x (governance) only filings.
    4. extract_via_gemini_grounded(ticker, date) — Gemini 2.5 Flash with
       Google Search grounding. We feed it (ticker, company, filing_date,
       item_codes) and ask it to find the company's press release for
       that date and extract the catalyst event. This is more robust than
       parsing 8-K legalese — the press releases are structured for humans.

Cost model:
  - data.sec.gov: free, polite limit 10 req/sec
  - LLM extraction: ~$0.0003/call (Gemini Flash + grounding)
  - Estimated 1500-3000 catalyst-relevant 8-Ks across 740 tickers / 10y
  - Total LLM cost: ~$0.50-1.00
"""
from __future__ import annotations
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# EDGAR requires a contact email in User-Agent. data.sec.gov accepts our
# Railway-IP requests with this UA (we tested — the block is www.sec.gov only).
EDGAR_UA = "AEGRA Biotech Research [email protected]"
EDGAR_HEADERS = {
    "User-Agent": EDGAR_UA,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

EDGAR_MIN_INTERVAL_SEC = 0.12  # ~8 req/sec
_last_edgar_call_at = 0.0


def _rate_limit():
    global _last_edgar_call_at
    elapsed = time.time() - _last_edgar_call_at
    if elapsed < EDGAR_MIN_INTERVAL_SEC:
        time.sleep(EDGAR_MIN_INTERVAL_SEC - elapsed)
    _last_edgar_call_at = time.time()


# Module-level CIK cache (populated lazily as tickers are looked up)
_cik_cache: Dict[str, Optional[str]] = {}


def resolve_cik(ticker: str) -> Optional[str]:
    """ticker → 10-digit CIK string. Looks up via data.sec.gov per-ticker
    (we can't use the master list because www.sec.gov is blocked).

    The trick: data.sec.gov/submissions/CIK{cik}.json works only with a
    known CIK. To get from ticker → CIK from this side, we need to use
    the EDGAR full-text search which IS on a third subdomain (efts.sec.gov)
    that may also work.
    """
    ticker_u = (ticker or "").upper().strip()
    if not ticker_u:
        return None
    if ticker_u in _cik_cache:
        return _cik_cache[ticker_u]

    # Strategy 1: efts.sec.gov full-text search (returns CIK in results)
    _rate_limit()
    try:
        # Use EDGAR's company search via efts (different from www)
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker_u}%22&forms=10-K&hits=1"
        r = requests.get(
            url,
            headers={"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                src = h.get("_source", {})
                ciks = src.get("ciks") or []
                tickers = src.get("tickers") or []
                if ticker_u in [t.upper() for t in tickers]:
                    cik = str(ciks[0]).zfill(10) if ciks else None
                    if cik:
                        _cik_cache[ticker_u] = cik
                        return cik
    except Exception as e:
        logger.debug(f"[edgar] efts search failed for {ticker_u}: {e}")

    # Strategy 2: data.sec.gov ticker → CIK via the ticker.txt feed (legacy)
    # SEC publishes a plain text list at https://www.sec.gov/include/ticker.txt
    # but that's also blocked. Try the data.sec.gov canonical alternative:
    # /api/xbrl/companyconcept/CIK{cik}/... — requires CIK already.

    # Strategy 3: Embedded fallback list — populated from a one-time scrape
    # of company_tickers.json. We bundle this with the deploy. See
    # services/edgar_cik_seed.py for the static dict.
    try:
        from services.edgar_cik_seed import CIK_BY_TICKER
        cik = CIK_BY_TICKER.get(ticker_u)
        if cik:
            _cik_cache[ticker_u] = cik
            return cik
    except ImportError:
        pass

    _cik_cache[ticker_u] = None
    return None


def list_8k_filings(
    cik: str,
    *,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List 8-K filings for a CIK in a date range via data.sec.gov.

    Returns list of {accession, filing_date, primary_doc, items} dicts.
    """
    _rate_limit()
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers={
            "User-Agent": EDGAR_UA, "Accept-Encoding": "gzip",
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[edgar] submissions fetch failed for CIK {cik}: {e}")
        return []

    end_date = end_date or date.today().isoformat()
    out: List[Dict[str, Any]] = []

    # Recent filings (latest 1000 — covers ~3-5 years for active filers)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates_arr = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    items_arr = recent.get("items", [])

    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        fdate = dates_arr[i]
        if fdate < start_date or fdate > end_date:
            continue
        out.append({
            "accession": accessions[i],
            "filing_date": fdate,
            "primary_doc": docs[i] if i < len(docs) else None,
            "items": items_arr[i] if i < len(items_arr) else "",
        })

    # Older filings — paginated archives in "files"
    older_files = data.get("filings", {}).get("files", [])
    for old_meta in older_files:
        f_from = old_meta.get("filingFrom", "9999-12-31")
        f_to = old_meta.get("filingTo", "0000-01-01")
        if f_to < start_date or f_from > end_date:
            continue
        try:
            _rate_limit()
            old_url = f"https://data.sec.gov/submissions/{old_meta['name']}"
            r = requests.get(old_url, headers={
                "User-Agent": EDGAR_UA, "Accept-Encoding": "gzip",
            }, timeout=20)
            r.raise_for_status()
            old_data = r.json()
            ofs = old_data.get("form", [])
            oaccs = old_data.get("accessionNumber", [])
            odates = old_data.get("filingDate", [])
            odocs = old_data.get("primaryDocument", [])
            oitems = old_data.get("items", [])
            for i, form in enumerate(ofs):
                if form != "8-K":
                    continue
                fdate = odates[i]
                if fdate < start_date or fdate > end_date:
                    continue
                out.append({
                    "accession": oaccs[i],
                    "filing_date": fdate,
                    "primary_doc": odocs[i] if i < len(odocs) else None,
                    "items": oitems[i] if i < len(oitems) else "",
                })
        except Exception as e:
            logger.warning(f"[edgar] older filings page {old_meta.get('name')} failed: {e}")

    return out


def is_catalyst_by_items_code(items: str) -> Tuple[bool, str]:
    """Filter 8-Ks based on their EDGAR Item codes alone.

    Items 7.01 (Reg FD Disclosure) — most common for clinical/PR
    Items 8.01 (Other Events)      — most common for FDA decisions
    Items 1.01 (Material Agreement) — partnerships, sometimes catalyst-adj
    Items 1.02 (Termination)        — clinical trial terminations
    Items 5.07 (Shareholder Vote)   — usually not a catalyst, skip

    Skip if the filing is ONLY 2.02 (financial results) and/or 5.x or 9.01
    with nothing catalyst-relevant.
    """
    item_set = {it.strip() for it in (items or "").split(",") if it.strip()}
    if not item_set:
        return (False, "no_items")

    catalyst_items = {"7.01", "8.01", "1.01", "1.02"}
    governance_only = {"2.02", "5.02", "5.03", "5.07", "9.01"}

    if item_set & catalyst_items:
        # Has at least one catalyst-relevant item
        matched = ",".join(sorted(item_set & catalyst_items))
        return (True, f"item_match:{matched}")

    if item_set <= governance_only:
        return (False, "governance_only")

    # Other items — uncertain; let it through
    return (True, "other_items")


# ────────────────────────────────────────────────────────────
# LLM extraction via Gemini grounding (no 8-K text needed)
# ────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a biotech catalyst extractor. A company filed an 8-K on the date below. Search for the company's press release published on or near that date and extract the catalyst event it announces.

Company: {company} ({ticker})
Filing date: {filing_date}
8-K Item codes: {items}

Search Google for:
  - Company press release on {filing_date} (or 1-2 days before/after)
  - PR Newswire / GlobeNewswire / BusinessWire announcement
  - Company IR site investor news
  - The 8-K filing itself on EDGAR

Return ONLY a JSON object (no markdown, no prose). If the filing has no clinical/regulatory catalyst event (e.g., financial results, governance), return {{"is_catalyst": false}}.

Schema:
{{
  "is_catalyst": true,
  "catalyst_type": "FDA Decision" | "Phase 1 Readout" | "Phase 2 Readout" | "Phase 3 Readout" | "Phase 1/2 Readout" | "AdComm" | "NDA Submission" | "BLA Submission" | "Designation Granted" | "Clinical Hold" | "Partnership" | "Other",
  "drug_name": "string or null",
  "indication": "string or null",
  "catalyst_date_iso": "YYYY-MM-DD",
  "outcome_class": "APPROVED" | "REJECTED" | "MET_ENDPOINT" | "MISSED_ENDPOINT" | "DELAYED" | "WITHDRAWN" | "MIXED" | "ANNOUNCED",
  "endpoint_met": true | false | null,
  "approval_granted": true | false | null,
  "evidence": "verbatim short quote from source (under 50 words)",
  "primary_source_url": "https://...",
  "confidence": 0.0 - 1.0
}}

Rules:
  - Extract ONE primary event per filing. If multiple, pick the most material.
  - "ANNOUNCED" is for forward-looking events (NDA filed, no decision yet).
  - confidence < 0.5 = borderline (e.g., business update mentioning clinical work in passing).
  - If is_catalyst=false, return ONLY {{"is_catalyst": false}}."""


def extract_catalyst_via_gemini(
    *,
    ticker: str,
    company: str,
    filing_date: str,
    items: str,
) -> Optional[Dict[str, Any]]:
    """Gemini-grounded extraction. Returns parsed JSON or None.
    Cost ~$0.0003 per call."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("[edgar-llm] GOOGLE_API_KEY not set")
        return None
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("[edgar-llm] google-genai not available")
        return None

    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker, company=company or ticker,
        filing_date=filing_date, items=items or "(none)",
    )
    client = genai.Client(api_key=google_api_key)
    config = genai_types.GenerateContentConfig(
        max_output_tokens=2000,
        temperature=0.1,
        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
    )

    t0 = time.time()
    status = "success"
    err_msg = None
    tokens_in = tokens_out = 0
    response_text = ""

    saw_503 = False
    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model=os.getenv("EDGAR_LLM_MODEL", "gemini-2.5-flash"),
                contents=prompt, config=config,
            )
            response_text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens_in = getattr(usage, "prompt_token_count", 0) or 0
                tokens_out = getattr(usage, "candidates_token_count", 0) or 0
            err_msg = None
            status = "success"
            break
        except Exception as e:
            status = "error"
            err_msg = str(e)[:300]
            if "503" in err_msg and "UNAVAILABLE" in err_msg:
                saw_503 = True
            if attempt == 1 and saw_503:
                time.sleep(3.0)
                continue
            break

    elapsed = time.time() - t0
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google",
            model=os.getenv("EDGAR_LLM_MODEL", "gemini-2.5-flash"),
            feature="edgar_scraper",
            ticker=ticker,
            tokens_input=tokens_in, tokens_output=tokens_out,
            duration_ms=int(elapsed * 1000),
            status=status, error_message=err_msg,
        )
    except Exception:
        pass

    if status != "success" or not response_text:
        return None

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        import json as _json
        parsed = _json.loads(cleaned)
    except Exception as e:
        logger.warning(f"[edgar-llm] JSON parse failed for {ticker}/{filing_date}: {e}")
        return None

    return parsed
