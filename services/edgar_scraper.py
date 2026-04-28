"""EDGAR 8-K backfill scraper — historical biotech catalyst extraction.

User selected: 2015-2025, source priority EDGAR > Finnhub > BPC, one-shot.

This module fetches 8-K filings for our 740-ticker biotech universe and
identifies clinical / regulatory catalysts. The pipeline:

  1. resolve_cik(ticker)          — ticker → CIK lookup via SEC company_tickers
  2. list_8k_filings(cik, range)  — submissions JSON, filter to 8-K
  3. fetch_filing_text(accession) — get the primary 8-K HTML, extract Items
  4. is_catalyst_8k(text)         — keyword classifier (Phase / topline /
                                     PDUFA / FDA / approval / CRL)
  5. extract_event(text)          — LLM extraction of structured event:
                                     {catalyst_type, drug_name, indication,
                                      outcome_class, evidence, source_url}
  6. write to catalyst_universe + post_catalyst_outcomes

Cost model:
  - EDGAR is free, no rate limit beyond 10 req/sec polite limit
  - LLM extraction: ~0.0003/call (Gemini Flash)
  - Estimated 4000-5000 catalyst-relevant 8-Ks across 740 tickers / 10y
  - Total LLM cost: ~$1.50

Scope notes:
  - We keyword-prefilter to avoid LLM-extracting every 8-K (most are
    earnings / dividends / management changes). Only ~10-15% of biotech
    8-Ks are catalyst-relevant.
  - Item codes 7.01 (Reg FD) and 8.01 (Other Events) cover most clinical /
    regulatory disclosures. Item 2.02 (Financial Results) is excluded.
"""
from __future__ import annotations
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# EDGAR requires a contact email in User-Agent. Use the AEGRA project owner
# (per the brief) so SEC can contact us if our crawler misbehaves.
EDGAR_UA = "AEGRA Biotech Research [email protected]"
EDGAR_HEADERS = {
    "User-Agent": EDGAR_UA,
    "Accept-Encoding": "gzip, deflate",
}

# Polite rate limit: SEC asks for max 10 req/sec
EDGAR_MIN_INTERVAL_SEC = 0.12  # ~8 req/sec to leave headroom

# In-process state for rate limiting
_last_edgar_call_at = 0.0


def _rate_limit():
    global _last_edgar_call_at
    elapsed = time.time() - _last_edgar_call_at
    if elapsed < EDGAR_MIN_INTERVAL_SEC:
        time.sleep(EDGAR_MIN_INTERVAL_SEC - elapsed)
    _last_edgar_call_at = time.time()


# Module-level CIK lookup cache (rebuilt on first use)
_cik_lookup_cache: Optional[Dict[str, str]] = None


def _build_cik_lookup() -> Dict[str, str]:
    """SEC publishes a master company_tickers.json with ticker → CIK."""
    global _cik_lookup_cache
    if _cik_lookup_cache is not None:
        return _cik_lookup_cache
    _rate_limit()
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        # Format: {"0": {"cik_str": 1234, "ticker": "ABC", "title": "..."}, ...}
        lookup = {}
        for v in data.values():
            ticker = (v.get("ticker") or "").upper().strip()
            cik = v.get("cik_str")
            if ticker and cik:
                lookup[ticker] = str(cik).zfill(10)
        _cik_lookup_cache = lookup
        logger.info(f"[edgar] built CIK lookup with {len(lookup)} tickers")
        return lookup
    except Exception as e:
        logger.exception(f"[edgar] CIK lookup build failed: {e}")
        return {}


def resolve_cik(ticker: str) -> Optional[str]:
    """ticker → 10-digit CIK string. Returns None if not found."""
    lookup = _build_cik_lookup()
    return lookup.get((ticker or "").upper().strip())


def list_8k_filings(
    cik: str,
    *,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List 8-K filings for a CIK in a date range. Uses
    data.sec.gov/submissions/CIK{cik}.json.

    Returns list of {accession, filing_date, primary_doc, items} dicts.
    """
    _rate_limit()
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"[edgar] submissions fetch failed for CIK {cik}: {e}")
        return []

    end_date = end_date or date.today().isoformat()

    out: List[Dict[str, Any]] = []
    # Recent filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates_arr = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    items_arr = recent.get("items", [])  # comma-separated string per filing

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

    # Older filings live in "files" list of paginated archives
    older_files = data.get("filings", {}).get("files", [])
    for old_meta in older_files:
        # Each entry has {"name": "CIK0001234567-submissions-001.json", ...}
        # Date range is in old_meta["filingFrom"] / "filingTo"
        f_from = old_meta.get("filingFrom", "9999-12-31")
        f_to = old_meta.get("filingTo", "0000-01-01")
        if f_to < start_date or f_from > end_date:
            continue
        try:
            _rate_limit()
            old_url = f"https://data.sec.gov/submissions/{old_meta['name']}"
            r = requests.get(old_url, headers=EDGAR_HEADERS, timeout=20)
            r.raise_for_status()
            old_data = r.json()
            forms = old_data.get("form", [])
            accessions = old_data.get("accessionNumber", [])
            dates_arr = old_data.get("filingDate", [])
            docs = old_data.get("primaryDocument", [])
            items_arr = old_data.get("items", [])
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
        except Exception as e:
            logger.warning(f"[edgar] older filings page {old_meta.get('name')} failed: {e}")

    return out


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """Fetch the primary 8-K document's text. accession comes with hyphens
    in submissions JSON (e.g. '0001193125-26-179401'); the URL needs them
    stripped from the directory part.
    """
    _rate_limit()
    try:
        acc_dir = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_dir}/{primary_doc}"
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return r.text
    except Exception as e:
        logger.warning(f"[edgar] fetch_filing_text failed for {accession}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Catalyst keyword classifier
# ────────────────────────────────────────────────────────────
# 8-K text is HTML — we strip tags then pattern-match. Most biotech 8-Ks
# falling in the catalyst category include one of these phrase clusters.

# Compiled once for performance
_CLINICAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"top-?line\s+(results?|data)",
        r"primary\s+endpoint",
        r"\bphase\s+(?:1|2|3|i|ii|iii)\b",
        r"\bpdufa\b",
        r"\bnda\s+(submission|filing|acceptance|approval)",
        r"\bbla\s+(submission|filing|acceptance|approval)",
        r"FDA\s+(approval|approved|approves|granted|accepted|cleared|grants)",
        r"complete\s+response\s+letter",
        r"\bCRL\b",
        r"breakthrough\s+therapy",
        r"orphan\s+drug",
        r"fast\s+track",
        r"priority\s+review",
        r"advisory\s+committee",
        r"\badcomm?\b",
        r"clinical\s+(hold|trial|study)",
        r"interim\s+(analysis|results)",
    ]
]


def is_catalyst_8k(text: str, items: str = "") -> Tuple[bool, List[str]]:
    """Returns (is_catalyst, matched_keywords). True if any clinical pattern
    matches the text. Items is a comma-separated string from EDGAR; we
    skip pure financial-results filings (Item 2.02 only).
    """
    if not text or len(text) < 500:
        return (False, [])

    # Skip if filing is only Item 2.02 (financial results) without 7.01/8.01
    item_set = {it.strip() for it in (items or "").split(",") if it.strip()}
    if item_set and item_set <= {"2.02", "9.01"}:
        return (False, [])

    # Strip HTML tags for keyword matching
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain)

    matched = []
    for pat in _CLINICAL_PATTERNS:
        m = pat.search(plain)
        if m:
            matched.append(m.group(0).lower())
            if len(matched) >= 3:
                break  # Enough signal; don't waste cycles
    return (len(matched) > 0, matched)


def extract_text_for_llm(html_text: str, max_chars: int = 8000) -> str:
    """Strip HTML and trim to a reasonable size for LLM extraction.
    Catalysts are usually announced in the first 3-5K chars of the 8-K."""
    if not html_text:
        return ""
    plain = re.sub(r"<script[^>]*>.*?</script>", "", html_text,
                   flags=re.DOTALL | re.IGNORECASE)
    plain = re.sub(r"<style[^>]*>.*?</style>", "", plain,
                   flags=re.DOTALL | re.IGNORECASE)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain = re.sub(r"&nbsp;", " ", plain)
    plain = re.sub(r"&amp;", "&", plain)
    plain = re.sub(r"&lt;", "<", plain)
    plain = re.sub(r"&gt;", ">", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) > max_chars:
        plain = plain[:max_chars]
    return plain


# ────────────────────────────────────────────────────────────
# LLM extraction
# ────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a biotech catalyst extractor. Given the text of an 8-K filing, extract any clinical or regulatory catalyst event(s) it announces.

Filing context:
  Ticker: {ticker}
  Filing date: {filing_date}
  CIK: {cik}

Filing text:
---
{text}
---

Return ONLY a JSON object (no markdown, no prose) with this shape. If the filing has no catalyst event, return {{"is_catalyst": false}}.

{{
  "is_catalyst": true,
  "catalyst_type": "FDA Decision" | "Phase 1 Readout" | "Phase 2 Readout" | "Phase 3 Readout" | "Phase 1/2 Readout" | "AdComm" | "NDA Submission" | "BLA Submission" | "Designation Granted" | "Clinical Hold",
  "drug_name": "string or null",
  "indication": "string or null",
  "catalyst_date_iso": "YYYY-MM-DD (the event date, often = filing date)",
  "outcome_class": "APPROVED" | "REJECTED" | "MET_ENDPOINT" | "MISSED_ENDPOINT" | "DELAYED" | "WITHDRAWN" | "MIXED" | "ANNOUNCED",
  "endpoint_met": true | false | null,
  "approval_granted": true | false | null,
  "evidence": "verbatim short quote from the filing, under 50 words",
  "confidence": 0.0 - 1.0
}}

Rules:
  - Extract ONE primary event per filing. If multiple are announced, pick the most material.
  - "ANNOUNCED" outcome_class is for forward-looking events (e.g., NDA filed but no decision yet).
  - Phase trials get "MET_ENDPOINT" / "MISSED_ENDPOINT" only if results are reported, not just announced.
  - confidence < 0.5 means the filing is borderline (e.g., business update touching on clinical work in passing).
  - If is_catalyst=false, return ONLY {{"is_catalyst": false}}."""


def extract_catalyst_via_llm(
    *,
    ticker: str,
    cik: str,
    filing_date: str,
    text: str,
) -> Optional[Dict[str, Any]]:
    """Run Gemini extraction. Returns parsed JSON or None on failure.
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

    plain = extract_text_for_llm(text, max_chars=6000)
    if len(plain) < 200:
        return None

    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker, cik=cik, filing_date=filing_date, text=plain,
    )
    client = genai.Client(api_key=google_api_key)
    config = genai_types.GenerateContentConfig(
        max_output_tokens=2000,
        temperature=0.1,
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
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_sec=elapsed, status=status, error=err_msg,
            context=f"edgar-extract:{ticker}",
        )
    except Exception:
        pass

    if status != "success" or not response_text:
        return None

    # Strip markdown fences if present
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
