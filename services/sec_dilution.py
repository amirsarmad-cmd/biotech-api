"""sec_dilution — narrative-level SEC parser for dilution capacity.

Closes ChatGPT pass-3 critique #1: "ATM / shelf / warrants are still missing.
The SEC XBRL parser extracts cash, debt, shares, burn, runway — excellent.
But ATM facilities, shelf registrations, warrants, convertibles with reset
terms, and recent financing capacity usually live in narrative filings."

For micro-cap biotechs, the existing pile of unexercised warrants + remaining
shelf capacity often matters more than current shares outstanding for
predicting realized dilution.

Strategy:
  1. EDGAR submissions endpoint → list of recent filings per CIK
  2. Filter to forms relevant to dilution capacity:
        S-3, S-3/A, 424B5  → shelf registration / takedown
        S-1, S-1/A         → IPO or follow-on
        8-K                → financing announcements (need to scan content)
  3. Fetch each document HTML, extract sections matching dilution patterns
  4. LLM extraction of: ATM facility size, shelf available, warrant counts +
     strikes + expirations, convertibles, recent issuances

This is the LLM-grounded parser path (per ChatGPT's source-precedence rule:
SEC narrative > LLM inference). The LLM sees actual filing text and just
extracts structured facts; it doesn't fabricate.
"""
import os
import re
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

EDGAR_DATA = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = os.getenv("SEC_USER_AGENT", "Biotech Screener research@biotech-screener.app")

# Forms relevant to dilution capacity, in priority order
DILUTION_FORMS = {
    "S-3", "S-3/A",       # shelf registration
    "424B5", "424B3",     # shelf takedown / prospectus supplement (actual ATM raise)
    "S-1", "S-1/A",       # IPO / follow-on
    "8-K",                # one-off announcements (filtered by content)
    "10-Q", "10-K",       # footnotes contain warrant tables, ATM disclosures
}

# Regex patterns for ATM / shelf / warrant sections
DILUTION_PATTERNS = [
    # ATM offerings
    re.compile(r"at[\s-]+the[\s-]+market.{0,200}?(\$[\d,.]+ million|\$[\d,.]+ billion|aggregate.{0,40}?(\$[\d,.]+))",
               re.IGNORECASE | re.DOTALL),
    # Shelf
    re.compile(r"shelf registration.{0,200}?(\$[\d,.]+ million|\$[\d,.]+ billion|aggregate.{0,40}?(\$[\d,.]+))",
               re.IGNORECASE | re.DOTALL),
    # Warrants
    re.compile(r"(\d{1,3}(?:,\d{3})*)\s+warrants.{0,200}?(?:exercise price|exercisable).{0,80}?\$([\d.]+)",
               re.IGNORECASE | re.DOTALL),
    # Convertibles
    re.compile(r"convertible (?:notes?|debt).{0,200}?(\$[\d,.]+ million|\$[\d,.]+ billion)",
               re.IGNORECASE | re.DOTALL),
]


def _redis():
    try:
        from services.cache import get_redis
        return get_redis()
    except Exception:
        return None


def _cached_get(key):
    r = _redis()
    if r is None: return None
    try:
        raw = r.get(key)
        if raw: return json.loads(raw)
    except Exception:
        pass
    return None


def _cached_set(key, val, ttl_sec=86400):
    r = _redis()
    if r is None: return
    try:
        r.setex(key, ttl_sec, json.dumps(val, default=str))
    except Exception:
        pass


def _http_get(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch from SEC. Returns response text or None."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json,*/*",
        })
        if resp.status_code in (404, 403):
            return None
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.info(f"SEC narrative fetch failed {url}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Filing discovery
# ────────────────────────────────────────────────────────────

def _list_recent_filings(cik: str, max_filings: int = 30) -> List[Dict]:
    """List recent filings via EDGAR submissions endpoint."""
    cache_key = f"sec:submissions:{cik}"
    cached = _cached_get(cache_key)
    if cached:
        return cached[:max_filings]

    url = f"{EDGAR_DATA}/submissions/CIK{cik}.json"
    text = _http_get(url, timeout=15)
    if not text:
        return []
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return []

    recent = d.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    acc = recent.get("accessionNumber", []) or []
    primary = recent.get("primaryDocument", []) or []

    out = []
    for i, f in enumerate(forms):
        if i >= len(dates) or i >= len(acc) or i >= len(primary):
            break
        if f not in DILUTION_FORMS:
            continue
        out.append({
            "form": f,
            "filing_date": dates[i],
            "accession_no": acc[i],
            "primary_doc": primary[i],
        })
        if len(out) >= max_filings * 3:  # gather extra, filter later
            break
    _cached_set(cache_key, out, ttl_sec=86400)
    return out[:max_filings]


def _filing_url(cik: str, accession_no: str, primary_doc: str) -> str:
    """Build the filing URL. Accession needs dashes removed for path."""
    cik_int = str(int(cik))  # strip leading zeros
    acc_clean = accession_no.replace("-", "")
    return f"{EDGAR_ARCHIVES}/{cik_int}/{acc_clean}/{primary_doc}"


# ────────────────────────────────────────────────────────────
# Content extraction
# ────────────────────────────────────────────────────────────

def _extract_relevant_sections(html: str, max_chars: int = 25000) -> str:
    """Strip HTML tags and extract sections likely to contain dilution language.
    Returns a concatenated text blob bounded by max_chars.
    """
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for sel in ["script", "style", "nav", "header", "footer"]:
            for el in soup.select(sel):
                el.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

    # Keyword search — keep only paragraphs that mention dilution-relevant terms
    keywords = re.compile(
        r"(at[\s-]?the[\s-]?market|shelf registration|"
        r"warrant|convertible|equity line|"
        r"prospectus supplement|public offering|registered direct|"
        r"private placement|rights offering|"
        r"shares.{0,30}authorized|shares.{0,30}reserved|"
        r"PIPE financing|series [A-Z] preferred)",
        re.IGNORECASE,
    )
    paragraphs = text.split("\n")
    kept = []
    char_count = 0
    for p in paragraphs:
        if not p or len(p) < 30:
            continue
        if keywords.search(p):
            kept.append(p)
            char_count += len(p)
            if char_count >= max_chars:
                break
    if not kept:
        return text[:max_chars]
    return "\n\n".join(kept)


# ────────────────────────────────────────────────────────────
# LLM extraction
# ────────────────────────────────────────────────────────────

def _llm_extract_dilution_facts(filing_meta: Dict, content: str) -> Dict:
    """Send extracted filing text to LLM for structured dilution facts."""
    if not content or len(content) < 100:
        return {"_status": "empty_content"}

    prompt = f"""Extract dilution-related facts from this SEC filing excerpt. Return ONLY a JSON object — no preamble.

FILING: {filing_meta.get('form')} dated {filing_meta.get('filing_date')}
ACCESSION: {filing_meta.get('accession_no')}

EXCERPT:
{content[:18000]}
---

Schema (use null where data isn't present):
{{
  "atm_facility": {{
    "exists": true | false,
    "aggregate_amount_usd": <integer USD or null>,    // total ATM size, e.g. 200000000 for $200M
    "amount_remaining_usd": <integer USD or null>,    // unused capacity, if disclosed
    "agent": "<sales agent name>",
    "established_date": "<YYYY-MM-DD or null>",
    "_quote": "<short ≤150 char direct quote — the basis for these facts>"
  }} | null,
  "shelf_registration": {{
    "exists": true | false,
    "aggregate_amount_usd": <integer USD or null>,
    "amount_remaining_usd": <integer USD or null>,
    "filed_date": "<YYYY-MM-DD or null>",
    "expiration_date": "<YYYY-MM-DD or null>",
    "_quote": "<short ≤150 char>"
  }} | null,
  "warrants": [
    {{
      "count": <integer shares>,
      "exercise_price_usd": <number>,
      "expiration_date": "<YYYY-MM-DD or null>",
      "category": "<warrant series/class>",
      "_quote": "<short ≤120 char>"
    }}
  ],
  "convertible_notes": [
    {{
      "principal_usd": <integer>,
      "conversion_price_usd": <number or null>,
      "maturity_date": "<YYYY-MM-DD or null>",
      "interest_rate_pct": <number or null>,
      "_quote": "<short ≤120 char>"
    }}
  ],
  "recent_issuance": {{
    "shares_issued": <integer or null>,
    "price_per_share_usd": <number or null>,
    "gross_proceeds_usd": <integer or null>,
    "type": "<at-the-market | registered direct | PIPE | follow-on | rights offering | other>",
    "_quote": "<short ≤150 char>"
  }} | null,
  "_filing_summary": "<1-2 sentence summary of the filing's dilution implications>",
  "_extraction_quality": "<high | medium | low — how confident you are these facts are accurate from the text>"
}}

RULES:
- Return facts ONLY if they're explicitly in the excerpt. Don't infer.
- Use null when the field isn't present rather than guessing.
- _quote fields anchor each fact to source text — keep them short and verbatim.
- If the filing is unrelated to dilution (e.g., a routine officer change), return atm_facility=null, shelf_registration=null, warrants=[], etc.
"""
    try:
        from services.llm_helper import call_llm_json
        result, err = call_llm_json(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
            feature="sec_dilution_extract",
        )
        if result is None:
            return {"_status": "llm_failed", "_error": err}
        return result
    except Exception as e:
        logger.info(f"LLM dilution extract failed: {e}")
        return {"_status": "llm_exception", "_error": str(e)[:120]}


# ────────────────────────────────────────────────────────────
# Top-level: fetch_dilution_capacity
# ────────────────────────────────────────────────────────────

def fetch_dilution_capacity(ticker: str, max_filings_to_parse: int = 5,
                              lookback_days: int = 540) -> Optional[Dict]:
    """Pull ATM/shelf/warrant/convertible disclosures from recent SEC filings.

    Strategy:
      1. ticker → CIK
      2. Pull recent submissions (last ~18 months by default)
      3. Filter to dilution-relevant forms in priority order
      4. Fetch + LLM-extract each filing
      5. Aggregate the facts: pick the MOST RECENT non-null for each capacity
         (e.g., latest S-3 wins for shelf size; latest 424B5 wins for actual
         ATM activity)

    Returns:
      {
        "ticker": str,
        "cik": str,
        "filings_inspected": [{form, date, accession, summary, ...}],
        "atm_facility": {...} | None,         // most recent
        "shelf_registration": {...} | None,
        "active_warrants": [...],             // all warrants from most recent
        "active_convertibles": [...],
        "recent_issuances": [...],            // last 18mo issuances
        "estimated_dilution_capacity_usd": int,
        "estimated_max_dilution_pct": float,  // if shelf + ATM fully used
        "_provenance": "sec_edgar_narrative",
        "_warnings": [str],
      }
    """
    from services.sec_financials import ticker_to_cik
    cik = ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "_error": "ticker not in SEC"}

    cache_key = f"sec:dilution:{cik}:{max_filings_to_parse}"
    cached = _cached_get(cache_key)
    if cached:
        cached["_from_cache"] = True
        return cached

    cutoff_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    all_filings = _list_recent_filings(cik, max_filings=30)
    relevant = [f for f in all_filings if f["filing_date"] >= cutoff_date]
    # Prioritize: S-3 / 424B5 first, then 8-K, then 10-Q/K
    priority = {"S-3": 0, "S-3/A": 0, "424B5": 1, "424B3": 1,
                "S-1": 2, "S-1/A": 2, "8-K": 3, "10-Q": 4, "10-K": 5}
    relevant.sort(key=lambda f: (priority.get(f["form"], 9), f["filing_date"]), reverse=False)
    # Most recent of each category — sort by date descending within priority bucket
    relevant.sort(key=lambda f: (priority.get(f["form"], 9), -int(f["filing_date"].replace("-", ""))))
    relevant = relevant[:max_filings_to_parse]

    inspected = []
    aggregate = {
        "atm_facility": None,
        "shelf_registration": None,
        "active_warrants": [],
        "active_convertibles": [],
        "recent_issuances": [],
    }

    for f in relevant:
        try:
            url = _filing_url(cik, f["accession_no"], f["primary_doc"])
            html = _http_get(url, timeout=20)
            if not html:
                inspected.append({**f, "_status": "fetch_failed", "url": url})
                continue
            content = _extract_relevant_sections(html, max_chars=20000)
            if len(content) < 200:
                inspected.append({**f, "_status": "no_relevant_content", "url": url})
                continue
            facts = _llm_extract_dilution_facts(f, content)
            inspected.append({**f, "url": url,
                              "summary": facts.get("_filing_summary"),
                              "extraction_quality": facts.get("_extraction_quality")})
            # Aggregate — most recent non-null wins
            if facts.get("atm_facility") and facts["atm_facility"].get("exists"):
                if not aggregate["atm_facility"] or f["filing_date"] > aggregate["atm_facility"].get("_filing_date", ""):
                    aggregate["atm_facility"] = {**facts["atm_facility"],
                                                  "_filing_date": f["filing_date"],
                                                  "_filing_form": f["form"]}
            if facts.get("shelf_registration") and facts["shelf_registration"].get("exists"):
                if not aggregate["shelf_registration"] or f["filing_date"] > aggregate["shelf_registration"].get("_filing_date", ""):
                    aggregate["shelf_registration"] = {**facts["shelf_registration"],
                                                        "_filing_date": f["filing_date"],
                                                        "_filing_form": f["form"]}
            for w in (facts.get("warrants") or []):
                if w and w.get("count"):
                    w["_filing_date"] = f["filing_date"]
                    aggregate["active_warrants"].append(w)
            for c in (facts.get("convertible_notes") or []):
                if c and c.get("principal_usd"):
                    c["_filing_date"] = f["filing_date"]
                    aggregate["active_convertibles"].append(c)
            if facts.get("recent_issuance"):
                ri = facts["recent_issuance"]
                ri["_filing_date"] = f["filing_date"]
                aggregate["recent_issuances"].append(ri)
        except Exception as e:
            inspected.append({**f, "_status": "exception", "_error": str(e)[:100]})

    # Compute estimated dilution capacity
    atm_remaining = (aggregate["atm_facility"] or {}).get("amount_remaining_usd") or \
                    (aggregate["atm_facility"] or {}).get("aggregate_amount_usd") or 0
    shelf_remaining = (aggregate["shelf_registration"] or {}).get("amount_remaining_usd") or \
                      (aggregate["shelf_registration"] or {}).get("aggregate_amount_usd") or 0
    warrant_value = sum(
        (w.get("count") or 0) * (w.get("exercise_price_usd") or 0)
        for w in aggregate["active_warrants"]
    )
    total_capacity = (atm_remaining or 0) + (shelf_remaining or 0) + (warrant_value or 0)

    warnings = []
    if atm_remaining and atm_remaining > 50_000_000:
        warnings.append(
            f"Active ATM facility with ${atm_remaining/1e6:.0f}M remaining — "
            f"company can dilute opportunistically without further announcement."
        )
    if shelf_remaining and shelf_remaining > 100_000_000:
        warnings.append(
            f"Shelf registration with ${shelf_remaining/1e6:.0f}M unused capacity."
        )
    if len(aggregate["active_warrants"]) > 0:
        total_warrants = sum(w.get("count", 0) for w in aggregate["active_warrants"])
        warnings.append(
            f"{total_warrants:,} warrants outstanding from past issuances — "
            f"dilutive on conversion."
        )

    out = {
        "ticker": ticker.upper(),
        "cik": cik,
        "filings_inspected": inspected,
        "atm_facility": aggregate["atm_facility"],
        "shelf_registration": aggregate["shelf_registration"],
        "active_warrants": aggregate["active_warrants"],
        "active_convertibles": aggregate["active_convertibles"],
        "recent_issuances": aggregate["recent_issuances"][:5],
        "estimated_dilution_capacity_usd": total_capacity if total_capacity else None,
        "warnings": warnings,
        "_provenance": "sec_edgar_narrative",
        "_extracted_at": datetime.utcnow().isoformat(),
    }
    # Cache 12 hours — narrative facts change with new filings
    _cached_set(cache_key, out, ttl_sec=12 * 3600)
    return out
