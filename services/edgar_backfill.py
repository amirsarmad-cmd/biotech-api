"""SEC EDGAR 8-K backfill scraper for clinical/regulatory catalysts.

Strategy:
  1. Get list of CIKs with biotech/pharma SIC codes (2836, 2834, 8731)
  2. For each CIK, fetch their 8-K filing index from EDGAR
  3. Filter to filings with item codes 7.01 (Reg FD), 8.01 (Other), 5.02
     (changes), or item descriptions matching clinical keywords
  4. For each candidate filing, fetch the cover page (EX-99 typically),
     run regex pre-filter for clinical/FDA keywords, write to staging
  5. A separate LLM-normalize pass (run later) extracts catalyst_type +
     drug + indication + date_precision and decides accept/reject

Rate limiting: SEC EDGAR's stated policy is 10 req/sec max with proper
User-Agent. We use 6 req/sec to stay safe and run with 0.18s sleep
between requests.

This is one-shot only — designed to be invoked once via an admin endpoint
that runs in the background and writes to catalyst_backfill_staging.

References:
  - https://www.sec.gov/os/accessing-edgar-data
  - https://www.sec.gov/cgi-bin/browse-edgar (CIK lookup)
  - https://data.sec.gov/submissions/CIK{cik}.json (filings list)
"""
from __future__ import annotations
import logging
import os
import re
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Per SEC fair-access policy
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "biotech-research-tool research@example.com",
)
EDGAR_REQ_DELAY = float(os.getenv("EDGAR_REQ_DELAY", "0.18"))  # ~5.5 req/sec

# Biotech/pharma SIC codes
BIOTECH_SIC_CODES = ["2834", "2836", "8731"]

# Regex pre-filter for 8-K text — must contain at least one of these tokens
# (case-insensitive) to be considered a candidate. False positives are OK
# (LLM filters them out later). False negatives are NOT OK (we'd miss real
# catalysts), so this is intentionally permissive.
CLINICAL_KEYWORDS_RE = re.compile(
    r"\b("
    r"FDA|PDUFA|approval|approve[sd]?|"
    r"clinical[\-\s]?trial|phase[\-\s]?[123I-V]+|topline|"
    r"endpoint|primary|readout|results|data|"
    r"BLA|NDA|CRL|complete[\-\s]response|breakthrough|"
    r"orphan|fast[\-\s]?track|priority[\-\s]?review|advisory[\-\s]?committee|"
    r"AdComm|adcom|advisory[\-\s]?panel|"
    r"investigational[\-\s]?new[\-\s]?drug|IND|"
    r"meets[\-\s]?primary|missed[\-\s]?primary|statistically[\-\s]?significant|"
    r"safety|efficacy|pivotal|registrational|dose[\-\s]?escalation"
    r")\b",
    re.IGNORECASE,
)

# Item codes in 8-K cover pages we care about. Item 7.01 (Reg FD) and 8.01
# (Other Events) are where most clinical/regulatory news lives. Item 2.02
# (Earnings) often co-announces readouts. Item 5.02 is for management
# changes — usually not clinical, but sometimes "Chief Medical Officer"
# transitions get co-bundled.
RELEVANT_ITEM_CODES_RE = re.compile(
    r"\bItem\s+(2\.02|5\.02|7\.01|8\.01)\b", re.IGNORECASE,
)


_session = None


def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": EDGAR_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        })
        _session = s
    return _session


def _polite_get(url: str, host_header: Optional[str] = None,
                 timeout: int = 15) -> Optional[requests.Response]:
    """GET with rate limiting + standard headers. Returns None on failure."""
    s = _get_session()
    headers = {}
    if host_header:
        headers["Host"] = host_header
    try:
        time.sleep(EDGAR_REQ_DELAY)
        r = s.get(url, timeout=timeout, headers=headers)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            logger.warning(f"[edgar] 429 rate limit on {url}, backing off 5s")
            time.sleep(5.0)
            return None
        if r.status_code != 404:  # 404 is silent (filing might not have an EX-99)
            logger.info(f"[edgar] status={r.status_code} on {url}")
        return None
    except requests.RequestException as e:
        logger.warning(f"[edgar] request error on {url}: {e}")
        return None


def fetch_biotech_ciks_from_universe(db) -> List[Tuple[str, str, str]]:
    """Return [(cik, ticker, company_name)] for all tickers we have in our
    universe — these are the CIKs we'll backfill 8-Ks for. Filters out
    rows without CIK (can be enriched later via SEC company_tickers.json).
    """
    out = []
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Try to use screener_stocks if it has cik column; fall back to
            # external lookup. For now just pull tickers.
            cur.execute("""
                SELECT DISTINCT ticker FROM catalyst_universe WHERE ticker IS NOT NULL
                UNION
                SELECT DISTINCT ticker FROM screener_stocks WHERE ticker IS NOT NULL
            """)
            tickers = [r[0] for r in cur.fetchall()]

        # Look up CIKs via SEC's official ticker map
        cik_map = _fetch_sec_ticker_map()
        for t in tickers:
            entry = cik_map.get(t.upper())
            if entry:
                out.append((entry["cik_padded"], t, entry["title"]))
        return out
    except Exception as e:
        logger.exception(f"fetch_biotech_ciks_from_universe failed: {e}")
        return out


_ticker_map_cache: Dict = {}

def _fetch_sec_ticker_map() -> Dict[str, Dict]:
    """Fetch SEC's master ticker → CIK map. Cached for the process lifetime."""
    if _ticker_map_cache:
        return _ticker_map_cache
    url = "https://www.sec.gov/files/company_tickers.json"
    s = _get_session()
    try:
        time.sleep(EDGAR_REQ_DELAY)
        r = s.get(url, timeout=30)
        if r.status_code != 200:
            logger.warning(f"[edgar] ticker-map fetch failed: {r.status_code}")
            return {}
        data = r.json()
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        for _, entry in data.items():
            cik = int(entry.get("cik_str", 0))
            if not cik:
                continue
            ticker = (entry.get("ticker") or "").upper()
            cik_padded = f"{cik:010d}"
            _ticker_map_cache[ticker] = {
                "cik": cik,
                "cik_padded": cik_padded,
                "ticker": ticker,
                "title": entry.get("title") or "",
            }
        logger.info(f"[edgar] loaded {len(_ticker_map_cache)} ticker→CIK mappings")
        return _ticker_map_cache
    except Exception as e:
        logger.exception(f"fetch_sec_ticker_map failed: {e}")
        return {}


def fetch_filings_for_cik(cik_padded: str,
                           start_year: int,
                           end_year: int,
                           form_types: List[str] = None,
                           ) -> List[Dict[str, Any]]:
    """Fetch all filings for a CIK between start_year and end_year.
    Default form_types = ['8-K'] which is what we want for catalysts.

    Returns list of {accession_no, filing_date, form_type, primary_doc_url}
    """
    if form_types is None:
        form_types = ["8-K"]

    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    # data.sec.gov needs different Host header
    s = _get_session()
    try:
        time.sleep(EDGAR_REQ_DELAY)
        r = s.get(url, timeout=30, headers={"Host": "data.sec.gov"})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception as e:
        logger.warning(f"[edgar] submissions fetch failed for CIK {cik_padded}: {e}")
        return []

    out: List[Dict[str, Any]] = []
    recent = (data.get("filings") or {}).get("recent") or {}
    if not recent:
        return out

    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    items_list = recent.get("items") or [""] * len(forms)

    for i in range(len(forms)):
        if forms[i] not in form_types:
            continue
        try:
            f_date = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if f_date.year < start_year or f_date.year > end_year:
            continue

        accession = accessions[i].replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else None
        out.append({
            "accession_no": accessions[i],
            "accession_padded": accession,
            "filing_date": f_date,
            "form_type": forms[i],
            "primary_doc": primary_doc,
            "items": items_list[i] if i < len(items_list) else "",
            "cik_padded": cik_padded,
        })
    return out


def fetch_filing_text_excerpt(filing: Dict[str, Any], max_chars: int = 3000
                                ) -> Optional[str]:
    """Fetch the primary document HTML and extract the first max_chars of
    plain text. Returns None on failure.
    """
    cik_padded = filing["cik_padded"].lstrip("0") or "0"
    accession = filing["accession_padded"]
    primary = filing.get("primary_doc")
    if not primary:
        return None

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession}/{primary}"
    r = _polite_get(url)
    if not r:
        return None

    text = r.text
    # Strip HTML
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if text else None


def looks_like_clinical_catalyst(text: str, items: str = "") -> bool:
    """Quick regex pre-filter: does this 8-K mention clinical/FDA keywords?
    False positives are fine — LLM filters more rigorously later.
    """
    if not text:
        return False
    return bool(CLINICAL_KEYWORDS_RE.search(text))


def edgar_backfill_for_cik(db, cik_padded: str, ticker: str,
                            start_year: int, end_year: int,
                            run_id: int) -> Dict[str, int]:
    """Backfill all 8-K filings for one CIK over [start_year, end_year]
    that pass the clinical-keyword regex pre-filter. Writes to staging.

    Returns {scraped, inserted, skipped, errored}.
    """
    counts = {"scraped": 0, "inserted": 0, "skipped": 0, "errored": 0}

    filings = fetch_filings_for_cik(cik_padded, start_year, end_year)
    counts["scraped"] = len(filings)

    for f in filings:
        try:
            # Skip if not in items list (e.g. just acquisitions)
            items = f.get("items") or ""
            if items and not RELEVANT_ITEM_CODES_RE.search(items):
                counts["skipped"] += 1
                continue

            # Fetch text and apply keyword filter
            excerpt = fetch_filing_text_excerpt(f, max_chars=3000)
            if not excerpt or not looks_like_clinical_catalyst(excerpt, items):
                counts["skipped"] += 1
                continue

            # Build staging row
            cik_padded_str = f["cik_padded"]
            accession = f["accession_no"]
            source_id = f"edgar:{cik_padded_str}:{accession}"
            primary = f.get("primary_doc") or ""
            cik_unpadded = cik_padded_str.lstrip("0") or "0"
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_unpadded}/{f['accession_padded']}/{primary}"
            )

            with db.get_conn() as conn:
                cur = conn.cursor()
                # Title heuristic: first 200 chars; LLM normalize will refine
                title = excerpt[:200]
                cur.execute("""
                    INSERT INTO catalyst_backfill_staging (
                        source, source_id, ticker, cik, filing_date,
                        catalyst_date, date_precision, raw_title,
                        raw_text_excerpt, source_url, status
                    ) VALUES (
                        'edgar', %s, %s, %s, %s, %s, 'day', %s, %s, %s, 'pending'
                    )
                    ON CONFLICT (source, source_id) DO NOTHING
                """, (
                    source_id, ticker, cik_padded_str,
                    f["filing_date"], f["filing_date"],
                    title, excerpt, source_url,
                ))
                inserted = cur.rowcount
                conn.commit()

            if inserted:
                counts["inserted"] += 1
            else:
                counts["skipped"] += 1
        except Exception as e:
            counts["errored"] += 1
            logger.warning(f"[edgar] error processing filing {f.get('accession_no')}: {e}")

    return counts
