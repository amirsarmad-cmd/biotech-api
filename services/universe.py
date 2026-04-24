"""
Universe expansion: pulls biotech/pharma tickers with upcoming catalysts from
multiple public sources (FDA Calendar, ClinicalTrials.gov).

Usage:
    # from services.universe import seed_universe  (doc example)
    result = seed_universe(db, fetcher, mode="both", progress_cb=print)
    # result = {"added": 185, "updated": 23, "skipped": 12, "source_counts": {...}}
"""
import os
import re
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ============================================================
# SOURCE 1: FDA Calendar (scraped from BiopharmCatalyst public data)
# ============================================================

def fetch_fda_calendar(max_results: int = 200) -> List[Dict]:
    """Scrape near-term FDA PDUFA and AdCom dates from public biotech calendar sites.
    
    Falls back through several sources. Returns list of catalysts with:
      ticker, drug, catalyst_type, catalyst_date, description
    """
    results = []
    
    # Primary: biopharmcatalyst.com has a public FDA calendar page
    try:
        results.extend(_scrape_biopharmcatalyst_fda(max_results))
        logger.info(f"FDA calendar: {len(results)} catalysts from biopharmcatalyst")
    except Exception as e:
        logger.warning(f"biopharmcatalyst scrape failed: {e}")
    
    # Fallback: RTT News / Drugs.com public PDUFA lists
    if len(results) < 20:
        try:
            more = _scrape_drugs_com_pdufa(max_results - len(results))
            results.extend(more)
            logger.info(f"drugs.com fallback added {len(more)}")
        except Exception as e:
            logger.warning(f"drugs.com scrape failed: {e}")
    
    # Dedupe by (ticker, catalyst_date)
    seen = set()
    dedup = []
    for r in results:
        key = (r.get("ticker","").upper(), r.get("catalyst_date",""))
        if key not in seen and key[0]:
            seen.add(key)
            dedup.append(r)
    
    return dedup[:max_results]


def _scrape_biopharmcatalyst_fda(limit: int) -> List[Dict]:
    """Scrape biopharmcatalyst.com/calendar/fda-calendar (public page)."""
    url = "https://www.biopharmcatalyst.com/calendars/fda-calendar"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"biopharmcatalyst fetch failed: {e}")
        return []
    
    # Parse tickers + drug + date from the HTML (they expose it via data attributes + JS)
    # Pattern: table rows with ticker symbol, drug name, date
    results = []
    # Look for ticker patterns like symbol="TICKER" or href="/company/TICKER"
    ticker_pattern = re.compile(
        r'href=["\']/company/([A-Z]{1,5})["\'][^>]*>([^<]+)</a>.*?'
        r'(?:drug-name|d\-title)[^>]*>([^<]+)</(?:td|div|span).*?'
        r'(\d{4}-\d{2}-\d{2}|[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})',
        re.DOTALL | re.IGNORECASE
    )
    
    # Simpler pattern: any /company/TICKER link followed by a date within next ~500 chars
    simple = re.compile(r'/company/([A-Z]{1,5})[/"]', re.IGNORECASE)
    date_pat = re.compile(r'(\d{4}-\d{2}-\d{2})')
    
    # Split by table rows
    rows = re.split(r'<tr[^>]*>', html)
    for row in rows[:limit*2]:
        tk_m = simple.search(row)
        if not tk_m: continue
        ticker = tk_m.group(1).upper()
        if len(ticker) > 5 or len(ticker) < 1: continue
        
        dt_m = date_pat.search(row)
        if not dt_m: continue
        
        # Extract drug name (between ticker link and date)
        drug = "Unknown"
        drug_m = re.search(r'(?:drug|product|name)[^>]*>([^<]{3,80})</', row, re.IGNORECASE)
        if drug_m:
            drug = drug_m.group(1).strip()[:80]
        
        results.append({
            "ticker": ticker,
            "company_name": ticker,  # Will be enriched later
            "catalyst_type": "FDA Decision",
            "catalyst_date": dt_m.group(1),
            "description": f"FDA decision on {drug}",
            "drug": drug,
            "source": "BiopharmCatalyst",
        })
        if len(results) >= limit: break
    
    return results


def _scrape_drugs_com_pdufa(limit: int) -> List[Dict]:
    """Fallback: drugs.com/pipeline.html and recent FDA decision pages."""
    url = "https://www.drugs.com/new-drug-approvals.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    
    # drugs.com uses structured rows but doesn't list tickers directly
    # We'd need a drug-name → company → ticker mapping
    # For now, return empty (the biopharmcatalyst source usually works)
    return []


# ============================================================
# SOURCE 2: ClinicalTrials.gov API v2
# ============================================================

def fetch_clinicaltrials_upcoming(max_results: int = 500, days_ahead: int = 180) -> List[Dict]:
    """Fetch Phase 3 trials with Primary Completion Date in the next `days_ahead` days.
    Uses ClinicalTrials.gov API v2 (stable public API).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    horizon = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # API v2 — combined query.term handles both phase and date range
    query_term = f"AREA[Phase]PHASE3 AND AREA[PrimaryCompletionDate]RANGE[{today},{horizon}]"
    params = {
        "query.term": query_term,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING",
        "pageSize": "200",
        "format": "json",
    }
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)
    
    all_results = []
    page_token = None
    pages_fetched = 0
    
    while pages_fetched < 3 and len(all_results) < max_results:  # max 3 pages = 600 trials
        full_url = url + (f"&pageToken={page_token}" if page_token else "")
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=25).read())
        except Exception as e:
            logger.warning(f"ClinicalTrials.gov fetch failed: {e}")
            break
        
        studies = data.get("studies", [])
        for study in studies:
            parsed = _parse_ct_study(study)
            if parsed:
                all_results.append(parsed)
        
        page_token = data.get("nextPageToken")
        pages_fetched += 1
        if not page_token: break
    
    logger.info(f"ClinicalTrials.gov: {len(all_results)} Phase 3 trials in next {days_ahead}d")
    return all_results[:max_results]


def _parse_ct_study(study: Dict) -> Optional[Dict]:
    """Parse a single ClinicalTrials.gov study into our catalyst format."""
    try:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
        
        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        sponsor_name = sponsor.get("name", "")
        
        pcd = status.get("primaryCompletionDateStruct", {}).get("date", "")
        if not pcd: return None
        
        # Normalize date to YYYY-MM-DD
        if len(pcd) == 7:  # YYYY-MM
            pcd = f"{pcd}-15"
        
        # Map sponsor to ticker via company_to_ticker lookup
        ticker = company_to_ticker(sponsor_name)
        if not ticker:
            return None  # Skip if we can't ID the company
        
        return {
            "ticker": ticker,
            "company_name": sponsor_name,
            "catalyst_type": "Phase 3 Readout",
            "catalyst_date": pcd,
            "description": title[:300],
            "nct_id": nct_id,
            "source": "ClinicalTrials.gov",
        }
    except Exception as e:
        logger.debug(f"CT parse error: {e}")
        return None


# ============================================================
# COMPANY NAME → TICKER MAPPING
# ============================================================

# Known biotech sponsors → tickers. Extended with common variants.
# yfinance's search API fills gaps at runtime.
KNOWN_SPONSORS = {
    # Large caps
    "pfizer": "PFE", "pfizer inc": "PFE", "pfizer, inc.": "PFE",
    "merck sharp & dohme": "MRK", "merck": "MRK", "merck & co": "MRK",
    "johnson & johnson": "JNJ", "janssen": "JNJ", "janssen pharmaceuticals": "JNJ",
    "novartis": "NVS", "novartis pharmaceuticals": "NVS",
    "roche": "RHHBY", "hoffmann-la roche": "RHHBY", "genentech": "RHHBY",
    "astrazeneca": "AZN",
    "glaxosmithkline": "GSK", "gsk": "GSK",
    "abbvie": "ABBV",
    "eli lilly": "LLY", "eli lilly and company": "LLY", "lilly": "LLY",
    "bristol-myers squibb": "BMY", "bristol myers squibb": "BMY",
    "sanofi": "SNY",
    "bayer": "BAYRY",
    "takeda": "TAK", "takeda pharmaceutical": "TAK",
    "amgen": "AMGN",
    "gilead sciences": "GILD", "gilead": "GILD",
    "regeneron": "REGN", "regeneron pharmaceuticals": "REGN",
    "vertex pharmaceuticals": "VRTX", "vertex": "VRTX",
    "biogen": "BIIB",
    "moderna": "MRNA", "modernatx": "MRNA",
    "biontech": "BNTX",
    "illumina": "ILMN",
    "alnylam": "ALNY", "alnylam pharmaceuticals": "ALNY",
    # Mid-caps
    "seagen": "SGEN",
    "bluebird bio": "BLUE",
    "sarepta therapeutics": "SRPT", "sarepta": "SRPT",
    "ionis pharmaceuticals": "IONS",
    "bmrn": "BMRN", "biomarin pharmaceutical": "BMRN",
    "incyte": "INCY",
    "exelixis": "EXEL",
    "united therapeutics": "UTHR",
    "neurocrine biosciences": "NBIX",
    "jazz pharmaceuticals": "JAZZ",
    "horizon therapeutics": "HZNP",
    "arrowhead pharmaceuticals": "ARWR",
    # Small-caps
    "iovance biotherapeutics": "IOVA",
    "agenus": "AGEN",
    "karuna therapeutics": "KRTX",
    "relay therapeutics": "RLAY",
    "blueprint medicines": "BPMC",
    "editas medicine": "EDIT",
    "crispr therapeutics": "CRSP",
    "intellia therapeutics": "NTLA",
    "beam therapeutics": "BEAM",
    "verve therapeutics": "VERV",
}

# Cache yfinance-resolved tickers in-memory for this seed run
_yf_resolve_cache: Dict[str, Optional[str]] = {}


def company_to_ticker(name: str) -> Optional[str]:
    """Resolve a company name to a public ticker.
    Tries known map first, then yfinance search API, then returns None."""
    if not name: return None
    clean = name.strip().lower()
    
    # Direct hit
    if clean in KNOWN_SPONSORS:
        return KNOWN_SPONSORS[clean]
    
    # Partial match on known sponsors
    for known_name, ticker in KNOWN_SPONSORS.items():
        if known_name in clean or clean in known_name:
            if len(clean) > 5 and len(known_name) > 5:  # avoid tiny matches
                return ticker
    
    # yfinance search — costs an HTTP call, cache result
    if clean in _yf_resolve_cache:
        return _yf_resolve_cache[clean]
    
    ticker = _yf_search_ticker(name)
    _yf_resolve_cache[clean] = ticker
    return ticker


def _yf_search_ticker(name: str) -> Optional[str]:
    """Use yfinance's search endpoint to find ticker from name.
    Only returns ticker if the result looks like a biotech/pharma company
    whose name substring-matches the sponsor name (avoids false positives like
    assigning 'ZIM' shipping to Chinese pharma companies).
    """
    if not name or len(name) < 3: return None
    clean_name = name.lower().strip()
    
    # Reject obvious non-biotech sponsor patterns
    reject_patterns = ["university", "hospital", "national cancer", "memorial",
                       "health network", "medical center", "clinic", "institute of",
                       "va medical", "research group", "cooperative group", "unicancer"]
    for pat in reject_patterns:
        if pat in clean_name:
            return None
    
    # Reject non-US (or yet-unlisted foreign) pharma by obvious tells
    # These companies often don't have US-listed securities
    foreign_hints = ["co.ltd", "co., ltd", "co.,ltd", "pharmaceutical technology", 
                     "pharma technology", "jiangsu", "shanghai", "hangzhou", "beijing", 
                     "chengdu", "wuhan", "shandong", "sichuan", "chigenovo"]
    for hint in foreign_hints:
        if hint in clean_name:
            return None
    
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(name)}&quotesCount=5&newsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        quotes = data.get("quotes", [])
        
        # Prefer: US-listed EQUITY whose short/long name contains the searched company name
        # Extract first word of sponsor name as the key signal (e.g. "Novo" from "Novo Nordisk A/S")
        name_words = [w for w in re.split(r"[\s,.\-]+", clean_name) if len(w) >= 4]
        key_word = name_words[0] if name_words else clean_name[:6]
        
        candidates = []
        for q in quotes:
            if q.get("quoteType") != "EQUITY": continue
            sym = q.get("symbol", "").upper()
            if not sym or "." in sym or "-" in sym: continue
            # Check if the sponsor name appears in the company's long/short name
            long_n = (q.get("longname") or "").lower()
            short_n = (q.get("shortname") or "").lower()
            name_match = (key_word in long_n or key_word in short_n or
                          clean_name[:8] in long_n or clean_name[:8] in short_n)
            # Industry check — pharma/biotech only
            industry = (q.get("industry") or "").lower()
            sector = (q.get("sector") or "").lower()
            is_pharma = ("drug" in industry or "biotech" in industry or 
                        "pharma" in industry or "healthcare" in sector or
                        "health" in sector)
            if name_match and is_pharma:
                return sym
            if name_match:
                candidates.append((sym, 2))
            elif is_pharma:
                candidates.append((sym, 1))
        
        # Fallback: if we have a name-match non-pharma candidate, use it (small chance of miss)
        if candidates:
            candidates.sort(key=lambda x: -x[1])
            if candidates[0][1] >= 2:
                return candidates[0][0]
        return None
    except Exception:
        return None


# ============================================================
# ORCHESTRATOR
# ============================================================

def seed_universe(db, fetcher, mode: str = "both",
                  progress_cb: Optional[Callable[[float, str], None]] = None,
                  clear_existing: bool = False) -> Dict:
    """Main entry point. Orchestrates FDA + ClinicalTrials fetch and adds to DB.
    
    Args:
        db: BiotechDatabase instance
        fetcher: Fetcher instance for enrichment
        mode: "fda" | "clinical" | "both"
        progress_cb: callback(pct 0-1, msg) for UI progress
        clear_existing: wipe screener_stocks before seeding (destructive)
    
    Returns: {"added", "updated", "skipped", "source_counts", "total_catalysts"}
    """
    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)
        logger.info(f"[seed {pct*100:.0f}%] {msg}")
    
    _p(0.01, "Starting universe expansion...")
    
    if clear_existing:
        db.clear_all_data()
        _p(0.02, "Cleared existing data")
    
    catalysts: List[Dict] = []
    source_counts = {"FDA": 0, "ClinicalTrials": 0}
    
    # ---- FDA Calendar ----
    if mode in ("fda", "both"):
        _p(0.05, "Fetching FDA calendar...")
        try:
            fda = fetch_fda_calendar(max_results=200)
            catalysts.extend(fda)
            source_counts["FDA"] = len(fda)
            _p(0.20, f"FDA: {len(fda)} catalysts found")
        except Exception as e:
            logger.error(f"FDA calendar error: {e}")
            _p(0.20, f"FDA failed ({e})")
    
    # ---- ClinicalTrials.gov ----
    if mode in ("clinical", "both"):
        _p(0.25, "Fetching ClinicalTrials.gov Phase 3 trials...")
        try:
            ct = fetch_clinicaltrials_upcoming(max_results=400, days_ahead=180)
            catalysts.extend(ct)
            source_counts["ClinicalTrials"] = len(ct)
            _p(0.40, f"ClinicalTrials: {len(ct)} Phase 3 trials found")
        except Exception as e:
            logger.error(f"ClinicalTrials error: {e}")
            _p(0.40, f"ClinicalTrials failed ({e})")
    
    # Dedupe by (ticker, catalyst_type, catalyst_date)
    seen = set()
    unique = []
    for c in catalysts:
        key = (c["ticker"].upper(), c["catalyst_type"], c["catalyst_date"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    _p(0.45, f"Deduped to {len(unique)} unique catalysts across {len(set(c['ticker'] for c in unique))} tickers")
    
    # ---- Enrich with yfinance data (market cap, etc.) and write to DB ----
    unique_tickers = list(set(c["ticker"].upper() for c in unique))
    
    # Parallel enrichment
    ticker_data: Dict[str, Dict] = {}
    _p(0.50, f"Enriching {len(unique_tickers)} tickers with yfinance data...")
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetcher.get_comprehensive_data, t): t for t in unique_tickers}
        completed = 0
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                ticker_data[t] = fut.result(timeout=20)
            except Exception as e:
                logger.debug(f"enrich fail {t}: {e}")
                ticker_data[t] = None
            completed += 1
            if completed % 20 == 0:
                pct = 0.50 + (completed / len(unique_tickers)) * 0.40  # 50-90%
                _p(pct, f"Enriched {completed}/{len(unique_tickers)}...")
    
    _p(0.90, "Writing catalysts to database...")
    
    added = 0
    updated = 0
    skipped = 0
    
    for c in unique:
        t = c["ticker"].upper()
        enriched = ticker_data.get(t)
        if not enriched:
            skipped += 1
            logger.debug(f"Skipping {t}: no enrichment data")
            continue
        
        # Merge source catalyst with enriched data
        record = {
            "ticker": t,
            "company_name": enriched.get("company_name") or c.get("company_name") or t,
            "industry": enriched.get("industry", "Biotechnology"),
            "market_cap": enriched.get("market_cap", 0),
            "catalyst_type": c["catalyst_type"],
            "catalyst_date": c["catalyst_date"],
            "probability": c.get("probability", 0.5),  # fetcher may have computed this
            "description": c["description"][:500],
            "news_count": enriched.get("news_count", 0),
            "sentiment_score": enriched.get("sentiment_score", 0),
            "overall_score": enriched.get("overall_score", 0),
            "last_updated": datetime.now().isoformat(),
        }
        
        try:
            existed = bool(db.add_stock(record))
            if existed: added += 1
            else: updated += 1
        except Exception as e:
            logger.warning(f"add_stock fail for {t}: {e}")
            skipped += 1
    
    _p(1.0, f"Done: {added} added, {updated} updated, {skipped} skipped")
    
    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "source_counts": source_counts,
        "total_catalysts": len(unique),
        "total_tickers": len(unique_tickers),
    }
