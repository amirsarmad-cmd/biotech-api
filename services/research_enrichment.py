"""
Research enrichment for AI analysis.

Provides:
  - Earnings transcripts fetcher (public sources)
  - Earnings history / EPS trend via yfinance
  - Additional news sources (STAT News, Endpoints, BioSpace, Google News)
  - Full article body fetch for selected authoritative sources
"""
import os
import re
import logging
import hashlib
import requests
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import quote

logger = logging.getLogger(__name__)

# In-memory cache
_enrich_cache: Dict = {}
_CACHE_TTL = 6 * 3600  # 6h


def _cached(key: str, fn):
    now = time.time()
    if key in _enrich_cache:
        ts, val = _enrich_cache[key]
        if now - ts < _CACHE_TTL: return val
    val = fn()
    _enrich_cache[key] = (now, val)
    return val


# ============================================================
# EPS / EARNINGS HISTORY via yfinance
# ============================================================

def get_earnings_history(ticker: str) -> Dict:
    """Fetch historical quarterly EPS, revenue, surprise %, dates.
    
    Returns:
    {
        "available": bool,
        "rows": [
            {"date": "2025-11-05", "eps_actual": 1.25, "eps_estimate": 1.10, "surprise_pct": 13.6,
             "revenue_actual": 2500M, "revenue_estimate": 2400M, "revenue_surprise_pct": 4.2},
            ...
        ],
        "next_earnings_date": "2026-02-05",
        "trailing_4q_avg_surprise": 8.5,  # %
    }
    """
    def _do():
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            
            # earnings_history has last 4 quarters of EPS surprises
            eh = t.get_earnings_history()
            rows = []
            surprises = []
            
            if eh is not None and not eh.empty:
                for idx, r in eh.iterrows():
                    eps_a = r.get("epsActual") or r.get("eps_actual")
                    eps_e = r.get("epsEstimate") or r.get("eps_estimate")
                    surprise = r.get("surprisePercent") or r.get("surprise_percent")
                    
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    
                    try:
                        eps_a_f = float(eps_a) if eps_a is not None else None
                        eps_e_f = float(eps_e) if eps_e is not None else None
                        surprise_f = float(surprise) if surprise is not None else None
                    except (ValueError, TypeError):
                        eps_a_f = eps_e_f = surprise_f = None
                    
                    rows.append({
                        "date": date_str,
                        "eps_actual": eps_a_f,
                        "eps_estimate": eps_e_f,
                        "surprise_pct": surprise_f,
                    })
                    if surprise_f is not None:
                        surprises.append(surprise_f)
            
            # Try to get next earnings date from calendar
            next_date = None
            try:
                cal = t.calendar
                if cal is not None and not cal.empty if hasattr(cal, "empty") else cal:
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed: next_date = str(ed[0] if isinstance(ed, list) else ed)[:10]
                    else:
                        if "Earnings Date" in cal.index:
                            next_date = str(cal.loc["Earnings Date"].iloc[0])[:10]
            except Exception as e:
                logger.debug(f"calendar unavailable for {ticker}: {e}")
            
            # Income statement for rev growth
            try:
                info = t.info or {}
                rev_growth_qoq = info.get("revenueQuarterlyGrowth")
                rev_growth_yoy = info.get("revenueGrowth")
            except:
                rev_growth_qoq = rev_growth_yoy = None
            
            return {
                "available": bool(rows) or next_date is not None,
                "rows": rows[:8],  # last 8 quarters max
                "next_earnings_date": next_date,
                "trailing_4q_avg_surprise": round(sum(surprises[:4]) / max(1, min(4, len(surprises))), 1) if surprises else None,
                "revenue_growth_qoq": rev_growth_qoq,
                "revenue_growth_yoy": rev_growth_yoy,
            }
        except Exception as e:
            logger.warning(f"earnings history {ticker}: {e}")
            return {"available": False, "error": str(e)[:200]}
    
    return _cached(f"earnings:{ticker}", _do)


# ============================================================
# EARNINGS TRANSCRIPTS
# ============================================================

def fetch_earnings_transcript(ticker: str) -> Optional[Dict]:
    """Try to locate the latest earnings call transcript for AI consumption.
    
    Public sources:
      - roic.ai (has earnings transcripts, free tier)
      - fool.com (free transcripts)
      - motleyfool.com (same, different endpoint)
    
    Returns:
    {
        "title": "Moderna Q3 2025 Earnings Call Transcript",
        "url": "https://...",
        "date": "2025-11-05",
        "excerpt": "first 3000 chars",  # for AI consumption
        "source": "Motley Fool",
    } or None
    """
    def _do():
        # 1) Try fool.com search
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; biotech-screener/1.0)"}
            url = f"https://www.fool.com/quote/{ticker}/"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                # Look for transcript link
                m = re.search(r'href="(/earnings/call-transcripts/[^"]+)"[^>]*>\s*([^<]{5,200})', r.text)
                if m:
                    transcript_url = "https://www.fool.com" + m.group(1)
                    title = m.group(2).strip()
                    # Fetch the transcript body
                    tr = requests.get(transcript_url, headers=headers, timeout=15)
                    if tr.status_code == 200:
                        # Extract body text (rough)
                        text = re.sub(r'<[^>]+>', ' ', tr.text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        # Find transcript start (usually after "Prepared Remarks" or similar)
                        for marker in ["Prepared Remarks", "Operator", "Company Representatives"]:
                            if marker in text:
                                text = text[text.find(marker):]
                                break
                        excerpt = text[:4000]
                        return {
                            "title": title,
                            "url": transcript_url,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "excerpt": excerpt,
                            "source": "Motley Fool",
                        }
        except Exception as e:
            logger.debug(f"fool.com transcript for {ticker}: {e}")
        
        return None
    
    return _cached(f"transcript:{ticker}", _do)


# ============================================================
# ADDITIONAL NEWS SOURCES
# ============================================================

def fetch_stat_news(ticker: str, company: str, cap: int = 8) -> List[Dict]:
    """STAT News — authoritative biotech/health news. Uses Google-site-search fallback
    since STAT's internal search often returns HTML-rendered later by JS."""
    try:
        import feedparser
        # Google News search restricted to statnews.com — reliable
        query = quote(f'"{ticker}" OR "{company}" site:statnews.com')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:cap * 2]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not title or "statnews.com" not in (link + entry.get("summary", "")): continue
            # Strip "- STAT" suffix
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            results.append({
                "title": title, "url": link, "source": "STAT News",
                "date": entry.get("published", "")[:10] if entry.get("published") else "",
                "summary": entry.get("summary","")[:300], "provider": "STAT News",
            })
            if len(results) >= cap: break
        return results
    except Exception as e:
        logger.warning(f"STAT News {ticker}: {e}")
        return []


def fetch_endpoints_news(ticker: str, company: str, cap: int = 6) -> List[Dict]:
    """Endpoints News — via Google site-search for reliability."""
    try:
        import feedparser
        query = quote(f'"{ticker}" OR "{company}" site:endpts.com')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:cap * 2]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not title or "endpts.com" not in (link + entry.get("summary", "")): continue
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            results.append({
                "title": title, "url": link, "source": "Endpoints News",
                "date": entry.get("published", "")[:10] if entry.get("published") else "",
                "summary": entry.get("summary","")[:300], "provider": "Endpoints News",
            })
            if len(results) >= cap: break
        return results
    except Exception as e:
        logger.warning(f"Endpoints {ticker}: {e}")
        return []


def fetch_biospace(ticker: str, company: str, cap: int = 6) -> List[Dict]:
    """BioSpace — biotech-focused news aggregator. Google site-search for reliability."""
    try:
        import feedparser
        query = quote(f'"{ticker}" OR "{company}" site:biospace.com')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:cap * 2]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            if not title or "biospace.com" not in (link + entry.get("summary", "")): continue
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            results.append({
                "title": title, "url": link, "source": "BioSpace",
                "date": entry.get("published", "")[:10] if entry.get("published") else "",
                "summary": entry.get("summary","")[:300], "provider": "BioSpace",
            })
            if len(results) >= cap: break
        return results
    except Exception as e:
        logger.warning(f"BioSpace {ticker}: {e}")
        return []


def fetch_google_news(ticker: str, company: str, cap: int = 10) -> List[Dict]:
    """Google News RSS — broadest coverage, picks up long tail sources."""
    try:
        import feedparser
        query = quote(f'"{ticker}" OR "{company}" biotech')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries[:cap]:
            title = entry.get("title", "")
            if not title: continue
            # Google News titles include publisher: "Title - Publisher"
            publisher = "Google News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                publisher = parts[1] if len(parts) > 1 else publisher
            results.append({
                "title": title,
                "url": entry.get("link", ""),
                "source": publisher,
                "date": entry.get("published", "")[:10] if entry.get("published") else "",
                "summary": entry.get("summary", "")[:400],
                "provider": f"Google News → {publisher}",
            })
        return results
    except Exception as e:
        logger.warning(f"Google News {ticker}: {e}")
        return []


# ============================================================
# FULL ARTICLE BODY FETCH
# ============================================================

# Sources we consider authoritative enough to fetch body for
AUTHORITATIVE_DOMAINS = [
    "sec.gov", "fda.gov",
    "fiercebiotech.com", "fiercepharma.com",
    "statnews.com", "endpts.com",
    "biospace.com", "reuters.com",
    "bloomberg.com", "wsj.com",  # may hit paywall
    "finance.yahoo.com",  # Yahoo Finance articles are usually free
    "fool.com", "motleyfool.com",
    "seekingalpha.com",  # will need auth
    "marketwatch.com",
    "bioworld.com",
]


def is_authoritative(url: str) -> bool:
    """Check if URL is from a source we trust enough to spend time fetching."""
    if not url: return False
    return any(d in url.lower() for d in AUTHORITATIVE_DOMAINS)


def fetch_article_body(url: str, max_chars: int = 4000) -> Optional[str]:
    """Fetch article body text. Returns None if failed, paywalled, or too short.
    
    Does basic HTML-to-text extraction. For production-grade we'd use readability/newspaper3k,
    but this keeps it lightweight and deploy-safe.
    """
    if not url or not is_authoritative(url):
        return None
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return None
        
        html = r.text
        
        # Paywall detection — common markers
        paywall_markers = [
            "paywall", "subscribe now", "subscription required",
            "members only", "unlock this article",
        ]
        if any(m in html[:10000].lower() for m in paywall_markers):
            # Still try to return what we got
            pass
        
        # Strip script/style
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Try to find article body
        # Common patterns: <article>, <div class="article-body">, <div class="story-body">
        article_match = re.search(r'<article[^>]*>(.*?)</article>', html, re.DOTALL | re.IGNORECASE)
        if article_match:
            body = article_match.group(1)
        else:
            # Fallback: main content
            main_match = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL | re.IGNORECASE)
            body = main_match.group(1) if main_match else html
        
        # Strip all HTML tags
        text = re.sub(r'<[^>]+>', ' ', body)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Too short → probably paywalled or failed
        if len(text) < 200:
            return None
        
        return text[:max_chars]
    except Exception as e:
        logger.debug(f"fetch_article_body {url[:50]}: {e}")
        return None


def enrich_with_full_articles(articles: List[Dict], max_fetches: int = 8,
                              max_chars_each: int = 3000) -> List[Dict]:
    """Given a list of article dicts, fetch full body for authoritative ones (up to max_fetches).
    Adds 'full_text' field + 'was_fully_read': bool to each enriched article.
    
    Parallel fetch with ThreadPool for speed.
    """
    # Filter to authoritative
    to_fetch = [(i, a) for i, a in enumerate(articles) if is_authoritative(a.get("url", ""))]
    to_fetch = to_fetch[:max_fetches]
    
    def _fetch_one(idx, article):
        body = fetch_article_body(article.get("url", ""), max_chars=max_chars_each)
        return idx, body
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_fetch_one, i, a) for i, a in to_fetch]
        for fut in as_completed(futures, timeout=40):
            try:
                idx, body = fut.result(timeout=20)
                if body:
                    articles[idx]["full_text"] = body
                    articles[idx]["was_fully_read"] = True
                else:
                    articles[idx]["was_fully_read"] = False
            except Exception as e:
                logger.debug(f"enrich article failed: {e}")
    
    # Mark all others as NOT fully read (title+snippet only)
    for a in articles:
        a.setdefault("was_fully_read", False)
    
    return articles


# ============================================================
# UNIFIED ORCHESTRATOR — replaces fetcher_news's fetch_all_sources when called
# ============================================================

def fetch_all_sources_v2(ticker: str, company: str, catalyst: str = "",
                          include_full_article_bodies: bool = True,
                          max_full_fetches: int = 8) -> Dict:
    """Expanded source fetch + optional full-article enrichment.
    
    Returns:
    {
        "articles": [...],
        "earnings_history": {...},
        "earnings_transcript": {...} or None,
        "stats": {
            "total_articles": 42,
            "articles_fully_read": 7,
            "sources": {"Finnhub": 20, "Yahoo": 10, ...},
            "fetch_time_sec": 12.3,
        }
    }
    """
    t0 = time.time()
    
    from services.fetcher_news import fetch_all_sources as _base_fetch
    
    # Run base sources + new sources in parallel
    sources = []
    source_map = [
        ("base",           lambda: _base_fetch(ticker, company, catalyst)),
        ("stat_news",      lambda: fetch_stat_news(ticker, company, cap=8)),
        ("endpoints",      lambda: fetch_endpoints_news(ticker, company, cap=6)),
        ("biospace",       lambda: fetch_biospace(ticker, company, cap=6)),
        ("google_news",    lambda: fetch_google_news(ticker, company, cap=12)),
    ]
    
    with ThreadPoolExecutor(max_workers=len(source_map)) as ex:
        futs = {ex.submit(fn): name for name, fn in source_map}
        for fut in as_completed(futs, timeout=30):
            try:
                r = fut.result(timeout=25)
                if r: sources.extend(r)
            except Exception as e:
                logger.warning(f"{futs[fut]} failed: {e}")
    
    # Dedupe by URL
    seen = set()
    deduped = []
    for a in sources:
        url = a.get("url", "")
        key = url if url else (a.get("title","") + a.get("source",""))
        if key in seen: continue
        seen.add(key); deduped.append(a)
    
    # Fetch earnings history + transcript in parallel
    earnings = {}
    transcript = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_earnings = ex.submit(get_earnings_history, ticker)
        f_transcript = ex.submit(fetch_earnings_transcript, ticker)
        try: earnings = f_earnings.result(timeout=10)
        except Exception as e: logger.warning(f"earnings: {e}")
        try: transcript = f_transcript.result(timeout=15)
        except Exception as e: logger.warning(f"transcript: {e}")
    
    # Optionally fetch full article bodies for authoritative sources
    if include_full_article_bodies:
        deduped = enrich_with_full_articles(deduped, max_fetches=max_full_fetches)
    
    # Aggregate stats
    source_counts = {}
    fully_read = 0
    for a in deduped:
        src = a.get("provider") or a.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        if a.get("was_fully_read"): fully_read += 1
    
    elapsed = time.time() - t0
    
    return {
        "articles": deduped,
        "earnings_history": earnings,
        "earnings_transcript": transcript,
        "stats": {
            "total_articles": len(deduped),
            "articles_fully_read": fully_read,
            "sources": source_counts,
            "fetch_time_sec": round(elapsed, 1),
        },
    }
