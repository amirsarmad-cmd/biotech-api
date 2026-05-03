"""STAT News + STAT+ collector.

Two-step ingestion:
  1. Pull the public biotech RSS feed for URLs + headlines (free).
  2. For STAT+ tagged articles, fetch the full body using cookies — the
     stat_paywall_token JWT is honored from any IP (verified via
     /admin/news/stat-direct-test 2026-05-04). No PerimeterX / anti-bot
     layer to defeat.

Persists to catalyst_event_news with:
  - source='stat_news_rss'    (free article — body via RSS or unpaywalled fetch)
  - source='stat_plus'        (paywalled body fetched with auth cookies)

Ticker mention extraction:
  - Cashtag regex `\\$([A-Z]{1,5})` in headline+body
  - "(NASDAQ|NYSE: TICKER)" parens regex
  - company_name → ticker dict from screener_stocks (loaded once per run)

A single article that mentions N tickers fans out to N rows (one per ticker)
with the same url_hash but different ticker; UNIQUE (ticker, url_hash) keeps
the table clean.
"""
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup

from services.fetcher_news import _decode_cookies_b64
from services.news_library import insert_news_row

logger = logging.getLogger(__name__)


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

# Default STAT category feed
DEFAULT_FEED = "https://www.statnews.com/category/biotech/feed/"

# Cashtag and listing regexes
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_LISTING_RE = re.compile(r"\((?:NASDAQ|NYSE|AMEX)[:\s]+([A-Z]{1,5})\)", re.IGNORECASE)


# ── Session helpers ─────────────────────────────────────────

def _build_session(cookies_env: str = "STAT_PLUS_COOKIES_B64") -> requests.Session:
    """Build a requests.Session with STAT+ auth cookies if available.
    Falls back to an unauthenticated session if env not set."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    cookies_list = _decode_cookies_b64(cookies_env)
    if cookies_list:
        for c in cookies_list:
            s.cookies.set(c.get("name"), c.get("value"),
                          domain=c.get("domain", "").lstrip("."),
                          path=c.get("path", "/"))
    return s


def _is_paywall_blocked(html: str) -> bool:
    """Tighter paywall-detection: look for the actual paywall barrier
    structure, not stray words."""
    if len(html) < 5000:
        # Very short body is almost always a paywall stub
        bl = html.lower()
        if "subscribe" in bl or "log in" in bl or "free trial" in bl:
            return True
    bl = html.lower()
    # Specific STAT paywall barrier text
    barriers = [
        "to continue reading",
        "already a subscriber",
        "subscribe to continue",
        "this article is for stat+ subscribers",
        "log in to read this article",
    ]
    return any(b in bl for b in barriers)


# ── Fetchers ────────────────────────────────────────────────

def fetch_stat_rss(feed_url: str = DEFAULT_FEED, timeout: int = 15) -> List[Dict]:
    """Pull the STAT biotech RSS feed. Returns list of
    {headline, url, summary, published_at, is_premium}."""
    try:
        r = requests.get(feed_url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"STAT RSS HTTP {r.status_code}")
            return []
    except Exception as e:
        logger.warning(f"STAT RSS request failed: {e}")
        return []
    feed = feedparser.parse(r.text)
    out: List[Dict] = []
    for e in feed.entries or []:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        # Strip ?utm_campaign= UTM by passing through normalize_url at insert time
        pub = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            try:
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        is_premium = title.startswith("STAT+:") or "STAT+" in title
        out.append({
            "headline": title[:400],
            "url": link,
            "summary": (e.get("summary") or "").strip()[:1500],
            "published_at": pub,
            "is_premium": is_premium,
        })
    return out


def fetch_stat_article_body(url: str, session: requests.Session,
                             timeout: int = 20) -> Optional[str]:
    """Fetch a STAT article and extract the body text. Returns None if
    the page is paywalled or fetch fails."""
    try:
        r = session.get(url, timeout=timeout)
    except Exception as e:
        logger.debug(f"STAT body fetch error {url[:80]}: {e}")
        return None
    if r.status_code != 200:
        logger.debug(f"STAT body HTTP {r.status_code} for {url[:80]}")
        return None
    body = r.text or ""
    if _is_paywall_blocked(body):
        logger.debug(f"STAT body PAYWALLED {url[:80]}")
        return None
    try:
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
            tag.decompose()
        # STAT articles use <article> tag with content inside
        article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile("entry-content|article-content"))
        text = ""
        if article:
            text = article.get_text(separator="\n", strip=True)
        if len(text) < 500:
            # Fallback: all <p> elements
            ps = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n".join([p for p in ps if len(p) > 30])
        return text[:50000] or None
    except Exception as e:
        logger.debug(f"STAT body parse failed {url[:80]}: {e}")
        return None


# ── Ticker mention extraction ──────────────────────────────

def _build_ticker_dict(conn) -> Dict[str, str]:
    """Load company_name -> ticker map from screener_stocks. Lower-cases
    company_name for case-insensitive matching. Skips short/ambiguous names."""
    out: Dict[str, str] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, company_name FROM screener_stocks WHERE company_name IS NOT NULL")
            for ticker, name in cur.fetchall():
                if not name or len(name) < 4:
                    continue
                # Strip trailing "Inc"/"Corp"/"Therapeutics" qualifiers for fuzzy match
                # but keep both forms in the dict
                clean = re.sub(r"\b(inc|corp|corporation|llc|ltd|sa|plc|nv|gmbh)\.?$",
                               "", name, flags=re.IGNORECASE).strip()
                out[name.lower()] = ticker
                if clean and clean.lower() != name.lower():
                    out[clean.lower()] = ticker
    except Exception as e:
        logger.warning(f"_build_ticker_dict failed: {e}")
    return out


def extract_ticker_mentions(
    text: str, ticker_dict: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Return list of (ticker, mention_method). May contain duplicate
    tickers from different methods — caller dedupes."""
    if not text:
        return []
    found: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def _add(t: str, method: str):
        t = t.upper().strip()
        if t and t not in seen:
            seen.add(t)
            found.append((t, method))

    for m in _CASHTAG_RE.finditer(text):
        _add(m.group(1), "cashtag")
    for m in _LISTING_RE.finditer(text):
        _add(m.group(1), "parens_listing")

    # Company-name match — lowercase and look for whole-word matches
    text_lower = text.lower()
    for name, ticker in ticker_dict.items():
        if ticker in seen:
            continue
        # Fast prefilter
        if name in text_lower:
            # Tighter: require word boundary
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, text_lower):
                _add(ticker, "company_name_match")
    return found


# ── Public entry point ──────────────────────────────────────

def collect_to_library(
    conn, *,
    feed_url: str = DEFAULT_FEED,
    fetch_bodies: bool = True,
    max_articles: int = 30,
) -> Dict:
    """Pull STAT RSS, optionally fetch bodies for each article, extract
    ticker mentions, persist to catalyst_event_news. Returns aggregate
    stats. Single-transaction batch; caller commits."""
    items = fetch_stat_rss(feed_url)[:max_articles]
    if not items:
        return {"items_fetched": 0, "articles_inserted": 0, "errors": "rss empty"}

    session = _build_session()
    ticker_dict = _build_ticker_dict(conn)

    inserted = 0
    enriched = 0
    untouched = 0
    bodies_fetched = 0
    bodies_paywalled = 0
    articles_with_no_ticker = 0

    for it in items:
        body = None
        if fetch_bodies:
            body = fetch_stat_article_body(it["url"], session)
            if body:
                bodies_fetched += 1
            else:
                bodies_paywalled += 1

        # Extract ticker mentions from headline + summary + body
        text_for_extraction = "\n".join([
            it["headline"] or "",
            it.get("summary") or "",
            body or "",
        ])
        mentions = extract_ticker_mentions(text_for_extraction, ticker_dict)
        if not mentions:
            articles_with_no_ticker += 1
            continue

        source = "stat_plus" if (it["is_premium"] and body) else "stat_news_rss"

        for ticker, method in mentions:
            res = insert_news_row(
                conn,
                ticker=ticker,
                source=source,
                url=it["url"],
                headline=it["headline"],
                summary=it["summary"],
                body=body,
                published_at=it["published_at"],
                discovery_path="forward_collection",
                ticker_mention_method=method,
            )
            if res is None:
                untouched += 1
            elif res.get("action") == "inserted":
                inserted += 1
            elif res.get("action") == "enriched":
                enriched += 1
            else:
                untouched += 1

    return {
        "items_fetched": len(items),
        "bodies_fetched": bodies_fetched,
        "bodies_paywalled": bodies_paywalled,
        "articles_with_no_ticker_match": articles_with_no_ticker,
        "rows_inserted": inserted,
        "rows_enriched": enriched,
        "rows_untouched": untouched,
    }
