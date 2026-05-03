"""Seeking Alpha XML feed fetcher.

We discovered (admin/news/sa-direct-test) that
  https://seekingalpha.com/api/sa/combined/{TICKER}.xml
is a public RSS endpoint that bypasses PerimeterX from any IP — no auth,
no headless browser, just a plain HTTP GET. Returns ~30 most-recent SA
news items per ticker (titles + URLs + dates), including Premium-tier
article titles. Premium article *bodies* still require residential-IP
auth (handled by the home-PC scraper that POSTs to /admin/news/ingest).
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import requests
import feedparser

logger = logging.getLogger(__name__)


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)


def fetch_sa_xml(ticker: str, timeout_s: int = 15) -> List[Dict]:
    """Fetch and parse SA's per-ticker combined XML feed.

    Returns a list of dicts with: headline, url, published_at (datetime
    or None), summary. Caller maps these into catalyst_event_news rows
    via news_library.insert_news_row. Empty list on any failure (logged).
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []
    url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/xml, text/xml, */*",
            },
            timeout=timeout_s,
        )
    except Exception as e:
        logger.warning(f"SA XML {ticker}: request failed: {e}")
        return []
    if r.status_code != 200:
        logger.warning(f"SA XML {ticker}: HTTP {r.status_code}")
        return []
    body = r.text or ""
    if "px-captcha" in body.lower() or "access to this page" in body.lower():
        logger.warning(f"SA XML {ticker}: PerimeterX challenge served (unexpected for XML feed)")
        return []

    try:
        feed = feedparser.parse(body)
    except Exception as e:
        logger.warning(f"SA XML {ticker}: feedparser failed: {e}")
        return []

    items: List[Dict] = []
    for e in feed.entries or []:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        # Drop the ?source= UTM param via news_library.normalize_url at insert time
        published_at: Optional[datetime] = None
        # feedparser provides published_parsed (struct_time) when available
        if hasattr(e, "published_parsed") and e.published_parsed:
            try:
                published_at = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        summary = (e.get("summary") or "").strip()[:1000]
        items.append({
            "headline": title[:400],
            "url": link,
            "published_at": published_at,
            "summary": summary or None,
        })
    return items


def collect_to_library(conn, ticker: str, *, catalyst_id: Optional[int] = None,
                       catalyst_date: Optional[str] = None) -> Dict[str, int]:
    """Fetch SA XML for a ticker and persist via news_library.

    Returns {"fetched": N, "inserted": K, "skipped": N-K}.
    Single-transaction batch; caller commits.
    """
    from services.news_library import insert_news_row

    items = fetch_sa_xml(ticker)
    inserted = enriched = untouched = 0
    for it in items:
        res = insert_news_row(
            conn,
            ticker=ticker,
            source="seeking_alpha_xml",
            url=it["url"],
            headline=it["headline"],
            summary=it["summary"],
            published_at=it["published_at"],
            catalyst_id=catalyst_id,
            catalyst_date=catalyst_date,
            discovery_path="sa_xml_per_ticker",
            ticker_mention_method="sa_xml_ticker",
        )
        if res is None:
            untouched += 1
        elif res.get("action") == "inserted":
            inserted += 1
        elif res.get("action") == "enriched":
            enriched += 1
        else:
            untouched += 1
    return {"fetched": len(items), "inserted": inserted,
            "enriched": enriched, "untouched": untouched}
