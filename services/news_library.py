"""Shared news-library writer.

Used by every news source path that lands articles in
catalyst_event_news:
  - Finviz quote-page parser (services/feature_store._fill_finviz_news)
  - SA XML feed fetcher (services/news_sa_xml)
  - RSS forward collector (services/news_rss_collector)
  - Home-PC SA Premium body scraper (POSTs via /admin/news/ingest)
  - Multi-LLM consensus Pass-1 grounding-chunk merge

One canonical normalize_url() + url_hash() so the same press release
syndicated across PR Newswire / GlobeNewswire / BusinessWire (each with
different UTM params) collapses to one row, not three.
"""
from __future__ import annotations
import hashlib
import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

logger = logging.getLogger(__name__)


# Strip these tracking params; keep the rest of the query string intact.
# Conservative list — only universally tracking-only params. We keep
# things like ?id=, ?article=, ?ticker= which can be path-meaningful.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "utm_brand",
    "fbclid", "gclid", "gbraid", "wbraid", "msclkid", "yclid", "dclid",
    "mc_cid", "mc_eid",
    "_ga", "_gl",
    "ref", "ref_src",  # twitter ref params
    "source",          # SA-style "?source=feed_symbol_MRNA"
}


def normalize_url(url: str) -> str:
    """Canonicalize a URL for dedup. Drops tracking params, lowercases the
    host, removes the fragment, drops a trailing slash on the path. Keeps
    the path's case (some publishers care). Idempotent."""
    if not url:
        return ""
    try:
        s = urlsplit(url.strip())
        scheme = s.scheme.lower() or "https"
        netloc = s.netloc.lower()
        path = s.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # Filter query
        kept = [(k, v) for (k, v) in parse_qsl(s.query, keep_blank_values=True)
                if k.lower() not in _TRACKING_PARAMS]
        query = urlencode(kept, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))  # drop fragment
    except Exception as e:
        logger.warning(f"normalize_url({url[:80]}) failed: {e}")
        return (url or "").strip()


def url_hash(url: str) -> str:
    """SHA-256 hex of the normalized URL."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


# ── DB helpers ────────────────────────────────────────────────

# Caller passes either a psycopg2 connection or a BiotechDatabase-like
# object that exposes get_conn(). We keep the signature flexible.

_INSERT_SQL = """
    INSERT INTO catalyst_event_news (
        ticker, catalyst_id, catalyst_date, source, url, url_hash,
        headline, summary, body, published_at,
        discovery_path, ticker_mention_method
    )
    VALUES (
        %(ticker)s, %(catalyst_id)s, %(catalyst_date)s, %(source)s, %(url)s, %(url_hash)s,
        %(headline)s, %(summary)s, %(body)s, %(published_at)s,
        %(discovery_path)s, %(ticker_mention_method)s
    )
    ON CONFLICT (ticker, url_hash) DO UPDATE SET
        -- Enrich existing row with body when a body-fetcher (e.g. home-PC
        -- SA Premium scraper) finds one. Don't downgrade an existing body
        -- to NULL by overwriting with EXCLUDED.body unconditionally.
        body = COALESCE(EXCLUDED.body, catalyst_event_news.body),
        -- Backfill summary if the new row has one and the existing doesn't.
        summary = COALESCE(catalyst_event_news.summary, EXCLUDED.summary),
        -- Mark the source as the "richer" one — premium > xml feed.
        source = CASE
            WHEN EXCLUDED.body IS NOT NULL AND catalyst_event_news.body IS NULL
            THEN EXCLUDED.source
            ELSE catalyst_event_news.source
        END
    -- Return the row ID only when we actually changed something (body
    -- was added). xmax=0 means it was inserted (not updated); we treat
    -- updates that added body as "newly enriched" by checking if body
    -- changed. Simpler: return the id when EXCLUDED.body is not null
    -- and the existing body was null.
    RETURNING (
        CASE
            WHEN xmax = 0 THEN 'inserted'
            WHEN body IS NOT NULL THEN 'enriched'
            ELSE 'untouched'
        END
    ) AS action, id
"""


def insert_news_row(
    conn,
    *,
    ticker: str,
    source: str,
    url: str,
    headline: str,
    summary: Optional[str] = None,
    body: Optional[str] = None,
    published_at=None,                         # datetime | None
    catalyst_id: Optional[int] = None,
    catalyst_date: Optional[str] = None,
    discovery_path: Optional[str] = None,
    ticker_mention_method: Optional[str] = None,
) -> Optional[int]:
    """Insert one row into catalyst_event_news.

    Returns the new id if inserted, or None on conflict (already seen).
    Caller is responsible for commit (we do not commit per-row to allow
    bulk-insert callers to batch).

    `conn` may be a raw psycopg2 connection.
    """
    if not ticker or not url or not headline:
        return None
    nu = normalize_url(url)
    if not nu:
        return None
    params = {
        "ticker": ticker.upper().strip(),
        "catalyst_id": catalyst_id,
        "catalyst_date": catalyst_date,
        "source": source,
        "url": nu,
        "url_hash": url_hash(nu),
        "headline": headline.strip()[:1000],
        "summary": (summary or "").strip()[:4000] or None,
        "body": (body or None),                 # bodies can be long; no hard cap
        "published_at": published_at,
        "discovery_path": discovery_path,
        "ticker_mention_method": ticker_mention_method,
    }
    try:
        with conn.cursor() as cur:
            cur.execute(_INSERT_SQL, params)
            row = cur.fetchone()
            if not row:
                return None
            # Row is (action, id) — action ∈ {inserted, enriched, untouched}
            return {"action": row[0], "id": row[1]}
    except Exception as e:
        logger.warning(f"insert_news_row({ticker}, {source}) failed: {e}")
        return None


def insert_news_rows_bulk(conn, rows: list) -> Tuple[int, int, int]:
    """Convenience: bulk-insert a list of dicts (same keys as
    insert_news_row's kwargs). Returns (inserted, enriched, untouched).
    Single transaction; caller commits.
    """
    inserted = 0
    enriched = 0
    untouched = 0
    for r in rows:
        try:
            res = insert_news_row(conn, **r)
            if res is None:
                untouched += 1
            elif res["action"] == "inserted":
                inserted += 1
            elif res["action"] == "enriched":
                enriched += 1
            else:
                untouched += 1
        except Exception as e:
            logger.warning(f"bulk insert row failed: {e}")
            untouched += 1
    return inserted, enriched, untouched
