"""Home-PC Seeking Alpha scraper.

Runs on the user's Windows machine (residential IP) every 6h via Task
Scheduler. Uses real-browser SA cookies to fetch Premium article bodies
that PerimeterX blocks from Railway, then POSTs the articles to the
Railway backend's authenticated /admin/news/ingest endpoint.

Why this exists: the public /api/sa/combined/{TICKER}.xml endpoint works
from Railway (no PerimeterX), so we get headlines + URLs there. But
SA's /article/ URLs (where the actual body lives) ARE behind PerimeterX,
which blocks all datacenter IPs — including Railway. The only way to
get bodies is from a residential IP with valid auth cookies. That's
this script.

Setup:
  1. Install deps:
     pip install requests beautifulsoup4 feedparser
  2. Create config dir: %USERPROFILE%/.biotech-news-scraper/
     - sa-cookies.json   (Cookie Editor export from logged-in browser)
     - ingest-token.txt  (one line: the value of NEWS_INGEST_TOKEN env on Railway)
     - tickers.txt       (one ticker per line; or rely on default biotech list)
  3. Test run:
     python home_pc_sa_scraper.py --tickers MRNA,PFE --max-bodies 3 --verbose
  4. Schedule via Windows Task Scheduler:
     Trigger: every 6 hours
     Action: python C:\\path\\to\\home_pc_sa_scraper.py
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── Config ───────────────────────────────────────────────────

DEFAULT_CONFIG_DIR = Path(os.environ.get("USERPROFILE", "~")).expanduser() / ".biotech-news-scraper"
DEFAULT_API_BASE = "https://biotech-api-production-7ec4.up.railway.app"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

# Default biotech tickers to scrape if no tickers.txt provided
DEFAULT_TICKERS = [
    "MRNA", "BNTX", "PFE", "JNJ", "MRK", "ABBV", "BMY", "LLY", "AMGN",
    "GILD", "REGN", "VRTX", "BIIB", "INCY", "ALNY", "BMRN", "EXEL", "JAZZ",
    "NBIX", "RIGL", "RYTM", "SAGE", "BHC", "CELG", "SRPT", "BLUE", "RARE",
    "FOLD", "CRSP", "EDIT", "NTLA", "BEAM", "VERV", "ARWR", "IONS", "MDGL",
    "VKTX", "ARQT", "PTGX", "CABA", "NVAX",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sa-scraper")


# ── Helpers ──────────────────────────────────────────────────

def load_cookies(path: Path) -> List[Dict]:
    """Load Cookie Editor JSON; return list of cookie dicts ready for
    requests.Session.cookies.set()."""
    if not path.exists():
        log.error(f"cookies file not found: {path}")
        log.error("export from Chrome via the Cookie Editor extension while"
                  " logged into seekingalpha.com → ⚙ → Export → JSON, then save here.")
        sys.exit(2)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in raw:
        if not c.get("name") or c.get("value") is None:
            continue
        out.append({
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", "").lstrip("."),
            "path": c.get("path", "/"),
        })
    log.info(f"loaded {len(out)} cookies from {path.name}")
    return out


def load_ingest_token(path: Path) -> str:
    if not path.exists():
        log.error(f"ingest token file not found: {path}")
        log.error("create it: file containing the value of NEWS_INGEST_TOKEN env on Railway")
        sys.exit(2)
    return path.read_text(encoding="utf-8").strip()


def load_tickers(path: Optional[Path]) -> List[str]:
    if path and path.exists():
        lines = [ln.strip().upper() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [t for t in lines if t and not t.startswith("#")]
    return list(DEFAULT_TICKERS)


def make_session(cookies: List[Dict]):
    import requests
    s = requests.Session()
    for c in cookies:
        s.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


# ── SA fetch logic ──────────────────────────────────────────

def fetch_sa_xml(session, ticker: str, timeout: int = 15) -> List[Dict]:
    """Fetch the public SA combined XML feed for a ticker. Returns list of
    {headline, url, published_at, summary} dicts. Same shape as the
    Railway-side fetcher; we run from home-PC anyway because it's cheap
    and lets us collect bodies in the same pass."""
    import feedparser
    try:
        url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        r = session.get(url, timeout=timeout,
                        headers={"Accept": "application/xml, text/xml, */*"})
        if r.status_code != 200:
            log.warning(f"{ticker} XML feed HTTP {r.status_code}")
            return []
        if "px-captcha" in r.text.lower():
            log.warning(f"{ticker} XML feed challenged by PerimeterX (unexpected)")
            return []
        feed = feedparser.parse(r.text)
        out = []
        for e in feed.entries or []:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue
            pub = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                try:
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            out.append({
                "headline": title[:400],
                "url": link,
                "published_at": pub,
                "summary": (e.get("summary") or "").strip()[:1000] or None,
            })
        return out
    except Exception as e:
        log.warning(f"{ticker} XML fetch failed: {e}")
        return []


def fetch_sa_article_body(session, url: str, timeout: int = 20) -> Optional[str]:
    """Fetch the body of a SA Premium article. Returns the extracted text
    body, or None if PerimeterX challenged us / not a real article."""
    try:
        r = session.get(url, timeout=timeout)
    except Exception as e:
        log.debug(f"  body fetch exception {url[:80]}: {e}")
        return None
    if r.status_code != 200:
        log.debug(f"  body fetch HTTP {r.status_code} for {url[:80]}")
        return None
    body = r.text or ""
    # Tighter PerimeterX-block detection. The real captcha page is small
    # (~3-5 KB) and has <title>Access to this page has been denied</title>.
    # Real article pages can be 100+ KB and may CONTAIN a px-captcha
    # telemetry script in <head>, but that's passive monitoring, not a
    # block. So check size + page title, not just substring presence.
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body, "html.parser")
        title_text = (soup.title.string if soup.title and soup.title.string else "").strip().lower()
        meta_desc = ""
        m = soup.find("meta", attrs={"name": "description"})
        if m and m.get("content"):
            meta_desc = m["content"].strip().lower()
        is_block_page = (
            len(body) < 15000 and
            ("access to this page has been denied" in title_text or meta_desc == "px-captcha")
        )
        if is_block_page:
            log.debug(f"  body PerimeterX-blocked (title={title_text[:60]}): {url[:80]}")
            return None
        # Strip script/style, then collect article paragraphs
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        # SA's article body is usually inside <article> or within div[data-test-id="article-content"]
        candidates = []
        candidates.extend(soup.select("[data-test-id='content-container']"))
        candidates.extend(soup.select("div[data-test-id*='article']"))
        candidates.extend(soup.find_all("article"))
        text = ""
        if candidates:
            text = candidates[0].get_text(separator="\n", strip=True)
        if not text or len(text) < 200:
            # Fallback: grab all <p> text
            ps = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n".join([p for p in ps if len(p) > 30])
        # Cap at 50K chars (long-form SA articles can hit ~20K typical)
        return text[:50000] or None
    except Exception as e:
        log.debug(f"  parse failed {url[:80]}: {e}")
        return None


def is_premium_article_url(url: str) -> bool:
    """True if URL looks like a real SA article body URL (worth fetching)
    rather than a redirect-to-news-index URL. The XML feed serves both."""
    u = url.lower()
    return "/article/" in u and "/news/" not in u


# ── Ingest ──────────────────────────────────────────────────

def post_to_ingest(api_base: str, token: str, articles: List[Dict],
                   batch_size: int = 50, timeout: int = 30) -> Dict[str, int]:
    """Batch-POST articles to /admin/news/ingest. Returns aggregate counts."""
    import requests
    totals = {"received": 0, "inserted": 0, "enriched": 0,
              "untouched": 0, "errors_count": 0}
    for i in range(0, len(articles), batch_size):
        chunk = articles[i:i + batch_size]
        payload = {"articles": chunk}
        try:
            r = requests.post(
                f"{api_base.rstrip('/')}/admin/news/ingest",
                json=payload,
                headers={"X-Ingest-Token": token, "Content-Type": "application/json"},
                timeout=timeout,
            )
            if r.status_code != 200:
                log.warning(f"  ingest batch HTTP {r.status_code}: {r.text[:200]}")
                totals["errors_count"] += len(chunk)
                continue
            data = r.json()
            totals["received"] += data.get("received", 0)
            totals["inserted"] += data.get("inserted", 0)
            totals["enriched"] += data.get("enriched", 0)
            totals["untouched"] += data.get("untouched", data.get("skipped", 0))
        except Exception as e:
            log.warning(f"  ingest batch failed: {e}")
            totals["errors_count"] += len(chunk)
    return totals


# ── Main ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Home-PC SA scraper for biotech consensus library.")
    p.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--api-base", default=os.environ.get("BIOTECH_API_BASE", DEFAULT_API_BASE))
    p.add_argument("--tickers", default=None,
                   help="Comma-separated ticker list. If omitted, reads tickers.txt or uses defaults.")
    p.add_argument("--max-bodies", type=int, default=10,
                   help="Max Premium article bodies to fetch per ticker (rate-limit guard).")
    p.add_argument("--ticker-sleep", type=float, default=2.0,
                   help="Seconds to sleep between tickers.")
    p.add_argument("--body-sleep", type=float, default=1.0,
                   help="Seconds to sleep between body fetches.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Do everything except the final POST to Railway.")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cookies = load_cookies(args.config_dir / "sa-cookies.json")
    token = load_ingest_token(args.config_dir / "ingest-token.txt")
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else load_tickers(args.config_dir / "tickers.txt")
    )

    log.info(f"scraping {len(tickers)} tickers, max {args.max_bodies} bodies/ticker, api={args.api_base}")
    session = make_session(cookies)

    # Sanity: hit a single SA URL to validate cookies didn't expire
    try:
        r = session.get("https://seekingalpha.com/api/sa/combined/MRNA.xml", timeout=15)
        if r.status_code != 200:
            log.error(f"SA sanity HTTP {r.status_code} — cookies may have expired")
            sys.exit(3)
    except Exception as e:
        log.error(f"SA sanity check failed: {e}")
        sys.exit(3)

    all_articles: List[Dict] = []
    summary_per_ticker: Dict[str, Dict] = {}

    for idx, ticker in enumerate(tickers, 1):
        log.info(f"[{idx}/{len(tickers)}] {ticker}")
        items = fetch_sa_xml(session, ticker)
        log.info(f"  XML: {len(items)} items")
        article_urls = [it for it in items if is_premium_article_url(it["url"])]
        log.info(f"  /article/ URLs: {len(article_urls)} (capped to {args.max_bodies})")

        bodies_fetched = 0
        bodies_blocked = 0
        articles_for_ticker: List[Dict] = []

        # Always submit headlines (with empty body) — Railway already has these from
        # the SA-XML collector, but ON CONFLICT DO NOTHING dedupes on (ticker, url_hash)
        for it in items:
            articles_for_ticker.append({
                "ticker": ticker,
                "source": "seeking_alpha_xml",
                "url": it["url"],
                "headline": it["headline"],
                "summary": it["summary"],
                "body": None,
                "published_at": it["published_at"],
                "discovery_path": "home_pc_scraper",
                "ticker_mention_method": "sa_xml_ticker",
            })

        # Fetch bodies for the top N /article/ URLs
        for it in article_urls[:args.max_bodies]:
            body = fetch_sa_article_body(session, it["url"])
            if body is None:
                bodies_blocked += 1
            else:
                bodies_fetched += 1
                # Insert as `seeking_alpha_premium` (different source so we
                # can distinguish "headline-only XML" from "body-fetched")
                articles_for_ticker.append({
                    "ticker": ticker,
                    "source": "seeking_alpha_premium",
                    "url": it["url"],
                    "headline": it["headline"],
                    "summary": it["summary"],
                    "body": body,
                    "published_at": it["published_at"],
                    "discovery_path": "home_pc_scraper",
                    "ticker_mention_method": "sa_xml_ticker",
                })
            if args.body_sleep > 0:
                time.sleep(args.body_sleep)

        log.info(f"  bodies: {bodies_fetched} fetched, {bodies_blocked} blocked")
        all_articles.extend(articles_for_ticker)
        summary_per_ticker[ticker] = {
            "xml_items": len(items),
            "article_urls": len(article_urls),
            "bodies_fetched": bodies_fetched,
            "bodies_blocked": bodies_blocked,
        }
        if args.ticker_sleep > 0 and idx < len(tickers):
            time.sleep(args.ticker_sleep)

    log.info(f"total articles to ingest: {len(all_articles)}")

    if args.dry_run:
        log.info("DRY RUN — skipping ingest POST")
        log.info(f"summary: {json.dumps(summary_per_ticker, indent=2)}")
        return

    log.info("posting to Railway /admin/news/ingest …")
    totals = post_to_ingest(args.api_base, token, all_articles)
    log.info(f"INGEST done: {json.dumps(totals)}")
    log.info(f"per-ticker summary: {json.dumps(summary_per_ticker)}")


if __name__ == "__main__":
    main()
