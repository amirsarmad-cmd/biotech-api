"""Extended news fetching — PARALLEL execution across all providers."""
import os, requests, time, logging
from datetime import datetime, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser

logger = logging.getLogger(__name__)

_news_cache = {}
_CACHE_TTL = 6 * 3600  # 6 hours

def _cached(key, fn):
    now = time.time()
    if key in _news_cache:
        ts, val = _news_cache[key]
        if now - ts < _CACHE_TTL: return val
    val = fn()
    _news_cache[key] = (now, val)
    return val

def fetch_finnhub(ticker, cap=20):
    fk = os.getenv("FINNHUB_API_KEY","")
    if not fk: return []
    try:
        fd = (datetime.now()-timedelta(days=90)).strftime("%Y-%m-%d")
        td = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fd}&to={td}&token={fk}", timeout=8)
        items = r.json()[:cap]
        return [{"title": i.get("headline",""), "source": i.get("source","Finnhub"),
                 "url": i.get("url",""),
                 "date": datetime.fromtimestamp(i.get("datetime",0)).strftime("%Y-%m-%d") if i.get("datetime") else "",
                 "summary": i.get("summary","")[:400], "provider": "Finnhub"} for i in items]
    except Exception as e:
        logger.warning(f"Finnhub {ticker}: {e}"); return []

def fetch_newsapi(company, catalyst, cap=15):
    nk = os.getenv("NEWSAPI_KEY","")
    if not nk: return []
    try:
        q = quote(f"{company} {catalyst} FDA biotech")
        r = requests.get(f"https://newsapi.org/v2/everything?q={q}&sortBy=publishedAt&pageSize={cap}&apiKey={nk}", timeout=8)
        articles = r.json().get("articles",[])[:cap]
        return [{"title": a.get("title",""), "source": a.get("source",{}).get("name","NewsAPI"),
                 "url": a.get("url",""), "date": a.get("publishedAt","")[:10],
                 "summary": a.get("description","")[:400], "provider": "NewsAPI"} for a in articles]
    except Exception as e:
        logger.warning(f"NewsAPI {company}: {e}"); return []

def fetch_yahoo(ticker, cap=15):
    """yfinance has changed news shape over versions. Handle both:
       - Legacy flat: {title, link, publisher, providerPublishTime, summary}
       - New nested: {content: {title, description, pubDate, provider:{displayName},
                                canonicalUrl:{url}, clickThroughUrl:{url}, summary}}
    """
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        out = []
        for n in news[:cap]:
            # Try new nested format first
            content = n.get("content") if isinstance(n.get("content"), dict) else None
            if content:
                title = content.get("title","")
                summary = (content.get("summary") or content.get("description") or "")[:400]
                # URL can be in canonicalUrl.url or clickThroughUrl.url
                url = ""
                for key in ("clickThroughUrl", "canonicalUrl"):
                    u = content.get(key)
                    if isinstance(u, dict) and u.get("url"):
                        url = u["url"]; break
                    if isinstance(u, str):
                        url = u; break
                publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else "Yahoo"
                # Date
                pub = content.get("pubDate") or content.get("displayTime") or ""
                try:
                    if "T" in str(pub):
                        date = str(pub).split("T")[0]
                    else:
                        date = str(pub)[:10]
                except:
                    date = ""
            else:
                # Legacy format
                title = n.get("title","")
                summary = (n.get("summary") or "")[:400]
                url = n.get("link","") or n.get("url","")
                publisher = n.get("publisher","Yahoo")
                ts = n.get("providerPublishTime", 0)
                try:
                    date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
                except:
                    date = ""
            
            # Fallback: construct Yahoo Finance URL from ticker if no URL available
            if not url and title:
                url = f"https://finance.yahoo.com/quote/{ticker}/news"
            
            if title:
                out.append({"title": title, "source": publisher, "url": url,
                            "date": date, "summary": summary, "provider": "Yahoo Finance"})
        return out
    except Exception as e:
        logger.warning(f"Yahoo {ticker}: {e}"); return []

def fetch_benzinga_rss(ticker, cap=10):
    try:
        feed = feedparser.parse(f"https://www.benzinga.com/stock/{ticker.lower()}/rss")
        return [{"title": e.get("title",""), "source": "Benzinga",
                 "url": e.get("link",""), "date": e.get("published","")[:10],
                 "summary": e.get("summary","")[:400], "provider": "Benzinga"}
                for e in feed.entries[:cap]]
    except Exception as e:
        logger.warning(f"Benzinga {ticker}: {e}"); return []

def fetch_fierce_biotech(cap=5):
    try:
        feed = feedparser.parse("https://www.fiercebiotech.com/rss/xml")
        return [{"title": e.get("title",""), "source": "FierceBiotech",
                 "url": e.get("link",""), "date": e.get("published","")[:10],
                 "summary": e.get("summary","")[:400], "provider": "FierceBiotech"}
                for e in feed.entries[:cap]]
    except Exception as e:
        logger.warning(f"FierceBiotech: {e}"); return []

def fetch_stocktwits(ticker, cap=10):
    try:
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json", timeout=6)
        msgs = r.json().get("messages",[])[:cap]
        return [{"title": (m.get("body","")[:80] + "..." if len(m.get("body",""))>80 else m.get("body","")),
                 "source": f"@{m.get('user',{}).get('username','anon')}",
                 "url": f"https://stocktwits.com/{m.get('user',{}).get('username','')}/message/{m.get('id','')}",
                 "date": m.get("created_at","")[:10], "summary": m.get("body","")[:300],
                 "provider": "StockTwits"} for m in msgs]
    except Exception as e:
        logger.warning(f"StockTwits {ticker}: {e}"); return []

def fetch_tipranks_public(ticker, cap=8):
    try:
        r = requests.get(f"https://www.tipranks.com/api/stocks/getNews?ticker={ticker}",
            headers={"User-Agent":"Mozilla/5.0","Referer":f"https://www.tipranks.com/stocks/{ticker}"},
            timeout=8)
        if r.status_code != 200: return []
        items = r.json().get("news",[])[:cap] if isinstance(r.json(), dict) else []
        return [{"title": i.get("title",""), "source": i.get("source","TipRanks"),
                 "url": i.get("link",""), "date": i.get("published","")[:10] if i.get("published") else "",
                 "summary": i.get("description","")[:400], "provider": "TipRanks"} for i in items]
    except Exception as e:
        logger.warning(f"TipRanks {ticker}: {e}"); return []

def _run_playwright_login(ticker, site):
    """Common Playwright login routine. site = 'tipranks' or 'sa'."""
    if site == "tipranks":
        user, pwd = os.getenv("TIPRANKS_USER"), os.getenv("TIPRANKS_PASS")
        if not user or not pwd: return []
        login_url, news_url = "https://www.tipranks.com/login", f"https://www.tipranks.com/stocks/{ticker}/stock-news"
        label = "TipRanks (Premium)"
        provider = "TipRanks-Login"
    else:
        user, pwd = os.getenv("SA_USER"), os.getenv("SA_PASS")
        if not user or not pwd: return []
        login_url, news_url = "https://seekingalpha.com/login", f"https://seekingalpha.com/symbol/{ticker}/news"
        label = "Seeking Alpha (Premium)"
        provider = "SA-Login"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
            ctx = browser.new_context(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
            page = ctx.new_page()
            page.goto(login_url, timeout=20000); time.sleep(3)
            try:
                page.fill("input[type=email], input[name=email]", user); time.sleep(1)
                page.fill("input[type=password], input[name=password]", pwd); time.sleep(1)
                page.click("button[type=submit]"); time.sleep(5)
            except: pass
            page.goto(news_url, timeout=20000); time.sleep(3)
            articles = page.evaluate('''() => {
                const sel = 'article, [class*="news"], [data-test-id="post-list-item"]';
                return Array.from(document.querySelectorAll(sel)).slice(0,10).map(i => ({
                    title: i.querySelector('h2,h3,h4,a[data-test-id="post-list-item-title"]')?.innerText || '',
                    url: i.querySelector('a')?.href || '',
                    summary: (i.innerText || '').substring(0, 400)
                })).filter(a => a.title);
            }''')
            browser.close()
            return [{"title": a["title"], "source": label, "url": a["url"],
                     "date": datetime.now().strftime("%Y-%m-%d"),
                     "summary": a["summary"], "provider": provider} for a in articles[:10]]
    except Exception as e:
        logger.warning(f"{site} login {ticker}: {e}"); return []

def fetch_tipranks_logged_in(ticker, cap=10):
    return _run_playwright_login(ticker, "tipranks")[:cap]

def fetch_sa_logged_in(ticker, cap=10):
    return _run_playwright_login(ticker, "sa")[:cap]


def fetch_all_sources(ticker, company, catalyst):
    """Fetch from ALL providers in PARALLEL. Cached per-ticker for 6h."""
    def _do():
        fetchers = [
            ("finnhub",         lambda: fetch_finnhub(ticker, cap=20)),
            ("newsapi",         lambda: fetch_newsapi(company, catalyst, cap=15)),
            ("yahoo",           lambda: fetch_yahoo(ticker, cap=10)),
            ("benzinga",        lambda: fetch_benzinga_rss(ticker, cap=8)),
            ("stocktwits",      lambda: fetch_stocktwits(ticker, cap=5)),
            ("tipranks_public", lambda: fetch_tipranks_public(ticker, cap=5)),
            ("fierce",          lambda: fetch_fierce_biotech(cap=3)),
            # Premium scrapers skipped in parallel (slow, would block) — run optionally after
        ]
        results = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(fetchers)) as ex:
            futs = {ex.submit(fn): name for name, fn in fetchers}
            for fut in as_completed(futs, timeout=20):
                try:
                    r = fut.result(timeout=12)
                    if r: results.extend(r)
                except Exception as e:
                    logger.warning(f"{futs[fut]} failed: {e}")
        # Optional premium — run only if creds set; do AFTER parallel fetch finishes
        if os.getenv("TIPRANKS_USER"): 
            try: results.extend(fetch_tipranks_logged_in(ticker, cap=8))
            except: pass
        if os.getenv("SA_USER"):
            try: results.extend(fetch_sa_logged_in(ticker, cap=8))
            except: pass
        results.append({"title": f"SEC EDGAR 8-K Filings — {ticker}", "source": "SEC EDGAR",
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K",
            "date": "", "summary": "Material announcements & FDA correspondence.", "provider": "SEC"})
        results.append({"title": "FDA Novel Drug Approvals Database", "source": "FDA",
            "url": "https://www.fda.gov/drugs/drug-approvals-and-databases/novel-drug-approvals-fda",
            "date": "", "summary": "PDUFA dates, approval history.", "provider": "FDA"})
        logger.info(f"Sources for {ticker}: {len(results)} items in {time.time()-t0:.1f}s")
        return results
    return _cached(f"news:{ticker}", _do)
