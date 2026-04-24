"""
Social sentiment sources: StockTwits + Reddit.

StockTwits: free public API, returns recent messages with bullish/bearish sentiment tags.
Reddit: Pushshift-compatible + Reddit.com JSON APIs, free, covers r/biotech, r/wallstreetbets,
        r/stocks, r/options, r/biostocks mentions.

Returns unified format consistent with fetcher_news.py + research_enrichment.py:
    {"title": "...", "source": "StockTwits" | "Reddit", "url": "...",
     "date": "YYYY-MM-DD", "summary": "...", "provider": "...",
     "sentiment": "bullish" | "bearish" | "neutral",  # social-specific
     "score": int,  # upvotes / likes
     "comments": int}
"""
import os
import re
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_social_cache: Dict = {}
_CACHE_TTL = 30 * 60  # 30 min — social moves fast


def _cached(key: str, fn):
    now = time.time()
    if key in _social_cache:
        ts, val = _social_cache[key]
        if now - ts < _CACHE_TTL: return val
    val = fn()
    _social_cache[key] = (now, val)
    return val


# ============================================================
# STOCKTWITS: public /symbols/{ticker}.json endpoint
# ============================================================

def fetch_stocktwits(ticker: str, cap: int = 30) -> List[Dict]:
    """Fetch latest StockTwits messages for a ticker. Public endpoint.
    
    Returns up to 30 messages with title, sentiment (bullish/bearish/null), user stats.
    """
    def _do():
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; biotech-screener/1.0)"}
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json?limit={cap}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                logger.warning(f"StockTwits {ticker} HTTP {r.status_code}")
                return []
            data = r.json()
            messages = data.get("messages", [])
            results = []
            for m in messages[:cap]:
                body = m.get("body", "")
                if not body or len(body) < 10: continue
                
                # Sentiment tag (StockTwits-native)
                entities = m.get("entities") or {}
                sentiment_obj = entities.get("sentiment") if isinstance(entities, dict) else None
                sent = None
                if isinstance(sentiment_obj, dict):
                    sent = sentiment_obj.get("basic", "").lower()
                sentiment = "bullish" if sent == "bullish" else ("bearish" if sent == "bearish" else "neutral")
                
                # User engagement stats
                user = m.get("user") or {}
                followers = user.get("followers", 0)
                
                date_str = m.get("created_at", "")[:10] if m.get("created_at") else ""
                msg_id = m.get("id", "")
                
                results.append({
                    "title": body[:200],  # message body becomes "title" for UI
                    "source": "StockTwits",
                    "url": f"https://stocktwits.com/message/{msg_id}" if msg_id else f"https://stocktwits.com/symbol/{ticker}",
                    "date": date_str,
                    "summary": body[:400],
                    "provider": "StockTwits",
                    "sentiment": sentiment,
                    "author": user.get("username", "anonymous"),
                    "author_followers": followers,
                    "likes": m.get("likes", {}).get("total", 0) if isinstance(m.get("likes"), dict) else 0,
                    "reshares": m.get("reshares", {}).get("reshared_count", 0) if isinstance(m.get("reshares"), dict) else 0,
                })
            return results
        except Exception as e:
            logger.warning(f"StockTwits {ticker}: {e}")
            return []
    
    return _cached(f"st:{ticker}", _do)


def stocktwits_sentiment_summary(messages: List[Dict]) -> Dict:
    """Aggregate sentiment across StockTwits messages.
    
    Returns:
    {
        "total": 30,
        "bullish": 18,
        "bearish": 5,
        "neutral": 7,
        "bullish_pct": 60.0,
        "bearish_pct": 16.7,
        "sentiment_score": 0.43,  # -1 to +1
        "top_bullish": [top 3 highest-engagement bullish msgs],
        "top_bearish": [top 3 highest-engagement bearish msgs],
    }
    """
    if not messages:
        return {"total": 0, "bullish": 0, "bearish": 0, "neutral": 0,
                "bullish_pct": 0, "bearish_pct": 0, "sentiment_score": 0,
                "top_bullish": [], "top_bearish": []}
    
    bull = [m for m in messages if m.get("sentiment") == "bullish"]
    bear = [m for m in messages if m.get("sentiment") == "bearish"]
    neutral = [m for m in messages if m.get("sentiment") == "neutral"]
    total = len(messages)
    
    bull_pct = len(bull) / total * 100 if total else 0
    bear_pct = len(bear) / total * 100 if total else 0
    # Sentiment score: net bullish, weighted
    score = (len(bull) - len(bear)) / total if total else 0
    
    # Engagement weighting: likes + reshares + followers/100
    def engagement(m):
        return m.get("likes", 0) + m.get("reshares", 0) * 2 + (m.get("author_followers", 0) / 100)
    
    top_bull = sorted(bull, key=engagement, reverse=True)[:3]
    top_bear = sorted(bear, key=engagement, reverse=True)[:3]
    
    return {
        "total": total,
        "bullish": len(bull),
        "bearish": len(bear),
        "neutral": len(neutral),
        "bullish_pct": round(bull_pct, 1),
        "bearish_pct": round(bear_pct, 1),
        "sentiment_score": round(score, 2),
        "top_bullish": top_bull,
        "top_bearish": top_bear,
    }


# ============================================================
# REDDIT: OAuth-first, with unauth fallback
# ============================================================

_reddit_token_cache = {"token": None, "expires_at": 0}


def _get_reddit_oauth_token():
    """Get a client_credentials OAuth token. Cached in memory for 55min."""
    import time as _t
    if _reddit_token_cache["token"] and _reddit_token_cache["expires_at"] > _t.time() + 60:
        return _reddit_token_cache["token"]
    
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    
    import base64
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "User-Agent": "biotech-screener/1.0",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    try:
        r = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            headers=headers,
            data="grant_type=client_credentials",
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            tok = data.get("access_token")
            expires_in = int(data.get("expires_in", 3600))
            _reddit_token_cache["token"] = tok
            _reddit_token_cache["expires_at"] = _t.time() + expires_in - 60
            logger.info(f"Reddit OAuth token refreshed, expires in {expires_in}s")
            return tok
        else:
            logger.warning(f"Reddit OAuth {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        logger.warning(f"Reddit OAuth error: {e}")
        return None


def fetch_reddit(ticker: str, company: str = "", cap: int = 30,
                 days_back: int = 30) -> List[Dict]:
    """Search relevant subreddits for recent mentions of ticker/company.
    
    Prefers OAuth via REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET env vars (rate limit ~60 req/min).
    Falls back to unauthenticated reddit.com/.json endpoints (often blocked by cloud IPs).
    """
    def _do():
        results = []
        
        oauth_token = _get_reddit_oauth_token()
        if oauth_token:
            base_url = "https://oauth.reddit.com"
            headers = {
                "Authorization": f"Bearer {oauth_token}",
                "User-Agent": "biotech-screener/1.0",
            }
            logger.debug("Using Reddit OAuth")
        else:
            base_url = "https://www.reddit.com"
            headers = {"User-Agent": "biotech-screener/1.0 (by /u/anonymous)"}
            logger.debug("Using unauthenticated Reddit (may be rate-limited)")
        
        queries = [ticker.upper()]
        if company and len(company) > 3:
            company_clean = re.split(r'[\s,.\-]+', company.strip())[0]
            if len(company_clean) >= 4 and company_clean.lower() != ticker.lower():
                queries.append(company_clean)
        
        seen_ids = set()
        per_subreddit_cap = max(3, cap // len(RELEVANT_SUBREDDITS))
        
        def fetch_one_subreddit(sub_name: str, query: str) -> List[Dict]:
            try:
                url = (f"{base_url}/r/{sub_name}/search.json"
                       f"?q={quote(query)}&restrict_sr=1&sort=new&limit={per_subreddit_cap}&t=month")
                # OAuth endpoint needs a raw query (no .json suffix sometimes)
                if oauth_token and ".json" in url:
                    url = url.replace(".json", "")
                r = requests.get(url, headers=headers, timeout=8)
                if r.status_code != 200:
                    return []
                data = r.json()
                items = []
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    post_id = post.get("id")
                    if not post_id or post_id in seen_ids: continue
                    created_ts = post.get("created_utc", 0)
                    if created_ts:
                        created_dt = datetime.fromtimestamp(created_ts)
                        if (datetime.now() - created_dt).days > days_back: continue
                        date_str = created_dt.strftime("%Y-%m-%d")
                    else:
                        date_str = ""
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")[:500]
                    if not title: continue
                    text_check = (title + " " + selftext).lower()
                    if ticker.lower() not in text_check and (not company or company.lower()[:5] not in text_check):
                        continue
                    sentiment = _classify_reddit_sentiment(title + " " + selftext)
                    items.append({
                        "id": post_id,
                        "title": title[:250],
                        "source": f"Reddit /r/{sub_name}",
                        "url": f"https://reddit.com{post.get('permalink','')}",
                        "date": date_str,
                        "summary": selftext[:400],
                        "provider": f"Reddit /r/{sub_name}",
                        "sentiment": sentiment,
                        "author": post.get("author", "anonymous"),
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                        "subreddit": sub_name,
                    })
                return items
            except Exception as e:
                logger.debug(f"Reddit {sub_name}: {e}")
                return []
        
        pairs = [(sub, q) for sub in RELEVANT_SUBREDDITS[:6] for q in queries[:2]]
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_one_subreddit, sub, q): (sub, q) for sub, q in pairs}
            for fut in as_completed(futs, timeout=20):
                try:
                    items = fut.result(timeout=12)
                    for it in items:
                        if it["id"] not in seen_ids:
                            seen_ids.add(it["id"])
                            results.append(it)
                except Exception as e:
                    logger.debug(f"reddit subfut failed: {e}")
        
        results.sort(key=lambda x: (-(x.get("score", 0) or 0), x.get("date", "")), reverse=False)
        return results[:cap]
    
    return _cached(f"reddit:{ticker}", _do)



def _classify_reddit_sentiment(text: str) -> str:
    """Rough sentiment from Reddit post content.
    Bullish: calls, moon, squeeze, pump, approval, buy, long, bullish, YOLO
    Bearish: puts, tank, crash, CRL, reject, sell, short, bearish, dumpster
    """
    text_l = text.lower()[:600]
    bull_words = ["calls", "moon", "🚀", "squeeze", "pump", "approval", "approved", "buy",
                  "long", "bullish", "yolo", "rocket", "to the moon", "upside", "breakout"]
    bear_words = ["puts", "tank", "crash", "crl", "reject", "rejected", "sell", "short",
                  "bearish", "dumpster", "bust", "downside", "breakdown", "failed"]
    
    bull_count = sum(1 for w in bull_words if w in text_l)
    bear_count = sum(1 for w in bear_words if w in text_l)
    
    if bull_count > bear_count + 1: return "bullish"
    if bear_count > bull_count + 1: return "bearish"
    return "neutral"


def reddit_sentiment_summary(posts: List[Dict]) -> Dict:
    """Aggregate Reddit sentiment + top posts by engagement."""
    if not posts:
        return {"total": 0, "bullish": 0, "bearish": 0, "neutral": 0,
                "bullish_pct": 0, "bearish_pct": 0, "sentiment_score": 0,
                "total_score": 0, "total_comments": 0,
                "top_bullish": [], "top_bearish": [], "subreddits": {}}
    
    bull = [p for p in posts if p.get("sentiment") == "bullish"]
    bear = [p for p in posts if p.get("sentiment") == "bearish"]
    neutral = [p for p in posts if p.get("sentiment") == "neutral"]
    total = len(posts)
    
    total_score = sum(p.get("score", 0) or 0 for p in posts)
    total_comments = sum(p.get("comments", 0) or 0 for p in posts)
    
    # By subreddit
    subreddit_counts = {}
    for p in posts:
        sub = p.get("subreddit", "unknown")
        subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
    
    def engagement(p):
        return (p.get("score", 0) or 0) + (p.get("comments", 0) or 0) * 2
    
    return {
        "total": total,
        "bullish": len(bull),
        "bearish": len(bear),
        "neutral": len(neutral),
        "bullish_pct": round(len(bull) / total * 100, 1) if total else 0,
        "bearish_pct": round(len(bear) / total * 100, 1) if total else 0,
        "sentiment_score": round((len(bull) - len(bear)) / total, 2) if total else 0,
        "total_score": total_score,
        "total_comments": total_comments,
        "top_bullish": sorted(bull, key=engagement, reverse=True)[:3],
        "top_bearish": sorted(bear, key=engagement, reverse=True)[:3],
        "subreddits": subreddit_counts,
    }


# ============================================================
# UNIFIED ORCHESTRATOR
# ============================================================

def fetch_social_all(ticker: str, company: str = "") -> Dict:
    """Fetch StockTwits + Reddit in parallel. 30min cache.
    
    Returns:
    {
        "stocktwits": {"messages": [...], "summary": {...}},
        "reddit": {"posts": [...], "summary": {...}},
        "combined_sentiment_score": -1 to +1,
        "total_mentions": int,
    }
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_st = ex.submit(fetch_stocktwits, ticker, 30)
        f_rd = ex.submit(fetch_reddit, ticker, company, 30, 30)
        try: st_msgs = f_st.result(timeout=12)
        except: st_msgs = []
        try: rd_posts = f_rd.result(timeout=25)
        except: rd_posts = []
    
    st_summary = stocktwits_sentiment_summary(st_msgs)
    rd_summary = reddit_sentiment_summary(rd_posts)
    
    # Combined sentiment: simple avg of the two, weighted by sample size
    st_w = min(st_summary["total"], 30)
    rd_w = min(rd_summary["total"], 30)
    total_w = st_w + rd_w
    if total_w > 0:
        combined = (st_summary["sentiment_score"] * st_w + rd_summary["sentiment_score"] * rd_w) / total_w
    else:
        combined = 0
    
    return {
        "stocktwits": {"messages": st_msgs, "summary": st_summary},
        "reddit": {"posts": rd_posts, "summary": rd_summary},
        "combined_sentiment_score": round(combined, 2),
        "total_mentions": st_summary["total"] + rd_summary["total"],
    }
