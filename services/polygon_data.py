"""polygon_data — Polygon.io integration for historical options chains and news.

Polygon Massive plan: $399/mo for unified bundle, OR $79+$79 for Stocks Advanced
+ Options Advanced separately. Either way we get:
  - 15-year historical options chains (end-of-day + minute bars)
  - Real-time + historical news with full article bodies
  - Tick-level trades
  - 5+ years of fundamentals
  - Unlimited API calls

This module solves two problems we have right now:

1. options_implied_move_pct backfill
   Current N=358 backtest captures options-implied move using TODAY's chain,
   not the chain at the actual catalyst date. Polygon historical chains
   give us the real implied vol at T-1 from the catalyst, so backtest accuracy
   becomes meaningful.

2. News ingestion (Layer 2 of source stack)
   Replaces brittle scraping. Polygon ingests full article bodies from
   Benzinga, Zacks, MarketWatch, MT Newswires, Reuters → indexed by ticker
   with sentiment tags. Cleaner than Finnhub for biotech-specific news.

Reads POLYGON_API_KEY env var (also tolerates trailing-space variant for
robustness against config typos). All calls cached in Redis 1h for live
data, 30d for historical.
"""
import os
import logging
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def _get_api_key() -> Optional[str]:
    """Tolerant key reader — handles trailing-space env var name."""
    for k in ("POLYGON_API_KEY", "POLYGON_API_KEY ", "POLYGON_KEY"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def is_configured() -> bool:
    return _get_api_key() is not None


# ────────────────────────────────────────────────────────────
# Cache helpers
# ────────────────────────────────────────────────────────────

def _redis_client():
    try:
        from services.cache import get_redis
        return get_redis()
    except Exception:
        return None


def _cached_get(cache_key: str):
    r = _redis_client()
    if r is None: return None
    try:
        raw = r.get(cache_key)
        if raw: return json.loads(raw)
    except Exception:
        pass
    return None


def _cached_set(cache_key: str, value, ttl_sec: int = 3600):
    r = _redis_client()
    if r is None: return
    try:
        r.setex(cache_key, ttl_sec, json.dumps(value, default=str))
    except Exception:
        pass


def _http_get(url: str, params: Dict = None, timeout: int = 15) -> Optional[Dict]:
    """Polygon GET with auth + retry on 429."""
    api_key = _get_api_key()
    if not api_key:
        logger.warning("Polygon: API key not configured")
        return None
    params = dict(params or {})
    params["apiKey"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": "biotech-screener/1.0"})
        if resp.status_code == 429:
            # Rate limited — back off briefly. Polygon Massive is unlimited
            # but per-second caps still apply.
            time.sleep(2)
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": "biotech-screener/1.0"})
        if resp.status_code == 401:
            logger.warning(f"Polygon: 401 unauthorized (check key)")
            return None
        if resp.status_code == 403:
            logger.warning(f"Polygon: 403 forbidden — endpoint not in plan tier")
            return {"_status": 403, "_message": "endpoint not available on current plan"}
        if resp.status_code == 404:
            return {"_status": 404}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.info(f"polygon http_get failed {url}: {e}")
        return None
    except ValueError as e:
        logger.info(f"polygon json parse failed: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Historical options chain — for options_implied_move backfill
# ────────────────────────────────────────────────────────────

def fetch_historical_options_chain(ticker: str, as_of_date: str,
                                    expiration_after: Optional[str] = None,
                                    expiration_before: Optional[str] = None) -> Optional[Dict]:
    """Fetch options chain that was active on `as_of_date` (YYYY-MM-DD).
    
    Strategy: query /v3/snapshot/options/{ticker} which returns the chain
    at the requested date. Filter to expirations near the catalyst window
    (typically ±30 days from event).
    
    Returns:
      {
        "ticker": str,
        "as_of_date": str,
        "underlying_price": float,
        "contracts": [
          {
            "ticker": str,         # OPRA symbol e.g. "O:NTLA241115C00045000"
            "type": "call"|"put",
            "strike": float,
            "expiration": str,     # YYYY-MM-DD
            "iv": float,           # implied volatility
            "delta": float,
            "open_interest": int,
            "last_price": float,
            "bid": float,
            "ask": float,
            "volume": int,
          }, ...
        ],
        "n_contracts": int,
      }
    Or {"_status": "not_available"} if the chain isn't available for that date.
    """
    cache_key = f"polygon:opt_chain:{ticker}:{as_of_date}:{expiration_after or ''}:{expiration_before or ''}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    # Polygon's snapshot endpoint gives us the full chain at the as_of_date
    # for historical: /v3/snapshot/options/{underlyingTicker}?as_of=YYYY-MM-DD
    params = {
        "as_of": as_of_date,
        "limit": 250,  # max per page; biotech options chains usually <100 contracts
    }
    if expiration_after:
        params["expiration_date.gte"] = expiration_after
    if expiration_before:
        params["expiration_date.lte"] = expiration_before

    data = _http_get(f"{POLYGON_BASE}/v3/snapshot/options/{ticker}", params=params)
    if not data:
        return None
    if data.get("_status") in (403, 404):
        return {"_status": "not_available", "_polygon_status": data.get("_status"),
                "_message": data.get("_message")}

    contracts = []
    underlying_price = None
    for r in (data.get("results") or []):
        details = r.get("details") or {}
        day = r.get("day") or {}
        greeks = r.get("greeks") or {}
        underlying = r.get("underlying_asset") or {}
        if underlying_price is None and underlying.get("price"):
            underlying_price = float(underlying["price"])
        contracts.append({
            "ticker": details.get("ticker"),
            "type": (details.get("contract_type") or "").lower(),
            "strike": float(details.get("strike_price")) if details.get("strike_price") is not None else None,
            "expiration": details.get("expiration_date"),
            "iv": r.get("implied_volatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "open_interest": r.get("open_interest"),
            "last_price": day.get("close") if day.get("close") is not None else day.get("last_quote_price"),
            "volume": day.get("volume"),
        })

    out = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "underlying_price": underlying_price,
        "contracts": contracts,
        "n_contracts": len(contracts),
        "_source": "polygon_options",
        "_from_cache": False,
    }
    # Historical data — cache 30 days (it doesn't change)
    _cached_set(cache_key, out, ttl_sec=30 * 86400)
    return out


def compute_implied_move_from_chain(chain: Dict, target_dte_min: int = 7,
                                     target_dte_max: int = 60) -> Optional[Dict]:
    """Given a Polygon options chain, compute the ATM straddle-implied move %.
    
    Methodology (industry standard):
      - Pick the expiration closest to (target_dte_min, target_dte_max) range
      - Find ATM call + put (closest strike to underlying_price)
      - Implied move % = (call_mid + put_mid) / underlying_price * 100
      - Equivalent: 0.85 * straddle / underlying ≈ 1-stdev implied move
    
    Returns:
      {
        "implied_move_pct": float,     # absolute percentage
        "expiration": str,
        "dte": int,
        "atm_strike": float,
        "atm_call_mid": float,
        "atm_put_mid": float,
        "underlying_price": float,
      }
    Or None if no suitable straddle could be priced.
    """
    if not chain or not chain.get("contracts") or not chain.get("underlying_price"):
        return None

    underlying = float(chain["underlying_price"])
    as_of = chain.get("as_of_date")
    if not as_of:
        return None
    try:
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d").date()
    except ValueError:
        return None

    # Group contracts by expiration → find best expiration in DTE range
    by_exp: Dict[str, Dict[str, list]] = {}
    for c in chain["contracts"]:
        exp_str = c.get("expiration")
        if not exp_str: continue
        try:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_dt - as_of_dt).days
        if dte < target_dte_min or dte > target_dte_max:
            continue
        by_exp.setdefault(exp_str, {"calls": [], "puts": []})
        if c.get("type") == "call":
            by_exp[exp_str]["calls"].append(c)
        elif c.get("type") == "put":
            by_exp[exp_str]["puts"].append(c)

    if not by_exp:
        return None

    # Pick the expiration with the most contracts (deepest market = better pricing)
    best_exp = max(by_exp.keys(), key=lambda e: len(by_exp[e]["calls"]) + len(by_exp[e]["puts"]))
    exp_data = by_exp[best_exp]
    if not exp_data["calls"] or not exp_data["puts"]:
        return None

    # Find ATM strike — closest to underlying price
    all_strikes = set(c["strike"] for c in exp_data["calls"] + exp_data["puts"]
                      if c.get("strike") is not None)
    if not all_strikes:
        return None
    atm_strike = min(all_strikes, key=lambda s: abs(s - underlying))

    atm_call = next((c for c in exp_data["calls"] if c.get("strike") == atm_strike), None)
    atm_put = next((c for c in exp_data["puts"] if c.get("strike") == atm_strike), None)
    if not atm_call or not atm_put:
        return None

    # Use last_price; could fall back to (bid+ask)/2 if available
    call_mid = atm_call.get("last_price") or 0
    put_mid = atm_put.get("last_price") or 0
    if not call_mid or not put_mid:
        return None

    straddle = float(call_mid) + float(put_mid)
    # 1-stdev expected move ≈ 85% of straddle premium / underlying
    # (Empirical adjustment for the fact that ATM straddle slightly overstates 1-stdev)
    implied_move_pct = 0.85 * straddle / underlying * 100

    exp_dt = datetime.strptime(best_exp, "%Y-%m-%d").date()
    return {
        "implied_move_pct": round(implied_move_pct, 2),
        "expiration": best_exp,
        "dte": (exp_dt - as_of_dt).days,
        "atm_strike": atm_strike,
        "atm_call_mid": call_mid,
        "atm_put_mid": put_mid,
        "underlying_price": underlying,
        "method": "atm_straddle_85pct",
    }


def backfill_historical_implied_move(ticker: str, catalyst_date: str,
                                       target_dte_min: int = 7,
                                       target_dte_max: int = 60) -> Optional[Dict]:
    """End-to-end: fetch the chain at T-1 from catalyst_date, compute implied move.
    
    This is the function that backfills `options_implied_move_pct` on
    historical post_catalyst_outcomes. T-1 (one trading day before) avoids
    the noise of catalyst-day vol crush.
    """
    try:
        cat_dt = datetime.strptime(catalyst_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    # Skip back one business day — handle weekends approximately
    as_of = cat_dt - timedelta(days=1)
    if as_of.weekday() == 6: as_of -= timedelta(days=2)  # Sunday → Friday
    if as_of.weekday() == 5: as_of -= timedelta(days=1)  # Saturday → Friday

    # Look for expirations 7-60d after catalyst date — that's the standard
    # event-window straddle
    exp_after = (cat_dt + timedelta(days=target_dte_min)).strftime("%Y-%m-%d")
    exp_before = (cat_dt + timedelta(days=target_dte_max + 30)).strftime("%Y-%m-%d")

    chain = fetch_historical_options_chain(
        ticker=ticker,
        as_of_date=as_of.strftime("%Y-%m-%d"),
        expiration_after=exp_after,
        expiration_before=exp_before,
    )
    if not chain or chain.get("_status") == "not_available":
        return None

    move = compute_implied_move_from_chain(
        chain,
        target_dte_min=(cat_dt - as_of).days + target_dte_min,
        target_dte_max=(cat_dt - as_of).days + target_dte_max,
    )
    if not move:
        return None

    return {
        "ticker": ticker,
        "catalyst_date": catalyst_date,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        **move,
    }


# ────────────────────────────────────────────────────────────
# News — Layer 2 source for research_corpus
# ────────────────────────────────────────────────────────────

def fetch_news(ticker: str, published_utc_gte: Optional[str] = None,
                published_utc_lte: Optional[str] = None,
                limit: int = 50) -> Optional[Dict]:
    """Fetch Polygon news for a ticker, optionally bounded by date.
    
    Returns:
      {
        "ticker": str,
        "articles": [
          {
            "id": str,
            "publisher": str,
            "title": str,
            "author": str,
            "published_utc": str,
            "article_url": str,
            "tickers": [str],
            "amp_url": str,
            "image_url": str,
            "description": str,
            "keywords": [str],
            "insights": [{"ticker": str, "sentiment": str, "sentiment_reasoning": str}],
          }, ...
        ],
        "count": int,
      }
    """
    cache_key = f"polygon:news:{ticker}:{published_utc_gte or ''}:{published_utc_lte or ''}:{limit}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    params = {
        "ticker": ticker,
        "order": "desc",
        "limit": min(limit, 1000),
        "sort": "published_utc",
    }
    if published_utc_gte:
        params["published_utc.gte"] = published_utc_gte
    if published_utc_lte:
        params["published_utc.lte"] = published_utc_lte

    data = _http_get(f"{POLYGON_BASE}/v2/reference/news", params=params, timeout=20)
    if not data or data.get("_status") in (403, 404):
        return None

    articles = data.get("results") or []
    out = {
        "ticker": ticker,
        "articles": articles,
        "count": len(articles),
        "_source": "polygon_news",
        "_from_cache": False,
    }
    # Live news cached 1h; historical (gte set) cached 24h
    ttl = 3600 if not published_utc_gte else 86400
    _cached_set(cache_key, out, ttl_sec=ttl)
    return out


# ────────────────────────────────────────────────────────────
# Diagnostic helper
# ────────────────────────────────────────────────────────────

def diagnostic() -> Dict:
    """Probe Polygon connectivity + plan tier. Used by /admin/polygon/status."""
    key = _get_api_key()
    if not key:
        return {
            "status": "no_api_key",
            "checked_env_vars": ["POLYGON_API_KEY", "POLYGON_API_KEY ", "POLYGON_KEY"],
            "_note": "Polygon API key not configured. Set POLYGON_API_KEY (no trailing space) on Railway.",
        }
    # Tickers reference endpoint is on every plan
    out = {"status": "ok", "key_present": True, "key_prefix": key[:6] + "...", "checks": {}}
    # Check 1: Reference data
    ref = _http_get(f"{POLYGON_BASE}/v3/reference/tickers", params={"ticker": "AAPL", "limit": 1})
    out["checks"]["reference_tickers"] = {
        "ok": bool(ref and ref.get("results")),
        "count": len(ref.get("results", [])) if ref else 0,
    }
    # Check 2: News
    news = _http_get(f"{POLYGON_BASE}/v2/reference/news", params={"ticker": "AAPL", "limit": 1})
    out["checks"]["news"] = {
        "ok": bool(news and news.get("results")),
        "count": len(news.get("results", [])) if news else 0,
    }
    # Check 3: Options snapshot (current)
    opt = _http_get(f"{POLYGON_BASE}/v3/snapshot/options/AAPL", params={"limit": 1})
    out["checks"]["options_snapshot"] = {
        "ok": bool(opt and not opt.get("_status")),
        "status_code": opt.get("_status") if opt else None,
    }
    # Check 4: Historical options snapshot (1 month ago)
    one_mo_ago = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    opt_hist = _http_get(
        f"{POLYGON_BASE}/v3/snapshot/options/AAPL",
        params={"as_of": one_mo_ago, "limit": 1},
    )
    out["checks"]["options_historical"] = {
        "ok": bool(opt_hist and not opt_hist.get("_status")),
        "as_of": one_mo_ago,
        "status_code": opt_hist.get("_status") if opt_hist else None,
    }
    return out
