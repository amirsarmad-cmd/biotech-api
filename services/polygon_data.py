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

    Polygon's /v3/snapshot/options/{ticker} ignores as_of and returns the
    current chain. The correct historical flow uses two endpoints:

      1. /v3/reference/options/contracts?underlying_ticker=X&as_of=Y
         → list of contract tickers active on date Y (with strike, expiry,
           contract_type as metadata only; no prices)
      2. /v2/aggs/ticker/{contract_ticker}/range/1/day/{as_of}/{as_of}
         → historical OHLC bar for each contract on the requested date

    To keep the response tractable, we only fetch aggs for contracts that
    are within ±20% of the underlying_close on as_of_date (near-the-money
    only — that's all we need for ATM straddle computation).

    Returns:
      {
        "ticker": str,
        "as_of_date": str,
        "underlying_price": float,
        "contracts": [
          {"ticker", "type", "strike", "expiration",
           "open", "high", "low", "close", "volume"}, ...
        ],
        "n_contracts": int,
        "n_priced": int,
      }
    """
    cache_key = f"polygon:opt_chain:{ticker}:{as_of_date}:{expiration_after or ''}:{expiration_before or ''}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    # ─── Step 1: Find underlying price on as_of_date ─────────────────
    # /v2/aggs/ticker/{ticker}/range/1/day/{as_of}/{as_of}
    underlying_data = _http_get(
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{as_of_date}/{as_of_date}",
        params={"adjusted": "true"}, timeout=10,
    )
    underlying_price = None
    if underlying_data and underlying_data.get("results"):
        bar = underlying_data["results"][0]
        underlying_price = float(bar.get("c") or bar.get("close") or 0) or None

    # ─── Step 2: List contracts active on as_of_date ─────────────────
    params = {
        "underlying_ticker": ticker,
        "as_of": as_of_date,
        "expired": "true",  # include contracts that have since expired (we want historical)
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
    }
    if expiration_after:
        params["expiration_date.gte"] = expiration_after
    if expiration_before:
        params["expiration_date.lte"] = expiration_before

    contracts_data = _http_get(f"{POLYGON_BASE}/v3/reference/options/contracts", params=params)
    if not contracts_data:
        return None
    if contracts_data.get("_status") in (403, 404):
        return {"_status": "not_available", "_polygon_status": contracts_data.get("_status")}

    raw_contracts = contracts_data.get("results") or []
    if not raw_contracts:
        return {
            "ticker": ticker, "as_of_date": as_of_date,
            "underlying_price": underlying_price,
            "contracts": [], "n_contracts": 0, "n_priced": 0,
            "_source": "polygon_options",
            "_note": "no contracts found for this date/range",
        }

    # ─── Step 3: Filter to near-the-money contracts ──────────────────
    # If underlying_price unknown, take everything (rare but possible)
    near_money = []
    if underlying_price:
        # ±25% strikes (covers all reasonable ATM straddle candidates)
        lo = underlying_price * 0.75
        hi = underlying_price * 1.25
        for c in raw_contracts:
            strike = c.get("strike_price")
            if strike and lo <= float(strike) <= hi:
                near_money.append(c)
    else:
        near_money = raw_contracts[:60]  # fallback safety cap

    # ─── Step 4: Fetch OHLC for each near-the-money contract on as_of_date ─
    # This is the expensive step — sequential calls with 100ms gap to respect
    # any per-second cap. For 60 contracts at 100ms = 6s total.
    contracts = []
    n_priced = 0
    for c in near_money:
        contract_ticker = c.get("ticker")
        if not contract_ticker:
            continue
        bar_data = _http_get(
            f"{POLYGON_BASE}/v2/aggs/ticker/{contract_ticker}/range/1/day/{as_of_date}/{as_of_date}",
            params={"adjusted": "true"}, timeout=8,
        )
        bar = None
        if bar_data and bar_data.get("results"):
            bar = bar_data["results"][0]
        contract = {
            "ticker": contract_ticker,
            "type": (c.get("contract_type") or "").lower(),
            "strike": float(c.get("strike_price")) if c.get("strike_price") is not None else None,
            "expiration": c.get("expiration_date"),
            "open": float(bar["o"]) if bar and bar.get("o") is not None else None,
            "high": float(bar["h"]) if bar and bar.get("h") is not None else None,
            "low": float(bar["l"]) if bar and bar.get("l") is not None else None,
            "close": float(bar["c"]) if bar and bar.get("c") is not None else None,
            "vwap": float(bar["vw"]) if bar and bar.get("vw") is not None else None,
            "volume": int(bar["v"]) if bar and bar.get("v") is not None else None,
            "last_price": float(bar["c"]) if bar and bar.get("c") is not None else None,
            "mid": float(bar["c"]) if bar and bar.get("c") is not None else None,  # use close as mid
        }
        if contract["close"] is not None:
            n_priced += 1
        contracts.append(contract)

    out = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "underlying_price": underlying_price,
        "contracts": contracts,
        "n_contracts": len(contracts),
        "n_priced": n_priced,
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

    # In the new historical flow, each contract has 'close' from /v2/aggs.
    # Use close as the primary price source, fall back to mid if present
    # (mid is set to close in the new fetcher for backward compat).
    def _option_price(c):
        if c.get("close") is not None and float(c["close"]) > 0:
            return float(c["close"]), "historical_close"
        if c.get("vwap") is not None and float(c["vwap"]) > 0:
            return float(c["vwap"]), "historical_vwap"
        if c.get("mid") is not None and float(c["mid"]) > 0:
            return float(c["mid"]), "mid"
        if c.get("last_price") and float(c["last_price"]) > 0:
            return float(c["last_price"]), "last_price"
        return None, None

    call_price, call_method = _option_price(atm_call)
    put_price, put_method = _option_price(atm_put)

    if not call_price or not put_price:
        # Without historical OHLC for ATM contracts, no straddle pricing possible.
        # IV-based fallback isn't applicable here since the new fetcher doesn't
        # return IV (only OHLC bars).
        return None

    straddle = call_price + put_price
    # 1-stdev expected move ≈ 85% of straddle premium / underlying
    implied_move_pct = 0.85 * straddle / underlying * 100

    exp_dt = datetime.strptime(best_exp, "%Y-%m-%d").date()
    return {
        "implied_move_pct": round(implied_move_pct, 2),
        "expiration": best_exp,
        "dte": (exp_dt - as_of_dt).days,
        "atm_strike": atm_strike,
        "atm_call_mid": call_price,
        "atm_put_mid": put_price,
        "atm_call_method": call_method,
        "atm_put_method": put_method,
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
    # Also surface what's actually in os.environ — Railway might be dropping
    # variable names with whitespace because POSIX env names can't have spaces.
    env_polygon_keys = sorted([k for k in os.environ.keys() if "POLYGON" in k.upper()])
    if not key:
        return {
            "status": "no_api_key",
            "checked_env_vars": ["POLYGON_API_KEY", "POLYGON_API_KEY ", "POLYGON_KEY"],
            "actual_env_vars_with_polygon": env_polygon_keys,
            "_note": ("Polygon API key not visible to the process. "
                      "POSIX requires env var names be alphanumeric+underscore — "
                      "if the Railway variable name has whitespace, the OS strips it. "
                      "Rename to plain POLYGON_API_KEY (no trailing space) on Railway."),
        }
    # Tickers reference endpoint is on every plan
    out = {"status": "ok", "key_present": True, "key_prefix": key[:6] + "...",
           "actual_env_vars_with_polygon": env_polygon_keys, "checks": {}}
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
    # Check 4: Historical options reference (the right endpoint for as_of)
    one_mo_ago = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    opt_hist = _http_get(
        f"{POLYGON_BASE}/v3/reference/options/contracts",
        params={"underlying_ticker": "AAPL", "as_of": one_mo_ago,
                "expired": "true", "limit": 1},
    )
    out["checks"]["options_historical_reference"] = {
        "ok": bool(opt_hist and opt_hist.get("results")),
        "as_of": one_mo_ago,
        "n_results": len(opt_hist.get("results", [])) if opt_hist else 0,
        "status_code": opt_hist.get("_status") if opt_hist else None,
    }
    # Check 5: Historical options aggregates (per-contract OHLC)
    # Pick a contract from check 4 if available, else just probe AAPL stock aggs
    aggs = _http_get(
        f"{POLYGON_BASE}/v2/aggs/ticker/AAPL/range/1/day/{one_mo_ago}/{one_mo_ago}",
        params={"adjusted": "true"},
    )
    out["checks"]["historical_aggs"] = {
        "ok": bool(aggs and aggs.get("results")),
        "as_of": one_mo_ago,
        "status_code": aggs.get("_status") if aggs else None,
    }
    return out
