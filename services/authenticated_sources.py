"""
Analyst data sources: yfinance (primary) + optional authenticated scrapers.

yfinance gives us:
- Mean/high/low price targets
- Number of analysts covering
- Recommendation mean (1=Strong Buy, 5=Strong Sell)
- Recent upgrades/downgrades
- Recommendation breakdown

No Playwright required by default. If TIPRANKS_USER/PASS are set AND playwright
browser is available, attempt scraping for premium TipRanks data.

SA_USER/PASS similarly optional — off by default since Playwright on Railway
has been unreliable.
"""
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_cache: Dict = {}
_CACHE_TTL = 12 * 3600  # 12h


def _cached(key: str, fn):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _CACHE_TTL: return val
    val = fn()
    _cache[key] = (now, val)
    return val


# ============================================================
# Analyst data from yfinance (primary — always works)
# ============================================================

def fetch_analyst_yfinance(ticker: str) -> Dict:
    """Analyst consensus + targets from yfinance. No auth needed."""
    def _do():
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
            
            target_mean = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low = info.get("targetLowPrice")
            n_analysts = info.get("numberOfAnalystOpinions")
            rec_key = info.get("recommendationKey", "")
            rec_mean = info.get("recommendationMean")
            analyst_rating = info.get("averageAnalystRating", "")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            # Parse "3.0 - Hold" style string
            consensus = None
            if analyst_rating:
                parts = analyst_rating.split(" - ")
                if len(parts) == 2:
                    consensus = parts[1].strip()
            if not consensus and rec_key:
                consensus = rec_key.replace("_"," ").title()
            
            # Upside calc
            upside_pct = None
            if target_mean and current_price and current_price > 0:
                upside_pct = (target_mean - current_price) / current_price * 100
            
            # Recommendation breakdown
            rec_breakdown = {}
            try:
                recs = t.recommendations_summary
                if recs is not None and not recs.empty:
                    # Most recent row
                    latest = recs.iloc[0] if len(recs) > 0 else None
                    if latest is not None:
                        for col in ["strongBuy","buy","hold","sell","strongSell"]:
                            if col in latest:
                                rec_breakdown[col] = int(latest.get(col, 0) or 0)
            except Exception:
                pass
            
            # Recent upgrades/downgrades
            recent_changes = []
            try:
                ugd = t.upgrades_downgrades
                if ugd is not None and not ugd.empty:
                    for idx, row in ugd.head(5).iterrows():
                        recent_changes.append({
                            "date": str(idx)[:10] if idx else "",
                            "firm": str(row.get("Firm","")),
                            "to_grade": str(row.get("ToGrade","")),
                            "from_grade": str(row.get("FromGrade","")),
                            "action": str(row.get("Action","")),
                        })
            except Exception:
                pass
            
            return {
                "target_mean": target_mean,
                "target_high": target_high,
                "target_low": target_low,
                "current_price": current_price,
                "upside_pct": upside_pct,
                "analyst_count": n_analysts,
                "consensus": consensus or "Unknown",
                "recommendation_mean": rec_mean,  # 1-5 scale
                "buy": rec_breakdown.get("strongBuy", 0) + rec_breakdown.get("buy", 0),
                "strong_buy": rec_breakdown.get("strongBuy", 0),
                "hold": rec_breakdown.get("hold", 0),
                "sell": rec_breakdown.get("sell", 0) + rec_breakdown.get("strongSell", 0),
                "strong_sell": rec_breakdown.get("strongSell", 0),
                "recent_changes": recent_changes,
                "error": None,
                "source": "yfinance",
            }
        except Exception as e:
            logger.warning(f"yfinance analyst {ticker}: {e}")
            return {"error": str(e)[:200], "analyst_count": 0,
                    "consensus": "Unknown", "source": "yfinance"}
    
    return _cached(f"ya:{ticker}", _do)


# ============================================================
# TipRanks (public, no auth needed) — supplements yfinance with analyst name + firm
# ============================================================

def fetch_tipranks_public(ticker: str) -> Dict:
    """Public TipRanks data via their JSON endpoints. No Playwright needed."""
    def _do():
        import requests
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json, text/plain, */*",
            }
            # TipRanks has a public JSON endpoint for stock summaries
            url = f"https://www.tipranks.com/api/stocks/getData/?name={ticker.upper()}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "consensus": "N/A"}
            data = r.json()
            
            result = {
                "consensus": data.get("consensus", "N/A"),
                "analyst_count": data.get("numOfAnalysts", 0),
                "target_mean": data.get("priceTarget"),
                "upside_pct": None,
                "buy": 0, "hold": 0, "sell": 0,
                "source_url": f"https://www.tipranks.com/stocks/{ticker.lower()}/forecast",
                "error": None,
            }
            
            # Rating breakdown if present
            if "consensusRatings" in data:
                cr = data["consensusRatings"] or {}
                result["buy"] = cr.get("buy", 0) or 0
                result["hold"] = cr.get("hold", 0) or 0
                result["sell"] = cr.get("sell", 0) or 0
            
            return result
        except Exception as e:
            return {"error": str(e)[:200], "consensus": "N/A"}
    
    return _cached(f"tr:{ticker}", _do)


# ============================================================
# Seeking Alpha — skip scraping (unreliable without auth + Playwright)
# Return stub unless SA_USER/SA_PASS are set (future work)
# ============================================================

def fetch_sa_public(ticker: str) -> Dict:
    """Stub — SA requires auth for most data. Returns empty unless creds are set."""
    if not os.getenv("SA_USER") or not os.getenv("SA_PASS"):
        return {
            "quant_rating": None,
            "quant_label": "Not configured",
            "valuation_grade": None,
            "profitability_grade": None,
            "recent_articles": [],
            "error": "SA credentials not set (SA_USER/SA_PASS)",
            "source_url": f"https://seekingalpha.com/symbol/{ticker.upper()}",
        }
    # With credentials set, would attempt Playwright login
    return {
        "quant_rating": None,
        "quant_label": "SA scraping disabled — set up Playwright-free API integration",
        "valuation_grade": None,
        "profitability_grade": None,
        "recent_articles": [],
        "error": None,
        "source_url": f"https://seekingalpha.com/symbol/{ticker.upper()}",
    }


# ============================================================
# Unified orchestrator
# ============================================================

def fetch_analyst_data_all(ticker: str) -> Dict:
    """Parallel fetch of yfinance + TipRanks + SA."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_yf = ex.submit(fetch_analyst_yfinance, ticker)
        f_tr = ex.submit(fetch_tipranks_public, ticker)
        f_sa = ex.submit(fetch_sa_public, ticker)
        try: yf_data = f_yf.result(timeout=20)
        except Exception as e: yf_data = {"error": str(e)[:200], "source": "yfinance"}
        try: tr_data = f_tr.result(timeout=15)
        except Exception as e: tr_data = {"error": str(e)[:200]}
        try: sa_data = f_sa.result(timeout=15)
        except Exception as e: sa_data = {"error": str(e)[:200]}
    
    return {
        "yfinance": yf_data,
        "tipranks": tr_data,
        "seeking_alpha": sa_data,
    }


# Back-compat for the old UI (which expects `tipranks` as primary)
# Merge yfinance into tipranks shape so the UI can use either
def fetch_analyst_data_all_merged(ticker: str) -> Dict:
    """Returns a merged shape where `tipranks` includes yfinance-derived data
    if TipRanks itself failed. Useful for single-source UI."""
    data = fetch_analyst_data_all(ticker)
    tr = data.get("tipranks", {})
    yf_d = data.get("yfinance", {})
    
    # If TipRanks returned error or no data, fill tr with yf data
    if tr.get("error") or not tr.get("consensus") or tr.get("consensus") == "N/A":
        if not yf_d.get("error"):
            tr["consensus"] = yf_d.get("consensus") or "N/A"
            tr["analyst_count"] = yf_d.get("analyst_count") or 0
            tr["target_mean"] = yf_d.get("target_mean")
            tr["target_high"] = yf_d.get("target_high")
            tr["target_low"] = yf_d.get("target_low")
            tr["upside_pct"] = yf_d.get("upside_pct")
            tr["buy"] = yf_d.get("buy", 0)
            tr["hold"] = yf_d.get("hold", 0)
            tr["sell"] = yf_d.get("sell", 0)
            tr["current_price"] = yf_d.get("current_price")
            tr["source_url"] = tr.get("source_url") or f"https://finance.yahoo.com/quote/{ticker}"
            tr["error"] = None  # clear error since we have data now
            tr["_source_fallback"] = "yfinance"
        data["tipranks"] = tr
    
    return data
