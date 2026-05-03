"""feature_store — snapshot every feature any future algo could want at the
catalyst-event grain, ONCE, so algo iterations don't re-fetch data each time.

Design (from spec-04 + user 2026-05-03 feedback):
  - One row per (catalyst_id, ticker, catalyst_date) in catalyst_event_features.
  - Backfill is incremental + idempotent — call repeatedly safely.
  - Per-source modules: each fills a slice of columns. Failures are logged
    in `backfill_source_status` JSON; the row is still written with NULLs.
  - Two phases:
      Phase A (this commit): non-LLM features that don't need Gemini.
      Phase B (when Gemini cap returns): LLM-enrichment columns. Picked up
        by a separate `enrich_with_llm()` pass.

Public API:
  - compute_event_features(catalyst_id, refresh=False) -> dict
       Builds the row for one event. Idempotent — skip if row exists and
       refresh=False.
  - backfill_features_batch(limit=N, only_labeled=True) -> dict
       Find catalyst_ids without a feature row (or stale) and fill them.

Per-source backfillers are small focused functions named `_fill_*` so a
future infra chat can swap implementations without touching the orchestrator.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Cap-bucket helper (kept consistent with services/scenario_algo.py)
# ────────────────────────────────────────────────────────────
def _cap_bucket(market_cap_m: Optional[float]) -> str:
    if not market_cap_m:
        return "unknown"
    if market_cap_m < 500:
        return "micro_lt500m"
    if market_cap_m < 2000:
        return "small_500m_2b"
    return "mid_or_above"


# ────────────────────────────────────────────────────────────
# Outcome label price-proxy (no LLM required — works on 100% of events)
# ────────────────────────────────────────────────────────────
def _outcome_label_price_proxy(actual_move_pct_7d: Optional[float]) -> Optional[str]:
    """Cheap 7d-move-based label for events the LLM hasn't classified yet.
    Lower-confidence than the Gemini label but always available, so backtest
    code can use the price-proxy when the LLM label is missing.
    """
    if actual_move_pct_7d is None:
        return None
    if actual_move_pct_7d > 15:
        return "POSITIVE"
    if actual_move_pct_7d < -15:
        return "NEGATIVE"
    return "MIXED"


# ────────────────────────────────────────────────────────────
# Per-source fillers
# ────────────────────────────────────────────────────────────
def _fill_price_action(
    *, ticker: str, catalyst_date: str, db: BiotechDatabase,
) -> Dict[str, Any]:
    """Runup, realized vol, max drawdown — fetched from yfinance (the
    `price_history_daily` cache table referenced earlier doesn't exist
    in this DB, so we go straight to yfinance live).
    """
    out: Dict[str, Any] = {}
    status = "ok"
    try:
        try:
            import yfinance as yf
        except ImportError:
            return {"_status": "yfinance_unavailable"}
        cd = datetime.fromisoformat(catalyst_date[:10])
        start = cd - timedelta(days=210)
        try:
            hist = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                              end=cd.strftime("%Y-%m-%d"))
            rows = [(idx.date(), float(close))
                    for idx, close in zip(hist.index, hist["Close"])
                    if not math.isnan(close)]
        except Exception as e:
            return {"_status": f"yfinance_fetch_failed:{type(e).__name__}"}
        if not rows or len(rows) < 30:
            return {"_status": f"insufficient_history:n={len(rows)}"}

        closes = [float(r[1]) for r in rows]
        last = closes[-1]

        def _runup(window_days: int) -> Optional[float]:
            if len(closes) <= window_days:
                return None
            past = closes[-(window_days + 1)]
            if past <= 0:
                return None
            return (last - past) / past * 100.0

        out["runup_pct_30d"] = _runup(30)
        out["runup_pct_90d"] = _runup(90)
        out["runup_pct_180d"] = _runup(180)

        # Realized vol = stdev of daily log returns, annualized
        def _vol(window_days: int) -> Optional[float]:
            if len(closes) <= window_days:
                return None
            window = closes[-window_days:]
            log_returns = []
            for i in range(1, len(window)):
                if window[i - 1] > 0 and window[i] > 0:
                    log_returns.append(math.log(window[i] / window[i - 1]))
            if len(log_returns) < 5:
                return None
            mean = sum(log_returns) / len(log_returns)
            variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
            return math.sqrt(variance) * math.sqrt(252) * 100.0

        out["realized_vol_30d"] = _vol(30)
        out["realized_vol_90d"] = _vol(90)

        # Max drawdown over the last 30 days
        if len(closes) >= 30:
            window30 = closes[-30:]
            peak = window30[0]
            max_dd = 0.0
            for px in window30:
                if px > peak:
                    peak = px
                dd = (px - peak) / peak * 100.0
                if dd < max_dd:
                    max_dd = dd
            out["max_drawdown_30d"] = max_dd
    except Exception as e:
        status = f"error:{type(e).__name__}:{str(e)[:80]}"
    out["_status"] = status
    return out


def _fill_peer_relative(
    *, ticker: str, catalyst_date: str, db: BiotechDatabase,
    ticker_runup_30d: Optional[float], ticker_runup_90d: Optional[float],
) -> Dict[str, Any]:
    """XBI / IBB peer-index runups + relative strength + beta-to-XBI."""
    out: Dict[str, Any] = {}
    status = "ok"
    try:
        # Try yfinance for both peer indices (cheap, cached by yfinance)
        try:
            import yfinance as yf
        except ImportError:
            return {"_status": "yfinance_unavailable"}
        cd = datetime.fromisoformat(catalyst_date[:10])
        start = cd - timedelta(days=210)

        def _index_runup(symbol: str, window_days: int) -> Optional[float]:
            try:
                hist = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"),
                                                  end=cd.strftime("%Y-%m-%d"))
                closes = [float(c) for c in hist["Close"] if not math.isnan(c)]
                if len(closes) <= window_days:
                    return None
                past = closes[-(window_days + 1)]
                last = closes[-1]
                if past <= 0:
                    return None
                return (last - past) / past * 100.0
            except Exception:
                return None

        out["xbi_runup_30d"] = _index_runup("XBI", 30)
        out["xbi_runup_90d"] = _index_runup("XBI", 90)
        out["ibb_runup_30d"] = _index_runup("IBB", 30)
        if ticker_runup_30d is not None and out["xbi_runup_30d"] is not None:
            out["relative_strength_xbi_30d"] = ticker_runup_30d - out["xbi_runup_30d"]
        if ticker_runup_90d is not None and out["xbi_runup_90d"] is not None:
            out["relative_strength_xbi_90d"] = ticker_runup_90d - out["xbi_runup_90d"]
        # beta_to_xbi_180d deferred — needs paired daily returns regression
    except Exception as e:
        status = f"error:{type(e).__name__}"
    out["_status"] = status
    return out


def _fill_massive_options(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """massive.com options chain — current snapshot for upcoming events;
    two-step historical (contracts list + per-contract snapshot) for past
    events. Vendor is massive.com (Polygon-compatible REST endpoints; both
    api.massive.com and api.polygon.io accept the key, we use api.massive.com
    as canonical to avoid the polygon/massive naming confusion).

    Endpoints used:
      - /v3/snapshot/options/{ticker}                          — current chain
      - /v3/reference/options/contracts?as_of=                 — historical chain inventory
      - /v3/snapshot/options/{ticker}/{contract}?as_of=        — historical per-contract IV
      - /v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}     — underlying close
    """
    # Strip whitespace defensively — Railway's variableUpsert sometimes
    # stores values with trailing newlines (verified 2026-05-03 via diag).
    api_key = (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_massive_api_key"}

    try:
        import requests
        cd = datetime.fromisoformat(catalyst_date[:10])
        days_from_today = abs((cd - datetime.now()).days)
        if days_from_today <= 7:
            return _massive_current_snapshot(ticker, api_key, requests)
        return _massive_historical_snapshot(ticker, cd, api_key, requests)
    except Exception as e:
        return {"_status": f"massive_error:{type(e).__name__}:{str(e)[:80]}"}


def _massive_aggregate_chain(results: list, underlying_price: Optional[float]) -> Dict[str, Any]:
    """Common aggregation logic for both current + historical chain results."""
    out: Dict[str, Any] = {"options_source": "massive"}
    call_oi = put_oi = call_volume = put_volume = 0
    atm_calls: List[Tuple[float, float]] = []
    atm_puts: List[Tuple[float, float]] = []

    for opt in results:
        details = opt.get("details") or {}
        ctype = (details.get("contract_type") or "").lower()
        strike = details.get("strike_price")
        if underlying_price is None:
            ua = opt.get("underlying_asset") or {}
            underlying_price = ua.get("price")
        day = opt.get("day") or {}
        oi = day.get("open_interest") or 0
        vol = day.get("volume") or 0
        iv = (opt.get("implied_volatility") or 0) * 100
        if ctype == "call":
            call_oi += oi
            call_volume += vol
            if strike and iv > 0:
                atm_calls.append((strike, iv))
        elif ctype == "put":
            put_oi += oi
            put_volume += vol
            if strike and iv > 0:
                atm_puts.append((strike, iv))

    out["put_call_oi_ratio"] = (put_oi / call_oi) if call_oi else None
    out["put_call_volume_ratio"] = (put_volume / call_volume) if call_volume else None

    if underlying_price and atm_calls and atm_puts:
        atm_calls.sort(key=lambda x: abs(x[0] - underlying_price))
        atm_puts.sort(key=lambda x: abs(x[0] - underlying_price))
        atm_call_iv = atm_calls[0][1]
        atm_put_iv = atm_puts[0][1]
        out["atm_iv_at_date"] = (atm_call_iv + atm_put_iv) / 2
        # 25d skew approx — strikes ±10% from spot
        target_low = underlying_price * 0.90
        target_high = underlying_price * 1.10
        puts_25d = [iv for s, iv in atm_puts if s <= target_low]
        calls_25d = [iv for s, iv in atm_calls if s >= target_high]
        if puts_25d and calls_25d:
            out["iv_skew_25d"] = sum(calls_25d) / len(calls_25d) - sum(puts_25d) / len(puts_25d)
        # Implied move = ATM IV × sqrt(days_to_expiry / 252); use 7 as the calibration window
        out["options_implied_move_pct"] = (atm_call_iv + atm_put_iv) / 2 * math.sqrt(7 / 252)
    return out


def _massive_current_snapshot(ticker: str, api_key: str, requests) -> Dict[str, Any]:
    """Current chain — works on Options Starter+ plans."""
    url = f"https://api.massive.com/v3/snapshot/options/{ticker}"
    r = requests.get(url, params={"limit": 250, "apiKey": api_key}, timeout=20)
    if r.status_code != 200:
        return {"_status": f"massive_current_status_{r.status_code}"}
    results = (r.json() or {}).get("results", []) or []
    if not results:
        return {"_status": "massive_empty_chain_current"}
    out = _massive_aggregate_chain(results, underlying_price=None)
    out["_status"] = "ok_current"
    return out


def _massive_historical_snapshot(
    ticker: str, target_date: datetime, api_key: str, requests,
    max_atm_contracts: int = 8,
) -> Dict[str, Any]:
    """Historical chain via two-step: contracts list as_of, then per-contract
    snapshot for the ATM-band contracts (limit to ~8 to control API spend).
    """
    target_str = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")

    # Step 1: front-month contracts as_of target
    r = requests.get(
        "https://api.massive.com/v3/reference/options/contracts",
        params={"underlying_ticker": ticker, "as_of": target_str, "limit": 250, "apiKey": api_key},
        timeout=20,
    )
    if r.status_code != 200:
        return {"_status": f"massive_contracts_status_{r.status_code}"}
    contracts = (r.json() or {}).get("results", []) or []
    if not contracts:
        return {"_status": "massive_no_contracts_at_date"}

    # Step 2: underlying close on target_str.
    # Try massive.com /v2/aggs first; fall back to yfinance because
    # Stocks Basic plan returns 403 for dates >2y old. yfinance has
    # unlimited history and we already use it for price_action.
    underlying_price = None
    r = requests.get(
        f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{target_str}/{target_str}",
        params={"apiKey": api_key},
        timeout=15,
    )
    if r.status_code == 200:
        px_data = (r.json() or {}).get("results") or []
        if px_data:
            underlying_price = px_data[0].get("c")
    if underlying_price is None:
        try:
            import yfinance as yf
            cd = datetime.fromisoformat(target_str)
            hist = yf.Ticker(ticker).history(
                start=cd.strftime("%Y-%m-%d"),
                end=(cd + timedelta(days=3)).strftime("%Y-%m-%d"),
            )
            for c in hist["Close"]:
                if not math.isnan(c):
                    underlying_price = float(c)
                    break
        except Exception:
            pass
    if underlying_price is None:
        return {"_status": f"underlying_unavailable_for_{target_str}"}

    # Step 3: filter to front-month (20-60 days to expiry from target)
    target_min_exp = (target_date + timedelta(days=20)).date()
    target_max_exp = (target_date + timedelta(days=60)).date()
    front: List[Dict[str, Any]] = []
    for c in contracts:
        exp = c.get("expiration_date")
        if not exp:
            continue
        try:
            exp_d = datetime.fromisoformat(exp).date()
        except Exception:
            continue
        if target_min_exp <= exp_d <= target_max_exp and c.get("strike_price"):
            front.append(c)
    if not front:
        return {"_status": "massive_no_front_month_contracts"}

    # Step 4: pick the ATM band (closest strikes to underlying), fetch per-contract IV
    front.sort(key=lambda c: abs((c.get("strike_price") or 0) - (underlying_price or 0)))
    sample = front[:max_atm_contracts]

    chain_results: List[Dict[str, Any]] = []
    for c in sample:
        ctk = c.get("ticker")
        if not ctk:
            continue
        rr = requests.get(
            f"https://api.massive.com/v3/snapshot/options/{ticker}/{ctk}",
            params={"as_of": target_str, "apiKey": api_key},
            timeout=12,
        )
        if rr.status_code != 200:
            continue
        snap = (rr.json() or {}).get("results") or {}
        # Inject the contract details in the shape _massive_aggregate_chain expects
        snap["details"] = {
            "contract_type": c.get("contract_type"),
            "strike_price": c.get("strike_price"),
        }
        chain_results.append(snap)

    if not chain_results:
        return {"_status": "massive_no_per_contract_iv"}

    out = _massive_aggregate_chain(chain_results, underlying_price=underlying_price)
    out["_status"] = f"ok_historical (n_contracts_sampled={len(chain_results)})"
    return out


def _fill_capital_structure(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Cash, debt, runway from SEC EDGAR via existing services/sec_financials.
    The existing function `fetch_capital_structure(ticker)` returns the
    LATEST filing — accepts the lookback bias for now (point-in-time
    filtering deferred to a later iteration that traverses _filings).
    """
    try:
        from services import sec_financials  # type: ignore[attr-defined]
    except ImportError:
        return {"_status": "sec_financials_module_unavailable"}
    try:
        if not hasattr(sec_financials, "fetch_capital_structure"):
            return {"_status": "no_fetch_capital_structure_function"}
        cs = sec_financials.fetch_capital_structure(ticker)
        if not cs:
            return {"_status": "no_filing_found"}
        out = {
            "cash_at_date_m": (cs.get("total_cash") or 0) / 1e6 if cs.get("total_cash") else None,
            "debt_at_date_m": (cs.get("total_debt") or 0) / 1e6 if cs.get("total_debt") else None,
            "runway_months_at_date": cs.get("cash_runway_months"),
            "sec_filing_date_used": cs.get("as_of_filing"),
            "_status": "ok_latest_filing_lookback_bias",
        }
        if out["cash_at_date_m"] is not None and out["debt_at_date_m"] is not None:
            out["net_cash_at_date_m"] = out["cash_at_date_m"] - out["debt_at_date_m"]
        return out
    except Exception as e:
        return {"_status": f"sec_error:{type(e).__name__}:{str(e)[:60]}"}


def _fill_microstructure(
    *, ticker: str, db: BiotechDatabase,
) -> Dict[str, Any]:
    """Market cap from screener_stocks (only column it actually has);
    short interest + avg volume + shares outstanding from yfinance live
    (current snapshot — lookback bias accepted for v1; FINRA historical
    short interest backfill deferred to a later iteration).
    """
    out: Dict[str, Any] = {}
    # 1. Market cap from screener_stocks (only column it actually stores)
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT market_cap FROM screener_stocks WHERE ticker = %s LIMIT 1",
                (ticker,),
            )
            row = cur.fetchone()
        if row and row[0] is not None:
            # screener_stocks.market_cap is in MILLIONS USD (per memory note)
            out["market_cap_at_date_m"] = float(row[0])
    except Exception as e:
        out["_screener_err"] = f"{type(e).__name__}"

    # 2. yfinance for short interest + shares outstanding + avg volume
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        short_pct = info.get("shortPercentOfFloat")  # already decimal (0.05 = 5%)
        if short_pct is not None:
            out["short_interest_pct_at_date"] = float(short_pct) * 100
        shares_out = info.get("sharesOutstanding")
        if shares_out:
            out["shares_out_at_date_m"] = float(shares_out) / 1e6
        avg_vol = info.get("averageVolume") or info.get("averageVolume10days")
        if avg_vol:
            out["avg_volume_30d"] = float(avg_vol)
        # Fallback for market cap if screener missed it
        if "market_cap_at_date_m" not in out and info.get("marketCap"):
            out["market_cap_at_date_m"] = float(info["marketCap"]) / 1e6
        # Short ratio = short_int_shares / avg_daily_volume
        if shares_out and short_pct and avg_vol:
            short_shares = float(short_pct) * float(shares_out)
            out["short_ratio_days_at_date"] = short_shares / float(avg_vol) if avg_vol else None
    except Exception as e:
        out["_yfinance_err"] = f"{type(e).__name__}"

    if "market_cap_at_date_m" not in out and "shares_out_at_date_m" not in out:
        out["_status"] = f"all_sources_failed (screener:{out.get('_screener_err','')} yf:{out.get('_yfinance_err','')})"
    else:
        out["_status"] = "ok_current_snapshot"
    out.pop("_screener_err", None)
    out.pop("_yfinance_err", None)
    return out


def _fill_sec_insider(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Form 4 insider buys/sells in the 30 days BEFORE the catalyst.
    Stub — full implementation calls SEC EDGAR `/cgi-bin/browse-edgar` filtered
    to ownership filings + parses the XML. Leaving structured shell so a
    future chat can implement without touching the orchestrator.

    NOTE: now superseded by Finviz Elite insider feed (_fill_finviz) which
    is faster + cleaner than parsing SEC Form 4 XML. Keep this as a backup
    for tickers Finviz doesn't cover.
    """
    return {
        "_status": "deferred_to_section_chat (use Finviz instead)",
        "_note": "Form 4 parser implementation deferred. Finviz preferred.",
    }


# ────────────────────────────────────────────────────────────
# Free / paid source fillers added 2026-05-03
# ────────────────────────────────────────────────────────────

def _fill_openfda_faers(
    *, ticker: str, drug_name: Optional[str], catalyst_date: str,
) -> Dict[str, Any]:
    """OpenFDA FAERS adverse event report counts in 90d / 365d windows
    pre-catalyst. Free REST API at https://api.fda.gov/drug/event.json.

    FAERS uses several drug-name fields (drugs report under brand, generic,
    or substance names depending on the reporter). We OR-query across
    medicinalproduct, openfda.brand_name, openfda.generic_name, and
    openfda.substance_name in a single call so we don't miss reports just
    because the name field varies.

    Raw URL string (not requests' params dict) to avoid `+` getting
    URL-encoded as %2B which would break Lucene's syntax.
    """
    if not drug_name:
        return {"_status": "no_drug_name", "faers_data_source": None}
    try:
        import requests
        from urllib.parse import quote
        cd = datetime.fromisoformat(catalyst_date[:10])
        d90_start = (cd - timedelta(days=90)).strftime("%Y%m%d")
        d365_start = (cd - timedelta(days=365)).strftime("%Y%m%d")
        end = cd.strftime("%Y%m%d")
        # Take the first parenthesized name — for "nexiguran ziclumeran (NTLA-2001)"
        # we keep "nexiguran ziclumeran" which is the generic INN. Drop
        # plus signs, quote-escape for Lucene exact-phrase match.
        base = (drug_name or "").split("(")[0].strip()
        if not base:
            return {"_status": "drug_name_unusable", "faers_data_source": None}
        # Lucene exact-phrase match — quoted + URL-encoded
        # FAERS is case-insensitive but most fields are stored uppercase
        phrase = quote(f'"{base}"')
        out: Dict[str, Any] = {"faers_data_source": "openfda"}
        notes: List[str] = []
        for window_label, start in (("90d", d90_start), ("365d", d365_start)):
            # OR across drug-name fields — single API call, broader match
            search = (
                f"(patient.drug.medicinalproduct:{phrase}"
                f"+patient.drug.openfda.brand_name:{phrase}"
                f"+patient.drug.openfda.generic_name:{phrase}"
                f"+patient.drug.openfda.substance_name:{phrase})"
                f"+AND+receivedate:[{start}+TO+{end}]"
            )
            url = f"https://api.fda.gov/drug/event.json?search={search}&limit=1"
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                count = 0
                notes.append(f"{window_label}:404=no_matches")
            elif r.status_code == 200:
                meta = (r.json() or {}).get("meta", {}) or {}
                count = (meta.get("results") or {}).get("total", 0) or 0
                notes.append(f"{window_label}:{count}")
            else:
                # Non-200/404 — record the status but don't drop the row
                notes.append(f"{window_label}:status_{r.status_code}")
                count = None
            if window_label == "90d":
                out["adverse_event_count_90d_pre"] = count
            else:
                out["adverse_event_count_365d_pre"] = count
        out["_status"] = f"ok ({', '.join(notes)})"
        return out
    except Exception as e:
        return {"_status": f"faers_error:{type(e).__name__}:{str(e)[:60]}",
                "faers_data_source": None}


def _fill_clinicaltrials_gov(
    *, ticker: str, drug_name: Optional[str], catalyst_date: str,
) -> Dict[str, Any]:
    """clinicaltrials.gov v2 API — count of active trials at catalyst_date,
    total enrollment, count of protocol amendments in the 180 days pre-event.
    Free at https://clinicaltrials.gov/api/v2/studies.
    """
    if not drug_name:
        return {"_status": "no_drug_name", "ctgov_data_source": None}
    try:
        import requests
        clean = (drug_name or "").split("(")[0].strip()
        if not clean:
            return {"_status": "drug_name_unusable"}
        url = "https://clinicaltrials.gov/api/v2/studies"
        # Free-text search on intervention name
        params = {
            "query.intr": clean,
            "fields": "NCTId,EnrollmentCount,OverallStatus,LastUpdateSubmitDate,StudyFirstSubmitDate",
            "pageSize": 100,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return {"_status": f"ctgov_status_{r.status_code}", "ctgov_data_source": None}
        data = r.json() or {}
        studies = data.get("studies", []) or []
        cd = datetime.fromisoformat(catalyst_date[:10]).date()
        amendments_180d = 0
        active_count = 0
        total_enrollment = 0
        for s in studies:
            proto = s.get("protocolSection", {}) or {}
            status_module = proto.get("statusModule", {}) or {}
            status = status_module.get("overallStatus")
            enr = (proto.get("designModule", {}) or {}).get("enrollmentInfo", {}).get("count")
            if isinstance(enr, int):
                total_enrollment += enr
            if status in ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"):
                active_count += 1
            # Last update date
            last_update = status_module.get("lastUpdateSubmitDate")
            if last_update:
                try:
                    lu = datetime.fromisoformat(last_update).date()
                    if (cd - timedelta(days=180)) <= lu <= cd:
                        amendments_180d += 1
                except Exception:
                    pass
        return {
            "trial_count_active_at_date": active_count,
            "trial_total_enrollment": total_enrollment,
            "trial_amendments_180d_pre": amendments_180d,
            "ctgov_data_source": "clinicaltrials.gov_v2",
            "_status": "ok",
        }
    except Exception as e:
        return {"_status": f"ctgov_error:{type(e).__name__}", "ctgov_data_source": None}


def _fill_finviz(
    *, ticker: str,
) -> Dict[str, Any]:
    """Finviz Elite snapshot via `export.ashx?v=111` (Custom screener view).

    Column codes are well-documented for v=111 (NOT v=151 which uses
    different numbering — that was the prior bug). Reference:
    https://github.com/lit26/finvizfinance — Finviz screener overview/options/...

    v=111 column codes (subset we care about):
      1   Ticker          26  Insider Own       46  Perf Year
      6   Market Cap      27  Insider Trans     47  Perf YTD
      25  Float           28  Inst Own          48  Beta
                          29  Inst Trans        50  Volatility (Week)
                          30  Float Short       60  Earnings Date
                          31  Short Ratio       61  Price
                                                64  Recom (1=Strong Buy ... 5=Strong Sell)
                                                65  Target Price

    Returned in order requested. We parse the CSV by column NAME
    (case-insensitive) so we're resilient to Finviz reordering columns
    in their response.
    """
    api_key = (os.getenv("FINVIZ_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_finviz_key"}
    try:
        import requests
        # Snapshot columns: ticker, market cap, float, insider/inst own,
        # short interest, perf, volatility, earnings date, price, recom,
        # target price.
        cols = "1,6,25,26,27,28,29,30,31,47,48,50,60,61,64,65"
        url = (
            "https://elite.finviz.com/export.ashx"
            f"?v=111&t={ticker}&c={cols}&auth={api_key}"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return {"_status": f"finviz_status_{r.status_code}"}
        body = r.text or ""
        if not body or "<html" in body[:200].lower() or "login" in body[:200].lower():
            return {"_status": "finviz_auth_returned_html (check FINVIZ_API_KEY)"}
        # CSV — header row + one data row for our single-ticker filter.
        # Parse by NAME (case-insensitive) — Finviz column codes are
        # quirky and depend on the v= view; reading by name is resilient.
        import csv, io
        reader = csv.DictReader(io.StringIO(body))
        row = next(reader, None) or {}
        if not row:
            return {"_status": "finviz_empty_response"}

        def _to_float(s):
            if s is None or s in ("", "-", "N/A"):
                return None
            try:
                return float(str(s).replace("%", "").replace("$", "").replace(",", ""))
            except Exception:
                return None

        # Build a case-insensitive lookup of header → value
        norm = {(k or "").strip().lower(): v for k, v in row.items()}
        def _get(*candidates):
            for name in candidates:
                v = norm.get(name.lower())
                if v not in (None, "", "-", "N/A"):
                    return v
            return None

        recom  = _to_float(_get("Recom", "Analyst Recom", "Recommendation"))
        target = _to_float(_get("Target Price", "Target", "Price Target"))
        price  = _to_float(_get("Price"))
        out = {
            "analyst_recommendation_avg": recom,
            "analyst_target_price_usd": target,
            "analyst_target_upside_pct": ((target - price) / price * 100) if (target and price) else None,
            "finviz_perf_ytd_pct": _to_float(_get("Perf YTD", "Performance YTD")),
            "finviz_data_source": "finviz_elite_export_v151",
            "_status": (
                "ok" if (recom or target or price) else
                f"finviz_returned_{len(norm)}_cols_but_no_recom_target_price"
            ),
            "_extra_columns_seen": list(norm.keys()),  # surfaces what we DID get
        }
        return out
    except Exception as e:
        return {"_status": f"finviz_error:{type(e).__name__}:{str(e)[:80]}"}


def _fill_pubmed(*, drug_name: Optional[str], catalyst_date: str) -> Dict[str, Any]:
    """PubMed publication count per drug — deferred. NCBI E-utilities API:
    https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=DRUG&datetype=pdat&mindate=YYYY/MM/DD
    """
    return {"_status": "deferred_v2"}


def _fill_uspto_patent(*, drug_name: Optional[str]) -> Dict[str, Any]:
    """USPTO patent runway — deferred. Drug-to-patent mapping is non-trivial
    without Orange Book wrapper; openFDA Orange Book API is the better path
    when we wire it later.
    """
    return {"_status": "deferred_v2"}


def _fill_nih_reporter(*, drug_name: Optional[str], ticker: str) -> Dict[str, Any]:
    """NIH RePORTER active grants — deferred."""
    return {"_status": "deferred_v2"}


def _fill_sec_institutional(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Form 13F institutional holdings (most recent quarter before catalyst).
    Stub for the same reason as _fill_sec_insider.
    """
    return {"_status": "deferred_to_section_chat"}


def _fill_catalyst_metadata(
    *, catalyst_id: int, catalyst_type: Optional[str], cap_bucket: str,
    db: BiotechDatabase,
) -> Dict[str, Any]:
    """Catalyst-specific snapshot: probability, regime, product class."""
    out: Dict[str, Any] = {"cap_bucket": cap_bucket, "_status": "ok"}
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT confidence_score, drug_name, indication
                FROM catalyst_universe
                WHERE id = %s
                """,
                (catalyst_id,),
            )
            row = cur.fetchone()
        if row:
            out["p_approval_at_pred"] = float(row[0]) if row[0] is not None else None
            drug_aliases = [row[1]] if row[1] else []
            indication = row[2] or ""
            try:
                from services.drug_programs import classify_product
                out["product_class"] = classify_product(
                    drug_aliases, indication, [catalyst_type or ""],
                )
            except Exception:
                out["product_class"] = "unknown"
        try:
            from services.disclosure_regime import classify_disclosure_regime
            out["regime"] = classify_disclosure_regime(catalyst_type or "")
        except Exception:
            out["regime"] = None
    except Exception as e:
        out["_status"] = f"error:{type(e).__name__}"
    return out


def _fill_outcomes_mirror(
    *, catalyst_id: int, db: BiotechDatabase,
) -> Dict[str, Any]:
    """Mirror realized outcomes + Gemini label into features table for
    one-row joins."""
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT actual_move_pct_1d, actual_move_pct_7d, actual_move_pct_30d,
                       outcome_label_class, pre_event_price
                FROM post_catalyst_outcomes
                WHERE catalyst_id = %s
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (catalyst_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"_status": "no_outcome_row"}
        m1d, m7d, m30d, gemini_label, pre_price = row
        out = {
            "actual_move_pct_1d": float(m1d) if m1d is not None else None,
            "actual_move_pct_7d": float(m7d) if m7d is not None else None,
            "actual_move_pct_30d": float(m30d) if m30d is not None else None,
            "outcome_label_gemini": gemini_label,
            "outcome_label_price_proxy": _outcome_label_price_proxy(
                float(m7d) if m7d is not None else None
            ),
            "pre_event_price": float(pre_price) if pre_price is not None else None,
            "_status": "ok",
        }
        # Consensus: if gemini_label exists AND price_proxy agrees, mark consensus.
        # (Future chat: extend to anthropic/openai labels for multi-source vote.)
        if out["outcome_label_gemini"] and out["outcome_label_price_proxy"]:
            g = out["outcome_label_gemini"].upper()
            p = out["outcome_label_price_proxy"]
            agree = (
                (g in ("APPROVED", "MET_ENDPOINT") and p == "POSITIVE") or
                (g in ("REJECTED", "MISSED_ENDPOINT", "WITHDRAWN") and p == "NEGATIVE") or
                (g in ("MIXED", "DELAYED") and p == "MIXED")
            )
            if agree:
                out["outcome_label_consensus"] = "AGREE_GEMINI_PRICE"
                out["outcome_confidence"] = 0.85
            else:
                out["outcome_label_consensus"] = "DISAGREE_GEMINI_PRICE"
                out["outcome_confidence"] = 0.40
        return out
    except Exception as e:
        return {"_status": f"error:{type(e).__name__}"}


# ────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────
def compute_event_features(
    catalyst_id: int,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Compute (or refresh) feature row for one catalyst event. Idempotent.

    Returns the dict that was upserted. `backfill_source_status` in the
    return value tells you which sources succeeded.
    """
    db = BiotechDatabase()

    # Look up catalyst metadata first
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ticker, catalyst_date::text, catalyst_type, drug_name
            FROM catalyst_universe WHERE id = %s
            """,
            (catalyst_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"error": f"catalyst_id {catalyst_id} not found"}
    cid, ticker, catalyst_date, catalyst_type, drug_name = row

    # Skip if already backfilled and refresh=False
    if not refresh:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM catalyst_event_features WHERE catalyst_id = %s",
                (cid,),
            )
            if cur.fetchone():
                return {"skipped": "already_exists", "catalyst_id": cid}

    # Run all fillers
    feats: Dict[str, Any] = {
        "catalyst_id": cid,
        "ticker": ticker,
        "catalyst_date": catalyst_date,
        "catalyst_type": catalyst_type,
    }
    statuses: Dict[str, str] = {}

    ms = _fill_microstructure(ticker=ticker, db=db)
    statuses["microstructure"] = ms.pop("_status", "")
    feats.update(ms)
    cap_bucket = _cap_bucket(feats.get("market_cap_at_date_m"))

    md = _fill_catalyst_metadata(
        catalyst_id=cid, catalyst_type=catalyst_type, cap_bucket=cap_bucket, db=db,
    )
    statuses["metadata"] = md.pop("_status", "")
    feats.update(md)

    pa = _fill_price_action(ticker=ticker, catalyst_date=catalyst_date, db=db)
    statuses["price_action"] = pa.pop("_status", "")
    feats.update(pa)

    pr = _fill_peer_relative(
        ticker=ticker, catalyst_date=catalyst_date, db=db,
        ticker_runup_30d=feats.get("runup_pct_30d"),
        ticker_runup_90d=feats.get("runup_pct_90d"),
    )
    statuses["peer_relative"] = pr.pop("_status", "")
    feats.update(pr)

    cs = _fill_capital_structure(ticker=ticker, catalyst_date=catalyst_date)
    statuses["capital_structure"] = cs.pop("_status", "")
    feats.update(cs)

    po = _fill_massive_options(ticker=ticker, catalyst_date=catalyst_date)
    statuses["massive_options"] = po.pop("_status", "")
    feats.update(po)

    ins = _fill_sec_insider(ticker=ticker, catalyst_date=catalyst_date)
    statuses["sec_insider"] = ins.pop("_status", "")
    feats.update({k: v for k, v in ins.items() if not k.startswith("_")})

    inst = _fill_sec_institutional(ticker=ticker, catalyst_date=catalyst_date)
    statuses["sec_institutional"] = inst.pop("_status", "")
    feats.update({k: v for k, v in inst.items() if not k.startswith("_")})

    om = _fill_outcomes_mirror(catalyst_id=cid, db=db)
    statuses["outcomes_mirror"] = om.pop("_status", "")
    feats.update(om)

    # New sources added 2026-05-03
    fae = _fill_openfda_faers(ticker=ticker, drug_name=drug_name, catalyst_date=catalyst_date)
    statuses["openfda_faers"] = fae.pop("_status", "")
    feats.update({k: v for k, v in fae.items() if not k.startswith("_")})

    ct = _fill_clinicaltrials_gov(ticker=ticker, drug_name=drug_name, catalyst_date=catalyst_date)
    statuses["clinicaltrials_gov"] = ct.pop("_status", "")
    feats.update({k: v for k, v in ct.items() if not k.startswith("_")})

    fv = _fill_finviz(ticker=ticker)
    statuses["finviz"] = fv.pop("_status", "")
    feats.update({k: v for k, v in fv.items() if not k.startswith("_")})

    # Stubs for v2:
    statuses["pubmed"] = _fill_pubmed(drug_name=drug_name, catalyst_date=catalyst_date)["_status"]
    statuses["uspto_patent"] = _fill_uspto_patent(drug_name=drug_name)["_status"]
    statuses["nih_reporter"] = _fill_nih_reporter(drug_name=drug_name, ticker=ticker)["_status"]

    # Days-until-catalyst (negative for past events)
    try:
        cd = date.fromisoformat(catalyst_date[:10])
        feats["days_until_catalyst_at_pred"] = (cd - date.today()).days
    except Exception:
        feats["days_until_catalyst_at_pred"] = None

    feats["backfill_source_status"] = statuses

    # ─── UPSERT ───
    cols = [k for k in feats if k != "_status"]
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join(cols)
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "catalyst_id")
    sql = f"""
        INSERT INTO catalyst_event_features ({col_names}, updated_at)
        VALUES ({placeholders}, NOW())
        ON CONFLICT (catalyst_id) DO UPDATE SET
          {update_clause},
          updated_at = NOW()
    """
    import json as _json
    values = []
    for c in cols:
        v = feats[c]
        if isinstance(v, dict):
            v = _json.dumps(v)
        values.append(v)

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()

    return feats


def backfill_features_batch(
    *,
    limit: int = 100,
    only_labeled: bool = False,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Find catalyst events without a feature row (or stale) and fill them.

    Idempotent + safe to run repeatedly. Returns batch summary.
    """
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        # Find catalysts that need backfill: have an outcome row, no feature row
        # (or refresh=True), filtered by labeled-only when requested.
        if only_labeled:
            cur.execute(
                """
                SELECT cu.id
                FROM catalyst_universe cu
                JOIN post_catalyst_outcomes pco ON pco.catalyst_id = cu.id
                LEFT JOIN catalyst_event_features cef ON cef.catalyst_id = cu.id
                WHERE pco.outcome_label_class IS NOT NULL
                  AND (cef.id IS NULL OR %s)
                ORDER BY cu.catalyst_date DESC
                LIMIT %s
                """,
                (refresh, limit),
            )
        else:
            cur.execute(
                """
                SELECT cu.id
                FROM catalyst_universe cu
                LEFT JOIN catalyst_event_features cef ON cef.catalyst_id = cu.id
                WHERE cu.status = 'active'
                  AND (cef.id IS NULL OR %s)
                ORDER BY cu.catalyst_date DESC
                LIMIT %s
                """,
                (refresh, limit),
            )
        catalyst_ids = [r[0] for r in cur.fetchall()]

    if not catalyst_ids:
        return {"checked": 0, "filled": 0, "skipped_existing": 0, "errors": 0,
                "_note": "No catalysts need backfill — try refresh=True to recompute"}

    filled = 0
    skipped = 0
    errors = 0
    error_samples: List[Dict[str, Any]] = []
    for cid in catalyst_ids:
        try:
            result = compute_event_features(cid, refresh=refresh)
            if result.get("error"):
                errors += 1
                if len(error_samples) < 5:
                    error_samples.append({"catalyst_id": cid, "error": result["error"]})
            elif result.get("skipped"):
                skipped += 1
            else:
                filled += 1
        except Exception as e:
            errors += 1
            if len(error_samples) < 5:
                error_samples.append({
                    "catalyst_id": cid,
                    "error": f"{type(e).__name__}: {str(e)[:120]}",
                })
            logger.exception("backfill_features_batch row %s failed", cid)

    return {
        "checked": len(catalyst_ids),
        "filled": filled,
        "skipped_existing": skipped,
        "errors": errors,
        "error_samples": error_samples,
        "_filter": {"only_labeled": only_labeled, "refresh": refresh, "limit": limit},
    }


def get_coverage_report() -> Dict[str, Any]:
    """How many feature-store rows exist + per-column population %."""
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM catalyst_event_features")
        total_features = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM catalyst_universe WHERE status = 'active'")
        total_universe = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM catalyst_universe cu "
            "JOIN post_catalyst_outcomes pco ON pco.catalyst_id = cu.id "
            "WHERE pco.outcome_label_class IS NOT NULL"
        )
        total_labeled = cur.fetchone()[0]
        # Population % per key column
        cur.execute(
            """
            SELECT
              ROUND(100.0 * COUNT(market_cap_at_date_m)         / NULLIF(COUNT(*), 0), 1) AS pct_market_cap,
              ROUND(100.0 * COUNT(runup_pct_30d)                / NULLIF(COUNT(*), 0), 1) AS pct_runup_30d,
              ROUND(100.0 * COUNT(xbi_runup_30d)                / NULLIF(COUNT(*), 0), 1) AS pct_xbi_runup,
              ROUND(100.0 * COUNT(cash_at_date_m)               / NULLIF(COUNT(*), 0), 1) AS pct_cash,
              ROUND(100.0 * COUNT(short_interest_pct_at_date)   / NULLIF(COUNT(*), 0), 1) AS pct_short_interest,
              ROUND(100.0 * COUNT(atm_iv_at_date)               / NULLIF(COUNT(*), 0), 1) AS pct_massive_options,
              ROUND(100.0 * COUNT(p_approval_at_pred)           / NULLIF(COUNT(*), 0), 1) AS pct_p_approval,
              ROUND(100.0 * COUNT(product_class)                / NULLIF(COUNT(*), 0), 1) AS pct_product_class,
              ROUND(100.0 * COUNT(actual_move_pct_7d)           / NULLIF(COUNT(*), 0), 1) AS pct_actual_7d,
              ROUND(100.0 * COUNT(outcome_label_gemini)         / NULLIF(COUNT(*), 0), 1) AS pct_label_gemini,
              ROUND(100.0 * COUNT(outcome_label_price_proxy)    / NULLIF(COUNT(*), 0), 1) AS pct_label_proxy,
              ROUND(100.0 * COUNT(drug_npv_b_at_date)           / NULLIF(COUNT(*), 0), 1) AS pct_drug_npv,
              ROUND(100.0 * COUNT(priced_in_fraction_at_date)   / NULLIF(COUNT(*), 0), 1) AS pct_priced_in
            FROM catalyst_event_features
            """
        )
        cols = [d[0] for d in cur.description]
        pop_row = cur.fetchone()
        population_pct = dict(zip(cols, [float(v) if v is not None else None for v in pop_row]))
    return {
        "total_features_rows": total_features,
        "total_active_catalysts": total_universe,
        "total_labeled_catalysts": total_labeled,
        "feature_coverage_pct": round(100 * total_features / total_universe, 1) if total_universe else 0,
        "feature_coverage_of_labeled_pct": round(100 * total_features / total_labeled, 1) if total_labeled else 0,
        "column_population_pct": population_pct,
    }
