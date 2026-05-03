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
    """Finviz snapshot via the public quote.ashx HTML page.

    Why scrape instead of API: Finviz Elite's `export.ashx?c=` parameter
    only respects column codes for certain views (v=151 returns 10 fixed
    fields; v=111 ignores c= and returns its default basic columns).
    Recom + Target Price aren't in the snapshot/performance views' code
    space at all. The public quote.ashx page DOES surface every snapshot
    field (incl. Recom, Target Price, Insider Trans, Inst Own, Short Float,
    Perf YTD) in a stable HTML table — this is the same approach the
    finvizfinance Python library takes. No API key needed for the page.

    Insider-transactions backfill (Form 4 detail) uses a separate
    insider_export.ashx Elite endpoint and stays deferred.
    """
    try:
        import requests, re
        # User-Agent needed; Finviz blocks the default python-requests UA.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        }
        # Try Elite first (slightly fresher data + auth-respected on the page),
        # fall back to public if no key.
        api_key = (os.getenv("FINVIZ_API_KEY") or "").strip()
        url = (
            f"https://elite.finviz.com/quote.ashx?t={ticker}&auth={api_key}"
            if api_key else f"https://finviz.com/quote.ashx?t={ticker}"
        )
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {"_status": f"finviz_quote_status_{r.status_code}"}
        html = r.text or ""
        if "snapshot-td" not in html:
            return {"_status": "finviz_no_snapshot_table (page format may have changed)"}

        # Finviz's modern (2025+) snapshot table uses inner divs:
        #   <td class="snapshot-td2 cursor-pointer ...">
        #     <div class="snapshot-td-label">LABEL</div>
        #   </td>
        #   <td class="snapshot-td2 ...">
        #     <div class="snapshot-td-content"><b>VALUE</b></div>
        #   </td>
        # Match each label-div followed by the next content-div.
        # Allow nested HTML in BOTH label and value (analyst-linked fields
        # like Recom / Target Price wrap the label in an <a href> tag).
        pairs = re.findall(
            r'<div class="snapshot-td-label[^"]*">(.*?)</div>'
            r'.*?<div class="snapshot-td-content[^"]*">(.*?)</div>',
            html, flags=re.DOTALL,
        )

        # Strip nested HTML tags (<a>, <b>, <span class="color-text...">,
        # <small>, links) from BOTH labels and values, decode common entities.
        def _clean(s):
            s = re.sub(r"<[^>]+>", "", s).strip()
            s = s.replace("&nbsp;", " ").replace("&amp;", "&")
            return s

        snap = {_clean(label): _clean(val) for label, val in pairs}
        if not snap:
            return {"_status": "finviz_snapshot_empty (regex matched 0 pairs)"}

        def _to_float(s):
            if s is None or s in ("", "-", "N/A"):
                return None
            try:
                return float(str(s).replace("%", "").replace("$", "").replace(",", ""))
            except Exception:
                return None

        # Try multiple label variants Finviz has used for these fields
        def _get(*candidates):
            for n in candidates:
                v = snap.get(n)
                if v not in (None, "", "-"):
                    return v
            return None

        recom  = _to_float(_get("Recom", "Analyst Recom", "Recommendation"))
        target = _to_float(_get("Target Price", "Price Target", "Target"))
        price  = _to_float(_get("Price"))
        ytd    = _to_float(_get("Perf YTD", "Performance YTD"))
        # When Recom/Target are missing, surface which keys we DID see so
        # we can adjust the variant list (or confirm the ticker really has
        # no analyst coverage on Finviz).
        sample_keys = sorted(snap.keys())[:30]
        out = {
            "analyst_recommendation_avg": recom,
            "analyst_target_price_usd": target,
            "analyst_target_upside_pct": ((target - price) / price * 100) if (target and price) else None,
            "finviz_perf_ytd_pct": ytd,
            "finviz_data_source": "finviz_quote_html_scrape",
            "_status": (
                "ok" if (recom is not None or target is not None) else
                f"finviz_no_recom_target_in_{len(snap)}_fields"
            ),
            "_snapshot_field_count": len(snap),
            "_snapshot_sample_keys": sample_keys,
        }
        return out
    except Exception as e:
        return {"_status": f"finviz_error:{type(e).__name__}:{str(e)[:80]}"}


# ────────────────────────────────────────────────────────────
# Finviz quote-page extensions: news / ratings / insider full table
# All three scrape from the same page — single fetch, three parses.
# ────────────────────────────────────────────────────────────

# Sources we treat as "high quality" for the news_high_quality_present flag
_FINVIZ_HIGH_QUALITY_SOURCES = {
    "Bloomberg", "Reuters", "WSJ", "Wall Street Journal", "FT",
    "Financial Times", "Barron's", "CNBC", "Forbes", "MarketWatch",
}


def _fetch_finviz_quote_html(ticker: str) -> Optional[str]:
    """Single fetch of the Finviz quote page — reused by news / ratings /
    insider scrapers. Returns None on failure."""
    try:
        import requests
        api_key = (os.getenv("FINVIZ_API_KEY") or "").strip()
        url = (f"https://elite.finviz.com/quote.ashx?t={ticker}&auth={api_key}"
               if api_key else f"https://finviz.com/quote.ashx?t={ticker}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        }
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        return None


def _fill_finviz_news(*, ticker: str, catalyst_date: str, html: Optional[str] = None) -> Dict[str, Any]:
    """News-table scrape: count + source diversity + high-quality-source flag.
    Date format on Finviz: 'May-01-26 04:01PM' for first of day, 'HH:MMAM/PM'
    for subsequent same-day rows.
    """
    if html is None:
        html = _fetch_finviz_quote_html(ticker)
    if not html:
        return {"_status": "no_html"}
    try:
        import re
        # News table rows have onclick="trackAndOpenNews(event, 'SOURCE', '...')"
        # paired with the date in the first <td>. Extract source + date pairs.
        # Pattern matches: onclick="trackAndOpenNews(event, 'GlobeNewswire', '/news/...')"
        # immediately following or preceding a <td> with date text.
        # Simpler: parse the whole news-table block.
        m = re.search(r'<table[^>]*id="news-table"[^>]*>(.*?)</table>', html, re.DOTALL)
        if not m:
            return {"_status": "no_news_table"}
        block = m.group(1)

        # Each row pairs a date-td with a source onclick handler.
        rows = re.findall(
            r'<td[^>]*align="right"[^>]*>\s*([A-Za-z]{3}-\d{1,2}-\d{2}\s+\d{1,2}:\d{2}[AP]M|\d{1,2}:\d{2}[AP]M)\s*</td>'
            r'.*?trackAndOpenNews\(event,\s*[\'"]([^\'"]+)[\'"]',
            block, re.DOTALL,
        )
        if not rows:
            return {"_status": "news_table_empty_after_parse"}

        # Forward-fill dates: rows with only HH:MMAM/PM inherit the previous full date
        cd = datetime.fromisoformat(catalyst_date[:10])
        cd_only = cd.date()
        d7_cutoff = cd_only - timedelta(days=7)
        d30_cutoff = cd_only - timedelta(days=30)
        last_date: Optional[date] = None
        count_7d = 0
        count_30d = 0
        sources_30d: List[str] = []
        latest_date_str: Optional[str] = None

        for raw_date, source in rows:
            # If the date string contains a hyphenated month (e.g. "May-01-26"),
            # parse fully; otherwise use the previously seen date.
            full_date = None
            mm = re.match(r'([A-Za-z]{3})-(\d{1,2})-(\d{2})', raw_date)
            if mm:
                month_str, day_str, year_str = mm.groups()
                try:
                    parsed = datetime.strptime(
                        f"{month_str} {int(day_str)} 20{year_str}", "%b %d %Y"
                    ).date()
                    full_date = parsed
                    last_date = parsed
                    if latest_date_str is None:
                        latest_date_str = parsed.isoformat()
                except ValueError:
                    pass
            elif last_date:
                full_date = last_date

            if not full_date:
                continue

            if full_date <= cd_only and full_date >= d30_cutoff:
                count_30d += 1
                sources_30d.append(source)
            if full_date <= cd_only and full_date >= d7_cutoff:
                count_7d += 1

        unique_sources = sorted(set(sources_30d))
        from collections import Counter
        top_sources = ", ".join(s for s, _ in Counter(sources_30d).most_common(5))
        high_quality_present = any(
            any(hq.lower() in src.lower() for hq in _FINVIZ_HIGH_QUALITY_SOURCES)
            for src in unique_sources
        )
        return {
            "finviz_news_count_7d_pre": count_7d,
            "finviz_news_count_30d_pre": count_30d,
            "finviz_news_source_count_30d": len(unique_sources),
            "finviz_news_top_sources": top_sources,
            "finviz_news_high_quality_present": high_quality_present,
            "finviz_news_latest_date": latest_date_str,
            "_status": f"ok ({count_30d} news in 30d pre)",
        }
    except Exception as e:
        return {"_status": f"news_error:{type(e).__name__}:{str(e)[:60]}"}


def _fill_finviz_ratings(*, ticker: str, catalyst_date: str, html: Optional[str] = None) -> Dict[str, Any]:
    """Analyst ratings table — chronological upgrade/downgrade events with
    firm names + price target changes.
    """
    if html is None:
        html = _fetch_finviz_quote_html(ticker)
    if not html:
        return {"_status": "no_html"}
    try:
        import re
        m = re.search(
            r'<table[^>]*class="js-table-ratings[^"]*"[^>]*>(.*?)</table>',
            html, re.DOTALL,
        )
        if not m:
            return {"_status": "no_ratings_table"}
        block = m.group(1)

        # Each row: <td>Date</td> <td><span>Action</span></td> <td>Firm</td>
        # <td>RatingChange</td> <td>Price Target Change</td>
        # NOTE: action is wrapped in <span class="fv-label ...">; firm/rating/PT
        # cells have inline color classes. Use (.*?) + _clean() to handle
        # nested HTML.
        def _clean_cell(s):
            s = re.sub(r"<[^>]+>", "", s).strip()
            s = s.replace("&nbsp;", " ").replace("&rarr;", "→").replace("&amp;", "&")
            return s

        rows = re.findall(
            r'<tr[^>]*>\s*'
            r'<td[^>]*>([A-Za-z]{3}-\d{1,2}-\d{2})</td>\s*'
            r'<td[^>]*>(.*?)</td>\s*'
            r'<td[^>]*>(.*?)</td>\s*'
            r'<td[^>]*>(.*?)</td>\s*'
            r'<td[^>]*>(.*?)</td>',
            block, re.DOTALL,
        )
        if not rows:
            return {"_status": "ratings_table_empty"}

        cd_only = datetime.fromisoformat(catalyst_date[:10]).date()
        d30_cutoff = cd_only - timedelta(days=30)
        upgrades = downgrades = pt_changes = 0
        latest_action = latest_firm = latest_date_str = None

        for raw_date, action, firm, rating_change, pt_change in rows:
            try:
                month_str, day_str, year_str = raw_date.split("-")
                d = datetime.strptime(
                    f"{month_str} {int(day_str)} 20{year_str}", "%b %d %Y"
                ).date()
            except Exception:
                continue
            action = _clean_cell(action)
            firm = _clean_cell(firm)
            pt_change_clean = _clean_cell(pt_change)
            if latest_action is None:
                latest_action = action
                latest_firm = firm
                latest_date_str = d.isoformat()
            if d <= cd_only and d >= d30_cutoff:
                if "Upgrade" in action:
                    upgrades += 1
                elif "Downgrade" in action:
                    downgrades += 1
                if pt_change_clean and pt_change_clean not in ("-", ""):
                    pt_changes += 1

        return {
            "finviz_analyst_upgrades_30d_pre": upgrades,
            "finviz_analyst_downgrades_30d_pre": downgrades,
            "finviz_analyst_pt_changes_30d_pre": pt_changes,
            "finviz_analyst_latest_action": latest_action,
            "finviz_analyst_latest_firm": latest_firm,
            "finviz_analyst_latest_date": latest_date_str,
            "_status": f"ok (parsed {len(rows)} rows)",
        }
    except Exception as e:
        return {"_status": f"ratings_error:{type(e).__name__}:{str(e)[:60]}"}


# Insider transaction codes that count as a real BUY (not grant/option exercise)
_INSIDER_BUY_CODES = {"P", "Buy"}
_INSIDER_SELL_CODES = {"S", "Sale"}
_NAMED_OFFICER_TITLES = {"CEO", "CFO", "Director", "President", "Chairman", "Chief"}


def _fill_finviz_insider_full(*, ticker: str, catalyst_date: str, html: Optional[str] = None) -> Dict[str, Any]:
    """Insider transactions full table — extract named-officer transactions
    (filter out compensation grants). Finviz HTML structure: rows in the
    table after the snapshot table, with columns: Insider | Relationship |
    Date | Transaction | Cost | #Shares | Value ($) | #Shares Total | SEC Form 4.
    """
    if html is None:
        html = _fetch_finviz_quote_html(ticker)
    if not html:
        return {"_status": "no_html"}
    try:
        import re
        # Insider table is identified by 'body-table' class + 'insider' in the row classes
        # OR by the 'Insider Trading' header. Conservative match: find table with
        # 'Insider Trading' nearby.
        # Easier path: find rows with class="cursor-pointer..." that come after
        # an <a href> linking to /insidertrading/. Limited to 100 rows.
        m = re.search(
            r'<table[^>]*class="[^"]*body-table[^"]*"[^>]*>(.*?)</table>',
            html, re.DOTALL,
        )
        if not m:
            return {"_status": "no_insider_table_found"}
        block = m.group(1)

        # Match rows: insider name (wrapped in <a> with no closing </a>),
        # relationship, date (format "Mar 02 '26"), transaction (wrapped in
        # <span>), cost, shares, value, etc. Use (.*?) + clean for cells
        # with nested HTML.
        def _clean_cell(s):
            s = re.sub(r"<[^>]+>", "", s).strip()
            s = s.replace("&nbsp;", " ").replace("&amp;", "&")
            return s

        rows = re.findall(
            r'<tr[^>]*class="fv-insider-row[^"]*"[^>]*>\s*'
            r'<td[^>]*>(.*?)</td>\s*'                                # insider name (in <a>)
            r'<td[^>]*>(.*?)</td>\s*'                                # relationship
            r'<td[^>]*>([A-Za-z]{3}\s+\d{1,2}\s+\'\d{2})</td>\s*'   # date "Mar 02 '26"
            r'<td[^>]*>(.*?)</td>\s*'                                # transaction (in <span>)
            r'<td[^>]*>(.*?)</td>\s*'                                # cost
            r'<td[^>]*>(.*?)</td>\s*'                                # shares
            r'<td[^>]*>(.*?)</td>',                                  # value $
            block, re.DOTALL,
        )
        if not rows:
            return {"_status": "insider_table_no_rows_parsed"}

        cd = datetime.fromisoformat(catalyst_date[:10])
        cd_only = cd.date()
        d30_cutoff = cd_only - timedelta(days=30)

        def _parse_finviz_insider_date(s: str) -> Optional[date]:
            # Format: "Mar 02 '26" — year is the apostrophe-2-digit
            try:
                m = re.match(r"([A-Za-z]{3})\s+(\d{1,2})\s+'(\d{2})", s.strip())
                if not m:
                    return None
                month_str, day_str, year_str = m.groups()
                return datetime.strptime(
                    f"{month_str} {int(day_str)} 20{year_str}", "%b %d %Y"
                ).date()
            except Exception:
                return None

        def _to_int(s: str) -> int:
            s = re.sub(r"[^\d-]", "", s)
            try:
                return int(s)
            except Exception:
                return 0

        def _to_float(s: str) -> float:
            s = re.sub(r"[^\d.\-]", "", s)
            try:
                return float(s)
            except Exception:
                return 0.0

        named_buys = named_sells = 0
        named_buy_value = 0.0
        top_buyer_title: Optional[str] = None
        top_buyer_value = 0.0

        for name, rel, date_str, tx, cost, shares, value in rows:
            d = _parse_finviz_insider_date(date_str)
            if d is None or d > cd_only or d < d30_cutoff:
                continue
            tx_clean = _clean_cell(tx)
            value_usd = _to_float(_clean_cell(value))
            rel_clean = _clean_cell(rel)
            is_named = any(t.lower() in rel_clean.lower() for t in _NAMED_OFFICER_TITLES)
            if not is_named:
                continue
            if any(c.lower() in tx_clean.lower() for c in _INSIDER_BUY_CODES):
                named_buys += 1
                named_buy_value += value_usd
                if value_usd > top_buyer_value:
                    top_buyer_value = value_usd
                    top_buyer_title = rel_clean[:60]
            elif any(c.lower() in tx_clean.lower() for c in _INSIDER_SELL_CODES):
                named_sells += 1

        return {
            "finviz_insider_buys_named_30d": named_buys,
            "finviz_insider_sells_named_30d": named_sells,
            "finviz_insider_buy_value_named_usd_30d": round(named_buy_value, 2) if named_buy_value else 0,
            "finviz_insider_top_buyer_title": top_buyer_title,
            "_status": f"ok (named: {named_buys} buys / {named_sells} sells in 30d)",
        }
    except Exception as e:
        return {"_status": f"insider_full_error:{type(e).__name__}:{str(e)[:60]}"}


# ────────────────────────────────────────────────────────────
# Finnhub (insider transactions, price-target dispersion, news sentiment)
# ────────────────────────────────────────────────────────────

def _fill_finnhub_insider(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Insider transactions (Form 4) + sentiment from Finnhub.
    Endpoint: /stock/insider-transactions?symbol=&from=&to= and /stock/insider-sentiment.
    Replaces the SEC Form 4 stub — Finnhub already parses the raw filings.
    """
    api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_finnhub_key"}
    try:
        import requests
        cd = datetime.fromisoformat(catalyst_date[:10])
        d30_start = (cd - timedelta(days=30)).strftime("%Y-%m-%d")
        d90_start = (cd - timedelta(days=90)).strftime("%Y-%m-%d")
        end = cd.strftime("%Y-%m-%d")
        out: Dict[str, Any] = {"finnhub_data_source": "finnhub_v1"}

        # 1. Insider transactions in 30d window pre-catalyst
        r = requests.get(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": ticker, "from": d30_start, "to": end, "token": api_key},
            timeout=15,
        )
        if r.status_code != 200:
            return {"_status": f"insider_status_{r.status_code}", **out}
        txs = (r.json() or {}).get("data", []) or []
        # transactionCode: P=purchase, S=sale, A=award/grant, D=disposition
        buys = [t for t in txs if (t.get("transactionCode") or "").upper() == "P"]
        sells = [t for t in txs if (t.get("transactionCode") or "").upper() == "S"]
        out["finnhub_insider_buys_30d_pre"] = len(buys)
        out["finnhub_insider_sells_30d_pre"] = len(sells)
        # Net value: sum(buy_value) - sum(sell_value); value = share × transactionPrice
        def _val(t):
            sh = t.get("share") or 0
            pr = t.get("transactionPrice") or 0
            return float(sh) * float(pr)
        buy_value = sum(_val(t) for t in buys)
        sell_value = sum(_val(t) for t in sells)
        out["finnhub_insider_net_value_usd_30d_pre"] = round(buy_value - sell_value, 2)

        # 2. Insider sentiment (mspr = monthly share purchase ratio, avg over 3 months)
        r = requests.get(
            "https://finnhub.io/api/v1/stock/insider-sentiment",
            params={"symbol": ticker, "from": d90_start, "to": end, "token": api_key},
            timeout=15,
        )
        if r.status_code == 200:
            sent_data = (r.json() or {}).get("data", []) or []
            msprs = [d.get("mspr") for d in sent_data if d.get("mspr") is not None]
            if msprs:
                out["finnhub_insider_sentiment_3m"] = round(sum(msprs) / len(msprs), 3)

        out["_status"] = "ok"
        return out
    except Exception as e:
        return {"_status": f"finnhub_insider_error:{type(e).__name__}:{str(e)[:60]}"}


def _fill_finnhub_price_target(
    *, ticker: str,
) -> Dict[str, Any]:
    """Analyst price target dispersion. Endpoint: /stock/price-target.
    Returns count + low/high/median — dispersion is a calibration signal
    (narrow consensus = high analyst confidence; wide = uncertainty)."""
    api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_finnhub_key"}
    try:
        import requests
        r = requests.get(
            "https://finnhub.io/api/v1/stock/price-target",
            params={"symbol": ticker, "token": api_key},
            timeout=15,
        )
        if r.status_code != 200:
            return {"_status": f"target_status_{r.status_code}"}
        d = r.json() or {}
        high = d.get("targetHigh")
        low = d.get("targetLow")
        median = d.get("targetMedian") or d.get("targetMean")
        out = {
            "finnhub_target_high_usd": float(high) if high else None,
            "finnhub_target_low_usd": float(low) if low else None,
            "finnhub_target_median_usd": float(median) if median else None,
        }
        if high and low and median and median > 0:
            out["finnhub_target_dispersion_pct"] = round(
                (float(high) - float(low)) / float(median) * 100, 2
            )
        out["_status"] = "ok" if any(out.values()) else "no_target_data"
        return out
    except Exception as e:
        return {"_status": f"finnhub_target_error:{type(e).__name__}"}


def _fill_finnhub_recommendation(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Analyst recommendation trends — buy/hold/sell counts, latest period
    + change vs 3 months ago. Endpoint: /stock/recommendation."""
    api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_finnhub_key"}
    try:
        import requests
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": ticker, "token": api_key},
            timeout=15,
        )
        if r.status_code != 200:
            return {"_status": f"recom_status_{r.status_code}"}
        recs = r.json() or []
        if not recs:
            return {"_status": "no_recommendation_data"}
        # Sort by period DESC (Finnhub usually returns sorted, but be safe)
        recs.sort(key=lambda x: x.get("period", ""), reverse=True)
        latest = recs[0]
        out = {
            "finnhub_buy_count": (latest.get("buy") or 0) + (latest.get("strongBuy") or 0),
            "finnhub_hold_count": latest.get("hold") or 0,
            "finnhub_sell_count": (latest.get("sell") or 0) + (latest.get("strongSell") or 0),
        }
        # 3-month delta in buy count (latest vs ~3 periods back, since Finnhub returns monthly)
        if len(recs) >= 4:
            three_ago = recs[3]
            three_ago_buy = (three_ago.get("buy") or 0) + (three_ago.get("strongBuy") or 0)
            out["finnhub_buy_count_change_3m"] = out["finnhub_buy_count"] - three_ago_buy
        out["_status"] = "ok"
        return out
    except Exception as e:
        return {"_status": f"finnhub_recom_error:{type(e).__name__}"}


def _fill_finnhub_news_sentiment(
    *, ticker: str,
) -> Dict[str, Any]:
    """News sentiment + buzz scores. Endpoint: /news-sentiment.
    Totally new signal — we had no pre-event sentiment data before this."""
    api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not api_key:
        return {"_status": "no_finnhub_key"}
    try:
        import requests
        r = requests.get(
            "https://finnhub.io/api/v1/news-sentiment",
            params={"symbol": ticker, "token": api_key},
            timeout=15,
        )
        if r.status_code != 200:
            return {"_status": f"news_status_{r.status_code}"}
        d = r.json() or {}
        sent = d.get("sentiment") or {}
        buzz = d.get("buzz") or {}
        out = {
            "finnhub_news_bullish_pct": (sent.get("bullishPercent") * 100) if sent.get("bullishPercent") is not None else None,
            "finnhub_news_bearish_pct": (sent.get("bearishPercent") * 100) if sent.get("bearishPercent") is not None else None,
            "finnhub_news_buzz_articles_week": buzz.get("articlesInLastWeek"),
            "finnhub_news_buzz_score": buzz.get("buzz"),
            "finnhub_company_news_score": d.get("companyNewsScore"),
        }
        out["_status"] = "ok" if any(v is not None for v in out.values()) else "no_news_sentiment_data"
        return out
    except Exception as e:
        return {"_status": f"finnhub_news_error:{type(e).__name__}"}


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

    # Finviz quote-page extensions: news + ratings + insider full table.
    # Single fetch shared across the three fillers (avoids triple-fetching
    # the same ~250KB page).
    finviz_html = _fetch_finviz_quote_html(ticker)
    fv_news = _fill_finviz_news(ticker=ticker, catalyst_date=catalyst_date, html=finviz_html)
    statuses["finviz_news"] = fv_news.pop("_status", "")
    feats.update({k: v for k, v in fv_news.items() if not k.startswith("_")})

    fv_ratings = _fill_finviz_ratings(ticker=ticker, catalyst_date=catalyst_date, html=finviz_html)
    statuses["finviz_ratings"] = fv_ratings.pop("_status", "")
    feats.update({k: v for k, v in fv_ratings.items() if not k.startswith("_")})

    fv_ins_full = _fill_finviz_insider_full(ticker=ticker, catalyst_date=catalyst_date, html=finviz_html)
    statuses["finviz_insider_full"] = fv_ins_full.pop("_status", "")
    feats.update({k: v for k, v in fv_ins_full.items() if not k.startswith("_")})

    # Finnhub — insider, price-target dispersion, recommendation trends, news sentiment
    fh_ins = _fill_finnhub_insider(ticker=ticker, catalyst_date=catalyst_date)
    statuses["finnhub_insider"] = fh_ins.pop("_status", "")
    feats.update({k: v for k, v in fh_ins.items() if not k.startswith("_")})

    fh_tgt = _fill_finnhub_price_target(ticker=ticker)
    statuses["finnhub_price_target"] = fh_tgt.pop("_status", "")
    feats.update({k: v for k, v in fh_tgt.items() if not k.startswith("_")})

    fh_rec = _fill_finnhub_recommendation(ticker=ticker, catalyst_date=catalyst_date)
    statuses["finnhub_recommendation"] = fh_rec.pop("_status", "")
    feats.update({k: v for k, v in fh_rec.items() if not k.startswith("_")})

    fh_news = _fill_finnhub_news_sentiment(ticker=ticker)
    statuses["finnhub_news"] = fh_news.pop("_status", "")
    feats.update({k: v for k, v in fh_news.items() if not k.startswith("_")})

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
    # COALESCE: don't clobber existing populated columns when a source
    # transiently fails (returns NULL). Without this, a flaky SEC EDGAR
    # call on refresh=true would erase previously-populated cash/debt.
    # Bookkeeping cols always overwrite (they should reflect latest run).
    BOOKKEEPING = {"backfill_source_status", "backfilled_at", "updated_at",
                   "backfill_version", "llm_enriched_at"}
    update_clause = ", ".join(
        (f"{c} = EXCLUDED.{c}" if c in BOOKKEEPING
         else f"{c} = COALESCE(EXCLUDED.{c}, catalyst_event_features.{c})")
        for c in cols if c != "catalyst_id"
    )
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


def refresh_dynamic_columns_batch(*, days_ahead: int = 90, limit: int = 200) -> Dict[str, Any]:
    """Refresh ONLY the rolling/dynamic columns (Finviz news, ratings,
    insider; Finnhub recommendation + news sentiment) for upcoming events
    where catalyst_date is within the next `days_ahead` days. These sources
    change continuously; past events stay frozen at their backfill snapshot.

    Single fetch per ticker (the shared Finviz quote-page HTML); per-ticker
    cost is the same as one full backfill but only touches ~9 columns,
    not the full row.
    """
    db = BiotechDatabase()
    today_iso = date.today().isoformat()
    horizon_iso = (date.today() + timedelta(days=days_ahead)).isoformat()

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cef.catalyst_id, cef.ticker, cef.catalyst_date::text
            FROM catalyst_event_features cef
            WHERE cef.catalyst_date >= %s
              AND cef.catalyst_date <= %s
            ORDER BY cef.catalyst_date ASC
            LIMIT %s
            """,
            (today_iso, horizon_iso, limit),
        )
        targets = cur.fetchall()

    if not targets:
        return {"checked": 0, "refreshed": 0, "_note": f"no upcoming events between {today_iso} and {horizon_iso}"}

    refreshed = errors = 0
    error_samples: List[Dict[str, Any]] = []
    for cid, ticker, cdate in targets:
        try:
            html = _fetch_finviz_quote_html(ticker)
            updates: Dict[str, Any] = {}
            updates.update({k: v for k, v in
                            _fill_finviz_news(ticker=ticker, catalyst_date=cdate, html=html).items()
                            if not k.startswith("_")})
            updates.update({k: v for k, v in
                            _fill_finviz_ratings(ticker=ticker, catalyst_date=cdate, html=html).items()
                            if not k.startswith("_")})
            updates.update({k: v for k, v in
                            _fill_finviz_insider_full(ticker=ticker, catalyst_date=cdate, html=html).items()
                            if not k.startswith("_")})
            updates.update({k: v for k, v in
                            _fill_finnhub_recommendation(ticker=ticker, catalyst_date=cdate).items()
                            if not k.startswith("_")})
            updates.update({k: v for k, v in
                            _fill_finnhub_news_sentiment(ticker=ticker).items()
                            if not k.startswith("_")})

            if not updates:
                continue
            # COALESCE-style update: NULLs from a failed source don't clobber
            set_clauses = ", ".join(
                f"{c} = COALESCE(%s, {c})" for c in updates.keys()
            )
            sql = (f"UPDATE catalyst_event_features SET {set_clauses}, "
                   f"dynamic_refresh_at = NOW(), updated_at = NOW() "
                   f"WHERE catalyst_id = %s")
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, list(updates.values()) + [cid])
                conn.commit()
            refreshed += 1
        except Exception as e:
            errors += 1
            if len(error_samples) < 5:
                error_samples.append({"catalyst_id": cid, "error": f"{type(e).__name__}: {str(e)[:100]}"})

    return {
        "checked": len(targets),
        "refreshed": refreshed,
        "errors": errors,
        "error_samples": error_samples,
        "horizon_days": days_ahead,
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
