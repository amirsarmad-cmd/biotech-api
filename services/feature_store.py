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
    """Runup, realized vol, max drawdown — from yfinance history we already
    cache in `price_history_daily` (or compute on demand).
    """
    out: Dict[str, Any] = {}
    status = "ok"
    try:
        # Try the cached price_history_daily table first; fall back to yfinance fetch.
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT date, close
                FROM price_history_daily
                WHERE ticker = %s
                  AND date <= %s::date
                  AND date >= (%s::date - INTERVAL '200 days')
                ORDER BY date ASC
                """,
                (ticker, catalyst_date, catalyst_date),
            )
            rows = cur.fetchall()
        if not rows or len(rows) < 30:
            # Try yfinance live fetch as fallback (slower but works for any ticker).
            try:
                import yfinance as yf
                from datetime import datetime
                cd = datetime.fromisoformat(catalyst_date[:10])
                start = cd - timedelta(days=210)
                hist = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                                  end=cd.strftime("%Y-%m-%d"))
                rows = [(idx.date(), float(close))
                        for idx, close in zip(hist.index, hist["Close"])
                        if not math.isnan(close)]
            except Exception as e:
                return {"_status": f"yfinance_fallback_failed:{type(e).__name__}"}
        if not rows or len(rows) < 30:
            return {"_status": "insufficient_history"}

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


def _fill_polygon_options(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Historical options chain at the day before catalyst — Polygon's
    options aggregates endpoint. Polygon API key is in Railway env as
    POLYGON_API_KEY. Polygon Stocks Starter ($30/mo) gives 2y historical
    options aggs; Options Starter ($100/mo) gives 5y including chain snapshots.

    This fill is best-effort — if the Polygon plan doesn't include the
    needed endpoint or the chain isn't available for the date, returns
    NULL columns + a status string. Intentionally kept short — the priced-in
    calculator can be retrofitted later to read from these columns.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return {"_status": "no_polygon_key"}
    out: Dict[str, Any] = {}
    try:
        import requests
        # Day before catalyst (best proxy for "into the event")
        cd = datetime.fromisoformat(catalyst_date[:10])
        target = (cd - timedelta(days=1)).strftime("%Y-%m-%d")

        # Step 1: get the underlying close price at target (cheap, free tier)
        # Step 2: pull the options chain snapshot at target — Polygon endpoint:
        #   /v3/snapshot/options/{underlying}?as_of={date}
        # NOTE: snapshot 'as_of' is a paid feature on Options Starter+. If the
        # account is on the free tier, this returns a 403 and we fall back
        # to setting `options_source=null`.
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {"limit": 250, "as_of": target, "apiKey": api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 403:
            return {"_status": "polygon_plan_does_not_include_historical_chain"}
        if r.status_code != 200:
            return {"_status": f"polygon_status_{r.status_code}"}
        data = r.json()
        results = data.get("results", []) or []
        if not results:
            return {"_status": "polygon_empty_chain"}

        # Compute aggregates from the chain
        call_oi = 0
        put_oi = 0
        call_volume = 0
        put_volume = 0
        atm_calls: List[Tuple[float, float]] = []   # (strike, iv)
        atm_puts: List[Tuple[float, float]] = []
        underlying_price = None
        for opt in results:
            details = opt.get("details") or {}
            ctype = (details.get("contract_type") or "").lower()
            strike = details.get("strike_price")
            ua = opt.get("underlying_asset") or {}
            if underlying_price is None:
                underlying_price = ua.get("price")
            day = opt.get("day") or {}
            oi = day.get("open_interest") or 0
            vol = day.get("volume") or 0
            iv = (opt.get("implied_volatility") or 0) * 100  # to %
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

        # ATM IV: closest-strike IV per side, then average
        if underlying_price and atm_calls and atm_puts:
            atm_calls.sort(key=lambda x: abs(x[0] - underlying_price))
            atm_puts.sort(key=lambda x: abs(x[0] - underlying_price))
            atm_call_iv = atm_calls[0][1]
            atm_put_iv = atm_puts[0][1]
            out["atm_iv_at_date"] = (atm_call_iv + atm_put_iv) / 2

            # 25-delta skew: approximate with strikes ±10% from spot
            target_low = underlying_price * 0.90
            target_high = underlying_price * 1.10
            puts_25d = [iv for s, iv in atm_puts if s <= target_low]
            calls_25d = [iv for s, iv in atm_calls if s >= target_high]
            if puts_25d and calls_25d:
                out["iv_skew_25d"] = sum(calls_25d) / len(calls_25d) - sum(puts_25d) / len(puts_25d)

            # Implied move from straddle proxy
            out["options_implied_move_pct"] = (atm_call_iv + atm_put_iv) / 2 * math.sqrt(7 / 252)

        out["options_source"] = "polygon"
        out["_status"] = "ok"
    except Exception as e:
        return {"_status": f"polygon_error:{type(e).__name__}:{str(e)[:60]}"}
    return out


def _fill_capital_structure(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Cash, debt, runway from SEC EDGAR via existing services/sec_financials.
    Picks the most recent 10-Q/K filed BEFORE catalyst_date (point-in-time)
    rather than the always-current snapshot.
    """
    try:
        from services import sec_financials  # type: ignore[attr-defined]
    except ImportError:
        return {"_status": "sec_financials_module_unavailable"}
    try:
        # The existing module exposes get_capital_structure(ticker) which
        # returns the latest filing — adapt to point-in-time by filtering
        # filings <= catalyst_date.
        if hasattr(sec_financials, "get_capital_structure_at"):
            cs = sec_financials.get_capital_structure_at(ticker, catalyst_date)
        elif hasattr(sec_financials, "get_capital_structure"):
            cs = sec_financials.get_capital_structure(ticker)
        else:
            return {"_status": "no_capital_structure_function"}
        if not cs:
            return {"_status": "no_filing_found"}
        out = {
            "cash_at_date_m": (cs.get("total_cash") or 0) / 1e6 if cs.get("total_cash") else None,
            "debt_at_date_m": (cs.get("total_debt") or 0) / 1e6 if cs.get("total_debt") else None,
            "runway_months_at_date": cs.get("cash_runway_months"),
            "sec_filing_date_used": cs.get("as_of_filing"),
            "_status": "ok",
        }
        if out["cash_at_date_m"] is not None and out["debt_at_date_m"] is not None:
            out["net_cash_at_date_m"] = out["cash_at_date_m"] - out["debt_at_date_m"]
        return out
    except Exception as e:
        return {"_status": f"sec_error:{type(e).__name__}"}


def _fill_microstructure(
    *, ticker: str, db: BiotechDatabase,
) -> Dict[str, Any]:
    """Short interest + avg volume from screener_stocks (current snapshot —
    point-in-time short interest deferred to a later iteration that joins
    historical FINRA data).
    """
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT short_pct_of_float, avg_volume_3m, market_cap, shares_outstanding
                FROM screener_stocks
                WHERE ticker = %s
                """,
                (ticker,),
            )
            row = cur.fetchone()
        if not row:
            return {"_status": "ticker_not_in_screener"}
        short_pct, avg_vol_3m, mcap, shares_out = row
        out = {
            "short_interest_pct_at_date": float(short_pct) if short_pct is not None else None,
            "avg_volume_30d": float(avg_vol_3m) if avg_vol_3m is not None else None,
            "market_cap_at_date_m": float(mcap) if mcap is not None else None,
            "shares_out_at_date_m": (float(shares_out) / 1e6) if shares_out else None,
            "_status": "ok_current_snapshot",  # signals lookback bias
        }
        # Short ratio = short_int_shares / avg_daily_volume; need short_int in shares
        if short_pct is not None and shares_out and avg_vol_3m:
            short_shares = float(short_pct) / 100 * float(shares_out)
            out["short_ratio_days_at_date"] = short_shares / float(avg_vol_3m) if avg_vol_3m else None
        return out
    except Exception as e:
        return {"_status": f"error:{type(e).__name__}"}


def _fill_sec_insider(
    *, ticker: str, catalyst_date: str,
) -> Dict[str, Any]:
    """Form 4 insider buys/sells in the 30 days BEFORE the catalyst.
    Stub — full implementation calls SEC EDGAR `/cgi-bin/browse-edgar` filtered
    to ownership filings + parses the XML. Leaving structured shell so a
    future chat can implement without touching the orchestrator.
    """
    return {
        "_status": "deferred_to_section_chat",
        "_note": "Form 4 parser implementation deferred. Schema columns reserved.",
    }


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
            SELECT id, ticker, catalyst_date::text, catalyst_type
            FROM catalyst_universe WHERE id = %s
            """,
            (catalyst_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"error": f"catalyst_id {catalyst_id} not found"}
    cid, ticker, catalyst_date, catalyst_type = row

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

    po = _fill_polygon_options(ticker=ticker, catalyst_date=catalyst_date)
    statuses["polygon_options"] = po.pop("_status", "")
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
              ROUND(100.0 * COUNT(atm_iv_at_date)               / NULLIF(COUNT(*), 0), 1) AS pct_polygon_options,
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
