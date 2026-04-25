"""
llm_usage — central LLM cost accounting + budget enforcement.

Public API:
  - record_usage(...)          : log a single call after it completes
  - check_budget(provider, feature) -> dict : pre-call check, optional hard cutoff
  - compute_cost_usd(...)      : pricing table lookup
  - get_usage_summary(...)     : aggregated stats for the dashboard
  - get_recent_usage(...)      : last N rows for the calls table
  - get_budgets() / set_budget(...) / delete_budget(...)

Pricing table is intentionally kept simple — flat per-1k-token rates per
(provider, model). Update PRICING_USD as needed.
"""
import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# Pricing table — update as providers change rates
# ============================================================
# Rates are USD per 1K tokens (input, output)
PRICING_USD = {
    # Anthropic — https://www.anthropic.com/api (Claude 4 family)
    ("anthropic", "claude-sonnet-4-5"):     (3.0e-3,  15.0e-3),  # Sonnet 4.5
    ("anthropic", "claude-sonnet-4"):       (3.0e-3,  15.0e-3),
    ("anthropic", "claude-opus-4-7"):       (15.0e-3, 75.0e-3),
    ("anthropic", "claude-opus-4"):         (15.0e-3, 75.0e-3),
    ("anthropic", "claude-haiku-4-5"):      (1.0e-3,  5.0e-3),

    # OpenAI — https://openai.com/api/pricing
    ("openai", "gpt-4o"):                   (2.5e-3,  10.0e-3),
    ("openai", "gpt-4o-mini"):              (0.15e-3, 0.60e-3),
    ("openai", "gpt-4.5-preview"):          (75.0e-3, 150.0e-3),  # legacy preview model

    # Google — https://ai.google.dev/pricing
    ("google", "gemini-2.5-flash"):         (0.30e-3, 2.50e-3),
    ("google", "gemini-2.5-pro"):           (1.25e-3, 10.0e-3),
    ("google", "gemini-flash"):             (0.30e-3, 2.50e-3),
}


def compute_cost_usd(provider: str, model: str, tokens_input: int, tokens_output: int) -> float:
    """Lookup pricing for (provider, model) and compute cost. Falls back to a
    conservative high estimate if model not in table (so unknown models get
    surfaced as expensive)."""
    if not provider:
        return 0.0
    p = provider.lower()
    m = (model or "").lower()
    rates = PRICING_USD.get((p, m))
    if rates is None:
        # Try fuzzy match — strip versioning
        for (pp, mm), rr in PRICING_USD.items():
            if pp == p and mm in m:
                rates = rr
                break
    if rates is None:
        # Conservative fallback
        rates = (5.0e-3, 25.0e-3)
        logger.debug(f"No pricing for ({provider}, {model}) — using fallback")
    in_per_1k, out_per_1k = rates
    cost = (tokens_input or 0) / 1000.0 * in_per_1k + (tokens_output or 0) / 1000.0 * out_per_1k
    return round(cost, 6)


# ============================================================
# DB helpers
# ============================================================

def _db():
    from services.database import BiotechDatabase
    return BiotechDatabase()


def record_usage(
    provider: str,
    model: Optional[str] = None,
    feature: Optional[str] = None,
    ticker: Optional[str] = None,
    tokens_input: int = 0,
    tokens_output: int = 0,
    duration_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    request_id: Optional[str] = None,
    cost_usd_override: Optional[float] = None,
) -> bool:
    """Record one LLM call. Returns True on success, False on failure (always
    swallowed — accounting failures should not break the calling code path)."""
    try:
        cost = cost_usd_override if cost_usd_override is not None else \
               compute_cost_usd(provider, model or "", tokens_input, tokens_output)
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO llm_usage
                  (provider, model, feature, ticker, tokens_input, tokens_output,
                   cost_usd, duration_ms, status, error_message, request_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                provider, model, feature, ticker,
                int(tokens_input or 0), int(tokens_output or 0),
                cost, duration_ms, status,
                (error_message or "")[:500] if error_message else None,
                request_id,
            ))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"record_usage failed (non-fatal): {e}")
        return False


def get_today_spend(provider: Optional[str] = None, feature: Optional[str] = None) -> float:
    """Sum of cost_usd today (UTC). Optional filter by provider/feature."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage WHERE day = (NOW() AT TIME ZONE 'UTC')::date"
            params: List = []
            if provider:
                sql += " AND provider = %s"; params.append(provider)
            if feature:
                sql += " AND feature = %s"; params.append(feature)
            cur.execute(sql, params)
            row = cur.fetchone()
            return float(row[0] or 0)
    except Exception as e:
        logger.warning(f"get_today_spend failed: {e}")
        return 0.0


def get_month_spend(provider: Optional[str] = None, feature: Optional[str] = None) -> float:
    """Sum cost_usd for current calendar month (UTC)."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            sql = """SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage
                     WHERE day >= date_trunc('month', NOW() AT TIME ZONE 'UTC')::date"""
            params: List = []
            if provider:
                sql += " AND provider = %s"; params.append(provider)
            if feature:
                sql += " AND feature = %s"; params.append(feature)
            cur.execute(sql, params)
            row = cur.fetchone()
            return float(row[0] or 0)
    except Exception as e:
        logger.warning(f"get_month_spend failed: {e}")
        return 0.0


# ============================================================
# Budget enforcement
# ============================================================

def get_applicable_budgets(provider: Optional[str], feature: Optional[str]) -> List[Dict]:
    """Return all budget rows that apply to this (provider, feature) combo,
    ordered most-specific to least-specific."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            scopes = [("global", "global")]
            if provider:
                scopes.append(("provider", provider))
            if feature:
                scopes.append(("feature", feature))
            if provider and feature:
                scopes.append(("provider_feature", f"{provider}:{feature}"))
            
            results = []
            for stype, sval in scopes:
                cur.execute("""
                    SELECT id, scope_type, scope_value, daily_limit_usd, monthly_limit_usd,
                           hard_cutoff, alert_at_pct, enabled, notes, updated_at
                    FROM llm_budgets
                    WHERE scope_type = %s AND scope_value = %s AND enabled = TRUE
                """, (stype, sval))
                row = cur.fetchone()
                if row:
                    results.append({
                        "id": row[0], "scope_type": row[1], "scope_value": row[2],
                        "daily_limit_usd": float(row[3]) if row[3] is not None else None,
                        "monthly_limit_usd": float(row[4]) if row[4] is not None else None,
                        "hard_cutoff": bool(row[5]), "alert_at_pct": float(row[6] or 80),
                        "enabled": bool(row[7]), "notes": row[8],
                        "updated_at": row[9].isoformat() if row[9] else None,
                    })
            # Order: most specific first
            order = {"provider_feature": 0, "feature": 1, "provider": 2, "global": 3}
            results.sort(key=lambda b: order.get(b["scope_type"], 99))
            return results
    except Exception as e:
        logger.warning(f"get_applicable_budgets failed: {e}")
        return []


def check_budget(provider: Optional[str] = None, feature: Optional[str] = None) -> Dict:
    """Pre-call budget check. Returns:
      {
        allowed: bool,             # False if hard_cutoff hit
        reason: 'ok' | 'daily_exceeded' | 'monthly_exceeded' | 'no_budget',
        warnings: [..],            # if approaching limit
        budgets_evaluated: [...],
        today_spend_usd: float,
        month_spend_usd: float,
      }
    """
    today_global = get_today_spend()
    month_global = get_month_spend()
    today_scoped = get_today_spend(provider=provider, feature=feature) if (provider or feature) else today_global
    month_scoped = get_month_spend(provider=provider, feature=feature) if (provider or feature) else month_global

    budgets = get_applicable_budgets(provider, feature)
    warnings = []
    blocked_reason = None

    for b in budgets:
        # Pick the right spend number based on scope
        if b["scope_type"] == "global":
            today_s, month_s = today_global, month_global
        else:
            today_s, month_s = today_scoped, month_scoped

        daily_limit = b.get("daily_limit_usd")
        monthly_limit = b.get("monthly_limit_usd")

        if daily_limit and today_s >= daily_limit:
            warnings.append({
                "scope": f"{b['scope_type']}:{b['scope_value']}",
                "kind": "daily_exceeded",
                "spent_usd": round(today_s, 4),
                "limit_usd": daily_limit,
                "hard_cutoff": b["hard_cutoff"],
            })
            if b["hard_cutoff"]:
                blocked_reason = "daily_exceeded"
                break
        elif daily_limit and today_s >= daily_limit * (b["alert_at_pct"] / 100.0):
            warnings.append({
                "scope": f"{b['scope_type']}:{b['scope_value']}",
                "kind": "daily_alert_threshold",
                "spent_usd": round(today_s, 4),
                "limit_usd": daily_limit,
                "pct_used": round(today_s / daily_limit * 100, 1),
            })

        if monthly_limit and month_s >= monthly_limit:
            warnings.append({
                "scope": f"{b['scope_type']}:{b['scope_value']}",
                "kind": "monthly_exceeded",
                "spent_usd": round(month_s, 2),
                "limit_usd": monthly_limit,
                "hard_cutoff": b["hard_cutoff"],
            })
            if b["hard_cutoff"]:
                blocked_reason = "monthly_exceeded"
                break

    return {
        "allowed": blocked_reason is None,
        "reason": blocked_reason or "ok",
        "warnings": warnings,
        "budgets_evaluated": [{"scope_type": b["scope_type"], "scope_value": b["scope_value"],
                                "daily": b.get("daily_limit_usd"), "monthly": b.get("monthly_limit_usd"),
                                "hard": b["hard_cutoff"]} for b in budgets],
        "today_spend_usd": round(today_scoped, 4),
        "month_spend_usd": round(month_scoped, 2),
        "today_spend_global_usd": round(today_global, 4),
        "month_spend_global_usd": round(month_global, 2),
    }


# ============================================================
# Reporting
# ============================================================

def get_usage_summary(days: int = 7, group_by: str = "day") -> Dict:
    """Aggregate usage stats. group_by: 'day' | 'provider' | 'feature' | 'day_provider'."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            since = (date.today() - timedelta(days=days)).isoformat()

            if group_by == "day":
                cur.execute("""
                    SELECT day, COUNT(*), COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),
                           COUNT(*) FILTER (WHERE status != 'success')
                    FROM llm_usage WHERE day >= %s
                    GROUP BY day ORDER BY day DESC
                """, (since,))
                rows = [{"day": str(r[0]), "calls": r[1], "cost_usd": float(r[2]),
                         "tokens_in": int(r[3]), "tokens_out": int(r[4]), "errors": r[5]}
                        for r in cur.fetchall()]
            elif group_by == "provider":
                cur.execute("""
                    SELECT provider, COUNT(*), COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),
                           COUNT(*) FILTER (WHERE status != 'success')
                    FROM llm_usage WHERE day >= %s
                    GROUP BY provider ORDER BY SUM(cost_usd) DESC NULLS LAST
                """, (since,))
                rows = [{"provider": r[0], "calls": r[1], "cost_usd": float(r[2]),
                         "tokens_in": int(r[3]), "tokens_out": int(r[4]), "errors": r[5]}
                        for r in cur.fetchall()]
            elif group_by == "feature":
                cur.execute("""
                    SELECT feature, COUNT(*), COALESCE(SUM(cost_usd), 0),
                           COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),
                           COUNT(*) FILTER (WHERE status != 'success')
                    FROM llm_usage WHERE day >= %s
                    GROUP BY feature ORDER BY SUM(cost_usd) DESC NULLS LAST
                """, (since,))
                rows = [{"feature": r[0] or "(unspecified)", "calls": r[1], "cost_usd": float(r[2]),
                         "tokens_in": int(r[3]), "tokens_out": int(r[4]), "errors": r[5]}
                        for r in cur.fetchall()]
            elif group_by == "day_provider":
                cur.execute("""
                    SELECT day, provider, COUNT(*), COALESCE(SUM(cost_usd), 0)
                    FROM llm_usage WHERE day >= %s
                    GROUP BY day, provider ORDER BY day DESC, provider
                """, (since,))
                rows = [{"day": str(r[0]), "provider": r[1], "calls": r[2], "cost_usd": float(r[3])}
                        for r in cur.fetchall()]
            else:
                rows = []

            # Headline totals
            cur.execute("""
                SELECT COUNT(*), COALESCE(SUM(cost_usd), 0),
                       COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0)
                FROM llm_usage WHERE day >= %s
            """, (since,))
            total_row = cur.fetchone()

            return {
                "since": since, "days": days, "group_by": group_by,
                "totals": {
                    "calls": int(total_row[0] or 0),
                    "cost_usd": float(total_row[1] or 0),
                    "tokens_in": int(total_row[2] or 0),
                    "tokens_out": int(total_row[3] or 0),
                },
                "today_spend_usd": get_today_spend(),
                "month_spend_usd": get_month_spend(),
                "rows": rows,
            }
    except Exception as e:
        logger.warning(f"get_usage_summary failed: {e}")
        return {"error": str(e)[:200], "rows": [], "totals": {}, "since": "?", "days": days, "group_by": group_by}


def get_recent_usage(limit: int = 50, provider: Optional[str] = None,
                      feature: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
    """Last N usage rows for the calls table."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            sql = """SELECT id, ts, provider, model, feature, ticker,
                            tokens_input, tokens_output, cost_usd, duration_ms,
                            status, error_message, request_id
                     FROM llm_usage WHERE 1=1"""
            params: List = []
            if provider: sql += " AND provider = %s"; params.append(provider)
            if feature:  sql += " AND feature = %s";  params.append(feature)
            if ticker:   sql += " AND ticker = %s";   params.append(ticker.upper())
            sql += " ORDER BY ts DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                # Convert types for JSON
                d["ts"] = d["ts"].isoformat() if d.get("ts") else None
                d["cost_usd"] = float(d["cost_usd"]) if d.get("cost_usd") is not None else None
                rows.append(d)
            return rows
    except Exception as e:
        logger.warning(f"get_recent_usage failed: {e}")
        return []


# ============================================================
# Budget management
# ============================================================

def get_budgets() -> List[Dict]:
    """List all budget rows."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT id, scope_type, scope_value, daily_limit_usd, monthly_limit_usd,
                                  hard_cutoff, alert_at_pct, enabled, notes, updated_at
                           FROM llm_budgets ORDER BY scope_type, scope_value""")
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                for k in ("daily_limit_usd", "monthly_limit_usd", "alert_at_pct"):
                    if d.get(k) is not None: d[k] = float(d[k])
                d["hard_cutoff"] = bool(d.get("hard_cutoff"))
                d["enabled"] = bool(d.get("enabled"))
                d["updated_at"] = d["updated_at"].isoformat() if d.get("updated_at") else None
                rows.append(d)
            return rows
    except Exception as e:
        logger.warning(f"get_budgets failed: {e}")
        return []


def set_budget(scope_type: str, scope_value: str,
                daily_limit_usd: Optional[float] = None,
                monthly_limit_usd: Optional[float] = None,
                hard_cutoff: Optional[bool] = None,
                alert_at_pct: Optional[float] = None,
                enabled: Optional[bool] = None,
                notes: Optional[str] = None) -> Dict:
    """UPSERT a budget row by (scope_type, scope_value)."""
    if scope_type not in ("global", "provider", "feature", "provider_feature"):
        return {"error": f"invalid scope_type: {scope_type}"}
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO llm_budgets
                  (scope_type, scope_value, daily_limit_usd, monthly_limit_usd,
                   hard_cutoff, alert_at_pct, enabled, notes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (scope_type, scope_value) DO UPDATE SET
                   daily_limit_usd = COALESCE(EXCLUDED.daily_limit_usd, llm_budgets.daily_limit_usd),
                   monthly_limit_usd = COALESCE(EXCLUDED.monthly_limit_usd, llm_budgets.monthly_limit_usd),
                   hard_cutoff = COALESCE(EXCLUDED.hard_cutoff, llm_budgets.hard_cutoff),
                   alert_at_pct = COALESCE(EXCLUDED.alert_at_pct, llm_budgets.alert_at_pct),
                   enabled = COALESCE(EXCLUDED.enabled, llm_budgets.enabled),
                   notes = COALESCE(EXCLUDED.notes, llm_budgets.notes),
                   updated_at = NOW()
                RETURNING id, scope_type, scope_value
            """, (scope_type, scope_value, daily_limit_usd, monthly_limit_usd,
                  hard_cutoff if hard_cutoff is not None else False,
                  alert_at_pct if alert_at_pct is not None else 80.0,
                  enabled if enabled is not None else True,
                  notes))
            row = cur.fetchone()
            conn.commit()
        return {"id": row[0], "scope_type": row[1], "scope_value": row[2], "ok": True}
    except Exception as e:
        logger.warning(f"set_budget failed: {e}")
        return {"error": str(e)[:200]}


def delete_budget(budget_id: int) -> Dict:
    """Delete budget by id. Refuses to delete global default (id with scope_type=global, scope_value=global)."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT scope_type, scope_value FROM llm_budgets WHERE id=%s", (budget_id,))
            row = cur.fetchone()
            if not row:
                return {"error": "budget not found"}
            if row[0] == "global" and row[1] == "global":
                return {"error": "cannot delete global default — use enabled=false to disable"}
            cur.execute("DELETE FROM llm_budgets WHERE id=%s", (budget_id,))
            conn.commit()
        return {"ok": True, "deleted_id": budget_id}
    except Exception as e:
        logger.warning(f"delete_budget failed: {e}")
        return {"error": str(e)[:200]}
