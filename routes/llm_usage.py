"""
/admin/llm/* routes — token usage dashboard + budget management.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.llm_usage import (
    get_usage_summary, get_recent_usage, get_today_spend, get_month_spend,
    get_budgets, set_budget, delete_budget, check_budget,
    PRICING_USD,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---- Usage analytics ----

@router.get("/admin/llm/usage/summary")
async def usage_summary(days: int = Query(7, ge=1, le=365),
                          group_by: str = Query("day", regex="^(day|provider|feature|day_provider)$")):
    """Aggregated usage stats. group_by: day | provider | feature | day_provider."""
    return get_usage_summary(days=days, group_by=group_by)


@router.get("/admin/llm/usage/recent")
async def usage_recent(limit: int = Query(50, ge=1, le=500),
                        provider: Optional[str] = None,
                        feature: Optional[str] = None,
                        ticker: Optional[str] = None):
    """Recent calls table. Filterable by provider, feature, or ticker."""
    rows = get_recent_usage(limit=limit, provider=provider, feature=feature, ticker=ticker)
    return {"count": len(rows), "rows": rows}


@router.get("/admin/llm/usage/headline")
async def usage_headline():
    """Dashboard headline numbers: today/month spend, both global + per-provider."""
    return {
        "today_global_usd": round(get_today_spend(), 4),
        "month_global_usd": round(get_month_spend(), 2),
        "by_provider_today": {
            p: round(get_today_spend(provider=p), 4)
            for p in ["anthropic", "openai", "google"]
        },
        "by_provider_month": {
            p: round(get_month_spend(provider=p), 4)
            for p in ["anthropic", "openai", "google"]
        },
    }


# ---- Budget management ----

@router.get("/admin/llm/budgets")
async def list_budgets():
    """All configured budgets."""
    return {"budgets": get_budgets()}


@router.get("/admin/llm/budgets/check")
async def budget_check(provider: Optional[str] = None, feature: Optional[str] = None):
    """Pre-call check what would happen for a (provider, feature) call right now.
    Returns allowed/reason/warnings + applicable budget rows."""
    return check_budget(provider=provider, feature=feature)


class BudgetSetRequest(BaseModel):
    scope_type: str           # 'global' | 'provider' | 'feature' | 'provider_feature'
    scope_value: str          # 'global' | 'anthropic' | 'npv_v2' | 'anthropic:npv_v2'
    daily_limit_usd: Optional[float] = None
    monthly_limit_usd: Optional[float] = None
    hard_cutoff: Optional[bool] = None
    alert_at_pct: Optional[float] = None
    enabled: Optional[bool] = None
    notes: Optional[str] = None


@router.post("/admin/llm/budgets")
async def upsert_budget(req: BudgetSetRequest):
    """Create or update a budget. UPSERT by (scope_type, scope_value)."""
    result = set_budget(
        scope_type=req.scope_type, scope_value=req.scope_value,
        daily_limit_usd=req.daily_limit_usd,
        monthly_limit_usd=req.monthly_limit_usd,
        hard_cutoff=req.hard_cutoff,
        alert_at_pct=req.alert_at_pct,
        enabled=req.enabled, notes=req.notes,
    )
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


@router.delete("/admin/llm/budgets/{budget_id}")
async def delete_budget_route(budget_id: int):
    """Delete budget. Cannot delete global default."""
    result = delete_budget(budget_id)
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return result


# ---- Pricing reference ----

@router.get("/admin/llm/pricing")
async def show_pricing():
    """Show the pricing table used for cost computation. Useful for sanity checks
    when costs look wrong."""
    return {
        "rates_usd_per_1k_tokens": [
            {"provider": p, "model": m, "input_per_1k": rates[0], "output_per_1k": rates[1]}
            for (p, m), rates in sorted(PRICING_USD.items())
        ],
        "fallback_rate_per_1k": {"input": 5.0e-3, "output": 25.0e-3},
        "notes": "Update services/llm_usage.py PRICING_USD when providers change rates.",
    }
