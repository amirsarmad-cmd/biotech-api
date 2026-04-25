"""
Catalyst-specific risk factors endpoint (Feature #5).

Drug + indication aware risk factor computation, cached in stock_risk_factors.
"""
import logging
import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor, Json

logger = logging.getLogger(__name__)
router = APIRouter()


def _pg_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(db_url)


@router.get("/v2/catalysts/{catalyst_id}/risk-factors")
async def get_catalyst_risk_factors(catalyst_id: int, refresh: bool = False):
    """Drug-specific risk factors for a catalyst. Cached 24h in stock_risk_factors."""
    try:
        with _pg_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Look up the catalyst
                cur.execute("""
                    SELECT id, ticker, company_name, catalyst_type, catalyst_date,
                           drug_name, indication, phase, description
                    FROM catalyst_universe 
                    WHERE id = %s AND status = 'active'
                """, (catalyst_id,))
                cat = cur.fetchone()
                if not cat:
                    raise HTTPException(404, f"Catalyst {catalyst_id} not found")
                
                # 2. Check cache (computed_at column, freshness 24h)
                if not refresh:
                    cur.execute("""
                        SELECT * FROM stock_risk_factors
                        WHERE ticker = %s 
                          AND (catalyst_id = %s OR catalyst_id IS NULL)
                          AND (drug_name = %s OR drug_name IS NULL OR %s IS NULL)
                        ORDER BY 
                          CASE WHEN catalyst_id = %s THEN 0 ELSE 1 END,
                          CASE WHEN drug_name = %s THEN 0 ELSE 1 END,
                          computed_at DESC
                        LIMIT 1
                    """, (cat["ticker"], catalyst_id, cat["drug_name"], cat["drug_name"],
                          catalyst_id, cat["drug_name"]))
                    cached = cur.fetchone()
                    if cached and cached.get("computed_at"):
                        age = datetime.now(cached["computed_at"].tzinfo) - cached["computed_at"]
                        if age < timedelta(hours=24):
                            factors = cached.get("factors") or {}
                            return {
                                "cached": True,
                                "age_hours": round(age.total_seconds() / 3600, 1),
                                "catalyst_id": catalyst_id,
                                "ticker": cat["ticker"],
                                "drug_name": cat.get("drug_name"),
                                "indication": cat.get("indication"),
                                "factors": factors,
                                "total_discount": float(factors.get("total_discount", 0)) if isinstance(factors, dict) else 0,
                                "computed_at": cached["computed_at"].isoformat(),
                            }
                
                # 3. Compute fresh
                from services.risk_factors import estimate_risk_factors
                try:
                    import yfinance as yf
                    info = yf.Ticker(cat["ticker"]).info or {}
                except Exception:
                    info = {}
                
                # Build drug-specific context
                drug_ctx_parts = []
                if cat.get("drug_name"):
                    drug_ctx_parts.append(f"Drug: {cat['drug_name']}")
                if cat.get("indication"):
                    drug_ctx_parts.append(f"Indication: {cat['indication']}")
                if cat.get("phase"):
                    drug_ctx_parts.append(f"Phase: {cat['phase']}")
                if cat.get("catalyst_type"):
                    drug_ctx_parts.append(f"Upcoming: {cat['catalyst_type']} on {cat['catalyst_date']}")
                if cat.get("description"):
                    drug_ctx_parts.append(f"Context: {cat['description'][:300]}")
                ai_context = "\n".join(drug_ctx_parts)
                
                result = estimate_risk_factors(
                    ticker=cat["ticker"],
                    company_name=cat.get("company_name") or cat["ticker"],
                    info=info,
                    news_excerpt="",
                    ai_context=ai_context,
                )
                
                # 4. Cache it
                cur.execute("""
                    INSERT INTO stock_risk_factors 
                        (ticker, catalyst_id, drug_name, factors, prior_crls, active_litigation,
                         insider_transactions, short_data, computed_at, llm_provider)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (ticker, catalyst_id, drug_name) DO UPDATE SET
                        factors = EXCLUDED.factors,
                        prior_crls = EXCLUDED.prior_crls,
                        active_litigation = EXCLUDED.active_litigation,
                        insider_transactions = EXCLUDED.insider_transactions,
                        short_data = EXCLUDED.short_data,
                        computed_at = NOW(),
                        llm_provider = EXCLUDED.llm_provider
                """, (
                    cat["ticker"], catalyst_id, cat.get("drug_name"),
                    Json(result),  # full result as factors json
                    Json([]),
                    Json([]),
                    Json({"insiderHeldPct": (info.get("heldPercentInsiders", 0) or 0) * 100}),
                    Json({"shortPctFloat": (info.get("shortPercentOfFloat", 0) or 0) * 100}),
                    "claude",
                ))
                conn.commit()
                
                return {
                    "cached": False,
                    "computed_at": datetime.utcnow().isoformat() + "Z",
                    "catalyst_id": catalyst_id,
                    "ticker": cat["ticker"],
                    "drug_name": cat.get("drug_name"),
                    "indication": cat.get("indication"),
                    "factors": result,
                    "total_discount": float(result.get("total_discount", 0)),
                    "context_used": ai_context,
                }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"risk-factors {catalyst_id}")
        raise HTTPException(500, f"risk-factors error: {e}")
