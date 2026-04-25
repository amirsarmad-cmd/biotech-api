"""
Catalyst-specific risk factors endpoint (Feature #5).

Extends the stock-level risk_factors with drug+indication context for sharper estimates:
- Drug name, MOA, indication
- Prior CRLs / AdComm history specifically for THIS drug
- Recent news around this drug program (not just company-wide)

Cached in stock_risk_factors table (UNIQUE on ticker, catalyst_id, drug_name).
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
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
    """Get cached + computed risk factors for a specific catalyst (drug + indication context).
    
    Returns the cached entry from stock_risk_factors if recent, otherwise computes fresh.
    """
    try:
        with _pg_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Look up the catalyst from catalyst_universe
                cur.execute("""
                    SELECT id, ticker, company_name, catalyst_type, catalyst_date,
                           drug_name, indication, phase, description
                    FROM catalyst_universe 
                    WHERE id = %s AND status = 'active'
                """, (catalyst_id,))
                cat = cur.fetchone()
                if not cat:
                    raise HTTPException(404, f"Catalyst {catalyst_id} not found or inactive")
                
                # 2. Check cache
                if not refresh:
                    cur.execute("""
                        SELECT * FROM stock_risk_factors
                        WHERE ticker = %s 
                          AND (catalyst_id = %s OR catalyst_id IS NULL)
                          AND (drug_name = %s OR drug_name IS NULL)
                        ORDER BY 
                          CASE WHEN catalyst_id = %s THEN 0 ELSE 1 END,
                          CASE WHEN drug_name = %s THEN 0 ELSE 1 END,
                          last_updated DESC
                        LIMIT 1
                    """, (cat["ticker"], catalyst_id, cat["drug_name"], catalyst_id, cat["drug_name"]))
                    cached = cur.fetchone()
                    if cached:
                        # Check freshness — stale after 24h or if news_hash changed
                        from datetime import datetime as _dt, timedelta as _td
                        age = _dt.now(cached["last_updated"].tzinfo) - cached["last_updated"]
                        if age < _td(hours=24):
                            return {
                                "cached": True,
                                "age_hours": round(age.total_seconds() / 3600, 1),
                                "catalyst_id": catalyst_id,
                                "ticker": cat["ticker"],
                                "drug_name": cat["drug_name"],
                                "factors": cached.get("factors") or {},
                                "prior_crls": cached.get("prior_crls") or [],
                                "litigation": cached.get("litigation") or [],
                                "insider": cached.get("insider") or {},
                                "short": cached.get("short") or {},
                                "total_discount": cached.get("total_discount", 0),
                                "last_updated": cached["last_updated"].isoformat(),
                            }
                
                # 3. Compute fresh
                from services.risk_factors import estimate_risk_factors
                # Pull yfinance info
                try:
                    import yfinance as yf
                    info = yf.Ticker(cat["ticker"]).info or {}
                except Exception:
                    info = {}
                
                # Build drug-specific context
                drug_ctx = []
                if cat.get("drug_name"):
                    drug_ctx.append(f"Drug: {cat['drug_name']}")
                if cat.get("indication"):
                    drug_ctx.append(f"Indication: {cat['indication']}")
                if cat.get("phase"):
                    drug_ctx.append(f"Phase: {cat['phase']}")
                if cat.get("catalyst_type"):
                    drug_ctx.append(f"Upcoming catalyst: {cat['catalyst_type']} on {cat['catalyst_date']}")
                if cat.get("description"):
                    drug_ctx.append(f"Description: {cat['description'][:300]}")
                
                ai_context = "\n".join(drug_ctx) if drug_ctx else ""
                
                # Compute risks with drug-specific context
                result = estimate_risk_factors(
                    ticker=cat["ticker"],
                    company_name=cat.get("company_name") or cat["ticker"],
                    info=info,
                    news_excerpt="",
                    ai_context=ai_context,
                )
                
                # 4. Cache in stock_risk_factors
                factors_json = {k: v for k, v in result.items() 
                                if k not in {"prior_crls", "litigation", "insider", "short", "total_discount"}}
                
                cur.execute("""
                    INSERT INTO stock_risk_factors 
                        (ticker, catalyst_id, drug_name, factors, prior_crls, litigation,
                         insider, short, total_discount, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (ticker, catalyst_id, drug_name) DO UPDATE SET
                        factors = EXCLUDED.factors,
                        prior_crls = EXCLUDED.prior_crls,
                        litigation = EXCLUDED.litigation,
                        insider = EXCLUDED.insider,
                        short = EXCLUDED.short,
                        total_discount = EXCLUDED.total_discount,
                        last_updated = NOW()
                """, (
                    cat["ticker"], catalyst_id, cat.get("drug_name"),
                    Json(factors_json),
                    Json([]),  # prior_crls — TODO: web search for "ticker drug CRL"
                    Json([]),  # litigation — TODO: SEC EDGAR query
                    Json({"shortPctFloat": (info.get("shortPercentOfFloat", 0) or 0) * 100,
                          "insiderHeld": (info.get("heldPercentInsiders", 0) or 0) * 100}),
                    Json({"shortPctFloat": (info.get("shortPercentOfFloat", 0) or 0) * 100}),
                    result.get("total_discount", 0),
                ))
                conn.commit()
                
                return {
                    "cached": False,
                    "computed_at": datetime.utcnow().isoformat() + "Z",
                    "catalyst_id": catalyst_id,
                    "ticker": cat["ticker"],
                    "drug_name": cat["drug_name"],
                    "indication": cat["indication"],
                    "factors": factors_json,
                    "total_discount": result.get("total_discount", 0),
                    "context_used": ai_context,
                }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"risk-factors {catalyst_id}")
        raise HTTPException(500, f"risk-factors error: {e}")
