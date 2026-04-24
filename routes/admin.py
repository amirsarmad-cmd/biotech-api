"""
/admin — universe refresh, DB stats.
"""
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/universe/refresh")
async def refresh_universe():
    """Re-seed the 70-stock universe from Finnhub + FDA calendar."""
    try:
        from services.universe import seed_universe
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        stocks = seed_universe(expand=True)
        added = 0
        for s in stocks:
            if db.add_stock(s):
                added += 1
        return {"status": "ok", "fetched": len(stocks), "upserted": added}
    except Exception as e:
        logger.exception("universe refresh failed")
        raise HTTPException(500, f"universe refresh error: {e}")


@router.get("/db/stats")
async def db_stats():
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        stats = db.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(500, f"stats error: {e}")
