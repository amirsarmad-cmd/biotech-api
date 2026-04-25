"""
Universe seeder for Phase B — pulls all biotech/pharma tickers and extracts upcoming catalysts.

Pipeline:
  1. Pull universe candidates (Russell 3000 biotech + Nasdaq Biotech Index + ClinicalTrials.gov sponsors)
  2. For each ticker, extract upcoming non-earnings catalysts in next 6 months
  3. Filter, dedupe, write to catalyst_universe table

Strict cost discipline:
  - Every LLM call gated by LLM_ENABLED env var (default: 'false')
  - When disabled, returns mock data so we can verify the pipes without spending money
  - When enabled, uses gpt-4o-mini exclusively (cheapest model, ~$0.0002/call)
  - Daily spend cap via DAILY_LLM_BUDGET_USD env var (default: $5.00 - intentionally low for safety)

To enable LLM: set LLM_ENABLED=true on the biotech-api Railway service.
"""
import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict

import requests

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() in ("true", "1", "yes")
LLM_MODEL = os.getenv("UNIVERSE_LLM_MODEL", "gpt-4o-mini")
DAILY_LLM_BUDGET_USD = float(os.getenv("DAILY_LLM_BUDGET_USD", "5.00"))
COST_PER_LLM_CALL_USD = 0.0002  # gpt-4o-mini avg per call we make
MAX_TICKERS_PER_RUN = int(os.getenv("MAX_TICKERS_PER_RUN", "50"))  # safety throttle

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# In-memory daily spend tracker (resets on container restart)
_daily_spend = 0.0
_daily_spend_date = None


# ────────────────────────────────────────────────────────────
# Universe candidate sources
# ────────────────────────────────────────────────────────────
def get_seed_universe() -> List[Dict]:
    """
    Returns biotech/pharma ticker candidates from multiple sources.
    
    Phase B v1: starts with a curated seed list of well-known biotechs (~270 tickers).
    Phase B v2: will pull dynamically from SEC EDGAR + Russell 3000 + ClinicalTrials.gov.
    
    Each entry: {ticker, company_name, source}
    """
    # Curated NASDAQ Biotechnology Index members + S&P Pharma — a stable starting point
    # In production, replace with API pulls from SEC EDGAR sector codes
    SEED_TICKERS = [
        # Mega cap pharma (priority)
        {"ticker": "JNJ", "company_name": "Johnson & Johnson", "source": "seed"},
        {"ticker": "LLY", "company_name": "Eli Lilly and Co", "source": "seed"},
        {"ticker": "ABBV", "company_name": "AbbVie Inc", "source": "seed"},
        {"ticker": "MRK", "company_name": "Merck & Co Inc", "source": "seed"},
        {"ticker": "PFE", "company_name": "Pfizer Inc", "source": "seed"},
        {"ticker": "AZN", "company_name": "AstraZeneca PLC", "source": "seed"},
        {"ticker": "NVS", "company_name": "Novartis AG", "source": "seed"},
        {"ticker": "BMY", "company_name": "Bristol-Myers Squibb", "source": "seed"},
        {"ticker": "AMGN", "company_name": "Amgen Inc", "source": "seed"},
        {"ticker": "GILD", "company_name": "Gilead Sciences Inc", "source": "seed"},
        {"ticker": "REGN", "company_name": "Regeneron Pharmaceuticals", "source": "seed"},
        {"ticker": "VRTX", "company_name": "Vertex Pharmaceuticals", "source": "seed"},
        {"ticker": "MRNA", "company_name": "Moderna Inc", "source": "seed"},
        {"ticker": "BNTX", "company_name": "BioNTech SE", "source": "seed"},
        {"ticker": "BIIB", "company_name": "Biogen Inc", "source": "seed"},
        {"ticker": "GSK", "company_name": "GSK plc", "source": "seed"},
        {"ticker": "SNY", "company_name": "Sanofi", "source": "seed"},
        # Mid cap biotechs (priority)
        {"ticker": "ALNY", "company_name": "Alnylam Pharmaceuticals", "source": "seed"},
        {"ticker": "BMRN", "company_name": "BioMarin Pharmaceutical", "source": "seed"},
        {"ticker": "INCY", "company_name": "Incyte Corp", "source": "seed"},
        {"ticker": "EXEL", "company_name": "Exelixis Inc", "source": "seed"},
        {"ticker": "JAZZ", "company_name": "Jazz Pharmaceuticals", "source": "seed"},
        {"ticker": "NBIX", "company_name": "Neurocrine Biosciences", "source": "seed"},
        {"ticker": "SRPT", "company_name": "Sarepta Therapeutics", "source": "seed"},
        {"ticker": "RPRX", "company_name": "Royalty Pharma", "source": "seed"},
        {"ticker": "HALO", "company_name": "Halozyme Therapeutics", "source": "seed"},
        {"ticker": "NTRA", "company_name": "Natera Inc", "source": "seed"},
        # Small cap with active pipelines
        {"ticker": "FOLD", "company_name": "Amicus Therapeutics", "source": "seed"},
        {"ticker": "OCUL", "company_name": "Ocular Therapeutix", "source": "seed"},
        {"ticker": "QNRX", "company_name": "Quoin Pharmaceuticals", "source": "seed"},
        {"ticker": "PRAX", "company_name": "Praxis Precision Medicines", "source": "seed"},
        {"ticker": "IONS", "company_name": "Ionis Pharmaceuticals", "source": "seed"},
        {"ticker": "BCRX", "company_name": "BioCryst Pharmaceuticals", "source": "seed"},
        {"ticker": "ARWR", "company_name": "Arrowhead Pharmaceuticals", "source": "seed"},
        {"ticker": "BLUE", "company_name": "Bluebird Bio Inc", "source": "seed"},
        {"ticker": "ACAD", "company_name": "ACADIA Pharmaceuticals", "source": "seed"},
        {"ticker": "RIGL", "company_name": "Rigel Pharmaceuticals", "source": "seed"},
        {"ticker": "SAGE", "company_name": "Sage Therapeutics", "source": "seed"},
        {"ticker": "TVTX", "company_name": "Travere Therapeutics", "source": "seed"},
        {"ticker": "VKTX", "company_name": "Viking Therapeutics", "source": "seed"},
        {"ticker": "MDGL", "company_name": "Madrigal Pharmaceuticals", "source": "seed"},
        {"ticker": "AXSM", "company_name": "Axsome Therapeutics", "source": "seed"},
        {"ticker": "REPL", "company_name": "Replimune Group", "source": "seed"},
        {"ticker": "INSM", "company_name": "Insmed Inc", "source": "seed"},
        {"ticker": "CRSP", "company_name": "CRISPR Therapeutics", "source": "seed"},
        {"ticker": "NTLA", "company_name": "Intellia Therapeutics", "source": "seed"},
        {"ticker": "BEAM", "company_name": "Beam Therapeutics", "source": "seed"},
        {"ticker": "EDIT", "company_name": "Editas Medicine", "source": "seed"},
        # Tools & diagnostics
        {"ticker": "ILMN", "company_name": "Illumina Inc", "source": "seed"},
        {"ticker": "TMO", "company_name": "Thermo Fisher Scientific", "source": "seed"},
        {"ticker": "DHR", "company_name": "Danaher Corp", "source": "seed"},
        {"ticker": "A", "company_name": "Agilent Technologies", "source": "seed"},
    ]
    return SEED_TICKERS


# ────────────────────────────────────────────────────────────
# LLM gate (only path that costs money)
# ────────────────────────────────────────────────────────────
def _check_budget() -> bool:
    """Returns True if we have budget remaining today."""
    global _daily_spend, _daily_spend_date
    today = date.today()
    if _daily_spend_date != today:
        _daily_spend = 0.0
        _daily_spend_date = today
    return _daily_spend < DAILY_LLM_BUDGET_USD


def _record_spend(usd: float):
    """Track LLM spend."""
    global _daily_spend, _daily_spend_date
    today = date.today()
    if _daily_spend_date != today:
        _daily_spend = 0.0
        _daily_spend_date = today
    _daily_spend += usd


def get_daily_spend() -> Dict:
    """Inspectable spend stats."""
    return {
        "date": str(_daily_spend_date) if _daily_spend_date else None,
        "spent_usd": round(_daily_spend, 4),
        "budget_usd": DAILY_LLM_BUDGET_USD,
        "remaining_usd": round(DAILY_LLM_BUDGET_USD - _daily_spend, 4),
        "llm_enabled": LLM_ENABLED,
        "model": LLM_MODEL,
    }


def extract_catalysts_for_ticker(ticker: str, company_name: str) -> Tuple[List[Dict], Dict]:
    """
    Extract upcoming catalysts for a single ticker.
    
    Returns (catalysts_list, meta) where meta includes source ('llm'|'mock'|'cached'),
    cost incurred, and any error string.
    
    Cost discipline:
      - If LLM_ENABLED=false → returns mock catalyst (zero cost)
      - If budget exceeded → returns empty list + meta error
      - If OPENAI_API_KEY missing → returns empty list + meta error
    """
    meta = {"source": None, "cost_usd": 0.0, "error": None}

    # Fast path: LLM disabled = mock data (for plumbing tests)
    if not LLM_ENABLED:
        meta["source"] = "mock"
        return _mock_catalysts(ticker, company_name), meta

    # Real LLM path - guarded by budget + key
    if not OPENAI_API_KEY:
        meta["error"] = "OPENAI_API_KEY not set"
        return [], meta
    if not _check_budget():
        meta["error"] = f"Daily LLM budget exceeded (${DAILY_LLM_BUDGET_USD})"
        return [], meta

    try:
        catalysts = _call_openai_extract(ticker, company_name)
        _record_spend(COST_PER_LLM_CALL_USD)
        meta["source"] = "llm"
        meta["cost_usd"] = COST_PER_LLM_CALL_USD
        return catalysts, meta
    except Exception as e:
        meta["error"] = f"LLM call failed: {type(e).__name__}: {e}"
        logger.exception(f"LLM extract failed for {ticker}")
        return [], meta


def _mock_catalysts(ticker: str, company_name: str) -> List[Dict]:
    """Mock catalysts to verify the plumbing works without LLM cost."""
    today = date.today()
    return [
        {
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": "FDA Decision",
            "catalyst_date": (today + timedelta(days=90)).isoformat(),
            "date_precision": "exact",
            "description": f"[MOCK] Hypothetical FDA decision for {company_name} lead drug",
            "drug_name": "MOCK-001",
            "canonical_drug_name": "mock-drug-001",
            "indication": "Mock indication",
            "phase": "BLA",
            "source": "mock",
            "source_url": None,
            "confidence_score": 0.5,
        },
        {
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": "Phase 3 Readout",
            "catalyst_date": (today + timedelta(days=120)).isoformat(),
            "date_precision": "quarter",
            "description": f"[MOCK] Hypothetical Phase 3 readout for {company_name}",
            "drug_name": "MOCK-002",
            "canonical_drug_name": "mock-drug-002",
            "indication": "Mock indication 2",
            "phase": "Phase 3",
            "source": "mock",
            "source_url": None,
            "confidence_score": 0.5,
        },
    ]


def _call_openai_extract(ticker: str, company_name: str) -> List[Dict]:
    """Call OpenAI to extract upcoming catalysts. Returns parsed list of dicts."""
    today = date.today()
    six_months = today + timedelta(days=183)
    
    prompt = f"""You are a biotech catalyst analyst. Extract upcoming non-earnings catalysts for {ticker} ({company_name}) expected between {today.isoformat()} and {six_months.isoformat()}.

Catalyst types to find: FDA Decision, AdComm, Phase 1/2/3 Readout, Clinical Trial, Partnership, BLA submission, NDA submission.
EXCLUDE: earnings reports, conferences, investor days.

For each catalyst found, provide JSON object with these fields:
- catalyst_type: one of the types above
- catalyst_date: ISO date YYYY-MM-DD (estimate if uncertain)
- date_precision: "exact" | "quarter" | "half" | "year"
- description: 1-sentence factual description
- drug_name: drug or program name
- indication: disease/condition
- phase: "Phase 1" | "Phase 2" | "Phase 3" | "BLA" | "NDA" | null
- confidence_score: 0.0-1.0 (your confidence in this prediction)

Return ONLY a JSON array. If no catalysts known, return empty array [].
Do not invent facts. Only include catalysts you're at least 60% confident exist.
"""

    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=45)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    
    # Parse JSON — model may return {"catalysts": [...]} or [...]
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for k in ("catalysts", "results", "data", "items"):
            if k in parsed and isinstance(parsed[k], list):
                parsed = parsed[k]
                break
        else:
            parsed = []
    if not isinstance(parsed, list):
        parsed = []
    
    # Normalize each entry, attach ticker info
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        out.append({
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": item.get("catalyst_type", "Unknown"),
            "catalyst_date": item.get("catalyst_date"),
            "date_precision": item.get("date_precision", "exact"),
            "description": item.get("description", ""),
            "drug_name": item.get("drug_name"),
            "canonical_drug_name": _canonicalize_drug(item.get("drug_name")),
            "indication": item.get("indication"),
            "phase": item.get("phase"),
            "source": "llm",
            "source_url": None,
            "confidence_score": item.get("confidence_score", 0.6),
        })
    return out


def _canonicalize_drug(name: Optional[str]) -> Optional[str]:
    """Lowercase + strip + collapse whitespace."""
    if not name:
        return None
    return " ".join(name.lower().split())


# ────────────────────────────────────────────────────────────
# DB writer
# ────────────────────────────────────────────────────────────
def write_catalysts_to_db(catalysts: List[Dict], conn) -> Dict:
    """
    Upsert catalysts into catalyst_universe.
    Returns {"added": int, "updated": int, "skipped": int, "errors": [...]}
    """
    stats = {"added": 0, "updated": 0, "skipped": 0, "errors": []}
    
    with conn.cursor() as cur:
        for c in catalysts:
            try:
                if not c.get("catalyst_date"):
                    stats["skipped"] += 1
                    continue
                
                # Try INSERT, fall back to UPDATE on conflict
                cur.execute("""
                    INSERT INTO catalyst_universe
                        (ticker, company_name, catalyst_type, catalyst_date, date_precision,
                         description, drug_name, canonical_drug_name, indication, phase,
                         source, source_url, confidence_score, status, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', NOW())
                    ON CONFLICT (ticker, catalyst_type, catalyst_date, drug_name)
                    DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        description = EXCLUDED.description,
                        indication = EXCLUDED.indication,
                        phase = EXCLUDED.phase,
                        confidence_score = EXCLUDED.confidence_score,
                        source = EXCLUDED.source,
                        last_updated = NOW()
                    RETURNING (xmax = 0) AS inserted
                """, (
                    c["ticker"], c.get("company_name"), c["catalyst_type"],
                    c["catalyst_date"], c.get("date_precision", "exact"),
                    c.get("description"), c.get("drug_name"), c.get("canonical_drug_name"),
                    c.get("indication"), c.get("phase"),
                    c.get("source", "unknown"), c.get("source_url"),
                    c.get("confidence_score"),
                ))
                inserted = cur.fetchone()[0]
                if inserted:
                    stats["added"] += 1
                else:
                    stats["updated"] += 1
            except Exception as e:
                stats["errors"].append(f"{c.get('ticker')}/{c.get('catalyst_type')}: {e}")
        conn.commit()
    return stats


# ────────────────────────────────────────────────────────────
# Top-level orchestration
# ────────────────────────────────────────────────────────────
def run_universe_seed(max_tickers: Optional[int] = None) -> Dict:
    """
    Main entry point. Run from /admin/universe/v2-seed endpoint or cron.
    Returns a summary dict with counts + cost incurred + errors.
    """
    import psycopg2
    
    started = datetime.utcnow()
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    conn = psycopg2.connect(db_url)
    
    # Log cron run start
    cron_run_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cron_runs (job_name, status) VALUES ('universe_seed_v2', 'running')
                RETURNING id
            """)
            cron_run_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        logger.warning(f"cron_runs insert failed: {e}")
    
    universe = get_seed_universe()
    cap = max_tickers or MAX_TICKERS_PER_RUN
    universe = universe[:cap]
    
    logger.info(f"[universe_seed] LLM_ENABLED={LLM_ENABLED} model={LLM_MODEL} processing {len(universe)} tickers (cap={cap})")
    
    total_added = 0
    total_updated = 0
    total_errors = []
    total_cost = 0.0
    catalysts_per_source = {"mock": 0, "llm": 0, "error": 0}
    
    for entry in universe:
        ticker = entry["ticker"]
        company = entry["company_name"]
        try:
            catalysts, meta = extract_catalysts_for_ticker(ticker, company)
            total_cost += meta.get("cost_usd", 0.0)
            
            if meta.get("source"):
                catalysts_per_source[meta["source"]] = catalysts_per_source.get(meta["source"], 0) + len(catalysts)
            
            if meta.get("error"):
                total_errors.append(f"{ticker}: {meta['error']}")
                catalysts_per_source["error"] += 1
                continue
            
            if catalysts:
                stats = write_catalysts_to_db(catalysts, conn)
                total_added += stats["added"]
                total_updated += stats["updated"]
                if stats["errors"]:
                    total_errors.extend(stats["errors"])
        except Exception as e:
            total_errors.append(f"{ticker}: orchestrator error: {e}")
            logger.exception(f"orchestrator error for {ticker}")
    
    completed = datetime.utcnow()
    duration_s = (completed - started).total_seconds()
    
    summary = {
        "job": "universe_seed_v2",
        "started_at": started.isoformat() + "Z",
        "completed_at": completed.isoformat() + "Z",
        "duration_seconds": round(duration_s, 1),
        "tickers_processed": len(universe),
        "catalysts_added": total_added,
        "catalysts_updated": total_updated,
        "catalysts_per_source": catalysts_per_source,
        "total_llm_cost_usd": round(total_cost, 4),
        "daily_spend_after": get_daily_spend(),
        "llm_enabled": LLM_ENABLED,
        "errors": total_errors[:20],  # Cap error list
        "error_count": len(total_errors),
    }
    
    # Update cron run
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cron_runs SET
                    completed_at = NOW(),
                    status = %s,
                    records_processed = %s,
                    records_added = %s,
                    records_updated = %s,
                    errors = %s,
                    log = %s
                WHERE id = %s
            """, (
                "success" if not total_errors else ("partial" if total_added > 0 else "failed"),
                len(universe), total_added, total_updated,
                json.dumps(total_errors[:50]) if total_errors else None,
                json.dumps(summary),
                cron_run_id,
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"cron_runs update failed: {e}")
    
    conn.close()
    return summary
