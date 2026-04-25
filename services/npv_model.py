"""
NPV-based catalyst impact model.

Replaces generic +35%/-45% heuristics with stock-specific impact calculations:
  expected_move_% = (NPV_of_drug × P_commercial) / market_cap × sentiment_adjustment

Uses AI to estimate:
  - peak_annual_sales_usd
  - commercial_success_probability_given_approval
  - relevant multiple for valuation

And market data for:
  - market cap
  - 60-day prior price baseline for implied move calc
"""
import os
import json
import math
import logging
import hashlib

# Risk factor integration (optional — falls back if module missing)
try:
    from services.risk_factors import estimate_risk_factors, apply_risk_discount
    RISK_FACTORS_AVAILABLE = True
except ImportError:
    RISK_FACTORS_AVAILABLE = False
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# DEFAULT WEIGHTS / CONFIG
# ============================================================

# Multi-provider LLM helper (standalone module, no circular import risk)
from services.llm_helper import call_llm_json as _call_llm_json


DEFAULT_NPV_WEIGHTS = {
    # Revenue multiples by company size (used for NPV calc)
    "small_cap_multiple":  4.0,   # <$5B market cap — higher multiples
    "mid_cap_multiple":    3.5,   # $5B-$50B
    "large_cap_multiple":  3.0,   # >$50B — more mature pricing
    
    # Implied move baseline
    "baseline_days":       60,    # how far back to check for pre-catalyst baseline
    
    # Rejection overshoot: beyond just losing the priced-in upside,
    # stocks typically take a disappointment hit
    "rejection_overshoot_pct":  12.0,  # additional % hit on top of giving back priced-in move
    
    # Sentiment dampeners (still useful for stock-specific variation)
    "institutional_dampener":    0.10,   # max 10% reduction for heavy institutional ownership
    "high_short_amplifier":      0.30,   # max 30% amplification for high short interest
    
    # Fallback peak-sales estimates (if AI fails)
    "fallback_peak_sales_b": {
        "FDA": 1.0,       # $1B
        "Clinical": 0.6,  # $600M (earlier stage, more uncertain)
        "Earnings": 0.0,  # earnings don't have peak sales
        "Other": 0.3,
    },
    "fallback_commercial_prob": 0.60,
}


# ============================================================
# AI ESTIMATOR: peak sales, commercial success probability
# ============================================================

def estimate_drug_economics(ticker: str, company_name: str, catalyst_type: str,
                            catalyst_date: str, description: str,
                            market_cap_m: float, ai_context_sources: Optional[str] = None) -> Dict:
    """
    Claude estimates the drug's peak annual sales, time to peak, revenue multiple,
    and probability of commercial success given regulatory approval.
    
    Returns dict:
    {
        "peak_sales_usd_b": 2.0,
        "peak_sales_year": 2029,
        "peak_sales_rationale": "...",
        "multiple": 4.0,
        "multiple_rationale": "...",
        "commercial_success_prob": 0.70,
        "commercial_success_rationale": "...",
        "first_in_class": true/false,
        "competitive_intensity": "low|medium|high",
        "error": None | "error message"
    }
    """
    cap_size = "small-cap" if market_cap_m < 5000 else "mid-cap" if market_cap_m < 50000 else "large-cap"
    
    try:
        prompt = f"""You are a senior biotech equity analyst specializing in drug valuation.

Estimate the commercial economics of this catalyst for NPV modeling.

STOCK: {ticker} — {company_name}
MARKET CAP: ${market_cap_m:,.0f}M ({cap_size})
CATALYST: {catalyst_type} on {catalyst_date}
DESCRIPTION: {description}

{f'CONTEXT FROM RESEARCH:{chr(10)}{ai_context_sources[:3000]}' if ai_context_sources else ''}

Provide a rigorous estimate in JSON format. Do not include markdown fences, only valid JSON:

{{
  "peak_sales_usd_b": <number — global peak annual revenue in $B, e.g. 2.0 = $2B>,
  "peak_sales_year": <integer year when peak reached, e.g. 2029>,
  "peak_sales_rationale": "<1-2 sentences: TAM, market share assumption, pricing>",
  "multiple": <revenue multiple to apply for NPV, typically 3-5 for biotech>,
  "multiple_rationale": "<1 sentence: why this multiple>",
  "commercial_success_prob": <0-1 probability the drug sells well AFTER approval (not approval prob itself)>,
  "commercial_success_rationale": "<1-2 sentences: competitive landscape, market access, differentiation>",
  "first_in_class": <true/false>,
  "competitive_intensity": "<'low' | 'medium' | 'high'>"
}}

GUIDANCE:
- For FDA/PDUFA catalysts, estimate the DRUG's peak revenue as if approved and commercialized well
- commercial_success_prob is P(commercial success | approval). Consider:
  * First-in-class in unmet-need indication = 0.75-0.90
  * Best-in-class with strong efficacy vs incumbents = 0.60-0.75
  * Me-too in crowded market = 0.30-0.55
  * Major competitor already dominant = 0.20-0.40
- Use multiple = 4x for innovative/orphan/specialty drugs, 3x for primary care, 5x for blockbusters/oncology
- Do NOT include the probability of FDA approval in these numbers (that's separate)
- For earnings catalysts: set peak_sales_usd_b to 0 and commercial_success_prob to 1.0

Return ONLY the JSON object."""
        
        result, err = _call_llm_json(prompt, max_tokens=1200)
        if result is None:
            raise RuntimeError(f"All LLM providers failed: {err}")
        result["error"] = None
        return result
    except Exception as e:
        logger.warning(f"estimate_drug_economics failed for {ticker}: {e}")
        # Fallback: size-based defaults
        cap_size = "small-cap" if market_cap_m < 5000 else "mid-cap" if market_cap_m < 50000 else "large-cap"
        default_sales = {"small-cap": 0.8, "mid-cap": 1.5, "large-cap": 0.8}[cap_size]
        default_mult = DEFAULT_NPV_WEIGHTS["small_cap_multiple"] if cap_size == "small-cap" \
            else DEFAULT_NPV_WEIGHTS["mid_cap_multiple"] if cap_size == "mid-cap" \
            else DEFAULT_NPV_WEIGHTS["large_cap_multiple"]
        return {
            "peak_sales_usd_b": default_sales,
            "peak_sales_year": datetime.now().year + 4,
            "peak_sales_rationale": f"Fallback estimate — AI unavailable. Typical {cap_size} drug.",
            "multiple": default_mult,
            "multiple_rationale": f"Default {cap_size} multiple.",
            "commercial_success_prob": DEFAULT_NPV_WEIGHTS["fallback_commercial_prob"],
            "commercial_success_rationale": "Fallback — AI unavailable.",
            "first_in_class": False,
            "competitive_intensity": "medium",
            "error": str(e)[:200],
        }


# ============================================================
# HISTORICAL PRICE FETCHER for implied move calc
# ============================================================

def get_baseline_price(ticker: str, days_back: int = 60) -> Optional[float]:
    """Fetch the stock price from `days_back` trading days ago for implied-move calc."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{days_back+30}d")
        if hist is None or hist.empty:
            return None
        target_date = datetime.now() - timedelta(days=days_back)
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
        before = hist[hist.index <= target_date]
        if before.empty:
            # Use earliest available
            return float(hist["Close"].iloc[0])
        return float(before["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"baseline price fetch failed for {ticker}: {e}")
        return None


# ============================================================
# CORE NPV-BASED ESTIMATOR
# ============================================================

def compute_npv_estimate(
    ticker: str,
    current_price: float,
    market_cap_m: float,
    p_approval: float,
    economics: Dict,
    baseline_price: Optional[float] = None,
    weights: Optional[Dict] = None,
    info: Optional[Dict] = None,
    risk_factors: Optional[Dict] = None,
) -> Dict:
    """
    Compute NPV-based catalyst impact using stock-specific fundamentals.
    
    Returns a dict with:
      - current / approval / rejection / expected prices
      - breakdown for UI display
    """
    w = {**DEFAULT_NPV_WEIGHTS, **(weights or {})}
    info = info or {}
    
    # ---- STAGE 1: NPV of drug ----
    peak_sales_b = float(economics.get("peak_sales_usd_b", 0.0))
    multiple = float(economics.get("multiple", 3.5))
    p_commercial = float(economics.get("commercial_success_prob", w["fallback_commercial_prob"]))
    
    # NPV = peak annual sales × multiple × P(commercial success | approval)
    # Peak sales in $M, market cap in $M — keep units consistent
    peak_sales_m = peak_sales_b * 1000.0
    raw_drug_npv_m = peak_sales_m * multiple * p_commercial
    
    # Apply adverse risk factor discounts (7 factors: litigation, FDA hist, SEC/short,
    # insider selling, going concern, patent cliff, governance)
    risk_discount_applied = 0.0
    risk_factor_breakdown = None
    if risk_factors and isinstance(risk_factors, dict):
        risk_discount_applied = float(risk_factors.get("total_discount", 0) or 0)
        risk_factor_breakdown = risk_factors
        drug_npv_m = raw_drug_npv_m * (1 - risk_discount_applied)
    else:
        drug_npv_m = raw_drug_npv_m
    
    # Fundamental impact capacity on stock (% of market cap)
    fundamental_impact_pct = (drug_npv_m / market_cap_m * 100.0) if market_cap_m > 0 else 0.0
    
    # ---- STAGE 2: Sentiment adjustment ----
    # Institutional ownership dampens moves (less retail reactivity)
    # High short interest amplifies both directions (squeeze + flush risk)
    sentiment_adj_factor = 1.0
    sentiment_notes = []
    
    inst_pct = (info.get("heldPercentInstitutions", 0) or 0)
    if inst_pct > 0.7:
        dampening = w["institutional_dampener"] * (inst_pct - 0.7) / 0.3  # scale 0.7→0.10 to 1.0→0.10
        sentiment_adj_factor *= (1 - min(dampening, w["institutional_dampener"]))
        sentiment_notes.append(f"Institutional ownership {inst_pct:.0%} dampens moves by {dampening*100:.1f}%")
    
    short_pct = (info.get("shortPercentOfFloat", 0) or 0)
    if short_pct > 0.15:
        amp = w["high_short_amplifier"] * min((short_pct - 0.15) / 0.20, 1.0)  # caps at 35% short
        sentiment_adj_factor *= (1 + amp)
        sentiment_notes.append(f"High short interest {short_pct:.0%} amplifies moves by {amp*100:.1f}%")
    
    # ---- STAGE 3: Implied move already priced in ----
    # What % of the potential upside has the market ALREADY priced in since baseline?
    if baseline_price and baseline_price > 0:
        implied_move_pct = (current_price - baseline_price) / baseline_price * 100.0
    else:
        implied_move_pct = 0.0
    
    # ---- STAGE 4: Approval / Rejection prices ----
    # Approval: fundamental upside × sentiment × (minus what's already priced in)
    full_approval_pct = fundamental_impact_pct * sentiment_adj_factor
    # Only the REMAINING upside will show post-event
    remaining_upside_pct = max(full_approval_pct - max(implied_move_pct, 0), 0)
    
    # Rejection: give back the priced-in upside + disappointment overshoot
    rejection_pct = -(max(implied_move_pct, 0) + w["rejection_overshoot_pct"])
    rejection_pct *= sentiment_adj_factor  # short/inst adjustments apply here too
    
    # Also: if fundamental impact is huge, rejection may be even worse
    if full_approval_pct > 30:
        additional_rejection = -(full_approval_pct * 0.3)  # additional 30% of upside lost
        rejection_pct += additional_rejection
    
    # Final prices
    approval_price = current_price * (1 + remaining_upside_pct / 100.0)
    rejection_price = current_price * (1 + rejection_pct / 100.0)
    
    # Expected = prob-weighted
    expected_price = (p_approval * approval_price) + ((1 - p_approval) * rejection_price)
    expected_pct = (expected_price / current_price - 1) * 100.0
    
    # Combined probability of value realization
    combined_prob = p_approval * p_commercial
    
    return {
        "current": current_price,
        "approval": approval_price,
        "rejection": rejection_price,
        "expected": expected_price,
        
        # Percentages for display
        "upside_pct": remaining_upside_pct,
        "downside_pct": rejection_pct,
        "expected_pct": expected_pct,
        
        # NPV breakdown
        "peak_sales_b": peak_sales_b,
        "peak_sales_rationale": economics.get("peak_sales_rationale", ""),
        "peak_sales_year": economics.get("peak_sales_year"),
        "multiple": multiple,
        "multiple_rationale": economics.get("multiple_rationale", ""),
        "raw_drug_npv_m": raw_drug_npv_m,
        "drug_npv_m": drug_npv_m,  # After risk discount
        "risk_discount_pct": risk_discount_applied * 100,
        "risk_factor_breakdown": risk_factor_breakdown,
        "market_cap_m": market_cap_m,
        "fundamental_impact_pct": fundamental_impact_pct,
        "full_approval_pct_theoretical": full_approval_pct,
        
        # Probability math
        "p_approval": p_approval,
        "p_commercial": p_commercial,
        "commercial_rationale": economics.get("commercial_success_rationale", ""),
        "combined_prob": combined_prob,
        
        # Implied move
        "baseline_price": baseline_price,
        "baseline_days": w["baseline_days"],
        "implied_move_pct": implied_move_pct,
        
        # Sentiment
        "sentiment_adj_factor": sentiment_adj_factor,
        "sentiment_notes": sentiment_notes,
        
        # Metadata
        "ai_error": economics.get("error"),
        "first_in_class": economics.get("first_in_class", False),
        "competitive_intensity": economics.get("competitive_intensity", "medium"),
    }


# ============================================================
# UTILITY: hash key for caching
# ============================================================

def make_cache_key(ticker: str, catalyst_type: str, catalyst_date: str) -> str:
    h = hashlib.md5(f"{ticker}|{catalyst_type}|{catalyst_date}".encode()).hexdigest()
    return f"npv_{h[:16]}"


# ============================================================
# V2: STRUCTURED DRUG ECONOMICS + TRUE rNPV (Phase 2B)
# ============================================================
# These functions WRAP the legacy compute_npv_estimate. They:
#   1. Fetch structured drug economics (pop × price × penetration) from cache
#      or LLM, persist to drug_economics_cache table
#   2. Read npv_defaults singleton from DB (so admin-tuned defaults apply)
#   3. Compute year-by-year rNPV with discounting, LOE drop-off
#   4. Cache final NPV to catalyst_npv_cache table
#
# Backward compatible: if any V2 component fails, callers can still fall
# back to legacy compute_npv_estimate(). All V2 functions return None or
# error-tagged dicts on failure rather than raising.
# ============================================================

import time
from typing import Tuple

# Cache for npv_defaults singleton (in-process, 60s TTL)
_NPV_DEFAULTS_CACHE = {"data": None, "ts": 0.0}
_NPV_DEFAULTS_TTL = 60.0


def load_npv_defaults_from_db() -> Dict:
    """Read npv_defaults singleton row, merge with in-code DEFAULT_NPV_WEIGHTS.
    Returns a dict suitable for passing as `weights` to compute_npv_estimate.
    Cached in-process for 60s. Falls back to DEFAULT_NPV_WEIGHTS on any error.
    """
    now = time.time()
    if _NPV_DEFAULTS_CACHE["data"] is not None and (now - _NPV_DEFAULTS_CACHE["ts"]) < _NPV_DEFAULTS_TTL:
        return _NPV_DEFAULTS_CACHE["data"]

    merged = dict(DEFAULT_NPV_WEIGHTS)
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT discount_rate, tax_rate, cogs_pct,
                                  default_penetration_pct, default_time_to_peak_years,
                                  rating_weights
                           FROM npv_defaults WHERE scope='global' LIMIT 1""")
            row = cur.fetchone()
            if row:
                merged["discount_rate"] = float(row[0]) if row[0] is not None else 0.12
                merged["tax_rate"] = float(row[1]) if row[1] is not None else 0.21
                merged["cogs_pct"] = float(row[2]) if row[2] is not None else 0.15
                merged["default_penetration_pct"] = float(row[3]) if row[3] is not None else 0.15
                merged["default_time_to_peak_years"] = int(row[4]) if row[4] is not None else 5
                # rating_weights JSONB -> dict
                rw = row[5]
                if isinstance(rw, str):
                    try: rw = json.loads(rw)
                    except Exception: rw = {}
                merged["rating_weights"] = rw if isinstance(rw, dict) else {}
                logger.debug(f"npv_defaults loaded: discount={merged['discount_rate']}, tax={merged['tax_rate']}, cogs={merged['cogs_pct']}")
    except Exception as e:
        logger.warning(f"load_npv_defaults_from_db failed (using in-code defaults): {e}")
        # Provide hardcoded fallbacks so downstream code can rely on these keys
        merged.setdefault("discount_rate", 0.12)
        merged.setdefault("tax_rate", 0.21)
        merged.setdefault("cogs_pct", 0.15)
        merged.setdefault("default_penetration_pct", 0.15)
        merged.setdefault("default_time_to_peak_years", 5)
        merged.setdefault("rating_weights", {})

    _NPV_DEFAULTS_CACHE["data"] = merged
    _NPV_DEFAULTS_CACHE["ts"] = now
    return merged


def _canonicalize_drug_name(name: str) -> str:
    """Normalize drug name for cache lookup. Drops parenthetical generics, lowercases."""
    if not name: return ""
    n = name.strip().lower()
    # Drop parentheticals
    if "(" in n: n = n[:n.index("(")].strip()
    # Common cleanup
    return " ".join(n.split())


def get_drug_economics_v2_from_cache(ticker: str, drug_name: str) -> Optional[Dict]:
    """Read drug_economics_cache by (ticker, canonical_drug_name). Returns None if
    not present, expired, or on error."""
    canon = _canonicalize_drug_name(drug_name)
    if not canon or not ticker:
        return None
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT addressable_population_us, addressable_population_global,
                                  annual_cost_min_usd, annual_cost_max_usd,
                                  standard_of_care_cost_usd,
                                  penetration_min_pct, penetration_max_pct, penetration_mid_pct,
                                  launch_year, peak_sales_year, patent_expiry_date,
                                  competitors, competitive_intensity, first_in_class,
                                  llm_rationale, llm_provider, computed_at, ttl, indication
                           FROM drug_economics_cache
                           WHERE ticker=%s AND canonical_drug_name=%s""",
                        (ticker, canon))
            row = cur.fetchone()
            if not row: return None
            # If TTL set and expired, treat as miss
            ttl = row[17]
            if ttl is not None:
                # Postgres returns timezone-aware datetime
                try:
                    if ttl < datetime.now().replace(tzinfo=ttl.tzinfo):
                        return None
                except Exception:
                    pass
            return {
                "ticker": ticker,
                "canonical_drug_name": canon,
                "addressable_population_us": int(row[0]) if row[0] else None,
                "addressable_population_global": int(row[1]) if row[1] else None,
                "annual_cost_min_usd": float(row[2]) if row[2] else None,
                "annual_cost_max_usd": float(row[3]) if row[3] else None,
                "standard_of_care_cost_usd": float(row[4]) if row[4] else None,
                "penetration_min_pct": float(row[5]) if row[5] else None,
                "penetration_max_pct": float(row[6]) if row[6] else None,
                "penetration_mid_pct": float(row[7]) if row[7] else None,
                "launch_year": int(row[8]) if row[8] else None,
                "peak_sales_year": int(row[9]) if row[9] else None,
                "patent_expiry_date": str(row[10]) if row[10] else None,
                "competitors": row[11] if isinstance(row[11], (list, dict)) else (json.loads(row[11]) if row[11] else []),
                "competitive_intensity": row[12],
                "first_in_class": bool(row[13]) if row[13] is not None else None,
                "llm_rationale": row[14],
                "llm_provider": row[15],
                "computed_at": str(row[16]) if row[16] else None,
                "indication": row[18],
                "_from_cache": True,
            }
    except Exception as e:
        logger.warning(f"get_drug_economics_v2_from_cache failed: {e}")
        return None


def write_drug_economics_v2_to_cache(ticker: str, drug_name: str, econ_v2: Dict, ttl_days: int = 7) -> bool:
    """Persist structured drug economics to drug_economics_cache.
    UPSERT on (ticker, canonical_drug_name). Returns True on success."""
    canon = _canonicalize_drug_name(drug_name)
    if not canon or not ticker:
        return False
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        ttl = datetime.now() + timedelta(days=ttl_days)
        comps = econ_v2.get("competitors") or []
        comps_json = json.dumps(comps) if not isinstance(comps, str) else comps
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""INSERT INTO drug_economics_cache
                (ticker, canonical_drug_name, indication,
                 addressable_population_us, addressable_population_global,
                 annual_cost_min_usd, annual_cost_max_usd, standard_of_care_cost_usd,
                 penetration_min_pct, penetration_max_pct, penetration_mid_pct,
                 launch_year, peak_sales_year, patent_expiry_date,
                 competitors, competitive_intensity, first_in_class,
                 llm_rationale, llm_provider, computed_at, ttl)
                VALUES (%s,%s,%s, %s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,
                        %s::jsonb,%s,%s, %s,%s, NOW(), %s)
                ON CONFLICT (ticker, canonical_drug_name) DO UPDATE SET
                    indication = EXCLUDED.indication,
                    addressable_population_us = EXCLUDED.addressable_population_us,
                    addressable_population_global = EXCLUDED.addressable_population_global,
                    annual_cost_min_usd = EXCLUDED.annual_cost_min_usd,
                    annual_cost_max_usd = EXCLUDED.annual_cost_max_usd,
                    standard_of_care_cost_usd = EXCLUDED.standard_of_care_cost_usd,
                    penetration_min_pct = EXCLUDED.penetration_min_pct,
                    penetration_max_pct = EXCLUDED.penetration_max_pct,
                    penetration_mid_pct = EXCLUDED.penetration_mid_pct,
                    launch_year = EXCLUDED.launch_year,
                    peak_sales_year = EXCLUDED.peak_sales_year,
                    patent_expiry_date = EXCLUDED.patent_expiry_date,
                    competitors = EXCLUDED.competitors,
                    competitive_intensity = EXCLUDED.competitive_intensity,
                    first_in_class = EXCLUDED.first_in_class,
                    llm_rationale = EXCLUDED.llm_rationale,
                    llm_provider = EXCLUDED.llm_provider,
                    computed_at = NOW(),
                    ttl = EXCLUDED.ttl""",
                (ticker, canon, econ_v2.get("indication"),
                 econ_v2.get("addressable_population_us"),
                 econ_v2.get("addressable_population_global"),
                 econ_v2.get("annual_cost_min_usd"),
                 econ_v2.get("annual_cost_max_usd"),
                 econ_v2.get("standard_of_care_cost_usd"),
                 econ_v2.get("penetration_min_pct"),
                 econ_v2.get("penetration_max_pct"),
                 econ_v2.get("penetration_mid_pct"),
                 econ_v2.get("launch_year"),
                 econ_v2.get("peak_sales_year"),
                 econ_v2.get("patent_expiry_date"),
                 comps_json,
                 econ_v2.get("competitive_intensity"),
                 econ_v2.get("first_in_class"),
                 econ_v2.get("llm_rationale"),
                 econ_v2.get("llm_provider"),
                 ttl))
            conn.commit()
        logger.info(f"drug_economics_cache written: {ticker}/{canon}")
        return True
    except Exception as e:
        logger.warning(f"write_drug_economics_v2_to_cache failed for {ticker}/{canon}: {e}")
        return False


def estimate_drug_economics_v2(ticker: str, company_name: str, drug_name: str,
                                catalyst_type: str, catalyst_date: str,
                                description: str, market_cap_m: float,
                                ai_context_sources: Optional[str] = None) -> Dict:
    """STRUCTURED LLM call for drug economics. Returns:
    {
        addressable_population_us / global: int (number of patients)
        annual_cost_min_usd / max_usd: float
        standard_of_care_cost_usd: float (current SOC pricing)
        penetration_min/max/mid_pct: float (peak market share)
        launch_year: int
        peak_sales_year: int
        patent_expiry_date: str (YYYY-MM-DD)
        competitors: [str, ...]
        competitive_intensity: 'low'|'medium'|'high'
        first_in_class: bool
        modality: 'small_molecule'|'biologic'|'cell_gene'|'antibody'|'other'
        cogs_pct_estimate: float (modality-specific)
        time_to_peak_years: float
        loe_dropoff_pct: float (e.g. 0.75 = 75% drop year 1 post-LOE)
        peak_sales_usd_b: float (computed: pop_global × price × penetration_mid)
        commercial_success_prob: 0-1
        llm_rationale: str
        llm_provider: str
        indication: str
    }
    On total failure returns dict with 'error' key set, partial data otherwise.
    """
    cap_size = "small-cap" if market_cap_m < 5000 else "mid-cap" if market_cap_m < 50000 else "large-cap"
    drug = drug_name or "(unspecified)"

    prompt = f"""You are a senior biotech equity analyst building a structured rNPV model.

Estimate drug economics in PRECISE STRUCTURED form (not narrative).

STOCK: {ticker} — {company_name}
DRUG: {drug}
INDICATION (from description): {description[:400]}
CATALYST: {catalyst_type} on {catalyst_date}
MARKET CAP: ${market_cap_m:,.0f}M ({cap_size})

{f'CONTEXT FROM RESEARCH:{chr(10)}{ai_context_sources[:2500]}' if ai_context_sources else ''}

Return ONLY a JSON object. No markdown, no preamble. Use realistic, evidence-based numbers.
If you genuinely don't know a field, set it to null. Don't fabricate. Schema:

{{
  "indication": "<concise indication, e.g. 'Type 2 diabetes' or 'COPD with eosinophilic phenotype'>",
  "modality": "<one of: small_molecule | biologic | antibody | cell_gene | rna | other>",
  "first_in_class": <true|false>,
  "addressable_population_us": <integer — US patients eligible for this drug at peak. e.g. 250000>,
  "addressable_population_global": <integer — global eligible patients>,
  "annual_cost_min_usd": <number — annual list price US, low end>,
  "annual_cost_max_usd": <number — annual list price US, high end>,
  "standard_of_care_cost_usd": <number — current standard of care annual cost in US, or null if none>,
  "penetration_min_pct": <number 0-100 — pessimistic peak market share %>,
  "penetration_max_pct": <number 0-100 — optimistic peak market share %>,
  "penetration_mid_pct": <number 0-100 — realistic mid-case peak market share %>,
  "launch_year": <integer — first commercial year if approved>,
  "peak_sales_year": <integer — year of peak sales>,
  "time_to_peak_years": <number — years from launch to peak (typical 4-7)>,
  "patent_expiry_date": "<YYYY-MM-DD or YYYY of LOE / patent cliff>",
  "loe_dropoff_pct": <number 0-1 — first-year revenue drop after LOE. 0.75 typical small-mol, 0.40 biologic>,
  "cogs_pct_estimate": <number 0-1 — COGS / revenue. small-mol: 0.10-0.15, biologic: 0.18-0.25, cell-gene: 0.35-0.50>,
  "commercial_success_prob": <number 0-1 — P(strong commercial success | approval). first-in-class unmet need: 0.75-0.90; me-too crowded: 0.30-0.55>,
  "competitors": [<list of named competitor drugs/companies, max 5>],
  "competitive_intensity": "<low | medium | high>",
  "key_risks": [<2-4 short risk bullets specific to this drug — pricing, pre-existing therapy, manufacturing, etc.>],
  "rationale": "<3-4 sentence summary justifying the numbers above. Cite specific epidemiology / pricing benchmarks where possible.>"
}}

CALCULATION SANITY CHECK (do this mentally before returning):
peak_sales_estimate = (addressable_population_global × penetration_mid_pct/100 × annual_cost_avg) / 1e9 → in $B
This should match what you'd quote as peak sales for this drug. If your structured numbers imply $20B peak but the drug realistically tops out at $3B, REVISE penetration or pricing or population down to be consistent.

Return ONLY the JSON object."""

    try:
        result, err = _call_llm_json(prompt, max_tokens=1800)
        if result is None:
            raise RuntimeError(f"All LLM providers failed: {err}")

        provider = result.pop("_llm_provider", "unknown") if isinstance(result, dict) else "unknown"

        # Compute derived peak_sales from structured fields, with safety
        try:
            pop = float(result.get("addressable_population_global") or 0)
            price_min = float(result.get("annual_cost_min_usd") or 0)
            price_max = float(result.get("annual_cost_max_usd") or 0)
            price_avg = (price_min + price_max) / 2.0 if (price_min and price_max) else (price_max or price_min or 0)
            pen = float(result.get("penetration_mid_pct") or 0) / 100.0
            peak_sales_b = (pop * pen * price_avg) / 1e9
            result["peak_sales_usd_b"] = round(peak_sales_b, 3)
        except Exception as e:
            logger.warning(f"peak sales derivation failed: {e}")
            result["peak_sales_usd_b"] = 0.0

        result["llm_provider"] = provider
        result["llm_rationale"] = result.pop("rationale", result.get("llm_rationale", ""))
        result["error"] = None
        return result

    except Exception as e:
        logger.warning(f"estimate_drug_economics_v2 failed for {ticker}/{drug}: {e}")
        return {
            "error": str(e)[:200],
            "indication": None,
            "modality": "other",
            "addressable_population_us": None,
            "addressable_population_global": None,
            "annual_cost_min_usd": None,
            "annual_cost_max_usd": None,
            "penetration_mid_pct": 15.0,
            "peak_sales_usd_b": 0.0,
            "commercial_success_prob": 0.5,
            "first_in_class": False,
            "competitive_intensity": "medium",
            "competitors": [],
            "key_risks": [],
            "loe_dropoff_pct": 0.6,
            "cogs_pct_estimate": 0.15,
            "time_to_peak_years": 5,
            "llm_rationale": f"Fallback (LLM failed: {str(e)[:100]})",
            "llm_provider": "fallback",
        }


def fetch_or_compute_drug_economics_v2(ticker: str, company_name: str,
                                        drug_name: str, catalyst_type: str,
                                        catalyst_date: str, description: str,
                                        market_cap_m: float,
                                        ai_context_sources: Optional[str] = None,
                                        force_refresh: bool = False) -> Dict:
    """Cache-first wrapper. Returns structured economics dict + adds
    `_from_cache: bool` and persists to drug_economics_cache when newly computed."""
    if not force_refresh:
        cached = get_drug_economics_v2_from_cache(ticker, drug_name)
        if cached:
            logger.info(f"drug_economics cache HIT: {ticker}/{drug_name}")
            return cached

    econ = estimate_drug_economics_v2(
        ticker=ticker, company_name=company_name, drug_name=drug_name,
        catalyst_type=catalyst_type, catalyst_date=catalyst_date,
        description=description, market_cap_m=market_cap_m,
        ai_context_sources=ai_context_sources,
    )
    # Persist if useful
    if not econ.get("error") and drug_name:
        write_drug_economics_v2_to_cache(ticker, drug_name, econ, ttl_days=7)
    econ["_from_cache"] = False
    return econ


# ============================================================
# TRUE rNPV — year-by-year cash flow discounting
# ============================================================

def compute_rnpv_full(econ_v2: Dict, p_approval: float,
                      market_cap_m: float = 0.0,
                      weights: Optional[Dict] = None) -> Dict:
    """True rNPV: build year-by-year revenue, apply COGS/tax, discount, drop at LOE.

    Inputs (from econ_v2):
      - addressable_population_global, penetration_mid_pct, annual_cost_avg
      - launch_year, time_to_peak_years, patent_expiry_date
      - cogs_pct_estimate, loe_dropoff_pct, commercial_success_prob

    Output dict:
      - rnpv_m: risk-adjusted NPV in $M (= deterministic_npv × p_approval × p_commercial)
      - deterministic_npv_m: undiscounted-by-prob NPV in $M
      - revenue_forecast: [{year, revenue_m, cash_flow_m, discount_factor, pv_m}, ...]
      - peak_sales_usd_b: peak annual revenue
      - assumptions_used: dict of all inputs used (for transparency)
      - error: None | string
    """
    w = load_npv_defaults_from_db()
    if weights:
        w = {**w, **weights}

    discount_rate = float(w.get("discount_rate", 0.12))
    tax_rate = float(w.get("tax_rate", 0.21))

    try:
        # Resolve inputs with sensible fallbacks
        pop_global = float(econ_v2.get("addressable_population_global") or 0)
        pop_us = float(econ_v2.get("addressable_population_us") or 0)
        # If global missing but US present, estimate global as 2.5x US
        if pop_global == 0 and pop_us > 0:
            pop_global = pop_us * 2.5

        price_min = float(econ_v2.get("annual_cost_min_usd") or 0)
        price_max = float(econ_v2.get("annual_cost_max_usd") or 0)
        price_avg = (price_min + price_max) / 2.0 if (price_min and price_max) else (price_max or price_min or 0)
        if price_avg == 0:
            return {"rnpv_m": 0.0, "deterministic_npv_m": 0.0, "revenue_forecast": [],
                    "peak_sales_usd_b": 0.0, "error": "no pricing data",
                    "assumptions_used": {}}

        pen_peak = float(econ_v2.get("penetration_mid_pct") or 15.0) / 100.0
        peak_revenue_usd = pop_global * pen_peak * price_avg
        peak_revenue_m = peak_revenue_usd / 1e6

        cogs_pct = float(econ_v2.get("cogs_pct_estimate") or w.get("cogs_pct", 0.15))
        cogs_pct = max(0.05, min(0.6, cogs_pct))  # sanity bounds

        time_to_peak = float(econ_v2.get("time_to_peak_years") or w.get("default_time_to_peak_years", 5))
        time_to_peak = max(2.0, min(10.0, time_to_peak))

        # Launch year
        now = datetime.now()
        launch_year = int(econ_v2.get("launch_year") or (now.year + 1))

        # LOE year — parse patent_expiry_date (YYYY-MM-DD or YYYY)
        loe_year = launch_year + 12  # default: 12 years post-launch
        pe = econ_v2.get("patent_expiry_date")
        if pe:
            try:
                if isinstance(pe, str):
                    loe_year = int(pe[:4])
            except Exception:
                pass

        loe_dropoff = float(econ_v2.get("loe_dropoff_pct") or 0.6)
        loe_dropoff = max(0.0, min(0.95, loe_dropoff))

        p_commercial = float(econ_v2.get("commercial_success_prob") or 0.6)
        p_commercial = max(0.0, min(1.0, p_commercial))

        # Build revenue curve year-by-year
        # Years: launch_year..launch_year + 20 (or until LOE+5)
        end_year = max(loe_year + 5, launch_year + 18)
        forecast = []
        cumulative_npv = 0.0

        for yr in range(launch_year, end_year + 1):
            t = yr - now.year  # years from today (can be 0 or negative if launch already happened)
            if t < 0:
                continue  # skip past years

            # S-curve penetration ramp: years_into_launch / time_to_peak
            yrs_into_launch = yr - launch_year
            if yrs_into_launch < 0:
                pen_this_year = 0.0
            elif yr < loe_year:
                # Pre-LOE: ramp up to peak
                ramp = min(1.0, yrs_into_launch / time_to_peak)
                # Light S-curve: x^1.3 for slower start
                pen_this_year = pen_peak * (ramp ** 1.3) if ramp < 1 else pen_peak
            else:
                # Post-LOE: revenue drops sharply, then continues at the lower level
                yrs_post_loe = yr - loe_year
                # Year 1 post-LOE: full drop; subsequent years: continue at residual
                residual_factor = (1.0 - loe_dropoff) * max(0.85, 1.0 - 0.05 * yrs_post_loe)
                pen_this_year = pen_peak * residual_factor

            revenue_m = (pop_global * pen_this_year * price_avg) / 1e6
            # Cash flow: revenue × (1 - cogs%) × (1 - tax_rate)
            # (Simplification: skip opex — biotech R&D is not allocable to a single drug well)
            ebit_m = revenue_m * (1.0 - cogs_pct)
            cash_flow_m = ebit_m * (1.0 - tax_rate)

            discount_factor = 1.0 / ((1.0 + discount_rate) ** t)
            pv_m = cash_flow_m * discount_factor
            cumulative_npv += pv_m

            forecast.append({
                "year": yr, "revenue_m": round(revenue_m, 1),
                "ebit_m": round(ebit_m, 1),
                "cash_flow_m": round(cash_flow_m, 1),
                "discount_factor": round(discount_factor, 4),
                "pv_m": round(pv_m, 1),
                "penetration_pct": round(pen_this_year * 100, 2),
            })

        deterministic_npv_m = cumulative_npv
        # rNPV = NPV × P(approval) × P(commercial | approval)
        rnpv_m = deterministic_npv_m * p_approval * p_commercial

        # Implied % move on stock = rnpv / market_cap
        fundamental_impact_pct = (rnpv_m / market_cap_m * 100.0) if market_cap_m > 0 else 0.0

        return {
            "rnpv_m": round(rnpv_m, 1),
            "deterministic_npv_m": round(deterministic_npv_m, 1),
            "peak_sales_usd_b": round(peak_revenue_m / 1000.0, 3),
            "fundamental_impact_pct": round(fundamental_impact_pct, 2),
            "revenue_forecast": forecast,
            "assumptions_used": {
                "addressable_population_global": pop_global,
                "annual_cost_avg_usd": price_avg,
                "penetration_peak_pct": pen_peak * 100,
                "launch_year": launch_year,
                "loe_year": loe_year,
                "loe_dropoff_pct": loe_dropoff,
                "time_to_peak_years": time_to_peak,
                "cogs_pct": cogs_pct,
                "tax_rate": tax_rate,
                "discount_rate": discount_rate,
                "p_approval": p_approval,
                "p_commercial": p_commercial,
                "modality": econ_v2.get("modality", "?"),
                "first_in_class": econ_v2.get("first_in_class", False),
            },
            "error": None,
        }
    except Exception as e:
        logger.exception(f"compute_rnpv_full error: {e}")
        return {
            "rnpv_m": 0.0, "deterministic_npv_m": 0.0,
            "peak_sales_usd_b": 0.0, "fundamental_impact_pct": 0.0,
            "revenue_forecast": [], "assumptions_used": {},
            "error": str(e)[:200],
        }


# ============================================================
# catalyst_npv_cache — cache final NPV by params hash
# ============================================================

def _params_hash(payload: Dict) -> str:
    """Stable hash of NPV inputs for cache key."""
    # Pick stable fields, ignore order
    keys_of_interest = sorted([
        "ticker", "catalyst_type", "catalyst_date", "drug_name",
        "discount_rate", "tax_rate", "cogs_pct", "p_approval",
        "annual_cost_min_usd", "annual_cost_max_usd",
        "addressable_population_global", "penetration_mid_pct",
        "patent_expiry_date", "loe_dropoff_pct", "time_to_peak_years",
    ])
    canonical = json.dumps({k: payload.get(k) for k in keys_of_interest}, default=str, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:24]


def get_npv_cached(ticker: str, catalyst_id: Optional[int], params_hash: str) -> Optional[Dict]:
    """Lookup catalyst_npv_cache. Returns None on miss/error."""
    if not ticker or not params_hash:
        return None
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            if catalyst_id is not None:
                cur.execute("""SELECT full_payload, computed_at, ttl, drug_npv_b
                               FROM catalyst_npv_cache
                               WHERE ticker=%s AND catalyst_id=%s AND params_hash=%s""",
                            (ticker, catalyst_id, params_hash))
            else:
                cur.execute("""SELECT full_payload, computed_at, ttl, drug_npv_b
                               FROM catalyst_npv_cache
                               WHERE ticker=%s AND catalyst_id IS NULL AND params_hash=%s""",
                            (ticker, params_hash))
            row = cur.fetchone()
            if not row:
                return None
            ttl = row[2]
            if ttl is not None:
                try:
                    if ttl < datetime.now().replace(tzinfo=ttl.tzinfo):
                        return None
                except Exception:
                    pass
            payload = row[0]
            if isinstance(payload, str):
                try: payload = json.loads(payload)
                except Exception: return None
            if isinstance(payload, dict):
                payload["_from_cache"] = True
            return payload
    except Exception as e:
        logger.warning(f"get_npv_cached failed: {e}")
        return None


def write_npv_cached(ticker: str, catalyst_id: Optional[int], params_hash: str,
                     payload: Dict, ttl_days: int = 1) -> bool:
    """UPSERT into catalyst_npv_cache."""
    if not ticker or not params_hash:
        return False
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        ttl = datetime.now() + timedelta(days=ttl_days)
        # Pull headline fields out of payload for indexed columns
        rnpv = payload.get("rnpv", {}) if isinstance(payload, dict) else {}
        drug_npv_b = (rnpv.get("rnpv_m") or 0) / 1000.0
        econ = payload.get("economics_v2", {}) if isinstance(payload, dict) else {}
        peak_sales_b = econ.get("peak_sales_usd_b", 0)
        p_approval = (rnpv.get("assumptions_used", {}) or {}).get("p_approval")
        p_commercial = (rnpv.get("assumptions_used", {}) or {}).get("p_commercial")
        expected_pct = rnpv.get("fundamental_impact_pct", 0)

        with db.get_conn() as conn:
            cur = conn.cursor()
            if catalyst_id is not None:
                cur.execute("""INSERT INTO catalyst_npv_cache
                    (ticker, catalyst_id, params_hash, drug_npv_b, p_approval, p_commercial,
                     peak_sales_b, expected_pct, full_payload, computed_at, ttl)
                    VALUES (%s,%s,%s, %s,%s,%s, %s,%s, %s::jsonb, NOW(), %s)
                    ON CONFLICT (ticker, catalyst_id, params_hash) DO UPDATE SET
                        drug_npv_b = EXCLUDED.drug_npv_b,
                        p_approval = EXCLUDED.p_approval,
                        p_commercial = EXCLUDED.p_commercial,
                        peak_sales_b = EXCLUDED.peak_sales_b,
                        expected_pct = EXCLUDED.expected_pct,
                        full_payload = EXCLUDED.full_payload,
                        computed_at = NOW(),
                        ttl = EXCLUDED.ttl""",
                    (ticker, catalyst_id, params_hash, drug_npv_b, p_approval, p_commercial,
                     peak_sales_b, expected_pct, json.dumps(payload, default=str), ttl))
            else:
                # Without catalyst_id, the unique constraint won't apply — insert without conflict handling
                cur.execute("""INSERT INTO catalyst_npv_cache
                    (ticker, catalyst_id, params_hash, drug_npv_b, p_approval, p_commercial,
                     peak_sales_b, expected_pct, full_payload, computed_at, ttl)
                    VALUES (%s, NULL, %s, %s,%s,%s, %s,%s, %s::jsonb, NOW(), %s)""",
                    (ticker, params_hash, drug_npv_b, p_approval, p_commercial,
                     peak_sales_b, expected_pct, json.dumps(payload, default=str), ttl))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"write_npv_cached failed: {e}")
        return False
