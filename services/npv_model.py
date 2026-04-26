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
        
        result, err = _call_llm_json(prompt, max_tokens=1200, feature="npv_legacy", ticker=ticker)
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

    # Clamp rejection_pct so we can never imply a negative stock price.
    # Floor at -90% — even Theranos-grade rejections rarely take a stock below
    # 10% of its pre-event price in a single day. Anything more negative is
    # mathematically valid but operationally unrealistic and confuses users.
    rejection_pct = max(rejection_pct, -90.0)

    # Final prices
    approval_price = current_price * (1 + remaining_upside_pct / 100.0)
    rejection_price = max(0.01, current_price * (1 + rejection_pct / 100.0))
    
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
        
        # Methodology transparency — flags known overlaps the user should account for
        "methodology_notes": [n for n in [
            ("Short interest is used in TWO places: (a) sec_short risk_factor "
             "discounts NPV; (b) high_short_amplifier multiplies post-event move size. "
             "Both reflect the same underlying signal — for high-short tickers, the "
             "displayed expected move may be amplified-then-discounted from the same input.")
            if risk_factor_breakdown and (risk_factor_breakdown.get("sec_short", 0) or 0) > 0 else None,
            ("Going concern flag captures financing risk. If you're using the per-share "
             "NPV view with dilution_assumed_pct, the dilution penalty addresses the same "
             "risk — choose ONE view to avoid double-discounting.")
            if risk_factor_breakdown and (risk_factor_breakdown.get("going_concern", 0) or 0) > 0 else None,
        ] if n],
        
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
                                  llm_rationale, llm_provider, computed_at, ttl, indication,
                                  annual_cost_us_net_usd, annual_cost_exus_net_usd,
                                  revenue_split_us_pct, provenance, confidence_score
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
                "annual_cost_us_net_usd": float(row[19]) if row[19] else None,
                "annual_cost_exus_net_usd": float(row[20]) if row[20] else None,
                "revenue_split_us_pct": float(row[21]) if row[21] else None,
                "provenance": row[22] if isinstance(row[22], dict) else (json.loads(row[22]) if row[22] else None),
                "confidence_score": float(row[23]) if row[23] is not None else None,
                "_from_cache": True,
            }
    except Exception as e:
        logger.warning(f"get_drug_economics_v2_from_cache failed: {e}")
        return None


def _compute_confidence_score(provenance: Dict) -> float:
    """Roll up per-field provenance into a single confidence score 0-1.
    
    Weighting:
      high   = 1.0
      medium = 0.6
      low    = 0.2
      missing/null field = 0.0
    
    Returns the average across critical fields. Used as a single 'how much
    should you trust this analysis' indicator.
    """
    if not provenance or not isinstance(provenance, dict):
        return 0.5  # neutral default — unknown
    
    # Fields that materially drive the rNPV math
    critical = [
        "addressable_population_us", "addressable_population_global",
        "annual_cost_us_net_usd", "annual_cost_exus_net_usd",
        "penetration_mid_pct", "patent_expiry_date",
        "p_event_occurs", "p_positive_outcome", "commercial_success_prob",
    ]
    SCORE = {"high": 1.0, "medium": 0.6, "low": 0.2}
    total = 0.0
    n = 0
    for f in critical:
        entry = provenance.get(f)
        if not entry or not isinstance(entry, dict):
            continue  # skip entirely missing — not an automatic fail
        conf = entry.get("confidence", "low")
        total += SCORE.get(conf.lower() if isinstance(conf, str) else "low", 0.2)
        n += 1
    return round(total / max(n, 1), 3) if n > 0 else 0.5


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
        provenance = econ_v2.get("provenance") or {}
        provenance_json = json.dumps(provenance) if not isinstance(provenance, str) else provenance
        # Roll up confidence: % of fields tagged 'high' or 'medium' / total
        conf_score = _compute_confidence_score(provenance) if provenance else None
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""INSERT INTO drug_economics_cache
                (ticker, canonical_drug_name, indication,
                 addressable_population_us, addressable_population_global,
                 annual_cost_min_usd, annual_cost_max_usd, standard_of_care_cost_usd,
                 annual_cost_us_net_usd, annual_cost_exus_net_usd, revenue_split_us_pct,
                 penetration_min_pct, penetration_max_pct, penetration_mid_pct,
                 launch_year, peak_sales_year, patent_expiry_date,
                 competitors, competitive_intensity, first_in_class,
                 llm_rationale, llm_provider,
                 provenance, confidence_score,
                 computed_at, ttl)
                VALUES (%s,%s,%s, %s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,
                        %s::jsonb,%s,%s, %s,%s,
                        %s::jsonb,%s,
                        NOW(), %s)
                ON CONFLICT (ticker, canonical_drug_name) DO UPDATE SET
                    indication = EXCLUDED.indication,
                    addressable_population_us = EXCLUDED.addressable_population_us,
                    addressable_population_global = EXCLUDED.addressable_population_global,
                    annual_cost_min_usd = EXCLUDED.annual_cost_min_usd,
                    annual_cost_max_usd = EXCLUDED.annual_cost_max_usd,
                    standard_of_care_cost_usd = EXCLUDED.standard_of_care_cost_usd,
                    annual_cost_us_net_usd = EXCLUDED.annual_cost_us_net_usd,
                    annual_cost_exus_net_usd = EXCLUDED.annual_cost_exus_net_usd,
                    revenue_split_us_pct = EXCLUDED.revenue_split_us_pct,
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
                    provenance = EXCLUDED.provenance,
                    confidence_score = EXCLUDED.confidence_score,
                    computed_at = NOW(),
                    ttl = EXCLUDED.ttl""",
                (ticker, canon, econ_v2.get("indication"),
                 econ_v2.get("addressable_population_us"),
                 econ_v2.get("addressable_population_global"),
                 econ_v2.get("annual_cost_min_usd"),
                 econ_v2.get("annual_cost_max_usd"),
                 econ_v2.get("standard_of_care_cost_usd"),
                 econ_v2.get("annual_cost_us_net_usd"),
                 econ_v2.get("annual_cost_exus_net_usd"),
                 econ_v2.get("revenue_split_us_pct"),
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
                 provenance_json,
                 conf_score,
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

CRITICAL: estimate the INCREMENTAL VALUE of THIS SPECIFIC CATALYST'S OUTCOME — not the entire drug franchise.

If this is a NEW INDICATION / LABEL EXPANSION for an already-approved drug (e.g. Dupixent for COPD when Dupixent is already approved for atopic dermatitis), return ONLY the additional patient population, pricing, and revenue from the NEW INDICATION. Do NOT include existing approved indications.

If this is a FIRST APPROVAL of a new drug, return the full drug economics.

If this is a Phase 3 readout (drug not yet approved), return the full drug economics if approved.

If this is an Earnings catalyst, return zeros for population/pricing — earnings doesn't have a per-drug NPV.

STOCK: {ticker} — {company_name}
DRUG: {drug}
INDICATION (from description): {description[:400]}
CATALYST: {catalyst_type} on {catalyst_date}
MARKET CAP: ${market_cap_m:,.0f}M ({cap_size})

{f'CONTEXT FROM RESEARCH:{chr(10)}{ai_context_sources[:2500]}' if ai_context_sources else ''}

Return ONLY a JSON object. No markdown, no preamble. Use realistic, evidence-based numbers.
If you genuinely don't know a field, set it to null. Don't fabricate. Schema:

{{
  "catalyst_scope": "<one of: 'first_approval' | 'new_indication' | 'label_expansion' | 'phase_readout' | 'earnings' | 'other'>",
  "indication": "<the SPECIFIC indication tied to THIS catalyst (not all approved indications)>",
  "modality": "<one of: small_molecule | biologic | antibody | cell_gene | rna | other>",
  "first_in_class": <true|false — for THIS specific indication, is this drug the first MOA approved?>,
  "addressable_population_us": <integer — US patients eligible for THIS indication only. Not the drug's total patients across all indications. e.g. for Dupixent COPD: ~500k eosinophilic COPD patients, NOT the 27M total Dupixent global patients across all approved uses>,
  "addressable_population_global": <integer — global patients for THIS indication only>,
  "annual_cost_min_usd": <number — annual list price for this indication, low end (US gross / WAC)>,
  "annual_cost_max_usd": <number — annual list price for this indication, high end (US gross / WAC)>,
  "annual_cost_us_net_usd": <number — annual NET realized price in the US AFTER rebates, GPO discounts, 340B, Medicaid best price, copay assistance. Typical gross-to-net: small molecule branded ~50-70%, biologic ~70-85%, orphan ~80-95%, gene therapy ~95% (one-time). Be explicit: do NOT assume net = gross.>,
  "annual_cost_exus_net_usd": <number — annual NET realized price ex-US (Europe + Japan + RoW blended). Typical: 50-65% of US net for established markets, lower for cash-pay markets. For US-only / orphan products: 0.>,
  "revenue_split_us_pct": <number 0-100 — share of global revenue from US. Typical: 60-75% for novel branded; 50-60% for global biologics with strong EU presence; 90-100% for US-orphan-only.>,
  "standard_of_care_cost_usd": <number — current SOC annual cost for THIS indication (US net), or null if no SOC>,
  "penetration_min_pct": <number 0-100 — pessimistic peak market share within THIS indication's eligible patients>,
  "penetration_max_pct": <number 0-100 — optimistic peak market share>,
  "penetration_mid_pct": <number 0-100 — realistic mid-case peak market share>,
  "launch_year": <integer — effective LAUNCH year for THIS catalyst's revenue. For new_indication / label_expansion, this is the year of the FDA decision (revenue starts then), NOT the drug's original launch year>,
  "peak_sales_year": <integer — year of peak INCREMENTAL sales from this catalyst>,
  "time_to_peak_years": <number — years from launch to peak (typical 4-7)>,
  "patent_expiry_date": "<YYYY-MM-DD or YYYY of LOE / patent cliff>",
  "loe_dropoff_pct": <number 0-1 — first-year revenue drop after LOE. 0.75 typical small-mol, 0.40 biologic>,
  "cogs_pct_estimate": <number 0-1 — COGS / revenue. small-mol: 0.10-0.15, biologic: 0.18-0.25, cell-gene: 0.35-0.50>,
  "commercial_success_prob": <number 0-1 — P(strong commercial uptake in THIS indication | approval). first-in-class unmet need: 0.75-0.90; me-too crowded: 0.30-0.55>,
  "p_event_occurs": <number 0-1 — P(this catalyst event actually happens on the stated date — readout reported / PDUFA decision rendered / etc). For a confirmed PDUFA: 0.95+. For an "expected late 2026" trial readout: 0.6-0.8. NOT the probability of success — just timing certainty.>,
  "p_positive_outcome": <number 0-1 — P(outcome is favorable | event occurs). For a Phase 3 readout in a well-validated MOA: 0.55-0.75. For a first-in-class novel target: 0.30-0.50. For a confirmatory Phase 3 with strong Phase 2: 0.65-0.80. This is what investors typically mean by "probability of approval".>,
  "competitors": [<list of named competitor drugs/companies in THIS indication, max 5>],
  "competitive_intensity": "<low | medium | high>",
  "key_risks": [<2-4 short risk bullets specific to this catalyst's outcome>],
  "rationale": "<3-4 sentence summary justifying the numbers above. Make clear whether you're scoping to a single indication or whole franchise. Cite specific epidemiology / pricing benchmarks where possible.>",
  "provenance": {{
    "<field_name>": {{
      "source": "<one of: 'openfda' | 'clinicaltrials_gov' | 'sec_edgar' | 'orange_book' | 'polygon_options' | 'finnhub' | 'llm_grounded_web' | 'llm_inference'>",
      "confidence": "<one of: 'high' | 'medium' | 'low'>",
      "citation": "<short string — e.g. 'HAEA 2024 prevalence estimate' or 'Takhzyro WAC list price 2024' or 'analyst inference'>"
    }},
    ...
  }}
}}

PROVENANCE RULES:
- Every numeric field above (population, pricing, penetration, dates, probabilities, COGS, LOE) MUST have a provenance entry.
- Use 'llm_grounded_web' for facts you found in the CONTEXT FROM RESEARCH section above.
- Use 'llm_inference' for values you reasoned from analogues, modality typicals, or standard biotech rules of thumb.
- Use 'openfda' / 'clinicaltrials_gov' / 'sec_edgar' / 'orange_book' ONLY when those sources are explicitly cited in the research context.
- 'high' confidence: you have a specific cited number from a reliable source.
- 'medium' confidence: defensible inference from analogues and benchmarks.
- 'low' confidence: rough estimate without strong supporting data — also returns 'llm_inference' as source.
- If you genuinely cannot estimate a field, set the field to null AND its provenance source to 'llm_inference' with confidence 'low' and citation 'no data'.

CALCULATION SANITY CHECK (do this mentally before returning):
peak_sales_estimate = (addressable_population_global × penetration_mid_pct/100 × annual_cost_avg) / 1e9 → in $B
- For Dupixent COPD label expansion: ~500k US + 1.5M global eosinophilic COPD × ~10% peak share × $40K = ~$6B peak (NOT $70B+)
- For a first-approval blockbuster small-mol: ~$2-5B peak typical
- For an orphan drug: $200M-$1B peak typical
If your structured numbers imply unrealistic peak (e.g. >$10B for label expansion), REVISE down.

Return ONLY the JSON object."""

    try:
        result, err = _call_llm_json(prompt, max_tokens=1800, feature="npv_v2", ticker=ticker)
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


def _fetch_grounded_research(ticker: str, drug_name: str, catalyst_type: str,
                              description: str) -> Optional[str]:
    """Optional Gemini-grounded research step that pulls real epidemiology +
    pricing data BEFORE the V2 economics LLM call, so the structured
    estimator has facts to anchor on rather than estimating from training.
    
    Returns a short research summary string suitable for ai_context_sources,
    or None if disabled/failed. Records to llm_usage with feature='npv_research'.
    
    Disabled if GROUNDED_NPV_RESEARCH=0. Default ON.
    """
    import os, time as _t
    if os.getenv("GROUNDED_NPV_RESEARCH", "1") == "0":
        return None
    if not drug_name or drug_name == "(unspecified)":
        return None
    
    prompt = f"""You are a biotech equity research analyst. For the upcoming catalyst below, do focused research and return a concise factual summary.

TICKER: {ticker}
DRUG: {drug_name}
CATALYST TYPE: {catalyst_type}
CATALYST CONTEXT: {description[:500]}

Use Google Search to find:
1. INDICATION-SPECIFIC EPIDEMIOLOGY — real prevalence/incidence numbers from medical literature, not estimates
2. PRICING BENCHMARKS — actual list prices of comparable approved drugs in the same indication (with sources)
3. CURRENT STANDARD OF CARE — what patients are getting today + cost
4. KEY COMPETITORS — drugs in development or approved targeting same indication
5. ADDRESSABLE MARKET SIZE — analyst estimates if available, with citation

Return a 200-400 word summary in plain text (no JSON, no markdown headers). Cite specific numbers with sources where possible. Format example:

"Hereditary angioedema affects 10,000-15,000 US patients (HAEA estimate, 2024) and ~40,000 globally. Standard-of-care prophylaxis: Takhzyro (lanadelumab, $700K/year wholesale), Orladeyo (berotralstat, $480K/year). Total US prophylactic market ~$3B annually. Lonvoguran ziclumeran is a one-time CRISPR therapy with potential for functional cure, expected pricing $2-3M lifetime (one-time). Competitors: Donidalorsen (Ionis, Phase 3) and BCX9930 (BioCryst, Phase 2). Analyst estimates 30-35% prophylactic-population capture at $400-500K equivalent annualized."

Be factual. If you can't find a number, omit it rather than guess."""
    
    try:
        from google import genai as google_genai
        from google.genai import types
        from services.llm_usage import record_usage
    except Exception as e:
        logger.warning(f"grounded_research import failed: {e}")
        return None
    
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not google_key:
        return None
    
    t0 = _t.time()
    try:
        client = google_genai.Client(api_key=google_key)
        config = types.GenerateContentConfig(
            max_output_tokens=1500,
            temperature=0.1,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        text = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        try:
            record_usage(
                provider="google", model="gemini-2.5-flash",
                feature="npv_research", ticker=ticker,
                tokens_input=getattr(usage, "prompt_token_count", 0) or 0 if usage else 0,
                tokens_output=getattr(usage, "candidates_token_count", 0) or 0 if usage else 0,
                duration_ms=int((_t.time() - t0) * 1000),
                status="success" if text else "error",
                error_message=None if text else "empty response",
            )
        except Exception:
            pass
        return text if text else None
    except Exception as e:
        try:
            record_usage(
                provider="google", model="gemini-2.5-flash",
                feature="npv_research", ticker=ticker,
                tokens_input=0, tokens_output=0,
                duration_ms=int((_t.time() - t0) * 1000),
                status="error", error_message=str(e)[:300],
            )
        except Exception:
            pass
        logger.warning(f"_fetch_grounded_research failed: {e}")
        return None


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

    # NEW: grounded research step — pulls real epidemiology + pricing
    # benchmarks via Gemini Search before the V2 estimator runs. This
    # closes the gap where V2 estimates were generic vs the rich detail-
    # page narrative. Skipped if no drug_name or env-disabled.
    research_context = ai_context_sources
    if not research_context:
        research_context = _fetch_grounded_research(
            ticker=ticker, drug_name=drug_name,
            catalyst_type=catalyst_type, description=description,
        )

    econ = estimate_drug_economics_v2(
        ticker=ticker, company_name=company_name, drug_name=drug_name,
        catalyst_type=catalyst_type, catalyst_date=catalyst_date,
        description=description, market_cap_m=market_cap_m,
        ai_context_sources=research_context,
    )
    # Persist research context alongside economics for transparency
    if research_context:
        econ["research_context"] = research_context[:2000]
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
        pop_exus = max(0.0, pop_global - pop_us)

        # ─── Pricing: prefer US/ex-US net split when LLM provides it ─────
        # Old path (gross × global) systematically overstates peak revenue by
        # ~30-50% because: (a) US gross WAC isn't realized after rebates;
        # (b) ex-US prices are typically 50-65% of US.
        price_us_net = econ_v2.get("annual_cost_us_net_usd")
        price_exus_net = econ_v2.get("annual_cost_exus_net_usd")
        revenue_split_us_pct = econ_v2.get("revenue_split_us_pct")

        # Backward-compat: if LLM didn't return the new fields, use old gross
        # × an implicit gross-to-net haircut to avoid overstatement.
        price_min = float(econ_v2.get("annual_cost_min_usd") or 0)
        price_max = float(econ_v2.get("annual_cost_max_usd") or 0)
        price_avg_gross = (price_min + price_max) / 2.0 if (price_min and price_max) else (price_max or price_min or 0)

        # Default gross-to-net by modality (used only if LLM didn't return net price)
        modality = (econ_v2.get("modality") or "other").lower()
        DEFAULT_GTN = {
            "small_molecule": 0.60,
            "biologic": 0.78,
            "antibody": 0.78,
            "cell_gene": 0.92,  # one-time gene therapy ~95%
            "rna": 0.78,
            "other": 0.70,
        }
        default_gtn = DEFAULT_GTN.get(modality, 0.70)

        if price_us_net is not None and float(price_us_net) > 0:
            price_us_net = float(price_us_net)
        elif price_avg_gross > 0:
            price_us_net = price_avg_gross * default_gtn
        else:
            price_us_net = 0

        if price_exus_net is not None and float(price_exus_net) > 0:
            price_exus_net = float(price_exus_net)
        elif price_us_net > 0:
            # Default ex-US net = 60% of US net (standard EU/Japan reference pricing)
            price_exus_net = price_us_net * 0.60
        else:
            price_exus_net = 0

        if price_us_net == 0 and price_exus_net == 0:
            return {"rnpv_m": 0.0, "deterministic_npv_m": 0.0, "revenue_forecast": [],
                    "peak_sales_usd_b": 0.0, "error": "no pricing data",
                    "assumptions_used": {}}

        pen_peak = float(econ_v2.get("penetration_mid_pct") or 15.0) / 100.0

        # Build peak revenue from US + ex-US separately when populations are known
        peak_revenue_us = pop_us * pen_peak * price_us_net if pop_us > 0 else 0
        peak_revenue_exus = pop_exus * pen_peak * price_exus_net if pop_exus > 0 else 0
        peak_revenue_usd = peak_revenue_us + peak_revenue_exus

        # Fallback: if pop_us is missing but global isn't, allocate by revenue_split_us_pct
        if peak_revenue_usd == 0 and pop_global > 0:
            split_us = float(revenue_split_us_pct) / 100.0 if revenue_split_us_pct else 0.65
            blended_price = price_us_net * split_us + price_exus_net * (1 - split_us)
            peak_revenue_usd = pop_global * pen_peak * blended_price

        peak_revenue_m = peak_revenue_usd / 1e6
        # For backward-compat in assumptions output
        price_avg = (price_us_net + price_exus_net) / 2.0 if (price_us_net and price_exus_net) else (price_us_net or price_exus_net)

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
        # rNPV with split probability (preferred): NPV × P(event) × P(positive | event) × P(commercial | positive)
        # When LLM provided split fields, use them. Otherwise fall back to combined p_approval × p_commercial.
        p_event_occurs = econ_v2.get("p_event_occurs")
        p_positive_outcome = econ_v2.get("p_positive_outcome")
        if p_event_occurs is not None and p_positive_outcome is not None:
            try:
                p_event_f = max(0.0, min(1.0, float(p_event_occurs)))
                p_pos_f = max(0.0, min(1.0, float(p_positive_outcome)))
                rnpv_m = deterministic_npv_m * p_event_f * p_pos_f * p_commercial
                rnpv_method = "split_probability"
            except (ValueError, TypeError):
                rnpv_m = deterministic_npv_m * p_approval * p_commercial
                rnpv_method = "combined_p_approval"
        else:
            rnpv_m = deterministic_npv_m * p_approval * p_commercial
            rnpv_method = "combined_p_approval"

        # ─── Bear/base/bull scenarios via penetration sensitivity ──────
        # Critical: LLM economics are interpolative estimates, not point
        # truth. Compute three scenarios so the UI can show a range.
        # We re-compute peak_revenue + scale rnpv linearly with peak_revenue
        # ratio (since the entire revenue curve is proportional to peak).
        pen_min = float(econ_v2.get("penetration_min_pct") or pen_peak * 100 * 0.5) / 100.0
        pen_max = float(econ_v2.get("penetration_max_pct") or pen_peak * 100 * 1.5) / 100.0
        # Cap at sane bounds — LLM sometimes returns 0 or 100
        pen_min = max(0.005, min(pen_peak, pen_min))
        pen_max = max(pen_peak, min(0.95, pen_max))
        scenarios = {
            "bear": {
                "rnpv_m": round(rnpv_m * (pen_min / pen_peak), 1) if pen_peak > 0 else 0.0,
                "peak_sales_usd_b": round(pop_global * pen_min * price_avg / 1e9, 3),
                "penetration_pct": round(pen_min * 100, 1),
                "label": "Pessimistic — low market share",
            },
            "base": {
                "rnpv_m": round(rnpv_m, 1),
                "peak_sales_usd_b": round(peak_revenue_m / 1000.0, 3),
                "penetration_pct": round(pen_peak * 100, 1),
                "label": "Base case — LLM mid estimate",
            },
            "bull": {
                "rnpv_m": round(rnpv_m * (pen_max / pen_peak), 1) if pen_peak > 0 else 0.0,
                "peak_sales_usd_b": round(pop_global * pen_max * price_avg / 1e9, 3),
                "penetration_pct": round(pen_max * 100, 1),
                "label": "Optimistic — high market share",
            },
        }

        # ─── Per-share NPV with optional dilution adjustment ───────────
        # If shares_outstanding_m is provided in weights/econ_v2, compute
        # per-share value of the rNPV. Optionally apply assumed dilution.
        shares_outstanding_m = float(econ_v2.get("shares_outstanding_m") or 0.0)
        dilution_assumed_pct = float(weights.get("dilution_assumed_pct", 0.0)) if weights else 0.0
        dilution_assumed_pct = max(0.0, min(75.0, dilution_assumed_pct))
        per_share_drug_npv = None
        per_share_after_dilution = None
        if shares_outstanding_m > 0:
            per_share_drug_npv = (rnpv_m * 1e6) / (shares_outstanding_m * 1e6)
            if dilution_assumed_pct > 0:
                # New share count after dilution
                new_shares_m = shares_outstanding_m * (1.0 + dilution_assumed_pct / 100.0)
                per_share_after_dilution = (rnpv_m * 1e6) / (new_shares_m * 1e6)

        # Implied % move on stock = rnpv / market_cap
        fundamental_impact_pct = (rnpv_m / market_cap_m * 100.0) if market_cap_m > 0 else 0.0

        return {
            "rnpv_m": round(rnpv_m, 1),
            "deterministic_npv_m": round(deterministic_npv_m, 1),
            "peak_sales_usd_b": round(peak_revenue_m / 1000.0, 3),
            "fundamental_impact_pct": round(fundamental_impact_pct, 2),
            "scenarios": scenarios,
            "per_share_drug_npv_usd": round(per_share_drug_npv, 2) if per_share_drug_npv else None,
            "per_share_after_dilution_usd": round(per_share_after_dilution, 2) if per_share_after_dilution else None,
            "shares_outstanding_m": shares_outstanding_m if shares_outstanding_m else None,
            "dilution_assumed_pct": dilution_assumed_pct if dilution_assumed_pct else None,
            "revenue_forecast": forecast,
            "assumptions_used": {
                "addressable_population_us": pop_us,
                "addressable_population_exus": pop_exus,
                "addressable_population_global": pop_global,
                "annual_cost_us_net_usd": price_us_net,
                "annual_cost_exus_net_usd": price_exus_net,
                "annual_cost_avg_usd": price_avg,
                "modality": econ_v2.get("modality", "?"),
                "default_gross_to_net_used": default_gtn if not econ_v2.get("annual_cost_us_net_usd") else None,
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
                "p_event_occurs": p_event_occurs,
                "p_positive_outcome": p_positive_outcome,
                "rnpv_method": rnpv_method,
                "first_in_class": econ_v2.get("first_in_class", False),
            },
            "caveats": [
                "Drug economics (population, pricing, penetration) are LLM estimates anchored to public research, NOT proprietary biotech data",
                "Per-share NPV is enterprise-level — does NOT reflect cash position, debt, or pipeline beyond this catalyst",
                "Bear/base/bull scenarios scale linearly with peak penetration; real downside in failure cases is total loss",
                "Discount rate, tax rate, and LOE drop-off use defaults — adjustable via NPV settings",
            ],
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
        "discount_rate", "tax_rate", "cogs_pct", "p_approval", "p_commercial",
        "annual_cost_min_usd", "annual_cost_max_usd",
        "addressable_population_global", "penetration_mid_pct",
        "patent_expiry_date", "loe_dropoff_pct", "time_to_peak_years",
        # Methodology audit additions — different inputs MUST yield
        # different cache keys, otherwise dilution/per-share variants
        # collide with the no-dilution base case.
        "dilution_assumed_pct", "shares_outstanding_m_override",
        "drug_name_override", "description_override",
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
