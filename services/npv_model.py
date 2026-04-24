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
    from risk_factors import estimate_risk_factors, apply_risk_discount
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
from llm_helper import call_llm_json as _call_llm_json


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
