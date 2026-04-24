"""
Adverse Risk Factor modeling.

Models 7 categories of risk that discount the drug NPV:
  1. Litigation (securities class actions, patent suits, product liability)
  2. FDA history (prior CRLs, withdrawals, REMS)
  3. SEC/short-seller reports (active investigations, public short positions)
  4. Insider selling velocity (C-suite selling > 2% float in last 90d)
  5. Going concern warnings (cash runway < 12 months)
  6. Patent cliff (top drugs losing protection in next 24 months)
  7. Governance (CEO/CFO turnover, auditor changes, board disputes)

Each returns 0-0.30 discount. Sum capped at 0.70 total.
One Claude call per ticker, 24h cached.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

# Import shared LLM helper (standalone module, no circular dependency risk)
try:
    from services.llm_helper import call_llm_json as _call_llm_json
except Exception as _e:
    _call_llm_json = None
    logging.getLogger(__name__).error(f"Could not import llm_helper: {_e}")

logger = logging.getLogger(__name__)


DEFAULT_RISK_WEIGHTS = {
    # Max discount per factor. User-editable (future deploy).
    "max_litigation":     0.25,
    "max_fda_history":    0.20,
    "max_sec_short":      0.25,
    "max_insider_sell":   0.15,
    "max_going_concern":  0.30,
    "max_patent_cliff":   0.20,
    "max_governance":     0.15,
    # Total cap — even if individual factors sum higher, cap at this
    "total_cap":          0.70,
}


def estimate_risk_factors(ticker: str, company_name: str, info: Dict = None,
                          news_excerpt: str = "", ai_context: str = "") -> Dict:
    """
    Uses Claude to estimate all 7 risk factor discounts.
    
    Inputs:
      - ticker, company_name
      - info: yfinance info dict (for cash, market cap, insider data if available)
      - news_excerpt: recent news headlines string (helps AI identify litigation/SEC items)
      - ai_context: additional context from prior research
    
    Returns:
    {
      "litigation": 0.05,
      "litigation_rationale": "No active securities class actions...",
      "fda_history": 0.0,
      "fda_history_rationale": "...",
      "sec_short": 0.0, "sec_short_rationale": "...",
      "insider_sell": 0.0, "insider_sell_rationale": "...",
      "going_concern": 0.0, "going_concern_rationale": "...",
      "patent_cliff": 0.0, "patent_cliff_rationale": "...",
      "governance": 0.0, "governance_rationale": "...",
      "total_discount": 0.05,  # capped sum
      "error": None | "error msg"
    }
    """
    info = info or {}
    
    # Extract quantitative signals
    cash = info.get("totalCash", 0) or 0
    total_rev = info.get("totalRevenue", 1) or 1
    burn_rate_m = abs(info.get("netIncomeToCommon", 0) or 0) / 4  # quarterly approx
    cash_runway_months = (cash / (burn_rate_m / 3)) if burn_rate_m > 0 else 99
    
    short_pct = (info.get("shortPercentOfFloat", 0) or 0) * 100
    insider_held = (info.get("heldPercentInsiders", 0) or 0) * 100
    
    try:
        prompt = f"""You are a biotech equity analyst assessing ADVERSE risk factors for {ticker} ({company_name}).

QUANTITATIVE SIGNALS:
- Cash on hand: ${cash/1e6:.1f}M
- Quarterly burn: ${burn_rate_m/1e6:.1f}M
- Estimated cash runway: {cash_runway_months:.0f} months
- Short interest: {short_pct:.1f}% of float
- Insider ownership: {insider_held:.1f}%
- Total TTM revenue: ${total_rev/1e6:.1f}M

RECENT NEWS EXCERPT (if provided):
{news_excerpt[:2000] if news_excerpt else '(none)'}

{f'ADDITIONAL CONTEXT:{chr(10)}{ai_context[:1500]}' if ai_context else ''}

Assess each of 7 adverse risk factors. Return discount (0.0 to 0.30) that should be applied to drug NPV for that factor. Return JSON, no markdown fences:

{{
  "litigation": <0.0 - 0.25, active suits / class actions / product liability discount>,
  "litigation_rationale": "<1 sentence>",
  
  "fda_history": <0.0 - 0.20, prior CRLs on this drug, withdrawal history, REMS restrictions>,
  "fda_history_rationale": "<1 sentence>",
  
  "sec_short": <0.0 - 0.25, SEC investigations, major short-seller reports like Hindenburg/Citron>,
  "sec_short_rationale": "<1 sentence>",
  
  "insider_sell": <0.0 - 0.15, C-suite large sales in last 90d, pattern of selling>,
  "insider_sell_rationale": "<1 sentence>",
  
  "going_concern": <0.0 - 0.30, cash runway < 12 months, auditor going concern warning>,
  "going_concern_rationale": "<1 sentence>",
  
  "patent_cliff": <0.0 - 0.20, top drugs losing patent protection in next 24 months>,
  "patent_cliff_rationale": "<1 sentence>",
  
  "governance": <0.0 - 0.15, CEO/CFO turnover last 12m, auditor changes, board disputes>,
  "governance_rationale": "<1 sentence>"
}}

GUIDANCE:
- Use 0.0 when factor is not a concern (most common for healthy biotechs)
- Use 0.05-0.10 for minor concerns (rumors, one small suit)
- Use 0.15-0.20 for material concerns (active class action, prior CRL on same drug, runway < 6mo)
- Use 0.25-0.30 only for severe issues (active SEC probe, imminent bankruptcy, major fraud allegations)
- Be skeptical: if you don't know of specific concerns, return 0.0 not default small numbers
- Biotechs with strong cash + no recent news → usually all 0.0 except maybe small patent_cliff

Return ONLY the JSON object."""
        
        if _call_llm_json is None:
            raise RuntimeError("LLM helper unavailable")
        result, err = _call_llm_json(prompt, max_tokens=1500)
        if result is None:
            raise RuntimeError(f"All LLM providers failed: {err}")
        
        # Compute total_discount with caps
        factors = ["litigation", "fda_history", "sec_short", "insider_sell",
                   "going_concern", "patent_cliff", "governance"]
        total = 0.0
        for f in factors:
            val = float(result.get(f, 0) or 0)
            # Clamp to max per factor
            max_allowed = DEFAULT_RISK_WEIGHTS.get(f"max_{f}", 0.20)
            val = max(0.0, min(val, max_allowed))
            result[f] = val
            total += val
        
        # Total cap
        result["total_discount"] = min(total, DEFAULT_RISK_WEIGHTS["total_cap"])
        result["error"] = None
        return result
    
    except Exception as e:
        logger.warning(f"estimate_risk_factors failed for {ticker}: {e}")
        # Fallback: pure quantitative signals (no AI)
        fallback = _quant_fallback(cash_runway_months, short_pct, insider_held)
        fallback["error"] = str(e)[:200]
        return fallback


def _quant_fallback(cash_runway_months: float, short_pct: float, insider_held: float) -> Dict:
    """Fallback when AI unavailable — use quantitative signals only."""
    result = {
        "litigation": 0.0, "litigation_rationale": "AI unavailable — factor unchecked",
        "fda_history": 0.0, "fda_history_rationale": "AI unavailable",
        "sec_short": 0.0, "sec_short_rationale": "AI unavailable",
        "insider_sell": 0.0, "insider_sell_rationale": "AI unavailable",
        "going_concern": 0.0, "going_concern_rationale": "AI unavailable",
        "patent_cliff": 0.0, "patent_cliff_rationale": "AI unavailable",
        "governance": 0.0, "governance_rationale": "AI unavailable",
    }
    
    # Going concern: cash runway heuristic
    if cash_runway_months < 6:
        result["going_concern"] = 0.30
        result["going_concern_rationale"] = f"Cash runway ~{cash_runway_months:.0f} months — severe"
    elif cash_runway_months < 12:
        result["going_concern"] = 0.15
        result["going_concern_rationale"] = f"Cash runway ~{cash_runway_months:.0f} months — concern"
    elif cash_runway_months < 18:
        result["going_concern"] = 0.05
        result["going_concern_rationale"] = f"Cash runway ~{cash_runway_months:.0f} months — watch"
    
    # High short interest often signals institutional distrust
    if short_pct > 30:
        result["sec_short"] = 0.10
        result["sec_short_rationale"] = f"High short interest {short_pct:.0f}% signals market distrust"
    elif short_pct > 20:
        result["sec_short"] = 0.05
        result["sec_short_rationale"] = f"Elevated short interest {short_pct:.0f}%"
    
    total = sum(result[f] for f in ["litigation","fda_history","sec_short","insider_sell",
                                     "going_concern","patent_cliff","governance"])
    result["total_discount"] = min(total, 0.70)
    return result


def apply_risk_discount(drug_npv_m: float, risk_factors: Dict) -> Dict:
    """Apply total risk discount to drug NPV."""
    total = risk_factors.get("total_discount", 0)
    adjusted_npv = drug_npv_m * (1 - total)
    return {
        "original_npv_m": drug_npv_m,
        "adjusted_npv_m": adjusted_npv,
        "total_discount": total,
        "discount_pct": total * 100,
        "factor_breakdown": {
            k: {"discount": risk_factors.get(k, 0), "rationale": risk_factors.get(f"{k}_rationale", "")}
            for k in ["litigation","fda_history","sec_short","insider_sell",
                     "going_concern","patent_cliff","governance"]
        },
    }
