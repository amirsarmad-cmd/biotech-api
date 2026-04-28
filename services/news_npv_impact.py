"""
News × NPV Impact Analyzer — Deploy 16

Takes recent news articles + the current NPV calculation and asks an LLM:
  1. What events in recent news change our NPV assumptions?
  2. How much of the current priced-in move is already reflecting this news?
  3. What specific news items are bullish/bearish, and by how much (bps on NPV)?

Returns structured discounts/premiums that integrate INTO the NPV calc — so
a major pipeline win shows +5% NPV adjustment, a competitor approval shows
-8% NPV adjustment, etc.

Uses the same multi-provider fallback pattern as npv_model/risk_factors:
Claude → OpenAI → Google.
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Cache at module level; biotech-screener also caches via st.cache_data
_cache = {}
_CACHE_TTL = 6 * 3600  # 6h — news changes fast


def _cached(key: str, fn):
    now = time.time()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _CACHE_TTL:
            return val
    val = fn()
    _cache[key] = (now, val)
    return val


NEWS_NPV_PROMPT = """You are a senior biotech equity analyst. Given the drug NPV baseline and recent news, assess how the news should shift NPV assumptions.

=== DRUG CONTEXT ===
Ticker: {ticker}
Company: {company}
Catalyst: {catalyst_type} on {catalyst_date}
Current stock price: ${current_price:.2f}
Market cap: ${market_cap_b:.1f}B

=== NPV BASELINE (pre-news) ===
- Drug peak annual sales: ${peak_sales_b:.2f}B
- Revenue multiple: {multiple:.1f}x
- P(commercial success | approval): {p_commercial:.0%}
- Computed drug NPV: ${drug_npv_b:.2f}B
- Fundamental impact (NPV / market cap): {fundamental_impact_pct:.1f}%
- Already priced in (stock move vs {baseline_days}d ago): {implied_move_pct:+.1f}%

=== RECENT NEWS (sorted newest first, past 30 days) ===
{news_block}

=== YOUR TASK ===
Return a JSON object (ONLY the JSON, no other text) with this EXACT shape:

{{
  "summary": "2-3 sentence overall read of the news flow",
  "net_npv_adjustment_pct": -5.0,
  "priced_in_assessment": "describe how much of the news is likely already in the current price, and your confidence",
  "material_events": [
    {{
      "headline": "exact headline or paraphrase (≤140 chars)",
      "source": "e.g. Reuters, company PR",
      "date": "YYYY-MM-DD or approximate",
      "direction": "bullish" | "bearish" | "neutral",
      "npv_impact_pct": 3.5,
      "priced_in_pct": 60,
      "rationale": "1-2 sentence why this moves NPV by this amount, and why this % is/isnt priced in"
    }}
  ],
  "news_driven_probability_delta_pp": 2,
  "new_risks_flagged": ["risk 1", "risk 2"],
  "new_tailwinds_flagged": ["tailwind 1"],
  "hedge_suggestion": "optional — if one side of the trade looks obviously better, say so"
}}

RULES:
1. net_npv_adjustment_pct: signed percentage to apply to drug NPV (e.g. -5.0 means NPV down 5%, +3.2 means NPV up 3.2%). If no material news, 0.0.
2. material_events: up to 8 events. Only INCLUDE news that moves NPV by ≥1% in either direction. Ignore noise, reiterated guidance, boilerplate.
3. npv_impact_pct per event: signed. The sum of these SHOULD approximately equal net_npv_adjustment_pct (minor discrepancies OK for interactions).
4. priced_in_pct: 0-100. What % of this event's impact do you think is already reflected in the stock's {implied_move_pct:+.1f}% move?
5. news_driven_probability_delta_pp: PP change to P(approval). Integer -20 to +20. If news suggests FDA is more likely to approve, positive; if it suggests trouble, negative.
6. Be stock-specific. Do NOT make up events not in the news. Cite by source.
7. If news is sparse or all boilerplate, return net_npv_adjustment_pct: 0 and empty material_events.

Output JSON only."""


def _call_llm_json(prompt: str, max_tokens: int = 2500, temperature: float = 0.3,
                   feature: str = "news_impact", ticker: Optional[str] = None):
    """Try Claude → OpenAI → Google for JSON output. Returns (dict, err_str).
    Records every attempt to llm_usage table."""
    import time as _t
    
    def _record(provider, model, status, tokens_in, tokens_out, dur, err=None):
        try:
            from services.llm_usage import record_usage
            record_usage(provider=provider, model=model, feature=feature, ticker=ticker,
                         tokens_input=tokens_in, tokens_output=tokens_out,
                         duration_ms=dur, status=status, error_message=err)
        except Exception:
            pass
    
    # Try Claude
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=55.0)
            t0 = _t.time()
            resp = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=max_tokens, temperature=temperature,
                messages=[{"role":"user","content":prompt}])
            usage = getattr(resp, "usage", None)
            tin = getattr(usage, "input_tokens", 0) or 0 if usage else 0
            tout = getattr(usage, "output_tokens", 0) or 0 if usage else 0
            dur = int((_t.time() - t0) * 1000)
            txt = resp.content[0].text
            data = _parse_json(txt)
            if data:
                _record("anthropic", "claude-sonnet-4-5", "success", tin, tout, dur)
                return data, None
            _record("anthropic", "claude-sonnet-4-5", "parse_error", tin, tout, dur, "JSON parse failed")
    except Exception as e:
        logger.info(f"news_npv Claude: {e}")
        _record("anthropic", "claude-sonnet-4-5", "error", 0, 0, None, str(e)[:300])
    
    # Try OpenAI
    try:
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=55.0)
            t0 = _t.time()
            resp = client.chat.completions.create(
                model="gpt-4o", max_tokens=max_tokens, temperature=temperature,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":"You are a biotech equity analyst. Respond with JSON only."},
                    {"role":"user","content":prompt}])
            usage = getattr(resp, "usage", None)
            tin = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            tout = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            dur = int((_t.time() - t0) * 1000)
            txt = resp.choices[0].message.content
            data = _parse_json(txt)
            if data:
                _record("openai", "gpt-4o", "success", tin, tout, dur)
                return data, None
            _record("openai", "gpt-4o", "parse_error", tin, tout, dur, "JSON parse failed")
    except Exception as e:
        logger.info(f"news_npv OpenAI: {e}")
        _record("openai", "gpt-4o", "error", 0, 0, None, str(e)[:300])
    
    # Try Google Gemini
    try:
        if os.getenv("GOOGLE_API_KEY"):
            from google import genai as google_genai
            from google.genai import types as _gtypes
            client = google_genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY"),
                http_options=_gtypes.HttpOptions(timeout=50000),  # 50s, matches OAI/Anthropic
            )
            t0 = _t.time()
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{prompt}\n\nReturn ONLY valid JSON, no markdown, no preamble.")
            usage = getattr(resp, "usage_metadata", None)
            tin = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
            tout = getattr(usage, "candidates_token_count", 0) or 0 if usage else 0
            dur = int((_t.time() - t0) * 1000)
            txt = resp.text
            data = _parse_json(txt)
            if data:
                _record("google", "gemini-2.5-flash", "success", tin, tout, dur)
                return data, None
            _record("google", "gemini-2.5-flash", "parse_error", tin, tout, dur, "JSON parse failed")
    except Exception as e:
        logger.info(f"news_npv Gemini: {e}")
        _record("google", "gemini-2.5-flash", "error", 0, 0, None, str(e)[:300])
    
    return None, "All 3 LLM providers failed"


def _parse_json(txt: str) -> Optional[dict]:
    """Strip markdown fences, parse JSON, return dict or None."""
    if not txt:
        return None
    t = txt.strip()
    # Strip ```json ... ```
    if t.startswith("```"):
        t = t.split("```", 2)
        # rejoin ignoring language tag
        if len(t) >= 2:
            body = t[1]
            if body.startswith("json"):
                body = body[4:]
            # trim trailing backticks
            body = body.rstrip("` \n")
            t = body
        else:
            t = txt
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # Try finding the first { ... matching
        try:
            start = t.find("{")
            end = t.rfind("}")
            if start >= 0 and end > start:
                return json.loads(t[start:end+1])
        except Exception:
            pass
    return None


def _format_news_for_prompt(articles: List[Dict], max_articles: int = 15) -> str:
    """Format a news list compactly for the prompt."""
    if not articles:
        return "(no recent articles)"
    lines = []
    for i, a in enumerate(articles[:max_articles], 1):
        title = (a.get("title") or "")[:200]
        source = (a.get("source") or a.get("provider") or "unknown")[:40]
        date = (a.get("date") or a.get("published_on") or "?")[:10]
        summary = (a.get("summary") or a.get("body") or "")[:400]
        lines.append(f"{i}. [{date}] ({source}) {title}\n   {summary}")
    return "\n\n".join(lines)


def analyze_news_npv_impact(
    ticker: str,
    company_name: str,
    catalyst_type: str,
    catalyst_date: str,
    current_price: float,
    market_cap_b: float,
    drug_npv_b: float,
    peak_sales_b: float,
    multiple: float,
    p_commercial: float,
    fundamental_impact_pct: float,
    implied_move_pct: float,
    baseline_days: int,
    articles: List[Dict],
) -> Dict:
    """Run the news × NPV analysis.
    
    Returns dict with keys:
      summary, net_npv_adjustment_pct, priced_in_assessment,
      material_events, news_driven_probability_delta_pp,
      new_risks_flagged, new_tailwinds_flagged, hedge_suggestion,
      adjusted_drug_npv_b, adjusted_npv_impact_pct, _llm_provider, error
    """
    cache_key = f"news_npv:{ticker}:{catalyst_date}:{len(articles)}:{round(current_price,2)}"
    
    def _do():
        if not articles:
            return {
                "summary": "No recent articles available for news-driven NPV adjustment.",
                "net_npv_adjustment_pct": 0.0,
                "priced_in_assessment": "N/A (no news to evaluate)",
                "material_events": [],
                "news_driven_probability_delta_pp": 0,
                "new_risks_flagged": [],
                "new_tailwinds_flagged": [],
                "hedge_suggestion": None,
                "error": None,
                "_llm_provider": None,
            }
        
        news_block = _format_news_for_prompt(articles, max_articles=15)
        prompt = NEWS_NPV_PROMPT.format(
            ticker=ticker,
            company=company_name or ticker,
            catalyst_type=catalyst_type,
            catalyst_date=catalyst_date or "TBD",
            current_price=current_price,
            market_cap_b=market_cap_b,
            peak_sales_b=peak_sales_b,
            multiple=multiple,
            p_commercial=p_commercial,
            drug_npv_b=drug_npv_b,
            fundamental_impact_pct=fundamental_impact_pct,
            implied_move_pct=implied_move_pct,
            baseline_days=baseline_days,
            news_block=news_block,
        )
        
        data, err = _call_llm_json(prompt, max_tokens=2500, temperature=0.2, feature="news_impact", ticker=ticker)
        
        if data is None:
            return {
                "summary": f"News analysis unavailable: {err}",
                "net_npv_adjustment_pct": 0.0,
                "priced_in_assessment": "N/A (LLM unavailable)",
                "material_events": [],
                "news_driven_probability_delta_pp": 0,
                "new_risks_flagged": [],
                "new_tailwinds_flagged": [],
                "hedge_suggestion": None,
                "error": err or "Unknown LLM failure",
                "_llm_provider": None,
            }
        
        # Validate & clamp
        def _f(v, default=0.0):
            try: return float(v)
            except: return default
        
        def _i(v, default=0):
            try: return int(v)
            except: return default
        
        net_adj = _f(data.get("net_npv_adjustment_pct"), 0.0)
        # Clamp to sane range
        if net_adj > 50: net_adj = 50
        if net_adj < -50: net_adj = -50
        
        # Validate events
        events = []
        for ev in (data.get("material_events") or [])[:12]:
            if not isinstance(ev, dict): continue
            events.append({
                "headline": (ev.get("headline") or "")[:200],
                "source": (ev.get("source") or "unknown")[:40],
                "date": (ev.get("date") or "?")[:15],
                "direction": ev.get("direction") or "neutral",
                "npv_impact_pct": _f(ev.get("npv_impact_pct"), 0),
                "priced_in_pct": max(0, min(100, _i(ev.get("priced_in_pct"), 50))),
                "rationale": (ev.get("rationale") or "")[:500],
            })
        
        prob_delta = _i(data.get("news_driven_probability_delta_pp"), 0)
        if prob_delta > 20: prob_delta = 20
        if prob_delta < -20: prob_delta = -20
        
        # Derived fields
        adjusted_drug_npv_b = drug_npv_b * (1.0 + net_adj / 100.0)
        adjusted_npv_impact_pct = (
            adjusted_drug_npv_b / market_cap_b * 100.0 if market_cap_b > 0 else 0.0
        )
        
        return {
            "summary": (data.get("summary") or "")[:1000],
            "net_npv_adjustment_pct": net_adj,
            "priced_in_assessment": (data.get("priced_in_assessment") or "")[:600],
            "material_events": events,
            "news_driven_probability_delta_pp": prob_delta,
            "new_risks_flagged": (data.get("new_risks_flagged") or [])[:8],
            "new_tailwinds_flagged": (data.get("new_tailwinds_flagged") or [])[:8],
            "hedge_suggestion": (data.get("hedge_suggestion") or None),
            "adjusted_drug_npv_b": adjusted_drug_npv_b,
            "adjusted_npv_impact_pct": adjusted_npv_impact_pct,
            "error": None,
            "_llm_provider": "detected",  # we dont know which; _call_llm_json doesnt expose it back
        }
    
    return _cached(cache_key, _do)
