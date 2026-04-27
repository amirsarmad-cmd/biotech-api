"""
chat.py — context-aware Q&A about a specific stock's calculations.

User feedback: "I see 1980% upside but verdict says priced in. Doesn't make
sense. Build me a floating Ask AI chat that lets me critique calculations
and proposes system changes."

This endpoint takes a ticker and a question, builds a context bundle
containing ALL the data the system has computed for that ticker (rNPV,
setup_quality, materiality, risk factors, fundamentals), and calls Claude
with that as system prompt. The LLM:
  1. Explains calculations in plain English
  2. Walks through derivations when asked "why X?"
  3. Evaluates user critiques rigorously
  4. When a critique would change a calculation, outputs a JSON
     improvement_proposal block at the end of its reply that the FE
     can render as an "Implement this change" affordance.
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    ticker: str
    question: str
    history: List[ChatMessage] = []
    # Optional context override — if provided, used instead of fetching
    context: Optional[Dict[str, Any]] = None


class ImprovementProposal(BaseModel):
    """Structured suggestion the LLM can attach to its reply."""
    title: str
    target: str        # e.g. "services/setup_quality.py:_axis_runup"
    rationale: str     # why this change
    change_summary: str  # what to change (1-2 sentences)
    confidence: str    # 'high' | 'medium' | 'low'


class ChatResponse(BaseModel):
    reply: str
    improvement_proposal: Optional[ImprovementProposal] = None
    duration_ms: int
    model: str


SYSTEM_PROMPT = """You are a biotech investment analyst assistant integrated
into a stock screener. The user is on the detail page for {ticker}. They
can see calculations including rNPV fair value, setup quality, materiality,
catalyst probability, risks, and options-implied moves. They sometimes get
confused about what these numbers mean or want to critique the methodology.

Your job:

1. **Explain calculations in plain English.** When asked "why X%?" walk
   through the derivation step by step using the actual numbers in the
   context block below. Be specific.

2. **Surface time horizons.** rNPV fair value ≠ today's price target.
   It's the discounted PV of future cash flows over 8-15 years. Always
   mention horizons when discussing valuation. Approval timeline (BLA →
   FDA → launch) is 12-24 months from approval; peak sales 5-8 years.

3. **Distinguish the EVENT from the THESIS.** "Priced in" means the
   immediate binary catalyst is past; the long-term cash flow story
   (still ahead) is what makes the rNPV upside.

4. **Evaluate critiques rigorously.** If the user says "your run-up axis
   threshold is too low" or "you're missing X factor", actually think
   about whether they're right. Don't agree just to be agreeable.

5. **Propose system changes when warranted.** If a critique should result
   in a code change, after your prose reply, append a single line starting
   with `IMPROVEMENT_PROPOSAL:` followed by valid JSON with these fields:
       {{
         "title": "short headline",
         "target": "file path : function/section",
         "rationale": "why this is right (2-3 sentences)",
         "change_summary": "what to change (1-2 sentences)",
         "confidence": "high|medium|low"
       }}
   Only include this block when you genuinely believe a change is
   warranted. The user can click "implement" to have a developer apply it.

Style:
- Concise but rigorous. No hedging filler.
- Use the actual numbers from context, not generic ranges.
- Use markdown sparingly — just bullet points or short tables.
- Respond in the same language as the user's question.
- 4-8 short paragraphs maximum unless the user asks for depth.

CONTEXT (all data the system has for {ticker}):

```json
{context_json}
```

Today's date: {today}
"""


def _build_context(ticker: str) -> Dict[str, Any]:
    """Pull the full data bundle for a ticker.

    Mirrors what the FE displays. Includes rNPV (V2), setup_quality,
    materiality, risks, options_implied, fundamentals — everything the
    user can see on the detail page.
    """
    from services.database import BiotechDatabase
    ctx: Dict[str, Any] = {"ticker": ticker}

    db = BiotechDatabase()
    try:
        rows = db.get_stock(ticker) or []
        if rows:
            primary = sorted(rows, key=lambda r: r.get("probability") or 0, reverse=True)[0]
            ctx["company_name"] = primary.get("company_name")
            ctx["industry"] = primary.get("industry")
            ctx["primary_catalyst"] = {
                "type": primary.get("catalyst_type"),
                "date": primary.get("catalyst_date"),
                "probability": primary.get("probability"),
                "description": (primary.get("description") or "")[:400],
                "drug_name": primary.get("drug_name"),
                "indication": primary.get("indication"),
                "phase": primary.get("phase"),
                "sentiment_score": primary.get("sentiment_score"),
            }
            ctx["all_catalysts"] = [{
                "type": r.get("catalyst_type"),
                "date": r.get("catalyst_date"),
                "probability": r.get("probability"),
                "drug_name": r.get("drug_name"),
                "indication": r.get("indication"),
                "phase": r.get("phase"),
            } for r in rows[:10]]
    except Exception as e:
        logger.warning(f"chat _build_context db fetch failed for {ticker}: {e}")

    # Live price + market cap
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        ctx["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        ctx["market_cap_m"] = (info.get("marketCap") or 0) / 1e6 if info.get("marketCap") else None
        ctx["fundamentals"] = {
            "short_pct_of_float": info.get("shortPercentOfFloat"),
            "insider_held_pct": info.get("heldPercentInsiders"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "cash": info.get("totalCash"),
            "debt": info.get("totalDebt"),
        }
    except Exception as e:
        logger.warning(f"chat _build_context yfinance fetch failed for {ticker}: {e}")

    # Options-implied
    try:
        if ctx.get("primary_catalyst", {}).get("date"):
            from services.options_implied import get_implied_move_for_catalyst
            oi = get_implied_move_for_catalyst(
                ticker=ticker, catalyst_date=ctx["primary_catalyst"]["date"]
            )
            if oi:
                ctx["options_implied"] = {
                    "implied_move_pct": oi.get("implied_move_pct"),
                    "annualized_iv_pct": oi.get("annualized_iv_pct"),
                    "expiry": oi.get("expiry"),
                    "days_to_expiry": oi.get("days_to_expiry"),
                    "source": oi.get("source"),
                }
    except Exception as e:
        logger.info(f"chat _build_context options_implied failed for {ticker}: {e}")

    # rNPV V2 — pull from cache if available
    try:
        from services.cache import get_redis
        r = get_redis()
        if r:
            # The /analyze/npv route caches as "npv_v2:{ticker}:{cat_type}:{cat_date}:..."
            keys = r.keys(f"npv_v2:{ticker}:*") or []
            if keys:
                # Pull the most recent
                latest = max(keys, key=lambda k: r.ttl(k))
                npv_data = r.get(latest)
                if npv_data:
                    parsed = json.loads(npv_data)
                    ctx["rnpv_v2"] = {
                        "fair_value_per_share": (parsed.get("rnpv") or {}).get("fair_value_per_share"),
                        "rnpv_b": (parsed.get("rnpv") or {}).get("rnpv_b"),
                        "fundamental_impact_pct": (parsed.get("rnpv") or {}).get("fundamental_impact_pct"),
                        "upside_pct": (parsed.get("rnpv") or {}).get("upside_pct"),
                        "discount_rate": (parsed.get("rnpv") or {}).get("discount_rate"),
                        "drug_name": (parsed.get("economics_v2") or {}).get("drug_name"),
                        "peak_sales_b": (parsed.get("economics_v2") or {}).get("peak_sales_b"),
                        "p_commercial": (parsed.get("economics_v2") or {}).get("p_commercial"),
                        "confidence_score": (parsed.get("economics_v2") or {}).get("confidence_score"),
                    }
    except Exception as e:
        logger.info(f"chat _build_context rnpv_v2 cache fetch failed for {ticker}: {e}")

    # Setup quality
    try:
        from services.setup_quality import compute_setup_quality
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        h = t.history(period="3mo", auto_adjust=True)
        history_bars = []
        if not h.empty:
            for idx, row in h.iterrows():
                history_bars.append({
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10],
                    "close": float(row.get("Close", 0)) or None,
                })
        days_to = None
        try:
            from datetime import datetime as _dt, date as _date
            d_str = ctx.get("primary_catalyst", {}).get("date") or ""
            if d_str:
                d = _dt.strptime(d_str[:10], "%Y-%m-%d").date()
                days_to = max(0, (d - _date.today()).days)
        except Exception:
            pass
        ctx["setup_quality"] = compute_setup_quality(
            info=info, history=history_bars, fundamentals=None,
            options_implied=ctx.get("options_implied"),
            social_sentiment=ctx.get("primary_catalyst", {}).get("sentiment_score"),
            days_to_catalyst=days_to,
        )
    except Exception as e:
        logger.info(f"chat _build_context setup_quality failed for {ticker}: {e}")

    return ctx


def _parse_improvement_proposal(text: str) -> tuple[str, Optional[ImprovementProposal]]:
    """Extract IMPROVEMENT_PROPOSAL JSON if present. Returns (cleaned_text, proposal_or_None)."""
    marker = "IMPROVEMENT_PROPOSAL:"
    if marker not in text:
        return text, None
    idx = text.rfind(marker)
    prose = text[:idx].rstrip()
    json_part = text[idx + len(marker):].strip()
    # Strip markdown code fences if present
    if json_part.startswith("```"):
        json_part = json_part.split("\n", 1)[1] if "\n" in json_part else ""
        if json_part.endswith("```"):
            json_part = json_part.rsplit("```", 1)[0]
        json_part = json_part.strip()
        if json_part.startswith("json"):
            json_part = json_part[4:].strip()
    try:
        parsed = json.loads(json_part)
        return prose, ImprovementProposal(**parsed)
    except Exception as e:
        logger.info(f"chat: failed to parse IMPROVEMENT_PROPOSAL JSON: {e}")
        return text, None


@router.post("/chat/explain", response_model=ChatResponse)
async def chat_explain(req: ChatRequest):
    """Context-aware Q&A about the calculations on a stock's detail page."""
    t0 = time.time()
    ticker = req.ticker.upper().strip()

    # Build context if not provided
    ctx = req.context if req.context else _build_context(ticker)

    # Prune context to keep system prompt under control
    context_json = json.dumps(ctx, default=str, indent=2)
    if len(context_json) > 12000:
        # Trim oversized fields
        if "all_catalysts" in ctx and len(ctx["all_catalysts"]) > 5:
            ctx["all_catalysts"] = ctx["all_catalysts"][:5]
        context_json = json.dumps(ctx, default=str, indent=2)

    from datetime import date as _date
    system = SYSTEM_PROMPT.format(
        ticker=ticker,
        context_json=context_json,
        today=_date.today().isoformat(),
    )

    # Build messages from history + new question
    messages = []
    for m in req.history[-10:]:  # last 10 turns max
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.question})

    # Call Claude (using anthropic SDK directly so we can pass a system prompt)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), timeout=60.0)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=system,
            messages=messages,
        )
        text = msg.content[0].text if msg.content else ""
        prose, proposal = _parse_improvement_proposal(text)
        duration_ms = int((time.time() - t0) * 1000)
        # Log usage
        try:
            from services.llm_usage import record_call
            record_call(
                provider="anthropic", model="claude-sonnet-4-5",
                feature="chat_explain", ticker=ticker, status="success",
                tokens_input=getattr(msg.usage, "input_tokens", 0) if hasattr(msg, "usage") else 0,
                tokens_output=getattr(msg.usage, "output_tokens", 0) if hasattr(msg, "usage") else 0,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        return ChatResponse(
            reply=prose,
            improvement_proposal=proposal,
            duration_ms=duration_ms,
            model="claude-sonnet-4-5",
        )
    except Exception as e:
        logger.exception("chat_explain failed")
        raise HTTPException(500, f"chat_explain error: {e}")
