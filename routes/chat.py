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
import redis as _redis_mod

logger = logging.getLogger(__name__)
router = APIRouter()


# Cached Redis client. Used to memoize the context bundle so that follow-up
# turns in the same conversation don't re-pay 10-15 s of yfinance + backtest
# aggregation work on every message.
_redis_client = None


def _redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        _redis_client = _redis_mod.from_url(url, decode_responses=True)
    except Exception as e:
        logger.warning(f"chat: redis init failed: {e}")
        _redis_client = None
    return _redis_client


# TTL covers a typical chat session's follow-up questions without forcing
# yfinance + aggregate work on every turn. Short enough that price/IV
# numbers don't go visibly stale.
_CTX_CACHE_TTL_S = 300


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

   The "System-wide backtest scoreboard" card on the page surfaces how
   well the system has predicted past catalysts. It is in the context
   bundle as:
     - `backtest_scoreboard_v2` — three-tier breakdown:
         all_events (raw 30D direction across every backfilled catalyst,
         ~50% noise floor) vs tradeable_events (3D abnormal-vs-XBI on
         the high-confidence subset, the actual model edge) plus
         coverage (% of all events that became tradeable signals).
         The actionable target is tradeable accuracy ≥ 65-70% with
         coverage 25-40%.
     - `backtest_scoreboard_v3` — V1 (probability-bias only) vs V2
         (priced-in-aware: LONG_UNDERPRICED_POSITIVE / SHORT_SELL_THE_NEWS
         / SHORT_LOW_PROBABILITY) head-to-head with 95% Wilson CIs,
         and per-bucket accuracy. In-sample. `production_ready=True`
         on a bucket means n_judged ≥ 50 AND CI lower bound > 55%.
     - `backtest_oos` — out-of-sample: prediction snapshots frozen
         before outcomes were known, scored against actual abnormal_3d
         once they become judgeable. This is the unbiased number; if
         days_of_oos_data is small, treat it as preliminary.
     - `post_catalyst_history` — this ticker's last 5 catalysts with
         predicted vs actual moves and the inferred outcome label.
   Use these numbers verbatim if asked. Do NOT say "I don't have data
   on the backtest scoreboard" when these fields are populated.

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

   CRITICAL — read this before emitting a proposal:
   - You do NOT have access to the codebase. You cannot see file names,
     directory layout, function names, or imports. Do NOT invent file
     paths. Generic guesses like `frontend/components/foo_bar.tsx` or
     `backend/utils/calc.py` are almost always wrong and waste the
     developer's time.
   - Only emit IMPROVEMENT_PROPOSAL when (a) the user has named a
     specific file or function in this conversation, OR (b) the context
     block above contains an explicit module/function reference you can
     cite verbatim. Otherwise, put the suggestion in prose: "A developer
     should review the [methodology / threshold / weight] for X" — no
     JSON block.
   - Confidence "high" requires concrete evidence from the context
     numbers contradicting the calculation. "I think the UI might be
     wrong" is not evidence — it's speculation, and speculation is
     never high confidence.
   - When in doubt, omit the proposal. A missing proposal is fine. A
     wrong proposal makes the user distrust the whole feature.

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


async def _build_context(ticker: str) -> Dict[str, Any]:
    """Pull the full data bundle for a ticker.

    Mirrors what the FE displays. Includes rNPV (V2), setup_quality,
    materiality, risks, options_implied, fundamentals, plus the per-ticker
    post-catalyst history and the system-wide backtest scoreboard
    (aggregate v2, aggregate v3 with V1/V2 priced-in CIs, OOS validation)
    that the user can see on the detail page.
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

    # Live price + market cap. yf_ticker / yf_info are reused by the
    # setup_quality block below — previously we paid for `t.info` twice.
    yf_ticker = None
    yf_info: Dict[str, Any] = {}
    try:
        import yfinance as yf
        yf_ticker = yf.Ticker(ticker)
        yf_info = yf_ticker.info or {}
        ctx["current_price"] = yf_info.get("currentPrice") or yf_info.get("regularMarketPrice")
        ctx["market_cap_m"] = (yf_info.get("marketCap") or 0) / 1e6 if yf_info.get("marketCap") else None
        ctx["fundamentals"] = {
            "short_pct_of_float": yf_info.get("shortPercentOfFloat"),
            "insider_held_pct": yf_info.get("heldPercentInsiders"),
            "fifty_two_week_high": yf_info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": yf_info.get("fiftyTwoWeekLow"),
            "shares_outstanding": yf_info.get("sharesOutstanding"),
            "cash": yf_info.get("totalCash"),
            "debt": yf_info.get("totalDebt"),
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
        r = _redis()
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

    # Setup quality — reuses the yfinance handle/info fetched above instead
    # of re-paying the ~2-4 s `.info` round-trip a second time.
    try:
        from services.setup_quality import compute_setup_quality
        if yf_ticker is None:
            raise RuntimeError("yfinance unavailable (info fetch failed earlier)")
        h = yf_ticker.history(period="3mo", auto_adjust=True)
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
            info=yf_info, history=history_bars, fundamentals=None,
            options_implied=ctx.get("options_implied"),
            social_sentiment=ctx.get("primary_catalyst", {}).get("sentiment_score"),
            days_to_catalyst=days_to,
        )
    except Exception as e:
        logger.info(f"chat _build_context setup_quality failed for {ticker}: {e}")

    # Per-ticker post-catalyst history — table the user sees at the
    # bottom of "Post-Catalyst History". Cap at 5 rows; older ones rarely
    # matter for the conversation.
    try:
        from services.post_catalyst_tracker import get_outcomes_for_ticker
        rows = get_outcomes_for_ticker(ticker, limit=5) or []
        if rows:
            ctx["post_catalyst_history"] = [
                {
                    "catalyst_date": (r.get("catalyst_date").isoformat()
                                      if hasattr(r.get("catalyst_date"), "isoformat")
                                      else r.get("catalyst_date")),
                    "catalyst_type": r.get("catalyst_type"),
                    "drug_name": r.get("drug_name"),
                    "predicted_move_pct": (float(r["predicted_move_pct"])
                                           if r.get("predicted_move_pct") is not None else None),
                    "actual_move_pct_1d": (float(r["actual_move_pct_1d"])
                                           if r.get("actual_move_pct_1d") is not None else None),
                    "actual_move_pct_30d": (float(r["actual_move_pct_30d"])
                                            if r.get("actual_move_pct_30d") is not None else None),
                    "outcome": r.get("outcome"),
                    "direction_correct": r.get("direction_correct"),
                    "error_abs_pct": (float(r["error_abs_pct"])
                                      if r.get("error_abs_pct") is not None else None),
                }
                for r in rows
            ]
    except Exception as e:
        logger.info(f"chat _build_context post_catalyst_history failed for {ticker}: {e}")

    # System-wide backtest scoreboard — mirrors the ThreeTierScoreboard
    # + V2 priced-in classifier + OOS cards on the page. Without this
    # the AskAI chat says "I don't have any data about a System-wide
    # backtest scoreboard" — the data exists, the chat just couldn't see it.
    # Slim each section so we stay under the 12 KB context ceiling.
    try:
        from routes.admin import (
            post_catalyst_aggregate_v2 as _agg_v2_handler,
            post_catalyst_aggregate_v3 as _agg_v3_handler,
            oos_aggregate as _oos_handler,
        )
        try:
            agg_v2 = await _agg_v2_handler()
            if agg_v2:
                all_e = agg_v2.get("all_events") or {}
                tr_e = agg_v2.get("tradeable_events") or {}
                ctx["backtest_scoreboard_v2"] = {
                    "all_events": {
                        "count": all_e.get("count"),
                        "direction_accuracy_pct": all_e.get("direction_accuracy_pct"),
                        "_target": all_e.get("_target"),
                    },
                    "tradeable_events": {
                        "count": tr_e.get("count"),
                        "direction_accuracy_pct": tr_e.get("direction_accuracy_pct"),
                        "coverage_pct": tr_e.get("coverage_pct"),
                        "_target": tr_e.get("_target"),
                    },
                    "interpretation": agg_v2.get("interpretation"),
                }
        except Exception as e:
            logger.info(f"chat _build_context aggregate_v2 failed: {e}")

        try:
            agg_v3 = await _agg_v3_handler(min_outcome_confidence=0.7)
            if agg_v3:
                v1 = agg_v3.get("tradeable_v1") or {}
                v2 = agg_v3.get("tradeable_v2") or {}
                tradeable_signals = (
                    "LONG_UNDERPRICED_POSITIVE",
                    "SHORT_SELL_THE_NEWS",
                    "SHORT_LOW_PROBABILITY",
                    "LONG",
                    "SHORT",
                )
                ctx["backtest_scoreboard_v3"] = {
                    "tradeable_v1": {
                        "judged": v1.get("judged"),
                        "direction_accuracy_pct": v1.get("direction_accuracy_pct"),
                        "ci_95_pct": v1.get("ci_95_pct"),
                        "coverage_pct": v1.get("coverage_pct"),
                    },
                    "tradeable_v2": {
                        "judged": v2.get("judged"),
                        "direction_accuracy_pct": v2.get("direction_accuracy_pct"),
                        "ci_95_pct": v2.get("ci_95_pct"),
                        "coverage_pct": v2.get("coverage_pct"),
                    },
                    "v2_buckets": [
                        {
                            "signal": b.get("signal"),
                            "count": b.get("count"),
                            "judged": b.get("judged"),
                            "direction_accuracy_pct": b.get("direction_accuracy_pct"),
                            "ci_95_pct": b.get("ci_95_pct"),
                            "production_ready": b.get("production_ready"),
                        }
                        for b in (agg_v3.get("v2_buckets") or [])
                        if b.get("signal") in tradeable_signals
                    ],
                    "interpretation": agg_v3.get("interpretation"),
                }
        except Exception as e:
            logger.info(f"chat _build_context aggregate_v3 failed: {e}")

        try:
            oos = await _oos_handler(signal_version="v2")
            if oos:
                ctx["backtest_oos"] = {
                    "tradeable_total": oos.get("tradeable_total"),
                    "evaluated": oos.get("evaluated"),
                    "judged": oos.get("judged"),
                    "direction_accuracy_pct": oos.get("direction_accuracy_pct"),
                    "ci_95_pct": oos.get("ci_95_pct"),
                    "days_of_oos_data": oos.get("days_of_oos_data"),
                    "buckets": [
                        {
                            "signal": b.get("signal"),
                            "judged": b.get("judged"),
                            "direction_accuracy_pct": b.get("direction_accuracy_pct"),
                        }
                        for b in (oos.get("buckets") or [])
                        if b.get("signal") in (
                            "LONG_UNDERPRICED_POSITIVE",
                            "SHORT_SELL_THE_NEWS",
                            "SHORT_LOW_PROBABILITY",
                            "LONG",
                            "SHORT",
                        )
                    ],
                }
        except Exception as e:
            logger.info(f"chat _build_context oos_aggregate failed: {e}")
    except Exception as e:
        logger.info(f"chat _build_context backtest scoreboard import failed: {e}")

    return ctx


_REPO_ROOTS = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # biotech-api
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "biotech-frontend",
    ),
]


def _proposal_target_exists(target: str) -> bool:
    """Check whether the file portion of a `path : symbol` target resolves
    against either repo root. Models without codebase access tend to
    hallucinate plausible-sounding paths; rejecting bad targets keeps
    those proposals from reaching the user."""
    if not target:
        return False
    file_part = target.split(":", 1)[0].strip().replace("\\", "/")
    if not file_part:
        return False
    for root in _REPO_ROOTS:
        if os.path.isfile(os.path.join(root, *file_part.split("/"))):
            return True
    return False


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
        proposal = ImprovementProposal(**parsed)
    except Exception as e:
        logger.info(f"chat: failed to parse IMPROVEMENT_PROPOSAL JSON: {e}")
        return text, None
    if not _proposal_target_exists(proposal.target):
        logger.info(
            f"chat: dropped proposal with non-existent target {proposal.target!r}"
        )
        return prose, None
    return prose, proposal


@router.post("/chat/explain", response_model=ChatResponse)
async def chat_explain(req: ChatRequest):
    """Context-aware Q&A about the calculations on a stock's detail page."""
    t0 = time.time()
    ticker = req.ticker.upper().strip()

    # Build context if not provided. The bundle is cached per-ticker so
    # follow-up turns (which dominate a typical chat session) skip the
    # ~10-15 s of yfinance + backtest aggregation work.
    ctx: Optional[Dict[str, Any]] = req.context
    if ctx is None:
        r = _redis()
        cache_key = f"chat_ctx:{ticker}"
        if r is not None:
            try:
                cached = r.get(cache_key)
                if cached:
                    ctx = json.loads(cached)
            except Exception as e:
                logger.info(f"chat_explain ctx cache get failed for {ticker}: {e}")
        if ctx is None:
            ctx = await _build_context(ticker)
            if r is not None:
                try:
                    r.setex(cache_key, _CTX_CACHE_TTL_S, json.dumps(ctx, default=str))
                except Exception as e:
                    logger.info(f"chat_explain ctx cache set failed for {ticker}: {e}")

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

    # Route through the universal LLM gateway: rotates keys within
    # Anthropic, falls through to OpenAI then Google if Anthropic is
    # rate-limited or down. Telemetry recorded automatically.
    try:
        from services.llm_gateway import llm_call, LLMAllProvidersFailed
        result = llm_call(
            capability="text_freeform",
            feature="chat_explain",
            ticker=ticker,
            system=system,
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
            timeout_s=60.0,
        )
        prose, proposal = _parse_improvement_proposal(result.text)
        return ChatResponse(
            reply=prose,
            improvement_proposal=proposal,
            duration_ms=result.duration_ms,
            model=f"{result.provider}:{result.model}",
        )
    except LLMAllProvidersFailed as e:
        logger.warning(f"chat_explain all providers failed: {e.attempts}")
        raise HTTPException(503, "AI service temporarily unavailable — please retry")
    except Exception as e:
        logger.exception("chat_explain failed")
        raise HTTPException(500, f"chat_explain error: {e}")
