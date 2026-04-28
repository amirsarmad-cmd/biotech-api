"""Outcome labeler — produces structured labels from press release text.

ChatGPT critique: "Right now `outcome` is inferred by `>20%/<−20%` price
action heuristic. If we want to evaluate the event-outcome prediction
separately from the stock-reaction prediction, we need real labels
(approved/rejected, hit-primary-endpoint/missed)."

This module uses Gemini 2.5 Flash with Google Search grounding to find
the press release for a given catalyst and extract structured outcome
labels independent of price action. Cost: ~$0.0003 per call.

Output schema (stored in catalyst_outcomes.outcome_labeled_json):
  outcome_class       — APPROVED / REJECTED / MET_ENDPOINT / MISSED_ENDPOINT /
                        DELAYED / WITHDRAWN / MIXED / UNKNOWN
  endpoint_met        — bool | null (Phase 2/3 readouts only)
  approval_granted    — bool | null (FDA/regulatory only)
  side_effects_flag   — bool (any safety signal mentioned)
  primary_source_url  — URL of the press release / 8-K / FDA letter
  evidence            — short verbatim quote from source
  confidence          — 0..1, how confident the labeler is
  labeled_at          — timestamp
"""
from __future__ import annotations
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("OUTCOME_LABELER_MODEL", "gemini-2.5-flash")


OUTCOME_LABELER_PROMPT = """You are a biotech outcome classifier. Given a catalyst event, find the press release or news article about its outcome and return a STRICT JSON object describing what happened.

CATALYST:
  Ticker: {ticker}
  Company: {company}
  Catalyst type: {catalyst_type}
  Catalyst date: {catalyst_date}
  Drug: {drug}
  Indication: {indication}

Search Google for the press release, 8-K filing, FDA decision letter, or news article published on or near {catalyst_date} that announced the outcome. Look for:
  - Company press release on {catalyst_date} or 1-2 days after
  - PR Newswire / GlobeNewswire / BusinessWire announcement
  - SEC 8-K filing
  - FDA approval letter or CRL
  - Phase readout press release

Return ONLY valid JSON in this exact schema (no markdown, no commentary):

{{
  "outcome_class": "APPROVED" | "REJECTED" | "MET_ENDPOINT" | "MISSED_ENDPOINT" | "DELAYED" | "WITHDRAWN" | "MIXED" | "UNKNOWN",
  "endpoint_met": true | false | null,
  "approval_granted": true | false | null,
  "safety_signal_flag": true | false,
  "primary_source_url": "https://...",
  "evidence": "verbatim short quote from the source (under 50 words)",
  "confidence": 0.0 - 1.0,
  "reasoning": "1-2 sentences explaining the classification"
}}

Classification rules:
  - APPROVED: FDA/regulatory approval granted (catalyst_type involves PDUFA / NDA / BLA / approval).
  - REJECTED: CRL issued, refusal to file, withdrawal of application by company.
  - MET_ENDPOINT: Phase 2/3 trial hit primary endpoint with statistical significance.
  - MISSED_ENDPOINT: Phase 2/3 trial failed primary endpoint (p > 0.05 or trend toward placebo).
  - DELAYED: PDUFA pushed back, advisory committee postponed, no decision yet.
  - WITHDRAWN: Company withdrew application before FDA decision.
  - MIXED: Met some endpoints but not others, or approved with significant restrictions.
  - UNKNOWN: Cannot find a definitive press release.

If you cannot find a clear primary source, return outcome_class=UNKNOWN with confidence=0 and a brief reasoning.

ONLY return the JSON object. No prose, no markdown fences."""


def label_catalyst_outcome(
    *,
    ticker: str,
    company: str,
    catalyst_type: str,
    catalyst_date: str,  # YYYY-MM-DD
    drug: Optional[str] = None,
    indication: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the outcome labeler against one catalyst. Returns the labeled dict
    or None on failure. Cost ~$0.0003 per call (Gemini 2.5 Flash with grounding).
    """
    if not GOOGLE_API_KEY:
        logger.warning("[outcome-labeler] GOOGLE_API_KEY not set, skipping")
        return None

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("[outcome-labeler] google-genai package not available")
        return None

    prompt = OUTCOME_LABELER_PROMPT.format(
        ticker=ticker,
        company=company or ticker,
        catalyst_type=catalyst_type or "unknown",
        catalyst_date=catalyst_date,
        drug=drug or "(unspecified)",
        indication=indication or "(unspecified)",
    )

    client = genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=genai_types.HttpOptions(timeout=50000),  # 50s
    )
    config = genai_types.GenerateContentConfig(
        max_output_tokens=2000,
        temperature=0.1,
        # Grounded search to find the press release
        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
    )

    t0 = time.time()
    status = "success"
    err_msg = None
    tokens_in = 0
    tokens_out = 0
    text = ""

    saw_503 = False
    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=config,
            )
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens_in = getattr(usage, "prompt_token_count", 0) or 0
                tokens_out = getattr(usage, "candidates_token_count", 0) or 0
            err_msg = None
            status = "success"
            break
        except Exception as e:
            status = "error"
            err_msg = str(e)[:300]
            if "503" in err_msg and "UNAVAILABLE" in err_msg:
                saw_503 = True
            if attempt == 1 and saw_503:
                logger.info(f"[outcome-labeler] {ticker} 503, retrying in 3s")
                time.sleep(3.0)
                continue
            logger.warning(f"[outcome-labeler] {ticker} failed: {err_msg}")
            break

    elapsed = time.time() - t0
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google", model=LLM_MODEL,
            tokens_in=tokens_in, tokens_out=tokens_out,
            latency_sec=elapsed, status=status, error=err_msg,
            context=f"outcome-labeler:{ticker}",
        )
    except Exception:
        pass

    if status != "success" or not text:
        return None

    # Strip markdown fences if Gemini returned them despite instruction
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"[outcome-labeler] {ticker} JSON parse failed: {e}; text: {cleaned[:200]}")
        return None

    # Validate required fields and clamp confidence
    valid_classes = {
        "APPROVED", "REJECTED", "MET_ENDPOINT", "MISSED_ENDPOINT",
        "DELAYED", "WITHDRAWN", "MIXED", "UNKNOWN",
    }
    oc = parsed.get("outcome_class")
    if oc not in valid_classes:
        logger.warning(f"[outcome-labeler] {ticker} invalid outcome_class={oc}")
        return None

    conf = parsed.get("confidence")
    if conf is None:
        conf = 0.5
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.5
    parsed["confidence"] = conf

    parsed["labeled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    parsed["model"] = LLM_MODEL
    return parsed


def label_outcome_for_db_row(*, db, outcome_id: int) -> Optional[Dict[str, Any]]:
    """Convenience: load a post_catalyst_outcomes row, run the labeler, and
    persist the result back to outcome_labeled_json column."""
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.ticker, pco.catalyst_type, pco.catalyst_date,
                       cu.drug_name, cu.indication, s.company_name
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
                LEFT JOIN screener_stocks s ON s.ticker = pco.ticker
                WHERE pco.id = %s
            """, (outcome_id,))
            row = cur.fetchone()
        if not row:
            return None
        ticker, cat_type, cat_date, drug, indication, company = row
        cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
        labeled = label_catalyst_outcome(
            ticker=ticker, company=company or ticker,
            catalyst_type=cat_type, catalyst_date=cat_str,
            drug=drug, indication=indication,
        )
        if not labeled:
            return None
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE post_catalyst_outcomes
                SET outcome_labeled_json = %s,
                    outcome_label_class = %s,
                    outcome_label_confidence = %s,
                    outcome_labeled_at = now()
                WHERE id = %s
            """, (
                json.dumps(labeled),
                labeled.get("outcome_class"),
                labeled.get("confidence"),
                outcome_id,
            ))
            conn.commit()
        return labeled
    except Exception as e:
        logger.exception(f"label_outcome_for_db_row failed for id={outcome_id}: {e}")
        return None
