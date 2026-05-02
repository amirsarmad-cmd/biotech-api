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
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("OUTCOME_LABELER_MODEL", "gemini-2.5-flash")

# Model fallback chain — Flash first, Pro on Flash rate-limit.
# Pro has its own per-minute RPM bucket on the same key so it absorbs
# bursts that would 429 a single model. Cooldown is per-(key, model)
# so cooling Flash on key_1 does NOT lock Pro on key_1. Override
# whole chain with OUTCOME_LABELER_MODEL_CHAIN env (comma-separated).
def _build_model_chain() -> List[str]:
    raw = (os.getenv("OUTCOME_LABELER_MODEL_CHAIN") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [LLM_MODEL, "gemini-2.5-pro"]

MODEL_CHAIN: List[str] = _build_model_chain()


# ────────────────────────────────────────────────────────────
# Multi-key + multi-model rotation
# ────────────────────────────────────────────────────────────
# Reads GOOGLE_API_KEY plus GOOGLE_API_KEY_1..N from env. Each call
# picks the LRU (key, model) combination whose cooldown has expired —
# Flash is preferred, Pro is used when Flash is rate-capped on a key.
# Cooldowns are per-(key, model) so the bigger/slower Pro doesn't get
# locked when only Flash hits a per-minute RPM ceiling.
_KEY_COOLDOWN_SECONDS = 300
_MAX_KEY_SLOTS = 10  # supports GOOGLE_API_KEY_1..GOOGLE_API_KEY_10

_key_lock = threading.Lock()
_key_pool: Optional[List[Tuple[str, str]]] = None
_key_state: Dict[str, Dict[str, Any]] = {}


def _build_key_pool() -> List[Tuple[str, str]]:
    """Read all configured Gemini keys from env. Returns list of
    (label, value) tuples. Labels are stable for telemetry."""
    pool: List[Tuple[str, str]] = []
    primary = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if primary:
        pool.append(("primary", primary))
    for i in range(1, _MAX_KEY_SLOTS + 1):
        v = (os.getenv(f"GOOGLE_API_KEY_{i}") or "").strip()
        if v:
            pool.append((f"key_{i}", v))
    # De-dupe by value (operator may have set the same key twice)
    seen = set()
    unique: List[Tuple[str, str]] = []
    for label, val in pool:
        if val in seen:
            continue
        seen.add(val)
        unique.append((label, val))
    return unique


def _ensure_key_pool() -> None:
    global _key_pool
    if _key_pool is None:
        _key_pool = _build_key_pool()
        for label, _ in _key_pool:
            _key_state.setdefault(label, {
                "label": label,
                "last_used_at": None,
                "last_success_at": None,
                "last_error_at": None,
                "last_error_kind": None,
                "last_error_text": None,
                "cooling_until": None,
                # Per-(key, model) cooldown — see MODEL_CHAIN comment.
                "model_cooling": {},   # model_name -> cooling_until_ts
                "success_count": 0,
                "error_count": 0,
            })


def _select_key_and_model() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (label, api_key, model) for the LRU key whose first
    non-cooling model in MODEL_CHAIN is available. (None, None, None)
    if every (key, model) combination is cooling."""
    with _key_lock:
        _ensure_key_pool()
        if not _key_pool:
            return None, None, None
        now = time.time()
        candidates: List[Tuple[str, str, str, float]] = []
        for label, key in _key_pool:
            state = _key_state[label]
            if (state.get("cooling_until") or 0) > now:
                continue
            for model in MODEL_CHAIN:
                if (state["model_cooling"].get(model) or 0) <= now:
                    candidates.append((label, key, model,
                                       state["last_used_at"] or 0))
                    break
        if not candidates:
            return None, None, None
        candidates.sort(key=lambda c: c[3])
        label, key, model, _ = candidates[0]
        _key_state[label]["last_used_at"] = now
        return label, key, model


def _select_key() -> Tuple[Optional[str], Optional[str]]:
    """Backward-compat shim — returns (label, key) only. Use
    _select_key_and_model() for the model-aware selection."""
    label, key, _ = _select_key_and_model()
    return label, key


def _classify_error(err_text: str) -> str:
    """Categorise a Gemini exception. Drives whether we rotate keys
    or just retry/give up."""
    e = (err_text or "").lower()
    if "429" in err_text or "too many" in e or "rate limit" in e:
        return "rate_limit"
    if "resource_exhausted" in e or "quota" in e or "exceeded" in e:
        return "quota"
    if "timeout" in e or "deadline" in e or "504" in err_text:
        return "timeout"
    if "503" in err_text or "unavailable" in e or "500" in err_text:
        return "transient"
    if "401" in err_text or "403" in err_text or "permission" in e or "api key" in e:
        # Auth failures look like quota issues to the worker — rotate so
        # one bad key doesn't tank the whole batch.
        return "auth"
    return "other"


def _mark_key_failed(label: str, kind: str, err_text: str,
                     model: Optional[str] = None) -> None:
    with _key_lock:
        s = _key_state.get(label)
        if not s:
            return
        s["error_count"] += 1
        s["last_error_at"] = time.time()
        s["last_error_kind"] = kind
        s["last_error_text"] = (err_text or "")[:200]
        if kind in ("rate_limit", "quota", "timeout", "auth"):
            if model is not None:
                # Per-(key, model) cooldown so Pro can still try this
                # key while Flash is cooling.
                s["model_cooling"][model] = time.time() + _KEY_COOLDOWN_SECONDS
            else:
                # Legacy fallback: cool whole key
                s["cooling_until"] = time.time() + _KEY_COOLDOWN_SECONDS


def _mark_key_success(label: str, model: Optional[str] = None) -> None:
    with _key_lock:
        s = _key_state.get(label)
        if not s:
            return
        s["success_count"] += 1
        s["last_success_at"] = time.time()
        if s.get("cooling_until"):
            s["cooling_until"] = None
        if model is not None:
            s["model_cooling"].pop(model, None)


def get_key_status() -> Dict[str, Any]:
    """Read-only snapshot of the per-key pool state. Surfaced via
    /admin/labeler/key-status."""
    with _key_lock:
        _ensure_key_pool()
        now = time.time()
        return {
            "pool_size": len(_key_pool or []),
            "cooldown_seconds": _KEY_COOLDOWN_SECONDS,
            "model_chain": MODEL_CHAIN,
            "keys": [
                {
                    "label": s["label"],
                    "success_count": s["success_count"],
                    "error_count": s["error_count"],
                    "last_used_secs_ago": (
                        round(now - s["last_used_at"], 1)
                        if s["last_used_at"] else None
                    ),
                    "last_success_secs_ago": (
                        round(now - s["last_success_at"], 1)
                        if s["last_success_at"] else None
                    ),
                    "last_error_kind": s["last_error_kind"],
                    "last_error_text": s["last_error_text"],
                    "cooling": (s["cooling_until"] or 0) > now,
                    "cooling_remaining_secs": (
                        round(s["cooling_until"] - now, 1)
                        if (s.get("cooling_until") or 0) > now else 0
                    ),
                    "model_cooling": {
                        m: round(ts - now, 1)
                        for m, ts in (s.get("model_cooling") or {}).items()
                        if ts > now
                    },
                }
                for s in (_key_state[label] for label, _ in (_key_pool or []))
            ],
        }


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
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("[outcome-labeler] google-genai package not available")
        return None

    _ensure_key_pool()
    if not _key_pool:
        logger.warning("[outcome-labeler] no Gemini keys configured (GOOGLE_API_KEY*)")
        return None

    prompt = OUTCOME_LABELER_PROMPT.format(
        ticker=ticker,
        company=company or ticker,
        catalyst_type=catalyst_type or "unknown",
        catalyst_date=catalyst_date,
        drug=drug or "(unspecified)",
        indication=indication or "(unspecified)",
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
    used_label: Optional[str] = None
    used_model: Optional[str] = None

    # Rotate across (key, model) pairs. The selector hands back the
    # LRU key whose first non-cooling model in MODEL_CHAIN is
    # available — i.e. Flash is preferred but Pro takes over when
    # Flash is rate-capped on a key. We try up to (keys × models)
    # combinations before giving up.
    max_attempts = max(1, len(_key_pool) * len(MODEL_CHAIN))
    for combo_attempt in range(max_attempts):
        used_label, api_key, used_model = _select_key_and_model()
        if not api_key:
            err_msg = "no Gemini (key, model) combinations available — all cooling"
            status = "error"
            logger.warning(f"[outcome-labeler] {ticker}: {err_msg}")
            break

        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=50000),
        )

        last_kind: Optional[str] = None
        for attempt in (1, 2):
            try:
                response = client.models.generate_content(
                    model=used_model,
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
                _mark_key_success(used_label, model=used_model)
                last_kind = None
                break
            except Exception as e:
                err_msg = str(e)[:300]
                last_kind = _classify_error(err_msg)
                _mark_key_failed(used_label, last_kind, err_msg, model=used_model)
                if last_kind == "transient" and attempt == 1:
                    logger.info(
                        f"[outcome-labeler] {ticker} key={used_label} model={used_model} "
                        f"transient, retry in 3s"
                    )
                    time.sleep(3.0)
                    continue
                logger.warning(
                    f"[outcome-labeler] {ticker} key={used_label} model={used_model} "
                    f"{last_kind}: {err_msg}"
                )
                status = "error"
                break

        if status == "success":
            break
        # Only rotate combinations for failures that look key/model-
        # specific. For request-shape bugs ("other"), more attempts
        # won't help.
        if last_kind not in ("rate_limit", "quota", "timeout", "auth", "transient"):
            break

    elapsed = time.time() - t0
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google",
            model=used_model or MODEL_CHAIN[0],
            feature="outcome_labeler",
            ticker=ticker,
            tokens_input=tokens_in, tokens_output=tokens_out,
            duration_ms=int(elapsed * 1000),
            status=status, error_message=err_msg,
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
    parsed["model"] = used_model or MODEL_CHAIN[0]
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
