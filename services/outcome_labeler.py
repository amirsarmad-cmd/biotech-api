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


# ────────────────────────────────────────────────────────────
# Multi-key rotation
# ────────────────────────────────────────────────────────────
# Reads GOOGLE_API_KEY plus GOOGLE_API_KEY_1..N from env. Picks the
# least-recently-used non-cooling key for each call. On rate-limit /
# quota / timeout errors, the failing key is put on a 5-minute
# cooldown and the call is retried against the next key. If every
# key is cooling, the labeler returns None and the worker pool's
# circuit breaker takes over (see routes/admin.py _label_state).
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
                "success_count": 0,
                "error_count": 0,
            })


def _select_key() -> Tuple[Optional[str], Optional[str]]:
    """Return (label, key) for the LRU non-cooling key, or (None, None)
    if every key is on cooldown."""
    with _key_lock:
        _ensure_key_pool()
        if not _key_pool:
            return None, None
        now = time.time()
        available = [
            (label, k) for label, k in _key_pool
            if (_key_state[label]["cooling_until"] or 0) <= now
        ]
        if not available:
            return None, None
        # LRU — least recently used wins
        available.sort(key=lambda lk: _key_state[lk[0]]["last_used_at"] or 0)
        label, key = available[0]
        _key_state[label]["last_used_at"] = now
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


def _mark_key_failed(label: str, kind: str, err_text: str) -> None:
    with _key_lock:
        s = _key_state.get(label)
        if not s:
            return
        s["error_count"] += 1
        s["last_error_at"] = time.time()
        s["last_error_kind"] = kind
        s["last_error_text"] = (err_text or "")[:200]
        if kind in ("rate_limit", "quota", "timeout", "auth"):
            s["cooling_until"] = time.time() + _KEY_COOLDOWN_SECONDS


def _mark_key_success(label: str) -> None:
    with _key_lock:
        s = _key_state.get(label)
        if not s:
            return
        s["success_count"] += 1
        s["last_success_at"] = time.time()
        # Clear cooldown on first success — key is healthy again
        if s.get("cooling_until"):
            s["cooling_until"] = None


def get_key_status() -> Dict[str, Any]:
    """Read-only snapshot of the per-key pool state. Surfaced via
    /admin/labeler/key-status."""
    with _key_lock:
        _ensure_key_pool()
        now = time.time()
        return {
            "pool_size": len(_key_pool or []),
            "cooldown_seconds": _KEY_COOLDOWN_SECONDS,
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

    # Rotate across the pool. We try every available key once before
    # giving up, so a transient outage on one key doesn't fail the
    # whole call. Per-key, we still do the legacy 2-attempt retry on
    # 503 (transient backend hiccup).
    max_key_attempts = max(1, len(_key_pool))
    for key_attempt in range(max_key_attempts):
        used_label, api_key = _select_key()
        if not api_key:
            err_msg = "no Gemini keys available — all on cooldown"
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
                _mark_key_success(used_label)
                last_kind = None
                break
            except Exception as e:
                err_msg = str(e)[:300]
                last_kind = _classify_error(err_msg)
                _mark_key_failed(used_label, last_kind, err_msg)
                if last_kind == "transient" and attempt == 1:
                    logger.info(
                        f"[outcome-labeler] {ticker} key={used_label} transient, retry in 3s"
                    )
                    time.sleep(3.0)
                    continue
                logger.warning(
                    f"[outcome-labeler] {ticker} key={used_label} {last_kind}: {err_msg}"
                )
                status = "error"
                break

        if status == "success":
            break
        # Only rotate keys for failures that look key-specific. For
        # request-shape bugs ("other"), more keys won't help.
        if last_kind not in ("rate_limit", "quota", "timeout", "auth", "transient"):
            break

    elapsed = time.time() - t0
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google", model=LLM_MODEL,
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
