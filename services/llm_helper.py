"""
Shared LLM helper: multi-provider JSON call with automatic fallback.
Anthropic → OpenAI → Google. Returns (result_dict, error_str).

Records every attempted call to llm_usage table for cost accounting and
budget enforcement (Phase 3B). Failures to record are swallowed silently
so accounting never breaks the call.
"""
import os
import json
import time
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================
# Inner provider calls — return (text, error, meta)
# meta: {"provider", "model", "tokens_input", "tokens_output", "duration_ms"}
# ============================================================

def _try_anthropic(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    t0 = time.time()
    model = "claude-sonnet-4-5"
    meta = {"provider": "anthropic", "model": model, "tokens_input": 0, "tokens_output": 0, "duration_ms": None}
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), timeout=40.0)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(msg, "usage", None)
        if usage:
            meta["tokens_input"] = getattr(usage, "input_tokens", 0) or 0
            meta["tokens_output"] = getattr(usage, "output_tokens", 0) or 0
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return msg.content[0].text, None, meta
    except Exception as e:
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return None, f"anthropic: {str(e)[:200]}", meta


def _try_openai(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    t0 = time.time()
    model = "gpt-4o"
    meta = {"provider": "openai", "model": model, "tokens_input": 0, "tokens_output": 0, "duration_ms": None}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=40.0)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        usage = getattr(resp, "usage", None)
        if usage:
            meta["tokens_input"] = getattr(usage, "prompt_tokens", 0) or 0
            meta["tokens_output"] = getattr(usage, "completion_tokens", 0) or 0
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return resp.choices[0].message.content, None, meta
    except Exception as e:
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return None, f"openai: {str(e)[:200]}", meta


def _try_google(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    t0 = time.time()
    model = "gemini-2.5-flash"
    meta = {"provider": "google", "model": model, "tokens_input": 0, "tokens_output": 0, "duration_ms": None}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            http_options=types.HttpOptions(timeout=50000),  # 50s
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            meta["tokens_input"] = getattr(usage, "prompt_token_count", 0) or 0
            meta["tokens_output"] = getattr(usage, "candidates_token_count", 0) or 0
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return resp.text, None, meta
    except Exception as e:
        meta["duration_ms"] = int((time.time() - t0) * 1000)
        return None, f"google: {str(e)[:200]}", meta


def _parse_json(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Strip markdown fences, parse JSON."""
    if not text:
        return None, "empty response"
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse: {str(e)[:100]}"


def _record(provider: str, model: str, feature: Optional[str], ticker: Optional[str],
            meta: Dict, status: str, error: Optional[str] = None,
            request_id: Optional[str] = None) -> None:
    """Best-effort usage recording. Imported lazily so tests/imports don't break
    if llm_usage module is missing."""
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider=provider, model=model, feature=feature, ticker=ticker,
            tokens_input=meta.get("tokens_input", 0),
            tokens_output=meta.get("tokens_output", 0),
            duration_ms=meta.get("duration_ms"),
            status=status, error_message=error, request_id=request_id,
        )
    except Exception as e:
        logger.debug(f"_record skipped: {e}")


def call_llm_json(prompt: str, max_tokens: int = 1200, temperature: float = 0.2,
                   feature: Optional[str] = None, ticker: Optional[str] = None,
                   request_id: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Try multiple LLM providers until one returns valid JSON.

    Returns (result_dict_with_"_llm_provider"_key, error_str).
    On success, error_str is None and result has "_llm_provider": "anthropic" | "openai" | "google".
    On total failure, result is None and error_str contains each provider's failure reason.

    Records EVERY attempted call to llm_usage (Phase 3B) — including failures so
    error rate is visible in the analytics tab.
    
    feature: optional feature/route name for analytics grouping (e.g. 'npv_v2',
             'risk_factors', 'universe_seeder'). Passes through to llm_usage.
    ticker:  optional ticker for stock-scoped analytics
    request_id: optional correlation id (job id, etc) for tracing
    """
    # Pre-call budget check (only if budget hard_cutoff applies)
    try:
        from services.llm_usage import check_budget
        bc = check_budget(provider=None, feature=feature)
        if not bc.get("allowed", True):
            reason = bc.get("reason", "budget_blocked")
            # Record the blocked attempt for visibility
            _record(provider="budget", model=None, feature=feature, ticker=ticker,
                     meta={"tokens_input": 0, "tokens_output": 0, "duration_ms": 0},
                     status="budget_blocked", error=f"hard_cutoff: {reason}",
                     request_id=request_id)
            return None, f"budget hard cutoff: {reason}"
    except Exception:
        pass  # budget check is best-effort

    providers = [
        ("anthropic", _try_anthropic),
        ("openai", _try_openai),
        ("google", _try_google),
    ]
    errors = []
    for name, fn in providers:
        text, err, meta = fn(prompt, max_tokens, temperature)
        if text is not None:
            parsed, parse_err = _parse_json(text)
            if parsed is not None:
                parsed["_llm_provider"] = name
                # Record success
                _record(provider=name, model=meta.get("model"),
                        feature=feature, ticker=ticker, meta=meta,
                        status="success", request_id=request_id)
                return parsed, None
            errors.append(f"{name}: {parse_err}")
            # Record parse-fail (we got bytes but unusable)
            _record(provider=name, model=meta.get("model"),
                    feature=feature, ticker=ticker, meta=meta,
                    status="parse_error", error=parse_err, request_id=request_id)
        else:
            errors.append(err or f"{name}: unknown error")
            _record(provider=name, model=meta.get("model"),
                    feature=feature, ticker=ticker, meta=meta,
                    status="error", error=err, request_id=request_id)
    return None, " | ".join(errors)


# Backward-compat alias (some code still imports _call_llm_json)
_call_llm_json = call_llm_json
