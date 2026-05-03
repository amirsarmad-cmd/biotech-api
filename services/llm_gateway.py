"""Universal LLM gateway — multi-provider, multi-key, capability-aware.

Every LLM call in biotech-api should route through `llm_call()` or
`llm_embeddings()`. The gateway:

  1. Maintains a key pool per provider (GOOGLE_API_KEY,
     GOOGLE_API_KEY_1..10, ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_1..10,
     OPENAI_API_KEY, OPENAI_API_KEY_1..10).
  2. Picks the LRU non-cooling key on every call.
  3. On rate-limit / quota / timeout / auth errors, puts that key on a
     5-minute cooldown and tries the next key in the same provider.
  4. When every key in a provider is cooling, falls through to the
     next provider in the capability's chain (Anthropic → OpenAI →
     Google for text; Google-only for grounded search; OpenAI-only
     for embeddings).
  5. Tracks a per-provider rolling success rate. If it drops below
     5/100 the provider is paused for 5 min and skipped in chains.
  6. Records every attempt to llm_usage so post-hoc cost & failure
     attribution work.
  7. Exposes get_status() for /admin/llm/status.

Failure mode taxonomy (the 2026-05-02 stall): with this gateway,
no caller can hang on a degraded provider. The longest possible
wait is `len(chain) * len(keys_per_provider) * (timeout_s + retry_s)`
which is bounded — typically ~60s. After that the caller gets
`LLMAllProvidersFailed` and decides what to do (return None, surface
to user, etc.).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────

_KEY_COOLDOWN_SECONDS = 300
_MAX_KEY_SLOTS = 10
_CIRCUIT_WINDOW = 100
_CIRCUIT_MIN_SUCCESS = 5
_CIRCUIT_COOLDOWN_SECONDS = 300

# Default model fallback chains per (provider, capability). Order
# matters — the first model is tried first; on rate-limit / quota /
# timeout errors the gateway rotates to the next model on the same
# key (Pro and Flash share keys but have separate per-minute RPM
# buckets, so this absorbs bursts that would 429 a single model).
# Override per call via model_overrides=str | List[str].
_DEFAULT_MODEL_CHAINS: Dict[Tuple[str, str], List[str]] = {
    ("anthropic", "text_json"):       ["claude-sonnet-4-6"],
    ("anthropic", "text_freeform"):   ["claude-sonnet-4-6"],
    ("openai",    "text_json"):       ["gpt-4o"],
    ("openai",    "text_freeform"):   ["gpt-4o"],
    ("openai",    "embeddings"):      ["text-embedding-3-small"],
    ("google",    "text_json"):       ["gemini-2.5-flash", "gemini-2.5-pro"],
    ("google",    "text_freeform"):   ["gemini-2.5-flash", "gemini-2.5-pro"],
    ("google",    "grounded_search"): ["gemini-2.5-flash", "gemini-2.5-pro"],
}

# Default fallback chains per capability. Order matters — first
# provider is tried first. For capabilities only one provider
# supports (grounded_search, embeddings) the chain is single-element.
_DEFAULT_CHAINS: Dict[str, List[str]] = {
    "text_json":       ["anthropic", "openai", "google"],
    "text_freeform":   ["anthropic", "openai", "google"],
    "grounded_search": ["google"],
    "embeddings":      ["openai"],
}

# Env-var prefix per provider. ProviderPool reads
# {prefix} + "" and {prefix}_1..N.
_ENV_PREFIX: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


# ────────────────────────────────────────────────────────────
# Exceptions
# ────────────────────────────────────────────────────────────

class LLMGatewayError(Exception):
    """Base for gateway errors."""


class LLMAllProvidersFailed(LLMGatewayError):
    """Every provider in the chain is unavailable or has errored.
    Carries the list of (provider, key_label, error_kind, error_text)
    tuples so callers can decide what to log / show."""
    def __init__(self, attempts: List[Dict[str, Any]]):
        self.attempts = attempts
        kinds = ", ".join(sorted({a["kind"] for a in attempts}))
        super().__init__(
            f"all LLM providers failed after {len(attempts)} attempts ({kinds})"
        )


class LLMCapabilityUnavailable(LLMGatewayError):
    """No provider in the configured chain supports the requested
    capability. E.g. grounded_search with Google fully down."""


# ────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text: str
    parsed_json: Optional[dict]
    provider: str
    key_label: str
    model: str
    tokens_input: int
    tokens_output: int
    duration_ms: int
    # Populated only when capability='grounded_search' AND provider='google'.
    # grounding_chunks: list of {"uri", "title"} dicts from
    # response.candidates[0].grounding_metadata.grounding_chunks. These
    # are the URLs Gemini consulted; the multi-LLM consensus orchestrator
    # writes them into catalyst_event_news so all downstream readers see
    # the same enriched library.
    grounding_chunks: Optional[List[Dict]] = None
    # search_queries: the actual queries Gemini issued during grounding.
    # Useful for "why didn't grounded search find X" debugging.
    search_queries: Optional[List[str]] = None


@dataclass
class EmbeddingsResult:
    vectors: List[List[float]]
    provider: str
    key_label: str
    model: str
    duration_ms: int


# ────────────────────────────────────────────────────────────
# Key pool — one per provider
# ────────────────────────────────────────────────────────────

class ProviderPool:
    """Per-provider pool of API keys with LRU rotation + cooldown.

    Generalisation of the labeler-only pool that shipped in commit
    175ffec. Each provider gets one of these; pools are created
    lazily on first use and survive for the process lifetime.
    """

    def __init__(self, name: str, env_prefix: str):
        self.name = name
        self.env_prefix = env_prefix
        self._lock = threading.Lock()
        self._pool: List[Tuple[str, str]] = []
        self._state: Dict[str, Dict[str, Any]] = {}
        self._build()

    def _build(self) -> None:
        primary = (os.getenv(self.env_prefix) or "").strip()
        pool: List[Tuple[str, str]] = []
        if primary:
            pool.append(("primary", primary))
        for i in range(1, _MAX_KEY_SLOTS + 1):
            v = (os.getenv(f"{self.env_prefix}_{i}") or "").strip()
            if v:
                pool.append((f"key_{i}", v))
        # De-dupe by value
        seen, unique = set(), []
        for label, val in pool:
            if val in seen:
                continue
            seen.add(val)
            unique.append((label, val))
        self._pool = unique
        for label, _ in self._pool:
            self._state.setdefault(label, {
                "label": label,
                "last_used_at": None,
                "last_success_at": None,
                "last_error_at": None,
                "last_error_kind": None,
                "last_error_text": None,
                "cooling_until": None,
                # Per-(key, model) cooldown for the Flash→Pro fallback
                # pattern. Flash being capped on key_1 should NOT
                # block Pro on key_1, since they have separate RPM
                # buckets even though they share the same API key.
                "model_cooling": {},   # model_name -> cooling_until_ts
                "success_count": 0,
                "error_count": 0,
            })

    @property
    def size(self) -> int:
        return len(self._pool)

    def select(
        self, model_chain: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (label, api_key, model) for the LRU key whose first
        non-cooling model in `model_chain` is available. Returns all
        None if every (key, model) combination is cooling.

        Backward-compatible: if model_chain is None or empty, the
        returned model is None and the caller is responsible for
        passing one to the adapter.
        """
        with self._lock:
            if not self._pool:
                return None, None, None
            now = time.time()
            chain = model_chain or [None]
            candidates: List[Tuple[str, str, Optional[str], float]] = []
            for label, key in self._pool:
                state = self._state[label]
                # Skip key if its overall cooldown is set (legacy)
                if (state.get("cooling_until") or 0) > now:
                    continue
                # First non-cooling model in chain wins for this key
                for model in chain:
                    cool_ts = (
                        (state["model_cooling"].get(model) or 0)
                        if model is not None else 0
                    )
                    if cool_ts <= now:
                        candidates.append((label, key, model,
                                           state["last_used_at"] or 0))
                        break
            if not candidates:
                return None, None, None
            candidates.sort(key=lambda c: c[3])
            label, key, model, _ = candidates[0]
            self._state[label]["last_used_at"] = now
            return label, key, model

    def mark_success(self, label: str, model: Optional[str] = None) -> None:
        with self._lock:
            s = self._state.get(label)
            if not s:
                return
            s["success_count"] += 1
            s["last_success_at"] = time.time()
            if s.get("cooling_until"):
                s["cooling_until"] = None  # key recovered
            if model is not None:
                # Clear per-(key, model) cooldown on success
                s["model_cooling"].pop(model, None)

    def mark_failed(
        self, label: str, kind: str, err_text: str,
        model: Optional[str] = None,
    ) -> None:
        with self._lock:
            s = self._state.get(label)
            if not s:
                return
            s["error_count"] += 1
            s["last_error_at"] = time.time()
            s["last_error_kind"] = kind
            s["last_error_text"] = (err_text or "")[:200]
            if kind in ("rate_limit", "quota", "timeout", "auth", "not_found"):
                if model is not None:
                    # Cool only this (key, model) pair so the next
                    # model in the chain can still try this key.
                    # not_found is permanent for this model on the API;
                    # use a long cooldown so the gateway doesn't waste
                    # cycles re-trying it.
                    cooldown = _KEY_COOLDOWN_SECONDS * (12 if kind == "not_found" else 1)
                    s["model_cooling"][model] = time.time() + cooldown
                else:
                    # Legacy callers without a model: cool the whole key
                    s["cooling_until"] = time.time() + _KEY_COOLDOWN_SECONDS

    def status(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "provider": self.name,
                "pool_size": len(self._pool),
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
                    for s in (self._state[label] for label, _ in self._pool)
                ],
            }


_pool_lock = threading.Lock()
_pools: Dict[str, ProviderPool] = {}


def _get_pool(provider: str) -> ProviderPool:
    with _pool_lock:
        if provider not in _pools:
            _pools[provider] = ProviderPool(provider, _ENV_PREFIX[provider])
        return _pools[provider]


# ────────────────────────────────────────────────────────────
# Per-provider circuit breaker
# ────────────────────────────────────────────────────────────

_breaker_lock = threading.Lock()
_breaker_state: Dict[str, Dict[str, Any]] = {}


def _ensure_breaker(provider: str) -> Dict[str, Any]:
    with _breaker_lock:
        if provider not in _breaker_state:
            _breaker_state[provider] = {
                "recent_attempts": [],     # rolling window of bool
                "tripped_until": None,
                "trip_count": 0,
            }
        return _breaker_state[provider]


def _is_provider_paused(provider: str) -> bool:
    s = _ensure_breaker(provider)
    with _breaker_lock:
        return (s["tripped_until"] or 0) > time.time()


def _record_provider_attempt(provider: str, success: bool) -> None:
    s = _ensure_breaker(provider)
    with _breaker_lock:
        s["recent_attempts"].append(success)
        if len(s["recent_attempts"]) > _CIRCUIT_WINDOW:
            del s["recent_attempts"][: len(s["recent_attempts"]) - _CIRCUIT_WINDOW]
        if (
            len(s["recent_attempts"]) >= _CIRCUIT_WINDOW
            and sum(s["recent_attempts"]) < _CIRCUIT_MIN_SUCCESS
            and (s["tripped_until"] or 0) <= time.time()
        ):
            s["tripped_until"] = time.time() + _CIRCUIT_COOLDOWN_SECONDS
            s["trip_count"] += 1
            logger.warning(
                f"[llm_gateway] circuit breaker tripped for {provider}: "
                f"{sum(s['recent_attempts'])}/{_CIRCUIT_WINDOW} succeeded"
            )


def _breaker_status(provider: str) -> Dict[str, Any]:
    s = _ensure_breaker(provider)
    with _breaker_lock:
        window = s["recent_attempts"]
        now = time.time()
        return {
            "window_size": len(window),
            "success_count": sum(window),
            "success_pct": (round(100.0 * sum(window) / len(window), 1)
                            if window else None),
            "tripped": (s["tripped_until"] or 0) > now,
            "tripped_remaining_secs": (
                round(s["tripped_until"] - now, 1)
                if (s.get("tripped_until") or 0) > now else 0
            ),
            "trip_count_lifetime": s["trip_count"],
        }


# ────────────────────────────────────────────────────────────
# Error classification (reusable)
# ────────────────────────────────────────────────────────────

def classify_error(err_text: str) -> str:
    """Categorise an LLM exception. Drives whether to rotate keys,
    fall to next provider, or give up."""
    e = (err_text or "").lower()
    if "429" in err_text or "too many" in e or "rate limit" in e or "ratelimit" in e:
        return "rate_limit"
    if "resource_exhausted" in e or "quota" in e or "exceeded" in e:
        return "quota"
    if "timeout" in e or "deadline" in e or "504" in err_text or "timed out" in e:
        return "timeout"
    if "503" in err_text or "unavailable" in e or "500" in err_text or "502" in err_text:
        return "transient"
    if "401" in err_text or "403" in err_text or "permission" in e or "api key" in e or "unauthorized" in e:
        return "auth"
    # 404 on a specific model means "this model ID doesn't exist for this
    # API version" — try the next model in the chain instead of breaking
    # out of the provider entirely.
    if "404" in err_text or "not_found" in e or "is not found" in e or "not supported for" in e:
        return "not_found"
    return "other"


# ────────────────────────────────────────────────────────────
# Provider adapters — translate our shape to each SDK
# ────────────────────────────────────────────────────────────

def _coerce_messages(prompt: Optional[str], messages: Optional[List[Dict]]) -> List[Dict]:
    """Normalise to a list of {role, content} dicts."""
    if messages:
        return list(messages)
    if prompt is None:
        raise ValueError("llm_call requires either prompt or messages")
    return [{"role": "user", "content": prompt}]


def _call_anthropic(api_key, model, prompt, messages, system,
                    max_tokens, temperature, timeout_s) -> Tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    msgs = _coerce_messages(prompt, messages)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": msgs,
    }
    if system:
        kwargs["system"] = system
    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:
        # Newer Anthropic models (Opus 4.7+) deprecate `temperature`.
        # Retry without it on the specific 400.
        msg = str(e)
        if "temperature" in msg and "deprecated" in msg.lower() and "temperature" in kwargs:
            kwargs.pop("temperature", None)
            resp = client.messages.create(**kwargs)
        else:
            raise
    text = resp.content[0].text if resp.content else ""
    usage = getattr(resp, "usage", None)
    tin = (getattr(usage, "input_tokens", 0) or 0) if usage else 0
    tout = (getattr(usage, "output_tokens", 0) or 0) if usage else 0
    return text, tin, tout


def _openai_uses_max_completion_tokens(model: str) -> bool:
    """Newer OpenAI models (GPT-5/5.5, o-series) reject `max_tokens` and
    require `max_completion_tokens`. Older models (gpt-4o, gpt-4-turbo,
    gpt-3.5-turbo) still take `max_tokens`. Heuristic by model name."""
    m = (model or "").lower()
    return (
        m.startswith("gpt-5") or
        m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
    )


def _call_openai(api_key, model, prompt, messages, system,
                 max_tokens, temperature, timeout_s,
                 response_format_json: bool) -> Tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=timeout_s)
    msgs: List[Dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(_coerce_messages(prompt, messages))
    kwargs = {
        "model": model,
        "messages": msgs,
    }
    # GPT-5/o-series only accept the default temperature (1) — passing a
    # custom value gets a 400 "Unsupported value". Older gpt-4o and
    # earlier accept any 0.0-2.0.
    is_new_series = _openai_uses_max_completion_tokens(model)
    if not is_new_series:
        kwargs["temperature"] = temperature
    if is_new_series:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Fallbacks: retry with the alternate param shape if the API
        # complains. Cheap insurance against new model IDs.
        msg = str(e)
        retried = False
        if "max_completion_tokens" in msg and "max_tokens" in kwargs:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            retried = True
        elif "max_tokens" in msg and "max_completion_tokens" in kwargs:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = max_tokens
            retried = True
        if "temperature" in msg and "temperature" in kwargs:
            kwargs.pop("temperature", None)
            retried = True
        if retried:
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    tin = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tout = (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return text, tin, tout


def _call_google(api_key, model, prompt, messages, system,
                 max_tokens, temperature, timeout_s,
                 grounded: bool) -> Tuple[str, int, int, Optional[List[Dict]], Optional[List[str]]]:
    """Returns (text, tokens_in, tokens_out, grounding_chunks, search_queries).

    grounding_chunks/search_queries are populated only when grounded=True
    and the response had grounding_metadata (Gemini's grounded search did
    find sources). Both default to None for non-grounded calls.
    """
    from google import genai as g
    from google.genai import types as gt
    client = g.Client(
        api_key=api_key,
        http_options=gt.HttpOptions(timeout=int(timeout_s * 1000)),
    )
    # Google's SDK takes a single contents string; if we have
    # multi-turn messages we flatten to "Role: content" pairs.
    if messages:
        parts = []
        if system:
            parts.append(f"[System]\n{system}")
        for m in messages:
            role = (m.get("role") or "user").upper()
            parts.append(f"[{role}]\n{m.get('content', '')}")
        contents = "\n\n".join(parts)
    else:
        contents = (system + "\n\n" + (prompt or "")) if system else (prompt or "")
    cfg = gt.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    if grounded:
        cfg.tools = [gt.Tool(google_search=gt.GoogleSearch())]
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    tin = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    tout = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

    chunks: Optional[List[Dict]] = None
    queries: Optional[List[str]] = None
    if grounded:
        try:
            cands = getattr(resp, "candidates", None) or []
            gm = getattr(cands[0], "grounding_metadata", None) if cands else None
            if gm:
                raw_chunks = getattr(gm, "grounding_chunks", None) or []
                # Coerce typed objects into plain dicts
                chunks = []
                for c in raw_chunks:
                    web = getattr(c, "web", None)
                    if web and getattr(web, "uri", None):
                        chunks.append({
                            "uri": web.uri,
                            "title": getattr(web, "title", None) or "",
                        })
                queries = list(getattr(gm, "web_search_queries", None) or []) or None
                if not chunks:
                    chunks = None
        except Exception as e:
            logger.warning(f"grounding_metadata extract failed: {e}")
            chunks = None
            queries = None
    return text, tin, tout, chunks, queries


def _call_openai_embeddings(api_key, model, texts: List[str],
                             timeout_s) -> Tuple[List[List[float]], int]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=timeout_s)
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [d.embedding for d in resp.data]
    usage = getattr(resp, "usage", None)
    tin = (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    return vectors, tin


# ────────────────────────────────────────────────────────────
# JSON parsing helper (for text_json capability)
# ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    # Strip markdown fences if present
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t[3:]
        if t.startswith("json"):
            t = t[4:]
        t = t.rsplit("```", 1)[0].strip() if "```" in t else t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


# ────────────────────────────────────────────────────────────
# Telemetry
# ────────────────────────────────────────────────────────────

def _record_usage(*, provider, model, feature, ticker, status,
                   tokens_in, tokens_out, duration_ms, err=None) -> None:
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider=provider, model=model, feature=feature, ticker=ticker,
            tokens_input=tokens_in, tokens_output=tokens_out,
            duration_ms=duration_ms, status=status, error_message=err,
        )
    except Exception:
        pass


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def llm_call(
    *,
    capability: str,
    feature: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    system: Optional[str] = None,
    ticker: Optional[str] = None,
    fallback_chain: Optional[List[str]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,  # {provider: str | List[str]}
    timeout_s: float = 50.0,
    max_attempts_per_provider: int = 4,
    max_tokens: int = 1500,
    temperature: float = 0.3,
) -> LLMResult:
    """Universal LLM call. Within each provider, rotates keys AND
    iterates through the configured model chain (e.g. Flash → Pro
    for Google) so a per-minute rate-cap on one model doesn't fail
    the call. Falls through to the next provider only after every
    (key, model) combination is exhausted. Records every attempt
    to llm_usage.

    capability: "text_json", "text_freeform", or "grounded_search".
    feature:    short tag for telemetry.
    model_overrides: per-provider override of the default model
                    chain — accepts either a single model string or
                    a List[str] for explicit chain control.
    """
    if capability not in ("text_json", "text_freeform", "grounded_search"):
        raise ValueError(f"unknown capability: {capability}")

    chain = fallback_chain or _DEFAULT_CHAINS[capability]
    overrides = model_overrides or {}
    response_format_json = (capability == "text_json")
    grounded = (capability == "grounded_search")
    attempts: List[Dict[str, Any]] = []

    for provider in chain:
        if provider not in _ENV_PREFIX:
            continue
        if _is_provider_paused(provider):
            attempts.append({
                "provider": provider, "key_label": None, "model": None,
                "kind": "circuit_paused", "text": "provider on circuit-breaker cooldown",
            })
            continue
        pool = _get_pool(provider)
        if pool.size == 0:
            attempts.append({
                "provider": provider, "key_label": None, "model": None,
                "kind": "no_keys", "text": f"no {provider} keys configured",
            })
            continue

        # Resolve model chain for this provider
        override = overrides.get(provider)
        if isinstance(override, list):
            model_chain = override
        elif isinstance(override, str):
            model_chain = [override]
        else:
            model_chain = _DEFAULT_MODEL_CHAINS.get((provider, capability), [])
        if not model_chain:
            attempts.append({
                "provider": provider, "key_label": None, "model": None,
                "kind": "no_model",
                "text": f"no default model chain for ({provider}, {capability})",
            })
            continue

        # Try (key, model) combinations within this provider until one
        # succeeds, all are cooling, or we hit max_attempts_per_provider.
        provider_tries = 0
        while provider_tries < max_attempts_per_provider:
            label, api_key, model = pool.select(model_chain)
            if not api_key:
                attempts.append({
                    "provider": provider, "key_label": None, "model": None,
                    "kind": "all_keys_cooling",
                    "text": f"all ({provider}, model) combinations on cooldown",
                })
                break
            provider_tries += 1
            t0 = time.time()
            grounding_chunks: Optional[List[Dict]] = None
            search_queries: Optional[List[str]] = None
            try:
                if provider == "anthropic":
                    text, tin, tout = _call_anthropic(
                        api_key, model, prompt, messages, system,
                        max_tokens, temperature, timeout_s,
                    )
                elif provider == "openai":
                    text, tin, tout = _call_openai(
                        api_key, model, prompt, messages, system,
                        max_tokens, temperature, timeout_s,
                        response_format_json,
                    )
                elif provider == "google":
                    text, tin, tout, grounding_chunks, search_queries = _call_google(
                        api_key, model, prompt, messages, system,
                        max_tokens, temperature, timeout_s,
                        grounded,
                    )
                else:
                    raise ValueError(f"unknown provider: {provider}")

                duration_ms = int((time.time() - t0) * 1000)

                parsed: Optional[dict] = None
                if capability == "text_json":
                    parsed = _parse_json(text)

                pool.mark_success(label, model=model)
                _record_provider_attempt(provider, True)
                _record_usage(
                    provider=provider, model=model, feature=feature, ticker=ticker,
                    status="success", tokens_in=tin, tokens_out=tout,
                    duration_ms=duration_ms,
                )
                return LLMResult(
                    text=text, parsed_json=parsed, provider=provider,
                    key_label=label, model=model,
                    tokens_input=tin, tokens_output=tout,
                    duration_ms=duration_ms,
                    grounding_chunks=grounding_chunks,
                    search_queries=search_queries,
                )

            except Exception as e:
                err_text = str(e)[:300]
                kind = classify_error(err_text)
                duration_ms = int((time.time() - t0) * 1000)
                pool.mark_failed(label, kind, err_text, model=model)
                _record_provider_attempt(provider, False)
                _record_usage(
                    provider=provider, model=model, feature=feature, ticker=ticker,
                    status="error", tokens_in=0, tokens_out=0,
                    duration_ms=duration_ms, err=err_text,
                )
                attempts.append({
                    "provider": provider, "key_label": label, "model": model,
                    "kind": kind, "text": err_text,
                })
                logger.info(
                    f"[llm_gateway] {feature} provider={provider} key={label} "
                    f"model={model} {kind}: {err_text[:120]}"
                )
                # If the error is non-key-specific (e.g. "other" — bad
                # request shape), don't waste more keys/models on this
                # provider; fall through to next provider.
                # "not_found" is per-model: cool the (key, model) and let
                # the loop pick the next model in the chain (e.g. fall
                # gemini-3.0-pro → gemini-2.5-pro).
                if kind == "other":
                    break
                # Else loop and let pool.select() pick the next
                # (key, model) combination — this is what gives us the
                # Flash→Pro fallback within the same key, AND the
                # newer-model→stable-model fallback (gemini-3 → 2.5-pro).

        # Done with this provider's budget; loop to next provider.

    raise LLMAllProvidersFailed(attempts)


def llm_embeddings(
    *,
    texts: List[str],
    feature: str,
    model_overrides: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    max_attempts_per_provider: int = 2,
) -> EmbeddingsResult:
    """Universal embeddings call. Same key-rotation + circuit-breaker
    semantics as llm_call(). Currently OpenAI is the only provider
    supported for embeddings; the chain has one element."""
    chain = _DEFAULT_CHAINS["embeddings"]
    overrides = model_overrides or {}
    attempts: List[Dict[str, Any]] = []

    for provider in chain:
        if provider != "openai":
            continue
        if _is_provider_paused(provider):
            attempts.append({"provider": provider, "key_label": None,
                             "kind": "circuit_paused", "text": "provider paused"})
            continue
        pool = _get_pool(provider)
        if pool.size == 0:
            attempts.append({"provider": provider, "key_label": None,
                             "kind": "no_keys", "text": f"no {provider} keys"})
            continue
        # Embeddings model chain (single-element today; structurally
        # ready for future fallback like text-embedding-3-large).
        override = overrides.get(provider)
        if isinstance(override, list):
            model_chain = override
        elif isinstance(override, str):
            model_chain = [override]
        else:
            model_chain = _DEFAULT_MODEL_CHAINS.get((provider, "embeddings"), [])
        if not model_chain:
            attempts.append({"provider": provider, "key_label": None,
                             "model": None, "kind": "no_model",
                             "text": "no embeddings model configured"})
            continue

        provider_tries = 0
        while provider_tries < max_attempts_per_provider:
            label, api_key, model = pool.select(model_chain)
            if not api_key:
                attempts.append({"provider": provider, "key_label": None,
                                 "model": None, "kind": "all_keys_cooling",
                                 "text": "all (key, model) combinations cooling"})
                break
            provider_tries += 1
            t0 = time.time()
            try:
                vectors, tin = _call_openai_embeddings(api_key, model, texts, timeout_s)
                duration_ms = int((time.time() - t0) * 1000)
                pool.mark_success(label, model=model)
                _record_provider_attempt(provider, True)
                _record_usage(
                    provider=provider, model=model, feature=feature, ticker=None,
                    status="success", tokens_in=tin, tokens_out=0,
                    duration_ms=duration_ms,
                )
                return EmbeddingsResult(
                    vectors=vectors, provider=provider, key_label=label,
                    model=model, duration_ms=duration_ms,
                )
            except Exception as e:
                err_text = str(e)[:300]
                kind = classify_error(err_text)
                duration_ms = int((time.time() - t0) * 1000)
                pool.mark_failed(label, kind, err_text, model=model)
                _record_provider_attempt(provider, False)
                _record_usage(
                    provider=provider, model=model, feature=feature, ticker=None,
                    status="error", tokens_in=0, tokens_out=0,
                    duration_ms=duration_ms, err=err_text,
                )
                attempts.append({"provider": provider, "key_label": label,
                                 "model": model, "kind": kind, "text": err_text})
                if kind == "other":
                    break

    raise LLMAllProvidersFailed(attempts)


def get_status() -> Dict[str, Any]:
    """One-shot snapshot of the gateway state — every provider, every
    key, per-(key, model) cooldowns, circuit-breaker state. Surfaced
    via /admin/llm/status."""
    out: Dict[str, Any] = {
        "config": {
            "key_cooldown_seconds": _KEY_COOLDOWN_SECONDS,
            "max_key_slots": _MAX_KEY_SLOTS,
            "circuit_window": _CIRCUIT_WINDOW,
            "circuit_min_success": _CIRCUIT_MIN_SUCCESS,
            "circuit_cooldown_seconds": _CIRCUIT_COOLDOWN_SECONDS,
        },
        "default_chains": _DEFAULT_CHAINS,
        "default_model_chains": {
            f"{p}:{c}": chain for (p, c), chain in _DEFAULT_MODEL_CHAINS.items()
        },
        "providers": {},
    }
    for provider in _ENV_PREFIX:
        pool = _get_pool(provider)  # forces lazy init for status visibility
        out["providers"][provider] = {
            "pool": pool.status(),
            "circuit": _breaker_status(provider),
        }
    return out
