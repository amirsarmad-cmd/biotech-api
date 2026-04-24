"""
Shared LLM helper: multi-provider JSON call with automatic fallback.
Anthropic → OpenAI → Google. Returns (result_dict, error_str).

No external dependencies on other project modules — safe to import anywhere.
"""
import os
import json
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


def _try_anthropic(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""), timeout=40.0)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text, None
    except Exception as e:
        return None, f"anthropic: {str(e)[:200]}"


def _try_openai(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), timeout=40.0)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"openai: {str(e)[:200]}"


def _try_google(prompt: str, max_tokens: int, temperature: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return resp.text, None
    except Exception as e:
        return None, f"google: {str(e)[:200]}"


def _parse_json(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Strip markdown fences, parse JSON."""
    if not text:
        return None, "empty response"
    text = text.strip()
    if text.startswith("```"):
        # Strip ``` or ```json
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


def call_llm_json(prompt: str, max_tokens: int = 1200, temperature: float = 0.2) -> Tuple[Optional[Dict], Optional[str]]:
    """Try multiple LLM providers until one returns valid JSON.

    Returns (result_dict_with_"_llm_provider"_key, error_str).
    On success, error_str is None and result has "_llm_provider": "anthropic" | "openai" | "google".
    On total failure, result is None and error_str contains each provider's failure reason.
    """
    providers = [
        ("anthropic", _try_anthropic),
        ("openai", _try_openai),
        ("google", _try_google),
    ]
    errors = []
    for name, fn in providers:
        text, err = fn(prompt, max_tokens, temperature)
        if text is not None:
            parsed, parse_err = _parse_json(text)
            if parsed is not None:
                parsed["_llm_provider"] = name
                return parsed, None
            errors.append(f"{name}: {parse_err}")
        else:
            errors.append(err or f"{name}: unknown error")
    return None, " | ".join(errors)


# Backward-compat alias (some code still imports _call_llm_json)
_call_llm_json = call_llm_json
