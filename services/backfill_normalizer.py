"""Normalize staged backfill rows into catalyst_universe via LLM.

Pipeline:
  1. Read pending rows from catalyst_backfill_staging
  2. For each, run Gemini 2.5 Flash to extract:
       - is_clinical_catalyst (true/false — is this actually a catalyst?)
       - catalyst_type (FDA Decision / Phase 1/2/3 Readout / AdComm /
                       Submission / Approval / Rejection / Other)
       - drug_name
       - indication
       - extracted_catalyst_date (often != filing_date; e.g. PDUFA target)
       - date_precision (exact / day / month / quarter / unknown)
       - confidence
  3. Decide:
       - is_clinical_catalyst=false → status='rejected', reject_reason
       - confidence < 0.5 → status='unclear'
       - else → check catalyst_universe for dupe; if dupe, status='duplicate'
       - else → INSERT into catalyst_universe, set status='accepted',
                catalyst_id=new_id

Cost: ~$0.0003 per row (Gemini Flash + grounding). For 5000-row backfill,
total LLM cost ≈ $1.50.

Dedupe rule: same ticker + catalyst_type + catalyst_date within ±7 days.
"""
from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("BACKFILL_NORMALIZER_MODEL", "gemini-2.5-flash")


NORMALIZER_PROMPT = """You are a biotech catalyst classifier. Given an SEC 8-K filing excerpt, decide whether it announces a clinical/regulatory catalyst event and extract structured details.

TICKER: {ticker}
COMPANY: {company}
FILING_DATE: {filing_date}

8-K EXCERPT (first ~3000 chars):
{excerpt}

A "clinical/regulatory catalyst" is one of these specific event types:
  - FDA Decision (PDUFA approval, CRL rejection, accelerated approval, etc.)
  - Phase 1 Readout / Phase 2 Readout / Phase 3 Readout (trial topline results)
  - AdComm (FDA Advisory Committee meeting outcome)
  - Submission (NDA / BLA / IND / sNDA filed with FDA)
  - Trial Initiation (first patient dosed in a Phase 1/2/3)
  - Other (orphan designation, breakthrough designation, fast track, etc.)

IMPORTANT — handling earnings filings with embedded clinical news:
  - If the filing is a quarterly earnings report (Item 2.02) AND it
    announces a NEW clinical/regulatory milestone (a readout, trial
    initiation, FDA correspondence, etc.) in the body, classify by the
    embedded catalyst type with the catalyst's actual date.
  - Pure earnings reports without embedded clinical news → NOT a catalyst.
  - When in doubt, look at the most material clinical content — if it's
    forward-looking guidance about upcoming readouts, NOT a catalyst.
    If it's announcing what just happened or what was filed, IS a catalyst.

NON-catalysts (always return is_clinical_catalyst=false):
  - Pure earnings reports without embedded clinical news
  - Stock offerings, financings, dilution
  - M&A, acquisitions, partnerships (unless tied to a clinical milestone)
  - Equity compensation plan amendments, board changes (unless co-announced
    with clinical news)
  - Investor presentations, conferences (unless they include readout data)
  - Insider trading filings, S-1/S-3 registration statements
  - XBRL data files, financial taxonomy, document/entity information files
    (these are NOT real 8-K content — return is_clinical_catalyst=false
    with reject_reason="XBRL boilerplate, no narrative content")

Return ONLY valid JSON in this schema (no markdown, no commentary):

{{
  "is_clinical_catalyst": true | false,
  "catalyst_type": "FDA Decision" | "Phase 1 Readout" | "Phase 2 Readout" | "Phase 3 Readout" | "AdComm" | "Submission" | "Trial Initiation" | "Other" | null,
  "drug_name": "string or null",
  "indication": "string or null",
  "extracted_catalyst_date": "YYYY-MM-DD or null",
  "date_precision": "exact" | "day" | "month" | "quarter" | "unknown",
  "confidence": 0.0 - 1.0,
  "reject_reason": "string explaining why (only if is_clinical_catalyst=false)"
}}

Date extraction rules:
  - If the filing announces a PDUFA target date, use THAT as extracted_catalyst_date
    with precision=exact or day.
  - If the filing announces a trial readout that just happened, use the filing_date
    with precision=day.
  - If the filing previews a Q3 readout, use 2024-09-30 (last day of Q3) with
    precision=quarter.
  - If date can't be inferred, return filing_date with precision=day.

Confidence rules:
  - 0.9-1.0: explicit unambiguous catalyst (e.g. "Met primary endpoint")
  - 0.6-0.8: clear catalyst but some ambiguity (e.g. interim data only)
  - 0.3-0.5: probably clinical but might be a non-catalyst
  - 0.0-0.2: very unclear, lean toward rejection

ONLY return the JSON object. No prose, no markdown fences."""


def normalize_one_staging_row(*, db, staging_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run the LLM on one staged row, return parsed JSON dict.
    Returns None on failure. Sets staging_row['_last_error'] on failure for
    diagnostics.
    """
    if not GOOGLE_API_KEY:
        staging_row["_last_error"] = "GOOGLE_API_KEY not set"
        return None

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        staging_row["_last_error"] = "google-genai not installed"
        return None

    prompt = NORMALIZER_PROMPT.format(
        ticker=staging_row.get("ticker") or "?",
        company=staging_row.get("ticker") or "?",
        filing_date=staging_row.get("filing_date") or "?",
        excerpt=(staging_row.get("raw_text_excerpt") or "")[:3000],
    )

    client = genai.Client(api_key=GOOGLE_API_KEY)
    # Disable BLOCK_MEDIUM_AND_ABOVE safety thresholds — clinical/regulatory
    # content (cancer drugs, gene editing, FDA hold letters, adverse events)
    # is precisely what we want to classify, and default Gemini safety filters
    # block it as "medical" / "dangerous content" surprisingly often. Using
    # BLOCK_NONE keeps medical content flowing; this is a research/classification
    # task, not a generative one with end-user output.
    # Defensive: skip if SafetySetting symbol isn't available in this genai
    # version (we'll still get default safety, just without our override).
    safety_settings = None
    try:
        SafetySetting = getattr(genai_types, "SafetySetting", None)
        if SafetySetting is not None:
            safety_settings = [
                SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in (
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_HARASSMENT",
                )
            ]
    except Exception as e:
        logger.warning(f"[normalizer] safety_settings setup failed (using defaults): {e}")
        safety_settings = None

    config_kwargs: Dict[str, Any] = {
        "max_output_tokens": 1500,  # plenty for the simplified schema
        "temperature": 0.0,  # zero — we want deterministic structured output
        # Force structured JSON output. Gemini's JSON mode constrains output
        # to the schema. We keep the schema small to reduce the surface area
        # for Gemini to hallucinate (e.g. infinite newlines in nullable
        # string fields, which we observed on Alzheimer's-disease excerpts).
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "is_clinical_catalyst": {"type": "boolean"},
                "catalyst_type": {
                    "type": "string",
                    "enum": [
                        "FDA Decision", "Phase 1 Readout", "Phase 2 Readout",
                        "Phase 3 Readout", "AdComm", "Submission",
                        "Trial Initiation", "Other", "NONE",
                    ],
                },
                "drug_name": {"type": "string"},
                "indication": {"type": "string"},
                "extracted_catalyst_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD format, or 'unknown'",
                },
                "date_precision": {
                    "type": "string",
                    "enum": ["exact", "day", "month", "quarter", "unknown"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0, "maximum": 1.0,
                },
                "reject_reason": {"type": "string"},
            },
            "required": [
                "is_clinical_catalyst", "catalyst_type", "date_precision",
                "confidence",
            ],
            "propertyOrdering": [
                "is_clinical_catalyst", "catalyst_type", "drug_name",
                "indication", "extracted_catalyst_date", "date_precision",
                "confidence", "reject_reason",
            ],
        },
        # Belt-and-suspenders: if Gemini gets stuck in a newline loop,
        # this aborts the generation. Observed pattern was 50+ consecutive
        # \n inside a string field on certain biomedical excerpts.
        "stop_sequences": ["\n\n\n\n\n"],
    }
    if safety_settings is not None:
        config_kwargs["safety_settings"] = safety_settings
    config = genai_types.GenerateContentConfig(**config_kwargs)

    t0 = time.time()
    status = "success"
    err_msg = None
    text = ""
    finish_reason: Optional[str] = None
    tokens_in = 0
    tokens_out = 0

    def _is_transient(msg: str) -> bool:
        m = msg.lower()
        return any(s in m for s in (
            "503", "unavailable", "500", "internal", "502", "bad gateway",
            "504", "deadline", "timeout", "connection reset",
            "429", "rate limit", "resource_exhausted",
        ))

    for attempt in (1, 2, 3):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL, contents=prompt, config=config,
            )
            # Capture finish_reason & token counts even on partial responses.
            try:
                cand = (response.candidates or [None])[0]
                if cand is not None and getattr(cand, "finish_reason", None) is not None:
                    finish_reason = str(cand.finish_reason)
            except Exception:
                pass
            try:
                um = getattr(response, "usage_metadata", None)
                if um is not None:
                    tokens_in = int(getattr(um, "prompt_token_count", 0) or 0)
                    tokens_out = int(getattr(um, "candidates_token_count", 0) or 0)
            except Exception:
                pass

            # response.text raises if no text content (SAFETY block, MAX_TOKENS
            # before any text, RECITATION, etc.). Wrap defensively.
            try:
                text = response.text or ""
            except Exception as te:
                text = ""
                # Try to pull text from candidate parts as a fallback
                try:
                    cand = (response.candidates or [None])[0]
                    if cand is not None and getattr(cand, "content", None):
                        for part in (cand.content.parts or []):
                            t = getattr(part, "text", None)
                            if t:
                                text = (text + t) if text else t
                except Exception:
                    pass
                if not text:
                    err_msg = (f"finish_reason={finish_reason} "
                               f"text_access_failed={str(te)[:160]}")[:300]

            if text:
                err_msg = None
                status = "success"
                break
            # No text — treat as error and decide whether to retry.
            status = "error"
            err_msg = err_msg or f"empty_response finish_reason={finish_reason}"
            # SAFETY / RECITATION / OTHER are not transient — don't retry
            if finish_reason and any(s in finish_reason for s in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST")):
                break
            # MAX_TOKENS is not transient either — already gave it the budget
            if finish_reason and "MAX_TOKENS" in finish_reason:
                break
            if attempt < 3:
                time.sleep(2.0 * attempt)
                continue
            break
        except Exception as e:
            status = "error"
            err_msg = str(e)[:300]
            if attempt < 3 and _is_transient(err_msg):
                time.sleep(2.0 * attempt)
                continue
            break

    elapsed_ms = int((time.time() - t0) * 1000)
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google", model=LLM_MODEL,
            tokens_input=tokens_in, tokens_output=tokens_out,
            duration_ms=elapsed_ms, status=status, error_message=err_msg,
            feature="backfill-normalizer",
            ticker=staging_row.get("ticker"),
        )
    except Exception as e:
        logger.warning(f"[normalizer] record_usage failed: {e}")

    if status != "success" or not text:
        # Gemini failed (503, SAFETY block, MAX_TOKENS, or runaway newline
        # generation that aborted via stop_sequences). Try OpenAI as
        # fallback — its JSON mode is more reliable on biomedical text.
        # Same cost (~$0.0003/call for gpt-4o-mini).
        openai_result = _try_openai_fallback(staging_row, prompt)
        if openai_result is not None:
            return openai_result
        # Bubble the error up so it ends up in reject_reason for diagnostics
        staging_row["_last_error"] = f"LLM call: {err_msg or 'empty response'}"
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Same fix as above — try OpenAI when Gemini emits invalid JSON
        # (the runaway-newline-in-Alzheimer's bug we observed empirically).
        logger.warning(f"[normalizer] JSON parse failed: {e}; trying OpenAI fallback")
        openai_result = _try_openai_fallback(staging_row, prompt)
        if openai_result is not None:
            return openai_result
        staging_row["_last_error"] = f"JSON parse: {str(e)[:100]}; text[:200]={cleaned[:200]}"
        return None

    return parsed


# ────────────────────────────────────────────────────────────
# OpenAI fallback (used when Gemini fails)
# ────────────────────────────────────────────────────────────

OPENAI_API_KEY_BACKFILL = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_BACKFILL = os.getenv("BACKFILL_OPENAI_MODEL", "gpt-4o-mini")


def _try_openai_fallback(staging_row: Dict[str, Any], prompt: str) -> Optional[Dict[str, Any]]:
    """OpenAI gpt-4o-mini fallback for the normalizer. Triggered when
    Gemini fails (503s, runaway-newline JSON corruption, SAFETY blocks).

    Uses OpenAI's response_format=json_object which is more reliable than
    Gemini's response_schema for this kind of biomedical text. Same cost
    bracket (~$0.0003 per call).
    """
    if not OPENAI_API_KEY_BACKFILL:
        return None

    import requests as _requests
    body = {
        "model": OPENAI_MODEL_BACKFILL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY_BACKFILL}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    status = "success"
    err_msg = None
    tokens_in = 0
    tokens_out = 0
    text = ""
    try:
        r = _requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=body, headers=headers, timeout=30,
        )
        r.raise_for_status()
        resp = r.json()
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
    except Exception as e:
        status = "error"
        err_msg = str(e)[:300]
        logger.warning(f"[normalizer] OpenAI fallback failed: {err_msg}")
    finally:
        elapsed_ms = int((time.time() - t0) * 1000)
        try:
            from services.llm_usage import record_usage
            record_usage(
                provider="openai", model=OPENAI_MODEL_BACKFILL,
                feature="backfill-normalizer-fallback",
                ticker=staging_row.get("ticker"),
                tokens_input=tokens_in, tokens_output=tokens_out,
                duration_ms=elapsed_ms, status=status, error_message=err_msg,
            )
        except Exception:
            pass

    if status != "success" or not text:
        return None

    try:
        parsed = json.loads(text)
        # Tag which provider succeeded for downstream diagnostics
        if isinstance(parsed, dict):
            parsed["_normalizer_provider"] = "openai"
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"[normalizer] OpenAI returned invalid JSON: {e}")
        return None


def find_existing_catalyst(*, db, ticker: str, catalyst_type: str,
                            catalyst_date: str, window_days: int = 7,
                            ) -> Optional[int]:
    """Check catalyst_universe for a duplicate. Returns id of existing
    catalyst if found, else None.

    Match rule: same ticker + same catalyst_type + catalyst_date within
    ±window_days. We use ±7 because date_precision can be 'day' or 'month'
    on either side; same event might be staged with slightly different dates.
    """
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM catalyst_universe
                WHERE ticker = %s
                  AND catalyst_type = %s
                  AND status = 'active'
                  AND ABS(EXTRACT(EPOCH FROM (catalyst_date::timestamp - %s::timestamp)) / 86400) <= %s
                LIMIT 1
            """, (ticker, catalyst_type, catalyst_date, window_days))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning(f"find_existing_catalyst error: {e}")
        return None


def insert_into_catalyst_universe(*, db, normalized: Dict[str, Any],
                                    staging: Dict[str, Any]) -> Optional[int]:
    """Insert a new row into catalyst_universe from normalized data.
    Returns the new catalyst_universe.id, or None on failure.

    Schema match (from universe_seeder pattern):
      - column is `confidence_score` (not `confidence`)
      - has `canonical_drug_name` for dedupe via ON CONFLICT
      - includes `company_name`, `description`, `phase`, `source_url`
      - uses `last_updated` only (no `created_at` column)

    Sanitizes LLM outputs:
      - catalyst_type 'NONE'/'None'/empty → reject (return None)
      - extracted_catalyst_date 'unknown'/'None'/non-ISO → fall back to
        staging.catalyst_date (the filing date)
    """
    try:
        ticker = staging.get("ticker")
        cat_type = (normalized.get("catalyst_type") or "").strip()
        if not cat_type or cat_type.upper() in ("NONE", "NULL"):
            logger.info(f"insert_into_catalyst_universe: invalid cat_type='{cat_type}' for {ticker}, skipping")
            return None

        # Date handling
        raw_date = normalized.get("extracted_catalyst_date")
        cat_date = None
        if isinstance(raw_date, str):
            r = raw_date.strip()
            if r and r.upper() not in ("UNKNOWN", "NONE", "NULL", ""):
                if len(r) >= 10:
                    try:
                        datetime.strptime(r[:10], "%Y-%m-%d")
                        cat_date = r[:10]
                    except ValueError:
                        cat_date = None
                elif len(r) == 7:
                    try:
                        datetime.strptime(r + "-01", "%Y-%m-%d")
                        cat_date = r + "-01"
                    except ValueError:
                        cat_date = None
                elif len(r) == 4 and r.isdigit():
                    cat_date = r + "-01-01"
        if not cat_date:
            sd = staging.get("catalyst_date")
            if sd:
                cat_date = sd.isoformat() if hasattr(sd, "isoformat") else str(sd)[:10]
        if not cat_date:
            return None

        date_prec = (normalized.get("date_precision") or "day").lower()
        if date_prec not in ("exact", "day", "month", "quarter", "unknown"):
            date_prec = "day"

        drug = normalized.get("drug_name")
        if isinstance(drug, str):
            drug = drug.strip() or None
            if drug and drug.upper() in ("NONE", "NULL", "UNKNOWN"):
                drug = None
        # Compute canonical_drug_name (lowercase, strip suffixes) for dedupe
        canonical_drug = None
        if drug:
            canonical_drug = drug.lower().strip()
            # Strip trade-name parens like "FDA Approval of TYBOST (cobicistat)"
            if "(" in canonical_drug:
                canonical_drug = canonical_drug.split("(")[0].strip()

        indication = normalized.get("indication")
        if isinstance(indication, str):
            indication = indication.strip() or None
            if indication and indication.upper() in ("NONE", "NULL", "UNKNOWN"):
                indication = None

        confidence_score = float(normalized.get("confidence") or 0.5)
        source_url = staging.get("source_url")

        # Description: short LLM-derived blurb. Use reject_reason if it has
        # content, else cat_type as fallback.
        description = normalized.get("reject_reason") or cat_type

        with db.get_conn() as conn:
            cur = conn.cursor()
            # Use the same INSERT shape as universe_seeder, with ON CONFLICT
            # on the canonical-drug uniqueness constraint to avoid dupes.
            cur.execute("""
                INSERT INTO catalyst_universe (
                    ticker, company_name, catalyst_type, catalyst_date, date_precision,
                    description, drug_name, canonical_drug_name, indication, phase,
                    source, source_url, confidence_score, status, last_updated
                )
                VALUES (
                    %s, NULL, %s, %s, %s,
                    %s, %s, %s, %s, NULL,
                    'edgar_backfill', %s, %s, 'active', NOW()
                )
                ON CONFLICT (ticker, catalyst_type, catalyst_date, canonical_drug_name)
                WHERE canonical_drug_name IS NOT NULL AND status = 'active'
                DO NOTHING
                RETURNING id
            """, (
                ticker, cat_type, cat_date, date_prec,
                description, drug, canonical_drug, indication,
                source_url, confidence_score,
            ))
            row = cur.fetchone()
            conn.commit()
            if row is None:
                # ON CONFLICT DO NOTHING — duplicate found
                logger.info(f"catalyst_universe: dupe on {ticker}/{cat_type}/{cat_date}/{canonical_drug}")
                # Find the existing row id so we can link it
                cur.execute("""
                    SELECT id FROM catalyst_universe
                    WHERE ticker = %s AND catalyst_type = %s
                      AND catalyst_date = %s AND canonical_drug_name = %s
                      AND status = 'active'
                    LIMIT 1
                """, (ticker, cat_type, cat_date, canonical_drug))
                ex = cur.fetchone()
                return ex[0] if ex else None
            return row[0]
    except Exception as e:
        logger.warning(f"insert_into_catalyst_universe failed for {staging.get('ticker')}: {e}")
        return None


def normalize_pending_batch(*, db, batch_size: int = 50) -> Dict[str, int]:
    """Process up to batch_size pending staging rows. Returns counters.

    Idempotent — only touches rows with status='pending'.
    """
    counts = {
        "scanned": 0, "accepted": 0, "rejected": 0,
        "unclear": 0, "duplicate": 0, "errored": 0,
    }

    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, source, source_id, ticker, cik, filing_date,
                       catalyst_date, raw_title, raw_text_excerpt, source_url
                FROM catalyst_backfill_staging
                WHERE status = 'pending'
                ORDER BY filing_date DESC NULLS LAST
                LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall()

        for row in rows:
            (sid, source, source_id, ticker, cik, filing_date, cat_date,
             title, excerpt, source_url) = row
            counts["scanned"] += 1

            staging_dict = {
                "id": sid, "source": source, "source_id": source_id,
                "ticker": ticker, "cik": cik, "filing_date": filing_date,
                "catalyst_date": cat_date, "raw_title": title,
                "raw_text_excerpt": excerpt, "source_url": source_url,
            }

            try:
                normalized = normalize_one_staging_row(db=db, staging_row=staging_dict)
                if not normalized:
                    counts["errored"] += 1
                    last_err = (staging_dict.get("_last_error") or "LLM call failed")[:500]
                    # Mark errored so we don't retry forever
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'unclear', processed_at = now(),
                                reject_reason = %s
                            WHERE id = %s
                        """, (last_err, sid))
                        conn.commit()
                    continue

                # Decide based on LLM output
                if not normalized.get("is_clinical_catalyst"):
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'rejected', processed_at = now(),
                                reject_reason = %s,
                                normalized_json = %s
                            WHERE id = %s
                        """, (
                            (normalized.get("reject_reason") or "")[:500],
                            json.dumps(normalized), sid,
                        ))
                        conn.commit()
                    counts["rejected"] += 1
                    continue

                conf = float(normalized.get("confidence") or 0.0)
                if conf < 0.5:
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'unclear', processed_at = now(),
                                normalized_json = %s
                            WHERE id = %s
                        """, (json.dumps(normalized), sid))
                        conn.commit()
                    counts["unclear"] += 1
                    continue

                # Check for duplicate
                cat_type = normalized.get("catalyst_type")
                cat_date_extracted = (
                    normalized.get("extracted_catalyst_date") or filing_date
                )
                existing_id = find_existing_catalyst(
                    db=db, ticker=ticker, catalyst_type=cat_type,
                    catalyst_date=str(cat_date_extracted),
                )
                if existing_id:
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'duplicate', processed_at = now(),
                                catalyst_id = %s, normalized_json = %s
                            WHERE id = %s
                        """, (existing_id, json.dumps(normalized), sid))
                        conn.commit()
                    counts["duplicate"] += 1
                    continue

                # Insert into catalyst_universe
                new_id = insert_into_catalyst_universe(
                    db=db, normalized=normalized, staging=staging_dict,
                )
                if new_id:
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'accepted', processed_at = now(),
                                catalyst_id = %s, normalized_json = %s
                            WHERE id = %s
                        """, (new_id, json.dumps(normalized), sid))
                        conn.commit()
                    counts["accepted"] += 1
                else:
                    # INSERT failed — mark unclear with reject_reason so we
                    # don't keep retrying. Most common cause: cat_type='NONE'
                    # or extracted_catalyst_date sentinel. Storing the
                    # normalized JSON helps debug.
                    counts["errored"] += 1
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'unclear', processed_at = now(),
                                reject_reason = %s,
                                normalized_json = %s
                            WHERE id = %s
                        """, (
                            "INSERT into catalyst_universe failed (see normalized_json)",
                            json.dumps(normalized), sid,
                        ))
                        conn.commit()
            except Exception as e:
                counts["errored"] += 1
                logger.exception(f"normalize_pending_batch row id={sid}: {e}")
                # Mark as unclear with the exception message
                try:
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'unclear', processed_at = now(),
                                reject_reason = %s
                            WHERE id = %s
                        """, (f"Exception: {str(e)[:400]}", sid))
                        conn.commit()
                except Exception:
                    pass
    except Exception as e:
        logger.exception(f"normalize_pending_batch top-level: {e}")

    return counts
