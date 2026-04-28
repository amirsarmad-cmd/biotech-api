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

NON-catalysts (always return is_clinical_catalyst=false):
  - Quarterly earnings reports
  - Stock offerings, financings, dilution
  - M&A, acquisitions, partnerships (unless tied to a clinical milestone)
  - Management changes (CEO/CMO/CSO transitions) UNLESS the change is
    co-announced with clinical news in the same filing
  - Investor presentations, conferences (unless they include readout data)
  - Insider trading filings, registration statements

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
    Returns None on failure."""
    if not GOOGLE_API_KEY:
        return None

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return None

    prompt = NORMALIZER_PROMPT.format(
        ticker=staging_row.get("ticker") or "?",
        company=staging_row.get("ticker") or "?",
        filing_date=staging_row.get("filing_date") or "?",
        excerpt=(staging_row.get("raw_text_excerpt") or "")[:3000],
    )

    client = genai.Client(api_key=GOOGLE_API_KEY)
    config = genai_types.GenerateContentConfig(
        max_output_tokens=1500,
        temperature=0.1,
    )

    t0 = time.time()
    status = "success"
    err_msg = None
    text = ""
    saw_503 = False
    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL, contents=prompt, config=config,
            )
            text = response.text or ""
            err_msg = None
            status = "success"
            break
        except Exception as e:
            status = "error"
            err_msg = str(e)[:300]
            if "503" in err_msg and "UNAVAILABLE" in err_msg:
                saw_503 = True
            if attempt == 1 and saw_503:
                time.sleep(3.0)
                continue
            break

    elapsed = time.time() - t0
    try:
        from services.llm_usage import record_usage
        record_usage(
            provider="google", model=LLM_MODEL,
            tokens_in=0, tokens_out=0,
            latency_sec=elapsed, status=status, error=err_msg,
            context=f"backfill-normalizer:{staging_row.get('ticker')}",
        )
    except Exception:
        pass

    if status != "success" or not text:
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
        logger.warning(f"[normalizer] JSON parse failed: {e}")
        return None

    return parsed


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
    Returns the new catalyst_universe.id, or None on failure."""
    try:
        ticker = staging.get("ticker")
        cat_type = normalized.get("catalyst_type")
        cat_date = normalized.get("extracted_catalyst_date") or staging.get("catalyst_date")
        date_prec = normalized.get("date_precision") or "day"
        drug = normalized.get("drug_name")
        indication = normalized.get("indication")
        confidence = float(normalized.get("confidence") or 0.5)

        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO catalyst_universe (
                    ticker, catalyst_type, catalyst_date,
                    date_precision, drug_name, indication,
                    confidence, source, status,
                    created_at, last_updated
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    'edgar_backfill', 'active', now(), now()
                )
                RETURNING id
            """, (
                ticker, cat_type, cat_date,
                date_prec, drug, indication, confidence,
            ))
            new_id = cur.fetchone()[0]
            conn.commit()
        return new_id
    except Exception as e:
        logger.warning(f"insert_into_catalyst_universe failed: {e}")
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
                    # Mark errored so we don't retry forever
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE catalyst_backfill_staging
                            SET status = 'unclear', processed_at = now(),
                                reject_reason = 'LLM call failed'
                            WHERE id = %s
                        """, (sid,))
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
                    counts["errored"] += 1
            except Exception as e:
                counts["errored"] += 1
                logger.exception(f"normalize_pending_batch row id={sid}: {e}")
    except Exception as e:
        logger.exception(f"normalize_pending_batch top-level: {e}")

    return counts
