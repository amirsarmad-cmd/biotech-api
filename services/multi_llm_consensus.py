"""Multi-LLM consensus orchestrator.

Architecture (project_news_consensus_scope.md, revised plan):

  Pass 1 — Gemini scout via grounded search.
           Returns label JSON + grounding chunks (URLs Gemini consulted).
           We persist the chunks to catalyst_event_news so all readers
           see the same enriched library.

  Pass 2 — Three readers in parallel reading the SAME library:
             Sonnet 4.6, GPT-5.5 (gpt-4o fallback), Gemini 3 (2.5-pro fallback)
           Each returns the same JSON-schema label.

  Pass 3 — Opus 4.7 arbiter.
           Reads library + Pass-1 verdict + 3 Pass-2 verdicts → final
           consensus class + confidence + reasoning. Opus's verdict IS
           the consensus.

Costs (one full consensus per event):
  Pass 1 Gemini Flash grounded   ~$0.0003
  Sonnet reader                  ~$0.005-0.010
  GPT-5.5 reader                 ~$0.010-0.020
  Gemini 3 reader                ~$0.005-0.010
  Opus arbiter                   ~$0.025-0.040  (≈70% of total)
                                 ─────────────
  Total                          ~$0.045-0.080 per event

Set OPUS_OPT_OUT=1 to skip the arbiter and majority-vote across the 3
readers (cuts cost ~70%).

Idempotency: orchestrator only re-calls LLMs that returned NULL/errored
on a prior attempt; per-LLM JSONB columns are checkpoints.
"""
from __future__ import annotations
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.llm_gateway import (
    llm_call, LLMAllProvidersFailed, LLMResult,
)
from services.outcome_labeler import OUTCOME_LABELER_PROMPT
from services.news_library import insert_news_row, normalize_url

logger = logging.getLogger(__name__)


# ── Library load ─────────────────────────────────────────────

LIBRARY_WINDOW_DAYS = 30  # ±30d around catalyst_date
LIBRARY_MAX_ITEMS = 30


def load_library_for_event(
    db_conn, ticker: str, catalyst_date: str
) -> List[Dict[str, Any]]:
    """Pull the news library slice for a specific event.

    Selects from catalyst_event_news where ticker matches AND
    published_at is within ±30d of catalyst_date. Orders by proximity
    to the catalyst, not chronologically — so the LIMIT cap doesn't
    bias toward the late edge of the window.
    """
    sql = """
        SELECT source, headline, summary, body, published_at, url
        FROM catalyst_event_news
        WHERE ticker = %s
          AND published_at IS NOT NULL
          AND published_at BETWEEN %s::timestamptz AND %s::timestamptz
        ORDER BY ABS(EXTRACT(EPOCH FROM (published_at - %s::timestamptz))) ASC
        LIMIT %s
    """
    try:
        cd = datetime.fromisoformat(catalyst_date) if "T" in catalyst_date else \
             datetime.strptime(catalyst_date[:10], "%Y-%m-%d")
        cd = cd.replace(tzinfo=timezone.utc) if cd.tzinfo is None else cd
    except Exception:
        return []
    lo = (cd - timedelta(days=LIBRARY_WINDOW_DAYS)).isoformat()
    hi = (cd + timedelta(days=LIBRARY_WINDOW_DAYS)).isoformat()
    with db_conn.cursor() as cur:
        cur.execute(sql, (ticker.upper(), lo, hi, cd.isoformat(), LIBRARY_MAX_ITEMS))
        rows = cur.fetchall()
    out = []
    for source, headline, summary, body, pub, url in rows:
        out.append({
            "source": source,
            "headline": headline,
            "summary": summary or "",
            "body": (body[:4000] if body else ""),  # cap per-item to keep prompt size bounded
            "published_at": pub.isoformat() if pub else "",
            "url": url,
        })
    return out


def format_library_for_prompt(items: List[Dict[str, Any]]) -> str:
    """Render the library as a numbered list for inline LLM consumption."""
    if not items:
        return "(no articles in library)"
    lines = []
    for i, it in enumerate(items, 1):
        date = (it.get("published_at") or "")[:10]
        src = it.get("source", "?")
        headline = it.get("headline", "")[:300]
        url = it.get("url", "")
        body = it.get("body") or ""
        summary = it.get("summary") or ""
        excerpt = (body[:600] if body else summary[:300]).strip()
        block = f"[{i}] [{src} {date}] {headline}\n  {url}"
        if excerpt:
            block += f"\n  {excerpt}"
        lines.append(block)
    return "\n\n".join(lines)


# ── Prompts ──────────────────────────────────────────────────

# Note: braces are doubled because this string gets .format()'d after concat.
_OUTCOME_SCHEMA = """{{
  "outcome_class": "APPROVED" | "REJECTED" | "MET_ENDPOINT" | "MISSED_ENDPOINT" | "DELAYED" | "WITHDRAWN" | "MIXED" | "UNKNOWN",
  "endpoint_met": true | false | null,
  "approval_granted": true | false | null,
  "safety_signal_flag": true | false,
  "primary_source_url": "<one of the URLs in the library>",
  "evidence": "<short verbatim quote from the source (under 50 words)>",
  "confidence": 0.0 - 1.0,
  "reasoning": "<1-2 sentences explaining the classification>"
}}"""

LIBRARY_LABELER_PROMPT = """You are a biotech outcome classifier. Below is a catalyst event and a news library of articles published around that catalyst date. Read the library carefully and classify what happened.

CATALYST:
  Ticker: {ticker}
  Company: {company}
  Catalyst type: {catalyst_type}
  Catalyst date: {catalyst_date}
  Drug: {drug}
  Indication: {indication}

NEWS LIBRARY (n={library_size}):
{numbered_library}

Classification rules:
  - APPROVED: FDA/regulatory approval granted (catalyst_type involves PDUFA / NDA / BLA / approval).
  - REJECTED: CRL issued, refusal to file, withdrawal of application by company.
  - MET_ENDPOINT: Phase 2/3 trial hit primary endpoint with statistical significance.
  - MISSED_ENDPOINT: Phase 2/3 trial failed primary endpoint (p > 0.05 or trend toward placebo).
  - DELAYED: PDUFA pushed back, advisory committee postponed, no decision yet.
  - WITHDRAWN: Company withdrew application before FDA decision.
  - MIXED: Met some endpoints but not others, or approved with significant restrictions.
  - UNKNOWN: Library does not contain enough information for a confident call.

The primary_source_url MUST be one of the URLs listed in the library.

Return ONLY valid JSON in this exact schema (no markdown, no commentary):

""" + _OUTCOME_SCHEMA


ARBITER_PROMPT = """You are a senior biotech outcome reviewer arbitrating a multi-LLM consensus. Read the catalyst, the news library, the verdict from a grounded-search agent (Pass 1), and verdicts from three independent agents that read the same library (Pass 2). Synthesize a final consensus.

CATALYST:
  Ticker: {ticker}
  Company: {company}
  Catalyst type: {catalyst_type}
  Catalyst date: {catalyst_date}
  Drug: {drug}
  Indication: {indication}

NEWS LIBRARY (n={library_size}):
{numbered_library}

PASS 1 (grounded-search scout — Gemini):
{pass1_json}

PASS 2 READER VERDICTS:
1. Sonnet:    {sonnet_json}
2. GPT-5.5:   {gpt55_json}
3. Gemini-3:  {gemini3_json}

Your job:
- Where the readers agree, use that class with high confidence.
- Where they disagree, weigh evidence in the library and the strength of each reader's reasoning, then render your own judgment.
- If the library is thin or evidence weak, return UNKNOWN.

Return ONLY valid JSON in this schema (no markdown, no commentary):

{{
  "outcome_class": "APPROVED" | "REJECTED" | "MET_ENDPOINT" | "MISSED_ENDPOINT" | "DELAYED" | "WITHDRAWN" | "MIXED" | "UNKNOWN",
  "endpoint_met": true | false | null,
  "approval_granted": true | false | null,
  "safety_signal_flag": true | false,
  "primary_source_url": "<one URL from the library>",
  "evidence": "<short verbatim quote>",
  "confidence": 0.0 - 1.0,
  "agreement_with_readers": "unanimous" | "majority" | "split",
  "reasoning": "<2-4 sentences explaining the synthesis, especially how disagreement was resolved>"
}}"""


# ── Pass 1: Gemini grounded scout ────────────────────────────

def run_pass1_gemini_scout(
    *, ticker: str, company: str, catalyst_type: str, catalyst_date: str,
    drug: Optional[str], indication: Optional[str],
) -> Tuple[Optional[Dict], Optional[List[Dict]], Optional[List[str]]]:
    """Returns (label_dict | None, grounding_chunks, search_queries)."""
    prompt = OUTCOME_LABELER_PROMPT.format(
        ticker=ticker, company=company or ticker,
        catalyst_type=catalyst_type or "unknown",
        catalyst_date=catalyst_date,
        drug=drug or "(unspecified)",
        indication=indication or "(unspecified)",
    )
    try:
        result = llm_call(
            capability="grounded_search",
            feature="consensus_pass1_scout",
            prompt=prompt,
            ticker=ticker,
            model_overrides={"google": ["gemini-2.5-flash", "gemini-2.5-pro"]},
            timeout_s=60.0,
            max_tokens=2000,
            temperature=0.1,
        )
    except LLMAllProvidersFailed as e:
        logger.warning(f"[consensus] pass1 ALL_PROVIDERS_FAILED for {ticker}: {e}")
        return None, None, None
    except Exception as e:
        logger.warning(f"[consensus] pass1 unexpected error for {ticker}: {e}")
        return None, None, None
    return result.parsed_json, result.grounding_chunks, result.search_queries


# ── Pass 2: 3 readers in parallel ────────────────────────────

_READER_CONFIGS = [
    {"name": "sonnet",
     "feature": "consensus_pass2_sonnet",
     "model_overrides": {"anthropic": ["claude-sonnet-4-6"]},
     "fallback_chain": ["anthropic"]},
    {"name": "gpt55",
     "feature": "consensus_pass2_gpt55",
     "model_overrides": {"openai": ["gpt-5.5", "gpt-4o"]},
     "fallback_chain": ["openai"]},
    {"name": "gemini3",
     "feature": "consensus_pass2_gemini3",
     "model_overrides": {"google": ["gemini-3.0-pro", "gemini-2.5-pro"]},
     "fallback_chain": ["google"]},
]


def _run_one_reader(reader_cfg: Dict, prompt: str, ticker: str) -> Tuple[str, Optional[Dict]]:
    """Returns (reader_name, parsed_json | None)."""
    try:
        result = llm_call(
            capability="text_json",
            feature=reader_cfg["feature"],
            prompt=prompt,
            ticker=ticker,
            model_overrides=reader_cfg["model_overrides"],
            fallback_chain=reader_cfg["fallback_chain"],
            timeout_s=60.0,
            max_tokens=1500,
            temperature=0.2,
        )
        return reader_cfg["name"], result.parsed_json
    except LLMAllProvidersFailed as e:
        logger.warning(f"[consensus] reader {reader_cfg['name']} failed for {ticker}: {e}")
        return reader_cfg["name"], None
    except Exception as e:
        logger.warning(f"[consensus] reader {reader_cfg['name']} unexpected error for {ticker}: {e}")
        return reader_cfg["name"], None


def run_pass2_readers_parallel(
    *, ticker: str, company: str, catalyst_type: str, catalyst_date: str,
    drug: Optional[str], indication: Optional[str],
    library: List[Dict],
) -> Dict[str, Optional[Dict]]:
    """Run all 3 readers concurrently. Returns {reader_name: parsed_json | None}."""
    prompt = LIBRARY_LABELER_PROMPT.format(
        ticker=ticker, company=company or ticker,
        catalyst_type=catalyst_type or "unknown",
        catalyst_date=catalyst_date,
        drug=drug or "(unspecified)",
        indication=indication or "(unspecified)",
        library_size=len(library),
        numbered_library=format_library_for_prompt(library),
    )
    out: Dict[str, Optional[Dict]] = {}
    with ThreadPoolExecutor(max_workers=len(_READER_CONFIGS)) as ex:
        futs = {ex.submit(_run_one_reader, cfg, prompt, ticker): cfg["name"]
                for cfg in _READER_CONFIGS}
        for fut in as_completed(futs):
            name, parsed = fut.result()
            out[name] = parsed
    return out


# ── Pass 3: Opus arbiter ─────────────────────────────────────

def run_pass3_opus_arbiter(
    *, ticker: str, company: str, catalyst_type: str, catalyst_date: str,
    drug: Optional[str], indication: Optional[str],
    library: List[Dict],
    pass1_json: Optional[Dict],
    sonnet_json: Optional[Dict], gpt55_json: Optional[Dict],
    gemini3_json: Optional[Dict],
) -> Optional[Dict]:
    prompt = ARBITER_PROMPT.format(
        ticker=ticker, company=company or ticker,
        catalyst_type=catalyst_type or "unknown",
        catalyst_date=catalyst_date,
        drug=drug or "(unspecified)",
        indication=indication or "(unspecified)",
        library_size=len(library),
        numbered_library=format_library_for_prompt(library),
        pass1_json=json.dumps(pass1_json) if pass1_json else "null",
        sonnet_json=json.dumps(sonnet_json) if sonnet_json else "null",
        gpt55_json=json.dumps(gpt55_json) if gpt55_json else "null",
        gemini3_json=json.dumps(gemini3_json) if gemini3_json else "null",
    )
    try:
        result = llm_call(
            capability="text_json",
            feature="consensus_pass3_opus",
            prompt=prompt,
            ticker=ticker,
            model_overrides={"anthropic": ["claude-opus-4-7"]},
            fallback_chain=["anthropic"],
            timeout_s=120.0,
            max_tokens=2000,
            temperature=0.1,
        )
        return result.parsed_json
    except LLMAllProvidersFailed as e:
        logger.warning(f"[consensus] pass3 (Opus) failed for {ticker}: {e}")
        return None
    except Exception as e:
        logger.warning(f"[consensus] pass3 unexpected error for {ticker}: {e}")
        return None


# ── Persistence ──────────────────────────────────────────────

def _persist_grounding_chunks(
    db_conn, ticker: str, catalyst_date: str,
    chunks: List[Dict],
) -> int:
    """Write Gemini's grounded-search URLs into catalyst_event_news so
    they're visible to subsequent readers / re-runs."""
    inserted = 0
    for c in chunks or []:
        uri = (c.get("uri") or "").strip()
        title = (c.get("title") or "").strip()
        if not uri or not title:
            continue
        res = insert_news_row(
            db_conn,
            ticker=ticker,
            source="gemini_grounded",
            url=uri,
            headline=title[:400],
            summary=None,
            body=None,
            published_at=None,
            catalyst_date=catalyst_date,
            discovery_path="gemini_grounded_search",
            ticker_mention_method="grounded_attribution",
        )
        if res and res.get("action") == "inserted":
            inserted += 1
    return inserted


def _build_consensus_json(
    pass1: Optional[Dict],
    sonnet: Optional[Dict], gpt55: Optional[Dict], gemini3: Optional[Dict],
    opus: Optional[Dict],
    library_size: int,
) -> Dict:
    """Build the persisted outcome_label_consensus_json blob."""
    votes = []
    for label, j in [("google/gemini-2.5-flash", pass1),
                     ("anthropic/claude-sonnet-4-6", sonnet),
                     ("openai/gpt-5.5", gpt55),
                     ("google/gemini-3.0-pro", gemini3),
                     ("anthropic/claude-opus-4-7", opus)]:
        if j:
            votes.append({
                "llm": label,
                "class": j.get("outcome_class"),
                "confidence": j.get("confidence"),
                "primary_source_url": j.get("primary_source_url"),
            })
    reader_classes = [v["class"] for v in votes
                      if v["llm"].startswith(("anthropic/claude-sonnet", "openai/gpt", "google/gemini-3"))
                      and v["class"]]
    if reader_classes:
        # majority class among the 3 readers (if 2-2 split, pick mode arbitrarily)
        from collections import Counter
        c = Counter(reader_classes)
        top, top_n = c.most_common(1)[0]
        agreement_pct = top_n / len(reader_classes)
    else:
        agreement_pct = 0.0
    return {
        "votes": votes,
        "majority_reader_class": (Counter(reader_classes).most_common(1)[0][0]
                                  if reader_classes else None),
        "reader_agreement_pct": agreement_pct,
        "library_size": library_size,
        "library_window": f"[-{LIBRARY_WINDOW_DAYS}d, +{LIBRARY_WINDOW_DAYS}d]",
        "votes_received": len(votes),
    }


def _persist_consensus(
    db_conn, *, outcome_id: int,
    pass1: Optional[Dict], grounding_chunks: Optional[List[Dict]],
    sonnet: Optional[Dict], gpt55: Optional[Dict],
    gemini3: Optional[Dict], opus: Optional[Dict],
    library_size: int,
) -> None:
    """Single transactional UPDATE writing all per-LLM JSONs + consensus."""
    consensus_json = _build_consensus_json(
        pass1, sonnet, gpt55, gemini3, opus, library_size,
    )
    # Final consensus_class is Opus's if available; otherwise majority-reader fallback
    if opus:
        consensus_class = opus.get("outcome_class")
        opus_conf = float(opus.get("confidence") or 0.5)
        consensus_confidence = round(
            opus_conf * (consensus_json["reader_agreement_pct"] or 1.0), 4
        )
    else:
        consensus_class = consensus_json.get("majority_reader_class")
        consensus_confidence = round(consensus_json["reader_agreement_pct"] * 0.7, 4) \
            if consensus_class else None

    with db_conn.cursor() as cur:
        cur.execute("""
            UPDATE post_catalyst_outcomes SET
              outcome_label_sonnet_json   = COALESCE(%s::jsonb, outcome_label_sonnet_json),
              outcome_label_sonnet_class  = COALESCE(%s, outcome_label_sonnet_class),
              outcome_label_gpt55_json    = COALESCE(%s::jsonb, outcome_label_gpt55_json),
              outcome_label_gpt55_class   = COALESCE(%s, outcome_label_gpt55_class),
              outcome_label_gemini3_json  = COALESCE(%s::jsonb, outcome_label_gemini3_json),
              outcome_label_gemini3_class = COALESCE(%s, outcome_label_gemini3_class),
              outcome_label_opus_json     = COALESCE(%s::jsonb, outcome_label_opus_json),
              outcome_label_opus_class    = COALESCE(%s, outcome_label_opus_class),
              outcome_label_pass1_grounding_json = COALESCE(%s::jsonb, outcome_label_pass1_grounding_json),
              outcome_label_consensus_json = %s::jsonb,
              outcome_label_consensus_class = %s,
              outcome_label_consensus_confidence = %s,
              outcome_label_consensus_at = NOW(),
              news_library_size_at_consensus = %s
            WHERE id = %s
        """, (
            json.dumps(sonnet) if sonnet else None,
            sonnet.get("outcome_class") if sonnet else None,
            json.dumps(gpt55) if gpt55 else None,
            gpt55.get("outcome_class") if gpt55 else None,
            json.dumps(gemini3) if gemini3 else None,
            gemini3.get("outcome_class") if gemini3 else None,
            json.dumps(opus) if opus else None,
            opus.get("outcome_class") if opus else None,
            json.dumps({"chunks": grounding_chunks or [], "pass1_label": pass1})
                if (grounding_chunks or pass1) else None,
            json.dumps(consensus_json),
            consensus_class,
            consensus_confidence,
            library_size,
            outcome_id,
        ))


# ── Public entry point ───────────────────────────────────────

def consensus_label_for_outcome(db, outcome_id: int) -> Optional[Dict]:
    """Run the full consensus pipeline for one post_catalyst_outcomes row.

    `db` is a BiotechDatabase-like object (has `.get_conn()` returning a
    psycopg2 connection context manager). Returns the consensus JSON
    (with votes_received, library_size, etc.) on success; None when
    fewer than 2 readers responded (row stays retry-eligible).

    Idempotency: per-LLM JSONB columns act as checkpoints. If a row has
    e.g. sonnet_json already populated from a prior run, we skip that
    reader. If all 4 LLMs already done, we just rebuild consensus from
    existing data (no new LLM calls).
    """
    opus_opt_out = os.getenv("OPUS_OPT_OUT", "").strip().lower() in ("1", "true", "yes")

    with db.get_conn() as conn:
        # ── Load row metadata + existing checkpoints ──
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pco.ticker, pco.catalyst_type, pco.catalyst_date,
                       pco.drug_name, pco.indication, s.company_name,
                       pco.outcome_label_sonnet_json,
                       pco.outcome_label_gpt55_json,
                       pco.outcome_label_gemini3_json,
                       pco.outcome_label_opus_json,
                       pco.outcome_label_pass1_grounding_json
                FROM post_catalyst_outcomes pco
                LEFT JOIN screener_stocks s ON s.ticker = pco.ticker
                WHERE pco.id = %s
            """, (outcome_id,))
            row = cur.fetchone()
        if not row:
            return None
        (ticker, cat_type, cat_date, drug, indication, company,
         sonnet_existing, gpt55_existing, gemini3_existing,
         opus_existing, pass1_existing) = row
        cat_str = (cat_date.strftime("%Y-%m-%d")
                   if hasattr(cat_date, "strftime") else str(cat_date)[:10])

        # ── Pass 1 — Gemini scout (skip if existing) ──
        if pass1_existing:
            try:
                p1 = pass1_existing.get("pass1_label") if isinstance(pass1_existing, dict) else None
                grounding_chunks = (pass1_existing.get("chunks") if isinstance(pass1_existing, dict) else None) or []
            except Exception:
                p1 = None
                grounding_chunks = []
        else:
            p1, grounding_chunks, _queries = run_pass1_gemini_scout(
                ticker=ticker, company=company,
                catalyst_type=cat_type, catalyst_date=cat_str,
                drug=drug, indication=indication,
            )
            # Persist discovered URLs to the news library (so subsequent readers see them)
            if grounding_chunks:
                _persist_grounding_chunks(conn, ticker, cat_str, grounding_chunks)
                conn.commit()

        # ── Library load (after Pass-1 enrichment) ──
        library = load_library_for_event(conn, ticker, cat_str)
        library_size = len(library)

        # ── Pass 2 — readers in parallel (skip cached ones) ──
        existing = {"sonnet": sonnet_existing, "gpt55": gpt55_existing,
                    "gemini3": gemini3_existing}
        cfgs_to_run = [c for c in _READER_CONFIGS if not existing.get(c["name"])]
        if cfgs_to_run:
            prompt = LIBRARY_LABELER_PROMPT.format(
                ticker=ticker, company=company or ticker,
                catalyst_type=cat_type or "unknown", catalyst_date=cat_str,
                drug=drug or "(unspecified)",
                indication=indication or "(unspecified)",
                library_size=library_size,
                numbered_library=format_library_for_prompt(library),
            )
            with ThreadPoolExecutor(max_workers=len(cfgs_to_run)) as ex:
                futs = {ex.submit(_run_one_reader, cfg, prompt, ticker): cfg["name"]
                        for cfg in cfgs_to_run}
                for fut in as_completed(futs):
                    name, parsed = fut.result()
                    existing[name] = parsed
        sonnet = existing["sonnet"]
        gpt55 = existing["gpt55"]
        gemini3 = existing["gemini3"]

        # ── Vote count gate: require >= 2 readers (out of 3) ──
        readers_received = sum(1 for x in (sonnet, gpt55, gemini3) if x)
        if readers_received < 2:
            err = f"insufficient_reader_votes: {readers_received}/3 responded"
            logger.warning(f"[consensus] {ticker} {cat_str}: {err}")
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE post_catalyst_outcomes
                    SET outcome_label_consensus_attempts = COALESCE(outcome_label_consensus_attempts, 0) + 1,
                        outcome_label_consensus_last_attempt_at = NOW(),
                        outcome_label_consensus_last_error = %s,
                        outcome_label_sonnet_json   = COALESCE(%s::jsonb, outcome_label_sonnet_json),
                        outcome_label_gpt55_json    = COALESCE(%s::jsonb, outcome_label_gpt55_json),
                        outcome_label_gemini3_json  = COALESCE(%s::jsonb, outcome_label_gemini3_json)
                    WHERE id = %s
                """, (err,
                      json.dumps(sonnet) if sonnet else None,
                      json.dumps(gpt55) if gpt55 else None,
                      json.dumps(gemini3) if gemini3 else None,
                      outcome_id))
                conn.commit()
            return None

        # ── Pass 3 — Opus arbiter (skip if cached or opted out) ──
        if opus_existing:
            opus = opus_existing
        elif opus_opt_out:
            opus = None
        else:
            opus = run_pass3_opus_arbiter(
                ticker=ticker, company=company,
                catalyst_type=cat_type, catalyst_date=cat_str,
                drug=drug, indication=indication,
                library=library,
                pass1_json=p1,
                sonnet_json=sonnet, gpt55_json=gpt55, gemini3_json=gemini3,
            )

        # ── Persist everything in one UPDATE ──
        _persist_consensus(
            conn, outcome_id=outcome_id,
            pass1=p1, grounding_chunks=grounding_chunks,
            sonnet=sonnet, gpt55=gpt55, gemini3=gemini3, opus=opus,
            library_size=library_size,
        )
        conn.commit()

        consensus_json = _build_consensus_json(p1, sonnet, gpt55, gemini3, opus, library_size)
        return consensus_json
