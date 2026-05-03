"""drug_programs — group catalysts by product into a sequenced program view.

User feedback (2026-05-03): "we have too much individual dates and sub
catalysts. but they need to be classified according to group its
related to. so if one sub catalyst is approved it has % approval on
project its related to but is not absolutely definitive unless all
subcatalysts related to it are achieved."

Follow-up (same day): "not all biotech is drug related — some are
operational procedures like CRISPR, some are medical devices, some
are drugs. we need to somehow classify the product and its
peculiarities as different treatment/solution have different
valuations, approvals, and NPV effects."

This module takes the flat catalyst stream and reshapes it into:
  product program → sequenced milestones (per product class) → events

Where:
  - "product program" merges all aliases of the same product
    (NTLA-2001, nexiguran ziclumeran, nex-z all collapse to one)
  - "product_class" is one of {small_molecule, biologic,
    gene_editing_invivo, gene_therapy_aav, cell_therapy, vaccine,
    medical_device, diagnostic, platform_technology, unknown}
  - "milestones" are class-specific (drugs use Phase 1/2/3 + FDA
    Decision; devices use 510(k) clearance / PMA approval; etc.)
  - "completion %" uses industry-standard cumulative POS-to-approval
    per class

It does NOT trigger any LLM calls or expensive computation — pure
SQL aggregation + Python heuristics. A future LLM-backed classifier
can refine product_class accuracy.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Product classification + per-class milestone paths
# ────────────────────────────────────────────────────────────
# Each product class has different regulatory milestones, approval
# pathways, and NPV math. The default DRUG_MILESTONES below applies
# to small molecules, biologics, gene therapies, cell therapies, and
# vaccines (all use Phase 1→FDA Decision shape).
# DEVICE_MILESTONES applies to medical devices (510(k) or PMA).
# DIAGNOSTIC_MILESTONES applies to in-vitro diagnostics (CLIA + FDA).

DRUG_MILESTONES = [
    ("Preclinical",   0,  ()),
    ("Phase 1",       10, ("Phase 1", "Phase 1 Readout", "Phase 1/2", "Phase 1/2 Readout", "Phase 0 Readout")),
    ("Phase 2",       25, ("Phase 2", "Phase 2 Readout", "Phase 2/3", "Phase 2/3 Readout")),
    ("Phase 3",       55, ("Phase 3", "Phase 3 Readout", "Phase 1/2/3 Readout", "Phase 4 Readout")),
    ("Submission",    85, ("Submission", "BLA submission", "NDA submission")),
    ("FDA Decision",  95, ("FDA Decision", "PDUFA Decision", "Regulatory Decision",
                           "AdComm", "AdCom", "Advisory Committee")),
    ("Approved",     100, ()),
]

DEVICE_MILESTONES = [
    ("Concept",         0,  ()),
    ("Bench / Animal",  10, ("Trial Initiation",)),
    ("Clinical Trial",  35, ("Phase 1", "Phase 2", "Phase 3", "Clinical Trial",
                             "Clinical Trial Readout",
                             "Phase 1 Readout", "Phase 2 Readout", "Phase 3 Readout")),
    ("Submission",      75, ("Submission", "510(k) Submission", "PMA Submission",
                             "BLA submission", "NDA submission")),
    ("FDA Clearance",   95, ("FDA Decision", "510(k) Clearance", "PMA Approval",
                             "PDUFA Decision", "AdComm", "Advisory Committee")),
    ("Cleared",        100, ()),
]

DIAGNOSTIC_MILESTONES = [
    ("Development",       0,  ()),
    ("Validation Study", 25, ("Trial Initiation", "Clinical Trial",
                              "Clinical Trial Readout",
                              "Phase 1 Readout", "Phase 2 Readout", "Phase 3 Readout")),
    ("Submission",       70, ("Submission", "BLA submission", "NDA submission")),
    ("FDA Authorization",95, ("FDA Decision", "PDUFA Decision", "Regulatory Decision",
                              "AdComm", "Advisory Committee")),
    ("Authorized",      100, ()),
]

# Map product_class → milestone path
MILESTONE_PATHS = {
    "small_molecule":     DRUG_MILESTONES,
    "biologic":           DRUG_MILESTONES,
    "gene_editing_invivo": DRUG_MILESTONES,
    "gene_therapy_aav":   DRUG_MILESTONES,
    "cell_therapy":       DRUG_MILESTONES,
    "vaccine":            DRUG_MILESTONES,
    "platform_technology": DRUG_MILESTONES,  # treat platform R&D like drug pipeline
    "medical_device":     DEVICE_MILESTONES,
    "diagnostic":         DIAGNOSTIC_MILESTONES,
    "unknown":            DRUG_MILESTONES,   # safe default
}

# Human-readable product class labels + valuation hints (surfaced in UI)
PRODUCT_CLASS_META = {
    "small_molecule":      {"label": "Small molecule drug",
                             "valuation": "peak sales × duration × discount; recurring revenue"},
    "biologic":            {"label": "Biologic (mAb / protein)",
                             "valuation": "peak sales × duration × discount; recurring revenue"},
    "gene_editing_invivo": {"label": "In-vivo gene editing (CRISPR/base editing)",
                             "valuation": "one-shot revenue at high $/patient; durable cure premium"},
    "gene_therapy_aav":    {"label": "Gene therapy (AAV / viral vector)",
                             "valuation": "one-shot revenue at high $/patient; durable cure premium"},
    "cell_therapy":        {"label": "Cell therapy (CAR-T / NK)",
                             "valuation": "one-shot or short-course; high COGS, manufacturing-bound"},
    "vaccine":             {"label": "Vaccine",
                             "valuation": "volume × ASP; pandemic vs endemic dynamics"},
    "platform_technology": {"label": "Platform technology",
                             "valuation": "pipeline option value + partnership economics"},
    "medical_device":      {"label": "Medical device",
                             "valuation": "unit volume × ASP × replacement cycle; capital + service"},
    "diagnostic":          {"label": "In-vitro diagnostic",
                             "valuation": "test volume × reimbursement; lab partnerships"},
    "unknown":             {"label": "Unclassified",
                             "valuation": "—"},
}


def classify_product(
    drug_aliases: list[str], indication: str, catalyst_types: list[str],
) -> str:
    """Heuristic classifier — no LLM call. Pattern-matches against
    drug names + indications + catalyst types.

    Hierarchy (first match wins):
      1. Explicit modality keywords in the name
      2. Catalyst-type signals (510(k), PMA → device)
      3. Indication signals (vaccine, diagnostic)
      4. Default → small_molecule

    A future iteration can swap this for an LLM classifier (~$0.0003/program)
    if heuristics prove insufficient.
    """
    blob = " ".join((drug_aliases or [])).lower() + " " + (indication or "").lower()
    types_blob = " ".join((catalyst_types or [])).lower()

    # Known commercial-product overrides — heuristic blob misses brand-only
    # names that don't carry their modality in the string. These are the
    # widely-traded biotech brands; keep this list tight and only extend
    # when a misclassification shows up in practice.
    if any(k in blob for k in (
        "casgevy", "exagamglogene", "exa-cel",   # CRISPR-edited HSC cell therapy
        "lyfgenia", "bb305",                     # bluebird BCL11A lentiviral cell therapy
        "skysona", "elivaldogene",               # bluebird ALD lentiviral cell therapy
        "kymriah", "yescarta", "tecartus", "breyanzi", "abecma", "carvykti",  # CAR-T
    )):
        return "cell_therapy"
    if any(k in blob for k in (
        "comirnaty", "spikevax", "mrna-1273", "bnt162",   # COVID mRNA vaccines
        "shingrix", "fluzone", "fluarix", "gardasil",     # other commercial vaccines
        "mresvia", "mnexspike", "arexvy",
    )):
        return "vaccine"
    if any(k in blob for k in (
        "luxturna", "zolgensma", "elevidys", "hemgenix", "roctavian", "beqvez",
    )):
        return "gene_therapy_aav"

    # Modality keywords in name / indication
    if any(k in blob for k in ("crispr", "cas9", "base edit", "prime edit", "zinc finger", "talen")):
        # CRISPR-edited cell therapies (ex-vivo) are cell_therapy; in-vivo
        # editing (LNP-delivered) is gene_editing_invivo. Heuristic: ex-vivo
        # CRISPR usually pairs with sickle/beta-thal/transplant context.
        if any(k in blob for k in ("sickle", "beta-thal", "thalassemi",
                                    "ex-vivo", "ex vivo", "autologous")):
            return "cell_therapy"
        return "gene_editing_invivo"
    if any(k in blob for k in ("aav", "lentiviral", "adeno-associated", "viral vector", "gene therapy")):
        return "gene_therapy_aav"
    if any(k in blob for k in ("car-t", "car t", "tcr-t", "til-t", "cell therapy", "nk cell", "natural killer")):
        return "cell_therapy"
    if any(k in blob for k in ("vaccine", "mrna vaccine", "subunit vaccine", "viral vector vaccine")):
        return "vaccine"
    if any(k in blob for k in ("antibody", "monoclonal", "-mab ", "mab,", "mab.", "biologic")):
        return "biologic"
    if any(k in blob for k in ("ziclumeran", "ranibizumab")):  # explicit known biologic suffixes
        return "biologic"
    if any(k in blob for k in ("medical device", "implant", "stent", "catheter", "pump", "monitor",
                                "pacemaker", "defibrillator", "wearable")):
        return "medical_device"
    if any(k in blob for k in ("diagnostic", "biomarker test", "companion diagnostic", "in-vitro")):
        return "diagnostic"

    # Catalyst-type signals
    if any(k in types_blob for k in ("510(k)", "pma submission", "pma approval", "ide ")):
        return "medical_device"

    # Platform indicator (catalyst types span trial initiations across many indications)
    if "platform" in blob or "platform" in (indication or "").lower():
        return "platform_technology"

    # Default: small molecule (most common biotech product type)
    return "small_molecule"


def _milestones_for_class(product_class: str):
    return MILESTONE_PATHS.get(product_class, DRUG_MILESTONES)


# Legacy aliases (backwards-compat with v1 of this module)
MILESTONE_ORDER = DRUG_MILESTONES
MILESTONE_NAMES = [m[0] for m in DRUG_MILESTONES]
MILESTONE_PCT = {m[0]: m[1] for m in DRUG_MILESTONES}
TYPE_TO_MILESTONE: dict[str, str] = {}
for m_name, _, types in DRUG_MILESTONES:
    for t in types:
        TYPE_TO_MILESTONE[t] = m_name


def _milestone_for_with_class(catalyst_type: Optional[str], product_class: str) -> Optional[str]:
    """Resolve a catalyst_type to its milestone name in the given class's
    milestone path."""
    if not catalyst_type:
        return None
    for m_name, _, types in _milestones_for_class(product_class):
        if catalyst_type in types:
            return m_name
    # Fall back to drug-default lookup
    return TYPE_TO_MILESTONE.get(catalyst_type)


def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    # Drop parens content (often the alternate name) but keep both for matching
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_drug_name(name: str) -> set[str]:
    """Return a set of normalised tokens for substring-fuzzy matching.

    'nexiguran ziclumeran (nex-z, NTLA-2001)' →
       {'nexiguran', 'ziclumeran', 'nex-z', 'ntla-2001'}
    """
    if not name:
        return set()
    n = name.lower()
    n = re.sub(r"[(),]", " ", n)
    tokens = re.split(r"\s+", n)
    out = set()
    for t in tokens:
        t = t.strip()
        if not t or t in ("the", "a", "an", "and", "of"):
            continue
        # Skip pure numbers and very short tokens
        if len(t) < 3:
            continue
        out.add(t)
    return out


def _cluster_drug_names(drug_names: list[str]) -> list[list[str]]:
    """Cluster drug names that refer to the same molecule.

    Two names cluster if they share at least one **specific** token —
    where "specific" means the token appears in ≤ 50% of names for
    this ticker. This filters out class-level suffixes like
    "ziclumeran" (monoclonal-antibody-class suffix shared by nex-z
    AND lonvo-z) which would otherwise wrongly merge two distinct
    drug programs.

    For NTLA: filters "ziclumeran" (50%+ frequency) → nexiguran-program
    and lonvoguran-program stay disjoint. Within each, the
    NTLA-XXXX code, generic name, and brand abbreviation cluster
    correctly via the remaining specific tokens.
    """
    if not drug_names:
        return []
    # Token frequency across all distinct names
    token_counts: dict[str, int] = {}
    name_tokens: list[set[str]] = []
    for name in drug_names:
        toks = _tokenize_drug_name(name)
        name_tokens.append(toks)
        for t in toks:
            token_counts[t] = token_counts.get(t, 0) + 1
    n = len(drug_names)
    # A token is "generic" if it appears in more than half of distinct names
    generic_threshold = max(2, n * 0.5)
    generic_tokens = {t for t, c in token_counts.items() if c > generic_threshold}

    clusters: list[list[str]] = []
    cluster_tokens: list[set[str]] = []
    for i, name in enumerate(drug_names):
        specific = name_tokens[i] - generic_tokens
        if not specific:
            # Name has no specific tokens — keep alone
            clusters.append([name])
            cluster_tokens.append(name_tokens[i])
            continue
        merged_into = None
        for j, ct in enumerate(cluster_tokens):
            ct_specific = ct - generic_tokens
            if specific & ct_specific:
                clusters[j].append(name)
                cluster_tokens[j] |= name_tokens[i]
                merged_into = j
                break
        if merged_into is None:
            clusters.append([name])
            cluster_tokens.append(name_tokens[i])
    return clusters


def _milestone_for(catalyst_type: Optional[str]) -> Optional[str]:
    if not catalyst_type:
        return None
    return TYPE_TO_MILESTONE.get(catalyst_type)


def _status_from_outcome(
    outcome_label_class: Optional[str],
    actual_move_pct_calibration: Optional[float],
    is_future: bool,
) -> str:
    """Status for a single event: pending | success | mixed | failure | delayed | unknown.

    `actual_move_pct_calibration` should be the 7d move (or whatever calibration
    window the caller chooses) — 7d is the practitioner consensus for biotech
    catalysts because 1d misses analyst revisions + hedge unwind, while 30d
    is contaminated by exogenous news.
    """
    if is_future:
        return "pending"
    if not outcome_label_class or outcome_label_class == "UNKNOWN":
        # Fall back on price action if available
        if actual_move_pct_calibration is not None:
            if actual_move_pct_calibration > 15:
                return "success"
            if actual_move_pct_calibration < -15:
                return "failure"
            return "mixed"
        return "unknown"
    label = outcome_label_class.upper()
    if label in ("APPROVED", "MET_ENDPOINT"):
        return "success"
    if label in ("REJECTED", "MISSED_ENDPOINT", "WITHDRAWN"):
        return "failure"
    if label == "DELAYED":
        return "delayed"
    if label == "MIXED":
        return "mixed"
    return "unknown"


def _select_canonical_label(names: list[str]) -> str:
    """Pick the most informative name for the program label.

    Prefer the longest with parens (usually 'generic name (brand, code)').
    Fall back to the longest name."""
    if not names:
        return ""
    with_parens = [n for n in names if "(" in n]
    if with_parens:
        return max(with_parens, key=len)
    return max(names, key=len)


def _summarize_indication(indications: list[str]) -> str:
    """Pick the most common/longest indication string for the program."""
    if not indications:
        return ""
    # Normalise to lowercase, count
    counts: dict[str, int] = {}
    for ind in indications:
        if not ind:
            continue
        key = ind.lower().strip()
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    # Return the original-case version of the most common
    most_common_key = max(counts, key=lambda k: (counts[k], -len(k)))
    for ind in indications:
        if ind and ind.lower().strip() == most_common_key:
            return ind
    return most_common_key


def get_drug_programs_for_ticker(ticker: str) -> dict[str, Any]:
    """Return drug-program-grouped catalyst data for the ticker.

    Joins catalyst_universe with post_catalyst_outcomes to attach
    historical price action + v2 prediction columns where available.
    """
    if not ticker:
        return {"ticker": ticker, "drug_programs": []}
    ticker = ticker.upper().strip()
    db = BiotechDatabase()

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
              cu.id AS catalyst_id, cu.catalyst_type, cu.catalyst_date,
              cu.drug_name, cu.canonical_drug_name, cu.indication, cu.phase,
              cu.confidence_score, cu.source, cu.source_url, cu.description,
              -- joined outcome row (may be null for upcoming events)
              pco.id AS outcome_id, pco.pre_event_price, pco.day1_price,
              pco.day7_price, pco.day30_price, pco.actual_move_pct_1d,
              pco.actual_move_pct_7d, pco.actual_move_pct_30d,
              pco.predicted_move_pct, pco.predicted_prob,
              pco.outcome_label_class, pco.outcome_label_confidence,
              pco.options_implied_move_pct,
              -- v2 columns (may be null until shadow mode populates them)
              pco.predicted_move_npv_pct, pco.predicted_move_statistical_pct,
              pco.predicted_low_pct, pco.predicted_high_pct,
              pco.regime, pco.cap_bucket_at_prediction, pco.priced_in_fraction,
              pco.priced_in_method, pco.disagreement_pp, pco.disagreement_verdict,
              pco.prediction_source_v2, pco.abstained, pco.abstain_reason
            FROM catalyst_universe cu
            LEFT JOIN post_catalyst_outcomes pco
              ON pco.ticker = cu.ticker
             AND pco.catalyst_type = cu.catalyst_type
             AND pco.catalyst_date::text = cu.catalyst_date::text
            WHERE cu.ticker = %s
              AND cu.status = 'active'
            ORDER BY cu.catalyst_date DESC
        """, (ticker,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    if not rows:
        return {"ticker": ticker, "drug_programs": []}

    # Step 1: collect distinct drug names per ticker
    name_to_rows: dict[str, list[dict]] = {}
    for r in rows:
        name = (r.get("drug_name") or r.get("canonical_drug_name") or "Unknown")
        name_to_rows.setdefault(name, []).append(r)

    # Step 2: cluster names that share tokens (NTLA-2001 / nex-z / nexiguran ziclumeran → one)
    distinct_names = list(name_to_rows.keys())
    clusters = _cluster_drug_names(distinct_names)

    # Step 3: build a program per cluster
    from datetime import date as _date
    today = _date.today()
    programs: list[dict[str, Any]] = []
    for cluster_names in clusters:
        events: list[dict] = []
        for name in cluster_names:
            events.extend(name_to_rows[name])
        if not events:
            continue

        # Classify product class up-front so milestones use the right path
        cluster_indications = [e.get("indication") for e in events if e.get("indication")]
        cluster_indication_blob = " ".join(cluster_indications) if cluster_indications else ""
        cluster_catalyst_types = list({e.get("catalyst_type") for e in events
                                       if e.get("catalyst_type")})
        product_class = classify_product(
            cluster_names, cluster_indication_blob, cluster_catalyst_types,
        )
        product_meta = PRODUCT_CLASS_META.get(product_class, PRODUCT_CLASS_META["unknown"])

        # Dedup events by (catalyst_type, catalyst_date) — keep highest confidence
        dedup: dict[tuple, dict] = {}
        for e in events:
            key = (e.get("catalyst_type") or "", str(e.get("catalyst_date") or ""))
            existing = dedup.get(key)
            if not existing:
                dedup[key] = e
            else:
                # Keep the one with the higher confidence_score (or with an outcome row)
                conf_e = float(e.get("confidence_score") or 0)
                conf_x = float(existing.get("confidence_score") or 0)
                has_outcome_e = e.get("outcome_id") is not None
                has_outcome_x = existing.get("outcome_id") is not None
                if (has_outcome_e and not has_outcome_x) or (
                    has_outcome_e == has_outcome_x and conf_e > conf_x
                ):
                    dedup[key] = e
        deduped = sorted(dedup.values(),
                         key=lambda r: str(r.get("catalyst_date") or ""))

        # Build event list with status tagging
        clean_events = []
        for e in deduped:
            cd = e.get("catalyst_date")
            try:
                cd_d = _date.fromisoformat(str(cd)[:10])
                is_future = cd_d > today
            except Exception:
                is_future = False
            milestone = _milestone_for_with_class(e.get("catalyst_type"), product_class)
            # Prefer 7d for the calibration-window fallback; 30d as a backstop
            # if 7d hasn't been backfilled yet.
            calib_move = e.get("actual_move_pct_7d") if e.get("actual_move_pct_7d") is not None else e.get("actual_move_pct_30d")
            status = _status_from_outcome(
                e.get("outcome_label_class"),
                float(calib_move) if calib_move is not None else None,
                is_future,
            )
            clean_events.append({
                "catalyst_id": e.get("catalyst_id"),
                "catalyst_type": e.get("catalyst_type"),
                "catalyst_date": str(cd) if cd else None,
                "milestone": milestone,
                "drug_name_raw": e.get("drug_name"),
                "indication": e.get("indication"),
                "phase": e.get("phase"),
                "confidence_score": float(e["confidence_score"]) if e.get("confidence_score") is not None else None,
                "source": e.get("source"),
                "source_url": e.get("source_url"),
                "is_future": is_future,
                "status": status,
                # Pricing (only for past events with outcome row)
                "pre_event_price": float(e["pre_event_price"]) if e.get("pre_event_price") is not None else None,
                "day1_price": float(e["day1_price"]) if e.get("day1_price") is not None else None,
                "day30_price": float(e["day30_price"]) if e.get("day30_price") is not None else None,
                "actual_move_pct_1d": float(e["actual_move_pct_1d"]) if e.get("actual_move_pct_1d") is not None else None,
                # 7d move: better calibration than 1d for biotech catalysts —
                # captures analyst revisions + hedge unwind + approval-halo fade.
                "actual_move_pct_7d": float(e["actual_move_pct_7d"]) if e.get("actual_move_pct_7d") is not None else None,
                "actual_move_pct_30d": float(e["actual_move_pct_30d"]) if e.get("actual_move_pct_30d") is not None else None,
                "day7_price": float(e["day7_price"]) if e.get("day7_price") is not None else None,
                "options_implied_move_pct": float(e["options_implied_move_pct"]) if e.get("options_implied_move_pct") is not None else None,
                # Predictions (legacy + v2)
                "predicted_move_pct": float(e["predicted_move_pct"]) if e.get("predicted_move_pct") is not None else None,
                "predicted_prob": float(e["predicted_prob"]) if e.get("predicted_prob") is not None else None,
                "predicted_move_npv_pct": float(e["predicted_move_npv_pct"]) if e.get("predicted_move_npv_pct") is not None else None,
                "predicted_move_statistical_pct": float(e["predicted_move_statistical_pct"]) if e.get("predicted_move_statistical_pct") is not None else None,
                "predicted_low_pct": float(e["predicted_low_pct"]) if e.get("predicted_low_pct") is not None else None,
                "predicted_high_pct": float(e["predicted_high_pct"]) if e.get("predicted_high_pct") is not None else None,
                "regime": e.get("regime"),
                "prediction_source_v2": e.get("prediction_source_v2"),
                "abstained": bool(e["abstained"]) if e.get("abstained") is not None else None,
                "abstain_reason": e.get("abstain_reason"),
                "disagreement_pp": float(e["disagreement_pp"]) if e.get("disagreement_pp") is not None else None,
                "disagreement_verdict": e.get("disagreement_verdict"),
                # Outcome
                "outcome_label_class": e.get("outcome_label_class"),
                "outcome_label_confidence": float(e["outcome_label_confidence"]) if e.get("outcome_label_confidence") is not None else None,
            })

        # Build milestone summary: for each stage, what's the latest event +
        # status. If any event in a stage is "success" → stage marked completed.
        # If any "failure" → stage marked failed. If pending future → "pending".
        # Use the CLASS-SPECIFIC milestone path so devices/diagnostics get
        # their right stage labels (510(k), Validation, etc.).
        class_milestones = _milestones_for_class(product_class)
        class_milestone_names = [m[0] for m in class_milestones]
        class_milestone_pct = {m[0]: m[1] for m in class_milestones}
        # Synthetic milestones (no events ever land in them)
        synthetic_first = class_milestone_names[0]    # e.g. Preclinical / Concept / Development
        synthetic_last = class_milestone_names[-1]    # e.g. Approved / Cleared / Authorized
        milestone_summary: list[dict] = []
        for m_name in class_milestone_names:
            if m_name in (synthetic_first, synthetic_last):
                continue
            in_stage = [e for e in clean_events if e["milestone"] == m_name]
            if not in_stage:
                milestone_summary.append({
                    "stage": m_name, "status": "not_reached",
                    "n_events": 0, "latest_date": None, "latest_label": None,
                })
                continue
            # Status priority: failure > success > delayed > mixed > pending > unknown
            statuses = [e["status"] for e in in_stage]
            if any(s == "success" for s in statuses):
                stage_status = "completed"
            elif any(s == "failure" for s in statuses):
                stage_status = "failed"
            elif any(s == "delayed" for s in statuses):
                stage_status = "delayed"
            elif any(s == "mixed" for s in statuses):
                stage_status = "mixed"
            elif any(s == "pending" for s in statuses):
                stage_status = "pending"
            else:
                stage_status = "unknown"
            in_stage_sorted = sorted(in_stage, key=lambda e: e["catalyst_date"] or "")
            latest = in_stage_sorted[-1]
            milestone_summary.append({
                "stage": m_name,
                "status": stage_status,
                "n_events": len(in_stage),
                "latest_date": latest["catalyst_date"],
                "latest_label": latest["outcome_label_class"],
            })

        # Compute completion % — the highest-completed (or in-progress) stage
        completion_pct = 0
        highest_stage = synthetic_first
        for ms in milestone_summary:
            stage = ms["stage"]
            stage_pct = class_milestone_pct.get(stage, 0)
            if ms["status"] == "completed":
                if stage_pct > completion_pct:
                    completion_pct = stage_pct
                    highest_stage = stage
            elif ms["status"] in ("delayed", "pending", "mixed"):
                # Partial credit at that stage — half of the gap to its pct
                partial = stage_pct - 5  # one notch back from full
                if partial > completion_pct:
                    completion_pct = partial
                    highest_stage = stage + " (in progress)"
            elif ms["status"] == "failed":
                # Failed at this stage — keep prior completion but mark stage
                pass

        # Indication summary from non-null indications across cluster events
        indications = [e["indication"] for e in clean_events if e["indication"]]
        indication_summary = _summarize_indication(indications)

        program = {
            "program_id": f"{ticker.lower()}-{cluster_names[0].lower().replace(' ', '-')[:30]}",
            "drug_aliases": cluster_names,
            "canonical_label": _select_canonical_label(cluster_names),
            "indication_summary": indication_summary,
            "product_class": product_class,
            "product_class_label": product_meta["label"],
            "valuation_framework": product_meta["valuation"],
            "milestone_path_name": "device" if product_class == "medical_device"
                                   else "diagnostic" if product_class == "diagnostic"
                                   else "drug",
            "completion_pct": completion_pct,
            "highest_stage": highest_stage,
            "milestones": milestone_summary,
            "events": clean_events,
            "n_events": len(clean_events),
            "n_future_events": sum(1 for e in clean_events if e["is_future"]),
            "n_completed_events": sum(1 for e in clean_events if e["status"] == "success"),
            "n_failed_events": sum(1 for e in clean_events if e["status"] == "failure"),
        }
        programs.append(program)

    # Sort programs: most complete first, then most events
    programs.sort(
        key=lambda p: (-p["completion_pct"], -p["n_events"]),
    )
    return {
        "ticker": ticker,
        "drug_programs": programs,
        "milestone_definitions_by_class": {
            cls_key: [
                {"stage": s, "completion_pct": p, "catalyst_types": list(types)}
                for s, p, types in path
            ]
            for cls_key, path in MILESTONE_PATHS.items()
        },
        "product_class_meta": PRODUCT_CLASS_META,
        # Backwards-compat: include the drug-default order at the
        # top level so v1 callers still work.
        "milestone_definitions": [
            {"stage": s, "completion_pct": p, "catalyst_types": list(types)}
            for s, p, types in DRUG_MILESTONES
        ],
    }
