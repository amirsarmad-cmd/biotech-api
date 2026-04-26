"""fda_sources — official structured data sources used to anchor LLM economics.

Layer 1 of the source stack: official, structured, free, authoritative.
These sources REPLACE LLM-guessed values where available, and are passed
to the V2 LLM as 'verified_facts' anchors when only partial coverage exists.

Sources:
    - OpenFDA Drug Label    https://api.fda.gov/drug/label.json
    - OpenFDA Drug Event    https://api.fda.gov/drug/event.json
    - OpenFDA Drugs@FDA     https://api.fda.gov/drug/drugsfda.json
    - ClinicalTrials.gov v2 https://clinicaltrials.gov/api/v2/studies
    - Orange Book (CSV)     https://www.fda.gov/media/76860/download

All fetchers are cached in Redis (24h TTL) so we don't hammer the public APIs.
None require authentication.

Rate limits:
    - OpenFDA: 240 req/min anon, 120k/day with key (OPENFDA_API_KEY env, optional)
    - ClinicalTrials.gov: no documented limit (~1 req/sec is polite)
"""
import os
import logging
import json
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus
import requests

logger = logging.getLogger(__name__)

OPENFDA_KEY = os.getenv("OPENFDA_API_KEY", "").strip()  # optional
OPENFDA_BASE = "https://api.fda.gov"
CTG_BASE = "https://clinicaltrials.gov/api/v2"


def _redis_client():
    try:
        from services.cache import get_redis
        return get_redis()
    except Exception:
        return None


def _cached_get(cache_key: str, ttl_sec: int = 86400):
    """Helper to wrap a fetch with Redis cache."""
    r = _redis_client()
    if r is None:
        return None  # cache miss, but caller should still try the fetch
    try:
        raw = r.get(cache_key)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _cached_set(cache_key: str, value: Any, ttl_sec: int = 86400):
    r = _redis_client()
    if r is None:
        return
    try:
        r.setex(cache_key, ttl_sec, json.dumps(value, default=str))
    except Exception:
        pass


def _http_get(url: str, params: Optional[Dict] = None, timeout: int = 12) -> Optional[Dict]:
    """Robust GET with logging — returns parsed JSON or None on failure."""
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": "biotech-screener/1.0"})
        if resp.status_code == 404:
            return {"_status": 404, "_message": "not found"}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.info(f"http_get failed for {url}: {e}")
        return None
    except ValueError as e:
        logger.info(f"json parse failed for {url}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# OpenFDA Drug Label — official approved indication, dosing, warnings
# ────────────────────────────────────────────────────────────

def fetch_drug_label(drug_name: str) -> Optional[Dict]:
    """Fetch FDA-approved drug label by brand or generic name.
    
    Returns a structured summary:
        {
          "drug_name": str,
          "brand_names": [str],
          "generic_names": [str],
          "approved_indications": [str],
          "boxed_warning": str | None,
          "indications_and_usage": str,
          "approval_date": str | None,
          "_source": "openfda_drug_label",
          "_url": str,
          "_n_results": int,
        }
    Or None if not found / fetch failed.
    """
    if not drug_name:
        return None
    drug_name = drug_name.strip()
    cache_key = f"openfda:label:{drug_name.lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    # Try brand name first, fall back to generic name
    queries = [
        f'openfda.brand_name:"{drug_name}"',
        f'openfda.generic_name:"{drug_name}"',
        # Also try a stripped version (e.g., 'lonvoguran ziclumeran (lonvo-z)' → 'lonvoguran')
        f'openfda.generic_name:"{drug_name.split()[0]}"' if " " in drug_name else None,
    ]
    queries = [q for q in queries if q]

    result = None
    for q in queries:
        data = _http_get(f"{OPENFDA_BASE}/drug/label.json",
                         params={"search": q, "limit": 1, "api_key": OPENFDA_KEY or None})
        if not data or data.get("_status") == 404:
            continue
        if not data.get("results"):
            continue
        result = data
        break

    if not result or not result.get("results"):
        # Cache the negative result briefly to avoid retries
        _cached_set(cache_key, {"_not_found": True, "drug_name": drug_name}, ttl_sec=3600)
        return None

    r = result["results"][0]
    of = r.get("openfda", {}) or {}
    out = {
        "drug_name": drug_name,
        "brand_names": of.get("brand_name", [])[:5],
        "generic_names": of.get("generic_name", [])[:5],
        "approved_indications": [(s or "")[:300] for s in (r.get("indications_and_usage") or [])[:3]],
        "boxed_warning": (r.get("boxed_warning") or [None])[0],
        "indications_and_usage": (r.get("indications_and_usage") or [""])[0][:500],
        "manufacturer_name": of.get("manufacturer_name", []),
        "product_type": of.get("product_type", []),
        "route": of.get("route", []),
        "_source": "openfda_drug_label",
        "_url": f"https://api.fda.gov/drug/label.json?search={quote_plus(queries[0])}",
        "_n_results": result.get("meta", {}).get("results", {}).get("total", 0),
        "_from_cache": False,
    }
    _cached_set(cache_key, out, ttl_sec=86400)
    return out


# ────────────────────────────────────────────────────────────
# OpenFDA Drugs@FDA — approval / submission history
# ────────────────────────────────────────────────────────────

def fetch_approval_history(drug_name: str) -> Optional[Dict]:
    """Fetch FDA submission history (NDA / BLA / PMA approvals, supplements,
    CRLs, withdrawals) for a drug name.
    
    Returns:
        {
          "drug_name": str,
          "applications": [
            {
              "application_number": str,
              "submissions": [
                  {"type": "NDA"/"BLA"/"PMA", "status": str, "date": str, ...}
              ],
            }
          ],
          "approval_count": int,
          "rejection_count": int,
          "earliest_approval": str | None,
          "latest_action": str | None,
          "_source": "openfda_drugsfda",
        }
    """
    if not drug_name:
        return None
    drug_name = drug_name.strip()
    cache_key = f"openfda:drugsfda:{drug_name.lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    queries = [
        f'products.brand_name:"{drug_name}"',
        f'products.active_ingredients.name:"{drug_name}"',
        f'products.active_ingredients.name:"{drug_name.split()[0]}"' if " " in drug_name else None,
    ]
    queries = [q for q in queries if q]

    data = None
    matched_q = None
    for q in queries:
        data = _http_get(f"{OPENFDA_BASE}/drug/drugsfda.json",
                         params={"search": q, "limit": 5, "api_key": OPENFDA_KEY or None})
        if data and data.get("results"):
            matched_q = q
            break

    if not data or not data.get("results"):
        _cached_set(cache_key, {"_not_found": True, "drug_name": drug_name}, ttl_sec=3600)
        return None

    apps = []
    approval_count = 0
    rejection_count = 0
    all_dates = []
    for app in data["results"]:
        subs = []
        for s in (app.get("submissions") or [])[:20]:
            sub = {
                "type": s.get("submission_type"),
                "status": s.get("submission_status"),
                "date": s.get("submission_status_date"),
                "class_code": s.get("submission_class_code_description"),
            }
            subs.append(sub)
            stat = (s.get("submission_status") or "").upper()
            if stat == "AP":
                approval_count += 1
            elif stat in ("CL", "RW", "WD"):  # CL=closed/CRL, RW=rejected, WD=withdrawn
                rejection_count += 1
            if sub["date"]:
                all_dates.append(sub["date"])
        apps.append({
            "application_number": app.get("application_number"),
            "sponsor_name": app.get("sponsor_name"),
            "submissions": subs,
            "products": [
                {"brand_name": p.get("brand_name"),
                 "active_ingredients": [a.get("name") for a in (p.get("active_ingredients") or [])],
                 "marketing_status": p.get("marketing_status")}
                for p in (app.get("products") or [])[:5]
            ],
        })

    all_dates.sort()
    out = {
        "drug_name": drug_name,
        "matched_query": matched_q,
        "applications": apps,
        "approval_count": approval_count,
        "rejection_count": rejection_count,
        "earliest_approval": all_dates[0] if all_dates else None,
        "latest_action": all_dates[-1] if all_dates else None,
        "_source": "openfda_drugsfda",
        "_url": f"https://api.fda.gov/drug/drugsfda.json?search={quote_plus(matched_q or '')}",
        "_from_cache": False,
    }
    _cached_set(cache_key, out, ttl_sec=86400)
    return out


# ────────────────────────────────────────────────────────────
# OpenFDA Drug Event — adverse event counts (safety profile)
# ────────────────────────────────────────────────────────────

def fetch_adverse_event_summary(drug_name: str) -> Optional[Dict]:
    """Pull AERS adverse-event counts for a drug. Used to assess safety
    profile vs. competitors.
    
    Returns:
        {
          "drug_name": str,
          "total_reports": int,
          "serious_count": int,
          "death_count": int,
          "top_reactions": [{"reaction": str, "n": int}],
        }
    """
    if not drug_name:
        return None
    drug_name = drug_name.strip()
    cache_key = f"openfda:event:{drug_name.lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    # Top adverse reactions for this drug
    data = _http_get(f"{OPENFDA_BASE}/drug/event.json",
                     params={
                         "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                         "count": "patient.reaction.reactionmeddrapt.exact",
                         "limit": 10,
                         "api_key": OPENFDA_KEY or None,
                     })
    if not data or not data.get("results"):
        _cached_set(cache_key, {"_not_found": True, "drug_name": drug_name}, ttl_sec=3600)
        return None

    top_reactions = [{"reaction": r.get("term"), "n": r.get("count")}
                     for r in data["results"]][:10]

    # Total report count + seriousness breakdown
    seriousness = _http_get(f"{OPENFDA_BASE}/drug/event.json",
                            params={
                                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                                "count": "serious",
                                "limit": 5,
                                "api_key": OPENFDA_KEY or None,
                            })
    serious_count = 0
    total = 0
    if seriousness and seriousness.get("results"):
        for r in seriousness["results"]:
            total += r.get("count", 0)
            if r.get("term") == 1:
                serious_count = r.get("count", 0)

    death_data = _http_get(f"{OPENFDA_BASE}/drug/event.json",
                           params={
                               "search": f'patient.drug.medicinalproduct:"{drug_name}" AND seriousnessdeath:1',
                               "limit": 1,
                               "api_key": OPENFDA_KEY or None,
                           })
    death_count = 0
    if death_data and death_data.get("meta", {}).get("results", {}).get("total"):
        death_count = death_data["meta"]["results"]["total"]

    out = {
        "drug_name": drug_name,
        "total_reports": total,
        "serious_count": serious_count,
        "death_count": death_count,
        "top_reactions": top_reactions,
        "_source": "openfda_drug_event",
        "_from_cache": False,
    }
    _cached_set(cache_key, out, ttl_sec=86400)
    return out


# ────────────────────────────────────────────────────────────
# ClinicalTrials.gov v2 — official trial registry
# ────────────────────────────────────────────────────────────

def fetch_clinical_trials(drug_name: str = None, indication: str = None,
                           phase: str = None, status: str = None,
                           max_studies: int = 5) -> Optional[Dict]:
    """Search ClinicalTrials.gov v2 for studies matching drug / indication / phase.
    
    Returns:
        {
          "studies": [
            {
              "nct_id": str,
              "brief_title": str,
              "phase": str,
              "status": str,
              "enrollment": int,
              "primary_completion_date": str,
              "primary_outcome_measure": str,
              "interventions": [str],
              "conditions": [str],
              "sponsors": [str],
              "_url": str,
            }, ...
          ],
          "total_count": int,
          "_source": "clinicaltrials_gov",
        }
    """
    if not (drug_name or indication):
        return None
    
    cache_key = f"ctg:{(drug_name or '').lower()}:{(indication or '').lower()}:{phase or ''}:{status or ''}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    params = {
        "format": "json",
        "pageSize": min(max_studies, 20),
        "fields": ",".join([
            "NCTId", "BriefTitle", "Phase", "OverallStatus",
            "EnrollmentCount", "PrimaryCompletionDate",
            "PrimaryOutcomeMeasure", "InterventionName",
            "Condition", "LeadSponsorName",
        ]),
    }
    
    # query.intr = intervention, query.cond = condition
    query_parts = []
    if drug_name:
        params["query.intr"] = drug_name
    if indication:
        params["query.cond"] = indication
    if phase:
        params["filter.advanced"] = f"AREA[Phase]{phase}"
    if status:
        params["filter.overallStatus"] = status

    data = _http_get(f"{CTG_BASE}/studies", params=params, timeout=15)
    if not data or not data.get("studies"):
        _cached_set(cache_key, {"_not_found": True, "drug_name": drug_name, "indication": indication}, ttl_sec=3600)
        return None

    studies = []
    for s in data.get("studies", []):
        ps = s.get("protocolSection", {}) or {}
        ident = ps.get("identificationModule", {}) or {}
        status_mod = ps.get("statusModule", {}) or {}
        design = ps.get("designModule", {}) or {}
        intervention = ps.get("armsInterventionsModule", {}) or {}
        conditions = ps.get("conditionsModule", {}) or {}
        outcomes = ps.get("outcomesModule", {}) or {}
        sponsors = ps.get("sponsorCollaboratorsModule", {}) or {}

        nct = ident.get("nctId")
        studies.append({
            "nct_id": nct,
            "brief_title": ident.get("briefTitle", "")[:200],
            "phase": (design.get("phases") or [None])[0],
            "status": status_mod.get("overallStatus"),
            "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
            "enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
            "primary_completion_date": (status_mod.get("primaryCompletionDateStruct") or {}).get("date"),
            "study_completion_date": (status_mod.get("completionDateStruct") or {}).get("date"),
            "primary_outcome_measure": ((outcomes.get("primaryOutcomes") or [{}])[0] or {}).get("measure", "")[:200],
            "interventions": [
                (i or {}).get("name", "")[:80]
                for i in (intervention.get("interventions") or [])[:5]
            ],
            "conditions": (conditions.get("conditions") or [])[:5],
            "sponsors": [(sponsors.get("leadSponsor") or {}).get("name")] +
                        [c.get("name") for c in (sponsors.get("collaborators") or [])[:3]
                         if c.get("name")],
            "_url": f"https://clinicaltrials.gov/study/{nct}" if nct else None,
        })

    out = {
        "studies": studies,
        "total_count": data.get("totalCount", len(studies)),
        "drug_name": drug_name,
        "indication": indication,
        "phase": phase,
        "status": status,
        "_source": "clinicaltrials_gov",
        "_from_cache": False,
    }
    _cached_set(cache_key, out, ttl_sec=86400)
    return out


# ────────────────────────────────────────────────────────────
# Orange Book — patent + exclusivity (LOE dates)
# ────────────────────────────────────────────────────────────

# Orange Book is published as a CSV bundle. We fetch + parse it weekly.
# Endpoint: https://www.fda.gov/media/76860/download (the file structure
# changes occasionally; the canonical reference is
# https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files)
ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download?attachment"


def fetch_orange_book_loe(drug_name: str = None, ingredient: str = None) -> Optional[Dict]:
    """Look up patent + exclusivity expirations for a drug from Orange Book.
    
    Note: this is a heavy operation — the OB CSV is ~50MB. We cache the
    parsed lookup table for 7 days. First call after deploy will be slow.
    
    Returns:
        {
          "drug_name": str,
          "ingredient": str,
          "patents": [
            {"patent_no": str, "expire_date": str, "drug_substance_flag": bool, "drug_product_flag": bool},
            ...
          ],
          "exclusivities": [
            {"code": str, "date": str, "description": str},  # e.g. ODE = orphan
            ...
          ],
          "earliest_loe": str | None,
          "latest_loe": str | None,
        }
    """
    if not (drug_name or ingredient):
        return None
    
    cache_key = f"orangebook:lookup:{(drug_name or ingredient or '').lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    # For now, a stub that returns the cache key + indicates not implemented.
    # Full implementation requires download + parse of products.txt, patent.txt,
    # and exclusivity.txt. Will add in a follow-up commit once we verify
    # OpenFDA + ClinicalTrials.gov work end-to-end.
    out = {
        "drug_name": drug_name,
        "ingredient": ingredient,
        "_source": "orange_book",
        "_status": "not_yet_implemented",
        "_note": "Orange Book CSV ingest deferred; use OpenFDA Drugs@FDA approval_date as proxy",
    }
    _cached_set(cache_key, out, ttl_sec=300)
    return out


# ────────────────────────────────────────────────────────────
# Combined fetch — used by /analyze/npv
# ────────────────────────────────────────────────────────────

def gather_verified_facts(ticker: str, drug_name: str,
                          indication: str = None) -> Dict:
    """Pull all available official facts for a drug. Used to inject into the
    V2 LLM prompt as 'verified_facts' so the LLM can anchor its estimates.
    
    Returns dict with whatever was successfully fetched. Each section may
    be present or absent. Always includes a 'sources_attempted' list and
    'sources_succeeded' list for transparency.
    """
    t0 = time.time()
    attempted = []
    succeeded = []
    facts = {
        "ticker": ticker,
        "drug_name": drug_name,
        "indication": indication,
    }

    # OpenFDA drug label
    attempted.append("openfda_drug_label")
    label = fetch_drug_label(drug_name)
    if label and not label.get("_not_found"):
        facts["drug_label"] = label
        succeeded.append("openfda_drug_label")

    # OpenFDA Drugs@FDA approval history
    attempted.append("openfda_drugsfda")
    history = fetch_approval_history(drug_name)
    if history and not history.get("_not_found"):
        facts["approval_history"] = history
        succeeded.append("openfda_drugsfda")

    # ClinicalTrials.gov — find pivotal trials
    attempted.append("clinicaltrials_gov")
    trials = fetch_clinical_trials(drug_name=drug_name, indication=indication, max_studies=5)
    if trials and not trials.get("_not_found") and trials.get("studies"):
        facts["clinical_trials"] = trials
        succeeded.append("clinicaltrials_gov")

    # Orange Book — patents (stub for now)
    # attempted.append("orange_book")
    # orange = fetch_orange_book_loe(drug_name=drug_name)
    # if orange:
    #     facts["orange_book"] = orange
    #     succeeded.append("orange_book")

    facts["_sources_attempted"] = attempted
    facts["_sources_succeeded"] = succeeded
    facts["_fetch_duration_ms"] = int((time.time() - t0) * 1000)
    return facts


def format_verified_facts_for_prompt(facts: Dict) -> str:
    """Render verified_facts dict as a structured prompt block for the LLM.
    Returns empty string if no facts were successfully fetched.
    """
    if not facts.get("_sources_succeeded"):
        return ""
    
    lines = ["VERIFIED FACTS (use these to anchor your estimates — do NOT contradict):"]
    
    # Drug label section
    if facts.get("drug_label"):
        lab = facts["drug_label"]
        lines.append(f"\n[OpenFDA Drug Label]")
        if lab.get("brand_names"):
            lines.append(f"  Brand names: {', '.join(lab['brand_names'][:3])}")
        if lab.get("generic_names"):
            lines.append(f"  Generic names: {', '.join(lab['generic_names'][:3])}")
        if lab.get("indications_and_usage"):
            ind_text = lab['indications_and_usage'][:400].replace('\n', ' ')
            lines.append(f"  Approved indications: {ind_text}")
        if lab.get("boxed_warning"):
            warn = lab['boxed_warning'][:200].replace('\n', ' ')
            lines.append(f"  Boxed warning: {warn}")
        if lab.get("manufacturer_name"):
            lines.append(f"  Manufacturer: {lab['manufacturer_name'][0] if lab['manufacturer_name'] else '?'}")

    # Approval history section
    if facts.get("approval_history"):
        h = facts["approval_history"]
        lines.append(f"\n[OpenFDA Drugs@FDA — approval history]")
        lines.append(f"  Approval count: {h.get('approval_count', 0)}")
        lines.append(f"  Rejection/CRL count: {h.get('rejection_count', 0)}")
        if h.get("earliest_approval"):
            lines.append(f"  Earliest approval: {h['earliest_approval']}")
        if h.get("latest_action"):
            lines.append(f"  Latest action: {h['latest_action']}")
        # Show 3 most recent submissions
        all_subs = []
        for app in (h.get("applications") or []):
            for s in (app.get("submissions") or []):
                if s.get("date"):
                    all_subs.append(s)
        all_subs.sort(key=lambda s: s.get("date") or "", reverse=True)
        for s in all_subs[:3]:
            lines.append(f"  - {s.get('date')}: {s.get('type')} → {s.get('status')} ({s.get('class_code', '')})")

    # Clinical trials section
    if facts.get("clinical_trials"):
        ct = facts["clinical_trials"]
        lines.append(f"\n[ClinicalTrials.gov — relevant studies]")
        lines.append(f"  Total registered studies matching: {ct.get('total_count', 0)}")
        for s in (ct.get("studies") or [])[:4]:
            lines.append(f"  - {s.get('nct_id')} | {s.get('phase', '?')} | {s.get('status', '?')} | "
                         f"n={s.get('enrollment') or '?'} | {(s.get('brief_title') or '')[:80]}")
            if s.get("primary_outcome_measure"):
                pom = (s['primary_outcome_measure'] or '')[:120].replace('\n', ' ')
                lines.append(f"      Primary endpoint: {pom}")
            if s.get("primary_completion_date"):
                lines.append(f"      Primary completion: {s['primary_completion_date']}")

    lines.append(f"\n  (Sources fetched: {', '.join(facts['_sources_succeeded'])} in "
                 f"{facts.get('_fetch_duration_ms', '?')}ms)")
    lines.append("")
    return "\n".join(lines)
