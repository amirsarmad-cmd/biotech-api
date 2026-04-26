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

# Orange Book is published as a ZIP bundle on the FDA website. It contains
# pipe-separated text files (products, patents, exclusivity) updated monthly.
# Reference: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download?attachment"

# Cache structure: orange_book is parsed once per ~7 days into a single
# in-Redis dict keyed by appl_no, plus secondary indices on brand_name and
# ingredient. Subsequent lookups are O(1).
_OB_CACHE_KEY = "orangebook:parsed:v1"
_OB_TTL_SEC = 7 * 86400  # 7 days


def _download_and_parse_orange_book() -> Optional[Dict]:
    """Download the Orange Book ZIP, parse the three pipe-delimited files,
    and return a lookup dict.

    Output structure:
        {
          "by_appl_no": {...},
          "by_brand": {...},
          "by_ingredient": {...},
          "_meta": {...},
        }
    Returns None on failure with detailed reason logged.
    """
    import io, zipfile, urllib.request, urllib.error

    download_attempts = []
    # Try multiple candidate URLs — FDA changes them occasionally
    urls_to_try = [
        "https://www.fda.gov/media/76860/download?attachment",
        "https://www.fda.gov/media/76860/download",
    ]

    buf = None
    for url in urls_to_try:
        try:
            logger.info(f"Orange Book: trying {url} (~50MB)...")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; biotech-screener/1.0; Railway)",
                "Accept": "application/zip,application/octet-stream,*/*",
            })
            # The download is large — give it 120s
            with urllib.request.urlopen(req, timeout=120) as resp:
                buf = resp.read()
                download_attempts.append({"url": url, "status": "ok", "size": len(buf)})
                break
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            download_attempts.append({"url": url, "status": "fail", "error": str(e)[:200]})
            logger.warning(f"Orange Book download failed for {url}: {e}")
            continue

    if buf is None:
        logger.warning(f"Orange Book ALL downloads failed: {download_attempts}")
        # Return marker so caller can surface the reason
        return {"_download_failed": True, "_attempts": download_attempts}

    try:
        zf = zipfile.ZipFile(io.BytesIO(buf))
        names = zf.namelist()
        logger.info(f"Orange Book: ZIP extracted ({len(buf)} bytes), files: {names}")
    except zipfile.BadZipFile as e:
        logger.warning(f"Orange Book ZIP malformed: {e}")
        return {"_download_failed": True, "_zip_error": str(e), "_buf_size": len(buf)}

    # Find the three data files (case-insensitive — FDA renames occasionally)
    def _find(name_part: str) -> Optional[str]:
        for n in names:
            if name_part.lower() in n.lower() and (n.endswith(".txt") or n.endswith(".TXT")):
                return n
        return None
    products_file = _find("products")
    patent_file = _find("patent")
    exclusivity_file = _find("exclusivity")

    by_appl_no: Dict[str, Dict] = {}
    by_brand: Dict[str, set] = {}
    by_ingredient: Dict[str, set] = {}
    files_parsed = []

    # ─── Parse products.txt ─────
    if products_file:
        try:
            text = zf.read(products_file).decode("utf-8", errors="replace")
            lines = text.split("\n")
            header = [h.strip() for h in (lines[0] if lines else "").split("~")]
            # Some recent Orange Book files use ~ as separator instead of |
            if "Trade_Name" not in header and len(header) < 3:
                header = [h.strip() for h in lines[0].split("|")]
                sep = "|"
            else:
                sep = "~"
            # Map column indices
            idx = {h: i for i, h in enumerate(header)}
            for line in lines[1:]:
                if not line.strip():
                    continue
                cols = line.split(sep)
                if len(cols) < len(header):
                    continue
                appl_no = (cols[idx.get("Appl_No", 0)] or "").strip()
                if not appl_no:
                    continue
                trade = (cols[idx.get("Trade_Name", 1)] or "").strip().upper()
                ingredient = (cols[idx.get("Ingredient", 0)] or "").strip().upper()
                product = {
                    "trade_name": trade,
                    "ingredient": ingredient,
                    "strength": (cols[idx.get("Strength", 0)] or "").strip(),
                    "appl_type": (cols[idx.get("Appl_Type", 0)] or "").strip(),
                    "approval_date": (cols[idx.get("Approval_Date", 0)] or "").strip(),
                    "rld": (cols[idx.get("RLD", 0)] or "").strip(),
                    "te_code": (cols[idx.get("TE_Code", 0)] or "").strip(),
                }
                d = by_appl_no.setdefault(appl_no,
                                          {"products": [], "patents": [], "exclusivities": []})
                d["products"].append(product)
                if trade:
                    by_brand.setdefault(trade, set()).add(appl_no)
                if ingredient:
                    by_ingredient.setdefault(ingredient, set()).add(appl_no)
            files_parsed.append(products_file)
        except Exception as e:
            logger.warning(f"products.txt parse failed: {e}")

    # ─── Parse patent.txt ─────
    if patent_file:
        try:
            text = zf.read(patent_file).decode("utf-8", errors="replace")
            lines = text.split("\n")
            header = [h.strip() for h in (lines[0] if lines else "").split("~")]
            if "Patent_No" not in header and len(header) < 3:
                header = [h.strip() for h in lines[0].split("|")]
                sep = "|"
            else:
                sep = "~"
            idx = {h: i for i, h in enumerate(header)}
            for line in lines[1:]:
                if not line.strip():
                    continue
                cols = line.split(sep)
                if len(cols) < len(header):
                    continue
                appl_no = (cols[idx.get("Appl_No", 0)] or "").strip()
                if not appl_no or appl_no not in by_appl_no:
                    # Patent for an Appl_No we didn't see in products — still record
                    by_appl_no.setdefault(appl_no,
                                          {"products": [], "patents": [], "exclusivities": []})
                pat = {
                    "patent_no": (cols[idx.get("Patent_No", 0)] or "").strip(),
                    "expire_date": (cols[idx.get("Patent_Expire_Date_Text", 0)] or "").strip(),
                    "drug_substance_flag": (cols[idx.get("Drug_Substance_Flag", 0)] or "").strip() == "Y",
                    "drug_product_flag": (cols[idx.get("Drug_Product_Flag", 0)] or "").strip() == "Y",
                    "patent_use_code": (cols[idx.get("Patent_Use_Code", 0)] or "").strip(),
                    "delist_flag": (cols[idx.get("Delist_Flag", 0)] or "").strip() == "Y",
                }
                if pat["patent_no"] and not pat["delist_flag"]:
                    by_appl_no[appl_no]["patents"].append(pat)
            files_parsed.append(patent_file)
        except Exception as e:
            logger.warning(f"patent.txt parse failed: {e}")

    # ─── Parse exclusivity.txt ─────
    if exclusivity_file:
        try:
            text = zf.read(exclusivity_file).decode("utf-8", errors="replace")
            lines = text.split("\n")
            header = [h.strip() for h in (lines[0] if lines else "").split("~")]
            if "Exclusivity_Code" not in header and len(header) < 3:
                header = [h.strip() for h in lines[0].split("|")]
                sep = "|"
            else:
                sep = "~"
            idx = {h: i for i, h in enumerate(header)}
            for line in lines[1:]:
                if not line.strip():
                    continue
                cols = line.split(sep)
                if len(cols) < len(header):
                    continue
                appl_no = (cols[idx.get("Appl_No", 0)] or "").strip()
                if not appl_no:
                    continue
                if appl_no not in by_appl_no:
                    by_appl_no[appl_no] = {"products": [], "patents": [], "exclusivities": []}
                excl = {
                    "code": (cols[idx.get("Exclusivity_Code", 0)] or "").strip(),
                    "date": (cols[idx.get("Exclusivity_Date", 0)] or "").strip(),
                }
                if excl["code"]:
                    by_appl_no[appl_no]["exclusivities"].append(excl)
            files_parsed.append(exclusivity_file)
        except Exception as e:
            logger.warning(f"exclusivity.txt parse failed: {e}")

    # ─── Compute LOE per appl_no ─────
    for appl_no, d in by_appl_no.items():
        all_dates = []
        for p in d["patents"]:
            if p.get("expire_date"):
                all_dates.append(p["expire_date"])
        for e in d["exclusivities"]:
            if e.get("date"):
                all_dates.append(e["date"])
        # Convert MMM DD, YYYY format to ISO if needed
        norm_dates = []
        for ds in all_dates:
            iso = _parse_ob_date(ds)
            if iso:
                norm_dates.append(iso)
        norm_dates.sort()
        d["earliest_loe"] = norm_dates[0] if norm_dates else None
        d["latest_loe"] = norm_dates[-1] if norm_dates else None

    out = {
        "by_appl_no": by_appl_no,
        "by_brand": {k: list(v) for k, v in by_brand.items()},
        "by_ingredient": {k: list(v) for k, v in by_ingredient.items()},
        "_meta": {
            "fetched_at": __import__("datetime").datetime.utcnow().isoformat(),
            "files_parsed": files_parsed,
            "n_appl_no": len(by_appl_no),
            "n_brands": len(by_brand),
            "n_ingredients": len(by_ingredient),
        },
    }
    logger.info(f"Orange Book parsed: {out['_meta']}")
    return out


def _parse_ob_date(date_str: str) -> Optional[str]:
    """Parse Orange Book date (e.g. 'Mar 21, 2030', 'Jan 2, 2030') → ISO YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # Already ISO?
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str
    from datetime import datetime
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _get_orange_book_lookup(force_refresh: bool = False) -> Optional[Dict]:
    """Get the parsed Orange Book lookup dict — from Redis cache or by
    downloading + parsing. Cached for 7 days. Negative results cached
    for 5 minutes to avoid hammering FDA.
    """
    r = _redis_client()
    if r and not force_refresh:
        try:
            raw = r.get(_OB_CACHE_KEY)
            if raw:
                return json.loads(raw)
            # Check negative cache
            neg = r.get(_OB_CACHE_KEY + ":neg")
            if neg:
                return json.loads(neg)
        except Exception:
            pass
    # Cache miss — download + parse
    parsed = _download_and_parse_orange_book()
    if parsed and parsed.get("_download_failed"):
        # Store the error info as a negative cache entry briefly
        if r:
            try:
                r.setex(_OB_CACHE_KEY + ":neg", 300, json.dumps(parsed, default=str))
            except Exception:
                pass
        return parsed  # caller checks _download_failed
    if parsed and r:
        try:
            r.setex(_OB_CACHE_KEY, _OB_TTL_SEC, json.dumps(parsed, default=str))
            logger.info(f"Orange Book cached in Redis (TTL {_OB_TTL_SEC}s)")
        except Exception as e:
            logger.warning(f"Orange Book Redis set failed: {e}")
    return parsed


def fetch_orange_book_loe(drug_name: str = None, ingredient: str = None) -> Optional[Dict]:
    """Look up patent + exclusivity expirations for a drug from Orange Book.

    First call after deploy will be slow (~30-60s) as it downloads + parses
    the ZIP. Subsequent calls are O(1) Redis lookups.

    Returns:
        {
          "drug_name": str,
          "ingredient": str,
          "matched_appl_no": str,
          "products": [...],
          "patents": [{"patent_no", "expire_date", "drug_substance_flag", ...}],
          "exclusivities": [{"code", "date"}],
          "earliest_loe": str | None,  # ISO date
          "latest_loe": str | None,
          "_source": "orange_book",
        }
        Or {"_not_found": True} if the drug isn't in Orange Book (typical for
        biologics — Orange Book is for small-molecule drugs only; biologics
        are in the Purple Book which is a separate database).
    """
    if not (drug_name or ingredient):
        return None

    cache_key = f"orangebook:lookup:{(drug_name or ingredient or '').lower()}"
    cached = _cached_get(cache_key)
    if cached is not None:
        cached["_from_cache"] = True
        return cached

    lookup = _get_orange_book_lookup()
    if not lookup or lookup.get("_download_failed"):
        return {
            "drug_name": drug_name,
            "ingredient": ingredient,
            "_source": "orange_book",
            "_status": "download_failed",
            "_attempts": (lookup or {}).get("_attempts", []),
            "_note": "Could not download Orange Book ZIP from FDA — try again later",
        }

    # Find matching appl_no(s) — by brand or ingredient name (case-insensitive)
    candidates = []
    if drug_name:
        candidates.extend(lookup.get("by_brand", {}).get(drug_name.upper(), []))
        # Also try just the first word (e.g. 'Lonvoguran ziclumeran' → 'LONVOGURAN')
        if " " in drug_name:
            candidates.extend(lookup.get("by_brand", {}).get(drug_name.split()[0].upper(), []))
    if ingredient:
        candidates.extend(lookup.get("by_ingredient", {}).get(ingredient.upper(), []))
    elif drug_name:
        # Fallback: try drug_name as ingredient
        candidates.extend(lookup.get("by_ingredient", {}).get(drug_name.upper(), []))
        if " " in drug_name:
            candidates.extend(lookup.get("by_ingredient", {}).get(drug_name.split()[0].upper(), []))

    # Deduplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    if not candidates:
        out = {
            "drug_name": drug_name,
            "ingredient": ingredient,
            "_source": "orange_book",
            "_not_found": True,
            "_note": "Drug not in Orange Book — likely biologic (Purple Book) or pre-approval",
        }
        _cached_set(cache_key, out, ttl_sec=86400)
        return out

    # Use the first match — for now, simple. Could merge across multiple Appl_Nos
    # for drugs with multiple formulations.
    appl_no = candidates[0]
    d = lookup["by_appl_no"].get(appl_no, {})
    out = {
        "drug_name": drug_name,
        "ingredient": ingredient,
        "matched_appl_no": appl_no,
        "products": d.get("products", [])[:5],
        "patents": d.get("patents", []),
        "exclusivities": d.get("exclusivities", []),
        "earliest_loe": d.get("earliest_loe"),
        "latest_loe": d.get("latest_loe"),
        "n_other_matching_appl_nos": max(0, len(candidates) - 1),
        "_source": "orange_book",
        "_from_cache": False,
    }
    _cached_set(cache_key, out, ttl_sec=86400)
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

    # Orange Book — real LOE dates (small molecules only; biologics in Purple Book)
    # Skipped if explicitly disabled — first call after deploy is slow (~30-60s)
    # so allow opting out for performance-sensitive paths.
    if os.getenv("ORANGE_BOOK_ENABLED", "1") != "0":
        attempted.append("orange_book")
        try:
            orange = fetch_orange_book_loe(drug_name=drug_name)
            if orange and not orange.get("_not_found") and not orange.get("_status") == "download_failed":
                if orange.get("earliest_loe") or orange.get("patents"):
                    facts["orange_book"] = orange
                    succeeded.append("orange_book")
        except Exception as e:
            logger.info(f"orange_book lookup failed: {e}")

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

    # Orange Book section — real patents + exclusivities
    if facts.get("orange_book"):
        ob = facts["orange_book"]
        lines.append(f"\n[Orange Book — patent / exclusivity (small molecules only)]")
        lines.append(f"  Matched Appl_No: {ob.get('matched_appl_no')}")
        if ob.get("earliest_loe"):
            lines.append(f"  Earliest LOE date: {ob['earliest_loe']} (USE THIS as patent_expiry_date)")
        if ob.get("latest_loe"):
            lines.append(f"  Latest LOE date:   {ob['latest_loe']}")
        n_pat = len(ob.get("patents", []))
        n_excl = len(ob.get("exclusivities", []))
        lines.append(f"  Patents on record: {n_pat} | Exclusivities: {n_excl}")
        # Show 3 nearest-expiring patents
        all_loe = []
        for p in (ob.get("patents") or []):
            iso = _parse_ob_date(p.get("expire_date", ""))
            if iso:
                all_loe.append((iso, "patent", p.get("patent_no")))
        for e in (ob.get("exclusivities") or []):
            iso = _parse_ob_date(e.get("date", ""))
            if iso:
                all_loe.append((iso, "exclusivity", e.get("code")))
        all_loe.sort()
        for iso, kind, code in all_loe[:3]:
            lines.append(f"    {iso}  {kind}  {code}")

    lines.append(f"\n  (Sources fetched: {', '.join(facts['_sources_succeeded'])} in "
                 f"{facts.get('_fetch_duration_ms', '?')}ms)")
    lines.append("")
    return "\n".join(lines)
