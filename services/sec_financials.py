"""sec_financials — Layer 1 SEC EDGAR balance-sheet + capital structure ingest.

Solves ChatGPT critique item #5: "Add cash/debt/dilution-adjusted equity value,
not only asset rNPV divided by market cap."

For a small biotech, the equity value isn't just rNPV / market_cap. It's:

    equity_value = (rNPV − net_debt + cash) / shares_outstanding
                   adjusted for projected dilution from runway

If a $300M company has $60M cash, $0 debt, $30M/quarter burn, and a $1.2B
rNPV, the per-share fair value depends on whether they need to raise
$120M at a 30% discount before the catalyst (= 40% dilution = 71% of asset
value per existing share).

EDGAR endpoints (public, no auth):
  - companyconcept: tag-level XBRL values across all filings
    https://data.sec.gov/api/xbrl/companyconcept/CIK<10-digit>/<taxonomy>/<tag>.json
  - companyfacts:   all facts for a company in one response
    https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit>.json
  - tickers:        ticker → CIK mapping
    https://www.sec.gov/files/company_tickers.json

EDGAR requires User-Agent header with company + email contact (free).
Rate limit: 10 requests/sec (we cache aggressively to stay well under).
"""
import os
import json
import logging
import re
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

EDGAR_BASE_DATA = "https://data.sec.gov"
EDGAR_BASE_WWW = "https://www.sec.gov"
USER_AGENT = os.getenv("SEC_USER_AGENT", "Biotech Screener research@biotech-screener.app")

# Tags to pull. SEC's us-gaap taxonomy uses these standard names; some
# companies use alternative tags so we try multiple per concept.
BALANCE_SHEET_TAGS = {
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
        "ConvertibleDebtNoncurrent",
    ],
    "current_debt": [
        "DebtCurrent",
        "ShortTermBorrowings",
        "ShortTermDebt",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    # P&L for burn-rate
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromOperatingActivities",
    ],
    "net_loss": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
}


def _redis_client():
    try:
        from services.cache import get_redis
        return get_redis()
    except Exception:
        return None


def _cached_get(key: str):
    r = _redis_client()
    if r is None: return None
    try:
        raw = r.get(key)
        if raw: return json.loads(raw)
    except Exception:
        pass
    return None


def _cached_set(key: str, val, ttl_sec: int = 86400):
    r = _redis_client()
    if r is None: return
    try:
        r.setex(key, ttl_sec, json.dumps(val, default=str))
    except Exception:
        pass


def _http_get(url: str, timeout: int = 12) -> Optional[Dict]:
    """SEC EDGAR GET — requires User-Agent header per their fair-use policy."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        if resp.status_code == 404:
            return {"_status": 404}
        if resp.status_code == 403:
            logger.warning(f"SEC 403 — User-Agent may be missing/invalid: {USER_AGENT}")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.info(f"SEC http_get failed {url}: {e}")
        return None
    except ValueError as e:
        logger.info(f"SEC json parse failed: {e}")
        return None


# ────────────────────────────────────────────────────────────
# Ticker → CIK
# ────────────────────────────────────────────────────────────

_TICKER_CIK_CACHE_KEY = "sec:ticker_cik_map"


def _load_ticker_cik_map() -> Dict[str, str]:
    """Load SEC's ticker→CIK mapping. Cached 7 days (rarely changes)."""
    cached = _cached_get(_TICKER_CIK_CACHE_KEY)
    if cached:
        return cached
    data = _http_get(f"{EDGAR_BASE_WWW}/files/company_tickers.json")
    if not data:
        return {}
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {}
    for entry in data.values() if isinstance(data, dict) else []:
        if not isinstance(entry, dict):
            continue
        ticker = (entry.get("ticker") or "").upper().strip()
        cik = entry.get("cik_str")
        if ticker and cik:
            # Pad to 10 digits per EDGAR requirement
            mapping[ticker] = str(cik).zfill(10)
    _cached_set(_TICKER_CIK_CACHE_KEY, mapping, ttl_sec=7 * 86400)
    return mapping


def ticker_to_cik(ticker: str) -> Optional[str]:
    """Resolve a ticker to its 10-digit CIK string."""
    if not ticker:
        return None
    mapping = _load_ticker_cik_map()
    return mapping.get(ticker.upper().strip())


# ────────────────────────────────────────────────────────────
# Company facts (XBRL bulk endpoint)
# ────────────────────────────────────────────────────────────

def _fetch_company_facts(cik: str) -> Optional[Dict]:
    """Pull the full XBRL facts blob for a company."""
    cache_key = f"sec:facts:{cik}"
    cached = _cached_get(cache_key)
    if cached:
        return cached
    data = _http_get(f"{EDGAR_BASE_DATA}/api/xbrl/companyfacts/CIK{cik}.json")
    if not data or data.get("_status") == 404:
        return None
    # Cache 24h — companies file at most quarterly
    _cached_set(cache_key, data, ttl_sec=86400)
    return data


def _latest_value_for_tags(facts: Dict, tag_candidates: List[str],
                          taxonomies: Tuple[str, ...] = ("us-gaap", "ifrs-full", "dei")
                          ) -> Optional[Dict]:
    """Search a companyfacts blob for the most-recent value across multiple
    candidate tags. Returns the most recent unit-USD or shares value with
    its filing date.

    Returns: {"value": float, "unit": str, "end": "YYYY-MM-DD",
              "filed": "YYYY-MM-DD", "form": str, "tag": str, "taxonomy": str}
    Or None if no candidate tag had data.
    """
    if not facts or not isinstance(facts, dict):
        return None
    facts_block = facts.get("facts", {}) or {}
    best = None
    for taxonomy in taxonomies:
        tax_block = facts_block.get(taxonomy, {}) or {}
        for tag in tag_candidates:
            tag_block = tax_block.get(tag)
            if not tag_block:
                continue
            units = tag_block.get("units", {}) or {}
            # Prefer USD; fallback to first unit
            for unit_key in ("USD", "shares", "USD/shares"):
                if unit_key in units:
                    entries = units[unit_key]
                    if not entries:
                        continue
                    # Take most-recent by 'end' date
                    sorted_entries = sorted(
                        [e for e in entries if e.get("end")],
                        key=lambda e: e["end"], reverse=True,
                    )
                    if not sorted_entries:
                        continue
                    e = sorted_entries[0]
                    candidate = {
                        "value": float(e.get("val", 0)),
                        "unit": unit_key,
                        "end": e.get("end"),
                        "filed": e.get("filed"),
                        "form": e.get("form"),
                        "tag": tag,
                        "taxonomy": taxonomy,
                        "fy": e.get("fy"),
                        "fp": e.get("fp"),
                    }
                    if best is None or (candidate["end"] or "") > (best["end"] or ""):
                        best = candidate
                    break  # found this tag, move to next tag
    return best


def _quarterly_burn_from_facts(facts: Dict) -> Optional[Dict]:
    """Compute most-recent quarterly cash burn from operating cash flow.

    Uses NetCashProvidedByUsedInOperatingActivities for Q-period filings.
    Returns: {"quarterly_burn_usd": float (positive = burn), "as_of": str, "form": str}
    """
    if not facts: return None
    facts_block = facts.get("facts", {}).get("us-gaap", {}) or {}
    for tag in BALANCE_SHEET_TAGS["operating_cash_flow"]:
        block = facts_block.get(tag)
        if not block:
            continue
        usd = (block.get("units") or {}).get("USD") or []
        if not usd:
            continue
        # Filter to quarterly (form 10-Q) and YTD-like periods
        quarterly = []
        for e in usd:
            form = e.get("form", "")
            if form != "10-Q":
                continue
            # Period length: end - start should be ~90 days for a single Q
            try:
                start = datetime.fromisoformat(e.get("start"))
                end = datetime.fromisoformat(e.get("end"))
                days = (end - start).days
            except Exception:
                continue
            # Single-Q range: ~80-100 days
            if 75 <= days <= 100:
                quarterly.append({
                    "value": float(e.get("val", 0)),
                    "end": e.get("end"),
                    "filed": e.get("filed"),
                    "form": form,
                    "days": days,
                    "tag": tag,
                })
        if quarterly:
            quarterly.sort(key=lambda x: x["end"], reverse=True)
            most_recent = quarterly[0]
            # Operating CF negative for biotechs in burn mode — flip sign for burn
            burn = -float(most_recent["value"]) if most_recent["value"] < 0 else 0
            return {
                "quarterly_burn_usd": burn,
                "operating_cash_flow_usd": float(most_recent["value"]),
                "period_end": most_recent["end"],
                "period_days": most_recent["days"],
                "form": most_recent["form"],
                "filed": most_recent["filed"],
                "tag": most_recent["tag"],
            }
    return None


# ────────────────────────────────────────────────────────────
# Top-level: fetch_capital_structure
# ────────────────────────────────────────────────────────────

def fetch_capital_structure(ticker: str) -> Optional[Dict]:
    """Pull cash, debt, shares, and runway from SEC EDGAR for any US-listed ticker.
    
    Returns:
      {
        "ticker": str,
        "cik": str,
        "as_of_filing": str,           # most recent quarter end
        "cash_and_equivalents": float, # USD
        "short_term_investments": float,
        "total_cash": float,           # cash + ST investments
        "long_term_debt": float,
        "current_debt": float,
        "total_debt": float,
        "net_debt": float,             # debt - cash (negative = net cash)
        "shares_outstanding": float,
        "quarterly_burn_usd": float,   # positive = burning
        "cash_runway_months": float,   # total_cash / monthly_burn
        "needs_financing_within_12mo": bool,
        "_source": "sec_edgar",
        "_filings": {tag: {...}},      # provenance per field
      }
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "_error": "ticker not found in SEC company list"}

    facts = _fetch_company_facts(cik)
    if not facts:
        return {"ticker": ticker, "cik": cik, "_error": "facts fetch failed"}

    # Pull each balance-sheet field
    fields = {}
    filings = {}
    for label, tag_candidates in BALANCE_SHEET_TAGS.items():
        if label == "operating_cash_flow":
            continue  # handled separately for burn
        result = _latest_value_for_tags(facts, tag_candidates)
        if result:
            fields[label] = result["value"]
            filings[label] = {
                "tag": result["tag"], "taxonomy": result["taxonomy"],
                "end": result["end"], "filed": result["filed"], "form": result["form"],
            }

    # Burn rate (operating cash flow)
    burn = _quarterly_burn_from_facts(facts)

    # Compute derived fields
    cash = fields.get("cash_and_equivalents", 0)
    sti = fields.get("short_term_investments", 0)
    total_cash = cash + sti
    lt_debt = fields.get("long_term_debt", 0)
    cur_debt = fields.get("current_debt", 0)
    total_debt = lt_debt + cur_debt
    net_debt = total_debt - total_cash

    quarterly_burn = (burn or {}).get("quarterly_burn_usd", 0)
    monthly_burn = quarterly_burn / 3.0 if quarterly_burn > 0 else 0
    runway_months = (total_cash / monthly_burn) if monthly_burn > 0 else None
    if runway_months and runway_months > 999:
        runway_months = 999  # cap for display

    # Most recent filing date across all fields
    all_ends = sorted([f.get("end") for f in filings.values() if f.get("end")], reverse=True)
    as_of = all_ends[0] if all_ends else None

    out = {
        "ticker": ticker.upper(),
        "cik": cik,
        "as_of_filing": as_of,
        "cash_and_equivalents": cash,
        "short_term_investments": sti,
        "total_cash": total_cash,
        "long_term_debt": lt_debt,
        "current_debt": cur_debt,
        "total_debt": total_debt,
        "net_debt": net_debt,
        "shares_outstanding": fields.get("shares_outstanding"),
        "quarterly_burn_usd": quarterly_burn if quarterly_burn else None,
        "monthly_burn_usd": monthly_burn if monthly_burn else None,
        "cash_runway_months": round(runway_months, 1) if runway_months else None,
        "needs_financing_within_12mo": (runway_months is not None and runway_months < 12),
        "_source": "sec_edgar",
        "_filings": filings,
        "_burn_filing": burn,
    }
    return out


# ────────────────────────────────────────────────────────────
# Equity value computation — the actual ChatGPT critique fix
# ────────────────────────────────────────────────────────────

def compute_equity_value(rnpv_m: float, cap_struct: Dict,
                          dilution_assumed_pct: Optional[float] = None,
                          financing_discount_pct: float = 30.0) -> Dict:
    """Compute equity value from rNPV adjusting for cash, debt, dilution.

    Per ChatGPT critique: 'Add cash/debt/dilution-adjusted equity value,
    not only asset rNPV divided by market cap.'

    Formula (small biotech with runway concerns):
        equity_value_pre = rNPV − total_debt + total_cash
        if needs_financing:
            # Project required raise to fund next 12-18mo
            # Estimate dilution at financing_discount_pct
            raise_amount = max(0, 12 × monthly_burn − total_cash)
            dilution = raise_amount / (current_market_value × (1 − discount/100))
        else:
            dilution = dilution_assumed_pct / 100  (if user supplied)
        equity_value_post = equity_value_pre × (1 − dilution)
        per_share = equity_value_post / shares_outstanding

    Returns:
      {
        "rnpv_m": float,
        "total_cash_m": float,
        "total_debt_m": float,
        "enterprise_to_equity_adj_m": float,  # = -debt + cash
        "equity_value_pre_dilution_m": float,
        "projected_dilution_pct": float,
        "projected_raise_m": float,
        "dilution_source": "user_override" | "runway_projection" | "none",
        "equity_value_post_dilution_m": float,
        "shares_outstanding_m": float,
        "per_share_value_usd": float,
        "current_market_cap_m": float,        # passed in or None
        "implied_upside_pct": float,
        "warnings": [str],
        "_provenance": "sec_edgar",
      }
    """
    warnings = []
    cs = cap_struct or {}
    cash_m = (cs.get("total_cash") or 0) / 1e6
    debt_m = (cs.get("total_debt") or 0) / 1e6
    shares_m = (cs.get("shares_outstanding") or 0) / 1e6
    monthly_burn_m = (cs.get("monthly_burn_usd") or 0) / 1e6

    if shares_m <= 0:
        warnings.append("shares_outstanding from SEC missing — per-share calc unavailable")

    # Equity value before dilution = rNPV − debt + cash
    equity_pre = rnpv_m - debt_m + cash_m

    # Dilution projection
    dilution_pct = 0.0
    raise_m = 0.0
    dilution_source = "none"

    if dilution_assumed_pct is not None and dilution_assumed_pct > 0:
        # User override
        dilution_pct = float(dilution_assumed_pct)
        dilution_source = "user_override"
    elif cs.get("needs_financing_within_12mo") and monthly_burn_m > 0:
        # Project required raise: 18 months runway target
        target_runway_months = 18
        cash_needed_m = max(0, monthly_burn_m * target_runway_months - cash_m)
        if cash_needed_m > 0:
            raise_m = cash_needed_m
            # Estimate market cap for dilution math — use rNPV as proxy if no
            # current market cap supplied (the caller can override this)
            market_proxy_m = max(equity_pre, 50)  # floor at $50M to avoid division explosions
            # Discount applied to financing
            effective_price = market_proxy_m * (1 - financing_discount_pct / 100.0)
            if effective_price > 0:
                dilution_pct = 100.0 * raise_m / effective_price
                # Cap at 75% — anything more would rarely happen in practice
                dilution_pct = min(dilution_pct, 75.0)
                dilution_source = "runway_projection"
                if dilution_pct > 35:
                    warnings.append(
                        f"Projected dilution >35% to fund 18-month runway "
                        f"({raise_m:.0f}M raise at {financing_discount_pct}% discount). "
                        f"Material headwind to per-share value."
                    )

    # Apply dilution
    equity_post = equity_pre * (1 - dilution_pct / 100.0)

    # Per-share
    per_share = (equity_post / shares_m) if shares_m > 0 else None

    # Sanity warnings
    if cs.get("cash_runway_months") and cs["cash_runway_months"] < 6:
        warnings.append(
            f"Critical: cash runway only {cs['cash_runway_months']:.1f} months. "
            f"Near-term financing required at potentially distressed terms."
        )
    if debt_m > equity_pre and equity_pre > 0:
        warnings.append(
            f"Total debt (${debt_m:.0f}M) exceeds asset NPV (${equity_pre:.0f}M) — "
            f"shareholders likely subordinated."
        )

    return {
        "rnpv_m": rnpv_m,
        "total_cash_m": cash_m,
        "total_debt_m": debt_m,
        "net_cash_adjustment_m": cash_m - debt_m,
        "equity_value_pre_dilution_m": equity_pre,
        "projected_dilution_pct": round(dilution_pct, 2),
        "projected_raise_m": round(raise_m, 2),
        "dilution_source": dilution_source,
        "financing_discount_assumed_pct": financing_discount_pct,
        "equity_value_post_dilution_m": equity_post,
        "shares_outstanding_m": round(shares_m, 2) if shares_m else None,
        "per_share_value_usd": round(per_share, 2) if per_share else None,
        "monthly_burn_m": round(monthly_burn_m, 2) if monthly_burn_m else None,
        "cash_runway_months": cs.get("cash_runway_months"),
        "needs_financing_within_12mo": cs.get("needs_financing_within_12mo", False),
        "as_of_filing": cs.get("as_of_filing"),
        "warnings": warnings,
        "_provenance": "sec_edgar",
    }
