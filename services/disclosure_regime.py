"""disclosure_regime — classify a catalyst_type by how reliably its
outcome is publicly disclosed.

Drives the probability source decision tree in
services/probability_lookup.py:

  MANDATED       → outcome is publicly required to be reported
                   (FDA decisions, AdComm votes, NDA/BLA submissions).
                   Historical base rate is unbiased — use it.
  SEMI_MANDATED  → typically reported but with positive selection
                   (Phase 3 readouts; negatives often delayed).
                   Blend historical base rate with options-implied.
  VOLUNTARY      → only reported if positive (Phase 1/2 readouts,
                   trial initiations). Historical base rate is
                   destroyed by survivorship bias — use options or
                   abstain.

Classification verified against negative-outcome publish rate per
catalyst_type from the labeled set (see docs/spec-03-prediction.md
§ Regime classification — observed pct_neg per type 2026-05-03):

  AdComm           80.0% neg  → MANDATED (committee votes both ways)
  Submission       22.2% neg  → MANDATED (CRL is mandated disclosure)
  FDA Decision     19.8% neg  → MANDATED (PDUFA outcome is public)
  Phase 3 Readout  18.7% neg  → SEMI_MANDATED (negatives delayed)
  Other            10.4% neg  → SEMI_MANDATED (mixed bag)
  Phase 2 Readout   8.4% neg  → VOLUNTARY (negatives mostly suppressed)
  Trial Initiation  5.0% neg  → VOLUNTARY (non-initiation rarely announced)
  Phase 1 Readout   1.5% neg  → VOLUNTARY (heavy survivorship)

UNCLASSIFIED types (BLA submission, NDA submission, Phase 4 Readout,
Phase 0 Readout, Phase 1/2 Readout, Phase 1/2/3 Readout, New Product
Launch, Clinical Trial, Partnership, _no_history_known) default to
VOLUNTARY until backfill grows enough to classify them empirically.
"""
from __future__ import annotations

from typing import Literal

DisclosureRegime = Literal["MANDATED", "SEMI_MANDATED", "VOLUNTARY"]

# Authoritative map. Keep in sync with the regime classification table
# in docs/spec-03-prediction.md. Update when the labeled set adds
# enough rows to reclassify a previously UNCLASSIFIED type.
DISCLOSURE_REGIME: dict[str, DisclosureRegime] = {
    # MANDATED — outcome is publicly disclosed by regulation/process
    "FDA Decision":          "MANDATED",
    "PDUFA Decision":        "MANDATED",
    "Regulatory Decision":   "MANDATED",
    "AdComm":                "MANDATED",
    "AdCom":                 "MANDATED",
    "Advisory Committee":    "MANDATED",
    "Submission":            "MANDATED",
    "BLA submission":        "MANDATED",
    "NDA submission":        "MANDATED",

    # SEMI_MANDATED — typically reported but selection biased toward positive
    "Phase 3 Readout":       "SEMI_MANDATED",
    "Phase 3":               "SEMI_MANDATED",
    "Other":                 "SEMI_MANDATED",

    # VOLUNTARY — heavy survivorship bias; negatives often suppressed
    "Phase 2 Readout":       "VOLUNTARY",
    "Phase 2":               "VOLUNTARY",
    "Phase 1 Readout":       "VOLUNTARY",
    "Phase 1":               "VOLUNTARY",
    "Phase 1/2 Readout":     "VOLUNTARY",
    "Phase 1/2":             "VOLUNTARY",
    "Phase 1/2/3 Readout":   "VOLUNTARY",
    "Phase 0 Readout":       "VOLUNTARY",
    "Phase 4 Readout":       "VOLUNTARY",
    "Trial Initiation":      "VOLUNTARY",
    "Clinical Trial":        "VOLUNTARY",
    "Clinical Trial Readout": "VOLUNTARY",
    "Partnership":           "VOLUNTARY",
    "Earnings":              "VOLUNTARY",
    "Product Launch":        "VOLUNTARY",
    "Commercial Launch":     "VOLUNTARY",
    "New Product Launch":    "VOLUNTARY",
}

# Catalyst types where NPV-driven prediction is the primary source.
# Per the user-approved plan, this is the binary-resolving set.
NPV_PRIMARY_TYPES: frozenset[str] = frozenset({
    "FDA Decision",
    "PDUFA Decision",
    "Regulatory Decision",
    "Phase 3 Readout",
    "Phase 3",
    "Phase 2 Readout",
    "Phase 2",
})


def classify_disclosure_regime(catalyst_type: str | None) -> DisclosureRegime:
    """Return MANDATED / SEMI_MANDATED / VOLUNTARY for a catalyst_type.

    Unknown / null types default to VOLUNTARY (the conservative choice
    — VOLUNTARY requires options data or abstention, so an unknown
    type can't accidentally publish a survivored historical base rate).
    """
    if not catalyst_type:
        return "VOLUNTARY"
    return DISCLOSURE_REGIME.get(catalyst_type, "VOLUNTARY")


def is_npv_primary_type(catalyst_type: str | None) -> bool:
    """True if this catalyst_type uses NPV-driven prediction as the
    primary source (statistical model is the calibration check)."""
    if not catalyst_type:
        return False
    return catalyst_type in NPV_PRIMARY_TYPES
