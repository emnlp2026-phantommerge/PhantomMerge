"""Import shared PM VNEXT taxonomy from Shopping bench (see README.md)."""

from __future__ import annotations

import sys
from pathlib import Path

_SHOPPING_SRC = Path(__file__).resolve().parents[2] / "shopping" / "src"
if str(_SHOPPING_SRC) not in sys.path:
    sys.path.insert(0, str(_SHOPPING_SRC))

from analysis.pm_eval_vnext import (  # noqa: E402
    PM_FAILURE_LABELS,
    claim_label_counts_from_claims,
    compute_claim_structure_counts,
    derive_support_labels,
    is_pm_relevant_claim_fhir,
    load_fhir_slot_config,
    normalize_fhir_anchor_resolution,
    primary_support_label,
    process_fhir_claim_record,
    qa_claim_label_consistency,
    refresh_row_labels,
    trajectory_labels_from_claims,
)

FHIR_PM_PIPELINE_VERSION = "fhir_pm_judge_vNEXT"

__all__ = [
    "FHIR_PM_PIPELINE_VERSION",
    "PM_FAILURE_LABELS",
    "claim_label_counts_from_claims",
    "compute_claim_structure_counts",
    "derive_support_labels",
    "is_pm_relevant_claim_fhir",
    "load_fhir_slot_config",
    "normalize_fhir_anchor_resolution",
    "primary_support_label",
    "process_fhir_claim_record",
    "qa_claim_label_consistency",
    "refresh_row_labels",
    "trajectory_labels_from_claims",
]
