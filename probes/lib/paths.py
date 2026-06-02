"""Default paths for the anonymous release (P1 layout under results/)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
TABLE1 = RESULTS / "table1_characterization"
TABLE3 = RESULTS / "table3_representation"
TABLE4 = RESULTS / "table4_mitigation"

PER = "per_trajectory.jsonl"

FHIR_QWEN_VNEXT = TABLE1 / "fhir_qwen_n973" / PER
SHOPPING_QWEN_VNEXT = TABLE1 / "shopping_qwen_n249" / PER
FHIR_MISTRAL_VNEXT = TABLE1 / "fhir_mistral_n847" / PER
SHOPPING_MISTRAL_VNEXT = TABLE1 / "shopping_mistral_n250" / PER

DEFAULT_MODEL_DIR = REPO_ROOT / ".cache/huggingface"
DEFAULT_MODEL_ID = "Qwen/Qwen3-32B"

FHIR_PROBE_OUT = TABLE3
SHOPPING_PROBE_OUT = RESULTS / "probe_shopping"

# Legacy alias
PROBE_FHIR = TABLE3

VALID_PIPELINE_VERSIONS = frozenset(
    {"fhir_pm_judge_vNEXT", "shopping_pm_judge_vNEXT"}
)
