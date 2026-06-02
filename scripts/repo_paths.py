"""Repository path constants (anonymous EMNLP release, P1 results + P2 code layout)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ANONYMOUS_REPO_URL = "https://github.com/emnlp2026-phantommerge/PhantomMerge"
RESULTS = REPO_ROOT / "results"
RUNS = REPO_ROOT / "runs"

# P2 code layout
BINDING = REPO_ROOT / "binding"
BINDING_SHOPPING = BINDING / "shopping"
BINDING_FHIR = BINDING / "fhir"
PROBES = REPO_ROOT / "probes"

TABLE1 = RESULTS / "table1_characterization"
TABLE2 = RESULTS / "table2_global_support"
TABLE3 = RESULTS / "table3_representation"
TABLE4 = RESULTS / "table4_mitigation"
APPENDIX = RESULTS / "appendix"
SUPPLEMENTAL = RESULTS / "supplemental"

# Legacy aliases (internal scripts may still import these)
FAILURE_DETECTION = TABLE1
PROBE_FHIR = TABLE3
APPENDIX_TABLES = APPENDIX
PAPER_EXPORTS = APPENDIX
TRAJECTORY_DECOMP = SUPPLEMENTAL / "trajectory_decomposition"
BENCHMARKS_PROBE = PROBES  # pre-P2 alias

PER_TRAJECTORY = "per_trajectory.jsonl"
SUMMARY_ABSOLUTE = "summary_absolute.json"

COHORTS = {
    "shopping_qwen_n249": TABLE1 / "shopping_qwen_n249",
    "shopping_mistral_n250": TABLE1 / "shopping_mistral_n250",
    "fhir_qwen_n973": TABLE1 / "fhir_qwen_n973",
    "fhir_mistral_n847": TABLE1 / "fhir_mistral_n847",
}

SHOPPING_QWEN_PER = COHORTS["shopping_qwen_n249"] / PER_TRAJECTORY
FHIR_QWEN_PER = COHORTS["fhir_qwen_n973"] / PER_TRAJECTORY

PAPER_COUNTS = TABLE1 / "counts.json"
COHORT_MANIFEST = TABLE1 / "cohort_index.json"
COHORT_INDEX = COHORT_MANIFEST

TABLE2_GLOBAL_SUPPORT = TABLE2 / "baseline_checker.json"

BCP_DETECT = TABLE3 / "bcp_detect.json"
MSPS_TEST146 = TABLE4 / "msps_test146.json"
MSPS_ULTIMATE = TABLE4 / "mitigation" / "msps_ultimate_results.json"

PM_EVAL_VNEXT = BINDING_SHOPPING / "src/analysis/pm_eval_vnext.py"
FHIR_PM_JUDGE = BINDING_FHIR / "scripts/analyze_phantom_merge_fhir_v2.py"
PROBE_VALIDATE_P0 = PROBES / "validate_p0.py"

REPRODUCE = REPO_ROOT / "reproduce"
REPRODUCE_VERIFY = REPRODUCE / "verify.sh"
REPRODUCE_INDEX = REPRODUCE / "INDEX.json"

PROBE_RUNS_FHIR = RUNS / "probe" / "fhir_qwen_vnext"
PROBE_RUNS_SHOPPING = RUNS / "probe" / "shopping_qwen_vnext"
