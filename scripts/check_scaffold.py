#!/usr/bin/env python3
"""Release verification: sealed P2 layout, integrity checks, no forbidden leakage."""

from __future__ import annotations

import json
import subprocess
import sys

from repo_paths import (
    BCP_DETECT,
    COHORT_MANIFEST,
    COHORTS,
    FHIR_PM_JUDGE,
    FHIR_QWEN_PER,
    MSPS_ULTIMATE,
    PAPER_COUNTS,
    PM_EVAL_VNEXT,
    PROBE_VALIDATE_P0,
    PROBES,
    REPO_ROOT,
    REPRODUCE,
    REPRODUCE_INDEX,
    REPRODUCE_VERIFY,
    SHOPPING_QWEN_PER,
    TABLE3,
)

REQUIRED_DIRS = [
    "scripts",
    "reproduce",
    "binding/shopping",
    "binding/fhir",
    "probes",
    "results/table1_characterization",
    "results/table3_representation",
    "results/table4_mitigation",
    "results/appendix",
    "results/table2_global_support",
    "results/supplemental",
    "figures/appendix",
    "runs/probe",
]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
    "ARTIFACT.json",
    "results/MANIFEST.json",
    str(PAPER_COUNTS.relative_to(REPO_ROOT)),
    str(COHORT_MANIFEST.relative_to(REPO_ROOT)),
    str(SHOPPING_QWEN_PER.relative_to(REPO_ROOT)),
    str(FHIR_QWEN_PER.relative_to(REPO_ROOT)),
    str(BCP_DETECT.relative_to(REPO_ROOT)),
    str(MSPS_ULTIMATE.relative_to(REPO_ROOT)),
    str((TABLE3 / "claims.parquet").relative_to(REPO_ROOT)),
    "results/table2_global_support/baseline_checker.json",
    "results/supplemental/gold_reference_panel/manifest.json",
    "results/supplemental/gold_reference_panel/claims_100.jsonl",
    str(PM_EVAL_VNEXT.relative_to(REPO_ROOT)),
    str(FHIR_PM_JUDGE.relative_to(REPO_ROOT)),
    str(PROBE_VALIDATE_P0.relative_to(REPO_ROOT)),
    str(REPRODUCE_VERIFY.relative_to(REPO_ROOT)),
    str(REPRODUCE_INDEX.relative_to(REPO_ROOT)),
    "scripts/run_diagnosis.sh",
    "scripts/run_probe.sh",
    "scripts/run_mitigation.sh",
]

FORBIDDEN_PATHS = [
    "results/case_studies",
    "results/analysis",
    "figures/case_studies",
    "docs",
    "benchmarks",
    "experiments",
    "results/failure_detection",
    "results/probe_fhir",
    "results/appendix_tables",
    "results/table1_characterization/fhir_mistral_n847/excluded_pm_judge_failed.jsonl",
    "results/supplemental/probe_auxiliary/p4_mitigation_research_summary.json",
    "binding/shopping/scripts/qa_pm_labels.py",
    "binding/fhir/scripts/qa_fhir_pm_labels.py",
]

FORBIDDEN_MARKERS = (
    "/home/wangshu",
    "Phantom_Merge/new_version",
    "Phantom_Merge/runs",
    "human_audit",
    "validity_audit_summary",
    "validity_audit_three_way",
    "Paper-expert",
    "pgcs_v1_judge_failed",
    "deterministic_flip",
)

_MARKER_SCAN_SKIP_PREFIXES = (
    "results/supplemental/gold_reference_panel/",
)


def _extra_readmes() -> list[str]:
    bad: list[str] = []
    root_readme = (REPO_ROOT / "README.md").resolve()
    for path in REPO_ROOT.rglob("README.md"):
        if path.resolve() != root_readme:
            bad.append(str(path.relative_to(REPO_ROOT)))
    return bad


def _grep_forbidden() -> list[str]:
    hits: list[str] = []
    skip = {
        "p05_harden_release.py",
        "check_scaffold.py",
        "p01_migrate_results_layout.py",
        "p02_migrate_code_layout.py",
        "p03_migrate_reproduce_layout.py",
        "p04_finalize_release.py",
    }
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".sh", ".md", ".txt", ".json", ".csv"}:
            continue
        if path.name in skip:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(_MARKER_SCAN_SKIP_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in FORBIDDEN_MARKERS:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: contains {needle!r}")
    return hits


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_DIRS:
        if not (REPO_ROOT / rel).is_dir():
            errors.append(f"Missing directory: {rel}")

    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).is_file():
            errors.append(f"Missing file: {rel}")

    for rel in FORBIDDEN_PATHS:
        if (REPO_ROOT / rel).exists():
            errors.append(f"Forbidden path still present: {rel}")

    errors.extend(_extra_readmes())

    for key, cohort_dir in COHORTS.items():
        per_path = cohort_dir / "per_trajectory.jsonl"
        if not per_path.is_file():
            errors.append(f"No per_trajectory.jsonl in cohort {key}")
        if (cohort_dir / "task_eval.json").is_file():
            errors.append(f"task_eval.json must not ship in cohort {key}")

    if PAPER_COUNTS.is_file():
        data = json.loads(PAPER_COUNTS.read_text(encoding="utf-8"))
        expected = data.get("paper_denominators", {})
        for key, n in expected.items():
            manifest = COHORTS.get(key)
            if manifest and (manifest / "cohort_manifest.json").is_file():
                cm = json.loads((manifest / "cohort_manifest.json").read_text())
                manifest_n = cm.get("paper_denominator_n", cm.get("paper_n"))
                if manifest_n != n:
                    errors.append(f"Denominator mismatch {key}: manifest vs counts")

    cm = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    if cm.get("fhir_mistral_excluded", {}).get("list_path"):
        errors.append("cohort_index.json must not reference excluded_pm_judge_failed.jsonl")

    errors.extend(_grep_forbidden())

    print("Running validate_p0.py ...")
    proc = subprocess.run(
        [sys.executable, str(PROBE_VALIDATE_P0)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        errors.append("validate_p0.py failed")
        if proc.stderr:
            print(proc.stderr)

    print("Running verify_results_integrity.py ...")
    proc_i = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/verify_results_integrity.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(proc_i.stdout)
    if proc_i.returncode != 0:
        errors.append("verify_results_integrity.py failed")
        if proc_i.stderr:
            print(proc_i.stderr)

    print("Running verify_paper_counts.py ...")
    proc2 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/verify_paper_counts.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(proc2.stdout)
    if proc2.returncode != 0:
        errors.append("verify_paper_counts.py failed")

    print("Running audit_submission_url.py ...")
    proc_u = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/audit_submission_url.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(proc_u.stdout)
    if proc_u.returncode != 0:
        errors.append("audit_submission_url.py failed")
        if proc_u.stderr:
            print(proc_u.stderr)

    if errors:
        print("\nRELEASE CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nRelease check PASSED (P4 final anonymous release layout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
