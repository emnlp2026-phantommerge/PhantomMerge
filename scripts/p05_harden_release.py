#!/usr/bin/env python3
"""Phase 0.5: remove reviewer-risk artifacts; relocate sealed results; sanitize paths."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

FORBIDDEN_MARKERS = (
    "/home/wangshu",
    "Phantom_Merge/new_version",
    "Phantom_Merge/runs",
    "human_audit",
    "validity_audit_three_way",
    "validity_audit_summary",
    "Paper-expert",
    "pgcs_v1_judge_failed",
    "deterministic_flip",
)

# Shipped curated panel intentionally includes pipeline flip metadata.
_MARKER_SCAN_SKIP_PREFIXES = (
    "results/supplemental/gold_reference_panel/",
)

DELETE_PATHS = [
    "results/analysis/validity_audit_summary.json",
    "results/analysis/validity_audit_three_way.json",
    "results/case_studies",
    "figures/case_studies",
    "results/table1_characterization/fhir_mistral_n847/excluded_pm_judge_failed.jsonl",
    "results/table1_characterization/fhir_mistral_n847/task_eval.json",
    "results/table1_characterization/fhir_qwen_n973/task_eval.json",
    "results/supplemental/probe_auxiliary/p4_mitigation_research_summary.json",
    "results/supplemental/probe_auxiliary/summaries",
    "results/failure_detection",
    "results/probe_fhir",
    "results/appendix_tables",
    "scripts/merge_three_way_audit.py",
    "scripts/analyze_validity_audit.py",
    "scripts/audit_public_docs.py",
    "scripts/audit_code_hygiene.py",
    "scripts/audit_paper_alignment.py",
    "scripts/install_case_studies.py",
    "scripts/finalize_with_cases.sh",
    "scripts/compute_mitigation_utility_proxy.py",
    "scripts/compute_global_support_table.py",
    "probes/summarize_mitigation_research.py",
    "probes/run_pgcs_onepass.py",
    "probes/lib/pgcs_onepass.py",
    "probes/backfill_mitigated_claims.py",
    "probes/reresolve_anchor_mitigation.py",
    "11717_Phantom_Merge_When_Your_.pdf",
    "RELEASE_NOTES.txt",
    "probes/run_mitigation_research.sh",
    "binding/shopping/scripts/qa_pm_labels.py",
    "binding/fhir/scripts/qa_fhir_pm_labels.py",
    "benchmarks",
    "experiments",
    "experiments/exp5_validity_controls",
]

MOVE_MAP = {
    "results/analysis/global_support_baseline_table.json": "results/table2_global_support/baseline_checker.json",
    "results/analysis/mitigation_utility_proxy.json": "results/supplemental/mitigation_utility_proxy.json",
    "results/analysis/mitigation_utility_pareto_table.csv": "results/supplemental/mitigation_utility_pareto_table.csv",
}


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _relocate() -> None:
    for src_rel, dst_rel in MOVE_MAP.items():
        src = REPO / src_rel
        dst = REPO / dst_rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"MOVED {src_rel} -> {dst_rel}")


def _sanitize_json_obj(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_obj(x) for x in obj]
    if isinstance(obj, str):
        s = obj
        s = s.replace("/home/wangshu/Phantom_Merge/", "")
        s = s.replace("/home/wangshu/", "")
        s = re.sub(r"Phantom_Merge/new_version/\S+", "", s)
        if "human_audit" in s and s.startswith("/"):
            return "redacted"
        return s
    return obj


def _sanitize_json_file(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    new = _sanitize_json_obj(data)
    if isinstance(new, dict) and path.name == "mitigation_utility_proxy.json":
        new.pop("probe_dir", None)
        new.pop("data_source", None)
    path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")


def _sanitize_all_results_json() -> None:
    for path in RESULTS.rglob("*.json"):
        _sanitize_json_file(path)


def _update_cohort_manifest() -> None:
    path = RESULTS / "table1_characterization/cohort_index.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    ex = data.get("fhir_mistral_excluded", {})
    ex.pop("list_path", None)
    ex["note"] = (
        "Trajectories excluded from the sealed cohort due to judge timeout/parse failure "
        "(count only; per-trajectory rows not redistributed in this release)."
    )
    data["fhir_mistral_excluded"] = ex
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_results_manifest() -> None:
    from p01_migrate_results_layout import write_results_manifest

    write_results_manifest()


def _delete_nested_readmes() -> None:
    for path in REPO.rglob("README.md"):
        if path.resolve() != (REPO / "README.md").resolve():
            path.unlink()
            print(f"DELETED {path.relative_to(REPO)}")


def _scan_forbidden() -> list[str]:
    hits: list[str] = []
    skip_dirs = {".git", ".venv", "venv", "__pycache__"}
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(p in skip_dirs for p in path.parts):
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_MARKER_SCAN_SKIP_PREFIXES):
            continue
        if path.suffix not in {".json", ".py", ".sh", ".txt", ".md", ".csv"}:
            continue
        if path.name in {"p05_harden_release.py", "check_scaffold.py", "p01_migrate_results_layout.py", "p02_migrate_code_layout.py", "p03_migrate_reproduce_layout.py", "p04_finalize_release.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(REPO)}: {marker!r}")
    return hits


def main() -> int:
    analysis_dir = RESULTS / "analysis"
    for rel in DELETE_PATHS:
        _rm(REPO / rel)
    if analysis_dir.is_dir() and not any(analysis_dir.iterdir()):
        analysis_dir.rmdir()

    _relocate()
    if analysis_dir.is_dir():
        _rm(analysis_dir)

    _sanitize_all_results_json()
    _update_cohort_manifest()
    _write_results_manifest()
    _delete_nested_readmes()

    hits = _scan_forbidden()
    if hits:
        print("\nP0.5 FORBIDDEN MARKER SCAN FAILED:")
        for h in hits[:40]:
            print(" ", h)
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        return 1

    print("\nP0.5 hardening completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
