#!/usr/bin/env python3
"""Rewrite results/ metadata JSON to P1 release-relative paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repo_paths import (
    APPENDIX,
    COHORTS,
    PER_TRAJECTORY,
    REPO_ROOT,
    RESULTS,
    SUMMARY_ABSOLUTE,
    SUPPLEMENTAL,
    TABLE1,
    TABLE2,
    TABLE3,
    TABLE4,
)

PROBE_AUX = SUPPLEMENTAL / "probe_auxiliary"


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def sanitize_cohort_manifests() -> None:
    for key, cohort_dir in COHORTS.items():
        manifest_path = cohort_dir / "cohort_manifest.json"
        if not manifest_path.is_file():
            continue
        per_path = cohort_dir / PER_TRAJECTORY
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["source_per"] = _rel(per_path)
        data["sealed_per"] = data["source_per"]
        digest = hashlib.sha256(per_path.read_bytes()).hexdigest()
        data["sha256_per_source"] = digest
        data["sha256_per_sealed"] = digest
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sanitize_fhir_summaries() -> None:
    for key in ("fhir_qwen_n973", "fhir_mistral_n847"):
        summary_path = COHORTS[key] / SUMMARY_ABSOLUTE
        if not summary_path.is_file():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        task_eval = COHORTS[key] / "task_eval.json"
        if task_eval.is_file():
            data["eval_json"] = _rel(task_eval)
        elif "eval_json" in data:
            del data["eval_json"]
        summary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sanitize_probe_sidecars() -> None:
    mapping = {
        TABLE3 / "export_stats.json": {
            "per_path": _rel(COHORTS["fhir_qwen_n973"] / PER_TRAJECTORY),
            "out_dir": _rel(TABLE3),
            "claims_parquet": _rel(TABLE3 / "claims.parquet"),
        },
    }
    for path, updates in mapping.items():
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    manifest = TABLE3 / "feature_bank" / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["claims_parquet"] = _rel(TABLE3 / "claims.parquet")
        data["oc_triplets"] = _rel(TABLE3 / "feature_bank/oc_triplets.jsonl")
        if "model_path" in data and str(data["model_path"]).startswith("/"):
            data["model_path"] = ".cache/huggingface/hub/Qwen3-32B"
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _rewrite_path_string(s: str) -> str:
    replacements = (
        ("results/failure_detection/", "results/table1_characterization/"),
        ("phantom_merge_v2_fhir_per_trajectory.jsonl", PER_TRAJECTORY),
        ("phantom_merge_v2_per_trajectory.jsonl", PER_TRAJECTORY),
        ("phantom_merge_v2_summary_absolute.json", SUMMARY_ABSOLUTE),
        ("paper_counts_absolute.json", "counts.json"),
        ("results/probe_fhir/", "results/table3_representation/"),
        ("results/probe_fhir", "results/table3_representation"),
        ("p2_results_summary.json", "bcp_detect.json"),
        ("oc_bcp_summary.json", "oc_bcp.json"),
        ("mitigation_gating_curve.json", "gating_curve.json"),
        ("mitigation_oracle_upper_bound.json", "oracle_upper_bound.json"),
        ("results/appendix_tables/", "results/appendix/"),
        ("results/appendix_tables", "results/appendix"),
        ("results/trajectory_decomposition/", "results/supplemental/trajectory_decomposition/"),
        ("runs/probe/fhir_qwen_vnext", "results/table3_representation"),
        ("runs/probe/shopping_qwen_vnext", "results/probe_shopping"),
        (
            "mitigation_pgcs_onepass_v2_judge_gpu1_bf16",
            "mitigation/pgcs_onepass_gpu1_bf16",
        ),
        ("mitigation_pgcs_onepass_v2/", "mitigation/pgcs_onepass_gpu1_bf16/"),
        ("/paper_expert_exports", ""),
    )
    for old, new in replacements:
        s = s.replace(old, new)
    # mitigation artifacts live under table4 after P1
    if "results/table3_representation/mitigation/" in s:
        s = s.replace(
            "results/table3_representation/mitigation/",
            "results/table4_mitigation/mitigation/",
        )
    return s


def sanitize_json_tree(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_tree(x) for x in obj]
    if isinstance(obj, str) and (
        "failure_detection" in obj
        or "probe_fhir" in obj
        or "appendix_tables" in obj
        or "phantom_merge_v2" in obj
        or "runs/" in obj
        or "main_results" in obj
    ):
        return _rewrite_path_string(obj)
    return obj


def sanitize_all_results_json() -> None:
    roots = [TABLE1, TABLE2, TABLE3, TABLE4, APPENDIX, SUPPLEMENTAL]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.is_symlink():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            path.write_text(
                json.dumps(sanitize_json_tree(data), indent=2) + "\n",
                encoding="utf-8",
            )


def write_results_manifest() -> None:
    payload = {
        "release": "emnlp2026_phantom_merge_anonymous",
        "repository": "https://github.com/emnlp2026-phantommerge/PhantomMerge",
        "purpose": "Frozen VNEXT protocol labels and paper tables (P1 layout).",
        "verify": "bash reproduce/verify.sh",
        "artifact_index": "ARTIFACT.json",
        "table1": _rel(TABLE1),
        "table2": _rel(TABLE2),
        "table3": _rel(TABLE3),
        "table4": _rel(TABLE4),
        "appendix": _rel(APPENDIX),
        "supplemental": _rel(SUPPLEMENTAL),
    }
    (RESULTS / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    sanitize_cohort_manifests()
    sanitize_fhir_summaries()
    sanitize_probe_sidecars()
    sanitize_all_results_json()
    write_results_manifest()
    print("Sanitized results metadata (P1 layout) under", RESULTS)


if __name__ == "__main__":
    main()
