#!/usr/bin/env python3
"""P1: restructure results/ to paper-aligned layout (Table 1–4 + appendix + supplemental)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

TABLE1 = RESULTS / "table1_characterization"
TABLE2 = RESULTS / "table2_global_support"
TABLE3 = RESULTS / "table3_representation"
TABLE4 = RESULTS / "table4_mitigation"
APPENDIX = RESULTS / "appendix"
SUPPLEMENTAL = RESULTS / "supplemental"
PROBE_AUX = SUPPLEMENTAL / "probe_auxiliary"

COHORT_KEYS = (
    "shopping_qwen_n249",
    "shopping_mistral_n250",
    "fhir_qwen_n973",
    "fhir_mistral_n847",
)

PER_RENAMES = (
    "phantom_merge_v2_per_trajectory.jsonl",
    "phantom_merge_v2_fhir_per_trajectory.jsonl",
)


def _mv(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"MOVE {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")


def _rename_in_dir(directory: Path, old: str, new: str) -> None:
    p = directory / old
    if p.is_file() and not (directory / new).exists():
        p.rename(directory / new)
        print(f"RENAME {p.relative_to(REPO)} -> {new}")


def migrate_table1() -> None:
    legacy = RESULTS / "failure_detection"
    if legacy.is_dir() and not TABLE1.is_dir():
        shutil.move(str(legacy), str(TABLE1))
        print(f"MOVE {legacy.relative_to(REPO)} -> table1_characterization/")
    elif legacy.is_dir() and TABLE1.is_dir():
        for item in legacy.iterdir():
            target = TABLE1 / item.name
            if not target.exists():
                shutil.move(str(item), str(target))

    if not TABLE1.is_dir():
        TABLE1.mkdir(parents=True, exist_ok=True)

    _rename_in_dir(TABLE1, "paper_counts_absolute.json", "counts.json")
    _rename_in_dir(TABLE1, "COHORT_MANIFEST.json", "cohort_index.json")

    for key in COHORT_KEYS:
        cdir = TABLE1 / key
        if not cdir.is_dir():
            continue
        for old in PER_RENAMES:
            _rename_in_dir(cdir, old, "per_trajectory.jsonl")
        _rename_in_dir(cdir, "phantom_merge_v2_summary_absolute.json", "summary_absolute.json")


def migrate_probe_split() -> None:
    legacy_probe = RESULTS / "probe_fhir"
    if not legacy_probe.is_dir():
        return

    TABLE3.mkdir(parents=True, exist_ok=True)
    TABLE4.mkdir(parents=True, exist_ok=True)
    PROBE_AUX.mkdir(parents=True, exist_ok=True)

    table3_moves = {
        "claims.parquet": "claims.parquet",
        "splits.json": "splits.json",
        "p2_results_summary.json": "bcp_detect.json",
        "bcp_detect_metrics.json": "bcp_detect_metrics.json",
        "oc_bcp_summary.json": "oc_bcp.json",
        "cp_com_mechanism.json": "cp_com_mechanism.json",
        "export_stats.json": "export_stats.json",
        "hidden_extract_manifest_L63.json": "hidden_extract_manifest_L63.json",
        "ablation_input.json": "ablation_input.json",
        "ablation_pool.json": "ablation_pool.json",
        "bcp_layer_sweep.json": "bcp_layer_sweep.json",
        "bcp_multi_head.json": "bcp_multi_head.json",
    }
    for old, new in table3_moves.items():
        _mv(legacy_probe / old, TABLE3 / new)

    if (legacy_probe / "feature_bank").is_dir():
        _mv(legacy_probe / "feature_bank", TABLE3 / "feature_bank")

    table4_moves = {
        "mitigation_gating_curve.json": "gating_curve.json",
        "mitigation_oracle_upper_bound.json": "oracle_upper_bound.json",
    }
    for old, new in table4_moves.items():
        _mv(legacy_probe / old, TABLE4 / new)

    if (legacy_probe / "mitigation").is_dir():
        _mv(legacy_probe / "mitigation", TABLE4 / "mitigation")

    msps = TABLE4 / "mitigation" / "msps_ultimate_results.json"
    if msps.is_file() and not (TABLE4 / "msps_test146.json").is_file():
        shutil.copy2(msps, TABLE4 / "msps_test146.json")

    aux_names = [
        "p3_results_summary.json",
        "p4_results_summary.json",
        "shopping_ood_eval.json",
        "validity_controls.json",
    ]
    for name in aux_names:
        _mv(legacy_probe / name, PROBE_AUX / name)

    summaries = PROBE_AUX / "summaries"
    if summaries.is_dir():
        shutil.rmtree(summaries)

    # Remove empty probe_fhir
    if legacy_probe.is_dir():
        remaining = list(legacy_probe.rglob("*"))
        if not any(p.is_file() for p in remaining):
            legacy_probe.rmdir()
        elif legacy_probe.exists():
            # leftover files -> probe_auxiliary
            for p in legacy_probe.iterdir():
                _mv(p, PROBE_AUX / p.name)
            try:
                legacy_probe.rmdir()
            except OSError:
                pass


def migrate_appendix_and_supplemental() -> None:
    _mv(RESULTS / "appendix_tables", APPENDIX)

    decomp_legacy = RESULTS / "trajectory_decomposition"
    decomp_new = SUPPLEMENTAL / "trajectory_decomposition"
    if decomp_legacy.is_dir():
        _mv(decomp_legacy, decomp_new)

    # table2 already at results/table2_global_support from P0.5


def rewrite_cohort_manifest_paths() -> None:
    per_name = "per_trajectory.jsonl"
    for key in COHORT_KEYS:
        manifest = TABLE1 / key / "cohort_manifest.json"
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rel = f"results/table1_characterization/{key}/{per_name}"
        data["source_per"] = rel
        data["sealed_per"] = rel
        data["cohort_key"] = key
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_artifact_json() -> None:
    payload = {
        "release": "emnlp2026_phantom_merge_anonymous",
        "repository": "https://github.com/emnlp2026-phantommerge/PhantomMerge",
        "protocol_version": {
            "shopping": "shopping_pm_judge_vNEXT",
            "fhir": "fhir_pm_judge_vNEXT",
        },
        "verify": "bash reproduce/verify.sh",
        "tables": {
            "table1": {
                "counts": "results/table1_characterization/counts.json",
                "cohort_index": "results/table1_characterization/cohort_index.json",
                "cohorts": {
                    k: f"results/table1_characterization/{k}/per_trajectory.jsonl"
                    for k in COHORT_KEYS
                },
            },
            "table2": {
                "global_support_baseline": "results/table2_global_support/baseline_checker.json"
            },
            "table3": {
                "bcp_detect": "results/table3_representation/bcp_detect.json",
                "bcp_detect_metrics": "results/table3_representation/bcp_detect_metrics.json",
                "oc_bcp": "results/table3_representation/oc_bcp.json",
                "claims_parquet": "results/table3_representation/claims.parquet",
                "splits": "results/table3_representation/splits.json",
            },
            "table4": {
                "msps_test146": "results/table4_mitigation/msps_test146.json",
                "gating_curve": "results/table4_mitigation/gating_curve.json",
                "cb_retention_audit": "results/appendix/mitigation_cb_retention_audit.json",
                "tau_pm_curve": "results/appendix/mitigation_tau_pm_curve.json",
            },
        },
        "supplemental": {
            "mitigation_utility_proxy": "results/supplemental/mitigation_utility_proxy.json",
            "trajectory_decomposition": "results/supplemental/trajectory_decomposition/decomposition.csv",
            "probe_auxiliary": "results/supplemental/probe_auxiliary/",
            "gold_reference_panel": {
                "manifest": "results/supplemental/gold_reference_panel/manifest.json",
                "claims": "results/supplemental/gold_reference_panel/claims_100.jsonl",
            },
        },
        "figures": {
            "appendix_scripts": "figures/appendix/",
            "data": "results/appendix/manifest.json",
        },
        "not_shipped": [
            "agent_rollouts",
            "feature_bank_tensors_npy",
            "qualitative_case_trajectories",
            "excluded_judge_failure_trajectories",
        ],
    }
    (REPO / "ARTIFACT.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("WROTE ARTIFACT.json")


def write_results_manifest() -> None:
    payload = {
        "release": "emnlp2026_phantom_merge_anonymous",
        "repository": "https://github.com/emnlp2026-phantommerge/PhantomMerge",
        "purpose": "Frozen VNEXT protocol labels and paper tables (P1 layout).",
        "verify": "bash reproduce/verify.sh",
        "artifact_index": "ARTIFACT.json",
        "table1": "results/table1_characterization",
        "table2": "results/table2_global_support",
        "table3": "results/table3_representation",
        "table4": "results/table4_mitigation",
        "appendix": "results/appendix",
        "supplemental": "results/supplemental",
    }
    (RESULTS / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    if not (TABLE1 / "counts.json").is_file():
        migrate_table1()
        rewrite_cohort_manifest_paths()
    migrate_probe_split()
    migrate_appendix_and_supplemental()
    write_artifact_json()
    write_results_manifest()
    print("\nP1 layout migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
