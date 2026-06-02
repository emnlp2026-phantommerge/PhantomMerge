#!/usr/bin/env python3
"""Export appendix tables: CB audit, random control, τ curves, CSVs (optional cases)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from eval_mitigation_ultimate import (  # noqa: E402
    MSPSConfig,
    _attach_probs,
    _load_oc_margins,
    _should_drop,
    _trajectory_eval,
)
from lib.feature_bank import load_bcp_matrix, load_manifest  # noqa: E402
from lib.mitigation import PM_FAILURE, trajectory_has_pm_rows  # noqa: E402
from lib.mitigation_claim_audit import (  # noqa: E402
    audit_gating,
    is_correct_binding_claim,
    is_pm_claim,
    kept_counts_per_group,
    random_retention_matched_eval,
    write_json,
)
from lib.p2_eval import train_bcp_hidden  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402

TABLE1 = REPO / "results" / "table1_characterization"
FAILURE_DETECTION = TABLE1  # legacy alias
# Qualitative case bundles are not part of the anonymous release artifact.
CASE_TRAJ_DIR = REPO / "results" / "_case_studies_disabled"
PGCS_DIR_DEFAULT = FHIR_PROBE_OUT / "mitigation/pgcs_onepass_gpu1_bf16"


def _load_msps_probs(probe_dir: Path, layer_slot: int = 1) -> pd.DataFrame:
    df_base = pd.read_parquet(probe_dir / "claims.parquet")
    prob: dict[str, np.ndarray] = {}
    for label in ("y_pm", "y_com", "y_cp"):
        clf, _, H, _ = train_bcp_hidden(
            probe_dir, label=label, variant="full", pool="mean", layer_slot=layer_slot
        )
        prob[label] = clf.predict_proba(H)[:, 1]
    return _attach_probs(df_base, prob["y_pm"], prob["y_com"], prob["y_cp"])


def _load_bcp_probs(probe_dir: Path) -> pd.DataFrame:
    pack = joblib.load(probe_dir / "probe_bcp_detect.joblib")
    clf = pack["clf"]
    layer_slot = int(pack.get("layer_slot", 2))
    variant = str(pack.get("variant", "full"))
    pool = str(pack.get("pool", "mean"))
    df = pd.read_parquet(probe_dir / "claims.parquet")
    bank = probe_dir / "feature_bank"
    man = load_manifest(bank)
    if not man or int(man.get("completed_rows", 0)) < len(df):
        raise FileNotFoundError("Incomplete feature_bank")
    H = load_bcp_matrix(bank, variant=variant, pool=pool, layer_slot=layer_slot)
    df = df.copy()
    df["p_pm"] = clf.predict_proba(H)[:, 1]
    return df


def _msps_drop_fn(cfg: MSPSConfig, oc_margins: dict[str, float]):
    def drop(row: dict) -> bool:
        return _should_drop(row, cfg, oc_margins)

    return drop


def _bcp_drop_fn(tau: float):
    def drop(row: dict) -> bool:
        return float(row.get("p_pm", 1)) > tau

    return drop


def _msps_tau_sweep(
    df: pd.DataFrame,
    oc_margins: dict[str, float],
    *,
    taus: list[float],
    split: str,
) -> list[dict[str, Any]]:
    sub = df[df["split"].astype(str) == split]
    rows_out: list[dict[str, Any]] = []
    for tau in taus:
        cfg = MSPSConfig(tau, tau, tau, 0.0, False)
        ev = _trajectory_eval(sub, cfg, oc_margins)
        audit = audit_gating(sub, _msps_drop_fn(cfg, oc_margins))
        rows_out.append(
            {
                "method": "MSPS",
                "split": split,
                "tau": tau,
                "pm_rate": ev["gated_pm_rate"],
                "pm_count": ev["gated_pm_count"],
                "pm_reduction": ev["pm_reduction"],
                "claims_retained_mean": ev["claims_retained_mean"],
                "cb_retention_rate": audit.cb_retention_rate,
                "removed_cb_fraction": audit.removed_cb_fraction_of_all_removed,
                "cb_claims_removed": audit.cb_claims_removed,
            }
        )
    return rows_out


def _build_mitigation_extended_table(
    probe_dir: Path,
    df_msps: pd.DataFrame,
    df_bcp: pd.DataFrame,
    oc_margins: dict[str, float],
    msps_cfg: MSPSConfig,
    random_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    test = df_msps[df_msps["split"].astype(str) == "test"]
    test_bcp = df_bcp[df_bcp["split"].astype(str) == "test"]
    rows: list[dict[str, Any]] = []

    def add_row(method: str, audit, extra: dict | None = None) -> None:
        r = {
            "method": method,
            "split": "test",
            "n_trajectories": audit.n_trajectories,
            "pm_count": audit.gated_pm_count,
            "pm_rate": audit.gated_pm_rate,
            "claims_retained_total": audit.claims_total_gated,
            "claims_retained_mean": audit.claims_retained_mean,
            "pm_claims_total": audit.pm_claims_total_gated,
            "cb_claims_baseline": audit.cb_claims_total_baseline,
            "cb_claims_retained": audit.cb_claims_retained,
            "cb_claims_removed": audit.cb_claims_removed,
            "cb_retention_rate": audit.cb_retention_rate,
            "removed_cb_fraction_of_removed": audit.removed_cb_fraction_of_all_removed,
            "pm_claims_removed": audit.pm_claims_removed,
        }
        if extra:
            r.update(extra)
        rows.append(r)

    baseline_audit = audit_gating(test, lambda _r: False)
    add_row("baseline", baseline_audit)

    msps_audit = audit_gating(test, _msps_drop_fn(msps_cfg, oc_margins))
    add_row("MSPS_tau0.45", msps_audit)

    bcp_audit = audit_gating(test_bcp, _bcp_drop_fn(0.5))
    add_row("BCP_tau0.5", bcp_audit)

    oracle_audit = audit_gating(test, lambda r: is_pm_claim(r))
    add_row("oracle_delete_pm_claims", oracle_audit)

    # PGCS from per_case
    pgcs_path = PGCS_DIR_DEFAULT / "pgcs_onepass_per_case.jsonl"
    if pgcs_path.is_file():
        pgcs_pm = 0
        retained: list[float] = []
        judged: list[int] = []
        label_only = 0
        pgcs_records: list[dict] = []
        for line in pgcs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != "test":
                continue
            pgcs_records.append(rec)
            pgcs_pm += int(bool(rec.get("pgcs_has_phantom_merge")))
            retained.append(float(rec.get("claim_retention_rate") or 0))
            judged.append(int(rec.get("pgcs_n_claims") or 0))
            if rec.get("label_pm_after_filter_only"):
                label_only += 1
        n = len(pgcs_records) or 146
        rows.append(
            {
                "method": "PGCS_onepass",
                "split": "test",
                "n_trajectories": n,
                "pm_count": pgcs_pm,
                "pm_rate": pgcs_pm / n,
                "claims_retained_mean": float(np.mean(retained)) if retained else None,
                "pgcs_judged_claims_mean": float(np.mean(judged)) if judged else None,
                "pgcs_judged_claims_total": sum(judged),
                "label_pm_after_filter_only_rate": label_only / n,
            }
        )

    rows.append(
        {
            "method": "random_retention_matched_MSPS",
            "split": "test",
            "n_trajectories": baseline_audit.n_trajectories,
            "pm_count_mean": random_stats.get("pm_count_mean"),
            "pm_rate_mean": random_stats.get("pm_rate_mean"),
            "pm_rate_std": random_stats.get("pm_rate_std"),
            "pm_rate_p05": random_stats.get("pm_rate_p05"),
            "pm_rate_p95": random_stats.get("pm_rate_p95"),
            "n_seeds": random_stats.get("n_seeds"),
            "note": "Random claim deletion matching MSPS per-trajectory retention (200 seeds)",
        }
    )
    return rows


def _export_task_pm_heatmap(out_dir: Path) -> None:
    cells = [
        ("shopping_qwen_n249", "shopping", "qwen3-32b"),
        ("shopping_mistral_n250", "shopping", "mistral-24b"),
        ("fhir_qwen_n973", "fhir", "qwen3-32b"),
        ("fhir_mistral_n847", "fhir", "mistral-24b"),
    ]
    rows: list[dict[str, Any]] = []
    for key, domain, model in cells:
        path = FAILURE_DETECTION / key / "summary_absolute.json"
        if not path.is_file():
            continue
        summ = json.loads(path.read_text(encoding="utf-8"))
        txb = summ.get("task_correctness_x_binding") or {}
        denom = int(summ.get("paper_denominator_n") or summ.get("num_trajectories") or 0)
        for cell_key, count in txb.items():
            rows.append(
                {
                    "cohort_key": key,
                    "domain": domain,
                    "model": model,
                    "paper_denominator_n": denom,
                    "cell_key": cell_key,
                    "count": int(count),
                    "rate": (int(count) / denom) if denom else None,
                }
            )
    _write_csv(out_dir / "task_pm_heatmap.csv", rows)


def _export_fhir_primary_stratum(out_dir: Path) -> None:
    path = FAILURE_DETECTION / "fhir_qwen_n973" / "summary_absolute.json"
    summ = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for stratum, stats in (summ.get("by_primary_stratum") or {}).items():
        n = int(stats.get("n") or 0)
        rows.append(
            {
                "primary_stratum": stratum,
                "n_trajectories": n,
                "pm_count": int(stats.get("pm") or 0),
                "pm_rate": (int(stats.get("pm") or 0) / n) if n else 0.0,
                "task_correct_count": int(stats.get("task_ok") or 0),
                "cross_count": int(stats.get("cross") or 0),
                "projection_count": int(stats.get("proj") or 0),
                "anchored_hall_count": int(stats.get("hall") or 0),
            }
        )
    _write_csv(out_dir / "fhir_primary_stratum.csv", rows)

    dist = (summ.get("claims_per_trajectory") or {}).get("distribution") or {}
    dist_rows = [{"n_claims": int(k), "n_trajectories": int(v)} for k, v in dist.items()]
    _write_csv(out_dir / "fhir_claims_per_trajectory_distribution.csv", dist_rows)


def _export_oc_margin_plot(out_dir: Path, probe_dir: Path) -> None:
    csv_path = probe_dir / "oc_bcp_stage_table.csv"
    df = pd.read_csv(csv_path)
    df.to_csv(out_dir / "oc_margin_plot.csv", index=False)
    summary = {
        "n_rows": len(df),
        "label_counts": df["label"].value_counts().to_dict() if "label" in df.columns else {},
        "margin_evidence_only": {
            "mean": float(df["margin_evidence_only"].mean()),
            "std": float(df["margin_evidence_only"].std()),
            "min": float(df["margin_evidence_only"].min()),
            "max": float(df["margin_evidence_only"].max()),
        },
    }
    if "label" in df.columns:
        for lab in df["label"].unique():
            sub = df[df["label"] == lab]["margin_evidence_only"]
            summary[f"margin_evidence_only_{lab}"] = {
                "mean": float(sub.mean()),
                "std": float(sub.std()),
                "n": int(len(sub)),
            }
    write_json(out_dir / "oc_margin_plot_summary.json", summary)


def _export_tau_curves(
    probe_dir: Path,
    out_dir: Path,
    df: pd.DataFrame,
    oc_margins: dict[str, float],
) -> None:
    taus = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    rows: list[dict[str, Any]] = []

    gating = json.loads((probe_dir / "mitigation_gating_curve.json").read_text())
    for split_name, curve in (gating.get("curves_by_split") or {}).items():
        base_rate = float(curve.get("baseline_pm_rate") or 0)
        for entry in curve.get("tau_sweep") or []:
            rows.append(
                {
                    "method": "BCP_single_head_y_pm",
                    "split": split_name,
                    "tau": entry["tau"],
                    "pm_rate": entry["pm_rate"],
                    "pm_count": entry["pm_count"],
                    "pm_reduction": entry.get("pm_reduction", base_rate - float(entry["pm_rate"])),
                    "claims_retained_mean": entry["claims_retained_mean"],
                    "task_correct_pm_rate": entry.get("task_correct_pm_rate"),
                }
            )

    for split in ("val", "test"):
        rows.extend(_msps_tau_sweep(df, oc_margins, taus=taus, split=split))

    _write_csv(out_dir / "mitigation_tau_pm_curve.csv", rows)
    write_json(out_dir / "mitigation_tau_pm_curve.json", {"taus": taus, "rows": rows})


def _compact_claim(c: dict, domain: str) -> dict[str, Any]:
    anchor = c.get("anchor_pid") or c.get("anchor_resource_id")
    return {
        "claim": c.get("claim"),
        "slot": c.get("slot"),
        "value": c.get("value"),
        "response_quote": c.get("response_quote"),
        "anchor_evidence_state": c.get("anchor_evidence_state"),
        "anchor_evidence": (c.get("supporting_anchor_evidence") or "")[:800],
        "other_resource_ids": c.get("supporting_other_resource_ids")
        or c.get("supporting_other_pids"),
        "cross_supported": c.get("cross_supported"),
        "query_supported": c.get("query_supported"),
        "rationale": (c.get("rationale") or "")[:1200],
        "anchor_id": anchor,
        "pm_labels": {
            "cross_object_merge": bool(c.get("cross_supported")),
            "constraint_projection": c.get("anchor_evidence_state") == "absent"
            and bool(c.get("query_supported")),
        },
    }


def _load_case_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    per = data.get("pm_vnext_per_trajectory") or {}
    claims = per.get("claims_extracted") or []
    traj_labels = per.get("trajectory_labels") or {}
    killer = None
    for c in claims:
        if c.get("cross_supported") or c.get("anchor_evidence_state") == "absent":
            killer = _compact_claim(c, data.get("domain", ""))
            break
    if killer is None and claims:
        killer = _compact_claim(claims[0], data.get("domain", ""))

    return {
        "case_id": data.get("case_id"),
        "domain": data.get("domain"),
        "paper_failure_type": data.get("paper_failure_type"),
        "question_id": per.get("question_id") or data.get("question_id"),
        "orig_index": per.get("orig_index") or data.get("orig_index"),
        "query": per.get("query"),
        "anchor_ids": per.get("selected_anchor_pids")
        or per.get("selected_anchor_resource_ids"),
        "task_correct": per.get("task_correct_exact")
        if per.get("task_correct_exact") is not None
        else per.get("task_correct_llm"),
        "final_answer": per.get("final_answer"),
        "trajectory_has_phantom_merge": traj_labels.get("has_phantom_merge"),
        "trajectory_labels": traj_labels,
        "killer_claim": killer,
        "all_claims": [_compact_claim(c, data.get("domain", "")) for c in claims],
        "trajectory_json": str(path.relative_to(REPO)),
    }


def _pick_mitigation_cases(probe_dir: Path) -> list[dict[str, Any]]:
    pgcs_path = PGCS_DIR_DEFAULT / "pgcs_onepass_per_case.jsonl"
    cases: list[dict] = []
    if not pgcs_path.is_file():
        return cases

    records = []
    for line in pgcs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    # MSPS filter-only still PM on frozen labels (~7/146 = MSPS residual)
    still_pm_after_filter = [
        r
        for r in records
        if r.get("baseline_has_phantom_merge") and r.get("label_pm_after_filter_only")
    ]
    # PGCS full pipeline cleared PM (filter + rewrite + judge)
    pgcs_full_success = [
        r
        for r in records
        if r.get("baseline_has_phantom_merge")
        and not r.get("label_pm_after_filter_only")
        and not r.get("pgcs_has_phantom_merge")
    ]
    # Rewrite rebound: filter cleared on labels, judge reintroduces PM
    rewrite_rebound = [
        r
        for r in records
        if r.get("baseline_has_phantom_merge")
        and not r.get("label_pm_after_filter_only")
        and r.get("pgcs_has_phantom_merge")
        and int(r.get("n_rewritten") or 0) > 0
    ]

    def pack(rec: dict, role: str) -> dict[str, Any]:
        qid = rec["question_id"]
        claim_rows = []
        cp = probe_dir / "claims.parquet"
        if cp.is_file():
            df = pd.read_parquet(cp)
            sub = df[(df["group_id"].astype(str) == qid) & (df["split"].astype(str) == "test")]
            for _, row in sub.iterrows():
                sl = row.get("support_labels")
                if sl is not None and not isinstance(sl, list):
                    sl = list(sl)
                claim_rows.append(
                    {
                        "claim_id": row["claim_id"],
                        "claim_text": row["claim_text"],
                        "primary_support_label": row["primary_support_label"],
                        "y_pm": int(row["y_pm"]),
                        "support_labels": sl,
                    }
                )
        return {
            "role": role,
            "question_id": qid,
            "baseline_has_phantom_merge": rec.get("baseline_has_phantom_merge"),
            "label_pm_after_filter_only": rec.get("label_pm_after_filter_only"),
            "pgcs_has_phantom_merge": rec.get("pgcs_has_phantom_merge"),
            "n_dropped": rec.get("n_dropped"),
            "n_kept": rec.get("n_kept"),
            "n_rewritten": rec.get("n_rewritten"),
            "claim_retention_rate": rec.get("claim_retention_rate"),
            "pgcs_n_claims": rec.get("pgcs_n_claims"),
            "dropped_claim_ids": rec.get("dropped_claim_ids"),
            "rewritten_claim_ids": rec.get("rewritten_claim_ids"),
            "pgcs_answer_preview": (rec.get("pgcs_answer_preview") or "")[:500],
            "baseline_claims": claim_rows,
        }

    if still_pm_after_filter:
        cases.append(pack(still_pm_after_filter[0], "mitigation_msps_still_pm_after_filter"))
    if rewrite_rebound:
        cases.append(pack(rewrite_rebound[0], "mitigation_pgcs_rewrite_rebound"))
    if pgcs_full_success:
        cases.append(pack(pgcs_full_success[0], "mitigation_pgcs_full_success"))
    return cases


def _export_case_studies(out_dir: Path, probe_dir: Path) -> None:
    bundles = [
        ("shopping_CAP", CASE_TRAJ_DIR / "S-CP-star1_shopping_orig10.json", "CAP"),
        ("fhir_CEM", CASE_TRAJ_DIR / "F-COM-star1_fhir_03d2b60423371ca98d1a99bc.json", "CEM"),
        ("shopping_ASF", CASE_TRAJ_DIR / "S-AH-star1_shopping_orig13.json", "ASF"),
    ]
    cases: list[dict[str, Any]] = []
    for role, path, ptype in bundles:
        if not path.is_file():
            continue
        c = _load_case_bundle(path)
        c["role"] = role
        c["paper_failure_type"] = ptype
        cases.append(c)

    cases.extend(_pick_mitigation_cases(probe_dir))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "terminology": {
            "CAP": "constraint_projection",
            "CEM": "cross_object_merge",
            "ASF": "anchored_hallucination",
        },
        "cases": cases,
    }
    write_json(out_dir / "case_studies.json", out)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "results" / "appendix",
    )
    ap.add_argument("--layer-slot", type=int, default=1)
    ap.add_argument("--random-seeds", type=int, default=200)
    ap.add_argument(
        "--include-case-studies",
        action="store_true",
        help="Also export case_studies.json (disabled in anonymous release).",
    )
    args = ap.parse_args()

    probe_dir = args.probe_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MSPS probe probabilities (3 heads)...", flush=True)
    df = _load_msps_probs(probe_dir, layer_slot=args.layer_slot)
    print("Loading official BCP y_pm probabilities...", flush=True)
    df_bcp = _load_bcp_probs(probe_dir)
    oc_margins = _load_oc_margins(probe_dir)

    msps_path = probe_dir / "mitigation_ultimate_v1/msps_ultimate_results.json"
    if msps_path.is_file():
        msps_cfg_dict = json.loads(msps_path.read_text()).get("val_selected_config") or {}
    else:
        msps_cfg_dict = {"tau_max": 0.45, "tau_com": 0.45, "tau_cp": 0.45, "oc_margin_cut": 0.0, "use_oc_gate": False}
    msps_cfg = MSPSConfig(
        float(msps_cfg_dict.get("tau_max", 0.45)),
        float(msps_cfg_dict.get("tau_com", 0.45)),
        float(msps_cfg_dict.get("tau_cp", 0.45)),
        float(msps_cfg_dict.get("oc_margin_cut", 0.0)),
        bool(msps_cfg_dict.get("use_oc_gate", False)),
    )

    test = df[df["split"].astype(str) == "test"]
    print("Random retention-matched control...", flush=True)
    target_kept = kept_counts_per_group(test, _msps_drop_fn(msps_cfg, oc_margins))
    random_stats = random_retention_matched_eval(
        test, target_kept, n_seeds=args.random_seeds
    )
    write_json(out_dir / "mitigation_random_retention_matched.json", random_stats)

    print("Building extended mitigation table...", flush=True)
    ext_rows = _build_mitigation_extended_table(
        probe_dir, df, df_bcp, oc_margins, msps_cfg, random_stats
    )
    _write_csv(out_dir / "mitigation_table_extended.csv", ext_rows)
    write_json(out_dir / "mitigation_table_extended.json", ext_rows)

    print("Exporting τ curves...", flush=True)
    _export_tau_curves(probe_dir, out_dir, df, oc_margins)

    print("Exporting failure-detection CSVs...", flush=True)
    _export_task_pm_heatmap(out_dir)
    _export_fhir_primary_stratum(out_dir)
    _export_oc_margin_plot(out_dir, probe_dir)

    if args.include_case_studies:
        print("Exporting case studies...", flush=True)
        _export_case_studies(out_dir, probe_dir)

    # CB audit detail JSON
    cb_detail = {
        "msps_tau0.45": audit_gating(test, _msps_drop_fn(msps_cfg, oc_margins)).to_dict(),
        "bcp_tau0.5": audit_gating(
            df_bcp[df_bcp["split"].astype(str) == "test"], _bcp_drop_fn(0.5)
        ).to_dict(),
        "oracle": audit_gating(test, lambda r: is_pm_claim(r)).to_dict(),
        "random_retention_matched": random_stats,
    }
    write_json(out_dir / "mitigation_cb_retention_audit.json", cb_detail)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_dir": str(probe_dir),
        "out_dir": str(out_dir),
        "files": [
            "mitigation_table_extended.csv",
            "mitigation_table_extended.json",
            "mitigation_cb_retention_audit.json",
            "mitigation_random_retention_matched.json",
            "mitigation_tau_pm_curve.csv",
            "mitigation_tau_pm_curve.json",
            "task_pm_heatmap.csv",
            "fhir_primary_stratum.csv",
            "fhir_claims_per_trajectory_distribution.csv",
            "oc_margin_plot.csv",
            "oc_margin_plot_summary.json",
            "manifest.json",
        ],
    }
    if args.include_case_studies:
        manifest["files"].insert(-1, "case_studies.json")
    manifest["cb_retention_summary"] = {
        "MSPS": cb_detail["msps_tau0.45"],
        "BCP": cb_detail["bcp_tau0.5"],
    }
    write_json(out_dir / "manifest.json", manifest)

    # Refresh p4 summary with CB fields (engineering + docs copy)
    p4_paths = [
        probe_dir / "p4_mitigation_research_summary.json",
        REPO / "results" / "probe_fhir" / "p4_mitigation_research_summary.json",
    ]
    export_meta = {
        "dir": str(out_dir.relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir),
        "cb_retention_audit": cb_detail,
        "random_retention_matched": random_stats,
    }
    for p4_path in p4_paths:
        if p4_path.is_file():
            p4 = json.loads(p4_path.read_text(encoding="utf-8"))
            p4["appendix_tables"] = export_meta
            p4_path.write_text(json.dumps(p4, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
