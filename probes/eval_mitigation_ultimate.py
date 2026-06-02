#!/usr/bin/env python3
"""
P4 Ultimate — Multi-Signal Probe Stack (MSPS).

Best honest probe mitigation under fixed VNEXT PM definition:
  - Layer L2 (best y_pm AUROC in layer sweep)
  - Three BCP heads: y_pm, y_com, y_cp (same hidden, val-tuned)
  - OC margin gate for oc_eligible COM claims (mechanism-aligned)
  - Val grid → test report on full 146 trajectories + pm_test 52 subset
  - Regen claims path (mitigated_answer) with same stack

Does NOT modify judge, labels, or prior result directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.mitigation import PM_FAILURE, gating_curve, trajectory_has_pm_rows  # noqa: E402
from lib.p2_eval import train_bcp_hidden  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402


@dataclass(frozen=True)
class MSPSConfig:
    tau_max: float
    tau_com: float
    tau_cp: float
    oc_margin_cut: float
    use_oc_gate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_max": self.tau_max,
            "tau_com": self.tau_com,
            "tau_cp": self.tau_cp,
            "oc_margin_cut": self.oc_margin_cut,
            "use_oc_gate": self.use_oc_gate,
        }


def _load_oc_margins(probe_dir: Path) -> dict[str, float]:
    csv_path = probe_dir / "oc_bcp_stage_table.csv"
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        cid = str(row.get("claim_id", ""))
        m = row.get("margin_evidence_only")
        if cid and m == m:
            out[cid] = float(m)
    return out


def _records(grp: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in grp.to_dict("records"):
        sl = rec.get("support_labels")
        if sl is not None and not isinstance(sl, list):
            rec["support_labels"] = list(sl)
        rows.append(rec)
    return rows


def _should_drop(row: dict, cfg: MSPSConfig, oc_margins: dict[str, float]) -> bool:
    p_pm = float(row.get("p_pm", 0))
    p_com = float(row.get("p_com", 0))
    p_cp = float(row.get("p_cp", 0))
    p_max = float(row.get("p_max", max(p_pm, p_com, p_cp)))

    if p_pm > cfg.tau_max or p_com > cfg.tau_com or p_cp > cfg.tau_cp:
        return True

    if cfg.use_oc_gate and int(row.get("oc_eligible", 0) or 0) == 1:
        cid = str(row.get("claim_id", ""))
        margin = oc_margins.get(cid)
        if margin is not None and margin > cfg.oc_margin_cut:
            return True
    return False


def _trajectory_eval(
    df: pd.DataFrame,
    cfg: MSPSConfig,
    oc_margins: dict[str, float],
) -> dict[str, Any]:
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    pm_claims_base: list[int] = []
    pm_claims_gated: list[int] = []

    for _, grp in df.groupby("group_id"):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        pm_claims_base.append(
            sum(1 for r in rows if int(r.get("y_pm", 0)) == 1 or _support_pm(r))
        )
        kept = [r for r in rows if not _should_drop(r, cfg, oc_margins)]
        claims_kept.append(len(kept))
        gated_pm.append(int(trajectory_has_pm_rows(kept)))
        pm_claims_gated.append(
            sum(1 for r in kept if int(r.get("y_pm", 0)) == 1 or _support_pm(r))
        )

    n = len(baseline_pm) or 1
    return {
        "n_trajectories": n,
        "baseline_pm_count": sum(baseline_pm),
        "baseline_pm_rate": sum(baseline_pm) / n,
        "gated_pm_count": sum(gated_pm),
        "gated_pm_rate": sum(gated_pm) / n,
        "pm_reduction": sum(baseline_pm) / n - sum(gated_pm) / n,
        "claims_retained_mean": float(np.mean(claims_kept)) if claims_kept else 0.0,
        "mean_pm_claims_baseline": float(np.mean(pm_claims_base)) if pm_claims_base else 0.0,
        "mean_pm_claims_gated": float(np.mean(pm_claims_gated)) if pm_claims_gated else 0.0,
        "pm_claims_total_baseline": sum(pm_claims_base),
        "pm_claims_total_gated": sum(pm_claims_gated),
    }


def _support_pm(row: dict) -> bool:
    labels = set(row.get("support_labels") or [])
    return bool(labels & PM_FAILURE)


def _attach_probs(
    df: pd.DataFrame,
    prob_pm: np.ndarray,
    prob_com: np.ndarray,
    prob_cp: np.ndarray,
) -> pd.DataFrame:
    out = df.copy()
    out["p_pm"] = prob_pm
    out["p_com"] = prob_com
    out["p_cp"] = prob_cp
    out["p_max"] = np.maximum.reduce([prob_pm, prob_com, prob_cp])
    return out


def _val_search(
    df: pd.DataFrame,
    oc_margins: dict[str, float],
) -> tuple[MSPSConfig, dict[str, Any]]:
    val = df[df["split"].astype(str) == "val"]
    taus = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    oc_cuts = [0.0, 0.05, 0.10, 0.15]
    best_cfg: MSPSConfig | None = None
    best_score = -1.0
    best_eval: dict[str, Any] = {}

    for tau in taus:
        for oc_cut in oc_cuts:
            for use_oc in (False, True):
                cfg = MSPSConfig(
                    tau_max=tau,
                    tau_com=tau,
                    tau_cp=tau,
                    oc_margin_cut=oc_cut,
                    use_oc_gate=use_oc,
                )
                ev = _trajectory_eval(val, cfg, oc_margins)
                # score: PM reduction, penalize over-deletion
                if ev["claims_retained_mean"] < 1.2:
                    continue
                score = ev["pm_reduction"] - 0.02 * max(0.0, 1.5 - ev["claims_retained_mean"])
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
                    best_eval = ev

    if best_cfg is None:
        best_cfg = MSPSConfig(0.5, 0.5, 0.5, 0.0, True)
        best_eval = _trajectory_eval(val, best_cfg, oc_margins)
    return best_cfg, best_eval


def _load_mitigated_matrix(bank: Path, layer_slot: int) -> np.ndarray:
    mean = np.load(bank / "bcp_full_mean.npy")
    return mean[:, layer_slot, :]


def run_msps(
    probe_dir: Path,
    out_dir: Path,
    *,
    layer_slot: int = 1,
    mitigated_parquet: Path | None = None,
    mitigated_bank: Path | None = None,
    pm_test_qids: set[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    oc_margins = _load_oc_margins(probe_dir)
    df_base = pd.read_parquet(probe_dir / "claims.parquet")
    heads: dict[str, Any] = {}
    prob_by_label: dict[str, np.ndarray] = {}
    clfs: dict[str, Any] = {}
    for label in ("y_pm", "y_com", "y_cp"):
        clf, metrics, H, df = train_bcp_hidden(
            probe_dir, label=label, variant="full", pool="mean", layer_slot=layer_slot
        )
        clfs[label] = clf
        heads[label] = {"auroc_test": metrics["splits"]["test"].get("auroc"), "layer": metrics["layer_name"]}
        prob_by_label[label] = clf.predict_proba(H)[:, 1]

    df = _attach_probs(df_base, prob_by_label["y_pm"], prob_by_label["y_com"], prob_by_label["y_cp"])
    cfg, val_eval = _val_search(df, oc_margins)

    test_df = df[df["split"].astype(str) == "test"]
    test_eval = _trajectory_eval(test_df, cfg, oc_margins)

    subset_eval = None
    if pm_test_qids:
        sub = test_df[test_df["group_id"].astype(str).isin(pm_test_qids)]
        if len(sub):
            subset_eval = _trajectory_eval(sub, cfg, oc_margins)

    # legacy single-head reference (read-only compare)
    legacy_ref = None
    legacy_path = probe_dir / "p4_mitigation_research_summary.json"
    if legacy_path.is_file():
        legacy_ref = json.loads(legacy_path.read_text()).get("baseline_bcp_gating_test")

    regen_eval = None
    if mitigated_parquet and mitigated_parquet.is_file() and mitigated_bank:
        mdf = pd.read_parquet(mitigated_parquet)
        Hm = _load_mitigated_matrix(mitigated_bank, layer_slot)
        if Hm.shape[0] == len(mdf):
            mpm = clfs["y_pm"].predict_proba(Hm)[:, 1]
            mco = clfs["y_com"].predict_proba(Hm)[:, 1]
            mcp = clfs["y_cp"].predict_proba(Hm)[:, 1]
            mdf = _attach_probs(mdf, mpm, mco, mcp)
            regen_eval = _trajectory_eval(mdf, cfg, {})

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Multi-Signal Probe Stack (MSPS)",
        "protocol": {
            "pm_definition": "unchanged VNEXT (cross|projection|anchored_hall)",
            "layer_slot": layer_slot,
            "layer_name": "L2",
            "heads": ["y_pm", "y_com", "y_cp"],
            "oc_margin": "margin_evidence_only > cut for oc_eligible",
            "val_selection": "max PM reduction, claims_retained_mean >= 1.2",
            "prior_results_preserved": True,
        },
        "head_metrics_test_auroc": heads,
        "val_selected_config": cfg.to_dict(),
        "val_eval": val_eval,
        "test_full_146": test_eval,
        "test_pm_subset_52": subset_eval,
        "regen_pm_test_52": regen_eval,
        "legacy_single_head_y_pm_tau0.5_ref": legacy_ref,
        "oracle_test_ref": json.loads((probe_dir / "mitigation_oracle_upper_bound.json").read_text())
        .get("splits", {})
        .get("test"),
        "conclusion_template": (
            "MSPS is the pre-specified maximum probe stack under VNEXT; "
            "oracle remains upper bound; rollout-time intervention not included."
        ),
    }

    # Is this strictly better than legacy on test 146?
    legacy_pm = (legacy_ref or {}).get("pm_count")
    msps_pm = test_eval["gated_pm_count"]
    out["beats_legacy_single_head"] = (
        legacy_pm is not None and msps_pm < int(legacy_pm)
    )
    out["vs_legacy"] = {
        "legacy_pm_count_test146": legacy_pm,
        "msps_pm_count_test146": msps_pm,
        "delta_trajectories": (int(legacy_pm) - msps_pm) if legacy_pm is not None else None,
    }

    out_path = out_dir / "msps_ultimate_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FHIR_PROBE_OUT / "mitigation_ultimate_v1",
    )
    ap.add_argument("--layer-slot", type=int, default=1, help="1=L2 best y_pm AUROC")
    ap.add_argument(
        "--mitigated-parquet",
        type=Path,
        default=FHIR_PROBE_OUT / "mitigation_anchor_pm_test_standard/mitigated_claims.parquet",
    )
    ap.add_argument(
        "--mitigated-bank",
        type=Path,
        default=FHIR_PROBE_OUT
        / "mitigation_anchor_pm_test_standard/feature_bank_mitigated",
    )
    ap.add_argument(
        "--pm-test-jsonl",
        type=Path,
        default=FHIR_PROBE_OUT
        / "mitigation_anchor_pm_test_standard/mitigation_anchor_only_per_case.jsonl",
    )
    args = ap.parse_args()

    qids: set[str] = set()
    if args.pm_test_jsonl.is_file():
        for line in args.pm_test_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                qids.add(json.loads(line)["question_id"])

    out = run_msps(
        args.probe_dir,
        args.out_dir,
        layer_slot=args.layer_slot,
        mitigated_parquet=args.mitigated_parquet if args.mitigated_parquet.is_file() else None,
        mitigated_bank=args.mitigated_bank if args.mitigated_bank.is_dir() else None,
        pm_test_qids=qids or None,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
