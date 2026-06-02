#!/usr/bin/env python3
"""
Probe-in-the-loop after anchor-only regen:
  score mitigated claims with frozen BCP → τ gating → trajectory PM (standard labels).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from eval_mitigation_gating import run_gating  # noqa: E402
from lib.mitigation import gating_curve, pick_tau_balanced, trajectory_has_pm_rows  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402


def _load_mitigated_matrix(bank: Path, layer_slot: int) -> np.ndarray:
    mean = np.load(bank / "bcp_full_mean.npy")
    return mean[:, layer_slot, :]


def run_regen_gating(
    probe_dir: Path,
    mitigated_parquet: Path,
    mitigated_bank: Path,
    *,
    per_case_jsonl: Path | None = None,
    taus: list[float] | None = None,
) -> dict:
    taus = taus or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pack = joblib.load(probe_dir / "probe_bcp_detect.joblib")
    clf = pack["clf"]
    layer_slot = int(pack.get("layer_slot", 2))

    df = pd.read_parquet(mitigated_parquet)
    H = _load_mitigated_matrix(mitigated_bank, layer_slot)
    if H.shape[0] != len(df):
        raise ValueError(f"hidden {H.shape[0]} != claims {len(df)}")

    prob = clf.predict_proba(H)[:, 1]
    df = df.copy()
    df["p_pm"] = prob

    # τ from baseline val (paper rule)
    base_gating = json.loads((probe_dir / "mitigation_gating_curve.json").read_text())
    tau_balanced = float(
        (base_gating.get("tau_selection") or {}).get("balanced_tau_paper") or 0.5
    )

    curves: dict[str, dict] = {}
    for split in sorted(set(df["split"].astype(str))):
        mask = df["split"].astype(str).values == split
        if not mask.any():
            continue
        sub = df.loc[mask].copy()
        sub["p_pm"] = prob[mask]
        curves[split] = gating_curve(sub, prob[mask], split_name=split, taus=taus)

    def _traj_flags(grp: pd.DataFrame, tau: float) -> tuple[int, int]:
        rows = grp.to_dict("records")
        for r in rows:
            sl = r.get("support_labels")
            if sl is not None and not isinstance(sl, list):
                r["support_labels"] = list(sl)
        base = int(trajectory_has_pm_rows(rows))
        kept = [r for r in rows if float(r.get("p_pm", 1)) <= tau]
        gated = int(trajectory_has_pm_rows(kept))
        return base, gated

    traj_rows: list[dict] = []
    for gid, grp in df.groupby("group_id"):
        b0, g0 = _traj_flags(grp, tau_balanced)
        traj_rows.append(
            {
                "question_id": str(gid),
                "split": str(grp["split"].iloc[0]),
                "n_claims": len(grp),
                "mitigated_judge_pm": b0,
                "probe_gated_pm": g0,
                "probe_cleared": b0 and not g0,
            }
        )

    test_traj = [t for t in traj_rows if t["split"] == "test"]
    n_test = len(test_traj) or 1
    hybrid = {
        "tau": tau_balanced,
        "tau_source": "baseline_val_pick_tau_balanced",
        "n_test_trajectories": n_test,
        "mitigated_judge_pm_rate": sum(t["mitigated_judge_pm"] for t in test_traj) / n_test,
        "after_probe_gating_pm_rate": sum(t["probe_gated_pm"] for t in test_traj) / n_test,
        "probe_cleared_count": sum(t["probe_cleared"] for t in test_traj),
        "mitigated_judge_pm_count": sum(t["mitigated_judge_pm"] for t in test_traj),
        "after_probe_gating_pm_count": sum(t["probe_gated_pm"] for t in test_traj),
    }

    if per_case_jsonl and per_case_jsonl.is_file():
        regen_pm = []
        for line in per_case_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("probe_split") == "test" or r.get("probe_split") == "":
                    if str(r.get("probe_split", "")) == "test":
                        regen_pm.append(bool(r.get("mitigated_has_phantom_merge")))
        if regen_pm:
            hybrid["anchor_regen_pm_rate"] = sum(regen_pm) / len(regen_pm)
            hybrid["anchor_regen_pm_count"] = sum(regen_pm)
            hybrid["n_anchor_regen_test"] = len(regen_pm)

    out = {
        "description": "Mitigated-answer claims: VNEXT judge labels + frozen BCP p_pm gating",
        "mitigated_parquet": str(mitigated_parquet),
        "mitigated_bank": str(mitigated_bank),
        "probe": str(probe_dir / "probe_bcp_detect.joblib"),
        "curves_by_split": curves,
        "hybrid_test": hybrid,
        "per_trajectory": traj_rows,
    }
    out_path = mitigated_parquet.parent / "mitigation_regen_gating.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--mitigated-parquet", type=Path, required=True)
    ap.add_argument("--mitigated-bank", type=Path, default=None)
    ap.add_argument("--per-case-jsonl", type=Path, default=None)
    args = ap.parse_args()
    bank = args.mitigated_bank or (args.mitigated_parquet.parent / "feature_bank_mitigated")
    out = run_regen_gating(
        args.probe_dir,
        args.mitigated_parquet,
        bank,
        per_case_jsonl=args.per_case_jsonl,
    )
    print(json.dumps(out.get("hybrid_test") or {}, indent=2))


if __name__ == "__main__":
    main()
