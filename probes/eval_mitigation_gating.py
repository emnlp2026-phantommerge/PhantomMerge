#!/usr/bin/env python3
"""P4 auxiliary: BCP τ gating — val τ pick + test report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.feature_bank import bank_dir, load_bcp_matrix, load_manifest  # noqa: E402
from lib.mitigation import gating_curve, pick_tau_balanced, pick_tau_on_val  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402


def run_gating(
    probe_dir: Path,
    *,
    taus: list[float],
    pick_split: str = "val",
    report_split: str = "test",
) -> dict:
    probe_path = probe_dir / "probe_bcp_detect.joblib"
    if not probe_path.is_file():
        raise FileNotFoundError(f"Train probe first: {probe_path}")
    pack = joblib.load(probe_path)
    clf = pack["clf"]
    layer_slot = int(pack.get("layer_slot", 2))
    variant = str(pack.get("variant", "full"))
    pool = str(pack.get("pool", "mean"))

    df = pd.read_parquet(probe_dir / "claims.parquet")
    bank = bank_dir(probe_dir)
    man = load_manifest(bank)
    if not man or int(man.get("completed_rows", 0)) < len(df):
        raise FileNotFoundError("Incomplete feature_bank; run extract_feature_bank.py")
    H = load_bcp_matrix(bank, variant=variant, pool=pool, layer_slot=layer_slot)
    if H.shape[0] != len(df):
        raise ValueError(f"hidden rows {H.shape[0]} != claims {len(df)}")

    prob = clf.predict_proba(H)[:, 1]
    curves: dict[str, dict] = {}
    for split in sorted(set(df["split"].astype(str))):
        mask = df["split"].astype(str).values == split
        if not mask.any():
            continue
        curves[split] = gating_curve(df.loc[mask].copy(), prob[mask], split_name=split, taus=taus)

    val_curve = curves.get(pick_split, {})
    tau_legacy = pick_tau_on_val(val_curve)
    tau_balanced = pick_tau_balanced(val_curve)
    test_curve = curves.get(report_split, {})

    def _entry(curve: dict, tau: float | None) -> dict | None:
        if tau is None:
            return None
        for e in curve.get("tau_sweep", []):
            if float(e["tau"]) == float(tau):
                return e
        return None

    test_at_balanced = _entry(test_curve, tau_balanced)
    test_at_legacy = _entry(test_curve, tau_legacy)
    fixed_test = {
        str(t): _entry(test_curve, t)
        for t in (0.5, 0.6, 0.7)
    }

    out = {
        "probe": str(probe_path),
        "variant": variant,
        "pool": pool,
        "layer_slot": layer_slot,
        "taus": taus,
        "curves_by_split": curves,
        "tau_selection": {
            "val_split": pick_split,
            "test_split": report_split,
            "legacy_smallest_tau": tau_legacy,
            "balanced_tau_paper": tau_balanced,
            "balanced_rule": "val τ∈[0.5,0.7], pm_reduction≥0.10, claims_retained≥1.5",
        },
        "test_at_balanced_tau": test_at_balanced,
        "test_at_legacy_tau": test_at_legacy,
        "test_at_fixed_taus": fixed_test,
    }
    (probe_dir / "mitigation_gating_curve.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--taus", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--pick-split", type=str, default="val")
    ap.add_argument("--report-split", type=str, default="test")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    out = run_gating(
        args.probe_dir,
        taus=taus,
        pick_split=args.pick_split,
        report_split=args.report_split,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
