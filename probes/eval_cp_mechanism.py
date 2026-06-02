#!/usr/bin/env python3
"""P3 supplement: CP vs COM decomposition + BCP no_other ablation (CP mechanism)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.paths import FHIR_PROBE_OUT, PROBE_RUNS  # noqa: E402


def run_cp_mechanism(
    probe_dir: Path,
    *,
    decomp_csv: Path | None = None,
    ablation_json: Path | None = None,
) -> dict:
    decomp_csv = decomp_csv or (PROBE_RUNS / "trajectory_decomposition.csv")
    decomp = pd.read_csv(decomp_csv)
    fhir = decomp[(decomp["domain"] == "fhir") & (decomp["model"] == "qwen3-32b")]
    n = len(fhir) or 1

    traj_rates = {
        "n_trajectories": int(n),
        "has_phantom_merge": float(fhir["has_phantom_merge"].mean()),
        "has_com_only": float(fhir["has_com_only"].mean()),
        "has_cp_only": float(fhir["has_cp_only"].mean()),
        "has_com_and_cp": float(fhir["has_com_and_cp"].mean()),
        "has_anchored_hall_only": float(fhir["has_anchored_hall_only"].mean()),
    }

    claims = pd.read_parquet(probe_dir / "claims.parquet")
    claim_rates = {
        "n_claims": int(len(claims)),
        "y_pm_rate": float(claims["y_pm"].mean()),
        "y_com_rate": float(claims["y_com"].mean()),
        "y_cp_rate": float(claims["y_cp"].mean()),
        "both_com_cp_rate": float(((claims["y_com"] == 1) & (claims["y_cp"] == 1)).mean()),
    }

    abl_path = ablation_json or (probe_dir / "ablation_input.json")
    bcp_ablation = {}
    if abl_path.is_file():
        abl = json.loads(abl_path.read_text(encoding="utf-8"))
        by_var = {v["variant"]: v for v in abl.get("variants", [])}
        full_t = (by_var.get("full") or {}).get("test_auroc")
        no_other_t = (by_var.get("no_other") or {}).get("test_auroc")
        bcp_ablation = {
            "full_test_auroc_y_pm": full_t,
            "no_other_test_auroc_y_pm": no_other_t,
            "drop_no_other": (full_t - no_other_t) if full_t and no_other_t else None,
            "interpretation": "CP mechanism: removing Other Evidence reduces y_pm detection (query/anchor-only insufficient).",
        }

    out = {
        "trajectory_decomposition": traj_rates,
        "claim_labels": claim_rates,
        "bcp_input_ablation_y_pm": bcp_ablation,
    }
    out_path = probe_dir / "cp_com_mechanism.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--decomp-csv", type=Path, default=PROBE_RUNS / "trajectory_decomposition.csv")
    ap.add_argument("--ablation-json", type=Path, default=None)
    args = ap.parse_args()
    out = run_cp_mechanism(args.probe_dir, decomp_csv=args.decomp_csv, ablation_json=args.ablation_json)
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.probe_dir / 'cp_com_mechanism.json'}")


if __name__ == "__main__":
    main()
