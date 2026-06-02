#!/usr/bin/env python3
"""P4.0 oracle upper bound: keep only non-PM claims, recompute trajectory PM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.mitigation import PM_FAILURE, trajectory_has_pm_rows  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402


def _records(grp: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in grp.to_dict("records"):
        sl = rec.get("support_labels")
        if sl is not None and not isinstance(sl, list):
            rec["support_labels"] = list(sl)
        rows.append(rec)
    return rows


def _oracle_kept(rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        labels = set(row.get("support_labels") or [])
        if labels & PM_FAILURE:
            continue
        if int(row.get("y_pm", 0) or 0) == 1:
            continue
        kept.append(row)
    return kept


def oracle_curve(df: pd.DataFrame, *, split_name: str) -> dict:
    baseline_pm: list[int] = []
    oracle_pm: list[int] = []
    claims_kept: list[int] = []

    for _, grp in df.groupby("group_id"):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        kept = _oracle_kept(rows)
        claims_kept.append(len(kept))
        oracle_pm.append(int(trajectory_has_pm_rows(kept)))

    n = len(baseline_pm) or 1
    base_rate = sum(baseline_pm) / n
    ora_rate = sum(oracle_pm) / n
    return {
        "split": split_name,
        "n_trajectories": n,
        "baseline_pm_rate": base_rate,
        "baseline_pm_count": sum(baseline_pm),
        "oracle_pm_rate": ora_rate,
        "oracle_pm_count": sum(oracle_pm),
        "pm_reduction": base_rate - ora_rate,
        "claims_retained_mean": float(sum(claims_kept) / n) if claims_kept else 0.0,
    }


def run_oracle(probe_dir: Path) -> dict:
    df = pd.read_parquet(probe_dir / "claims.parquet")
    by_split = {}
    for split in sorted(set(df["split"].astype(str))):
        sub = df[df["split"].astype(str) == split]
        if len(sub):
            by_split[split] = oracle_curve(sub, split_name=split)
    out = {
        "description": "Keep claims with no PM failure label; recompute has_phantom_merge",
        "splits": by_split,
    }
    path = probe_dir / "mitigation_oracle_upper_bound.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    args = ap.parse_args()
    out = run_oracle(args.probe_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
