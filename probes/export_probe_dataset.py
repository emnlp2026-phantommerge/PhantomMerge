#!/usr/bin/env python3
"""Export claim-level probe dataset (parquet + splits.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.paths import (  # noqa: E402
    FHIR_PROBE_OUT,
    FHIR_QWEN_VNEXT,
    SHOPPING_PROBE_OUT,
    SHOPPING_QWEN_VNEXT,
)
from lib.per_trajectory import export_claim_rows, read_jsonl  # noqa: E402
from lib.splits import assign_split, grouped_split, write_splits  # noqa: E402


def export_domain(
    per_path: Path,
    *,
    domain: str,
    model: str,
    out_dir: Path,
    split_scope: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for traj in read_jsonl(per_path):
        rows.extend(export_claim_rows(traj, domain=domain, model=model))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No claims exported from {per_path}")

    if split_scope == "grouped":
        split_map = grouped_split(df["group_id"].astype(str).tolist(), seed=42)
        df["split"] = df["group_id"].astype(str).map(lambda g: assign_split(g, split_map))
        split_payload = {
            "seed": 42,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_by": "group_id",
            "groups": split_map,
            "n_claims": int(len(df)),
            "n_groups": int(df["group_id"].nunique()),
            "y_pm_rate": float(df["y_pm"].mean()),
            "y_com_rate": float(df["y_com"].mean()),
            "oc_eligible": int(df["oc_eligible"].sum()),
        }
    else:
        df["split"] = split_scope
        split_payload = {"split_scope": split_scope, "n_claims": int(len(df))}

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "claims.parquet"
    df.to_parquet(parquet_path, index=False)
    write_splits(out_dir / "splits.json", split_payload)

    stats = {
        "domain": domain,
        "model": model,
        "per_path": str(per_path),
        "n_claims": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "y_pm": int(df["y_pm"].sum()),
        "y_com": int(df["y_com"].sum()),
        "y_cp": int(df["y_cp"].sum()),
        "oc_eligible": int(df["oc_eligible"].sum()),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (out_dir / "export_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["fhir", "shopping", "both"], default="both")
    ap.add_argument("--fhir-per", type=Path, default=FHIR_QWEN_VNEXT)
    ap.add_argument("--shopping-per", type=Path, default=SHOPPING_QWEN_VNEXT)
    ap.add_argument("--fhir-out", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--shopping-out", type=Path, default=SHOPPING_PROBE_OUT)
    args = ap.parse_args()

    if args.domain in ("fhir", "both"):
        export_domain(
            args.fhir_per,
            domain="fhir",
            model="qwen3-32b",
            out_dir=args.fhir_out,
            split_scope="grouped",
        )
    if args.domain in ("shopping", "both"):
        export_domain(
            args.shopping_per,
            domain="shopping",
            model="qwen3-32b",
            out_dir=args.shopping_out,
            split_scope="ood",
        )


if __name__ == "__main__":
    main()
