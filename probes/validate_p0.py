#!/usr/bin/env python3
"""P0 acceptance: claims.parquet + splits + VNEXT paths (no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from lib.paths import FHIR_QWEN_VNEXT, FHIR_PROBE_OUT, SHOPPING_QWEN_VNEXT  # noqa: E402
from lib.per_trajectory import read_jsonl, validate_pipeline_version  # noqa: E402

REQUIRED_CLAIM_COLS = [
    "claim_id",
    "group_id",
    "split",
    "y_pm",
    "y_com",
    "y_cp",
    "primary_support_label",
    "oc_eligible",
    "bcp_prompt",
]


def main() -> None:
    errors: list[str] = []
    if not FHIR_QWEN_VNEXT.is_file():
        errors.append(f"Missing {FHIR_QWEN_VNEXT}")
    else:
        n_bad = 0
        for row in read_jsonl(FHIR_QWEN_VNEXT):
            try:
                validate_pipeline_version(row)
            except ValueError as e:
                errors.append(str(e))
                n_bad += 1
        print(f"FHIR per_trajectory: OK pipeline_version (checked rows, bad={n_bad})")

    fhir_pq = FHIR_PROBE_OUT / "claims.parquet"
    if not fhir_pq.is_file():
        errors.append(f"Missing {fhir_pq}")
    else:
        df = pd.read_parquet(fhir_pq)
        for c in REQUIRED_CLAIM_COLS:
            if c not in df.columns:
                errors.append(f"claims.parquet missing column: {c}")
        if "unverifiable" in df["primary_support_label"].astype(str).values:
            errors.append("primary_support_label contains deprecated unverifiable")
        splits = json.loads((FHIR_PROBE_OUT / "splits.json").read_text())
        if splits.get("n_claims") != len(df):
            errors.append("splits.json n_claims mismatch")
        print(
            f"FHIR claims: n={len(df)} y_pm={int(df.y_pm.sum())} "
            f"oc_eligible={int(df.oc_eligible.sum())} splits={df.split.value_counts().to_dict()}"
        )

    shop_pq = REPO / "results/probe_shopping/claims.parquet"
    if shop_pq.is_file():
        sdf = pd.read_parquet(shop_pq)
        if (sdf["split"] != "ood").any():
            errors.append("shopping claims must all be split=ood")
        print(f"Shopping OOD claims: n={len(sdf)}")

    bank = FHIR_PROBE_OUT / "feature_bank" / "manifest.json"
    if bank.is_file():
        man = json.loads(bank.read_text())
        if int(man.get("completed_rows", 0)) >= int(man.get("n_claims", 0)):
            print("P1 feature_bank: COMPLETE")
        else:
            errors.append(f"P1 incomplete: {man.get('completed_rows')}/{man.get('n_claims')}")
    else:
        print("P1 feature_bank: NOT STARTED (expected after P0)")

    legacy = FHIR_PROBE_OUT / "hidden_bcp_L63.npy"
    if legacy.is_file():
        import numpy as np

        arr = np.load(legacy)
        if arr.shape[0] < 100:
            print(f"WARN: legacy smoke hidden only {arr.shape[0]} rows — do not use for paper")

    if errors:
        print("P0/P1 VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("P0 validation PASSED.")


if __name__ == "__main__":
    main()
