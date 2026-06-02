#!/usr/bin/env python3
"""
P2 full suite — BCP-Detect + ablations + validity + Shopping OOD.

Requires P1 feature_bank COMPLETE on FHIR (and Shopping for OOD row).
CPU for linear probes; ModernBERT optional (--skip-bert).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.feature_bank import load_bcp_matrix, load_manifest  # noqa: E402
from lib.p2_eval import (  # noqa: E402
    LABELS,
    eval_lr_probe,
    layer_sweep,
    pool_ablation,
    prob_positive,
    require_complete_bank,
    split_masks,
    train_bcp_hidden,
    variant_ablation,
    write_json,
)
from lib.paths import FHIR_PROBE_OUT  # noqa: E402

REPO = PROBE_ROOT.parents[1]
SHOPPING_PROBE_OUT = REPO / "results/probe_shopping"


def _eval_ood(
    fhir_dir: Path,
    shopping_dir: Path,
    *,
    label: str,
    variant: str,
    pool: str,
    layer_slot: int,
) -> dict:
    clf, train_metrics, _, _ = train_bcp_hidden(
        fhir_dir,
        label=label,
        variant=variant,
        pool=pool,
        layer_slot=layer_slot,
    )
    bank = require_complete_bank(shopping_dir)
    sdf = pd.read_parquet(shopping_dir / "claims.parquet")
    H = load_bcp_matrix(bank, variant=variant, pool=pool, layer_slot=layer_slot)
    if H.shape[0] != len(sdf):
        raise ValueError(f"Shopping features {H.shape[0]} != claims {len(sdf)}")
    y = sdf[label].astype(int).values
    mask = (sdf["split"].astype(str) == "ood").values
    ood = eval_lr_probe(clf, H, y, {"ood": mask})
    return {
        "train_fhir": train_metrics["splits"].get("test"),
        "shopping_ood": ood.get("ood"),
        "n_ood": int(mask.sum()),
        "variant": variant,
        "pool": pool,
        "layer_slot": layer_slot,
        "label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full P2 experiment suite")
    ap.add_argument("--fhir-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--shopping-dir", type=Path, default=SHOPPING_PROBE_OUT)
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--layer-slot", type=int, default=2)
    args = ap.parse_args()

    fhir_dir = args.fhir_dir
    require_complete_bank(fhir_dir)

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fhir_dir": str(fhir_dir),
        "shopping_dir": str(args.shopping_dir),
    }

    # Primary probe (y_pm, full, mean, L)
    clf, main_metrics, _, df = train_bcp_hidden(
        fhir_dir,
        label="y_pm",
        variant="full",
        pool="mean",
        layer_slot=args.layer_slot,
    )
    joblib.dump(
        {
            "clf": clf,
            "label": "y_pm",
            "variant": "full",
            "pool": "mean",
            "layer_slot": args.layer_slot,
            "layer_name": main_metrics["layer_name"],
        },
        fhir_dir / "probe_bcp_detect.joblib",
    )
    write_json(fhir_dir / "bcp_detect_metrics.json", main_metrics)
    summary["primary_bcp_detect"] = main_metrics
    print("=== primary y_pm full mean L ===")
    print(json.dumps(main_metrics["splits"], indent=2))

    summary["layer_sweep"] = layer_sweep(fhir_dir, label="y_pm", variant="full", pool="mean")
    write_json(fhir_dir / "bcp_layer_sweep.json", summary["layer_sweep"])

    summary["ablation_input"] = variant_ablation(
        fhir_dir, label="y_pm", pool="mean", layer_slot=args.layer_slot
    )
    write_json(fhir_dir / "ablation_input.json", summary["ablation_input"])

    summary["ablation_pool"] = pool_ablation(
        fhir_dir, label="y_pm", variant="full", layer_slot=args.layer_slot
    )
    write_json(fhir_dir / "ablation_pool.json", summary["ablation_pool"])

    multi_head = {}
    for lab in LABELS:
        _, m, _, _ = train_bcp_hidden(
            fhir_dir,
            label=lab,
            variant="full",
            pool="mean",
            layer_slot=args.layer_slot,
        )
        multi_head[lab] = m["splits"]
    summary["multi_head"] = multi_head
    write_json(fhir_dir / "bcp_multi_head.json", multi_head)

    from run_validity_controls import run_validity_controls  # noqa: E402

    summary["validity_controls"] = run_validity_controls(
        fhir_dir,
        skip_bert=args.skip_bert,
        cache_bert=True,
    )

    if not args.skip_ood and (args.shopping_dir / "feature_bank/manifest.json").is_file():
        man = load_manifest(args.shopping_dir / "feature_bank")
        if man and int(man.get("completed_rows", 0)) >= int(man.get("n_claims", 0)):
            ood = _eval_ood(
                fhir_dir,
                args.shopping_dir,
                label="y_pm",
                variant="full",
                pool="mean",
                layer_slot=args.layer_slot,
            )
            summary["shopping_ood"] = ood
            write_json(fhir_dir / "shopping_ood_eval.json", ood)
            print("=== Shopping OOD ===")
            print(json.dumps(ood, indent=2))
        else:
            summary["shopping_ood"] = {"skipped": "incomplete shopping feature_bank"}
    else:
        summary["shopping_ood"] = {"skipped": True}

    write_json(fhir_dir / "p2_results_summary.json", summary)
    print(f"Wrote {fhir_dir / 'p2_results_summary.json'}")


if __name__ == "__main__":
    main()
