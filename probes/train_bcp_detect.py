#!/usr/bin/env python3
"""Train linear BCP-Detect probe — reads feature_bank/ (P2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.p2_eval import layer_sweep, train_bcp_hidden, write_json  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402
from lib.templates import BCP_VARIANTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--variant", type=str, default="full", choices=list(BCP_VARIANTS))
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--layer-slot", type=int, default=2, help="0=L4, 1=L2, 2=L")
    ap.add_argument("--label", type=str, default="y_pm", choices=["y_pm", "y_com", "y_cp"])
    ap.add_argument("--layer-sweep", action="store_true")
    args = ap.parse_args()

    if args.layer_sweep:
        metrics = layer_sweep(
            args.train_dir,
            label=args.label,
            variant=args.variant,
            pool=args.pool,
        )
        joblib.dump({"metrics": metrics}, args.train_dir / f"probe_{args.label}_{args.variant}_{args.pool}_sweep.joblib")
    else:
        clf, metrics, _, _ = train_bcp_hidden(
            args.train_dir,
            label=args.label,
            variant=args.variant,
            pool=args.pool,
            layer_slot=args.layer_slot,
        )
        joblib.dump(
            {
                "clf": clf,
                "label": args.label,
                "variant": args.variant,
                "pool": args.pool,
                "layer_slot": args.layer_slot,
                "layer_name": metrics["layer_name"],
            },
            args.train_dir / "probe_bcp_detect.joblib",
        )

    out = args.train_dir / "bcp_detect_metrics.json"
    write_json(out, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
