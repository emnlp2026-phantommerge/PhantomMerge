#!/usr/bin/env python3
"""P3 one-shot: OC-BCP margins + CP/COM mechanism stats."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from eval_cp_mechanism import run_cp_mechanism  # noqa: E402
from eval_oc_bcp import run_oc_eval  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402
from lib.p2_eval import write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    args = ap.parse_args()

    oc = run_oc_eval(args.probe_dir)
    print("=== OC-BCP ===")
    print(json.dumps(oc, indent=2))

    cp = run_cp_mechanism(args.probe_dir)
    print("=== CP/COM ===")
    print(json.dumps(cp, indent=2))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_dir": str(args.probe_dir),
        "oc_bcp": oc,
        "cp_com": cp,
    }
    write_json(args.probe_dir / "p3_results_summary.json", summary)
    print(f"Wrote {args.probe_dir / 'p3_results_summary.json'}")


if __name__ == "__main__":
    main()
