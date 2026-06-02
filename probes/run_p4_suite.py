#!/usr/bin/env python3
"""P4 one-shot: BCP gating + anchor-only mitigation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from eval_mitigation_gating import run_gating  # noqa: E402
from lib.paths import FHIR_PROBE_OUT, FHIR_QWEN_VNEXT  # noqa: E402
from lib.p2_eval import write_json  # noqa: E402
from run_anchor_only_mitigation import run_anchor_only  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--skip-anchor-only", action="store_true")
    ap.add_argument("--anchor-max-cases", type=int, default=50)
    ap.add_argument("--anchor-smoke", type=int, default=0, help="If >0, override max-cases")
    ap.add_argument("--dry-run-anchor", action="store_true")
    ap.add_argument("--resume-anchor", action="store_true")
    ap.add_argument("--device-map", type=str, default="none")
    ap.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:8001/v1")
    ap.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="FHIR rollout JSON (not shipped); required for anchor-only regen unless --dry-run-anchor.",
    )
    args = ap.parse_args()

    taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    gating = run_gating(args.probe_dir, taus=taus)
    print("=== BCP gating ===")
    print(json.dumps(gating.get("test_at_selected_tau"), indent=2))

    anchor_summary = {"skipped": True}
    if not args.skip_anchor_only:
        if args.rollout is None and not args.dry_run_anchor:
            anchor_summary = {
                "skipped": True,
                "reason": (
                    "No --rollout (FHIR rollout not in release). "
                    "Pass --rollout, --dry-run-anchor, or --skip-anchor-only."
                ),
            }
        else:
            max_cases = args.anchor_smoke or args.anchor_max_cases
            per_traj = FHIR_QWEN_VNEXT
            anchor_summary = run_anchor_only(
                per_path=per_traj,
                rollout_path=args.rollout,
                out_dir=args.probe_dir / "mitigation_anchor_only",
                model_path=str(REPO / ".cache/huggingface/hub/Qwen3-32B"),
                max_cases=max_cases,
                max_new_tokens=384,
                device_map=args.device_map,
                judge_base_url=args.judge_base_url,
                judge_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                resume=args.resume_anchor,
                dry_run=args.dry_run_anchor,
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_dir": str(args.probe_dir),
        "gating": gating,
        "anchor_only": anchor_summary,
    }
    write_json(args.probe_dir / "p4_results_summary.json", summary)
    print(f"Wrote {args.probe_dir / 'p4_results_summary.json'}")


if __name__ == "__main__":
    main()
