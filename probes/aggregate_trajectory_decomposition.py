#!/usr/bin/env python3
"""Trajectory-level COM/CP/hall decomposition from VNEXT per_trajectory.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.labels import trajectory_decomposition  # noqa: E402
from lib.paths import FHIR_QWEN_VNEXT, PROBE_RUNS, SHOPPING_QWEN_VNEXT  # noqa: E402
from lib.per_trajectory import read_jsonl, validate_pipeline_version  # noqa: E402


def aggregate_file(per_path: Path, *, domain: str, model: str) -> list[dict]:
    rows_out: list[dict] = []
    for row in read_jsonl(per_path):
        validate_pipeline_version(row)
        gid = (
            str(row.get("question_id"))
            if domain == "fhir"
            else str(row.get("orig_index") if row.get("orig_index") is not None else row.get("line_index"))
        )
        traj = dict(row.get("trajectory_labels") or {})
        decomp = trajectory_decomposition(traj)
        struct = dict(row.get("claim_structure_counts") or {})
        rows_out.append(
            {
                "domain": domain,
                "model": model,
                "group_id": gid,
                "n_claims": struct.get("n_claims_total", len(row.get("claims") or [])),
                "task_correct": int(
                    row.get("task_correct_exact")
                    if "task_correct_exact" in row
                    else row.get("task_correct_llm", 0)
                ),
                **{k: int(v) for k, v in decomp.items()},
                **{f"struct_{k}": v for k, v in struct.items()},
            }
        )
    return rows_out


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    keys = [
        "has_phantom_merge",
        "has_com_only",
        "has_cp_only",
        "has_com_and_cp",
        "has_anchored_hall_only",
    ]
    return {k: sum(int(r.get(k, 0)) for r in rows) / n for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROBE_RUNS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (FHIR_QWEN_VNEXT, "fhir", "qwen3-32b"),
        (SHOPPING_QWEN_VNEXT, "shopping", "qwen3-32b"),
    ]
    all_rows: list[dict] = []
    summary: dict = {}
    for per_path, domain, model in datasets:
        if not per_path.is_file():
            raise FileNotFoundError(per_path)
        part = aggregate_file(per_path, domain=domain, model=model)
        all_rows.extend(part)
        summary[f"{domain}_{model}"] = {"n_trajectories": len(part), **summarize(part)}

    csv_path = args.out_dir / "trajectory_decomposition.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    json_path = args.out_dir / "trajectory_decomposition_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} ({len(all_rows)} rows)")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
