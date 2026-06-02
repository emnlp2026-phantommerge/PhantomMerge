#!/usr/bin/env python3
"""Build claim-level parquet from anchor-only per_case.jsonl (mitigated judge claims)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.mitigation import load_probe_split_map  # noqa: E402
from lib.per_trajectory import export_claim_rows, read_jsonl  # noqa: E402
from lib.paths import FHIR_QWEN_VNEXT  # noqa: E402


def export_from_anchor_run(
    per_case_jsonl: Path,
    *,
    per_path: Path,
    splits_path: Path,
    out_parquet: Path,
) -> pd.DataFrame:
    per_by_qid = {str(r["question_id"]): r for r in read_jsonl(per_path)}
    split_map = load_probe_split_map(splits_path)

    rows: list[dict] = []
    meta: list[dict] = []
    for line in per_case_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["question_id"])
        mclaims = rec.get("mitigated_claims")
        if not mclaims:
            continue
        base = per_by_qid.get(qid)
        if not base:
            continue
        traj = dict(base)
        traj["claims"] = mclaims
        answer = str(rec.get("anchor_only_answer") or "")
        answer = re.sub(r"(?s)^\s*.*?</think>\s*", "", answer, count=1)
        traj["final_answer"] = answer[:4000] or traj.get("final_answer", "")

        for crow in export_claim_rows(traj, domain="fhir", model="qwen3-32b-mitigated"):
            crow["mitigation_source"] = str(rec.get("mitigation_version") or "standard_vnext")
            crow["split"] = str(rec.get("probe_split") or split_map.get(qid, ""))
            crow["regen_run"] = str(per_case_jsonl.parent.name)
            rows.append(crow)
        meta.append(
            {
                "question_id": qid,
                "split": crow["split"] if rows else split_map.get(qid, ""),
                "baseline_pm": rec.get("baseline_has_phantom_merge"),
                "mitigated_pm": rec.get("mitigated_has_phantom_merge"),
                "n_mitigated_claims": len(mclaims),
            }
        )

    if not rows:
        raise RuntimeError(f"No mitigated_claims in {per_case_jsonl}")

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    (out_parquet.parent / "mitigated_trajectory_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    stats = {
        "per_case_jsonl": str(per_case_jsonl),
        "n_claims": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "y_pm": int(df["y_pm"].sum()),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (out_parquet.parent / "export_mitigated_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-case-jsonl",
        type=Path,
        required=True,
    )
    ap.add_argument("--per-path", type=Path, default=FHIR_QWEN_VNEXT)
    ap.add_argument(
        "--splits",
        type=Path,
        default=REPO / "results/table3_representation/splits.json",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Default: <parent>/mitigated_claims.parquet",
    )
    args = ap.parse_args()
    out = args.out_parquet or (args.per_case_jsonl.parent / "mitigated_claims.parquet")
    export_from_anchor_run(
        args.per_case_jsonl,
        per_path=args.per_path,
        splits_path=args.splits,
        out_parquet=out,
    )


if __name__ == "__main__":
    main()
