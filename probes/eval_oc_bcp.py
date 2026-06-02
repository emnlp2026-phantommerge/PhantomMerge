#!/usr/bin/env python3
"""P3: Owner-Contrastive BCP margins + COM vs clean stats."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.feature_bank import bank_dir, load_manifest  # noqa: E402
from lib.paths import FHIR_PROBE_OUT  # noqa: E402
from lib.stats_utils import compare_groups, mean_ci  # noqa: E402


def _cos(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def margin(vectors: dict) -> float:
    return _cos(vectors["source"], vectors["claim"]) - _cos(vectors["anchor"], vectors["claim"])


def run_oc_eval(probe_dir: Path, triplets_path: Path | None = None) -> dict:
    triplets_path = triplets_path or (bank_dir(probe_dir) / "oc_triplets.jsonl")
    if not triplets_path.is_file():
        raise FileNotFoundError(triplets_path)

    man = load_manifest(bank_dir(probe_dir)) or {}
    owner_layer = (man.get("layer_indices") or [None])[-1]

    by_claim: dict[str, dict] = defaultdict(
        lambda: {"margins": {}, "oc_kind": "", "oc_source_id": "", "layer_index": None}
    )
    with triplets_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["claim_id"]
            stage = obj["stage"]
            by_claim[cid]["margins"][stage] = margin(obj["vectors"])
            by_claim[cid]["oc_kind"] = obj.get("oc_kind", "")
            by_claim[cid]["oc_source_id"] = obj.get("oc_source_id", "")
            by_claim[cid]["layer_index"] = obj.get("layer_index", owner_layer)

    rows: list[dict] = []
    for cid, data in by_claim.items():
        m = data["margins"]
        kind = data["oc_kind"]
        label = "com" if kind == "com" else ("clean" if kind == "clean_control" else "other")
        rec = {
            "claim_id": cid,
            "label": label,
            "oc_source_id": data["oc_source_id"],
            "layer_index": data["layer_index"],
            "margin_evidence_only": m.get("evidence_only"),
            "margin_after_anchor": m.get("after_anchor"),
            "margin_final_prefix": m.get("final_prefix"),
        }
        if rec["margin_evidence_only"] is not None and rec["margin_final_prefix"] is not None:
            rec["margin_shift"] = rec["margin_final_prefix"] - rec["margin_evidence_only"]
        rows.append(rec)

    if not rows:
        raise RuntimeError("No OC triplets found")

    out_csv = probe_dir / "oc_bcp_stage_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def _vals(lab: str, col: str) -> list[float]:
        return [float(r[col]) for r in rows if r["label"] == lab and r.get(col) is not None]

    stages = ["margin_evidence_only", "margin_after_anchor", "margin_final_prefix", "margin_shift"]
    stage_stats: dict[str, dict] = {}
    for col in stages:
        com_v = _vals("com", col)
        clean_v = _vals("clean", col)
        stage_stats[col] = {
            "com": mean_ci(com_v),
            "clean": mean_ci(clean_v),
            "compare_com_vs_clean": compare_groups(com_v, clean_v),
        }

    com_shift = _vals("com", "margin_shift")
    clean_shift = _vals("clean", "margin_shift")

    summary = {
        "n_claims": len(rows),
        "n_com": sum(1 for r in rows if r["label"] == "com"),
        "n_clean": sum(1 for r in rows if r["label"] == "clean"),
        "owner_layer_index": owner_layer,
        "owner_layer_note": "P1 OC triplets extracted @ last layer (L); P3 owner-layer selection = L",
        "com_mean_margin_evidence_only": stage_stats["margin_evidence_only"]["com"]["mean"],
        "com_mean_margin_final_prefix": stage_stats["margin_final_prefix"]["com"]["mean"],
        "clean_mean_margin_evidence_only": stage_stats["margin_evidence_only"]["clean"]["mean"],
        "clean_mean_margin_final_prefix": stage_stats["margin_final_prefix"]["clean"]["mean"],
        "com_mean_margin_shift": float(np.mean(com_shift)) if com_shift else None,
        "clean_mean_margin_shift": float(np.mean(clean_shift)) if clean_shift else None,
        "stage_stats": stage_stats,
        "mechanism_pattern": {
            "com_evidence_margin_gt_clean": (
                (stage_stats["margin_evidence_only"]["com"]["mean"] or 0)
                > (stage_stats["margin_evidence_only"]["clean"]["mean"] or 0)
            ),
            "com_final_margin_lt_evidence": (
                (stage_stats["margin_final_prefix"]["com"]["mean"] or 0)
                < (stage_stats["margin_evidence_only"]["com"]["mean"] or 0)
            ),
            "com_negative_shift_rate": (
                float(np.mean([1 if s < 0 else 0 for s in com_shift])) if com_shift else None
            ),
        },
    }
    summary_path = probe_dir / "oc_bcp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--triplets", type=Path, default=None)
    args = ap.parse_args()
    summary = run_oc_eval(args.probe_dir, args.triplets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
