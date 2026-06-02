#!/usr/bin/env python3
"""
Hidden-state feature bank extraction for FHIR probe training.

One scheduled job:
  - BCP variants full/no_other/no_anchor/no_query: 1 forward each → layers L4,L2,L + mean/last
  - OC-BCP: COM (oc_eligible) + stratified clean controls

Smoke:  --smoke 5
Full:   --resume (checkpoint every 25 rows)

Do NOT use extract_hidden_qwen32b.py for production.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.feature_bank import (  # noqa: E402
    SPEC_VERSION,
    alloc_bcp_arrays,
    bank_dir,
    load_manifest,
    resolve_layers,
    sample_clean_controls,
    validate_smoke_bank,
    write_bcp_slice,
    write_manifest,
    write_row_index,
)
from lib.hidden_extract import (  # noqa: E402
    claim_features_multi_layer,
    forward_hidden_states,
    load_model_and_tokenizer,
    mean_pool_hidden,
    oc_section_indices,
)
from lib.paths import DEFAULT_MODEL_DIR, FHIR_PROBE_OUT  # noqa: E402
from lib.templates import (  # noqa: E402
    BCP_VARIANTS,
    build_bcp_prompt_variant,
    build_oc_prompt,
    final_answer_prefix_from_text,
)

OC_STAGES = ("evidence_only", "after_anchor", "final_prefix")
CHECKPOINT_EVERY = 25


def _row_dict(row: pd.Series) -> dict:
    return {
        "query": str(row["query"]),
        "anchor_evidence": str(row["anchor_evidence_text"]),
        "other_evidence": str(row["other_evidence_text"]),
        "claim_text": str(row["claim_text"]),
        "final_answer": str(row.get("final_answer", "")),
        "oc_source_id": str(row.get("oc_source_id", "")),
    }


def _bcp_prompt(variant: str, fields: dict) -> str:
    return build_bcp_prompt_variant(
        variant,
        query=fields["query"],
        anchor_evidence=fields["anchor_evidence"],
        other_evidence=fields["other_evidence"],
        claim_text=fields["claim_text"],
    )


def _checkpoint_bcp(
    bank: Path,
    bcp_mean: dict[str, np.ndarray],
    bcp_last: dict[str, np.ndarray],
    manifest: dict,
) -> None:
    for v in BCP_VARIANTS:
        write_bcp_slice(bank, v, bcp_mean[v], bcp_last[v])
    write_manifest(bank, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="PROBE feature bank extract (spec v1)")
    ap.add_argument("--claims-parquet", type=Path, default=FHIR_PROBE_OUT / "claims.parquet")
    ap.add_argument("--out-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--oc-clean-max", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument(
        "--device-map",
        type=str,
        default="none",
        help="Prefer 'none' + single GPU (spec); 'auto' needs accelerate",
    )
    ap.add_argument("--skip-oc", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    df = pd.read_parquet(args.claims_parquet).reset_index(drop=True)
    n_total = len(df)
    if args.smoke > 0:
        df = df.head(args.smoke).copy()
        print(f"SMOKE: {len(df)} / {n_total} claims")

    bank = bank_dir(args.out_dir)
    bank.mkdir(parents=True, exist_ok=True)

    start_row = 0
    if args.resume and not args.smoke:
        prev = load_manifest(bank)
        if prev and prev.get("spec_version") == SPEC_VERSION:
            start_row = int(prev.get("completed_rows", 0))
            print(f"Resume from row {start_row}")

    model, tokenizer, n_layers = load_model_and_tokenizer(
        args.model_path, device_map=args.device_map
    )
    layer_spec = resolve_layers(n_layers)
    layer_list = list(layer_spec.indices)
    print(f"layers={layer_spec.names} indices={layer_list}")

    n = len(df)
    dim: int | None = None
    bcp_mean: dict[str, np.ndarray] = {}
    bcp_last: dict[str, np.ndarray] = {}
    oc_records: list[dict] = []

    if start_row > 0:
        for v in BCP_VARIANTS:
            bcp_mean[v] = np.load(bank / f"bcp_{v}_mean.npy")
            bcp_last[v] = np.load(bank / f"bcp_{v}_last.npy")
            dim = int(bcp_mean[v].shape[2])
        oc_path = bank / "oc_triplets.jsonl"
        if oc_path.is_file():
            with oc_path.open(encoding="utf-8") as f:
                for line in f:
                    oc_records.append(json.loads(line))
        else:
            oc_records.clear()

    oc_layer = layer_list[-1]
    t0 = time.time()
    for row_idx in range(n):
        if row_idx < start_row:
            continue
        row = df.iloc[row_idx]
        fields = _row_dict(row)

        for variant in BCP_VARIANTS:
            prompt = _bcp_prompt(variant, fields)
            hs_by_layer, offsets = forward_hidden_states(
                model,
                tokenizer,
                prompt,
                max_length=args.max_length,
                layer_indices=layer_list,
            )
            mean_arr, last_arr = claim_features_multi_layer(
                hs_by_layer,
                layer_list,
                prompt,
                fields["claim_text"],
                offsets,
            )
            if dim is None:
                dim = int(mean_arr.shape[1])
                for v in BCP_VARIANTS:
                    bcp_mean[v], bcp_last[v] = alloc_bcp_arrays(n, 3, dim)
            bcp_mean[variant][row_idx] = mean_arr
            bcp_last[variant][row_idx] = last_arr

        if args.smoke and dim:
            m = bcp_mean["full"][row_idx]
            if not np.isfinite(m).all() or np.linalg.norm(m) < 1e-6:
                raise RuntimeError(f"Smoke invalid vector at row {row_idx}")

        if not args.skip_oc and int(row.get("oc_eligible", 0)) == 1:
            other_ev = fields["other_evidence"] or f"Object {fields['oc_source_id']}"
            fprefix = final_answer_prefix_from_text(fields["final_answer"])
            for stage in OC_STAGES:
                oc_prompt = build_oc_prompt(
                    stage,
                    anchor_evidence=fields["anchor_evidence"],
                    source_evidence=other_ev,
                    claim=fields["claim_text"],
                    final_answer_prefix=fprefix,
                )
                hs_by_layer, off_oc = forward_hidden_states(
                    model,
                    tokenizer,
                    oc_prompt,
                    max_length=args.max_length,
                    layer_indices=[oc_layer],
                )
                hs = hs_by_layer[oc_layer]
                sec = oc_section_indices(oc_prompt, off_oc)
                oc_records.append(
                    {
                        "claim_id": row["claim_id"],
                        "row_idx": row_idx,
                        "oc_kind": "com",
                        "oc_source_id": fields["oc_source_id"],
                        "stage": stage,
                        "layer_index": oc_layer,
                        "vectors": {
                            k: mean_pool_hidden(hs, sec[k]).tolist()
                            for k in ("anchor", "source", "claim")
                        },
                    }
                )

        if (row_idx + 1) % args.checkpoint_every == 0 or row_idx == n - 1:
            completed = row_idx + 1
            if dim is not None:
                manifest = {
                    "spec_version": SPEC_VERSION,
                    "model_path": args.model_path,
                    "claims_parquet": str(args.claims_parquet),
                    "n_claims": n,
                    "hidden_dim": dim,
                    "layer_indices": list(layer_list),
                    "layer_names": list(layer_spec.names),
                    "bcp_variants": list(BCP_VARIANTS),
                    "pooling": ["mean", "last"],
                    "completed_rows": completed,
                    "smoke": bool(args.smoke),
                    "elapsed_sec": round(time.time() - t0, 1),
                }
                _checkpoint_bcp(bank, bcp_mean, bcp_last, manifest)
            print(f"  checkpoint row {completed}/{n}")

    # OC clean controls (full run only)
    if not args.skip_oc and not args.smoke and dim is not None:
        clean_df = sample_clean_controls(
            pd.read_parquet(args.claims_parquet).reset_index(drop=True),
            n=args.oc_clean_max,
        )
        print(f"OC clean controls: {len(clean_df)}")
        seen_oc = {r["claim_id"] for r in oc_records}
        for _, crow in clean_df.iterrows():
            if crow["claim_id"] in seen_oc:
                continue
            fields = _row_dict(crow)
            fprefix = final_answer_prefix_from_text(fields["final_answer"])
            for stage in OC_STAGES:
                oc_prompt = build_oc_prompt(
                    stage,
                    anchor_evidence=fields["anchor_evidence"],
                    source_evidence="(no other observed object)",
                    claim=fields["claim_text"],
                    final_answer_prefix=fprefix,
                )
                hs_by_layer, off_oc = forward_hidden_states(
                    model,
                    tokenizer,
                    oc_prompt,
                    max_length=args.max_length,
                    layer_indices=[oc_layer],
                )
                hs = hs_by_layer[oc_layer]
                sec = oc_section_indices(oc_prompt, off_oc)
                oc_records.append(
                    {
                        "claim_id": crow["claim_id"],
                        "row_idx": int(crow.name) if isinstance(crow.name, int) else None,
                        "oc_kind": "clean_control",
                        "oc_source_id": "",
                        "stage": stage,
                        "layer_index": oc_layer,
                        "vectors": {
                            k: mean_pool_hidden(hs, sec[k]).tolist()
                            for k in ("anchor", "source", "claim")
                        },
                    }
                )

    assert dim is not None
    write_row_index(bank, df)
    _checkpoint_bcp(bank, bcp_mean, bcp_last, manifest={
        "spec_version": SPEC_VERSION,
        "model_path": args.model_path,
        "claims_parquet": str(args.claims_parquet),
        "n_claims": n,
        "hidden_dim": dim,
        "layer_indices": list(layer_list),
        "layer_names": list(layer_spec.names),
        "bcp_variants": list(BCP_VARIANTS),
        "pooling": ["mean", "last"],
        "n_forward_bcp_per_claim": len(BCP_VARIANTS),
        "n_oc_records": len(oc_records),
        "completed_rows": n,
        "smoke": bool(args.smoke),
        "elapsed_sec": round(time.time() - t0, 1),
    })

    oc_path = bank / "oc_triplets.jsonl"
    with oc_path.open("w", encoding="utf-8") as f:
        for rec in oc_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    man = load_manifest(bank) or {}
    man["oc_triplets"] = str(oc_path)
    write_manifest(bank, man)

    if args.smoke:
        validate_smoke_bank(bank, n, dim)
        print(f"SMOKE OK: N={n} D={dim} variants={BCP_VARIANTS}")

    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
