#!/usr/bin/env python3
"""Forward-only BCP hidden (variant=full) for mitigated_claims.parquet — probe-in-the-loop."""

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

from lib.feature_bank import resolve_layers, write_manifest  # noqa: E402
from lib.hidden_extract import (  # noqa: E402
    claim_features_multi_layer,
    forward_hidden_states,
    load_model_and_tokenizer,
)
from lib.paths import DEFAULT_MODEL_DIR  # noqa: E402
from lib.templates import build_bcp_prompt_variant  # noqa: E402


def _row_dict(row: pd.Series) -> dict:
    return {
        "query": str(row["query"]),
        "anchor_evidence": str(row["anchor_evidence_text"]),
        "other_evidence": str(row["other_evidence_text"]),
        "claim_text": str(row["claim_text"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--device-map", type=str, default="none")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.claims_parquet.parent / "feature_bank_mitigated")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.claims_parquet).reset_index(drop=True)
    n = len(df)
    manifest_path = out_dir / "manifest.json"
    start = 0
    if args.resume and manifest_path.is_file():
        start = int(json.loads(manifest_path.read_text()).get("completed_rows", 0))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model, tokenizer, n_layers = load_model_and_tokenizer(
        args.model_path, device_map=args.device_map
    )
    layer_spec = resolve_layers(n_layers)
    layer_list = list(layer_spec.indices)

    mean_path = out_dir / "bcp_full_mean.npy"
    if start > 0 and mean_path.is_file():
        mean_all = np.load(mean_path)
        if mean_all.shape[0] != n:
            raise ValueError("resume shape mismatch")
    else:
        dim_probe = 5120
        mean_all = np.zeros((n, len(layer_list), dim_probe), dtype=np.float32)

    t0 = time.time()
    for row_idx in range(n):
        if row_idx < start:
            continue
        fields = _row_dict(df.iloc[row_idx])
        prompt = build_bcp_prompt_variant("full", **fields)
        hs_by_layer, offsets = forward_hidden_states(
            model,
            tokenizer,
            prompt,
            max_length=args.max_length,
            layer_indices=layer_list,
        )
        mean_arr, _ = claim_features_multi_layer(
            hs_by_layer,
            layer_list,
            prompt,
            fields["claim_text"],
            offsets,
        )
        if mean_all.shape[2] != mean_arr.shape[1]:
            mean_all = np.zeros((n, len(layer_list), mean_arr.shape[1]), dtype=np.float32)
        mean_all[row_idx] = mean_arr
        if (row_idx + 1) % 10 == 0:
            np.save(mean_path, mean_all)
            write_manifest(
                out_dir,
                {
                    "spec_version": "mitigated_bcp_full_v1",
                    "completed_rows": row_idx + 1,
                    "n_rows": n,
                    "layer_indices": layer_list,
                },
            )
            print(f"  [{row_idx+1}/{n}] {time.time()-t0:.0f}s", flush=True)

    np.save(mean_path, mean_all)
    write_manifest(
        out_dir,
        {
            "spec_version": "mitigated_bcp_full_v1",
            "completed_rows": n,
            "n_rows": n,
            "layer_indices": layer_list,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"Done {n} rows -> {mean_path}")


if __name__ == "__main__":
    main()
