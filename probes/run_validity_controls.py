#!/usr/bin/env python3
"""P2 validity controls: BoW+LR and ModernBERT-large on BCP prompt text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parent
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.p2_eval import (  # noqa: E402
    LABELS,
    eval_lr_probe,
    fit_balanced_lr,
    metrics_dict,
    prob_positive,
    split_masks,
    train_text_bow,
    write_json,
)
from lib.paths import FHIR_PROBE_OUT  # noqa: E402


def _encode_modernbert(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    cache_path: Path | None,
) -> np.ndarray:
    if cache_path and cache_path.is_file():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr.astype(np.float64)
        print(f"Ignore stale cache {cache_path}: shape {arr.shape} != {len(texts)}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    dev = torch.device(device if device != "cpu" and torch.cuda.is_available() else "cpu")
    model.to(dev)

    out_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            out_rows.append(pooled.cpu().numpy())

    X = np.vstack(out_rows).astype(np.float64)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X


def run_validity_controls(
    probe_dir: Path,
    *,
    label: str = "y_pm",
    text_col: str = "bcp_prompt",
    skip_bert: bool = False,
    modernbert_model: str = "answerdotai/ModernBERT-large",
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
    cache_bert: bool = True,
) -> dict:
    df = pd.read_parquet(probe_dir / "claims.parquet")
    if text_col not in df.columns:
        raise KeyError(f"Missing column {text_col}")

    results: dict[str, object] = {
        "probe_dir": str(probe_dir),
        "label": label,
        "text_col": text_col,
        "n_claims": int(len(df)),
    }

    _, bow = train_text_bow(df, text_col=text_col, label=label)
    results["bow_tfidf_lr"] = bow
    print(json.dumps(bow, indent=2))

    if not skip_bert:
        texts = df[text_col].astype(str).tolist()
        cache = None
        if cache_bert:
            safe = modernbert_model.replace("/", "__")
            cache = probe_dir / f"text_embeddings_{safe}.npy"
        try:
            X = _encode_modernbert(
                texts,
                model_name=modernbert_model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                cache_path=cache,
            )
            y = df[label].astype(int).values
            masks = split_masks(df)
            clf = fit_balanced_lr(X[masks["train"]], y[masks["train"]])
            bert_metrics = {
                "method": "modernbert_large_mean_pool",
                "model": modernbert_model,
                "device": device,
                "embedding_dim": int(X.shape[1]),
                "cache": str(cache) if cache else None,
                "splits": eval_lr_probe(clf, X, y, masks),
            }
            results["modernbert_large"] = bert_metrics
            print(json.dumps(bert_metrics, indent=2))
        except Exception as exc:  # noqa: BLE001
            results["modernbert_large"] = {"error": str(exc)[:800]}
            print(f"ModernBERT skipped/failed: {exc}")

    out = probe_dir / "validity_controls.json"
    write_json(out, results)
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 validity controls (text baselines)")
    ap.add_argument("--probe-dir", type=Path, default=FHIR_PROBE_OUT)
    ap.add_argument("--label", type=str, default="y_pm", choices=LABELS)
    ap.add_argument("--text-col", type=str, default="bcp_prompt")
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument(
        "--modernbert-model",
        type=str,
        default="answerdotai/ModernBERT-large",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cache-bert", action="store_true", help="Save/load text embedding cache")
    args = ap.parse_args()
    run_validity_controls(
        args.probe_dir,
        label=args.label,
        text_col=args.text_col,
        skip_bert=args.skip_bert,
        modernbert_model=args.modernbert_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_bert=args.cache_bert,
    )


if __name__ == "__main__":
    main()
