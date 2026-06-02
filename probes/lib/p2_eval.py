"""P2 training/eval helpers — BCP-Detect, ablations, OOD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from .feature_bank import bank_dir, load_bcp_matrix, load_manifest
from .templates import BCP_VARIANTS

LABELS = ("y_pm", "y_com", "y_cp")
POOLS = ("mean", "last")


def metrics_dict(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    out: dict[str, Any] = {"n": int(len(y_true)), "pos_rate": float(y_true.mean()) if len(y_true) else None}
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        out["auroc"] = None
        out["note"] = "single_class_split"
        return out
    out["auroc"] = float(roc_auc_score(y_true, prob))
    return out


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    split = df["split"].astype(str)
    return {
        "train": (split == "train").values,
        "val": (split == "val").values,
        "test": (split == "test").values,
        "ood": (split == "ood").values,
    }


def fit_balanced_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)
    return clf


def prob_positive(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


def eval_lr_probe(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {name: metrics_dict(y[m], prob_positive(clf, X[m])) for name, m in masks.items() if m.any()}


def require_complete_bank(probe_dir: Path) -> Path:
    bank = bank_dir(probe_dir)
    man = load_manifest(bank)
    if man is None:
        raise FileNotFoundError(f"Missing feature_bank under {probe_dir}")
    if int(man.get("completed_rows", 0)) < int(man.get("n_claims", 0)):
        raise RuntimeError(
            f"Incomplete bank: {man.get('completed_rows')}/{man.get('n_claims')}"
        )
    return bank


def train_bcp_hidden(
    probe_dir: Path,
    *,
    label: str = "y_pm",
    variant: str = "full",
    pool: str = "mean",
    layer_slot: int = 2,
    feature_bank: Path | None = None,
) -> tuple[LogisticRegression, dict[str, Any], np.ndarray, pd.DataFrame]:
    bank = feature_bank or require_complete_bank(probe_dir)
    man = load_manifest(bank)
    df = pd.read_parquet(probe_dir / "claims.parquet")
    H = load_bcp_matrix(bank, variant=variant, pool=pool, layer_slot=layer_slot)
    if H.shape[0] != len(df):
        raise ValueError(f"Features {H.shape[0]} != claims {len(df)}")
    y = df[label].astype(int).values
    masks = split_masks(df)
    if not masks["train"].any():
        raise RuntimeError("No train split rows")
    clf = fit_balanced_lr(H[masks["train"]], y[masks["train"]])
    layer_names = man.get("layer_names") or ["L4", "L2", "L"]
    meta = {
        "probe_dir": str(probe_dir),
        "feature_bank": str(bank),
        "label": label,
        "variant": variant,
        "pool": pool,
        "layer_slot": layer_slot,
        "layer_name": layer_names[layer_slot],
        "hidden_dim": int(H.shape[1]),
        "manifest_spec": man.get("spec_version"),
    }
    metrics = {
        **meta,
        "splits": eval_lr_probe(clf, H, y, masks),
    }
    return clf, metrics, H, df


def layer_sweep(
    probe_dir: Path,
    *,
    label: str = "y_pm",
    variant: str = "full",
    pool: str = "mean",
) -> dict[str, Any]:
    bank = require_complete_bank(probe_dir)
    man = load_manifest(bank)
    df = pd.read_parquet(probe_dir / "claims.parquet")
    y = df[label].astype(int).values
    masks = split_masks(df)
    layer_names = man.get("layer_names") or ["L4", "L2", "L"]
    rows = []
    for slot in range(3):
        H = load_bcp_matrix(bank, variant=variant, pool=pool, layer_slot=slot)
        clf = fit_balanced_lr(H[masks["train"]], y[masks["train"]])
        entry = {
            "layer_slot": slot,
            "layer_name": layer_names[slot],
            "splits": eval_lr_probe(clf, H, y, masks),
        }
        rows.append(entry)
    return {
        "label": label,
        "variant": variant,
        "pool": pool,
        "layer_sweep": rows,
    }


def variant_ablation(
    probe_dir: Path,
    *,
    label: str = "y_pm",
    pool: str = "mean",
    layer_slot: int = 2,
) -> dict[str, Any]:
    rows = []
    for variant in BCP_VARIANTS:
        _, metrics, _, _ = train_bcp_hidden(
            probe_dir,
            label=label,
            variant=variant,
            pool=pool,
            layer_slot=layer_slot,
        )
        rows.append(
            {
                "variant": variant,
                "test_auroc": metrics["splits"].get("test", {}).get("auroc"),
                "val_auroc": metrics["splits"].get("val", {}).get("auroc"),
                "splits": metrics["splits"],
            }
        )
    return {"label": label, "pool": pool, "layer_slot": layer_slot, "variants": rows}


def pool_ablation(
    probe_dir: Path,
    *,
    label: str = "y_pm",
    variant: str = "full",
    layer_slot: int = 2,
) -> dict[str, Any]:
    rows = []
    for pool in POOLS:
        _, metrics, _, _ = train_bcp_hidden(
            probe_dir,
            label=label,
            variant=variant,
            pool=pool,
            layer_slot=layer_slot,
        )
        rows.append({"pool": pool, "test_auroc": metrics["splits"].get("test", {}).get("auroc"), "splits": metrics["splits"]})
    return {"label": label, "variant": variant, "layer_slot": layer_slot, "pools": rows}


def train_text_bow(
    df: pd.DataFrame,
    text_col: str = "bcp_prompt",
    label: str = "y_pm",
) -> tuple[Pipeline, dict[str, Any]]:
    masks = split_masks(df)
    texts = df[text_col].astype(str).tolist()
    y = df[label].astype(int).values
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "clf",
                LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    pipe.fit([texts[i] for i in range(len(texts)) if masks["train"][i]], y[masks["train"]])

    def probs(mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask)[0]
        return pipe.predict_proba([texts[i] for i in idx])[:, 1]

    metrics = {
        "method": "bow_tfidf_lr",
        "text_col": text_col,
        "label": label,
        "splits": {name: metrics_dict(y[mask], probs(mask)) for name, mask in masks.items() if mask.any()},
    }
    return pipe, metrics


from .io_utils import write_json  # noqa: F401
