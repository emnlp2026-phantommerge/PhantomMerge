"""Feature bank I/O — PROBE_EXTRACT_SPEC_v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .templates import BCP_VARIANTS

SPEC_VERSION = "probe_extract_v1"


@dataclass(frozen=True)
class LayerSpec:
    names: tuple[str, ...]
    indices: tuple[int, ...]


def resolve_layers(n_layers: int) -> LayerSpec:
    last = n_layers - 1
    return LayerSpec(
        names=("L4", "L2", "L"),
        indices=(last - 5, last - 3, last),
    )


def bank_dir(out_dir: Path) -> Path:
    return out_dir / "feature_bank"


def alloc_bcp_arrays(n: int, n_layer_slots: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros((n, n_layer_slots, dim), dtype=np.float32)
    last = np.zeros((n, n_layer_slots, dim), dtype=np.float32)
    return mean, last


def write_bcp_slice(
    bank: Path,
    variant: str,
    mean: np.ndarray,
    last: np.ndarray,
) -> None:
    bank.mkdir(parents=True, exist_ok=True)
    np.save(bank / f"bcp_{variant}_mean.npy", mean)
    np.save(bank / f"bcp_{variant}_last.npy", last)


def load_manifest(bank: Path) -> dict[str, Any] | None:
    p = bank / "manifest.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_manifest(bank: Path, payload: dict[str, Any]) -> None:
    bank.mkdir(parents=True, exist_ok=True)
    (bank / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_row_index(bank: Path, df: pd.DataFrame) -> None:
    cols = [
        "claim_id",
        "group_id",
        "split",
        "y_pm",
        "y_com",
        "y_cp",
        "primary_support_label",
        "oc_eligible",
    ]
    with (bank / "row_index.jsonl").open("w", encoding="utf-8") as f:
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            rec: dict[str, Any] = {"row_idx": row_idx}
            for c in cols:
                if c in df.columns:
                    v = row[c]
                    if isinstance(v, (list, tuple)):
                        rec[c] = list(v)
                    else:
                        rec[c] = v
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def sample_clean_controls(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    pool = df[
        (df["y_com"] == 0) & (df["primary_support_label"].astype(str) == "correct_binding")
    ].copy()
    if len(pool) <= n:
        return pool
    parts: list[pd.DataFrame] = []
    splits = pool["split"].astype(str).unique().tolist()
    per = max(1, n // max(1, len(splits)))
    remaining = n
    for sp in sorted(splits):
        grp = pool[pool["split"].astype(str) == sp]
        k = min(len(grp), per, remaining)
        if k > 0:
            parts.append(grp.sample(n=k, random_state=seed))
            remaining -= k
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        rest = pool.drop(index=out.index, errors="ignore")
        need = n - len(out)
        if len(rest) > 0 and need > 0:
            out = pd.concat(
                [out, rest.sample(n=min(need, len(rest)), random_state=seed)],
                ignore_index=True,
            )
    return out.head(n)


def load_bcp_matrix(
    bank: Path,
    *,
    variant: str = "full",
    pool: str = "mean",
    layer_slot: int = 2,
) -> np.ndarray:
    if variant not in BCP_VARIANTS:
        raise ValueError(f"variant must be in {BCP_VARIANTS}")
    if pool not in ("mean", "last"):
        raise ValueError("pool must be mean or last")
    arr = np.load(bank / f"bcp_{variant}_{pool}.npy")
    return arr[:, layer_slot, :].astype(np.float64)


def expected_bcp_files(bank: Path) -> list[Path]:
    files = []
    for v in BCP_VARIANTS:
        files.append(bank / f"bcp_{v}_mean.npy")
        files.append(bank / f"bcp_{v}_last.npy")
    return files


def validate_smoke_bank(bank: Path, n: int, dim: int, n_layers: int = 3) -> None:
    for p in expected_bcp_files(bank):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")
        arr = np.load(p)
        if arr.shape != (n, n_layers, dim):
            raise ValueError(f"Bad shape {p}: {arr.shape} expected ({n},{n_layers},{dim})")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {p}")
