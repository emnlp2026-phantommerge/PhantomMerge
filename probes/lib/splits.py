"""Grouped train/val/test splits (70/15/15, seed=42)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def grouped_split(
    group_ids: list[str],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    unique = sorted({str(g) for g in group_ids if str(g).strip()})
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
    if n_train + n_val >= n:
        n_val = max(0, min(n_val, n - n_train - 1))
    train_ids = unique[:n_train]
    val_ids = unique[n_train : n_train + n_val]
    test_ids = unique[n_train + n_val :]
    if not test_ids and val_ids:
        test_ids = [val_ids.pop()]
    elif not test_ids and train_ids:
        test_ids = [train_ids.pop()]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def assign_split(group_id: str, split_map: dict[str, list[str]]) -> str:
    gid = str(group_id)
    for name, ids in split_map.items():
        if gid in ids:
            return name
    return "held_out"


def write_splits(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
