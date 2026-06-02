"""Lightweight stats for P3 mechanism tables."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_array(vals: list[float]) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def mean_ci(vals: list[float], *, alpha: float = 0.05, n_boot: int = 2000, seed: int = 42) -> dict[str, float | None]:
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    arr = _as_array(vals)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(samp.mean()))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": int(len(arr))}


def compare_groups(a: list[float], b: list[float]) -> dict[str, Any]:
    """a=COM, b=clean (typical). Returns mean diff + optional Mann-Whitney."""
    out: dict[str, Any] = {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)) if a else None,
        "mean_b": float(np.mean(b)) if b else None,
        "mean_diff_a_minus_b": (float(np.mean(a)) - float(np.mean(b))) if a and b else None,
    }
    if a and b:
        try:
            from scipy.stats import mannwhitneyu  # type: ignore

            u, p = mannwhitneyu(a, b, alternative="two-sided")
            out["mannwhitney_u"] = float(u)
            out["mannwhitney_p"] = float(p)
        except Exception:
            out["mannwhitney_note"] = "scipy unavailable"
    return out
