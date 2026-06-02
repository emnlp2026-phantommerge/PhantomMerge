"""Claim-level mitigation audit: CB retention, random retention-matched controls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from lib.mitigation import PM_FAILURE, trajectory_has_pm_rows


def is_correct_binding_claim(row: dict) -> bool:
    """Judge-frozen CB: primary stratum correct_binding and not trajectory PM claim."""
    if str(row.get("primary_support_label", "")) != "correct_binding":
        return False
    if int(row.get("y_pm", 0) or 0) == 1:
        return False
    if set(row.get("support_labels") or []) & PM_FAILURE:
        return False
    return True


def is_pm_claim(row: dict) -> bool:
    if int(row.get("y_pm", 0) or 0) == 1:
        return True
    labels = row.get("support_labels")
    if labels is None:
        return False
    if isinstance(labels, str):
        return labels in PM_FAILURE
    return bool(set(labels) & PM_FAILURE)


def _records(grp: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for rec in grp.to_dict("records"):
        sl = rec.get("support_labels")
        if sl is not None and not isinstance(sl, list):
            rec["support_labels"] = list(sl)
        rows.append(rec)
    return rows


@dataclass
class ClaimAuditResult:
    n_trajectories: int
    baseline_pm_count: int
    baseline_pm_rate: float
    gated_pm_count: int
    gated_pm_rate: float
    claims_total_baseline: int
    claims_total_gated: int
    claims_retained_mean: float
    pm_claims_total_baseline: int
    pm_claims_total_gated: int
    cb_claims_total_baseline: int
    cb_claims_retained: int
    cb_claims_removed: int
    cb_retention_rate: float
    removed_cb_fraction_of_all_removed: float
    pm_claims_removed: int
    non_cb_non_pm_removed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trajectories": self.n_trajectories,
            "baseline_pm_count": self.baseline_pm_count,
            "baseline_pm_rate": self.baseline_pm_rate,
            "gated_pm_count": self.gated_pm_count,
            "gated_pm_rate": self.gated_pm_rate,
            "claims_total_baseline": self.claims_total_baseline,
            "claims_total_gated": self.claims_total_gated,
            "claims_retained_mean": self.claims_retained_mean,
            "pm_claims_total_baseline": self.pm_claims_total_baseline,
            "pm_claims_total_gated": self.pm_claims_total_gated,
            "cb_claims_total_baseline": self.cb_claims_total_baseline,
            "cb_claims_retained": self.cb_claims_retained,
            "cb_claims_removed": self.cb_claims_removed,
            "cb_retention_rate": self.cb_retention_rate,
            "removed_cb_fraction_of_all_removed": self.removed_cb_fraction_of_all_removed,
            "pm_claims_removed": self.pm_claims_removed,
            "non_cb_non_pm_removed": self.non_cb_non_pm_removed,
        }


def audit_gating(
    df: pd.DataFrame,
    drop_fn: Callable[[dict], bool],
) -> ClaimAuditResult:
    """Evaluate trajectory PM + claim retention under per-claim drop_fn(row)->drop."""
    baseline_pm: list[int] = []
    gated_pm: list[int] = []
    claims_kept: list[int] = []
    claims_base = claims_gated = 0
    pm_base = pm_gated = 0
    cb_base = cb_ret = cb_rem = 0
    pm_removed = non_cb_removed = 0

    for _, grp in df.groupby("group_id"):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        claims_base += len(rows)
        pm_base += sum(1 for r in rows if is_pm_claim(r))
        cb_base += sum(1 for r in rows if is_correct_binding_claim(r))

        kept: list[dict] = []
        for r in rows:
            if drop_fn(r):
                if is_pm_claim(r):
                    pm_removed += 1
                elif is_correct_binding_claim(r):
                    cb_rem += 1
                else:
                    non_cb_removed += 1
            else:
                kept.append(r)
                if is_correct_binding_claim(r):
                    cb_ret += 1

        claims_kept.append(len(kept))
        claims_gated += len(kept)
        gated_pm.append(int(trajectory_has_pm_rows(kept)))
        pm_gated += sum(1 for r in kept if is_pm_claim(r))

    n = len(baseline_pm) or 1
    removed_total = pm_removed + cb_rem + non_cb_removed
    return ClaimAuditResult(
        n_trajectories=n,
        baseline_pm_count=sum(baseline_pm),
        baseline_pm_rate=sum(baseline_pm) / n,
        gated_pm_count=sum(gated_pm),
        gated_pm_rate=sum(gated_pm) / n,
        claims_total_baseline=claims_base,
        claims_total_gated=claims_gated,
        claims_retained_mean=float(np.mean(claims_kept)) if claims_kept else 0.0,
        pm_claims_total_baseline=pm_base,
        pm_claims_total_gated=pm_gated,
        cb_claims_total_baseline=cb_base,
        cb_claims_retained=cb_ret,
        cb_claims_removed=cb_rem,
        cb_retention_rate=(cb_ret / cb_base) if cb_base else 1.0,
        removed_cb_fraction_of_all_removed=(cb_rem / removed_total) if removed_total else 0.0,
        pm_claims_removed=pm_removed,
        non_cb_non_pm_removed=non_cb_removed,
    )


def kept_counts_per_group(df: pd.DataFrame, drop_fn: Callable[[dict], bool]) -> dict[str, int]:
    out: dict[str, int] = {}
    for gid, grp in df.groupby("group_id"):
        rows = _records(grp)
        out[str(gid)] = sum(1 for r in rows if not drop_fn(r))
    return out


def random_retention_matched_eval(
    df: pd.DataFrame,
    target_kept: dict[str, int],
    *,
    n_seeds: int = 200,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Randomly drop claims to match MSPS per-trajectory retention counts."""
    pm_rates: list[float] = []
    pm_counts: list[int] = []
    cb_removed_fracs: list[float] = []

    for seed in range(base_seed, base_seed + n_seeds):
        rng = np.random.default_rng(seed)

        # Build per-trajectory drop masks
        frames: list[pd.DataFrame] = []
        for gid, grp in df.groupby("group_id"):
            rows = _records(grp)
            k = int(target_kept.get(str(gid), len(rows)))
            n = len(rows)
            if n == 0:
                continue
            drop_mask = np.zeros(n, dtype=bool)
            if k < n:
                drop_idx = rng.choice(n, size=n - k, replace=False)
                drop_mask[drop_idx] = True
            sub = grp.copy()
            sub["_random_drop"] = drop_mask
            frames.append(sub)

        if not frames:
            continue
        sub_df = pd.concat(frames, ignore_index=True)

        def _drop(row: dict) -> bool:
            return bool(row.get("_random_drop", False))

        audit = audit_gating(sub_df, _drop)
        pm_rates.append(audit.gated_pm_rate)
        pm_counts.append(audit.gated_pm_count)
        cb_removed_fracs.append(audit.removed_cb_fraction_of_all_removed)

    n = len(pm_rates) or 1
    arr = np.array(pm_rates)
    return {
        "n_seeds": n_seeds,
        "pm_rate_mean": float(arr.mean()),
        "pm_rate_std": float(arr.std()),
        "pm_rate_median": float(np.median(arr)),
        "pm_rate_p05": float(np.percentile(arr, 5)),
        "pm_rate_p95": float(np.percentile(arr, 95)),
        "pm_count_mean": float(np.mean(pm_counts)),
        "pm_count_std": float(np.std(pm_counts)),
        "removed_cb_fraction_mean": float(np.mean(cb_removed_fracs)),
        "target_retention_source": "msps_tau0.45_per_trajectory",
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
