"""P4 mitigation helpers — gating + anchor-only cohort selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PM_FAILURE = frozenset(
    {"cross_object_merge", "constraint_projection", "anchored_hallucination"}
)


def _support_labels(row: dict) -> set[str]:
    raw = row.get("support_labels")
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    try:
        return {str(x) for x in raw}
    except TypeError:
        return set()


def trajectory_has_pm_rows(claim_rows: list[dict]) -> bool:
    for row in claim_rows:
        if int(row.get("y_pm", 0) or 0) == 1:
            return True
        if _support_labels(row) & PM_FAILURE:
            return True
    return False


def pm_claim_count(claim_rows: list[dict]) -> int:
    n = sum(1 for row in claim_rows if int(row.get("y_pm", 0) or 0) == 1)
    if n:
        return n
    return sum(1 for row in claim_rows if _support_labels(row) & PM_FAILURE)


def load_probe_split_map(splits_path: Path) -> dict[str, str]:
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for split, ids in (data.get("groups") or {}).items():
        for gid in ids:
            out[str(gid)] = str(split)
    return out


def select_mitigation_trajectories(
    per_rows: list[dict],
    *,
    max_cases: int = 50,
    prefer: str = "com_or_cp",
    split_filter: str | None = None,
    split_map: dict[str, str] | None = None,
) -> list[dict]:
    """Pick trajectories for anchor-only mitigation.

    prefer:
      - com_or_cp: legacy pilot (hardest COM/CP)
      - pm: all trajectories with has_phantom_merge
      - pm_test: PM trajectories in probe test split (paper-aligned)
    """
    scored: list[tuple[float, dict]] = []
    for row in per_rows:
        tl = row.get("trajectory_labels") or {}
        has_com = bool(tl.get("has_cross_object_merge"))
        has_cp = bool(tl.get("has_constraint_projection"))
        has_pm = bool(tl.get("has_phantom_merge"))
        gid = str(row.get("question_id") or row.get("group_id") or "")

        if prefer in ("pm", "pm_test") and not has_pm:
            continue
        if prefer == "pm_test":
            if split_map is None or split_map.get(gid) != "test":
                continue
        if prefer == "com_or_cp" and not (has_com or has_cp):
            continue
        if prefer == "pm" and split_filter and split_map:
            if split_map.get(gid) != split_filter:
                continue

        n_pm_claims = sum(
            1
            for c in row.get("claims") or []
            if set(c.get("support_labels") or []) & PM_FAILURE
        )
        score = (2.0 if has_com else 0.0) + (1.5 if has_cp else 0.0) + n_pm_claims
        scored.append((score, row))
    scored.sort(key=lambda x: -x[0])
    picked = [r for _, r in scored]
    if max_cases > 0:
        picked = picked[:max_cases]
    return picked


def aggregate_anchor_evidence(row: dict) -> str:
    """Legacy: snippets from all baseline claims (may leak other-resource wording)."""
    parts: list[str] = []
    seen: set[str] = set()
    for c in row.get("claims") or []:
        ev = str(c.get("supporting_anchor_evidence") or "").strip()
        if ev and ev not in seen:
            seen.add(ev)
            parts.append(ev)
    return "\n\n".join(parts)


def format_anchor_kb_evidence(
    kb: dict[str, dict],
    anchor_ids: list[str],
) -> str:
    """P4.2 v3: protocol-anchor KB blobs only (no PM-claim excerpts)."""
    parts: list[str] = []
    for aid in anchor_ids:
        aid = str(aid).strip()
        slot = kb.get(aid)
        if not slot:
            continue
        rtype = slot.get("resource_type") or "Resource"
        blob = str(slot.get("evidence_blob") or "").strip()
        parts.append(f"[{rtype}] {aid}\n{blob or '(empty)'}")
    return "\n\n".join(parts) if parts else "(no anchor evidence in KB)"


def build_anchor_only_user_prompt(query: str, anchor_evidence: str) -> str:
    return (
        "[User Query]\n"
        f"{query.strip()}\n\n"
        "[Anchor Evidence Only]\n"
        f"{anchor_evidence.strip() or '(none)'}\n\n"
        "Write a concise clinical answer using ONLY the anchor evidence above.\n"
        "Rules:\n"
        "- Do NOT mention or use other FHIR resources, encounters, or conditions.\n"
        "- If the anchor lacks a slot (e.g. diagnosis), do NOT answer yes/no; state that it is unknown.\n"
        "- Prefer one short paragraph; no lists of other resource IDs."
    )


def gating_curve(
    df: pd.DataFrame,
    prob: np.ndarray,
    *,
    split_name: str,
    taus: list[float],
) -> dict[str, Any]:
    sub = df.copy()
    sub["p_pm"] = prob

    def _records(grp: pd.DataFrame) -> list[dict]:
        rows = []
        for rec in grp.to_dict("records"):
            sl = rec.get("support_labels")
            if sl is not None and not isinstance(sl, list):
                rec["support_labels"] = list(sl)
            rows.append(rec)
        return rows

    baseline_pm: list[int] = []
    baseline_task_pm: list[int] = []
    gated: dict[float, list[int]] = {t: [] for t in taus}
    gated_task_pm: dict[float, list[int]] = {t: [] for t in taus}
    claims_kept: dict[float, list[int]] = {t: [] for t in taus}

    for _, grp in sub.groupby("group_id"):
        rows = _records(grp)
        baseline_pm.append(int(trajectory_has_pm_rows(rows)))
        if rows and int(rows[0].get("task_correct", 0)) == 1:
            baseline_task_pm.append(int(trajectory_has_pm_rows(rows)))
        for tau in taus:
            kept = [r for r in rows if r["p_pm"] <= tau]
            claims_kept[tau].append(len(kept))
            gated[tau].append(int(trajectory_has_pm_rows(kept)))
            if rows and int(rows[0].get("task_correct", 0)) == 1:
                gated_task_pm[tau].append(int(trajectory_has_pm_rows(kept)))

    n = len(baseline_pm) or 1
    n_task = len(baseline_task_pm) or 1
    curve: dict[str, Any] = {
        "split": split_name,
        "n_trajectories": n,
        "n_task_correct_trajectories": n_task,
        "baseline_pm_rate": sum(baseline_pm) / n,
        "baseline_pm_count": sum(baseline_pm),
        "baseline_task_correct_pm_rate": sum(baseline_task_pm) / n_task,
        "tau_sweep": [],
    }
    for tau in taus:
        flags = gated[tau]
        curve["tau_sweep"].append(
            {
                "tau": tau,
                "pm_rate": sum(flags) / n,
                "pm_count": sum(flags),
                "pm_reduction": curve["baseline_pm_rate"] - (sum(flags) / n),
                "claims_retained_mean": float(np.mean(claims_kept[tau])) if claims_kept[tau] else 0.0,
                "task_correct_pm_rate": sum(gated_task_pm[tau]) / n_task if gated_task_pm[tau] else None,
            }
        )
    return curve


def pick_tau_on_val(curve: dict[str, Any], *, target_pm_reduction: float = 0.15) -> float | None:
    """Legacy: smallest tau with >= target PM reduction (can be too aggressive)."""
    base = float(curve.get("baseline_pm_rate") or 0)
    best = None
    for entry in curve.get("tau_sweep") or []:
        tau = float(entry["tau"])
        rate = float(entry["pm_rate"])
        if base - rate >= target_pm_reduction:
            best = tau
            break
    if best is None and curve.get("tau_sweep"):
        entry = min(curve["tau_sweep"], key=lambda e: e["pm_rate"])
        best = float(entry["tau"])
    return best


def pick_tau_balanced(
    curve: dict[str, Any],
    *,
    min_pm_reduction: float = 0.10,
    min_claims_retained: float = 1.5,
    tau_lo: float = 0.5,
    tau_hi: float = 0.7,
) -> float | None:
    """Paper P4 gating: val-only τ in [0.5, 0.7] with PM drop + min claims kept."""
    candidates: list[dict[str, Any]] = []
    for entry in curve.get("tau_sweep") or []:
        tau = float(entry["tau"])
        if tau < tau_lo or tau > tau_hi:
            continue
        if float(entry.get("pm_reduction") or 0) < min_pm_reduction:
            continue
        if float(entry.get("claims_retained_mean") or 0) < min_claims_retained:
            continue
        candidates.append(entry)
    if candidates:
        best = max(candidates, key=lambda e: float(e.get("pm_reduction") or 0))
        return float(best["tau"])
    for entry in curve.get("tau_sweep") or []:
        tau = float(entry["tau"])
        if tau_lo <= tau <= tau_hi:
            return tau
    return pick_tau_on_val(curve, target_pm_reduction=min_pm_reduction)
