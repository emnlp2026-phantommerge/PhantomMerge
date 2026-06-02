"""Label helpers aligned with pm_eval_vnext."""

from __future__ import annotations

PM_FAILURE_LABELS = frozenset(
    {
        "cross_object_merge",
        "constraint_projection",
        "anchored_hallucination",
    }
)


def y_pm_from_labels(support_labels: list[str], primary: str) -> int:
    primary = str(primary or "").strip()
    labels = {str(x).strip() for x in (support_labels or [])}
    if primary in PM_FAILURE_LABELS:
        return 1
    return int(bool(labels & PM_FAILURE_LABELS))


def y_com(support_labels: list[str]) -> int:
    return int("cross_object_merge" in (support_labels or []))


def y_cp(support_labels: list[str]) -> int:
    return int("constraint_projection" in (support_labels or []))


def trajectory_decomposition(traj_labels: dict) -> dict[str, bool]:
    """Derived trajectory buckets (PM_PROBE_PLAN §1.4)."""
    com = bool(traj_labels.get("has_cross_object_merge"))
    cp = bool(traj_labels.get("has_constraint_projection"))
    hall = bool(traj_labels.get("has_anchored_hallucination"))
    return {
        "has_com_only": com and not cp and not hall,
        "has_cp_only": cp and not com and not hall,
        "has_com_and_cp": com and cp,
        "has_anchored_hall_only": hall and not com and not cp,
        "has_phantom_merge": bool(traj_labels.get("has_phantom_merge")),
    }
