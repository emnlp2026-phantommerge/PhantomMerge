"""Paper figure palette.

Heatmap (Fig.2a): four base hues, alpha scales with cell %.
All other charts: same four base hues + one unified alpha (FHIR Qwen PM/CAP/CEM mean).
"""

from __future__ import annotations

import numpy as np

# Four base hues — heatmap uses these with per-cell alpha
C_BASE = {
    "PM": "#6F8FAF",
    "CAP": "#9BB0A5",
    "CEM": "#D9B48F",
    "ASF": "#C98A7D",
    "text": "#24323C",
    "grid": "#D8DEE5",
    "bg": "#FFFFFF",
    "light": "#F7F5F0",
}

HEATMAP_ALPHA_MIN = 0.18
HEATMAP_ALPHA_SPAN = 0.72

TRAJ_COLS = ["PM", "CAP", "CEM", "ASF"]
TRAJ_PCT = np.array([
    [18.1, 16.1, 2.0, 3.2],
    [19.6, 16.8, 1.6, 4.0],
    [34.5, 32.6, 22.9, 0.1],
    [49.8, 44.0, 34.9, 1.8],
])
FHIR_QWEN_ROW = 2
MAX_BY_COL = TRAJ_PCT.max(axis=0)


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def blend_with_white(hex_color: str, alpha: float) -> str:
    rgb = hex_to_rgb01(hex_color)
    white = np.ones(3)
    out = white * (1 - alpha) + rgb * alpha
    return "#{:02x}{:02x}{:02x}".format(
        int(round(out[0] * 255)),
        int(round(out[1] * 255)),
        int(round(out[2] * 255)),
    )


def heatmap_cell_alpha(value_pct: float, col_max: float) -> float:
    if col_max <= 0:
        return HEATMAP_ALPHA_MIN
    return HEATMAP_ALPHA_MIN + HEATMAP_ALPHA_SPAN * (value_pct / col_max)


def fhir_qwen_heatmap_alpha(label: str) -> float:
    j = TRAJ_COLS.index(label)
    return heatmap_cell_alpha(TRAJ_PCT[FHIR_QWEN_ROW, j], MAX_BY_COL[j])


# Unified α: mean of FHIR Qwen blue/green/orange cells (PM, CAP, CEM) — one opacity for all bar/box plots
_fhir_qwen_main = [fhir_qwen_heatmap_alpha(k) for k in ("PM", "CAP", "CEM")]
CHART_FILL_ALPHA = round(float(np.mean(_fhir_qwen_main)), 4)

C_SOLID = {
    k: blend_with_white(C_BASE[k], CHART_FILL_ALPHA) for k in ("PM", "CAP", "CEM", "ASF")
}
C_SOLID["blue"] = C_SOLID["PM"]
C_SOLID["green"] = C_SOLID["CAP"]
C_SOLID["orange"] = C_SOLID["CEM"]
C_SOLID["red"] = C_SOLID["ASF"]
C_SOLID["Clean"] = C_SOLID["CAP"]
C_SOLID["CEM"] = C_SOLID["CEM"]
C_SOLID["baseline"] = C_SOLID["ASF"]
C_SOLID["bcp"] = C_SOLID["PM"]
C_SOLID["msps"] = C_SOLID["CAP"]

SCATTER_ALPHA = 0.55

RC_PARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}
