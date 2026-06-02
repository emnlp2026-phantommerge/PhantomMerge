"""Paper Fig.2 — failure decomposition (one PDF per panel)."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from palette import (
    C_BASE,
    C_SOLID,
    CHART_FILL_ALPHA,
    MAX_BY_COL,
    RC_PARAMS,
    TRAJ_COLS,
    TRAJ_PCT,
    blend_with_white,
    heatmap_cell_alpha,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(RC_PARAMS)

COHORT_ROWS = [
    ("Shopping", "Qwen"),
    ("Shopping", "Mistral"),
    ("FHIR", "Qwen"),
    ("FHIR", "Mistral"),
]
N_COHORTS = len(COHORT_ROWS)


def cohort_labels() -> list[str]:
    return [f"{a}\n{b}" for a, b in COHORT_ROWS]


traj_cnt = np.array([
    [45, 40, 5, 8],
    [49, 42, 4, 10],
    [336, 317, 223, 1],
    [422, 373, 296, 15],
])

claim_labels = ["CAP", "CEM", "ASF"]
claim_counts = np.array([
    [53, 7, 9],
    [52, 7, 12],
    [175, 320, 1],
    [226, 412, 16],
], dtype=float)
claim_pct = claim_counts / claim_counts.sum(axis=1, keepdims=True) * 100.0

# Stacked bars (panel b): inter-bar gap = BAR_H / 3
BAR_H = 0.56
BAR_GAP_FRAC = 1 / 3
BAR_PITCH = BAR_H * (1 + BAR_GAP_FRAC)


def panel_b_y_centers() -> np.ndarray:
    return np.arange(N_COHORTS) * BAR_PITCH


def _save(fig, name: str, *, pad_inches: float = 0.015) -> None:
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved {out}")


def _legend_below_axes(fig, ax, *, dy: float = 0.02) -> None:
    """Place legend in figure coords just under the axes box (not inside bars)."""
    fig.canvas.draw()
    pos = ax.get_position()
    x_center = (pos.x0 + pos.x1) / 2
    y = pos.y0 - dy
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(x_center, y),
        bbox_transform=fig.transFigure,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    for text in leg.get_texts():
        text.set_color(C_BASE["text"])


# gap between axes left edge and label column (figure fraction)
_LABEL_FIG_PAD = 0.006


def _label_column_x(fig, ax, renderer) -> float:
    """Figure-fraction x: left edge of label column; right edge clears axes."""
    font = FontProperties(family=RC_PARAMS["font.family"], size=7)
    max_w_px = 0.0
    for lab in cohort_labels():
        for line in lab.split("\n"):
            w, _h, _d = renderer.get_text_width_height_descent(line, font, ismath=False)
            max_w_px = max(max_w_px, w)
    pos = ax.get_position()
    return pos.x0 - max_w_px / fig.bbox.width - _LABEL_FIG_PAD


def _place_cohort_ylabels(fig, ax, y_centers) -> None:
    """Two-line labels left-aligned, fully left of axes (figure+data transform)."""
    ax.set_yticks(y_centers)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = _label_column_x(fig, ax, renderer)
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for y, lab in zip(y_centers, cohort_labels()):
        ax.text(
            x_left,
            y,
            lab,
            transform=trans,
            ha="left",
            va="center",
            fontsize=7,
            color=C_BASE["text"],
            clip_on=False,
        )


def _legend_align_panel_a_xticks(fig, ax_b, ax_a) -> None:
    """Wide layout: legend row at same height as panel (a) x-axis labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabels = ax_a.get_xticklabels()
    if xlabels:
        ext = xlabels[0].get_window_extent(renderer)
        y_fig = 0.5 * (ext.y0 + ext.y1) / fig.bbox.height
    else:
        pos_b = ax_b.get_position()
        y_fig = pos_b.y0 - 0.02

    pos_b = ax_b.get_position()
    x_fig = (pos_b.x0 + pos_b.x1) / 2
    handles, labels = ax_b.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=3,
        loc="center",
        bbox_to_anchor=(x_fig, y_fig),
        bbox_transform=fig.transFigure,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    for text in leg.get_texts():
        text.set_color(C_BASE["text"])


def draw_panel_a(ax):
    ax.set_xlim(0, len(TRAJ_COLS))
    ax.set_ylim(0, N_COHORTS)
    ax.invert_yaxis()

    for i in range(N_COHORTS):
        for j, col in enumerate(TRAJ_COLS):
            val = TRAJ_PCT[i, j]
            cnt = int(traj_cnt[i, j])
            alpha = heatmap_cell_alpha(val, MAX_BY_COL[j])
            face = blend_with_white(C_BASE[col], alpha)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.0))
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.1f}%\n({cnt})",
                ha="center",
                va="center",
                fontsize=7,
                color=C_BASE["text"],
                fontweight="bold" if col == "PM" else "normal",
            )

    for j in range(len(TRAJ_COLS) + 1):
        ax.plot([j, j], [0, N_COHORTS], color="white", lw=1)
    for i in range(N_COHORTS + 1):
        ax.plot([0, len(TRAJ_COLS)], [i, i], color="white", lw=1)

    ax.set_xticks(np.arange(len(TRAJ_COLS)) + 0.5)
    ax.set_xticklabels(TRAJ_COLS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_b(ax):
    y = panel_b_y_centers()
    left = np.zeros(N_COHORTS)
    bar_h = BAR_H
    pad = BAR_H * 0.12

    for lab in claim_labels:
        vals = claim_pct[:, claim_labels.index(lab)]
        color = C_SOLID[lab]
        ax.barh(
            y,
            vals,
            left=left,
            height=bar_h,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=lab,
            alpha=1.0,
        )
        for i, v in enumerate(vals):
            # Only label segments wide enough to stay inside x∈[0,100] (no outward 2% etc.)
            if v >= 8:
                ax.text(
                    left[i] + v / 2,
                    y[i],
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=C_BASE["text"],
                    fontweight="bold",
                )
        left += vals

    ax.set_xlim(0, 100)
    ax.margins(x=0)
    y_top = -BAR_H / 2 - pad
    y_bot = (N_COHORTS - 1) * BAR_PITCH + BAR_H / 2 + pad
    ax.set_ylim(y_top, y_bot)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)


def make_panel_a():
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_panel_a(ax)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.12)
    _place_cohort_ylabels(fig, ax, np.arange(N_COHORTS) + 0.5)
    _save(fig, "fig4_panel_a_heatmap.pdf")


def make_panel_b():
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_panel_b(ax)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.14)
    _place_cohort_ylabels(fig, ax, panel_b_y_centers())
    _legend_below_axes(fig, ax, dy=0.018)
    _save(fig, "fig4_panel_b_claim_bars.pdf", pad_inches=0.01)


def make_wide():
    """Optional combined figure; legend aligned to panel (a) x-tick row."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.05, 1.35]})
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.14, wspace=0.34)
    _place_cohort_ylabels(fig, axes[0], np.arange(N_COHORTS) + 0.5)
    _place_cohort_ylabels(fig, axes[1], panel_b_y_centers())
    _legend_align_panel_a_xticks(fig, axes[1], axes[0])
    _save(fig, "fig4_failure_decomposition_wide.pdf", pad_inches=0.015)


def main():
    print(f"Chart fill alpha={CHART_FILL_ALPHA}, colors:", {k: C_SOLID[k] for k in ("CAP", "CEM", "ASF")})
    make_panel_a()
    make_panel_b()
    make_wide()


if __name__ == "__main__":
    main()
