"""Paper Fig.4 — mitigation (one PDF per panel / variant)."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import C_BASE, C_SOLID, CHART_FILL_ALPHA, RC_PARAMS, blend_with_white

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(RC_PARAMS)

N = 146
LADDER_LINE = "#9AA3AA"
# Same ASF base as stacked bars; bar ~8% lighter, point uses higher α for contrast
LADDER_BAR = blend_with_white(C_BASE["ASF"], CHART_FILL_ALPHA * 0.90)
LADDER_POINT = C_BASE["ASF"]

ladder = pd.DataFrame([
    {"method": "Baseline", "pm": 52.0},
    {"method": "PGCS", "pm": 26.0},
    {"method": "Random", "pm": 16.6},
    {"method": "BCP", "pm": 14.0},
    {"method": "MSPS", "pm": 7.0},
    {"method": "Oracle", "pm": 0.0},
])
ladder["pm_rate"] = ladder["pm"] / N * 100.0

TAU_PATHS = [
    os.path.join(REPO_ROOT, "results/appendix/mitigation_tau_pm_curve.csv"),
    os.path.join(REPO_ROOT, "runs/probe/fhir_qwen_vnext/appendix_tables/mitigation_tau_pm_curve.csv"),
]


def find_tau():
    for p in TAU_PATHS:
        if os.path.exists(p):
            return p
    return None


def normalize_tau(df):
    cols = {c.lower(): c for c in df.columns}
    method_col = next((cols[k] for k in ["method", "filter", "name"] if k in cols), None)
    tau_col = next((cols[k] for k in ["tau", "threshold", "gamma"] if k in cols), None)
    pm_col = next((cols[k] for k in ["pm_count", "pm", "n_pm", "pm_trajectories"] if k in cols), None)
    rate_col = next((cols[k] for k in ["pm_rate", "rate"] if k in cols), None)
    if method_col is None or tau_col is None:
        raise ValueError(f"Cannot find method/tau columns in {df.columns.tolist()}")

    out = pd.DataFrame({"method": df[method_col].astype(str), "tau": df[tau_col].astype(float)})
    if pm_col is not None:
        out["pm_count"] = df[pm_col].astype(float)
        out["pm_rate"] = out["pm_count"] / N * 100
    elif rate_col is not None:
        out["pm_rate"] = df[rate_col].astype(float)
        if out["pm_rate"].max() <= 1.0:
            out["pm_rate"] *= 100
    else:
        raise ValueError(f"Cannot find pm count/rate columns in {df.columns.tolist()}")

    split_col = cols.get("split")
    if split_col is not None:
        out = out[df[split_col].astype(str).str.lower() == "test"].copy()

    out["method_norm"] = out["method"].str.upper().apply(
        lambda x: "MSPS" if "MSPS" in x else ("BCP" if "BCP" in x else x)
    )
    return out


def _ladder_annotations(ax, x, y, rates):
    for i, (xi, yi, rate) in enumerate(zip(x, y, rates)):
        label = f"{int(round(yi))}" if ladder.iloc[i]["method"] != "Random" else "16.6"
        ax.text(
            xi,
            yi + 3.0,
            f"{label}\n({rate:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=C_BASE["text"],
        )


def _style_ladder_axes(ax, x):
    ax.set_xticks(x)
    ax.set_xticklabels(ladder["method"], rotation=20, ha="right")
    ax.set_ylim(-2, 60)
    ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9AA3AA")
    ax.spines["bottom"].set_color("#9AA3AA")


def draw_ladder_line(ax):
    x = np.arange(len(ladder))
    y = ladder["pm"].values
    rates = ladder["pm_rate"].values

    ax.plot(x, y, color=LADDER_LINE, linewidth=1.0, zorder=1)
    ax.scatter(
        x,
        y,
        s=50,
        color=LADDER_POINT,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    _ladder_annotations(ax, x, y, rates)
    _style_ladder_axes(ax, x)


def draw_ladder_combo(ax):
    """Bars = PM count (light ASF); gray trend line + darker red markers at bar tops."""
    x = np.arange(len(ladder))
    y = ladder["pm"].values
    rates = ladder["pm_rate"].values

    ax.bar(
        x,
        y,
        width=0.58,
        color=LADDER_BAR,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.plot(x, y, color=LADDER_LINE, linewidth=1.0, zorder=3)
    ax.scatter(
        x,
        y,
        s=50,
        color=LADDER_POINT,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )

    _ladder_annotations(ax, x, y, rates)
    _style_ladder_axes(ax, x)


def draw_tau(ax):
    tau_path = find_tau()
    if tau_path is None:
        points = pd.DataFrame([
            {"method_norm": "BCP", "tau": 0.50, "pm_rate": 9.6},
            {"method_norm": "MSPS", "tau": 0.45, "pm_rate": 4.8},
        ])
        for _, row in points.iterrows():
            color = C_SOLID["bcp"] if row["method_norm"] == "BCP" else C_SOLID["msps"]
            ax.scatter(row["tau"], row["pm_rate"], s=55, color=color, edgecolor="white", linewidth=1.0)
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(0, 40)
    else:
        df = normalize_tau(pd.read_csv(tau_path))
        print(f"Loaded tau curve from {tau_path}")
        for method, color in [("BCP", C_SOLID["bcp"]), ("MSPS", C_SOLID["msps"])]:
            sub = df[df["method_norm"] == method].sort_values("tau")
            if sub.empty:
                continue
            ax.plot(
                sub["tau"],
                sub["pm_rate"],
                marker="o",
                markersize=3.8,
                linewidth=1.4,
                color=color,
                label=method,
                alpha=1.0,
            )
        ax.axvline(0.50, color=C_SOLID["bcp"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0.45, color=C_SOLID["msps"], linestyle="--", linewidth=0.8, alpha=0.7)

    ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9AA3AA")
    ax.spines["bottom"].set_color("#9AA3AA")

    leg = ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    for text in leg.get_texts():
        text.set_color(C_BASE["text"])


def _save(fig, name: str, bottom: float = 0.18) -> None:
    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=bottom)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print(f"Saved {out}")


def make_ladder_line():
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_ladder_line(ax)
    _save(fig, "fig6_panel_a_mitigation_ladder_line.pdf", bottom=0.22)


def make_ladder_combo():
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_ladder_combo(ax)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.22)
    for name in ("fig6_panel_a_mitigation_ladder.pdf", "fig6_panel_a_mitigation_ladder_combo.pdf"):
        out = os.path.join(OUT_DIR, name)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.012)
        print(f"Saved {out}")
    plt.close(fig)


def make_tau():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_tau(ax)
    _save(fig, "fig6_panel_b_mitigation_tau.pdf", bottom=0.28)


def main():
    print("Ladder bar:", LADDER_BAR, "| point:", LADDER_POINT, "| line:", LADDER_LINE)
    make_ladder_combo()
    make_ladder_line()
    make_tau()


if __name__ == "__main__":
    main()
