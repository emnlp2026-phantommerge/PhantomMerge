"""Paper Fig.3 — OC margins (one PDF per context panel)."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette import C_BASE, C_SOLID, RC_PARAMS, SCATTER_ALPHA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(RC_PARAMS)

RAW_PATHS = [
    os.path.join(REPO_ROOT, "results/appendix/oc_margin_plot.csv"),
    os.path.join(REPO_ROOT, "runs/probe/fhir_qwen_vnext/appendix_tables/oc_margin_plot.csv"),
]

SUMMARY = pd.DataFrame([
    {"context": "Evidence-only", "group": "CEM", "margin": 0.140},
    {"context": "Evidence-only", "group": "Clean", "margin": -0.287},
    {"context": "Final-prefix", "group": "CEM", "margin": 0.178},
    {"context": "Final-prefix", "group": "Clean", "margin": -0.269},
])


def find_raw():
    for p in RAW_PATHS:
        if os.path.exists(p):
            return p
    return None


def _label_to_group(s):
    x = str(s).strip().lower()
    if x in ("com", "cem", "1", "true"):
        return "CEM"
    return "Clean"


def normalize_raw(df):
    cols = {c.lower(): c for c in df.columns}
    ev_col = cols.get("margin_evidence_only")
    fin_col = cols.get("margin_final_prefix")
    if ev_col and fin_col:
        label_col = cols.get("label") or cols.get("primary_label")
        if label_col is None:
            raise ValueError(f"Cannot infer group from {df.columns.tolist()}")
        grp = df[label_col].map(_label_to_group)
        parts = [
            pd.DataFrame({"context": "Evidence-only", "group": grp, "margin": df[ev_col].astype(float)}),
            pd.DataFrame({"context": "Final-prefix", "group": grp, "margin": df[fin_col].astype(float)}),
        ]
        return pd.concat(parts, ignore_index=True)

    margin_col = None
    for k in ["margin", "oc_margin", "owner_contrastive_margin", "m"]:
        if k in cols:
            margin_col = cols[k]
            break
    if margin_col is None:
        raise ValueError(f"Cannot find margin column in {df.columns.tolist()}")

    out = pd.DataFrame({"margin": df[margin_col].astype(float), "context": "Final-prefix"})
    label_col = cols.get("label") or cols.get("primary_label")
    if label_col:
        out["group"] = df[label_col].map(_label_to_group)
    return out[["context", "group", "margin"]]


def draw_box_strip(ax, data, context: str):
    sub = data[data["context"].str.lower().str.replace("_", "-").str.contains(
        "evidence" if context == "Evidence-only" else "final"
    )]
    if sub.empty:
        sub = SUMMARY[SUMMARY["context"] == context]

    groups = ["Clean", "CEM"]
    xpos = [0, 1]
    colors = [C_SOLID["Clean"], C_SOLID["CEM"]]

    for x, g, color in zip(xpos, groups, colors):
        vals = sub[sub["group"] == g]["margin"].astype(float).values
        if len(vals) > 2:
            rng = np.random.default_rng(7)
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), x) + jitter,
                vals,
                s=9,
                alpha=SCATTER_ALPHA,
                color=color,
                edgecolor="none",
            )
            ax.boxplot(
                vals,
                positions=[x],
                widths=0.36,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=SCATTER_ALPHA, color=color, linewidth=1),
                medianprops=dict(color=C_BASE["text"], linewidth=1.2),
                whiskerprops=dict(color=color, linewidth=1, alpha=SCATTER_ALPHA),
                capprops=dict(color=color, linewidth=1, alpha=SCATTER_ALPHA),
                flierprops=dict(marker="", markersize=0),
            )
        else:
            ax.scatter([x], [vals.mean()], s=45, color=color, alpha=SCATTER_ALPHA, edgecolor="white", linewidth=0.8)

        mean = vals.mean()
        ax.text(
            x,
            mean + (0.055 if mean >= 0 else -0.075),
            f"{mean:+.3f}",
            ha="center",
            va="center",
            fontsize=7,
            color=C_BASE["text"],
        )

    ax.axhline(0, color="#7B848C", linewidth=0.9, linestyle="--")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9AA3AA")
    ax.spines["bottom"].set_color("#9AA3AA")


def make_one(context: str, out_name: str, data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(2.0, 1.85))
    fig.patch.set_facecolor(C_BASE["bg"])
    draw_box_strip(ax, data, context)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.98, bottom=0.18)
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    raw = find_raw()
    if raw:
        data = normalize_raw(pd.read_csv(raw))
        print(f"Loaded raw OC margins from {raw}")
    else:
        data = SUMMARY
        print("Raw OC file not found; using summary means only.")

    make_one("Evidence-only", "fig5_panel_a_oc_margin_evidence.pdf", data)
    make_one("Final-prefix", "fig5_panel_b_oc_margin_final.pdf", data)


if __name__ == "__main__":
    main()
