"""Appendix figures — reads sealed CSV/JSON under results/appendix/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from palette import (
    C_BASE,
    C_SOLID,
    CHART_FILL_ALPHA,
    RC_PARAMS,
    SCATTER_ALPHA,
    blend_with_white,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EXPORT_DIRS = [
    REPO_ROOT / "results" / "appendix",
    REPO_ROOT / "runs" / "probe" / "fhir_qwen_vnext" / "appendix_tables",
]
OUT_DIR = SCRIPT_DIR / "output" / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(RC_PARAMS)

# Legacy alias map (appendix labels → palette keys)
COLORS = {
    "PM": C_BASE["PM"],
    "CAP": C_BASE["CAP"],
    "CEM": C_BASE["CEM"],
    "ASF": C_BASE["ASF"],
    "PH": C_BASE["light"],
    "CB": blend_with_white(C_BASE["CAP"], CHART_FILL_ALPHA * 0.85),
    "GREY": "#7C8790",
    "LIGHT_GREY": C_BASE["grid"],
    "TEXT": C_BASE["text"],
}


def find_file(*names: str) -> Path:
    for name in names:
        for base in EXPORT_DIRS:
            p = base / name
            if p.is_file():
                return p
        for p in (Path(name), SCRIPT_DIR / name, Path("data") / name):
            if p.is_file():
                return p
    raise FileNotFoundError(f"Cannot find any of: {names}")


def savefig(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor=C_BASE["bg"])
    plt.close(fig)
    print(f"saved: {out}")


def cohort_label(cohort_key: str) -> str:
    return {
        "shopping_qwen_n249": "Shopping\nQwen",
        "shopping_mistral_n250": "Shopping\nMistral",
        "fhir_qwen_n973": "FHIR\nQwen",
        "fhir_mistral_n847": "FHIR\nMistral",
    }.get(cohort_key, cohort_key)


def _load_mitigation_reference_rates() -> dict[str, float]:
    """PM rates (%) on FHIR probe test split from mitigation_table_extended."""
    path = find_file("mitigation_table_extended.json", "mitigation_table_extended.csv")
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    df = df[df["split"] == "test"].copy()

    def rate_pct(method_prefix: str) -> float:
        sub = df[df["method"].astype(str).str.startswith(method_prefix)]
        if sub.empty:
            raise KeyError(f"No row for method prefix {method_prefix!r}")
        row = sub.iloc[0]
        if "pm_rate" in row and pd.notna(row.get("pm_rate")):
            return float(row["pm_rate"]) * 100.0
        if "pm_rate_mean" in row and pd.notna(row.get("pm_rate_mean")):
            return float(row["pm_rate_mean"]) * 100.0
        raise KeyError(f"No pm_rate on row {method_prefix}")

    return {
        "baseline": rate_pct("baseline"),
        "msps": rate_pct("MSPS"),
        "bcp": rate_pct("BCP"),
    }


def _load_random_matched_stats() -> dict:
    path = find_file(
        "mitigation_random_retention_matched.json",
        "mitigation_random_retention_matched(1).json",
    )
    return json.loads(path.read_text(encoding="utf-8"))


# =========================
# Figure A1: Task correctness × PM outcome heatmap
# =========================


def make_task_pm_heatmap() -> None:
    df = pd.read_csv(find_file("task_pm_heatmap.csv", "task_pm_heatmap(1).csv"))

    cohort_order = [
        "shopping_qwen_n249",
        "shopping_mistral_n250",
        "fhir_qwen_n973",
        "fhir_mistral_n847",
    ]

    columns = [
        ("TC clean", ["task_correct__binding_clean_or_correct"]),
        ("TC PM", ["task_correct__phantom_merge"]),
        ("TC PH", ["task_correct__pure_hallucination_only"]),
        ("TI clean", ["task_incorrect__binding_clean_or_correct"]),
        ("TI PM", ["task_incorrect__phantom_merge"]),
        ("TI PH", ["task_incorrect__pure_hallucination_only"]),
        ("No check", ["task_correct__no_checkable_claims", "task_incorrect__no_checkable_claims"]),
    ]

    mat_rate = np.zeros((len(cohort_order), len(columns)))
    mat_count = np.zeros_like(mat_rate, dtype=int)

    for i, cohort in enumerate(cohort_order):
        sub = df[df["cohort_key"] == cohort]
        for j, (_, keys) in enumerate(columns):
            rows = sub[sub["cell_key"].isin(keys)]
            mat_rate[i, j] = rows["rate"].sum() if len(rows) else 0.0
            mat_count[i, j] = int(rows["count"].sum()) if len(rows) else 0

    cmap = LinearSegmentedColormap.from_list(
        "pmblue", ["#FFFFFF", blend_with_white(C_BASE["PM"], CHART_FILL_ALPHA)]
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    fig.patch.set_facecolor(C_BASE["bg"])
    im = ax.imshow(mat_rate, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.38, mat_rate.max()))

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], rotation=0)
    ax.set_yticks(np.arange(len(cohort_order)))
    ax.set_yticklabels([cohort_label(c) for c in cohort_order])

    for i in range(mat_rate.shape[0]):
        for j in range(mat_rate.shape[1]):
            rate = mat_rate[i, j] * 100
            count = mat_count[i, j]
            if count == 0:
                text = "–"
            elif rate >= 10:
                text = f"{rate:.0f}%\n({count})"
            else:
                text = f"{rate:.1f}%\n({count})"
            ax.text(j, i, text, ha="center", va="center", color=C_BASE["text"], fontsize=7)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Rate among valid trajectories", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6)

    savefig(fig, "app_task_pm_heatmap.pdf")


# =========================
# Figure A2: FHIR claims per trajectory distribution
# =========================


def make_claim_distribution() -> None:
    df = pd.read_csv(
        find_file(
            "fhir_claims_per_trajectory_distribution.csv",
            "fhir_claims_per_trajectory_distribution(1).csv",
        )
    ).sort_values("n_claims")

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    fig.patch.set_facecolor(C_BASE["bg"])
    bars = ax.bar(
        df["n_claims"].astype(str),
        df["n_trajectories"],
        color=C_SOLID["PM"],
        edgecolor="white",
        linewidth=0.8,
        alpha=1.0,
    )

    ymax = df["n_trajectories"].max()
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + ymax * 0.015,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=C_BASE["text"],
        )

    ax.set_xlabel("Extracted claims per trajectory")
    ax.set_ylabel("Trajectories")
    ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, "app_fhir_claim_count_distribution.pdf")


# =========================
# Figure A3: FHIR primary stratum PM rate
# =========================


def make_fhir_primary_stratum() -> None:
    df = pd.read_csv(find_file("fhir_primary_stratum.csv", "fhir_primary_stratum(1).csv")).copy()
    label_map = {
        "medication_semantic_rivals": "Medication\nsemantic",
        "temporal_multi_instance_rivals": "Temporal\nmulti-instance",
        "numeric_value_rivals": "Numeric\nvalue",
        "encounter_procedure_rivals": "Encounter /\nprocedure",
    }
    df["label"] = df["primary_stratum"].map(label_map).fillna(df["primary_stratum"])
    df = df.sort_values("pm_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    fig.patch.set_facecolor(C_BASE["bg"])
    y = np.arange(len(df))
    ax.barh(
        y,
        df["pm_rate"] * 100,
        color=C_SOLID["CAP"],
        edgecolor="white",
        linewidth=0.8,
        alpha=1.0,
    )

    for ypos, row in enumerate(df.itertuples()):
        ax.text(
            row.pm_rate * 100 + 1.0,
            ypos,
            f"{row.pm_rate * 100:.1f}% ({int(row.pm_count)}/{int(row.n_trajectories)})",
            va="center",
            ha="left",
            fontsize=7,
            color=C_BASE["text"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Trajectory-level PM rate (%)")
    ax.set_xlim(0, max(55, df["pm_rate"].max() * 100 + 14))
    ax.grid(axis="x", color=C_BASE["grid"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, "app_fhir_primary_stratum.pdf")


# =========================
# Figure A4: OC margin detailed distributions
# =========================


def make_oc_margin_detail() -> None:
    df = pd.read_csv(find_file("oc_margin_plot.csv", "oc_margin_plot(1).csv")).copy()
    df["paper_label"] = df["label"].replace({"com": "CEM", "clean": "Clean"})
    label_order = ["Clean", "CEM"]

    contexts = [
        ("margin_evidence_only", "evidence"),
        ("margin_final_prefix", "final"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45), sharey=True)
    fig.patch.set_facecolor(C_BASE["bg"])

    rng = np.random.default_rng(123)
    for ax, (col, _panel) in zip(axes, contexts):
        data = [df[df["paper_label"] == lab][col].dropna().values for lab in label_order]
        positions = np.arange(len(label_order))

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": C_BASE["text"], "linewidth": 1.0},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8, "alpha": SCATTER_ALPHA},
            capprops={"linewidth": 0.8, "alpha": SCATTER_ALPHA},
        )
        for patch, lab in zip(bp["boxes"], label_order):
            patch.set_facecolor(C_SOLID["CEM"] if lab == "CEM" else C_SOLID["CAP"])
            patch.set_alpha(1.0)
            patch.set_edgecolor(
                C_BASE["CEM"] if lab == "CEM" else C_BASE["CAP"]
            )

        for pos, lab, values in zip(positions, label_order, data):
            jitter = rng.normal(0, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=8,
                alpha=SCATTER_ALPHA,
                color=C_BASE["CEM"] if lab == "CEM" else C_BASE["CAP"],
                edgecolors="none",
            )
            mean = float(np.mean(values)) if len(values) else np.nan
            ax.text(
                pos,
                mean + 0.055,
                f"{mean:+.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=C_BASE["text"],
            )

        ax.axhline(0, color=COLORS["GREY"], linestyle="--", linewidth=0.8, alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels(label_order)
        ax.set_xlabel("Evidence-only" if col == "margin_evidence_only" else "Final-prefix")
        ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Owner-contrastive margin")

    savefig(fig, "app_oc_margin_detail.pdf")


# =========================
# Figure A5: Mitigation threshold sensitivity
# =========================


def make_mitigation_tau_curve() -> None:
    df = pd.read_csv(find_file("mitigation_tau_pm_curve.csv", "mitigation_tau_pm_curve(1).csv"))
    df = df[df["split"] == "test"].copy()

    method_map = {"BCP_single_head_y_pm": "BCP", "MSPS": "MSPS"}
    df["method_label"] = df["method"].map(method_map).fillna(df["method"])

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    fig.patch.set_facecolor(C_BASE["bg"])

    for method, color, marker in [("BCP", C_BASE["PM"], "o"), ("MSPS", C_BASE["CAP"], "s")]:
        sub = df[df["method_label"] == method].sort_values("tau")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["tau"],
            sub["pm_rate"] * 100,
            marker=marker,
            markersize=3.5,
            linewidth=1.4,
            color=color,
            alpha=0.95,
            label=f"{method}: PM rate",
        )

    msps = df[df["method_label"] == "MSPS"].sort_values("tau")
    ax2 = None
    if "cb_retention_rate" in msps.columns and msps["cb_retention_rate"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(
            msps["tau"],
            msps["cb_retention_rate"] * 100,
            linestyle="--",
            marker="^",
            markersize=3.2,
            linewidth=1.1,
            color=COLORS["GREY"],
            alpha=0.85,
            label="MSPS: CB retention",
        )
        ax2.set_ylabel("CB retention (%)", color=COLORS["GREY"])
        ax2.tick_params(axis="y", labelcolor=COLORS["GREY"])
        ax2.set_ylim(0, 105)

    ax.axvline(0.50, color=C_BASE["PM"], linestyle=":", linewidth=1.0, alpha=0.75)
    ax.axvline(0.45, color=C_BASE["CAP"], linestyle=":", linewidth=1.0, alpha=0.75)

    ax.set_xlabel("Threshold $\\tau$")
    ax.set_ylabel("Trajectory PM rate (%)")
    ax.grid(axis="y", color=C_BASE["grid"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    if ax2 is not None:
        ax2.spines["top"].set_visible(False)
    else:
        ax.spines["right"].set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax.legend(lines, labels, loc="upper left", frameon=False, fontsize=6)

    savefig(fig, "app_mitigation_tau_curve.pdf")


# =========================
# Figure A6: Random matched deletion (summary interval — no per-seed list in JSON)
# =========================


def random_matched_overleaf_notes() -> str:
    """Numbers for caption / footnote — not drawn on the figure."""
    s = _load_random_matched_stats()
    refs = _load_mitigation_reference_rates()
    mean = float(s["pm_rate_mean"]) * 100
    std = float(s["pm_rate_std"]) * 100
    p05 = float(s["pm_rate_p05"]) * 100
    p95 = float(s["pm_rate_p95"]) * 100
    median = float(s["pm_rate_median"]) * 100
    n_seeds = int(s["n_seeds"])
    removed_cb = float(s["removed_cb_fraction_mean"]) * 100
    pm_count_mean = float(s.get("pm_count_mean", np.nan))
    msps = refs["msps"]
    bcp = refs["bcp"]
    baseline = refs["baseline"]

    lines = [
        "% --- paste into Overleaf (figure caption or footnote) ---",
        f"FHIR probe test ($n=146$). Random matched deletion: "
        f"{mean:.1f}\\% $\\pm$ {std:.1f}\\% (mean $\\pm$ std over {n_seeds} seeds); "
        f"5--95\\% interval [{p05:.1f}, {p95:.1f}]\\%; median {median:.1f}\\%.",
        f"Mean removed CB fraction {removed_cb:.1f}\\% (matched to MSPS per-trajectory retention).",
        f"Reference PM rates: MSPS {msps:.1f}\\%, BCP {bcp:.1f}\\%, baseline {baseline:.1f}\\% "
        f"(baseline omitted from axis; see main text / Table~3).",
    ]
    if not np.isnan(pm_count_mean):
        lines.append(f"Mean PM trajectory count after random deletion: {pm_count_mean:.1f}.")
    lines.append(
        "Dashed vertical lines: MSPS (green), BCP (blue). "
        "Tan bar: 5--95\\% interval; dot: mean $\\pm$ std."
    )
    return "\n".join(lines)


def make_random_matched_summary() -> None:
    """Compact interval plot (0--16\\% axis); annotation text → Overleaf only."""
    s = _load_random_matched_stats()
    refs = _load_mitigation_reference_rates()

    mean = float(s["pm_rate_mean"]) * 100
    std = float(s["pm_rate_std"]) * 100
    p05 = float(s["pm_rate_p05"]) * 100
    p95 = float(s["pm_rate_p95"]) * 100
    msps = refs["msps"]
    bcp = refs["bcp"]

    fig, ax = plt.subplots(figsize=(3.2, 1.35))
    fig.patch.set_facecolor(C_BASE["bg"])

    y = 0.0
    bar_h = 0.22

    # 5--95% interval (aggregated seeds; not a histogram).
    ax.barh(
        y,
        p95 - p05,
        left=p05,
        height=bar_h,
        color=C_SOLID["CEM"],
        edgecolor=C_BASE["CEM"],
        linewidth=0.6,
        alpha=1.0,
        zorder=1,
    )
    ax.vlines(
        [p05, p95],
        y - bar_h * 0.65,
        y + bar_h * 0.65,
        color=C_BASE["CEM"],
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )

    ax.errorbar(
        mean,
        y,
        xerr=std,
        fmt="o",
        color=C_BASE["text"],
        ecolor=C_BASE["text"],
        elinewidth=1.0,
        capsize=2.5,
        markersize=4,
        zorder=4,
    )

    ax.axvline(msps, color=C_BASE["CAP"], linestyle="--", linewidth=1.1, alpha=0.9, zorder=3)
    ax.axvline(bcp, color=C_BASE["PM"], linestyle="--", linewidth=1.1, alpha=0.9, zorder=3)

    ax.set_yticks([])
    ax.set_xlabel("Trajectory PM rate (%)")
    ax.set_xlim(0, 16)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_ylim(-0.55, 0.55)
    ax.grid(axis="x", color=C_BASE["grid"], linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    savefig(fig, "app_random_matched_summary.pdf")

    notes = random_matched_overleaf_notes()
    notes_path = OUT_DIR / "app_random_matched_summary_overleaf.txt"
    notes_path.write_text(notes + "\n", encoding="utf-8")
    print(f"Overleaf notes: {notes_path}")
    print(notes)


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"CHART_FILL_ALPHA={CHART_FILL_ALPHA}")
    print(f"Data roots: {[str(d) for d in EXPORT_DIRS if d.is_dir()]}")
    make_task_pm_heatmap()
    make_claim_distribution()
    make_fhir_primary_stratum()
    make_oc_margin_detail()
    make_mitigation_tau_curve()
    make_random_matched_summary()
