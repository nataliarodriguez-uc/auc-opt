"""
Tier 1 Experiment Plots
========================
Reads CSV files produced by experiment_runner.py and generates
publication-quality figures saved to FIGURES_DIR.

Figures produced
----------------
1.  auc_vs_sigma_grid.pdf        — AUC vs σ with CI bands, one panel per dataset
2.  auc_vs_sigma_heatmap.pdf     — Heat-map: rows=datasets, cols=σ, color=mean AUC
3.  timing_vs_sigma.pdf          — Wall-time + ALM-iters vs σ
4.  init_robustness_box.pdf      — Box plots of AUC distribution over w0 initializations
5.  convergence_residuals.pdf    — Constraint residual vs outer iteration
6.  convergence_ssn_iters.pdf    — SSN inner iterations vs outer iteration
7.  summary_table.pdf            — LaTeX-style table for paper (via matplotlib)

Usage
-----
    python experiment_plots.py          # plot everything
    python experiment_plots.py --fig 1  # plot only figure 1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Computer Modern Roman", "DejaVu Serif"],
    "text.usetex"      : False,       # set True if LaTeX installed
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 8,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from experiment_config import (
    RESULTS_DIR, FIGURES_DIR, DATASET_SPECS, DATASET_KEYS,
    SIGMA_GRID, CONVERGENCE_EXPERIMENTS, ALM_DEFAULTS,
)

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── colour palette (colour-blind friendly) ────────────────────────────────────
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

REGIME_MARKERS = {"m>>n": "o", "m<<n": "s"}
SEP_LINESTYLE  = {"low": "--", "high": "-"}

# γ regime boundaries (vertical reference lines on σ-axis)
GAMMA_BOUNDARIES = {
    "σ=0.5 (γ=2)": 0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — AUC vs σ (one sub-plot per dataset, CI bands over seeds)
# ─────────────────────────────────────────────────────────────────────────────

def fig_auc_vs_sigma_grid(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))

    keys    = [k for k in DATASET_KEYS if k in df.dataset_key.unique()]
    n_plots = len(keys)
    ncols   = 4
    nrows   = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    for ax_idx, dk in enumerate(keys):
        ax   = axes[ax_idx]
        sub  = df[df.dataset_key == dk]
        spec = DATASET_SPECS[dk]

        stats = sub.groupby("sigma")["auc"].agg(["mean", "std", "min", "max"]).reset_index()

        ax.fill_between(stats.sigma, stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"],
                        alpha=0.20, color=PALETTE[0])
        ax.fill_between(stats.sigma, stats["min"], stats["max"],
                        alpha=0.10, color=PALETTE[0])
        ax.plot(stats.sigma, stats["mean"],
                color=PALETTE[0], marker="o", ms=5, lw=1.8, label="Prox (mean)")
        ax.plot(stats.sigma, stats["min"],
                color=PALETTE[0], lw=0.8, ls=":", alpha=0.7)
        ax.plot(stats.sigma, stats["max"],
                color=PALETTE[0], lw=0.8, ls=":", alpha=0.7)

        # γ=2 boundary
        ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6, label="γ=2 (σ=0.5)")

        ax.set_xscale("log")
        ax.set_xlabel("σ  (log scale)")
        ax.set_ylabel("Test AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title(spec["label"], fontsize=9, pad=4)

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    # hide unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("AUC vs Penalty Parameter σ  (mean ± std / min-max over seeds)",
                 fontsize=12, y=1.01)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "auc_vs_sigma_grid.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — AUC Heat-map  (datasets × σ values)
# ─────────────────────────────────────────────────────────────────────────────

def fig_auc_heatmap(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))

    pivot = (df.groupby(["dataset_key", "sigma"])["auc"]
               .mean()
               .unstack("sigma"))

    # Row order: match DATASET_KEYS
    row_order = [k for k in DATASET_KEYS if k in pivot.index]
    pivot     = pivot.loc[row_order]
    labels    = [DATASET_SPECS[k]["label"] for k in row_order]

    fig, ax = plt.subplots(figsize=(len(pivot.columns) * 0.9 + 2.5,
                                    len(pivot.index) * 0.55 + 1.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s:.2g}" for s in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("σ")
    ax.set_title("Mean Test AUC across seeds", pad=8)

    # annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val > 0.65 else "white")

    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)

    # mark γ=2 boundary column
    sigma_cols = list(pivot.columns)
    if 0.5 in sigma_cols:
        col_idx = sigma_cols.index(0.5)
        ax.axvline(col_idx - 0.5, color="royalblue", lw=2, ls="--", label="γ=2 (σ=0.5)")
        ax.axvline(col_idx + 0.5, color="royalblue", lw=2, ls="--")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "auc_vs_sigma_heatmap.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Timing and ALM iteration count vs σ
# ─────────────────────────────────────────────────────────────────────────────

def fig_timing_vs_sigma(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))

    keys  = [k for k in DATASET_KEYS if k in df.dataset_key.unique()]
    ncols = 2
    nrows = int(np.ceil(len(keys) / ncols))

    fig, axes = plt.subplots(nrows, ncols * 2,
                             figsize=(5 * ncols * 2, 3.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols * 2)

    for idx, dk in enumerate(keys):
        row  = idx // ncols
        base = (idx % ncols) * 2
        ax_t = axes[row, base]        # timing axis
        ax_i = axes[row, base + 1]    # iteration axis

        sub   = df[df.dataset_key == dk]
        spec  = DATASET_SPECS[dk]
        stats = sub.groupby("sigma")[["alm_time", "alm_iter"]].agg(
                    ["mean", "std"]).reset_index()
        stats.columns = ["sigma", "time_mean", "time_std", "iter_mean", "iter_std"]

        # Wall time
        ax_t.errorbar(stats.sigma, stats.time_mean, yerr=stats.time_std,
                      color=PALETTE[1], marker="o", ms=5, lw=1.6, capsize=3)
        ax_t.set_xscale("log")
        ax_t.set_xlabel("σ")
        ax_t.set_ylabel("Wall time (s)")
        ax_t.set_title(f"{spec['label']}\nWall time", fontsize=8)
        ax_t.axvline(0.5, color="gray", ls="--", lw=0.8)

        # ALM iterations
        ax_i.errorbar(stats.sigma, stats.iter_mean, yerr=stats.iter_std,
                      color=PALETTE[2], marker="s", ms=5, lw=1.6, capsize=3)
        ax_i.set_xscale("log")
        ax_i.set_xlabel("σ")
        ax_i.set_ylabel("ALM iterations")
        ax_i.set_title(f"{spec['label']}\nALM iterations", fontsize=8)
        ax_i.axvline(0.5, color="gray", ls="--", lw=0.8)

    # hide unused axes
    total_slots = nrows * ncols * 2
    for idx in range(len(keys) * 2, total_slots):
        r = idx // (ncols * 2)
        c = idx %  (ncols * 2)
        if r < nrows:
            axes[r, c].set_visible(False)

    fig.suptitle("Wall-time and ALM Iteration Count vs σ  (mean ± std over seeds)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "timing_vs_sigma.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Initialization Robustness Box plots
# ─────────────────────────────────────────────────────────────────────────────

def fig_init_robustness(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "init_robustness.csv"))

    keys   = [k for k in DATASET_KEYS if k in df.dataset_key.unique()]
    labels = [DATASET_SPECS[k]["label"] for k in keys]

    data_per_dataset = [df[df.dataset_key == k]["auc"].values for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.3), 4.5))

    bp = ax.boxplot(data_per_dataset,
                    labels=labels,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker=".", alpha=0.5, ms=5))

    for i, (patch, key) in enumerate(zip(bp["boxes"], keys)):
        color = PALETTE[i % len(PALETTE)]
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, vals in enumerate(data_per_dataset):
        jitter = np.random.normal(i + 1, 0.06, size=len(vals))
        ax.scatter(jitter, vals, alpha=0.4, s=18,
                   color=PALETTE[i % len(PALETTE)], zorder=3)

    ax.set_ylabel("Test AUC")
    ax.set_title(
        f"Initialization Robustness  (σ = {df.sigma.iloc[0]:.2f},  "
        f"n_trials = {df.trial.max() + 1}  random w₀)",
        fontsize=10)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="random baseline")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "init_robustness_box.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Constraint Residual vs Outer Iteration
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence_residuals(experiments=None):
    if experiments is None:
        experiments = CONVERGENCE_EXPERIMENTS

    fig, axes = plt.subplots(1, len(experiments),
                             figsize=(4.5 * len(experiments), 3.8),
                             sharey=False)
    if len(experiments) == 1:
        axes = [axes]

    max_iter = ALM_DEFAULTS["max_iter_alm"]

    for ax, exp in zip(axes, experiments):
        label = exp["label"].replace(" ", "_").replace(",", "").replace("=", "")
        path  = os.path.join(RESULTS_DIR, f"convergence_{label}.csv")
        if not os.path.exists(path):
            ax.set_title(f"Missing: {exp['label']}", fontsize=8)
            continue

        trace = pd.read_csv(path)
        iters = trace["iteration"].values
        res   = trace["residual_inf"].values
        tol   = ALM_DEFAULTS["tol_alm"]

        ax.semilogy(iters, res, color=PALETTE[0], lw=1.8, marker="o", ms=4)
        ax.axhline(tol, color="red", ls="--", lw=1.0, label=f"tol = {tol:.0e}")

        ax.set_xlabel("ALM outer iteration")
        ax.set_ylabel("‖constraint residual‖∞")
        ax.set_title(exp["label"], fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0.5, max(iters) + 0.5)

    fig.suptitle("Convergence: Constraint Residual vs ALM Iteration",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "convergence_residuals.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — SSN inner iterations vs outer ALM iteration
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence_ssn_iters(experiments=None):
    if experiments is None:
        experiments = CONVERGENCE_EXPERIMENTS

    fig, axes = plt.subplots(1, len(experiments),
                             figsize=(4.5 * len(experiments), 3.8),
                             sharey=True)
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        label = exp["label"].replace(" ", "_").replace(",", "").replace("=", "")
        path  = os.path.join(RESULTS_DIR, f"convergence_{label}.csv")
        if not os.path.exists(path):
            ax.set_title(f"Missing: {exp['label']}", fontsize=8)
            continue

        trace      = pd.read_csv(path)
        iters      = trace["iteration"].values
        ssn_counts = trace["ssn_iters"].values
        sigma_t    = trace["sigma_t"].values

        # bar chart for SSN iters, colored by σ value (warmer = larger σ)
        norm   = plt.Normalize(sigma_t.min(), sigma_t.max())
        colors = plt.cm.plasma(norm(sigma_t))

        bars = ax.bar(iters, ssn_counts, color=colors, edgecolor="none", alpha=0.85)
        sm   = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="σₜ", shrink=0.75)

        ax.set_xlabel("ALM outer iteration")
        ax.set_ylabel("SSN inner iterations")
        ax.set_title(exp["label"], fontsize=9)
        ax.set_xlim(0.5, max(iters) + 0.5)

    fig.suptitle("SSN Inner Iterations per ALM Outer Iteration",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "convergence_ssn_iters.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Compact summary table (rendered as a figure for paper)
# ─────────────────────────────────────────────────────────────────────────────

def fig_summary_table(df=None, sigma_focal=None):
    """
    Recreate a table similar to the paper's Tables 1/2 but for ALL datasets
    and focal σ values.  Renders as a matplotlib figure (PDF-embeddable).
    """
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))
    if sigma_focal is None:
        from experiment_config import SIGMA_FOCAL
        sigma_focal = SIGMA_FOCAL

    sub = df[df.sigma.isin(sigma_focal)]
    tbl = (sub.groupby(["dataset_label", "m", "n", "sigma"])["auc"]
              .agg(["mean", "std"])
              .reset_index())
    tbl.columns = ["Dataset", "m", "n", "σ", "AUC mean", "AUC std"]
    tbl["AUC"] = tbl.apply(
        lambda r: f"{r['AUC mean']:.4f} ± {r['AUC std']:.4f}", axis=1)

    display_cols = ["Dataset", "m", "n", "σ", "AUC"]
    cell_data    = tbl[display_cols].values

    fig, ax = plt.subplots(figsize=(14, max(3, len(cell_data) * 0.32 + 1.2)))
    ax.axis("off")

    tbl_obj = ax.table(
        cellText  = cell_data,
        colLabels = display_cols,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl_obj.auto_set_font_size(False)
    tbl_obj.set_fontsize(8)
    tbl_obj.scale(1.15, 1.45)

    # Header styling
    for j in range(len(display_cols)):
        cell = tbl_obj[(0, j)]
        cell.set_facecolor("#2E4057")
        cell.set_text_props(color="white", fontweight="bold")

    # Row banding
    for i in range(1, len(cell_data) + 1):
        color = "#F0F4F8" if i % 2 == 0 else "white"
        for j in range(len(display_cols)):
            tbl_obj[(i, j)].set_facecolor(color)

    fig.suptitle(
        f"AUC Summary (mean ± std over {len(SEEDS)} seeds)  —  focal σ ∈ {sigma_focal}",
        fontsize=10, y=0.96)

    out = os.path.join(FIGURES_DIR, "summary_table.pdf")
    fig.savefig(out)
    print(f"  Saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

FIG_MAP = {
    "1": ("AUC vs σ grid",             fig_auc_vs_sigma_grid),
    "2": ("AUC heatmap",               fig_auc_heatmap),
    "3": ("Timing vs σ",               fig_timing_vs_sigma),
    "4": ("Init robustness box",       fig_init_robustness),
    "5": ("Convergence residuals",     fig_convergence_residuals),
    "6": ("Convergence SSN iters",     fig_convergence_ssn_iters),
    "7": ("Summary table",             fig_summary_table),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default="all",
                        help="Figure number (1-7) or 'all'")
    args = parser.parse_args()

    print("=" * 50)
    print("  Tier 1 — Generating Figures")
    print("=" * 50)

    targets = FIG_MAP.keys() if args.fig == "all" else [args.fig]
    for key in targets:
        if key not in FIG_MAP:
            print(f"  Unknown figure: {key}")
            continue
        name, fn = FIG_MAP[key]
        print(f"\n[{key}] {name}")
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  ⚠️  Data not found — run experiment_runner.py first.  ({e})")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    print("\n✅ Done.")