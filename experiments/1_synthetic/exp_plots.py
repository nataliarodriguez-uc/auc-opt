"""
Tier 1 Experiment Plots
========================
Reads CSV files produced by experiment_runner.py and generates
publication-quality figures saved to FIGURES_DIR.

Figures produced
----------------
1.  auc_vs_sigma_grid.png        — AUC vs σ with CI bands, one panel per dataset
2.  auc_vs_sigma_heatmap.png     — Heat-map: rows=datasets, cols=σ, color=mean AUC
3.  timing_vs_sigma.png          — Wall-time + ALM-iters vs σ
4.  init_robustness_box.png      — Box plots of AUC distribution over w0 initializations
5.  convergence_residuals.png    — Constraint residual vs outer iteration
6.  convergence_ssn_iters.png    — SSN inner iterations vs outer iteration
7.  summary_table.png            — LaTeX-style table for paper (via matplotlib)

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
from exp_config import (
    RESULTS_DIR, FIGURES_DIR, DATASET_SPECS, DATASET_KEYS,
    SIGMA_GRID, CONVERGENCE_EXPERIMENTS, ALM_DEFAULTS, CONVERGENCE_DIR
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

def fig_auc_vs_sigma_grid(df=None, df_baselines=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))
    if df_baselines is None:
        path = os.path.join(RESULTS_DIR, "baselines.csv")
        if os.path.exists(path):
            df_baselines = pd.read_csv(path)

    # Paired layout — each row is a regime, columns are low/high sep
    layout = [
        ("PI1_lowsep_mggn",       "PI4_highsep_mggn"),
        ("PI3_lowsep_mlln",       "PI6_highsep_mlln"),
        ("PI7_lowsep_imbalanced", "PI8_highsep_imbalanced"),
    ]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12))

    for row_idx, (dk_left, dk_right) in enumerate(layout):
        for col_idx, dk in enumerate([dk_left, dk_right]):
            ax   = axes[row_idx, col_idx]
            spec = DATASET_SPECS[dk]
            sub  = df[df.dataset_key == dk]

            if sub.empty:
                ax.set_visible(False)
                continue

            stats = sub.groupby("sigma")["auc"].agg(["mean", "std"]).reset_index()

            # ── Prox ──────────────────────────────────────────────
            ax.fill_between(stats.sigma,
                            stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"],
                            alpha=0.20, color=PALETTE[0])
            ax.plot(stats.sigma, stats["mean"],
                    color=PALETTE[0], marker="o", ms=5, lw=1.8,
                    label=f"Prox ({stats['mean'].mean():.3f} ± {stats['std'].mean():.3f})")

            # ── Baselines ──────────────────────────────────────────
            if df_baselines is not None:
                sub_b = df_baselines[df_baselines.dataset_key == dk]
                if not sub_b.empty:
                    bce_mean = sub_b["bce_auc"].mean()
                    bce_std  = sub_b["bce_auc"].std()
                    ax.axhline(bce_mean, color=PALETTE[1], ls="--", lw=1.5,
                               label=f"BCE ({bce_mean:.3f} ± {bce_std:.3f})")
                    ax.axhspan(max(0, bce_mean - bce_std),
                               min(1, bce_mean + bce_std),
                               alpha=0.08, color=PALETTE[1])

                    libauc_mean = sub_b["libauc_auc"].mean()
                    libauc_std  = sub_b["libauc_auc"].std()
                    ax.axhline(libauc_mean, color=PALETTE[2], ls="--", lw=1.5,
                               label=f"LibAUC ({libauc_mean:.3f} ± {libauc_std:.3f})")
                    ax.axhspan(max(0, libauc_mean - libauc_std),
                               min(1, libauc_mean + libauc_std),
                               alpha=0.08, color=PALETTE[2])

            # ── γ=2 boundary ───────────────────────────────────────
            ax.axvline(0.5, color="gray", ls=":", lw=0.8, label="γ=2 (σ=0.5)")

            # ── x-axis: log scale, literal labels ─────────────────
            sigma_vals = sorted(sub["sigma"].unique())
            ax.set_xscale("log")
            ax.set_xticks(sigma_vals)
            ax.set_xticklabels([str(s) for s in sigma_vals], rotation=45, ha="right")
            ax.xaxis.set_minor_locator(ticker.NullLocator())

            ax.set_xlabel("σ")
            ax.set_ylabel("Test AUC")
            ax.set_ylim(0, 1.05)
            ax.set_title(spec["label"], fontsize=9, pad=6)
            ax.legend(fontsize=7, loc="lower right",
                      framealpha=0.9, edgecolor="lightgray")

    # Column headers to make the layout readable at a glance
    axes[0, 0].set_title("Low Separation\n" + DATASET_SPECS["PI1_lowsep_mggn"]["label"],
                          fontsize=9, pad=6)
    axes[0, 1].set_title("High Separation\n" + DATASET_SPECS["PI4_highsep_mggn"]["label"],
                          fontsize=9, pad=6)

    fig.suptitle(
        "Effect of Penalty Parameter σ on Test AUC across Dataset Regimes\n"
        "Shaded bands = ±1 std over seeds  |  Dashed lines = baseline means",
        fontsize=12, y=1.01
    )
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "auc_vs_sigma_grid.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Timing and ALM iteration count vs σ
# ─────────────────────────────────────────────────────────────────────────────

def fig_timing_vs_sigma(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "sigma_sensitivity.csv"))

    layout = [
        ("PI1_lowsep_mggn",       "PI4_highsep_mggn"),
        ("PI3_lowsep_mlln",       "PI6_highsep_mlln"),
        ("PI7_lowsep_imbalanced", "PI8_highsep_imbalanced"),
    ]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12))

    for row_idx, (dk_left, dk_right) in enumerate(layout):
        for col_idx, dk in enumerate([dk_left, dk_right]):
            ax   = axes[row_idx, col_idx]
            spec = DATASET_SPECS[dk]
            sub  = df[df.dataset_key == dk]

            if sub.empty:
                ax.set_visible(False)
                continue

            # ── timing stats ───────────────────────────────────────
            time_stats = sub.groupby("sigma")["alm_time"].agg(
                ["mean", "std"]).reset_index()
            iter_stats = sub.groupby("sigma")["alm_iter"].agg(
                ["mean", "std"]).reset_index()

            sigma_vals = sorted(sub["sigma"].unique())

            # ── wall time on primary axis ──────────────────────────
            ax.fill_between(time_stats.sigma,
                           time_stats["mean"] - time_stats["std"],
                           time_stats["mean"] + time_stats["std"],
                           alpha=0.20, color=PALETTE[0])
            ax.plot(time_stats.sigma, time_stats["mean"],
                   color=PALETTE[0], marker="o", ms=5, lw=1.8,
                   label="wall time (s)")
            ax.set_ylabel("Wall time (s)", color=PALETTE[0])
            ax.tick_params(axis="y", labelcolor=PALETTE[0])

            # ── ALM iterations on secondary axis ───────────────────
            ax2 = ax.twinx()
            ax2.fill_between(iter_stats.sigma,
                            iter_stats["mean"] - iter_stats["std"],
                            iter_stats["mean"] + iter_stats["std"],
                            alpha=0.15, color=PALETTE[1])
            ax2.plot(iter_stats.sigma, iter_stats["mean"],
                    color=PALETTE[1], marker="s", ms=5, lw=1.8,
                    ls="--", label="ALM iters")
            ax2.set_ylabel("ALM iterations", color=PALETTE[1])
            ax2.tick_params(axis="y", labelcolor=PALETTE[1])
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # ── γ=2 boundary ───────────────────────────────────────
            ax.axvline(0.5, color="gray", ls=":", lw=0.8)

            # ── x axis ────────────────────────────────────────────
            ax.set_xscale("log")
            ax.set_xticks(sigma_vals)
            ax.set_xticklabels([str(s) for s in sigma_vals],
                              rotation=45, ha="right")
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xlabel("σ")

            # ── combined legend ────────────────────────────────────
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                     fontsize=7, loc="upper right",
                     framealpha=0.9, edgecolor="lightgray")

            # ── dataset info in title ──────────────────────────────
            imbalance = spec.get("imbalance", "50/50")
            ax.set_title(
                f"{spec['label']}\n"
                f"m={spec['m']}, n={spec['n']}, ratio={imbalance}",
                fontsize=8, pad=6
            )

    fig.suptitle(
        "Computational Cost vs Penalty Parameter σ across Dataset Regimes\n"
        "Wall time (solid) and ALM iterations (dashed)  |  "
        "Shaded bands = ±1 std over seeds",
        fontsize=11, y=1.01
    )
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "timing_vs_sigma.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Initialization Robustness Box plots
# ─────────────────────────────────────────────────────────────────────────────

def fig_init_robustness(df=None, df_baselines=None):
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "init_robustness.csv"))
    if df_baselines is None:
        path = os.path.join(RESULTS_DIR, "baselines.csv")
        if os.path.exists(path):
            df_baselines = pd.read_csv(path)

    layout = [
        ("PI1_lowsep_mggn",       "PI4_highsep_mggn"),
        ("PI3_lowsep_mlln",       "PI6_highsep_mlln"),
        ("PI7_lowsep_imbalanced", "PI8_highsep_imbalanced"),
    ]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12))

    for row_idx, (dk_left, dk_right) in enumerate(layout):
        for col_idx, dk in enumerate([dk_left, dk_right]):
            ax   = axes[row_idx, col_idx]
            spec = DATASET_SPECS[dk]
            sub  = df[df.dataset_key == dk]

            if sub.empty:
                ax.set_visible(False)
                continue

            # ── Random w0 box plot ─────────────────────────────────
            random_auc = sub[sub.init_type == "random"]["auc"].values
            bp = ax.boxplot(random_auc,
                           positions=[1],
                           patch_artist=True,
                           widths=0.35,
                           medianprops=dict(color="black", lw=2),
                           whiskerprops=dict(lw=1.2),
                           capprops=dict(lw=1.2),
                           flierprops=dict(marker=".", alpha=0.5, ms=5))
            bp["boxes"][0].set_facecolor(PALETTE[0])
            bp["boxes"][0].set_alpha(0.6)

            # jitter
            jitter = np.random.normal(1, 0.04, size=len(random_auc))
            ax.scatter(jitter, random_auc, alpha=0.3, s=15,
                      color=PALETTE[0], zorder=3)


            # ── Baseline reference lines ───────────────────────────
            if df_baselines is not None:
                sub_b = df_baselines[df_baselines.dataset_key == dk]
                if not sub_b.empty:
                    bce_mean = sub_b["bce_auc"].mean()
                    bce_std  = sub_b["bce_auc"].std()
                    ax.axhline(bce_mean, color=PALETTE[1], ls="--", lw=1.5,
                               label=f"BCE ({bce_mean:.3f} ± {bce_std:.3f})")
                    ax.axhspan(max(0, bce_mean - bce_std),
                               min(1, bce_mean + bce_std),
                               alpha=0.08, color=PALETTE[1])

                    libauc_mean = sub_b["libauc_auc"].mean()
                    libauc_std  = sub_b["libauc_auc"].std()
                    ax.axhline(libauc_mean, color=PALETTE[2], ls="--", lw=1.5,
                               label=f"LibAUC ({libauc_mean:.3f} ± {libauc_std:.3f})")
                    ax.axhspan(max(0, libauc_mean - libauc_std),
                               min(1, libauc_mean + libauc_std),
                               alpha=0.08, color=PALETTE[2])

            # ── Random baseline ────────────────────────────────────
            ax.axhline(0.5, color="gray", ls=":", lw=0.8,
                      label="random (AUC=0.5)")

            # ── Mean/std annotation for random ─────────────────────
            mean_auc = random_auc.mean()
            std_auc  = random_auc.std()
            ax.text(1.45, mean_auc,
                   f"rand: {mean_auc:.3f}±{std_auc:.3f}",
                   fontsize=7, va="center", color=PALETTE[0])

            # ── x axis labels ──────────────────────────────────────
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Random\n(20 trials)", "LR\nwarm", "LDA\nwarm"],
                              fontsize=8)
            ax.set_xlim(0.4, 3.6)

            # ── Formatting ─────────────────────────────────────────
            imbalance = spec.get("imbalance", "50/50")
            ax.set_title(
                f"{spec['label']}\n"
                f"m={spec['m']}, n={spec['n']}, ratio={imbalance}",
                fontsize=8, pad=6
            )
            ax.set_ylabel("Test AUC")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right",
                     framealpha=0.9, edgecolor="lightgray")

    sigma_val = df["sigma"].iloc[0]
    n_trials  = len(df[df.init_type == "random"]) // len(DATASET_KEYS)
    fig.suptitle(
        f"Initialization Robustness: Random w₀  |  σ={sigma_val}\n"
        f"Box = IQR over {n_trials} random trials  |  "
        f"Dashed = baseline means",
        fontsize=10, y=1.01
    )
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "init_robustness_box.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Constraint Residual vs Outer Iteration
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence_residuals_sigma_scale():
    """
    PI3 only — sigma_scale comparison + warm start comparison overlaid.
    Left panel: residual traces
    Right panel: SSN inner iterations
    """
    pi3_experiments = [e for e in CONVERGENCE_EXPERIMENTS
                      if e["dataset_key"] == "PI3_lowsep_mlln"]

    # separate by type for different styling
    scale_exps     = [e for e in pi3_experiments if e.get("warm_start") is None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scale_colors     = [PALETTE[0], PALETTE[1], PALETTE[2]]

    for panel_idx, ax in enumerate(axes):
        ylabel = "‖constraint residual‖∞" if panel_idx == 0 else "SSN inner iterations"
        title  = "Constraint Residual vs Iteration" if panel_idx == 0 \
                 else "SSN Inner Iterations per Outer Iteration"

        # ── sigma_scale comparison lines ───────────────────────────
        for exp, color in zip(scale_exps, scale_colors):
            label_key = exp["label"].replace(" ", "_").replace("=", "")
            path      = os.path.join(CONVERGENCE_DIR,
                                    f"convergence_{label_key}.csv")
            if not os.path.exists(path):
                print(f"  Missing: {path}")
                continue

            trace    = pd.read_csv(path)
            auc_val  = trace.get("auc", [None]).iloc[-1] if "auc" in trace.columns else "?"
            leg_label = (f"random, scale={exp['sigma_scale']} "
                        f"({len(trace)} iters)")

            if panel_idx == 0:
                ax.semilogy(trace["iteration"], trace["residual_inf"],
                           color=color, lw=1.8, marker="o", ms=4,
                           label=leg_label)
                # shade spikes
                spikes = trace[trace["residual_inf"] >
                               trace["residual_inf"].shift(1).fillna(0)]
                for _, row in spikes.iterrows():
                    ax.axvspan(row["iteration"] - 0.4, row["iteration"] + 0.4,
                              alpha=0.06, color="red")
            else:
                ax.plot(trace["iteration"], trace["ssn_iters"],
                       color=color, lw=1.8, marker="o", ms=4,
                       label=leg_label)

        # ── tolerance line (residual panel only) ───────────────────
        if panel_idx == 0:
            ax.axhline(ALM_DEFAULTS["tol_alm"], color="red", ls=":",
                      lw=1.0, label=f"tol={ALM_DEFAULTS['tol_alm']:.0e}")

        ax.set_xlabel("ALM outer iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, pad=6)
        ax.legend(fontsize=7, loc="upper right",
                 framealpha=0.9, edgecolor="lightgray")

        if panel_idx == 1:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # load summary to get AUC values for suptitle annotation
    sum_path = os.path.join(CONVERGENCE_DIR, "convergence_summary.csv")
    auc_note = ""
    if os.path.exists(sum_path):
        df_sum   = pd.read_csv(sum_path)
        pi3_rows = df_sum[df_sum["dataset_key"] == "PI3_lowsep_mlln"]
        if not pi3_rows.empty:
            parts = []
            for _, row in pi3_rows.iterrows():
                init = row.get("warm_start", "random")
                parts.append(f"{init} AUC={row['auc']:.3f}")
            auc_note = "  |  " + "   ".join(parts)

    fig.suptitle(
        "Convergence on PI3 low-sep m<<n  —  Random Init (solid) vs "
        "Warm Starts (dashed)\n"
        "Left: constraint residual  |  Right: SSN inner iterations  |  "
        f"Red shading = residual spike{auc_note}",
        fontsize=9, y=1.02
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "convergence_sigma_scale.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


def fig_convergence_residuals_by_dataset():
    """
    One panel per non-PI3 dataset showing residual + SSN iters.
    Handles warm start variants if present.
    """
    other_experiments = [e for e in CONVERGENCE_EXPERIMENTS
                        if e["dataset_key"] != "PI3_lowsep_mlln"]

    if not other_experiments:
        print("  No non-PI3 experiments found")
        return

    n = len(other_experiments)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    for ax, exp, color in zip(axes, other_experiments, colors):
        label_key = exp["label"].replace(" ", "_").replace("=", "")
        path      = os.path.join(CONVERGENCE_DIR,
                                f"convergence_{label_key}.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            continue

        trace     = pd.read_csv(path)
        spec      = DATASET_SPECS[exp["dataset_key"]]
        warm_start = exp.get("warm_start", None)
        ls        = "--" if warm_start else "-"
        marker    = "D" if warm_start else "o"

        # ── residual line ──────────────────────────────────────────
        ax.semilogy(trace["iteration"], trace["residual_inf"],
                   color=color, lw=1.8, marker=marker, ms=6, ls=ls)
        ax.axhline(ALM_DEFAULTS["tol_alm"], color="red",
                   ls=":", lw=1.0,
                   label=f"tol={ALM_DEFAULTS['tol_alm']:.0e}")

        # ── SSN iters as secondary axis bars ──────────────────────
        ax2 = ax.twinx()
        ax2.bar(trace["iteration"], trace["ssn_iters"],
               alpha=0.25, color=color, width=0.4)
        ax2.set_ylabel("SSN iters", fontsize=8)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.tick_params(labelsize=7)

        # ── annotations ───────────────────────────────────────────
        n_iters   = len(trace)
        final_res = trace["residual_inf"].iloc[-1]
        init_str  = f"warm={warm_start}" if warm_start else "random init"

        # get AUC from summary if available
        sum_path = os.path.join(CONVERGENCE_DIR, "convergence_summary.csv")
        auc_str  = ""
        if os.path.exists(sum_path):
            df_sum = pd.read_csv(sum_path)
            row    = df_sum[df_sum["label"] == exp["label"]]
            if not row.empty:
                auc_str = f"\nAUC={row['auc'].values[0]:.3f}"

        ax.text(0.05, 0.05,
               f"{init_str}\n{n_iters} iters  res={final_res:.2e}{auc_str}",
               transform=ax.transAxes, fontsize=7,
               verticalalignment="bottom",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="lightgray", alpha=0.8))

        imbalance  = spec.get("imbalance", "50/50")
        scale_used = exp.get("sigma_scale", ALM_DEFAULTS["sigma_scale"])
        ax.set_xlabel("ALM outer iteration")
        ax.set_ylabel("‖constraint residual‖∞")
        ax.set_title(
            f"{spec['label']}\n"
            f"σ={exp['sigma']}, scale={scale_used}, ratio={imbalance}",
            fontsize=8, pad=6
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(
        "Convergence Diagnostics — PI1, PI6, PI7\n"
        "Residual (line) and SSN inner iterations (bars) per ALM outer iteration",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "convergence_by_dataset.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)