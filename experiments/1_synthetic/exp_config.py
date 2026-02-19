"""
Experiment Configuration
================================
All datasets, parameter grids, and seeds are defined here.
To change an experiment, edit ONLY this file.
"""

import numpy as np

# ─────────────────────────────────────────────
# Random seeds (used for both data gen & init)
# ─────────────────────────────────────────────
SEEDS = [1034, 1234, 42, 99, 7, 2]
N_SEEDS = len(SEEDS)

# ─────────────────────────────────────────────
# σ grid  (γ = 1/σ changes the proximal regime)
#   γ > 2  ←→  σ < 0.5
#   γ = 2  ←→  σ = 0.5
#   γ < 2  ←→  σ > 0.5
# ─────────────────────────────────────────────
SIGMA_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]

# σ values used in the focused 3-column comparison tables (legacy)
SIGMA_FOCAL = [0.1, 1.0, 2.0]

# ─────────────────────────────────────────────
# ALM / SSN / LineSearch defaults
# ─────────────────────────────────────────────
ALM_DEFAULTS = dict(
    max_iter_alm = 30,
    tau_scale    = 0.5,
    sigma_scale  = 2.0,
    tol_alm      = 1e-4,
)

SSN_DEFAULTS = dict(
    tol_ssn      = 1e-4,
    max_iter_ssn = 25,
)

LS_DEFAULTS = dict(
    c            = 1e-4,
    max_iter_ls  = 20,
    beta         = 0.5,
)

# Fixed tau and alpha across all Tier-1 experiments
TAU0   = 1.0
ALPHA0 = 1.0

# Train/test split ratio
TRAIN_RATIO = 0.7

# ─────────────────────────────────────────────
# Dataset specifications
# Each entry is a dict consumed by DataSet_SVM
# ─────────────────────────────────────────────
DATASET_SPECS = {

    # ── Low separation ───────────────────────
    "PI1_lowsep_mggn": dict(
        label        = "PI1 low-sep  m>>n",
        m            = 1000,
        n            = 50,
        sep_distance = 0.5,
        class_ratios = [0.5, 0.5],
    ),
    "PI3_lowsep_mlln": dict(
        label        = "PI3 low-sep  m<<n",
        m            = 50,
        n            = 500,
        sep_distance = 0.5,
        class_ratios = [0.5, 0.5],
    ),
    # ── High separation ───────────────────────
    "PI4_highsep_mggn": dict(
        label        = "PI4 high-sep m>>n",
        m            = 1000,
        n            = 50,
        sep_distance = 3.0,
        class_ratios = [0.5, 0.5],
    ),

    "PI6_highsep_mlln": dict(
        label        = "PI6 high-sep m<<n",
        m            = 50,
        n            = 500,
        sep_distance = 3.0,
        class_ratios = [0.5, 0.5],
    ),
    
    # ── Imbalanced ────────────────────────────
    "PI7_lowsep_imbalanced": dict(
        label        = "PI7 low-sep  imbalanced (80/20)",
        m            = 600,
        n            = 50,
        sep_distance = 0.5,
        class_ratios = [0.8, 0.2],   # 80 % negative, 20 % positive
    ),
    "PI8_highsep_imbalanced": dict(
        label        = "PI8 high-sep imbalanced (80/20)",
        m            = 600,
        n            = 50,
        sep_distance = 3.0,
        class_ratios = [0.8, 0.2],
    )
}

#DATASET_SPECS = {

    # ── Low separation ───────────────────────
    #"PI1_lowsep_mggn": dict(
        #label        = "PI1 low-sep  m≫n",
        #m            = 1000,
        #n            = 50,
        #sep_distance = 0.5,
        #class_ratios = [0.5, 0.5],
    #),
    #"PI2_lowsep_balanced_small": dict(
        #label        = "PI2 low-sep  m≫n (small)",
        #m            = 300,
        #n            = 20,
        #sep_distance = 0.5,
        #class_ratios = [0.5, 0.5],
    #),
    #"PI3_lowsep_mlln": dict(
        #label        = "PI3 low-sep  m≪n",
        #m            = 50,
        #n            = 500,
        #sep_distance = 0.5,
        #class_ratios = [0.5, 0.5],
    #),

    # ── High separation ───────────────────────
    #"PI4_highsep_mggn": dict(
        #label        = "PI4 high-sep m≫n",
        #m            = 1000,
        #n            = 50,
        #sep_distance = 3.0,
        #class_ratios = [0.5, 0.5],
    #),
    #"PI5_highsep_balanced_small": dict(
        #label        = "PI5 high-sep m≫n (small)",
        #m            = 300,
        #n            = 20,
        #sep_distance = 3.0,
        #class_ratios = [0.5, 0.5],
    #),
    #"PI6_highsep_mlln": dict(
        #label        = "PI6 high-sep m≪n",
        #m            = 50,
        #n            = 500,
        #sep_distance = 3.0,
        #class_ratios = [0.5, 0.5],
    #),

    # ── Imbalanced ────────────────────────────
    #"PI7_lowsep_imbalanced": dict(
        #label        = "PI7 low-sep  imbalanced",
        #m            = 600,
        #n            = 50,
        #sep_distance = 0.5,
        #class_ratios = [0.8, 0.2],   # 80 % negative, 20 % positive
    #),
    #"PI8_highsep_imbalanced": dict(
        #label        = "PI8 high-sep imbalanced",
        #m            = 600,
        #n            = 50,
        #sep_distance = 3.0,
        #class_ratios = [0.8, 0.2],
    #),
#}

# Ordered list used for consistent iteration
DATASET_KEYS = list(DATASET_SPECS.keys())

# ─────────────────────────────────────────────
# Number of random initializations for the
# "initialization robustness" sub-experiment
# ─────────────────────────────────────────────
N_INIT_TRIALS = 20
SIGMA_FOR_INIT_EXPERIMENT = 1.0   # fix σ, vary w0

# ─────────────────────────────────────────────
# Convergence-diagnostic experiment
# (one representative seed + dataset per regime)
# ─────────────────────────────────────────────
CONVERGENCE_EXPERIMENTS = [
    # Low sep m<<n — most interesting, highest variance
    dict(dataset_key="PI3_lowsep_mlln", sigma=SIGMA_FOR_INIT_EXPERIMENT,  seed=SEEDS[0], label="PI3 low-sep m<<n, sigma=0.1"),
    dict(dataset_key="PI3_lowsep_mlln", sigma=1.0,  seed=SEEDS[0], label="PI3 low-sep m<<n, sigma=1.0"),
    
    # Low sep m>>n — Prox beats baselines
    dict(dataset_key="PI1_lowsep_mggn", sigma=1.0,  seed=SEEDS[0], label="PI1 low-sep m>>n, sigma=1.0"),
    
    # High sep m<<n — Prox hits 1.0, LibAUC struggles
    dict(dataset_key="PI6_highsep_mlln", sigma=1.0, seed=SEEDS[0], label="PI6 high-sep m<<n, sigma=1.0"),
]

CONVERGENCE_EXPERIMENTS = [
    # Same dataset, same seed, same sigma0 — only scale changes
    dict(dataset_key="PI3_lowsep_mlln", sigma=0.1, 
         sigma_scale=2.0, seed=SEEDS[0], label="PI3 scale=2.0 (current)"),
    dict(dataset_key="PI3_lowsep_mlln", sigma=0.1, 
         sigma_scale=1.5, seed=SEEDS[0], label="PI3 scale=1.5"),
    dict(dataset_key="PI3_lowsep_mlln", sigma=0.1, 
         sigma_scale=1.2, seed=SEEDS[0], label="PI3 scale=1.2"),
]

# ─────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────
RESULTS_DIR  = "results"
FIGURES_DIR  = "figures"
CONVERGENCE_DIR = "results/convergence"