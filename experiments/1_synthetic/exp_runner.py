"""
Tier 1 Experiment Runner
=========================
Runs four sub-experiments and saves raw results as CSV.

Sub-experiments
---------------
1. sigma_sensitivity   – AUC, wall-time, ALM-iters vs σ over N_SEEDS seeds
2. init_robustness     – AUC distribution over N_INIT_TRIALS random w0s +
                         warm start strategies (LR, LDA) on fixed dataset/σ
3. convergence_diag    – Per-outer-iteration residual + SSN-iter traces
4. baselines           – BCE and LibAUC on same datasets/seeds (σ-independent)

Usage
-----
    python exp_runner.py [--exp all|sigma|init|conv|baselines] [--dataset KEY]

Results saved to RESULTS_DIR (defined in exp_config.py).
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from aucopt.data.problem_svmdata import DataSet_SVM
from aucopt.data.problem_instance import ProblemInstance
from aucopt.optim.alm import run_alm
from aucopt.optim.variables import ALMParameters, SSNParameters, LineSearchParameters
from aucopt.eval.baselines import evaluate_pytorch_bce, evaluate_libauc
from sklearn.metrics import roc_auc_score

from exp_config import (
    SEEDS, SIGMA_GRID, SIGMA_FOCAL,
    ALM_DEFAULTS, SSN_DEFAULTS, LS_DEFAULTS,
    TAU0, ALPHA0, TRAIN_RATIO,
    DATASET_SPECS, DATASET_KEYS,
    N_INIT_TRIALS, SIGMA_FOR_INIT_EXPERIMENT,
    CONVERGENCE_EXPERIMENTS,
    RESULTS_DIR, FIGURES_DIR, CONVERGENCE_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_params():
    """Return fresh ALM / SSN / LS parameter objects from config defaults."""
    AP = ALMParameters(**ALM_DEFAULTS)
    SP = SSNParameters(**SSN_DEFAULTS)
    LS = LineSearchParameters(**LS_DEFAULTS)
    return AP, SP, LS


def build_dataset(spec, seed):
    """
    Build a DataSet_SVM and a ProblemInstance with train/test split.

    Returns
    -------
    ds : DataSet_SVM  (full data)
    PI : ProblemInstance with PI.X_test and PI.y_test attached
    """
    ds = DataSet_SVM(
        m            = spec["m"],
        n            = spec["n"],
        num_classes  = 2,
        class_ratios = spec["class_ratios"],
        sep_distance = spec["sep_distance"],
        seed         = seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        ds.X.T, ds.y,
        train_size   = TRAIN_RATIO,
        stratify     = ds.y,
        random_state = seed,
    )
    X_train, X_test = X_train.T, X_test.T   # back to (d, n) convention

    PI = ProblemInstance(X_train, y_train, seed=seed)

    # Attach test set so evaluate_auc() can find it
    PI.X_test = X_test
    PI.y_test = y_test

    return ds, PI


def evaluate_auc(w, PI):
    """Compute test AUC from a trained weight vector."""
    scores = w @ PI.X_test
    return roc_auc_score(PI.y_test, scores)


# ─────────────────────────────────────────────────────────────────────────────
# Core ALM solver with full diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_alm_with_diagnostics(sigma, PI, seed,
                             sigma_scale=None,
                             use_provided_w0=False):
    """
    Runs ALM while capturing constraint residual, objective value, and SSN
    inner iteration count at every outer ALM iteration.

    Parameters
    ----------
    sigma           : float   initial penalty parameter σ
    PI              : ProblemInstance
    seed            : int     used for random w0 (ignored if use_provided_w0=True)
    sigma_scale     : float   penalty scaling rate (default: ALM_DEFAULTS value)
    use_provided_w0 : bool    if True, PI.w0 is used as-is (warm start set externally)

    Returns
    -------
    dict with keys:
        auc, alm_iter, alm_time, L_final,
        residual_trace, obj_trace, ssn_iter_trace, w
    """
    import warnings
    from aucopt.optim.variables import ALMVar, ProxVar, SSNVar, ALMLog
    from aucopt.optim.ssn import run_ssn
    from aucopt.optim.parameters import (
        update_tol, update_iter, update_proxmethod, update_sigma_gamma
    )

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ── Initialization ─────────────────────────────────────────────
    if use_provided_w0:
        # warm start already set on PI.w0 externally — use it as-is
        pass
    else:
        np.random.seed(seed)
        PI.w0 = np.random.randn(PI.d)

    PI.lambda0 = np.zeros(PI.n_pairs)

    # ── Parameters ─────────────────────────────────────────────────
    AP, SP, LS = make_params()
    SP_local   = deepcopy(SP)
    AP_local   = deepcopy(AP)
    LS_local   = deepcopy(LS)

    if sigma_scale is not None:
        AP_local.sigma_scale = sigma_scale

    K_len     = PI.n_pairs
    inv_K_len = 1.0 / K_len

    # ── ALM state ──────────────────────────────────────────────────
    almlog = ALMLog(AP_local.max_iter_alm, SP_local.max_iter_ssn, LS_local.max_iter_ls)
    almlog.alm_time = time.time()

    almvar          = ALMVar(TAU0, sigma, PI)
    almvar.lambd    = PI.lambda0.copy()
    almvar.sigma    = sigma
    almvar.tau      = TAU0
    almvar.w[:]     = PI.w0
    almvar.y        = np.zeros(K_len)
    almvar.alpha    = ALPHA0

    ssnvar          = SSNVar(PI)
    ssnvar.w_ssn[:] = PI.w0
    proxvar         = ProxVar(PI.d, K_len, almvar.tau)
    temp_res        = np.empty(K_len)

    residual_trace  = []
    obj_trace       = []
    ssn_iter_trace  = []

    # ── Main ALM loop ──────────────────────────────────────────────
    for t in range(AP_local.max_iter_alm):
        update_tol(SP_local, t)
        update_iter(SP_local, t)
        update_proxmethod(almvar)

        t_ssn = time.time()
        run_ssn(t, almlog, almvar, ssnvar, proxvar, PI, SP_local, LS_local)
        almlog.ssn_times[t] = time.time() - t_ssn

        almvar.w[:]   = ssnvar.w_ssn
        almvar.y[:]   = ssnvar.y_ssn
        almvar.w_D[:] = ssnvar.w_ssn_D

        np.subtract(almvar.y, almvar.w_D, out=temp_res)
        np.multiply(temp_res, inv_K_len, out=almvar.cons_condition)

        cons_norm = float(np.linalg.norm(almvar.cons_condition, ord=np.inf))
        obj_val   = float(ssnvar.L_obj * inv_K_len)

        residual_trace.append(cons_norm)
        obj_trace.append(obj_val)
        ssn_iter_trace.append(int(almlog.ssn_iters[t]))

        if cons_norm <= AP_local.tol_alm:
            almlog.alm_iter = t + 1
            almlog.L_final  = obj_val
            break
        else:
            update_sigma_gamma(almvar, AP_local)
            almvar.lambd += almvar.sigma * inv_K_len * temp_res
    else:
        almlog.alm_iter = AP_local.max_iter_alm

    almlog.alm_time = time.time() - almlog.alm_time

    return dict(
        auc            = evaluate_auc(almvar.w, PI),
        alm_iter       = almlog.alm_iter,
        alm_time       = almlog.alm_time,
        L_final        = almlog.L_final,
        residual_trace = residual_trace,
        obj_trace      = obj_trace,
        ssn_iter_trace = ssn_iter_trace,
        w              = almvar.w.copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 1: σ Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_sigma_sensitivity(dataset_keys=None, sigma_grid=None, seeds=None, verbose=True):
    """
    For every (dataset, σ, seed) triple, run one ALM solve and record:
        dataset_key, sigma, gamma, seed, auc, alm_iter, alm_time, L_final

    Saves results/sigma_sensitivity.csv
    """
    if dataset_keys is None: dataset_keys = DATASET_KEYS
    if sigma_grid   is None: sigma_grid   = SIGMA_GRID
    if seeds        is None: seeds        = SEEDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    total = len(dataset_keys) * len(sigma_grid) * len(seeds)
    done  = 0

    for dk in dataset_keys:
        spec = DATASET_SPECS[dk]
        for sigma in sigma_grid:
            for seed in seeds:
                _, PI = build_dataset(spec, seed)
                res   = run_alm_with_diagnostics(sigma, PI, seed)

                records.append(dict(
                    dataset_key   = dk,
                    dataset_label = spec["label"],
                    m             = spec["m"],
                    n             = spec["n"],
                    imbalance     = spec.get("imbalance", "50/50"),
                    sep           = "high" if spec["sep_distance"] >= 2.0 else "low",
                    regime        = "m>>n" if spec["m"] > spec["n"] else "m<<n",
                    sigma         = sigma,
                    gamma         = round(1.0 / sigma, 6),
                    seed          = seed,
                    auc           = res["auc"],
                    alm_iter      = res["alm_iter"],
                    alm_time      = res["alm_time"],
                    L_final       = res["L_final"],
                ))

                done += 1
                if verbose:
                    print(f"  [{done}/{total}] {dk:35s} σ={sigma:.3f}  "
                          f"seed={seed}  AUC={res['auc']:.4f}  "
                          f"iter={res['alm_iter']}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "sigma_sensitivity.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ Saved sigma sensitivity → {out}  ({len(df)} rows)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 2: Initialization Robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_init_robustness(dataset_keys=None, sigma=None, n_trials=None,
                        data_seed=None, verbose=True):
    """
    Fix σ and dataset. Run N_INIT_TRIALS with different random w0 seeds,
    then run each warm start strategy (LR, LDA) for direct comparison.
    Dataset is fixed via data_seed — only w0 changes across trials.

    Results tagged with init_type: "random", "lr", or "lda"

    Saves results/init_robustness.csv
    """
    if dataset_keys is None: dataset_keys = DATASET_KEYS
    if sigma        is None: sigma        = SIGMA_FOR_INIT_EXPERIMENT
    if n_trials     is None: n_trials     = N_INIT_TRIALS
    if data_seed    is None: data_seed    = SEEDS[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []
    init_seeds = list(range(1000, 1000 + n_trials))

    for dk in dataset_keys:
        spec = DATASET_SPECS[dk]
        _, PI = build_dataset(spec, data_seed)   # fixed dataset, never changes

        # ── Random initializations ─────────────────────────────────
        for trial, iseed in enumerate(init_seeds):
            res = run_alm_with_diagnostics(sigma, PI, seed=iseed,
                                           use_provided_w0=False)
            records.append(dict(
                dataset_key   = dk,
                dataset_label = spec["label"],
                imbalance     = spec.get("imbalance", "50/50"),
                sigma         = sigma,
                trial         = trial,
                init_seed     = iseed,
                init_type     = "random",
                auc           = res["auc"],
                alm_iter      = res["alm_iter"],
                alm_time      = res["alm_time"],
                converged     = int(res["alm_iter"] < ALM_DEFAULTS["max_iter_alm"]),
            ))
            if verbose:
                print(f"  {dk:35s} random trial {trial+1:3d}/{n_trials}  "
                      f"AUC={res['auc']:.4f}  iter={res['alm_iter']}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "init_robustness.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ Saved init robustness → {out}  ({len(df)} rows)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 3: Convergence Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_convergence_diagnostics(experiments=None, verbose=True):
    """
    For each entry in CONVERGENCE_EXPERIMENTS, run the patched ALM and
    save the per-iteration residual + SSN-iter traces.

    Supports optional warm_start key in experiment dict.
    If warm_start is set ("lr" or "lda"), PI.w0 is set from the registry
    instead of random initialization.

    Saves
    -----
    results/convergence/convergence_<label>.csv  (one per experiment)
    results/convergence/convergence_summary.csv  (one row per experiment)
    """
    if experiments is None: experiments = CONVERGENCE_EXPERIMENTS
    os.makedirs(CONVERGENCE_DIR, exist_ok=True)

    summary = []

    for exp in experiments:
        dk          = exp["dataset_key"]
        sigma       = exp["sigma"]
        seed        = exp["seed"]
        sigma_scale = exp.get("sigma_scale", ALM_DEFAULTS["sigma_scale"])
        warm_start  = exp.get("warm_start", None)
        label       = exp["label"].replace(" ", "_").replace(",", "").replace("=", "")

        spec  = DATASET_SPECS[dk]
        _, PI = build_dataset(spec, seed)

        if verbose:
            ws_str = f"warm={warm_start}" if warm_start else "random init"
            print(f"  Convergence diag: {exp['label']}  ({ws_str}) ...")

    
        res   = run_alm_with_diagnostics(sigma, PI, seed,
                                             sigma_scale=sigma_scale,
                                             use_provided_w0=False)

        T = len(res["residual_trace"])
        df_trace = pd.DataFrame({
            "iteration"    : list(range(1, T + 1)),
            "residual_inf" : res["residual_trace"],
            "obj_value"    : res["obj_trace"],
            "ssn_iters"    : res["ssn_iter_trace"],
            "sigma_t"      : [sigma * (sigma_scale ** t) for t in range(T)],
            "warm_start"   : warm_start if warm_start else "random",
        })
        trace_path = os.path.join(CONVERGENCE_DIR, f"convergence_{label}.csv")
        df_trace.to_csv(trace_path, index=False)

        summary.append(dict(
            label       = exp["label"],
            dataset_key = dk,
            sigma       = sigma,
            sigma_scale = sigma_scale,
            warm_start  = warm_start if warm_start else "random",
            seed        = seed,
            auc         = res["auc"],
            alm_iter    = res["alm_iter"],
            alm_time    = res["alm_time"],
            converged   = int(res["alm_iter"] < ALM_DEFAULTS["max_iter_alm"]),
            trace_file  = trace_path,
        ))

        if verbose:
            print(f"    → AUC={res['auc']:.4f}  iter={res['alm_iter']}  "
                  f"time={res['alm_time']:.2f}s  scale={sigma_scale}  "
                  f"init={warm_start if warm_start else 'random'}")

    df_sum = pd.DataFrame(summary)
    sum_path = os.path.join(CONVERGENCE_DIR, "convergence_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n✅ Saved convergence summary → {sum_path}\n")
    return df_sum


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 4: Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(dataset_keys=None, seeds=None, verbose=True):
    """
    Run BCE and LibAUC once per (dataset, seed). σ-independent.
    Saves results/baselines.csv
    """
    if dataset_keys is None: dataset_keys = DATASET_KEYS
    if seeds        is None: seeds        = SEEDS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    total = len(dataset_keys) * len(seeds)
    done  = 0

    for dk in dataset_keys:
        spec = DATASET_SPECS[dk]
        for seed in seeds:
            _, PI = build_dataset(spec, seed)

            # baselines expect (n_samples, n_features)
            X_train = PI.X.T
            X_test  = PI.X_test.T

            bce_auc, _    = evaluate_pytorch_bce(X_train, X_test, PI.y, PI.y_test)
            libauc_result = evaluate_libauc(X_train, X_test, PI.y, PI.y_test)
            libauc_auc    = libauc_result[0] if isinstance(libauc_result, tuple) else libauc_result

            records.append(dict(
                dataset_key   = dk,
                dataset_label = spec["label"],
                imbalance     = spec.get("imbalance", "50/50"),
                seed          = seed,
                bce_auc       = bce_auc,
                libauc_auc    = libauc_auc,
            ))

            done += 1
            if verbose:
                print(f"  [{done}/{total}] {dk:35s} seed={seed}  "
                      f"BCE={bce_auc:.4f}  LibAUC={libauc_auc:.4f}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "baselines.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ Saved baselines → {out}  ({len(df)} rows)\n")
    return df
