"""
Tier 1 Experiment Runner
=========================
Runs three sub-experiments and saves raw results as CSV/npy.

Sub-experiments
---------------
1. sigma_sensitivity   – AUC, wall-time, ALM-iters vs σ over N_SEEDS seeds
2. init_robustness     – AUC distribution over N_INIT_TRIALS random w0s (fixed σ)
3. convergence_diag    – Per-outer-iteration residual + SSN-iter traces

Usage
-----
    python experiment_runner.py [--exp all|sigma|init|conv] [--dataset KEY]

Results saved to RESULTS_DIR (defined in experiment_config.py).
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

# ── project imports (adjust sys.path if needed) ───────────────────────────────
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
    RESULTS_DIR, FIGURES_DIR,CONVERGENCE_DIR
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
    ds = DataSet_SVM(
        m            = spec["m"],
        n            = spec["n"],
        num_classes  = 2,
        class_ratios = spec["class_ratios"],
        sep_distance = spec["sep_distance"],
        seed         = seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        ds.X.T, ds.y,                    # sklearn expects (n_samples, n_features)
        train_size = TRAIN_RATIO,
        stratify   = ds.y,
        random_state = seed,
    )
    X_train, X_test = X_train.T, X_test.T   # back to (d, n) convention

    PI = ProblemInstance(X_train, y_train, seed=seed)

    # Attach test set manually so evaluate_auc() can find it
    PI.X_test = X_test
    PI.y_test = y_test

    return ds, PI


def evaluate_auc(w, PI):
    """Compute test AUC from a trained weight vector."""
    scores  = w @ PI.X_test
    return roc_auc_score(PI.y_test, scores)


def run_single(sigma, PI, seed):
    """
    Run one ALM solve.

    Parameters
    ----------
    sigma : float   penalty parameter σ
    PI    : ProblemInstance  (already constructed, w0/lambda0 will be set here)
    seed  : int     for reproducible w0

    Returns
    -------
    result : dict with keys auc, alm_iter, alm_time, L_final, ssn_iters, residuals
    """
    np.random.seed(seed)
    PI.w0      = np.random.randn(PI.d)
    PI.lambda0 = np.zeros(PI.n_pairs)

    AP, SP, LS = make_params()
    t0 = time.time()
    almvar, almlog = run_alm(sigma, TAU0, ALPHA0, PI, AP, SP, LS)
    elapsed = time.time() - t0

    T = almlog.alm_iter  # actual iterations used

    # Constraint residual trace (inf-norm per outer iteration)
    residuals = []
    for t in range(T):
        # Reconstruct from ssn_times being non-zero as a proxy for "used"
        # The residual itself is stored in almvar only at the end;
        # we track it via a patched ALM – see ConvergenceTracker below.
        residuals.append(None)  # placeholder; see ConvergenceTracker

    return dict(
        auc       = evaluate_auc(almvar.w, PI),
        alm_iter  = T,
        alm_time  = almlog.alm_time,
        L_final   = almlog.L_final,
        ssn_iters = almlog.ssn_iters[:T].tolist(),
        w         = almvar.w.copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patched ALM that records residuals per outer iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_alm_with_diagnostics(sigma, PI, seed, sigma_scale = 2.0):
    """
    Wraps run_alm but additionally captures the constraint-residual
    and objective value at every outer ALM iteration.

    Returns
    -------
    result : dict   (same as run_single + residual_trace, obj_trace, ssn_iter_trace)
    """
    import warnings
    from aucopt.optim.variables import ALMVar, ProxVar, SSNVar, ALMLog
    from aucopt.optim.ssn import run_ssn
    from aucopt.optim.parameters import (
        update_tol, update_iter, update_proxmethod, update_sigma_gamma
    )

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    np.random.seed(seed)
    PI.w0      = np.random.randn(PI.d)
    PI.lambda0 = np.zeros(PI.n_pairs)

    AP, SP, LS = make_params()
    SP_local = deepcopy(SP)
    AP_local = deepcopy(AP)
    LS_local = deepcopy(LS)

    K_len     = PI.n_pairs
    inv_K_len = 1.0 / K_len
    AP_local.sigma_scale = sigma_scale

    almlog = ALMLog(AP_local.max_iter_alm, SP_local.max_iter_ssn, LS_local.max_iter_ls)
    almlog.alm_time = time.time()

    almvar        = ALMVar(TAU0, sigma, PI)
    almvar.lambd  = PI.lambda0.copy()
    almvar.sigma  = sigma
    almvar.tau    = TAU0
    almvar.w[:]   = PI.w0
    almvar.y      = np.zeros(K_len)
    almvar.alpha  = ALPHA0

    ssnvar        = SSNVar(PI)
    ssnvar.w_ssn[:] = PI.w0
    proxvar       = ProxVar(PI.d, K_len, almvar.tau)

    temp_res      = np.empty(K_len)

    residual_trace  = []
    obj_trace       = []
    ssn_iter_trace  = []

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
        auc             = evaluate_auc(almvar.w, PI),
        alm_iter        = almlog.alm_iter,
        alm_time        = almlog.alm_time,
        L_final         = almlog.L_final,
        residual_trace  = residual_trace,
        obj_trace       = obj_trace,
        ssn_iter_trace  = ssn_iter_trace,
        w               = almvar.w.copy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 1: σ Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_sigma_sensitivity(dataset_keys=None, sigma_grid=None, seeds=None, verbose=True):
    
    """
    For every (dataset, σ, seed) triple, run one ALM solve and record:
        dataset_key, sigma, gamma, seed, auc, alm_iter, alm_time, L_final

    Saves results/tier1/sigma_sensitivity.csv
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
                    dataset_key = dk,
                    dataset_label = spec["label"],
                    m           = spec["m"],
                    n           = spec["n"],
                    sep         = "high" if spec["sep_distance"] >= 2.0 else "low",
                    regime      = "m>>n" if spec["m"] > spec["n"] else "m<<n",
                    sigma       = sigma,
                    gamma       = round(1.0 / sigma, 6),
                    seed        = seed,
                    auc         = res["auc"],
                    alm_iter    = res["alm_iter"],
                    alm_time    = res["alm_time"],
                    L_final     = res["L_final"],
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
    Fix σ and a dataset.  Run N_INIT_TRIALS with different random w0 seeds.
    The dataset itself is generated with a single fixed seed (data_seed).

    Saves results/tier1/init_robustness.csv
    """
    if dataset_keys is None: dataset_keys = DATASET_KEYS
    if sigma        is None: sigma        = SIGMA_FOR_INIT_EXPERIMENT
    if n_trials     is None: n_trials     = N_INIT_TRIALS
    if data_seed    is None: data_seed    = SEEDS[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    # Use a dense set of init seeds
    init_seeds = list(range(1000, 1000 + n_trials))

    for dk in dataset_keys:
        spec = DATASET_SPECS[dk]
        _, PI = build_dataset(spec, data_seed)   # fixed data

        for trial, iseed in enumerate(init_seeds):
            res = run_alm_with_diagnostics(sigma, PI, iseed)

            records.append(dict(
                dataset_key   = dk,
                dataset_label = spec["label"],
                sigma         = sigma,
                trial         = trial,
                init_seed     = iseed,
                auc           = res["auc"],
                alm_iter      = res["alm_iter"],
                alm_time      = res["alm_time"],
                converged     = int(res["alm_iter"] < ALM_DEFAULTS["max_iter_alm"]),
            ))

            if verbose:
                print(f"  {dk:35s} trial {trial+1:3d}/{n_trials}  "
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

    Saves
    -----
    results/tier1/convergence_<label>.csv  (one per experiment)
    results/tier1/convergence_summary.csv  (one row per experiment)
    """
    if experiments is None: experiments = CONVERGENCE_EXPERIMENTS
    os.makedirs(CONVERGENCE_DIR, exist_ok=True)

    summary = []

    for exp in experiments:
        dk    = exp["dataset_key"]
        sigma = exp["sigma"]
        seed  = exp["seed"]
        sigma_scale = exp.get("sigma_scale", 2.0)
        label = exp["label"].replace(" ", "_").replace(",", "").replace("=", "")

        spec  = DATASET_SPECS[dk]
        _, PI = build_dataset(spec, seed)

        if verbose:
            print(f"  Convergence diag: {exp['label']} ...")

        res = run_alm_with_diagnostics(sigma, PI, seed, sigma_scale = sigma_scale)

        T = len(res["residual_trace"])
        df_trace = pd.DataFrame({
            "iteration"    : list(range(1, T + 1)),
            "residual_inf" : res["residual_trace"],
            "obj_value"    : res["obj_trace"],
            "ssn_iters"    : res["ssn_iter_trace"],
            "sigma_t"      : [sigma * (ALM_DEFAULTS["sigma_scale"] ** t) for t in range(T)],
        })
        trace_path = os.path.join(CONVERGENCE_DIR, f"convergence_{label}.csv")
        df_trace.to_csv(trace_path, index=False)

        summary.append(dict(
            label       = exp["label"],
            dataset_key = dk,
            sigma       = sigma,
            sigma_scale = sigma_scale,
            seed        = seed,
            auc         = res["auc"],
            alm_iter    = res["alm_iter"],
            alm_time    = res["alm_time"],
            converged   = int(res["alm_iter"] < ALM_DEFAULTS["max_iter_alm"]),
            trace_file  = trace_path,
        ))

        if verbose:
            print(f"    → AUC={res['auc']:.4f}  iter={res['alm_iter']}  "
                  f"time={res['alm_time']:.2f}s")

    df_sum = pd.DataFrame(summary)
    sum_path = os.path.join(CONVERGENCE_DIR, "convergence_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n✅ Saved convergence summary → {sum_path}\n")
    return df_sum

# ─────────────────────────────────────────────────────────────────────────────
# Sub-experiment 4: Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(verbose=True):
    """
    Run BCE and LibAUC once per dataset per seed (no sigma dependence).
    Saves results/tier1/baselines.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    for dk in DATASET_KEYS:
        spec = DATASET_SPECS[dk]
        for seed in SEEDS:
            _, PI = build_dataset(spec, seed)

            # BCE expects (n_samples, n_features) so transpose
            X_train = PI.X.T
            X_test  = PI.X_test.T

            bce_auc, _    = evaluate_pytorch_bce(X_train, X_test, PI.y, PI.y_test)
            libauc_result = evaluate_libauc(X_train, X_test, PI.y, PI.y_test)
            libauc_auc    = libauc_result[0] if isinstance(libauc_result, tuple) else None

            records.append(dict(
                dataset_key   = dk,
                dataset_label = spec["label"],
                seed          = seed,
                bce_auc       = bce_auc,
                libauc_auc    = libauc_auc,
            ))

            if verbose:
                print(f"  {dk:35s} seed={seed}  "
                      f"BCE={bce_auc:.4f}  LibAUC={libauc_auc:.4f}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "baselines.csv")
    df.to_csv(out, index=False)
    print(f"\n✅ Saved baselines -> {out}\n")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",
                        default="all",
                        choices=["all", "sigma", "init", "conv"],
                        help="Which sub-experiment to run")
    parser.add_argument("--dataset",
                        default=None,
                        help="Run on a single dataset key (default: all)")
    args = parser.parse_args()

    dk_filter = [args.dataset] if args.dataset else None

    print("=" * 60)
    print("  Tier 1 Experiments — SVM Synthetic")
    print("=" * 60)

    if args.exp in ("all", "sigma"):
        print("\n[1/3] σ Sensitivity")
        run_sigma_sensitivity(dataset_keys=dk_filter, verbose=True)

    if args.exp in ("all", "init"):
        print("\n[2/3] Initialization Robustness")
        run_init_robustness(dataset_keys=dk_filter, verbose=True)

    if args.exp in ("all", "conv"):
        print("\n[3/3] Convergence Diagnostics")
        run_convergence_diagnostics(verbose=True)

    print("\n✅ All requested experiments finished.")