import os
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from aucopt.optim.alm import run_alm
from aucopt.data.problem_instance import ProblemInstance
from initialize import WARM_START_REGISTRY

# ════════════════════════════════════════════════════════════════════════════
# FULL DATASET
# ════════════════════════════════════════════════════════════════════════════

def run_alm_on_full_dataset(
    ds,
    AP,
    SP,
    LS,
    dataset_name,
    sigma0,
    tau0,
    alpha0,
    warm_start,
    plot_weights,
    save_weights,
    output_dir,
):
    """
    Runs ALM on the full dataset as a single solve.

    Initialisation strategy
    -----------------------
    The primal is warm-started via the strategy named by the warm_start
    parameter. Available strategies are the keys of WARM_START_REGISTRY
    in warm_start.py — currently "lr" (logistic regression) and "lda"
    (linear discriminant analysis). Both are convex or closed-form solves
    that place w0 in a geometrically sensible basin before the first SSN
    step, addressing initialisation sensitivity without adding ALM
    parameters or continuation overhead.

    The sigma0 passed here should be small (e.g. 0.1) so that gamma = 1/sigma
    is large at the start and the landscape is smooth. run_alm's internal
    sigma_scale growth then sharpens the problem reactively as constraints
    tighten — no explicit schedule needed.

    Parameters
    ----------
    ds           : object with .X (d x n) and .y (n,) attributes
    AP, SP, LS   : ALM, SSN, and Line Search parameter objects
    dataset_name : str, used for saving results
    sigma0       : initial ALM penalty — start small, e.g. 0.1
    tau0         : regularisation weight — held fixed throughout
    alpha0       : initial SSN step size
    warm_start   : strategy key, one of WARM_START_REGISTRY — "lr" or "lda"
    plot_weights : show a bar plot of learned weights at the end
    save_weights : save final weight vector to output_dir
    output_dir   : path to directory where results are saved

    Returns
    -------
    w      : final learned weight vector
    almvar : ALMVar from the solve
    almlog : ALMLog from the solve
    """

    X = ds.X
    y = ds.y

    # ── Warm-start w from chosen strategy ───────────────────────────────────
    PI    = ProblemInstance(X, y)
    PI.w0 = WARM_START_REGISTRY[warm_start](X, y)
    # lambda0 stays at zeros — no prior dual information for a fresh full solve

    # ── Single ALM solve ─────────────────────────────────────────────────────
    # run_alm's internal residual test grows sigma reactively.
    # tau0 is fixed — it is a property of the problem, not a schedule parameter.
    print(f"Running ALM  (sigma0={sigma0:.4f},  gamma0={1/sigma0:.4f})")
    almvar, almlog = run_alm(sigma0, tau0, alpha0, PI, AP, SP, LS)

    w = almvar.w.copy()
    print(f"  → done  |  L_final={almlog.L_final:.6f}"
          f"  |  ALM iters={almlog.alm_iter}"
          f"  |  grown sigma={almvar.sigma:.4f}")

    # ── Save learned weights ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if save_weights:
        weight_path = os.path.join(output_dir, f"{dataset_name}_w_full.csv")
        np.savetxt(weight_path, w, delimiter=",")
        print(f"✅ Saved final weights to {weight_path}")

    # ── Optional plot ────────────────────────────────────────────────────────
    if plot_weights:
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(w)), w)
        plt.title(f"Final Learned Weights — {dataset_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return w, almvar, almlog


# ════════════════════════════════════════════════════════════════════════════
# SGD / BATCHED
# ════════════════════════════════════════════════════════════════════════════

def run_prox_sgd_on_dataset(
    ds,
    AP,
    SP,
    LS,
    dataset_name,
    n_epochs,
    n_batches,
    n_pos,
    n_neg,
    sigma0,
    tau0,
    tau_min,
    alpha0,
    warm_start,
    plot_weights,
    save_weights,
    output_dir,
):
    """
    Runs Prox-SGD (batched ALM) using disjoint batches on the given dataset.

    Initialisation strategy
    -----------------------
    w is warm-started via the strategy named by the warm_start parameter
    before the SGD loop begins. Available strategies are the keys of
    WARM_START_REGISTRY in warm_start.py — currently "lr" and "lda".

    Parameter schedules
    -------------------
    sigma and tau are decoupled because they serve different roles:

      current_sigma  starts at sigma0, grows reactively inside each run_alm
                     call, and is carried forward across batches. A per-epoch
                     decay of 0.95 prevents unbounded growth that would empty
                     Region 2 and collapse the Hessian. sigma is a constraint
                     enforcement parameter — it must grow monotonically to
                     drive constraint satisfaction.

      current_tau    starts at tau0 and decays slowly (0.99/epoch) to a floor
                     of tau_min. tau is a regularisation parameter — it
                     provides the guaranteed positive-definite τ·I floor in
                     the Hessian. Decaying it at the same rate as sigma would
                     keep τ/σ constant, preventing the loss from ever dominating
                     the regulariser at convergence. The floor ensures the
                     Hessian never becomes singular when Region 2 empties.

    Fixes over original
    -------------------
    1. LR warm start: w initialised from logistic regression, not randn.

    2. Global dual dictionary: dual values keyed by (i,j) pair persist across
       all batches and epochs. Pairs seen before reuse their accumulated dual
       estimate; unseen pairs default to 0.0. This makes the SGD loop a
       genuine distributed ALM rather than independent restarts sharing only w.

    3. Sigma carry-forward: current_sigma is updated from almvar.sigma after
       each batch so the penalty grown inside run_alm is not discarded.
       sigma grows monotonically across all batches, modulated by a per-epoch
       decay of 0.95 to prevent unbounded growth.

    4. Tau decoupled: current_tau decays slowly and independently of sigma,
       with a floor at tau_min. This keeps τ/σ shrinking naturally as sigma
       grows, allowing the loss to progressively dominate the regulariser.

    Parameters
    ----------
    ds           : object with .X (d x n) and .y (n,) attributes
    AP, SP, LS   : ALM, SSN, and Line Search parameter objects
    dataset_name : str, used for saving results
    n_epochs     : number of SGD epochs
    n_batches    : batches per epoch
    n_pos, n_neg : samples per class in each batch
    sigma0       : initial ALM penalty — start small, e.g. 0.1
    tau0         : initial regularisation weight
    tau_min      : floor on tau — prevents Hessian singularity, e.g. 0.01
    alpha0       : initial SSN step size
    warm_start   : strategy key, one of WARM_START_REGISTRY — "lr" or "lda"
    plot_weights : show a bar plot of learned weights at the end
    save_weights : save final weight vector to output_dir
    output_dir   : path to directory where results are saved

    Returns
    -------
    w : final learned weight vector
    """

    X = ds.X
    y = ds.y

    # ── Warm-start w from chosen strategy ───────────────────────────────────
    w = WARM_START_REGISTRY[warm_start](X, y)

    # ── Global dual dictionary ───────────────────────────────────────────────
    # Keys: (i, j) sample-pair tuples from PI.K.
    # Values: scalar lambda estimate for that pair.
    # Persists across all batches and epochs.
    global_lambda = {}

    # ── Live sigma and tau ───────────────────────────────────────────────────
    # current_sigma: grows via run_alm's internal residual test, carried
    #   forward across batches, decayed 0.95/epoch to prevent collapse.
    #   Never reset to sigma0 mid-run.
    #
    # current_tau: decays slowly at 0.99/epoch to a floor of tau_min.
    #   Decoupled from sigma so that τ/σ shrinks naturally as sigma grows.
    current_sigma = sigma0
    current_tau   = tau0

    # ── Prepare disjoint sampling plan ──────────────────────────────────────
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    total_batches = n_epochs * n_batches
    assert len(pos_idx) >= total_batches * n_pos, \
        "Not enough positive samples for disjoint batching"
    assert len(neg_idx) >= total_batches * n_neg, \
        "Not enough negative samples for disjoint batching"

    pos_batches = np.array_split(pos_idx[:total_batches * n_pos], total_batches)
    neg_batches = np.array_split(neg_idx[:total_batches * n_neg], total_batches)

    # ── Main SGD loop ────────────────────────────────────────────────────────
    batch_counter = 0

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}  "
              f"(sigma={current_sigma:.5f}  "
              f"gamma={1/current_sigma:.4f}  "
              f"tau={current_tau:.5f})")

        for _ in range(n_batches):
            pos_sample = pos_batches[batch_counter]
            neg_sample = neg_batches[batch_counter]
            selected   = np.concatenate([pos_sample, neg_sample])

            X_batch = X[:, selected]
            y_batch = y[selected]
            PI      = ProblemInstance(X_batch, y_batch)

            # Warm-start primal from previous batch
            PI.w0 = w.copy()

            # Look up duals for this batch's pairs.
            # Seen pairs: reuse accumulated estimate.
            # Unseen pairs: 0.0 — same as original behaviour for new pairs.
            PI.lambda0 = np.array([
                global_lambda.get(pair, 0.0)
                for pair in PI.K
            ])

            AP_batch = deepcopy(AP)
            AP_batch.max_iter_alm = 2

            almvar, _ = run_alm(
                current_sigma, current_tau, alpha0, PI, AP_batch, SP, LS
            )

            # Update primal
            w = almvar.w.copy()

            # Carry grown sigma forward.
            # almvar.sigma is the value run_alm grew to across its 2 outer
            # iterations. Discarding it would reset constraint enforcement and
            # break the monotonicity ALM dual convergence requires.
            current_sigma = almvar.sigma

            # Write duals back into global dict keyed by (i,j) pairs
            for idx, pair in enumerate(PI.K):
                global_lambda[pair] = almvar.lambd[idx]

            batch_counter += 1

        # ── Per-epoch parameter updates ──────────────────────────────────────
        # sigma: 0.95 cooling on the live value prevents unbounded growth.
        current_sigma *= 0.95

        # tau: slow independent decay toward floor.
        # 0.99 << 0.95 so tau/sigma shrinks over time — the loss gradually
        # dominates the regulariser, which is the correct direction at convergence.
        current_tau = max(tau_min, current_tau * 0.99)

    # ── Save learned weights ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if save_weights:
        weight_path = os.path.join(output_dir, f"{dataset_name}_w_sgd.csv")
        np.savetxt(weight_path, w, delimiter=",")
        print(f"✅ Saved final weights to {weight_path}")

    # ── Optional plot ────────────────────────────────────────────────────────
    if plot_weights:
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(w)), w)
        plt.title(f"Final Learned Weights — {dataset_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return w