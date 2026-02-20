import os
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from aucopt.optim.alm import run_alm
from aucopt.data.problem_instance import ProblemInstance

# ════════════════════════════════════════════════════════════════════════════
# FULL DATASET
# ════════════════════════════════════════════════════════════════════════════

def run_full_dataset(
    ds,
    AP,
    SP,
    LS,
    dataset_name,
    sigma_schedule,
    tau0,
    alpha0,
    save_weights,
    output_dir,
):
    """
    Runs ALM on the full dataset using a sigma continuation schedule.

    Rather than calling run_alm once at a fixed sigma0, this function calls
    run_alm once per level in sigma_schedule, warm-starting w and lambda from
    the solution of the previous level. This traverses the landscape from
    smooth (small sigma, large gamma, convex-like) to sharp (large sigma,
    small gamma, close to the true 0-1 loss), tracking the minimizer along
    the homotopy path rather than landing cold at a high sigma value.

    Because this is a full-dataset solve, PI.K is identical at every
    continuation level. Lambda transfers exactly between levels with no
    re-indexing complexity.

    Parameters:
    - ds              : object with .X and .y attributes (full training data)
    - AP, SP, LS      : ALM, SSN, and Line Search parameter objects
    - dataset_name    : str, used for saving results
    - sigma_schedule  : list of sigma values in increasing order, e.g.
                        [0.01, 0.1, 1.0, 10.0]. Each value is the sigma0
                        passed to run_alm at that level. run_alm's internal
                        sigma_scale growth still applies within each level.
                        Start small so gamma = 1/sigma is large and the
                        landscape is smooth on the first level.
    - tau0            : regularisation weight (same across all levels)
    - alpha0          : initial SSN step size (same across all levels)
    - save_weights    : save final weight vector to output_dir
    - output_dir      : path to directory where results are saved

    Returns:
    - w       : final learned weight vector
    - almvar  : ALMVar from the last continuation level
    - almlog  : ALMLog from the last continuation level
    """

    X = ds.X
    y = ds.y

    # Build the problem instance once — K and D are fixed for all levels
    PI = ProblemInstance(X, y)

    # ── Initial state ────────────────────────────────────────────────────────
    # w and lambda start from scratch at level 0. From level 1 onwards they
    # are warm-started from the previous level's solution.
    w   = PI.w0.copy()        # random init from ProblemInstance
    lam = PI.lambda0.copy()   # zeros

    almvar = None
    almlog = None

    # ── Continuation loop ────────────────────────────────────────────────────
    for level, sigma0 in enumerate(sigma_schedule):

        print(f"Continuation level {level + 1}/{len(sigma_schedule)}  "
              f"(sigma0={sigma0:.4f},  gamma0={1/sigma0:.4f})")

        # Warm-start this level from the previous solution.
        # On level 0 this is the random w and zero lambda from above.
        PI.w0      = w.copy()
        PI.lambda0 = lam.copy()

        # Lambda transfer is exact here — PI.K never changes between levels
        # because we are always solving the same full dataset.

        almvar, almlog = run_alm(sigma0, tau0, alpha0, PI, AP, SP, LS)

        # Carry w and lambda forward to the next level.
        # almvar.sigma is the grown sigma at the end of this run_alm call —
        # we do NOT carry it forward as sigma0 for the next level, because
        # the next level has its own sigma0 from the schedule. The schedule
        # controls the entry point; run_alm's internal growth controls what
        # happens within each level.
        w   = almvar.w.copy()
        lam = almvar.lambd.copy()

        print(f"  → done  |  L_final={almlog.L_final:.6f}  "
              f"|  ALM iters={almlog.alm_iter}  "
              f"|  grown sigma={almvar.sigma:.4f}")

    # ── Save learned weights ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if save_weights:
        weight_path = os.path.join(output_dir, f"{dataset_name}_w_full.csv")
        np.savetxt(weight_path, w, delimiter=",")
        print(f"✅ Saved final weights to {weight_path}")

    return w, almvar, almlog


# ════════════════════════════════════════════════════════════════════════════
# SGD / BATCHED
# ════════════════════════════════════════════════════════════════════════════

def run_sgd_dataset(
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
    alpha0,
    save_weights,
    output_dir,
):
    """
    Runs Prox-SGD (batched ALM) using disjoint batches on the given dataset.

    sigma0 should be set small (e.g. 0.1 or 0.01) so that the procedure
    begins in the smooth, well-conditioned regime (large gamma = wide Region 2)
    and naturally sharpens as sigma grows across batches.

    Parameters:
    - ds              : object with .X and .y attributes (training data)
    - AP, SP, LS      : ALM, SSN, and Line Search parameter objects
    - dataset_name    : str, used for saving results
    - n_epochs        : number of SGD epochs
    - n_batches       : batches per epoch
    - n_pos, n_neg    : samples per class in each batch
    - sigma0          : initial ALM penalty — start small, e.g. 0.01 or 0.1
    - tau0            : initial regularisation weight
    - alpha0          : initial SSN step size
    - save_weights    : save final weight vector to output_dir
    - output_dir      : path to directory where results are saved

    Fixes over original:
    1. Global dual dictionary: dual values keyed by (i,j) pair persist across
       all batches and epochs. Pairs seen before reuse their accumulated dual
       estimate; unseen pairs default to 0.0. This makes the SGD loop a
       genuine distributed ALM rather than independent restarts sharing only w.

    2. Sigma carry-forward: current_sigma is updated from almvar.sigma after
       each batch so the penalty grown inside run_alm is not discarded.
       sigma grows monotonically across all batches (the condition ALM theory
       requires for dual updates to drive constraint satisfaction), modulated
       by a per-epoch cooling factor of 0.95.
    """

    X = ds.X
    y = ds.y
    d = X.shape[0]
    w = np.random.randn(d)

    # ── Global dual dictionary ───────────────────────────────────────────────
    # Keys: (i, j) sample-pair tuples from PI.K.
    # Values: scalar lambda estimate for that pair.
    # Persists across all batches and epochs.
    global_lambda = {}

    # ── Live sigma and tau — global state for the entire SGD run ────────────
    # current_sigma starts at sigma0 (should be small) and grows via:
    #   - run_alm's internal sigma_scale at each ALM outer iteration
    #   - carry-forward of almvar.sigma after each batch
    #   - per-epoch decay of 0.95 to prevent unbounded growth
    # It is NEVER reset to sigma0 mid-run. It is a monotone (modulo decay)
    # global variable for the entire procedure, coupled to w.
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
              f"(current_sigma={current_sigma:.5f},  "
              f"gamma={1/current_sigma:.4f})")

        for _ in range(n_batches):
            pos_sample = pos_batches[batch_counter]
            neg_sample = neg_batches[batch_counter]
            selected   = np.concatenate([pos_sample, neg_sample])

            X_batch = X[:, selected]
            y_batch = y[selected]
            PI = ProblemInstance(X_batch, y_batch)

            # Warm-start primal from previous batch
            PI.w0 = w.copy()

            # Look up duals for this batch's pairs.
            # Seen pairs: reuse accumulated estimate.
            # Unseen pairs: 0.0, same as original behaviour.
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

            # Carry grown sigma forward — do not reset to current_sigma.
            # almvar.sigma is what run_alm grew it to across its 2 ALM iters.
            # This is the correct entry point for the next batch.
            current_sigma = almvar.sigma

            # Write duals back into global dict keyed by actual (i,j) pairs
            for idx, pair in enumerate(PI.K):
                global_lambda[pair] = almvar.lambd[idx]

            batch_counter += 1

        # Per-epoch cooling — applied to the live sigma, not the original sigma0.
        # This prevents unbounded growth while preserving the monotone trend.
        current_sigma *= 0.95
        current_tau   *= 0.95

    # ── Save learned weights ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    if save_weights:
        weight_path = os.path.join(output_dir, f"{dataset_name}_w_sgd.csv")
        np.savetxt(weight_path, w, delimiter=",")
        print(f"✅ Saved final weights to {weight_path}")

    return w