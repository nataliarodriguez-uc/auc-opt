import os
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from aucopt.optim.alm import run_alm
from aucopt.data.problem_instance import ProblemInstance

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
    alpha0,
    plot_weights,
    save_weights,
    output_dir
):
    """
    Runs Prox-SGD (batched ALM) using disjoint batches on the given dataset.

    Parameters:
    - ds: object with .X and .y attributes (training data)
    - AP, SP, LS: ALM, SSN, and Line Search parameter objects
    - dataset_name (str): name used for saving results
    - n_epochs (int): number of SGD epochs
    - n_batches (int): batches per epoch
    - n_pos, n_neg (int): samples per class in each batch
    - sigma0, tau0, alpha0 (float): initial ALM parameters
    - plot_weights (bool): show a bar plot of learned weights
    - save_weights (bool): save final weight vector to output_dir
    - output_dir (str): path to directory where results are saved
    """
    X = ds.X
    y = ds.y
    d = X.shape[0]
    w = np.random.randn(d)
    lam = None

    # === Prepare disjoint sampling plan ===
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    total_batches = n_epochs * n_batches
    assert len(pos_idx) >= total_batches * n_pos, "Not enough positive samples for disjoint batching"
    assert len(neg_idx) >= total_batches * n_neg, "Not enough negative samples for disjoint batching"

    pos_batches = np.array_split(pos_idx[:total_batches * n_pos], total_batches)
    neg_batches = np.array_split(neg_idx[:total_batches * n_neg], total_batches)

    batch_counter = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        for _ in range(n_batches):
            pos_sample = pos_batches[batch_counter]
            neg_sample = neg_batches[batch_counter]
            selected = np.concatenate([pos_sample, neg_sample])

            X_batch = X[:, selected]
            y_batch = y[selected]
            PI = ProblemInstance(X_batch, y_batch, mode='full')
            PI.w0 = w.copy()
            PI.lambda0 = np.zeros(len(PI.K))

            AP_batch = deepcopy(AP)
            AP_batch.max_iter_alm = 2

            almvar, _ = run_alm(sigma0, tau0, alpha0, PI, AP_batch, SP, LS)
            w = almvar.w.copy()
            lam = almvar.lambd

            batch_counter += 1

        sigma0 *= 0.95
        tau0 *= 0.95

    # Save learned weights
    os.makedirs(output_dir, exist_ok=True)
    if save_weights:
        weight_path = os.path.join(output_dir, f"{dataset_name}_w_sgd.csv")
        np.savetxt(weight_path, w, delimiter=",")
        print(f"✅ Saved final weights to {weight_path}")

    # Optional plot
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
