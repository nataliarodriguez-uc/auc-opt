import numpy as np
from sklearn.linear_model import LogisticRegression

# ════════════════════════════════════════════════════════════════════════════
# Binary classification strategies
# ════════════════════════════════════════════════════════════════════════════

def get_warm_start_w_lr(X, y, C=1.0, max_iter=200):
    """
    Warm-start from logistic regression.

    Logistic regression is convex and has a unique global optimum, so it
    produces a w that already separates the classes geometrically regardless
    of random state. This places any subsequent ALM solve in a sensible basin
    of attraction before the first SSN step, addressing initialisation
    sensitivity without adding ALM parameters or continuation overhead.

    Appropriate when
    ----------------
    - The dataset is moderate in size (LR converges quickly).
    - You want a probabilistically calibrated direction.
    - No strong prior on the scale of w is needed.

    Parameters
    ----------
    X        : (d x n) feature matrix — samples as columns
    y        : (n,) binary label vector
    C        : inverse regularisation strength (default 1.0)
    max_iter : maximum solver iterations (default 200)

    Returns
    -------
    w0 : (d,) weight vector
    """
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X.T, y)
    w0 = clf.coef_.flatten()
    print(f"  warm start [LR]   ‖w0‖ = {np.linalg.norm(w0):.4f}")
    return w0


def get_warm_start_w_lda(X, y):
    """
    Warm-start from the Linear Discriminant Analysis (LDA) direction.

    LDA computes the direction that maximally separates the class means
    relative to within-class variance. For AUC optimisation this is
    particularly natural: the ranking objective cares about the direction
    along which positive and negative score distributions are most
    separated, which is precisely what LDA finds.

    Unlike logistic regression, LDA requires no iterative solver — it is
    a single linear solve involving the within-class scatter matrix S_W
    and the class mean difference. It is therefore faster and fully
    deterministic.

    The LDA direction is the solution to:

        S_W  w  =  (mu_pos - mu_neg)

    where S_W = Σ_{c} Σ_{i in c} (x_i - mu_c)(x_i - mu_c)^T

    If S_W is singular (d > n or collinear features), the pseudoinverse
    is used as a fallback, which gives the minimum-norm LDA direction.

    Appropriate when
    ----------------
    - Speed matters and an iterative LR solve is too slow.
    - The dataset is high-dimensional (d >> n) where LR may struggle.
    - You want a fully deterministic warm start with no hyperparameters.
    - The between-class geometry is a reliable guide to the optimum
      (which is true for AUC / ranking problems).

    Parameters
    ----------
    X : (d x n) feature matrix — samples as columns
    y : (n,) binary label vector

    Returns
    -------
    w0 : (d,) weight vector (the LDA direction, not normalised)
    """
    X_pos = X[:, y == 1]   # (d x n_pos)
    X_neg = X[:, y == 0]   # (d x n_neg)

    mu_pos = X_pos.mean(axis=1)   # (d,)
    mu_neg = X_neg.mean(axis=1)   # (d,)

    # Within-class scatter matrix S_W = S_pos + S_neg
    X_pos_c = X_pos - mu_pos[:, None]   # centred positives
    X_neg_c = X_neg - mu_neg[:, None]   # centred negatives
    S_W = X_pos_c @ X_pos_c.T + X_neg_c @ X_neg_c.T   # (d x d)

    delta_mu = mu_pos - mu_neg   # (d,)

    # Solve S_W w = delta_mu for the LDA direction.
    # Fall back to pseudoinverse if S_W is singular.
    try:
        w0 = np.linalg.solve(S_W, delta_mu)
    except np.linalg.LinAlgError:
        w0 = np.linalg.pinv(S_W) @ delta_mu

    print(f"  warm start [LDA]  ‖w0‖ = {np.linalg.norm(w0):.4f}")
    return w0


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════
# Maps string keys to strategy functions so runners can select by name
# rather than importing each function individually.
#
# Usage in a runner:
#
#     from warm_start import WARM_START_REGISTRY
#     w0 = WARM_START_REGISTRY["lda"](X, y)
#     w0 = WARM_START_REGISTRY["lr"](X, y, C=0.5)

WARM_START_REGISTRY = {
    "lr":  get_warm_start_w_lr,
    "lda": get_warm_start_w_lda,
}