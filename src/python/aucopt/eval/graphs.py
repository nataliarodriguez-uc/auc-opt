import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import os

def plot_dataset_with_separator(dataset):
    
    """
    Plots a 2D synthetic dataset with two classes and the ground truth separator w_svm.

    Args:
    - dataset: A DataSet object with 2D features (n=2), class labels, and a separating hyperplane.

    Displays:
    - A scatter plot with class-colored points and a dashed line representing w_svm.
    """
    
    X = dataset.X
    y = dataset.y
    w = dataset.w_svm

    if X.shape[0] != 2:
        raise ValueError("This plot only works for 2D features (n=2)")

    plt.figure(figsize=(6, 6))

    # Plot data points
    plt.scatter(X[0, y == 0], X[1, y == 0], color='red', label='Class 0')
    plt.scatter(X[0, y == 1], X[1, y == 1], color='blue', label='Class 1')

    # Plot separator
    x_vals = np.linspace(X[0].min(), X[0].max(), 100)
    slope = -w[0] / w[1]
    intercept = 0
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='w_svm separator')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Synthetic Dataset with Separator')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def prox_approx_graph(delta=0.1, label="δ = 0.1"):
    
    """
    Plots the linear approximation of the 0-1 loss (indicator) used in the proximal operator.

    Args:
    - delta: The threshold shift for the approximation.
    - label: Optional label for display purposes.

    Returns:
    - fig, ax: Matplotlib figure and axis.
    - f: The approximation function ℓ(x) = min(1, max(0, x - delta)).
    """
    
    def f(x):
        return np.minimum(1, np.maximum(0, x - delta))
    
    x_vals = np.linspace(-5, 5, 1000)
    y_vals = f(x_vals)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_vals, y_vals, label=f"Linear Indicator Approximation ({label})", color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("ℓ(x)")
    ax.set_title("Linear Indicator Approximation")
    ax.grid(True)
    
    return fig, ax, f

def scatter_prox_points(ax, w, lambd, sigma, PI, f, label_prefix="initial"):
    
    """
    Plots (x, ℓ(x)) values for all (i,j) pairs in the problem, where
    x = wᵗ(D_ij) - λ_ij / σ, and ℓ(x) is the linear surrogate loss.

    Args:
    - ax: Matplotlib axis to plot on (reused from prox_approx_graph).
    - w: Current primal variable (weight vector).
    - lambda_vec: Dual variable vector.
    - sigma: Current penalty parameter.
    - PI: ProblemInstance with D and K.
    - f: Surrogate loss function (from prox_approx_graph).
    - label_prefix: "initial" or "final" to color-code the points.
    """
    
    color = "blue" if label_prefix == "initial" else "red"
    alpha = 0.3

    D = PI.D  # shape (d, N)
    K = PI.K  # list of (i, j) index pairs

    x_vals = []
    y_vals = []

    for idx, (i, j) in enumerate(K):
        Dij = D[:, idx]
        x = w @ Dij - lambd[idx] / sigma
        y = f(x)
        x_vals.append(x)
        y_vals.append(y)

    ax.scatter(x_vals, y_vals, color=color, alpha=alpha, s=10, label=label_prefix)

def visualize_prox_approximation(w_before, w_after, lambda_before, lambda_after, sigma, PI, delta=0.1, save_path=None):
    fig, ax, f = prox_approx_graph(delta)

    scatter_prox_points(ax, w_before, lambda_before, sigma, PI, f, label_prefix="initial")
    scatter_prox_points(ax, w_after, lambda_after, sigma, PI, f, label_prefix="final")

    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()


def plot_pca_projection(X, y, title="PCA Projection"):
    
    """
    Helps project points for multiple n variables onto a 2D plane using PCA components.
    
    """
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X.T)  # X is (n_features, n_samples)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_proj[y == 0, 0], X_proj[y == 0, 1], alpha=0.6, label="Class 0")
    plt.scatter(X_proj[y == 1, 0], X_proj[y == 1, 1], alpha=0.6, label="Class 1")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_auc_on_test(w, X_test, y_test, target_class=1,
                        dataset_name="dataset", output_dir="results"):
    """
    Evaluate AUC on test data.
    
    Args:
        w: Trained weight vector (d,)
        X_test: Test features (d, n_test)
        y_test: Test labels (n_test,)
        target_class: Which class to evaluate (for multi-class)
        dataset_name: Name for saving plots
        output_dir: Where to save results
    
    Returns:
        auc: AUC score on test set
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import os
    
    # Create binary labels for target class
    y_binary = (y_test == target_class).astype(int)
    
    # Compute scores
    scores = w @ X_test
    
    # Compute AUC
    auc = roc_auc_score(y_binary, scores)
    fpr, tpr, _ = roc_curve(y_binary, scores)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{dataset_name}_roc.png")
    plt.savefig(fig_path)
    plt.close()
    
    auc_csv_path = os.path.join(output_dir, f"{dataset_name}_auc.csv")
    with open(auc_csv_path, "w") as f:
        f.write("auc\n")
        f.write(f"{auc:.6f}\n")
    
    print(f"✅ Test AUC for {dataset_name}: {auc:.4f}")
    print(f"✅ Saved ROC plot to {fig_path}")
    print(f"✅ Saved AUC value to {auc_csv_path}")
    
    return auc

def linear_indicator_approx(delta=1.0):
    def f(x): return np.minimum(1, np.maximum(0, x - delta))
    return f

def plot_linear_indicator(delta=1.0, label="initial"):
    f = linear_indicator_approx(delta)
    x_vals = np.linspace(-5, 5, 1000)
    y_vals = f(x_vals)
    plt.figure()
    plt.plot(x_vals, y_vals, label=f"ℓ(x), δ={delta}")
    plt.title(f"Linear Indicator Approximation ({label})")
    plt.xlabel("x")
    plt.ylabel("ℓ(x)")
    plt.grid(True)
    plt.legend()
    return f

def scatter_prox_points(w, lambda_vec, sigma, PI, f, label="initial"):
    x_vals = []
    y_vals = []
    for idx in range(PI.D.shape[1]):
        Dij = PI.D[:, idx]
        x_ij = np.dot(w, Dij) - lambda_vec[idx] / sigma
        x_vals.append(x_ij)
        y_vals.append(f(x_ij))
    color = "blue" if label == "initial" else "red"
    plt.scatter(x_vals, y_vals, s=10, alpha=0.3, color=color, label=f"{label} points")

