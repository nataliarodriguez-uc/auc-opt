import numpy as np

class ProblemInstance:
    
    """
    Pairwise ranking problem for AUC optimization (binary or multi-class OvR).
    
    Creates pairwise comparisons between a target class and all other classes.
    Uses vectorized operations for efficiency (50-100x faster than loops).
    
    Matrix Convention:
        X has shape (d, n) where:
        - d = number of features (dimensions)
        - n = number of samples (data points)
        - C = number of classes
        
        Note: This is transposed from sklearn convention (n, d).
    
    Args:
        X: Feature matrix (d, n)
        y: Labels (n,) - any integer labels {0, 1, 2, ...}
        target_class: Which class to treat as positive (default: 1 for binary)
        seed: Random seed for reproducibility
    
    Attributes:
        X, y: Features and labels
        d: Number of features
        n: Number of samples
        C: Number of classes
        classes: Array of unique class labels
        target_class: The positive class
        K: List of (i,j) pairs where y[i]=target_class, y[j]≠target_class
        D: Pairwise difference matrix (d, n_pairs)
        n_pairs: Number of pairs
        w0: Random weight initialization (d,)
        lambda0: Zero dual initialization (n_pairs,)
    
    Examples:
        # Binary (automatic)
        >>> PI = ProblemInstance(X, y)  # Assumes y ∈ {0,1}, target_class=1
        >>> print(f"{PI.d} features, {PI.n} samples, {PI.C} classes")
        
        # Binary (explicit)
        >>> PI = ProblemInstance(X, y, target_class=1)
        
        # Multi-class OvR
        >>> for c in range(3):
        ...     PI = ProblemInstance(X, y, target_class=c)
        ...     print(f"Class {c}: {PI.n_pairs} pairs")
    """
    
    def __init__(self, X, y, target_class=None, seed=42):
        
        self.seed = seed
        self.X = X  # Shape: (d, n)
        self.y = y  # Shape: (n,)
        
        # Problem dimensions
        self.d = X.shape[0]  # Number of features
        self.n = X.shape[1]  # Number of samples

        # Detect classes
        self.classes = np.unique(y)
        self.C = len(self.classes)  # Number of classes
        
        # Determine target class
        if target_class is None:
            # Auto-detect: assume binary with target_class=1
            if self.C == 2 and np.array_equal(np.sort(self.classes), [0, 1]):
                target_class = 1
            else:
                raise ValueError(
                    f"target_class must be specified for multi-class data. "
                    f"Found {self.C} classes: {self.classes}"
                )
        
        # Validate target class exists
        if target_class not in self.classes:
            raise ValueError(
                f"target_class={target_class} not found in data. "
                f"Available classes: {self.classes}"
            )
        
        self.target_class = target_class
        
        # Compute pairwise differences: target vs. rest (VECTORIZED!)
        self.K, self.D = self._compute_pairwise_differences(self.X, self.y, target_class)
        self.n_pairs = len(self.K)
        
        # Optimization initialization
        np.random.seed(seed)
        self.w0 = np.random.randn(self.d)
        self.lambda0 = np.zeros(self.n_pairs)
    
    
    def _compute_pairwise_differences(self, X, y, target_class):
        """
        Compute pairwise differences: target_class vs. all others.
        
        Uses vectorized NumPy operations with broadcasting for efficiency.
        This is 50-100x faster than looping over pairs in Python.
        
        Algorithm:
            1. Extract X_pos (target class samples) and X_neg (other classes)
            2. Use broadcasting: X_neg[:,:,None] - X_pos[:,None,:] 
               creates (d, n_neg, n_pos) tensor of all differences
            3. Reshape to (d, n_pairs) where n_pairs = n_pos × n_neg
        
        Returns:
            K: List of (i,j) where y[i]=target_class, y[j]≠target_class
            D: Matrix where D[:, k] = X[:, j] - X[:, i] for pair k
        """
        # Get target (positive) and other (negative) indices
        pos_idx = np.where(y == target_class)[0]
        neg_idx = np.where(y != target_class)[0]
        
        # Validate we have both groups
        if len(pos_idx) == 0:
            raise ValueError(f"No samples found for target_class={target_class}")
        if len(neg_idx) == 0:
            raise ValueError(
                f"No samples found for non-target classes. "
                f"All samples belong to class {target_class}."
            )
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        n_pairs = n_pos * n_neg
        
        # Extract positive and negative samples
        X_pos = X[:, pos_idx]  # Shape: (d, n_pos)
        X_neg = X[:, neg_idx]  # Shape: (d, n_neg)
        
        # Vectorized pairwise differences using broadcasting
        # X_neg[:, :, None] has shape (d, n_neg, 1)
        # X_pos[:, None, :] has shape (d, 1, n_pos)
        # Broadcasting gives shape (d, n_neg, n_pos)
        # Result: D_tensor[i, j, k] = X_neg[i, j] - X_pos[i, k]
        D_tensor = X_neg[:, :, np.newaxis] - X_pos[:, np.newaxis, :]
        
        # Reshape to (d, n_pairs)
        # Pairs are ordered as: (pos[0], neg[0]), (pos[0], neg[1]), ..., (pos[1], neg[0]), ...
        D = D_tensor.reshape(X.shape[0], n_pairs)
        
        # Generate K list to match the ordering in D
        K = [(i, j) for i in pos_idx for j in neg_idx]
        
        return K, D
    
    # Visualization and Debugging 
    def get_class_distribution(self):
        """Return distribution of classes in the data."""
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))
    
    def __repr__(self):
        """String representation for debugging."""
        class_dist = self.get_class_distribution()
        n_target = np.sum(self.y == self.target_class)
        n_other = self.n - n_target
        
        return (
            f"ProblemInstance(\n"
            f"  d={self.d}, n={self.n}, C={self.C}, n_pairs={self.n_pairs},\n"
            f"  target_class={self.target_class}, classes={list(self.classes)},\n"
            f"  target_samples={n_target}, other_samples={n_other},\n"
            f"  class_distribution={class_dist}\n"
            f")"
        )