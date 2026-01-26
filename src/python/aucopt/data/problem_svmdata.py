import numpy as np

class DataSet_SVM:
    def __init__(self, m, n, num_classes, class_ratios, sep_distance,
                 feature_noise=0.0, flip_ratio=0.0, seed=1034):
        
        """
    Synthetic binary classification dataset generator.

    This class creates a linearly separable (or noisy) dataset in high-dimensional space,
    with adjustable separation, feature noise, and label flipping.

    Args:
        m (int): Total number of samples.
        n (int): Number of features (dimensionality).
        num_classes (int): Number of classes (usually 2).
        class_ratios (List[float]): Proportions for each class, must sum to 1.0.
        sep_distance (float): Distance between class means (controls separability).
        feature_noise (float, optional): Std. dev. of Gaussian noise added to features.
        flip_ratio (float, optional): Proportion of labels to randomly flip (adds label noise).
        seed (int, optional): Random seed for reproducibility.

    Attributes:
        X (ndarray): Feature matrix of shape (n, m).
        y (ndarray): Label vector of shape (m,).
        w_svm (ndarray): Ground truth separating hyperplane normal (unit vector).
    """
    
        assert abs(sum(class_ratios) - 1.0) < 1e-6, "Class ratios must sum to 1."
        np.random.seed(seed)

        # Determine number of samples per class
        class_sizes = np.round(np.array(class_ratios) * m).astype(int)
        if class_sizes.sum() != m:
            class_sizes[0] += m - class_sizes.sum()

        # Generate hyperplane normal
        hyperplane_normal = np.random.randn(n)
        hyperplane_normal /= np.linalg.norm(hyperplane_normal)

        X = np.empty((n, m))
        y = []

        start_idx = 0
        for class_id in range(num_classes):  # 0 or 1 for binary classification
            num_samples = class_sizes[class_id]
            points = np.random.randn(n, num_samples)
            shift = ((2 * class_id - 1) * sep_distance) * hyperplane_normal[:, np.newaxis]
            X[:, start_idx:start_idx + num_samples] = points + shift
            y.extend([class_id] * num_samples)
            start_idx += num_samples

        y = np.array(y)

        # Add feature noise
        if feature_noise > 0:
            X += feature_noise * np.random.randn(*X.shape)

        # Label flipping
        if flip_ratio > 0:
            num_to_flip = int(round(flip_ratio * len(y)))
            flip_indices = np.random.permutation(len(y))[:num_to_flip]
            y[flip_indices] = 1 - y[flip_indices]

        self.m = m
        self.n = n
        self.num_classes = num_classes
        self.class_ratios = class_ratios
        self.sep_distance = sep_distance
        self.X = X
        self.y = y
        self.w_svm = hyperplane_normal

