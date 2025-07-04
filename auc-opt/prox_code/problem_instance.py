import numpy as np
from sklearn.model_selection import train_test_split

class ProblemInstance:
    
    """
    Constructs a pairwise comparison problem instance from feature and label data.

    This class supports different modes for using all data, a train/test split, or a random batch.
    It computes the difference matrix D for all (i, j) such that y[i] > y[j], enabling AUC-type optimization.

    Args:
        X (ndarray): Feature matrix of shape (n_features, n_samples).
        y (ndarray): Label vector of shape (n_samples,).
        mode (str): 'full', 'train_test', or 'batch' to control how the data is used.
        train_ratio (float): Ratio of data used for training in 'train_test' mode.
        stratified (bool): Whether to preserve label ratios during train/test split.
        batch_size (int, optional): Number of samples in batch mode (split evenly between classes).

    Attributes:
        X (ndarray): Features used in this problem instance (mode-dependent).
        y (ndarray): Corresponding labels (mode-dependent).
        X_train, X_test, y_train, y_test: Only set in 'train_test' mode.
        K (list): List of index pairs (i, j) where y[i] > y[j].
        D (ndarray): Pairwise difference matrix of shape (n_features, len(K)).
        m (int): Number of samples in this instance.
        n (int): Number of features.
        w0 (ndarray): Randomly initialized primal variable (used in optimization).
        lambda0 (ndarray): Zero-initialized dual variable for each pair in K.
    """
    
    def __init__(self, X, y, mode='full', train_ratio=0.7, stratified=True, batch_size=None):
        self.mode = mode
        self.train_ratio = train_ratio
        self.stratified = stratified

        if mode == 'train_test':
            self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(X, y)
            self.X = self.X_train
            self.y = self.y_train
        elif mode == 'batch':
            self.X, self.y = self._sample_batch(X, y, batch_size)
        else:  # full
            self.X = X
            self.y = y

        self.K, self.D = self._compute_pairwise_differences(self.X, self.y)
        self.m = self.X.shape[1]
        self.n = self.X.shape[0]
        self.w0 = np.random.randn(self.n)
        self.lambda0 = np.zeros(len(self.K))

    def _split_data(self, X, y):
        X = X.T  # shape (samples, features)
        stratify = y if self.stratified else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_ratio, stratify=stratify, random_state=42
        )
        return X_train.T, X_test.T, y_train, y_test

    def _sample_batch(self, X, y, batch_size):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = n_neg = batch_size // 2
        pos_sample = np.random.choice(pos_idx, n_pos, replace=False)
        neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
        selected = np.concatenate([pos_sample, neg_sample])
        return X[:, selected], y[selected]

    def _compute_pairwise_differences(self, X, y):
        n_samples = X.shape[1]
        K = []
        for i in range(n_samples):
            for j in range(n_samples):
                if y[i] > y[j]:
                    K.append((i, j))
        num_pairs = len(K)
        D = np.zeros((X.shape[0], num_pairs))
        for idx, (i, j) in enumerate(K):
            D[:, idx] = X[:, j] - X[:, i]
        return K, D
