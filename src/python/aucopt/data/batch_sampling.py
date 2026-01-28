import numpy as np

class BatchSampler:
    
    """Utilities for creating batches for SGD training."""
    
    @staticmethod
    def create_disjoint_batches(y, n_epochs, n_batches, n_pos, n_neg, seed=None):
        
        """
        Pre-allocate disjoint batches (no sample overlap within epoch).
        
        Args:
            y: Labels
            n_epochs: Number of epochs
            n_batches: Batches per epoch
            n_pos: Positive samples per batch
            n_neg: Negative samples per batch
            seed: Random seed
        
        Returns:
            List of arrays, each containing sample indices for one batch
        """
        if seed is not None:
            np.random.seed(seed)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        
        total_batches = n_epochs * n_batches
        samples_needed_pos = total_batches * n_pos
        samples_needed_neg = total_batches * n_neg
        
        assert len(pos_idx) >= samples_needed_pos, \
            f"Need {samples_needed_pos} positive samples, have {len(pos_idx)}"
        assert len(neg_idx) >= samples_needed_neg, \
            f"Need {samples_needed_neg} negative samples, have {len(neg_idx)}"
        
        pos_batches = np.array_split(pos_idx[:samples_needed_pos], total_batches)
        neg_batches = np.array_split(neg_idx[:samples_needed_neg], total_batches)
        
        batch_indices = [np.concatenate([pos, neg]) 
                        for pos, neg in zip(pos_batches, neg_batches)]
        
        return batch_indices
    
    @staticmethod
    def create_random_batches(y, n_batches, n_pos, n_neg, seed=None):
        """
        Create random batches (with replacement across epochs).
        
        Samples can appear in multiple batches.
        """
        if seed is not None:
            np.random.seed(seed)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        batch_indices = []
        for _ in range(n_batches):
            pos_sample = np.random.choice(pos_idx, n_pos, replace=False)
            neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
            batch_indices.append(np.concatenate([pos_sample, neg_sample]))
        
        return batch_indices
    
    @staticmethod
    def create_stratified_batches(y, n_batches, batch_size, seed=None):
        """
        Create batches maintaining class proportions.
        
        Useful when class ratios are important.
        """
        if seed is not None:
            np.random.seed(seed)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        pos_ratio = len(pos_idx) / len(y)
        n_pos = int(batch_size * pos_ratio)
        n_neg = batch_size - n_pos
        
        return BatchSampler.create_disjoint_batches(
            y, 1, n_batches, n_pos, n_neg, seed
        )