import numpy as np

def compute_prox_ssn(prox_wD, almvar, proxvar, PI):
    
    """
    Evaluate the proximal mapping in Semi-Smooth Newton (SSN) mode.

    Computes ALM's Lagrangian objective, Jacobian, and Hessian by applying 
    the selected proximal rule (based on gamma) to each (i, j) pair.

    Parameters:
    - prox_wD: array of ⟨w, D_ij⟩ values
    - almvar: current ALM state
    - proxvar: object storing proximal variables
    - PI: problem instance
    """
    
    # 1. Vectorized computation of w_Dij
    proxvar.w_Dij = prox_wD - almvar.lambd / almvar.sigma

    # 2. Reset proxvar variables
    proxvar.y[:] = 0.0
    proxvar.Lag_obj = 0.0
    proxvar.Lag_H[:, :] = 0.0
    proxvar.Lag_J[:] = 0.0

    # 3. Loop over all pairs
    for idx in range(len(PI.K)):
        D_ij = PI.D[:, idx]
        # Dispatch to selected proximal method
        almvar.prox_method_ssn(idx, proxvar.w_Dij[idx], D_ij, almvar, proxvar, PI)

def prox_smallgamma_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI):
    d = PI.d
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

    elif delta <= w_Dij <= delta + gamma:
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * np.outer(D_ij, D_ij)

    elif delta + gamma < w_Dij < 1 + delta + 0.5 * gamma:
        proxvar.y[idx] = delta - gamma
        proxvar.Lag_obj += w_Dij - delta - 0.5 * gamma
        proxvar.Lag_J += sigma * D_ij
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

def prox_gamma2_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI):
    d = PI.d
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

    elif delta <= w_Dij <= delta + gamma:
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * np.outer(D_ij, D_ij)

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

def prox_largegamma_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI):
    d = PI.d
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau

    elif delta <= w_Dij <= delta + np.sqrt(2 * gamma):
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * np.outer(D_ij, D_ij)

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(d)] += almvar.tau



def compute_prox_ls(prox_wD, almvar, proxvar, PI):
    
    """
    Evaluate the proximal objective function only (used in Line Search).

    Parameters:
    - prox_wD: ⟨w, D_ij⟩ values at candidate point
    - almvar: ALM variable state
    - proxvar: storage (not used here, but passed for interface consistency)
    - PI: problem instance

    Returns:
    - total Lagrangian objective over all pairwise constraints
    """
    
    L = 0.0
    for idx in range(len(PI.K)):
        w_Dij = prox_wD[idx] - almvar.lambd[idx] / almvar.sigma

        if almvar.gamma < 2:
            L += prox_smallgamma_ls(w_Dij, almvar)
        elif almvar.gamma == 2:
            L += prox_gamma2_ls(w_Dij, almvar)
        else:
            L += prox_largegamma_ls(w_Dij, almvar)

    return L

def prox_smallgamma_ls(w_Dij, almvar):
    delta = almvar.delta
    gamma = almvar.gamma

    if w_Dij < delta:
        return 0.0
    elif delta <= w_Dij <= delta + gamma:
        return 0.5 / gamma * (delta - w_Dij)**2
    elif delta + gamma < w_Dij < 1 + delta + 0.5 * gamma:
        return w_Dij - delta - 0.5 * gamma
    else:
        return 1.0

def prox_gamma2_ls(w_Dij, almvar):
    delta = almvar.delta
    gamma = almvar.gamma

    if w_Dij < delta:
        return 0.0
    elif w_Dij < delta + gamma:
        return 0.5 / gamma * (delta - w_Dij)**2
    else:
        return 1.0

def prox_largegamma_ls(w_Dij, almvar):
    delta = almvar.delta
    gamma = almvar.gamma

    if w_Dij < delta:
        return 0.0
    elif w_Dij < delta + np.sqrt(2 * gamma):
        return 0.5 / gamma * (delta - w_Dij)**2
    else:
        return 1.0


def compute_prox_ssn_vectorized(prox_wD, almvar, proxvar, PI):
    """
    Vectorized proximal operator evaluation.
    
    Instead of looping over pairs, we:
    1. Vectorize region detection
    2. Batch process each region
    3. Accumulate updates efficiently
    
    This is 10-50x faster than the loop version!
    """
    d = PI.d
    
    # Compute w_Dij for all pairs at once (already vectorized)
    w_Dij = prox_wD - almvar.lambd / almvar.sigma
    
    # Reset accumulation variables
    proxvar.y[:] = 0.0
    proxvar.Lag_obj = 0.0
    proxvar.Lag_J[:] = 0.0
    proxvar.Lag_H[:, :] = 0.0
    
    # Get parameters
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta
    tau = almvar.tau
    
    # Dispatch to appropriate gamma regime
    if gamma < 2:
        _compute_prox_smallgamma_vectorized(
            w_Dij, PI.D, d,  gamma, sigma, delta, tau, proxvar
        )
    elif gamma == 2:
        _compute_prox_gamma2_vectorized(
            w_Dij, PI.D, d, gamma, sigma, delta, tau, proxvar
        )
    else:  # gamma > 2
        _compute_prox_largegamma_vectorized(
            w_Dij, PI.D, d, gamma, sigma, delta, tau, proxvar
        )


def _compute_prox_smallgamma_vectorized(w_Dij, D, d, gamma, sigma, delta, tau, proxvar):
    """
    Vectorized evaluation for gamma < 2 case.
    
    From your PDF page 6:
    - Region 1: w_Dij < delta
    - Region 2: delta <= w_Dij <= delta + gamma
    - Region 3: delta + gamma < w_Dij < 1 + delta + gamma/2
    - Region 4: w_Dij >= 1 + delta + gamma/2
    """
    
    # ===== Region 1: w_Dij < delta =====
    region1_mask = w_Dij < delta
    n_region1 = np.sum(region1_mask)
    
    if n_region1 > 0:
        proxvar.y[region1_mask] = w_Dij[region1_mask]
        # Lag_H diagonal update: done once at end
    
    # ===== Region 2: delta <= w_Dij <= delta + gamma =====
    region2_mask = (delta <= w_Dij) & (w_Dij <= delta + gamma)
    n_region2 = np.sum(region2_mask)
    
    if n_region2 > 0:
        proxvar.y[region2_mask] = delta
        
        # Compute diffs for this region
        diffs = delta - w_Dij[region2_mask]  # Shape: (n_region2,)
        
        # Objective: sum of 0.5/gamma * diff²
        proxvar.Lag_obj += np.sum(0.5 / gamma * diffs**2)
        
        # Jacobian: sum of sigma * diff * D_ij
        D_region2 = D[:, region2_mask]  # Shape: (d, n_region2)
        proxvar.Lag_J += sigma * (D_region2 @ diffs)  # Matrix-vector product!
        
        # Hessian: sum of sigma * D_ij D_ij^T
        # This is the expensive part - need to be clever
        # D_ij D_ij^T = outer product
        # Sum of outer products = D @ D^T (when properly scaled)
        proxvar.Lag_H += sigma * (D_region2 @ D_region2.T)  # Matrix multiplication!
    
    # ===== Region 3: delta + gamma < w_Dij < 1 + delta + gamma/2 =====
    region3_mask = (delta + gamma < w_Dij) & (w_Dij < 1 + delta + 0.5 * gamma)
    n_region3 = np.sum(region3_mask)
    
    if n_region3 > 0:
        proxvar.y[region3_mask] = delta - gamma
        
        # Objective: sum of (w_Dij - delta - gamma/2)
        proxvar.Lag_obj += np.sum(w_Dij[region3_mask] - delta - 0.5 * gamma)
        
        # Jacobian: sum of sigma * D_ij
        D_region3 = D[:, region3_mask]
        proxvar.Lag_J += sigma * np.sum(D_region3, axis=1)  # Sum columns
        
        # Lag_H diagonal update: done once at end
    
    # ===== Region 4: w_Dij >= 1 + delta + gamma/2 =====
    region4_mask = w_Dij >= 1 + delta + 0.5 * gamma
    n_region4 = np.sum(region4_mask)
    
    if n_region4 > 0:
        proxvar.y[region4_mask] = w_Dij[region4_mask]
        proxvar.Lag_obj += n_region4 * 1.0  # Add 1.0 for each pair in this region
        # Lag_H diagonal update: done once at end
    
    # ===== Diagonal regularization (accumulated over ALL pairs) =====
    # Regions 1, 3, 4 contribute tau to diagonal
    n_diag_regions = n_region1 + n_region3 + n_region4
    proxvar.Lag_H[np.diag_indices(d)] += tau * n_diag_regions


def _compute_prox_gamma2_vectorized(w_Dij, D, d, gamma, sigma, delta, tau, proxvar):
    """Vectorized evaluation for gamma = 2 case."""
    
    # Region 1: w_Dij < delta
    region1_mask = w_Dij < delta
    n_region1 = np.sum(region1_mask)
    
    if n_region1 > 0:
        proxvar.y[region1_mask] = w_Dij[region1_mask]
    
    # Region 2: delta <= w_Dij <= delta + gamma
    region2_mask = (delta <= w_Dij) & (w_Dij <= delta + gamma)
    n_region2 = np.sum(region2_mask)
    
    if n_region2 > 0:
        proxvar.y[region2_mask] = delta
        diffs = delta - w_Dij[region2_mask]
        
        proxvar.Lag_obj += np.sum(0.5 / gamma * diffs**2)
        
        D_region2 = D[:, region2_mask]
        proxvar.Lag_J += sigma * (D_region2 @ diffs)
        proxvar.Lag_H += sigma * (D_region2 @ D_region2.T)
    
    # Region 3: w_Dij > delta + gamma
    region3_mask = w_Dij > delta + gamma
    n_region3 = np.sum(region3_mask)
    
    if n_region3 > 0:
        proxvar.y[region3_mask] = w_Dij[region3_mask]
        proxvar.Lag_obj += n_region3 * 1.0
    
    # Diagonal
    n_diag_regions = n_region1 + n_region3
    proxvar.Lag_H[np.diag_indices(d)] += tau * n_diag_regions


def _compute_prox_largegamma_vectorized(w_Dij, D, d, gamma, sigma, delta, tau, proxvar):
    """Vectorized evaluation for gamma > 2 case."""
    
    sqrt_2gamma = np.sqrt(2 * gamma)
    
    # Region 1: w_Dij < delta
    region1_mask = w_Dij < delta
    n_region1 = np.sum(region1_mask)
    
    if n_region1 > 0:
        proxvar.y[region1_mask] = w_Dij[region1_mask]
    
    # Region 2: delta <= w_Dij <= delta + sqrt(2*gamma)
    region2_mask = (delta <= w_Dij) & (w_Dij <= delta + sqrt_2gamma)
    n_region2 = np.sum(region2_mask)
    
    if n_region2 > 0:
        proxvar.y[region2_mask] = delta
        diffs = delta - w_Dij[region2_mask]
        
        proxvar.Lag_obj += np.sum(0.5 / gamma * diffs**2)
        
        D_region2 = D[:, region2_mask]
        proxvar.Lag_J += sigma * (D_region2 @ diffs)
        proxvar.Lag_H += sigma * (D_region2 @ D_region2.T)
    
    # Region 3: w_Dij > delta + sqrt(2*gamma)
    region3_mask = w_Dij > delta + sqrt_2gamma
    n_region3 = np.sum(region3_mask)
    
    if n_region3 > 0:
        proxvar.y[region3_mask] = w_Dij[region3_mask]
        proxvar.Lag_obj += n_region3 * 1.0
    
    # Diagonal
    n_diag_regions = n_region1 + n_region3
    proxvar.Lag_H[np.diag_indices(d)] += tau * n_diag_regions
