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
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

    elif delta <= w_Dij <= delta + gamma:
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * almvar.D_product[idx]

    elif delta + gamma < w_Dij < 1 + delta + 0.5 * gamma:
        proxvar.y[idx] = delta - gamma
        proxvar.Lag_obj += w_Dij - delta - 0.5 * gamma
        proxvar.Lag_J += sigma * D_ij
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

def prox_gamma2_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI):
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

    elif delta <= w_Dij <= delta + gamma:
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * almvar.D_product[idx]

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

def prox_largegamma_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI):
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau

    elif delta <= w_Dij <= delta + np.sqrt(2 * gamma):
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff**2
        proxvar.Lag_J += sigma * diff * D_ij
        proxvar.Lag_H += sigma * almvar.D_product[idx]

    else:
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        proxvar.Lag_H[np.diag_indices(n)] += almvar.tau



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
