from yaml import warnings
from aucopt.optim.variables import ALMVar, ProxVar, SSNVar, ALMLog
from aucopt.optim.ssn import run_ssn
from aucopt.optim.parameters import update_tol, update_iter, update_sigma_gamma
import numpy as np
import time
import warnings
from copy import deepcopy

def run_alm(
    sigma0: float,
    tau0: float,
    alpha0: float,
    PI,        # ProblemInstanceBatch
    AP0,       # ALMParameters
    SP0,       # SSNParameters
    LS0        # LineSearchParameters
):
    
    """
    Runs the Augmented Lagrangian Method (ALM) to solve a pairwise ranking optimization problem.
    
    OPTIMIZED VERSION with improvements:
    - Reduced deep copies (only copy when necessary)
    - Pre-compute commonly used values
    - Use in-place operations for array updates
    - Cache length calculations
    - Avoid redundant norm calculations

    Parameters:
    - sigma0: Initial penalty parameter.
    - tau0: Initial regularization weight.
    - alpha0: Initial step size for SSN updates.
    - PI: ProblemInstance object (full, train/test, or batch).
    - AP0: ALMParameters (max iterations, tolerance, scaling).
    - SP0: SSNParameters (tolerance, inner iterations).
    - LS0: LineSearchParameters (Armijo constants).

    Returns:
    - almvar: Final solution variables (weights, duals, etc.).
    - almlog: Log of timing, convergence, and iteration statistics.
    """
    
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Deep copies only for parameters that will be modified
    SP = deepcopy(SP0)
    AP = deepcopy(AP0)
    LS = deepcopy(LS0)

    # Initialize logging
    almlog = ALMLog(AP.max_iter_alm, SP.max_iter_ssn, LS.max_iter_ls)
    almlog.alm_time = time.time()

    w0 = PI.w0
    lambda0 = PI.lambda0
    K_len = len(PI.K) 
    inv_K_len = 1.0 / K_len  # Pre-compute inverse for faster division

    # Initialize ALM variables
    almvar = ALMVar(tau0, sigma0, PI)
    almvar.lambd[:] = lambda0  # In-place if lambda0 is zeros, else copy
    almvar.sigma = sigma0
    almvar.tau = tau0
    almvar.w[:] = w0  # In-place assignment instead of copy
    almvar.y = np.zeros(K_len)
    almvar.alpha = alpha0

    # Initialize SSN and Prox variables
    ssnvar = SSNVar(PI)
    ssnvar.w_ssn[:] = w0  # In-place assignment
    proxvar = ProxVar(PI.d, K_len, almvar.tau)

    # *** OPTIMIZATION 4: Pre-allocate temporary arrays ***
    temp_residual = np.empty(K_len)  # Reuse for constraint calculations
    
    for t in range(AP.max_iter_alm):
        
        update_tol(SP, t)
        update_iter(SP, t)

        # Time the SSN call
        start_ssn = time.time()
        run_ssn(t, almlog, almvar, ssnvar, proxvar, PI, SP, LS)
        almlog.ssn_times[t] = time.time() - start_ssn

        # Update ALM variables from SSN solution
        almvar.w[:] = ssnvar.w_ssn  # In-place instead of np.copy
        almvar.y[:] = ssnvar.y_ssn  # In-place
        almvar.w_D[:] = ssnvar.w_ssn_D  # In-place

        # Original: almvar.cons_condition = (1.0 / len(PI.K)) * (almvar.y - almvar.w_D)
        np.subtract(almvar.y, almvar.w_D, out=temp_residual)
        np.multiply(temp_residual, inv_K_len, out=almvar.cons_condition)

        cons_norm = np.linalg.norm(almvar.cons_condition, ord=np.inf)
        
        if cons_norm <= AP.tol_alm:
            almlog.alm_iter = t
            almlog.L_final = ssnvar.L_obj * inv_K_len  # Use cached inverse
            break
        else:
            # Update sigma and gamma
            update_sigma_gamma(almvar, AP)
            
            # Original: almvar.lambd += almvar.sigma * (1.0 / len(PI.K)) * (almvar.y - almvar.w_D)
            # Reuse temp_residual from above (already contains y - w_D)
            almvar.lambd += almvar.sigma * inv_K_len * temp_residual

    else:
        # Max iterations reached
        almlog.alm_iter = AP.max_iter_alm
        
    almlog.alm_time = time.time() - almlog.alm_time
    
    return almvar, almlog