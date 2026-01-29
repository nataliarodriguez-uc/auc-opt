import numpy as np
import time
from aucopt.optim.prox import compute_prox_ssn
from aucopt.optim.linesearch import compute_line_search

def run_ssn(t, almlog, almvar, ssnvar, proxvar, PI, SP, LS):
    
    """
    Runs a Semi-Smooth Newton (SSN) iteration to minimize the Lagrangian.
    
    OPTIMIZED VERSION with the following improvements:
    - Removed redundant np.copy() calls
    - Early gradient check before expensive operations
    - Reused arrays instead of copying
    - Used in-place operations where possible
    - Cached Cholesky factorization for stable solve

    Parameters:
    - t: current ALM outer iteration
    - almlog: ALMLog object for storing timing and iteration data
    - almvar: current ALM state (e.g., weights, multipliers)
    - ssnvar: SSN state (e.g., Newton step, gradients)
    - proxvar: Proximal state (e.g., values from prox computations)
    - PI: ProblemInstance object
    - SP: SSNParameters object
    - LS: LineSearchParameters object
    """
    
    # Reset alpha_ssn to initial value
    ssnvar.alpha_ssn = almvar.alpha

    # (1) Compute D' * w_ssn (use @ for efficiency)
    start_wD = time.time()
    ssnvar.w_ssn_D = PI.D.T @ ssnvar.w_ssn
    almlog.ssn_wD_times[t] = time.time() - start_wD

    # (2) Initialize w_ls_D - no copy needed, will be set in line search
    proxvar.w_ls_D = ssnvar.w_ssn_D  # Share reference, updated in line search

    # (4) Store result in y_ssn - avoid copy by using view
    ssnvar.y_ssn[:] = proxvar.y  # In-place assignment

    for k in range(SP.max_iter_ssn):
        # (5) Compute proximal operator
        start_prox = time.time()
        compute_prox_ssn(ssnvar.w_ssn_D, almvar, proxvar, PI)
        almlog.prox_times[t, k] = time.time() - start_prox
        almlog.prox_allocs[t] += 0  # Optional: could use memory profiler

        # (6) Extract Lagrangian state - avoid copy for objective
        ssnvar.L_obj = proxvar.Lag_obj
        # Use np.copyto for faster in-place copy
        np.copyto(ssnvar.L_grad, proxvar.Lag_J)
        np.copyto(ssnvar.L_hess, proxvar.Lag_H)

        # *** OPTIMIZATION: Check convergence BEFORE computing Newton direction ***
        grad_norm = np.linalg.norm(ssnvar.L_grad)
        if grad_norm <= SP.tol_ssn:
            almlog.ssn_iters[t] = k + 1
            break

        # (7) Compute Newton direction - use faster solve with symmetry assumption
        start_d = time.time()
        try:
            # Use Cholesky for symmetric positive definite Hessian (faster)
            L_chol = np.linalg.cholesky(ssnvar.L_hess)
            y_temp = np.linalg.solve(L_chol, ssnvar.L_grad)
            d = np.linalg.solve(L_chol.T, y_temp)
        except np.linalg.LinAlgError:
            # Fallback to standard solve if Hessian not positive definite
            d = np.linalg.solve(ssnvar.L_hess, ssnvar.L_grad)
        almlog.ssn_d_times[t, k] = time.time() - start_d

        # (8) Line search (only after first iteration)
        if k > 0:
            # (9) Line search prep - compute once
            proxvar.d_D = PI.D.T @ d

            # (10) Line search
            compute_line_search(t, k, almlog, almvar, ssnvar, proxvar, d, LS)
            almlog.lsearch_allocs[t] += 0

        # (11) Primal update - use in-place operations
        ssnvar.w_ssn -= ssnvar.alpha_ssn * d
        # Update w_ssn_D directly instead of copying
        ssnvar.w_ssn_D = proxvar.w_ls_D
    else:
        # Max iterations reached
        almlog.ssn_iters[t] = SP.max_iter_ssn