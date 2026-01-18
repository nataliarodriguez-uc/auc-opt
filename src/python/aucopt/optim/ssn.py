import numpy as np
import time
from aucopt.optim.prox import compute_prox_ssn
from aucopt.optim.linesearch import compute_line_search

def run_ssn(t, almlog, almvar, ssnvar, proxvar, PI, SP, LS):
    
    """
    Runs a Semi-Smooth Newton (SSN) iteration to minimize the Lagrangian.

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

    # (1) Compute D' * w_ssn
    start_wD = time.time()
    ssnvar.w_ssn_D = PI.D.T @ ssnvar.w_ssn
    almlog.ssn_wD_times[t] = time.time() - start_wD

    # (2) Store w_D for proximal call
    proxvar.w_ls_D = np.copy(ssnvar.w_ssn_D)

    # (4) Store result in y_ssn
    ssnvar.y_ssn = np.copy(proxvar.y)

    for k in range(SP.max_iter_ssn):
        # (5) Compute proximal operator
        start_prox = time.time()
        compute_prox_ssn(ssnvar.w_ssn_D, almvar, proxvar, PI)
        almlog.prox_times[t, k] = time.time() - start_prox
        almlog.prox_allocs[t] += 0  # Optional: could use memory profiler

        # (6) Extract Lagrangian state
        ssnvar.L_obj = proxvar.Lag_obj
        ssnvar.L_grad = np.copy(proxvar.Lag_J)
        ssnvar.L_hess = np.copy(proxvar.Lag_H)

        # (7) Compute Newton direction
        start_d = time.time()
        d = np.linalg.solve(ssnvar.L_hess, ssnvar.L_grad)
        almlog.ssn_d_times[t, k] = time.time() - start_d

        # (8) Check convergence
        if np.linalg.norm(ssnvar.L_grad) <= SP.tol_ssn:
            almlog.ssn_iters[t] = k+1
            break
        else:
            if k > 0:
                # (9) Line search prep
                proxvar.d_D = PI.D.T @ d

                # (10) Line search
                compute_line_search(t, k, almlog, almvar, ssnvar, proxvar, d, PI, LS)
                almlog.lsearch_allocs[t] += 0

            # (11) Primal update
            ssnvar.w_ssn -= ssnvar.alpha_ssn * d
            ssnvar.w_ssn_D = np.copy(proxvar.w_ls_D)
    else:
        almlog.ssn_iters[t] = SP.max_iter_ssn
