import numpy as np
from src.python.aucopt.optim.prox import compute_prox_ls
import time

def compute_line_search(t, k, almlog, almvar, ssnvar, proxvar, d, PI, LS):
    
    """
    Performs backtracking line search to find a suitable step size along the Newton direction.

    Uses the Armijo condition to ensure sufficient decrease in the augmented Lagrangian.

    Args:
    - t: Current ALM iteration index.
    - k: Current SSN iteration index.
    - almlog: ALMLog object tracking iteration stats.
    - almvar: ALMVar object with current primal/dual variables.
    - ssnvar: SSNVar object containing Newton direction state.
    - proxvar: ProxVar object storing proximal-related variables.
    - d: Newton direction vector.
    - PI: ProblemInstance with data and pairwise differences.
    - LS: LineSearchParameters object with backtracking rules.

    Modifies:
    - ssnvar.alpha_ssn: Sets final step size after search.
    - almlog.lsearch_iters[t][k]: Records number of backtracking steps.
    """
    
    # Initialize alpha
    alpha_ls = ssnvar.alpha_ssn

    # Objective and gradient at current point
    L_current = proxvar.Lag_obj
    J_current = proxvar.Lag_J

    for l in range(LS.max_iter_ls):
        
        start_time = time.time()
        # (1) Candidate primal and D-product update
        ssnvar.w_ls = ssnvar.w_ssn + alpha_ls * d
        proxvar.w_ls_D = ssnvar.w_ssn_D + alpha_ls * proxvar.d_D  # linear update

        # (2) Compute new objective value
        L_new = compute_prox_ls(proxvar.w_ls_D, almvar, proxvar, PI)
        almlog.lsearch_times[t, k, l] = time.time() - start_time

        # (3) Armijo condition
        if L_new - L_current <= LS.c * alpha_ls * np.dot(J_current, d):
            almlog.lsearch_iters[t, k] = l + 1
            break
        else:
            alpha_ls *= LS.beta
            
    else:
        # If loop ends without break, max iterations reached
        almlog.lsearch_iters[t, k] = LS.max_iter_ls
        
    # Update step size
    ssnvar.alpha_ssn = alpha_ls
