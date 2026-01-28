import numpy as np
from aucopt.optim.prox import compute_prox_ls
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
    
    alpha_ls = ssnvar.alpha_ssn
    L_current = proxvar.Lag_obj
    
    # Pre-compute dot product (only optimization over original)
    grad_dot_d = np.dot(proxvar.Lag_J, d)
    armijo_threshold = LS.c * grad_dot_d
    
    for l in range(LS.max_iter_ls):
        ssnvar.w_ls = ssnvar.w_ssn + alpha_ls * d
        proxvar.w_ls_D = ssnvar.w_ssn_D + alpha_ls * proxvar.d_D
        
        L_new = compute_prox_ls(proxvar.w_ls_D, almvar, proxvar, PI)
        
        # Armijo condition
        if L_new - L_current <= armijo_threshold * alpha_ls:
            almlog.lsearch_iters[t, k] = l + 1
            ssnvar.alpha_ssn = alpha_ls
            return
        
        alpha_ls *= LS.beta
    
    almlog.lsearch_iters[t, k] = LS.max_iter_ls
    ssnvar.alpha_ssn = alpha_ls