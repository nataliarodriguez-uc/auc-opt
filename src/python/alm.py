from problem_instance import ProblemInstance
from problem_variables import ALMParameters, SSNParameters, LineSearchParameters
from problem_variables import ALMVar, ProxVar, SSNVar, ALMLog
from ssn import run_ssn
from linesearch import compute_line_search
from prox import compute_prox_ssn, compute_prox_ls
from prox import prox_smallgamma_ssn, prox_gamma2_ssn, prox_largegamma_ssn
from problem_updates import update_tol, update_iter, update_proxmethod, update_sigma_gamma
import numpy as np
import time
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

    # Deep copies of parameter objects
    SP = deepcopy(SP0)
    AP = deepcopy(AP0)
    LS = deepcopy(LS0)

    # Initialize logging
    almlog = ALMLog(AP.max_iter_alm, SP.max_iter_ssn, LS.max_iter_ls)
    almlog.alm_time = time.time()

    # Shortcut variables from problem instance
    w0 = PI.w0
    lambda0 = PI.lambda0

    # Initialize ALM variables
    almvar = ALMVar(tau0, sigma0, PI)
    almvar.lambd = np.copy(lambda0)
    almvar.sigma = sigma0
    almvar.tau = tau0
    almvar.w = np.copy(w0)
    almvar.y = np.zeros(len(PI.K))
    almvar.alpha = alpha0

    # Initialize SSN and Prox variables
    ssnvar = SSNVar(PI)
    ssnvar.w_ssn = np.copy(w0)
    proxvar = ProxVar(PI.n, len(PI.K), almvar.tau)

    for t in range(AP.max_iter_alm):
        
        update_tol(SP, t)
        update_iter(SP, t)
        update_proxmethod(almvar)

        # Time the SSN call
        start_ssn = time.time()
        run_ssn(t, almlog, almvar, ssnvar, proxvar, PI, SP, LS)
        almlog.ssn_times[t] = time.time() - start_ssn

        # Update ALM variables from SSN solution
        almvar.w = np.copy(ssnvar.w_ssn)
        almvar.y = np.copy(ssnvar.y_ssn)
        almvar.w_D = np.copy(ssnvar.w_ssn_D)

        # Compute constraint residuals
        almvar.cons_condition = (1.0 / len(PI.K)) * (almvar.y - almvar.w_D)

        if np.linalg.norm(almvar.cons_condition, ord=np.inf) <= AP.tol_alm:
            almlog.alm_iter = t
            almlog.L_final = ssnvar.L_obj / len(PI.K)
            break
        else:
            update_sigma_gamma(almvar, AP)
            almvar.lambd += almvar.sigma * (1.0 / len(PI.K)) * (almvar.y - almvar.w_D)

    else:
        almlog.alm_iter = AP.max_iter_alm
        
    almlog.alm_time = time.time() - almlog.alm_time
    
    return almvar, almlog
