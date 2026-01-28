import numpy as np

class ALMVar:
    
    """
    Stores variables and state used throughout the Augmented Lagrangian Method (ALM).

    Attributes:
    - tau, sigma, gamma: algorithm parameters
    - w: current primal variable
    - lambd: Lagrange multipliers for inequality constraints
    - y: proximal outputs
    - alpha: current step size
    - D_product: precomputed outer products D_ij D_ij^T for efficiency
    - w_D: inner products ⟨w, D_ij⟩
    - prox_method_ssn / prox_method_ls: chosen prox operators for SSN and LS
    - cons_condition: constraint residuals for termination
    """
    
    def __init__(self, tau_init, sigma_init, PI):
        self.cons_condition_norm = []
        self.cons_condition = np.zeros(len(PI.K))
        self.lambd = np.zeros(len(PI.K))
        self.tau = tau_init
        self.sigma = sigma_init
        self.gamma = 1.0 / sigma_init
        self.w = np.random.randn(PI.d)  
        self.y = np.zeros(len(PI.K))
        self.alpha = 1.0
        self.delta = 0.0
        #self.D_product = [np.outer(PI.D[:, k], PI.D[:, k]) for k in range(len(PI.K))]
        self.w_D = np.zeros(len(PI.K))
        self.prox_method_ssn = None  # to be assigned
        self.prox_method_ls = None


class ALMParameters:
    
    """
    Hyperparameters for controlling the ALM procedure.

    Attributes:
    - max_iter_alm: maximum ALM iterations
    - tau_scale: decay factor for tau
    - sigma_scale: growth factor for sigma
    - tol_alm: termination tolerance based on constraint residuals
    """
    
    def __init__(self, max_iter_alm, tau_scale, sigma_scale, tol_alm):
        self.max_iter_alm = max_iter_alm
        self.tau_scale = tau_scale
        self.sigma_scale = sigma_scale
        self.tol_alm = tol_alm


class ProxVar:
    
    """
    Stores intermediate values used during proximal mapping calls.

    Attributes:
    - Lprox, Jprox, Hprox: scalar, Jacobian, and Hessian terms for all pairs
    - Lag_obj, Lag_J, Lag_H: total ALM objective, gradient, and Hessian
    - w_ls_D, d_D, w_Dij: various projections needed for updates
    - Hprox_diag: regularization term for Hessians (tau * I)
    - prox_time: time spent in prox evaluations
    """
    
    def __init__(self, n, K, tau):
        self.y = np.zeros(K)
        self.Lprox = np.zeros(K)
        self.Jprox = np.zeros((n, K))
        self.Hprox = [np.zeros((n, n)) for _ in range(K)]
        self.Hprox_diag = tau * np.eye(n)
        self.w_ls_D = np.zeros(K)
        self.d_D = np.zeros(K)
        self.w_Dij = np.zeros(K)
        self.Lag_obj = 0.0
        self.Lag_J = np.zeros(n)
        self.Lag_H = np.zeros((n, n))
        self.prox_time = 0.0


class LineSearchParameters:
    
    """
    Parameters for the Armijo backtracking line search.

    Attributes:
    - c: Armijo parameter (controls sufficient decrease)
    - max_iter_ls: maximum number of line search steps
    - beta: multiplicative factor for reducing step size
    """
    
    def __init__(self, c, max_iter_ls, beta):
        self.c = c
        self.max_iter_ls = max_iter_ls
        self.beta = beta


class SSNVar:
    
    """
    Stores intermediate variables used during Semi-Smooth Newton iterations.

    Attributes:
    - L_obj: ALM objective value
    - L_grad, L_hess: gradient and Hessian of Lagrangian
    - alpha_ssn: current SSN step size
    - w_ssn: current primal guess
    - w_ls: candidate updated primal after line search
    - y_ssn: current prox outputs
    - w_ssn_D: ⟨w_ssn, D_ij⟩ values
    """
    
    def __init__(self, PI):
        self.L_obj = 0.0
        self.L_grad = np.zeros(PI.n)
        self.L_hess = np.zeros((PI.n, PI.n))
        self.alpha_ssn = 1.0
        self.w_ssn = np.zeros(PI.n)
        self.w_ls = np.zeros(PI.n)
        self.y_ssn = np.zeros(len(PI.K))
        self.w_ssn_D = np.zeros(len(PI.K))

class SSNVar:
    def __init__(self, PI):
        self.L_obj = 0.0
        self.L_grad = np.zeros(PI.d)      
        self.L_hess = np.zeros((PI.d, PI.d))  
        self.alpha_ssn = 1.0
        self.w_ssn = np.zeros(PI.d)      
        self.w_ls = np.zeros(PI.d)        
        self.y_ssn = np.zeros(len(PI.K))
        self.w_ssn_D = np.zeros(len(PI.K))


class SSNParameters:
    
    """
    Controls the behavior of the Semi-Smooth Newton method.

    Attributes:
    - tol_ssn: tolerance for SSN convergence
    - max_iter_ssn: maximum SSN iterations per ALM outer iteration
    """
    
    def __init__(self, tol_ssn, max_iter_ssn):
        self.tol_ssn = tol_ssn
        self.max_iter_ssn = max_iter_ssn


class ALMLog:
    
    """
    Stores logging information from the entire ALM run.

    Attributes:
    - alm_time: total wall time
    - alm_iter: number of ALM iterations performed
    - L_final: final objective value
    - ssn_times, ssn_iters, ssn_wD_times: SSN timing and iteration stats
    - prox_times, prox_allocs: proximal operator performance
    - lsearch_times, lsearch_iters, lsearch_allocs: line search diagnostics
    """
    def __init__(self, max_alm, max_ssn, max_lsearch):
        
        self.alm_time = 0.0
        self.alm_iter = 0
        self.L_final = 0.0
        self.ssn_times = np.zeros(max_alm)
        self.ssn_iters = np.zeros(max_alm, dtype=int)
        self.ssn_wD_times = np.zeros(max_alm)
        self.prox_allocs = np.zeros(max_alm)
        self.lsearch_allocs = np.zeros(max_alm)
        self.ssn_d_times = np.zeros((max_alm, max_ssn))
        self.prox_times = np.zeros((max_alm, max_ssn))
        self.lsearch_iters = np.zeros((max_alm, max_ssn), dtype=int)
        self.lsearch_times = np.zeros((max_alm, max_ssn, max_lsearch))