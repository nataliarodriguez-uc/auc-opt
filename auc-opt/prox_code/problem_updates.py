from problem_variables import ALMVar, ALMParameters, SSNParameters
from prox import *

def update_tol(SP: SSNParameters, t: int):
    
    """
    Dynamically adjust the SSN tolerance based on the outer ALM iteration number.

    Parameters:
    - SP: SSNParameters object whose tolerance will be updated
    - t: current ALM iteration index
    """
    
    if t < 2:
        SP.tol_ssn = 1e-2
    elif 2 <= t <= 10:
        SP.tol_ssn = 1e-4
    else:
        SP.tol_ssn = 1e-6

def update_iter(SP: SSNParameters, t: int):
    
    """
    Adjust the max SSN iterations based on the outer ALM iteration number.

    Parameters:
    - SP: SSNParameters object
    - t: current ALM iteration index
    """
    
    SP.max_iter_ssn = 10 if t <= 2 else 25

def update_sigma_gamma(almvar: ALMVar, AP: ALMParameters):
    
    """
    Increase the ALM penalty parameter and update gamma.

    Parameters:
    - almvar: current ALM variable state
    - AP: ALM parameter configuration (contains the scaling factor)
    """
    
    almvar.sigma *= AP.sigma_scale
    almvar.gamma = 1.0 / almvar.sigma

def update_proxmethod(almvar: ALMVar):
    
    """
    Choose appropriate proximal operator variant based on current gamma.

    This method assigns the correct proximal computation functions (for line search and SSN)
    to the ALM state object depending on the penalty regime.

    Parameters:
    - almvar: ALM variable object with current gamma value
    """
    
    gamma = almvar.gamma
    if gamma < 2:
        almvar.prox_method_ls = prox_smallgamma_ls
        almvar.prox_method_ssn = prox_smallgamma_ssn
    elif gamma == 2:
        almvar.prox_method_ls = prox_gamma2_ls
        almvar.prox_method_ssn = prox_gamma2_ssn
    else:
        almvar.prox_method_ls = prox_largegamma_ls
        almvar.prox_method_ssn = prox_largegamma_ssn

def update_proxmethod_thread(almvar: ALMVar):
    
    """
    Thread-safe version of `update_proxmethod`, used when calling threaded proximal operators.

    Parameters:
    - almvar: ALM variable object
    """
    
    gamma = almvar.gamma
    if gamma < 2:
        almvar.prox_method_ls = prox_smallgamma_ls
        almvar.prox_method_ssn = prox_smallgamma_ssn
    elif gamma == 2:
        almvar.prox_method_ls = prox_gamma2_ls
        almvar.prox_method_ssn = prox_gamma2_ssn
    else:
        almvar.prox_method_ls = prox_largegamma_ls
        almvar.prox_method_ssn = prox_largegamma_ssn
