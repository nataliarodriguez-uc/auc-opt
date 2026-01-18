mutable struct ALMVar
    cons_condition_norm::Vector{Float64}
    cons_condition::Vector{Float64}  
    lambda::Vector{Float64}  
    tau::Float64  
    sigma::Float64  
    gamma::Float64  
    w::Vector{Float64}  
    y::Vector{Float64}  
    alpha::Float64  
    delta::Float64
    D_product::Vector{Matrix{Float64}}
    w_D::Vector{Float64}
    prox_method_ls::Function  # ← added field
    prox_method_ssn::Function  # ← added field

    function ALMVar(tau_init::Float64, sigma_init::Float64, PI::ProblemInstanceBatch)
        w_init = randn(PI.n)
        y_init = zeros(Float64, length(PI.K))
        w_D_init = zeros(Float64, length(PI.K))

        sigma = sigma_init
        gamma = 1 / sigma
        D_product = [PI.D[:, k] * PI.D[:, k]' for k in 1:length(PI.K)]

        # Initialize with default prox method
        prox_method_ssn = gamma < 2 ? prox_smallgamma_ssn : gamma == 2 ? prox_gamma2_ssn : prox_largegamma_ssn
        prox_method_ls = gamma < 2 ? prox_smallgamma_ls : gamma == 2 ? prox_gamma2_ls : prox_largegamma_ls

        return new(
            Float64[],                             # cons_condition_norm
            zeros(Float64, length(PI.K)),         # cons_condition
            zeros(Float64, length(PI.K)),         # lambda
            tau_init,
            sigma,
            gamma,
            w_init,
            y_init,
            1.0,
            0.0,
            D_product,
            w_D_init,
            prox_method_ls,
            prox_method_ssn                            # ✅ here
        )
    end
end



mutable struct ALMParameters
    max_iter_alm::Int  
    tau_scale::Float64  
    sigma_scale::Float64 
    tol_alm::Float64 

    function ALMParameters(; max_iter_alm::Int, tau_scale::Float64, sigma_scale::Float64, tol_alm::Float64)
        return new(max_iter_alm, tau_scale, sigma_scale, tol_alm)
    end
end

mutable struct ProxVar

    y::Vector{Float64}  # Proximal solution for each (i, j) pair
    Lprox::Vector{Float64}  # Function values for each pair
    Jprox::Matrix{Float64}  # Jacobian (d × K)
    Hprox::Vector{Matrix{Float64}}  # Hessian (K separate matrices)

    Hprox_diag::Matrix{Float64}

    w_ls_D::Vector{Float64}  # h'w computation
    d_D::Vector{Float64}  # d * D 
    w_Dij::Vector{Float64}  # w_Dij vector

    Lag_obj::Float64  # Accumulated Lagrangian objective
    Lag_J::Vector{Float64}  # Accumulated Lagrangian gradient (d)
    Lag_H::Matrix{Float64}  # Accumulated Lagrangian Hessian (d × d)
    prox_time::Float64  # Time taken to compute the proximal mapping

    function ProxVar(n::Int, K::Int, tau::Float64)
        return new(
            zeros(K),                  # y
            zeros(K),                  # Lprox
            zeros(n, K),               # Jprox
            [zeros(n, n) for _ in 1:K],# Hprox
            tau * Matrix{Float64}(I, n, n),  # ← Hprox_diag
            zeros(K),                  # w_ls_D
            zeros(K),                  # d_D
            zeros(K),                  # w_Dij
            0.0,                       # Lag_obj
            zeros(n),                  # Lag_J
            zeros(n, n),               # Lag_H
            0.0                        # prox_time
        )
    end
end

mutable struct LineSearchParameters
    c::Float64
    max_iter_ls::Int
    beta::Float64

    function LineSearchParameters(;c::Float64, max_iter_ls::Int, beta::Float64)
        return new(c, max_iter_ls, beta)
    end

end 

mutable struct SSNVar

    L_obj::Float64
    L_grad::Vector{Float64}
    L_hess::Matrix{Float64}
    alpha_ssn::Float64
    w_ssn::Vector{Float64}
    w_ls::Vector{Float64}
    y_ssn::Vector{Float64}
    w_ssn_D::Vector{Float64}  # w_ssn * D

    function SSNVar(PI::ProblemInstanceBatch)
        new(0.0, zeros(PI.n), zeros(PI.n, PI.n), 1.0, zeros(PI.n), zeros(PI.n), zeros(Float64, length(PI.K)), zeros(Float64, length(PI.K)))
    end
end 

mutable struct SSNParameters

    tol_ssn::Float64  # Tolerance for SSN solver
    max_iter_ssn::Int  # Maximum SSN iterations

    function SSNParameters(;tol_ssn::Float64, max_iter_ssn::Int)
        return new(tol_ssn, max_iter_ssn)
    end

end

mutable struct ALMLog
    alm_time::Float64                    # Total ALM time
    alm_iter::Int                        # Number of ALM iterations run
    L_final::Float64                     # Final Lagrangian objective

    ssn_times::Vector{Float64}          # SSN time per ALM iteration
    ssn_iters::Vector{Int}              # Number of SSN iterations per ALM iteration
    ssn_wD_times::Vector{Float64}       # Time to compute wᵗD per ALM iteration
    ssn_d_times::Vector{Vector{Float64}}# Time per SSN Newton step (fixed length)

    prox_times::Vector{Vector{Float64}}     # Prox time per SSN iteration (fixed length)
    prox_allocs::Vector{Float64}            # Memory allocation for prox per ALM iter

    lsearch_times::Vector{Vector{Float64}}  # Line search time per SSN iter (fixed length)
    lsearch_iters::Vector{Vector{Int}}      # Line search iteration count per SSN iter
    lsearch_allocs::Vector{Float64}         # Memory allocation for line search per ALM iter

    function ALMLog(max_alm::Int, max_ssn::Int)
        return new(
            0.0,
            0,
            0.0,
            zeros(Float64, max_alm),
            zeros(Int, max_alm),
            zeros(Float64, max_alm),
            [zeros(Float64, max_ssn) for _ in 1:max_alm],
            [zeros(Float64, max_ssn) for _ in 1:max_alm],
            zeros(Float64, max_alm),
            [zeros(Float64, max_ssn) for _ in 1:max_alm],
            [zeros(Int, max_ssn) for _ in 1:max_alm],
            zeros(Float64, max_alm)
        )
    end
end
