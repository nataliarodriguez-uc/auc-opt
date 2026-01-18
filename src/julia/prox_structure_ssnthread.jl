using Base.Threads

function compute_prox_ssn_thread!(prox_wD::Vector{Float64}, almvar::ALMVar, proxvar::ProxVar, PI::ProblemInstance)
    @. proxvar.w_Dij = prox_wD .- almvar.lambda ./ almvar.sigma

    fill!(proxvar.y, 0.0)
    fill!(proxvar.Lag_J, 0.0)
    fill!(proxvar.Lag_H, 0.0)
    proxvar.Lag_obj = 0.0

    n = PI.n
    d = length(PI.K)
    T = nthreads()

    thread_objs = [Ref(0.0) for _ in 1:T]
    thread_Js   = [zeros(n) for _ in 1:T]
    thread_Hs   = [zeros(n, n) for _ in 1:T]
    thread_ys   = [zeros(d) for _ in 1:T]

    blk = div(d + T - 1, T)

    Threads.@threads for t in 1:T
        lo = (t - 1) * blk + 1
        hi = min(t * blk, d)

        local_obj = thread_objs[t]
        local_J   = thread_Js[t]
        local_H   = thread_Hs[t]
        local_y   = thread_ys[t]

        @inbounds for idx in lo:hi
            D_ij = view(PI.D, :, idx)
            w    = proxvar.w_Dij[idx]
            almvar.prox_method_ssn(idx, w, D_ij, almvar, local_obj, local_J, local_H, local_y, PI)
        end
    end

    proxvar.Lag_obj = sum(obj[] for obj in thread_objs)
    proxvar.Lag_J   .= sum(thread_Js)
    proxvar.Lag_H   .= reduce(+, thread_Hs)

    # Aggregate thread-local y vectors
    for t in 1:T
        @inbounds for i in 1:length(PI.K)
            proxvar.y[i] = thread_ys[t][i]  # assumes only one thread wrote to each i
        end
    end
end

function prox_smallgamma_ssn_thread(
    idx::Int,
    w_Dij::Float64,
    D_ij::AbstractVector{Float64},
    almvar::ALMVar,
    lag_obj::Base.RefValue{Float64},
    lag_J::Vector{Float64},
    lag_H::Matrix{Float64},
    local_y::Vector{Float64},  # ✅ thread-safe y
    PI::ProblemInstance
)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta
        local_y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end

    elseif w_Dij <= delta + gamma
        local_y[idx] = delta
        diff = delta - w_Dij
        lag_obj[] += 0.5 / gamma * diff^2
        @inbounds @simd for i in 1:n
            lag_J[i] += sigma * diff * D_ij[i]
        end
        @inbounds lag_H .+= sigma .* almvar.D_product[idx]

    elseif w_Dij < 1 + delta + 0.5 * gamma
        local_y[idx] = delta - gamma
        lag_obj[] += w_Dij - delta - 0.5 * gamma
        @inbounds @simd for i in 1:n
            lag_J[i] += sigma * D_ij[i]
        end
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end

    else
        local_y[idx] = w_Dij
        lag_obj[] += 1.0
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end
    end
end



function prox_gamma2_ssn_thread(
    idx::Int,
    w_Dij::Float64,
    D_ij::AbstractVector{Float64},
    almvar::ALMVar,
    lag_obj::Base.RefValue{Float64},
    lag_J::Vector{Float64},
    lag_H::Matrix{Float64},
    local_y::Vector{Float64},  # ✅ thread-safe
    PI::ProblemInstance
)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta
        local_y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end

    elseif w_Dij <= delta + gamma
        local_y[idx] = delta
        diff = delta - w_Dij
        lag_obj[] += 0.5 / gamma * diff^2
        @inbounds @simd for i in 1:n
            lag_J[i] += sigma * diff * D_ij[i]
        end
        @inbounds lag_H .+= sigma .* almvar.D_product[idx]

    else
        local_y[idx] = w_Dij
        lag_obj[] += 1.0
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end
    end
end


function prox_largegamma_ssn_thread(
    idx::Int,
    w_Dij::Float64,
    D_ij::AbstractVector{Float64},
    almvar::ALMVar,
    lag_obj::Base.RefValue{Float64},
    lag_J::Vector{Float64},
    lag_H::Matrix{Float64},
    local_y::Vector{Float64},  # ✅ thread-safe
    PI::ProblemInstance
)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta
    sqrt_term = delta + sqrt(2 * gamma)

    if w_Dij < delta
        local_y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end

    elseif w_Dij <= sqrt_term
        local_y[idx] = delta
        diff = delta - w_Dij
        lag_obj[] += 0.5 / gamma * diff^2
        @inbounds @simd for i in 1:n
            lag_J[i] += sigma * diff * D_ij[i]
        end
        @inbounds lag_H .+= sigma .* almvar.D_product[idx]

    else
        local_y[idx] = w_Dij
        lag_obj[] += 1.0
        @inbounds @simd for i in 1:n
            lag_H[i, i] += almvar.tau
        end
    end
end


