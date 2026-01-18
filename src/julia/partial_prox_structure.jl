function compute_prox_ssn!(prox_wD::Vector{Float64}, almvar::ALMVar, proxvar::ProxVar, PI::ProblemInstanceBatch)

    # 1. Vectorized computation of w_Dij (much more efficient than inside the loop)
    @. proxvar.w_Dij = prox_wD .- almvar.lambda ./ almvar.sigma

    # 2. Fill all proxvar values with zero to avoid stale data (DO THIS HERE)
    fill!(proxvar.y, 0.0)
    proxvar.Lag_obj = 0.0
    fill!(proxvar.Lag_H, 0.0)
    fill!(proxvar.Lag_J, 0.0)
    

    # 3. Loop over all idx values
    for idx in 1:length(PI.K)

        D_ij = view(PI.D, :, idx)  # Avoid copy
        almvar.prox_method_ssn(idx, proxvar.w_Dij[idx], D_ij, almvar, proxvar, PI)
    end

end

function prox_smallgamma_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta
        # Region I1: flat
        proxvar.y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end

    elseif delta <= w_Dij <= delta + gamma
        # Region I2: quadratic
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff^2

        @inbounds @simd for i in 1:n
            proxvar.Lag_J[i] += sigma * diff * D_ij[i]
        end

        @inbounds proxvar.Lag_H .+= sigma .* almvar.D_product[idx]

    elseif delta + gamma < w_Dij <=`` 1 + delta + 0.5 * gamma
        # Region I3: linear ramp
        proxvar.y[idx] = delta - gamma
        proxvar.Lag_obj += w_Dij - delta - 0.5 * gamma

        @inbounds @simd for i in 1:n
            proxvar.Lag_J[i] += sigma * D_ij[i]
        end
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end

    else
        # Region I4: saturated
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end
    end
end


function prox_gamma2_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta

    if w_Dij < delta

        # Region I1: flat
        proxvar.y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end

    elseif delta <= w_Dij <= delta + gamma

        # Region I2: quadratic
        diff = delta - w_Dij
        proxvar.Lag_obj += 0.5 / gamma * diff^2

        @inbounds @simd for i in 1:n
            proxvar.Lag_J[i] += sigma * diff * D_ij[i]
        end

        @inbounds proxvar.Lag_H .+= sigma .* almvar.D_product[idx]

    else
        # Region I4: saturated
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0

        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end
    end
end

function prox_largegamma_ssn(idx, w_Dij, D_ij, almvar, proxvar, PI)
    n = PI.n
    gamma = almvar.gamma
    sigma = almvar.sigma
    delta = almvar.delta
    sqrt_term = 

    if w_Dij < delta
        # Region I1: flat
        proxvar.y[idx] = w_Dij
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end

    elseif delta <= w_Dij <= delta + sqrt(2 * gamma)
        # Region I2: quadratic
        diff = delta - w_Dij
        proxvar.y[idx] = delta
        proxvar.Lag_obj += 0.5 / gamma * diff^2

        @inbounds @simd for i in 1:n
            proxvar.Lag_J[i] += sigma * diff * D_ij[i]
        end

        @inbounds proxvar.Lag_H .+= sigma .* almvar.D_product[idx]
        
    else
        # Region I4: saturated
        proxvar.y[idx] = w_Dij
        proxvar.Lag_obj += 1.0
        @inbounds @simd for i in 1:n
            proxvar.Lag_H[i, i] += almvar.tau
        end
    end
end


