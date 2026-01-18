function line_search!(t, k, almlog::ALMLog, almvar::ALMVar,ssnvar::SSNVar, proxvar::ProxVar, d::Vector{Float64}, PI::ProblemInstanceBatch, LS::LineSearchParameters)

    # Initialize alpha_ls
    alpha_ls = ssnvar.alpha_ssn  

    #Objective at current w_ssn
    L_current = proxvar.Lag_obj
    J_current = proxvar.Lag_J

    # Line search iterations
    for l in 1:LS.max_iter_ls
        
        # Compute potential update for w_ls
        ssnvar.w_ls .= ssnvar.w_ssn .+ alpha_ls * d 
        proxvar.w_ls_D .= ssnvar.w_ssn_D .+ alpha_ls .* proxvar.d_D #! linear update

        # Compute the new objective with the candidate w_ls
        L_new = compute_prox_ls!(proxvar.w_ls_D, almvar, proxvar, PI)

        # Armijo Rule
        if L_new - L_current <= LS.c * alpha_ls * dot(J_current, d)
            
            almlog.lsearch_iters[t][k] = l 
            break  
        else
            alpha_ls *= LS.beta  
        end

    end

    #push!(almlog.lsearch_iters, local_ls_count)
    ssnvar.alpha_ssn = alpha_ls  # Update alpha_ssn

end