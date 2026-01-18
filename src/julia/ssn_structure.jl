function ssn!(t::Int, almlog::ALMLog, almvar::ALMVar, ssnvar::SSNVar, proxvar::ProxVar,
    PI::ProblemInstanceBatch, SP::SSNParameters, LS::LineSearchParameters)

    #@profile begin 
        # Reset alpha_ssn to the initial value
        ssnvar.alpha_ssn = almvar.alpha  

        # (1) Time D' * w_ssn
        
        almlog.ssn_wD_times[t] = @elapsed mul!(ssnvar.w_ssn_D, PI.D', ssnvar.w_ssn)

        # (2) Store w_D for proximal call
        proxvar.w_ls_D .= ssnvar.w_ssn_D     

        # (4) Store result in y_ssn
        ssnvar.y_ssn .= proxvar.y

        # Start SSN iterations
        for k in 1:SP.max_iter_ssn

            timed_result = @timed compute_prox_ssn!(ssnvar.w_ssn_D, almvar, proxvar, PI)
            almlog.prox_times[t][k] = timed_result.time
            almlog.prox_allocs[t] += timed_result.bytes / 1024^2  # Convert bytes to MB
            

            # (5) Extract Lagrangian state
            ssnvar.L_obj = proxvar.Lag_obj
            ssnvar.L_grad .= proxvar.Lag_J
            ssnvar.L_hess .= proxvar.Lag_H

            # (6) Compute Newton direction
            d_time = @elapsed d = ssnvar.L_hess \ ssnvar.L_grad
            almlog.ssn_d_times[t][k] = d_time


            # (7) Check for convergence
            if norm(ssnvar.L_grad) <= SP.tol_ssn
                almlog.ssn_iters[t] = k
                break
            else
                if k > 1
                    # (8) Line search direction prep
                    mul!(proxvar.d_D, PI.D', d)

                    # (9) Line search + logging
                    timed_res = @timed line_search!(t, k, almlog, almvar, ssnvar, proxvar, d, PI, LS)
                    almlog.lsearch_times[t][k] = timed_res.time
                    almlog.lsearch_allocs[t] += timed_res.bytes / 1024^2


                end     
            
                # (10) Primal update
                ssnvar.w_ssn .-= ssnvar.alpha_ssn * d
                ssnvar.w_ssn_D .= proxvar.w_ls_D
                
            end

            
        end 

    #end 
    
end
