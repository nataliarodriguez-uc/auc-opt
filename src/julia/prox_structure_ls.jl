function compute_prox_ls!(prox_wD::Vector{Float64}, almvar::ALMVar, proxvar::ProxVar, PI::ProblemInstanceBatch)

    #Extract the Indexes of the Dot Product
    #@threads for idx in 1:length(PI.K)
    L = 0.0

    @inbounds for idx in 1:length(PI.K)

        w_Dij = prox_wD[idx] - almvar.lambda[idx] / almvar.sigma

        # Compute the Proximal Mapping
        if almvar.gamma < 2
            L += prox_smallgamma_ls(w_Dij, almvar)

        elseif almvar.gamma == 2

            L += prox_gamma2_ls(w_Dij, almvar)
            
        else
            L += prox_largegamma_ls(w_Dij, almvar)
        end
        

    end

    # Log time and memory usage for compute_prox!
    #println("  ⏱️ Prox computation time: $(prox_stats.time) sec, $(prox_stats.bytes / 1e6) MB allocated")

    return L 

end 

function prox_smallgamma_ls(w_Dij, almvar)


    if w_Dij < almvar.delta
        
        #Optimal value for the proximal term
        return 0     

    elseif almvar.delta <= w_Dij <= almvar.delta + almvar.gamma
        
        #Optimal value for the proximal term
       return 0.5 / almvar.gamma * (almvar.delta - w_Dij)^2
        

    elseif almvar.delta + almvar.gamma < w_Dij < 1 + almvar.delta + 0.5 * almvar.gamma
        
        #Optimal value for the proximal term
        return w_Dij - almvar.delta - 0.5 * almvar.gamma

    else
        
        #Optimal value for the proximal term
        return 1.0
    
    end

end

function prox_gamma2_ls(w_Dij, almvar)

    if w_Dij < almvar.delta

       return 0   

    elseif w_Dij < almvar.delta + almvar.gamma

        return 0.5 / almvar.gamma * (almvar.delta - w_Dij)^2

    else
        
        return 1.0
        
    end
end

function prox_largegamma_ls(w_Dij, almvar)

    if w_Dij < almvar.delta

        return 0   

    elseif w_Dij < almvar.delta + sqrt(2 * almvar.gamma)

        return 0.5 / almvar.gamma * (almvar.delta - w_Dij)^2
        
    else

        return 1.0
    
    end
end


