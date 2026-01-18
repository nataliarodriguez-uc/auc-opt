function compute_prox_ls_thread!(
    prox_wD::Vector{Float64},
    almvar::ALMVar,
    proxvar::ProxVar, # passed on to the prox_* fns if needed?
    PI::ProblemInstance)

    nt  = nthreads()              # number of Julia threads
    blk = div(length(PI.K) + nt - 1, nt)     # ⌈K / nt⌉ – size of each contiguous block
    partial = zeros(Float64, nt)  # one independent cache‑line per thread
    
    @threads for t in 1:nt        # t = thread id
      lo = (t-1)*blk + 1          # inclusive start of this block
      hi = min(t*blk, length(PI.K))          # inclusive end   of this block
      local_L = 0.0

      @inbounds @simd for idx in lo:hi # simd ok because no races
        w_Dij = prox_wD[idx] - almvar.lambda[idx] / almvar.sigma
        local_L += almvar.prox_method_ls(w_Dij, almvar)
      end
      partial[t] = local_L
  end
  # reduce
  return sum(partial)
end

function prox_smallgamma_ls_thread(w_Dij, almvar)


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

function prox_gamma2_ls_thread(w_Dij, almvar)

    if w_Dij < almvar.delta

       return 0   

    elseif w_Dij < almvar.delta + almvar.gamma

        return 0.5 / almvar.gamma * (almvar.delta - w_Dij)^2

    else
        
        return 1.0
        
    end
end

function prox_largegamma_ls_thread(w_Dij, almvar)

    if w_Dij < almvar.delta

        return 0   

    elseif w_Dij < almvar.delta + sqrt(2 * almvar.gamma)

        return 0.5 / almvar.gamma * (almvar.delta - w_Dij)^2
        
    else

        return 1.0
    
    end
end


