# Define update functions as methods for SSNParameters
function update_tol!(SP::SSNParameters, t::Int)
    if t < 2
        SP.tol_ssn = 1e-2  # Phase A: Rough tolerance
    elseif 2 <= t <= 10
        SP.tol_ssn = 1e-4  # Phase B: Moderate tolerance
    else
        SP.tol_ssn = 1e-6  # Phase C: High tolerance
    end
end

function update_iter!(SP::SSNParameters, t::Int)
    if t <= 2
        SP.max_iter_ssn = 10
    else
        SP.max_iter_ssn = 25
    end
end

function update_sigma_gamma!(almvar::ALMVar,AP::ALMParameters)

    almvar.sigma *= AP.sigma_scale

    almvar.gamma = 1.0 / almvar.sigma
end

function update_proxmethod!(almvar::ALMVar)

    # Update the gamma value based on the current sigma
    if almvar.gamma < 2
        # Small gamma case
        almvar.prox_method_ls = prox_smallgamma_ls  # Use small gamma method
        almvar.prox_method_ssn = prox_smallgamma_ssn
    elseif almvar.gamma == 2
        # Gamma equals 2 case
        almvar.prox_method_ls = prox_gamma2_ls  # Use gamma = 2 method
        almvar.prox_method_ssn = prox_smallgamma_ssn
     
    else
        # Large gamma case
        almvar.prox_method_ls = prox_largegamma_ls  # Use large gamma method
        almvar.prox_method_ssn = prox_smallgamma_ssn
   
    end
end

function update_proxmethod_thread!(almvar::ALMVar)
    if almvar.gamma < 2
        almvar.prox_method_ls = prox_smallgamma_ls_thread
        almvar.prox_method_ssn = prox_smallgamma_ssn_thread
    elseif almvar.gamma == 2
        almvar.prox_method_ls = prox_gamma2_ls_thread
        almvar.prox_method_ssn = prox_gamma2_ssn_thread        # ✅ correct method
    else
        almvar.prox_method_ls = prox_largegamma_ls_thread
        almvar.prox_method_ssn = prox_largegamma_ssn_thread    # ✅ correct method
    end
end
