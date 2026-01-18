function alm!(
    sigma0::Float64,  
    tau0::Float64,    
    alpha0::Float64,  
    PI::ProblemInstanceBatch, 
    AP0::ALMParameters, 
    SP0::SSNParameters,
    LS0::LineSearchParameters
)

    SP = deepcopy(SP0)  # Copy SSN parameters
    AP = deepcopy(AP0)  # Copy ALM parameters
    LS = deepcopy(LS0)  # Copy Line Search parameters

    almlog = ALMLog(AP.max_iter_alm, SP.max_iter_ssn)
    almlog.alm_time = time()
    
    #Shortcut variables from Problem Instance
    w0 = PI.w0
    lambda0 = PI.lambda0
    #println("w0",w0)

    # Initialize ALM Variables
    almvar = ALMVar(tau0, sigma0, PI)  
    almvar.lambda .= lambda0  # Correctly initialize lambda
    almvar.sigma = sigma0  # Explicitly set sigma
    almvar.tau = tau0  # Explicitly set tau
    almvar.w .= w0  # Initialize w
    almvar.y .= zeros(Float64, length(PI.K))  # Initialize y
    almvar.alpha = alpha0  

    #Initialize SSN variables
    ssnvar = SSNVar(PI)
    ssnvar.w_ssn .= w0  # Initialize w_ssn

    #Initialize Proximal Mapping
    proxvar = ProxVar(PI.n, length(PI.K), almvar.tau)

    #Profile.init(delay=1e-8)
    #Profile.clear()
    #Profile.clear_malloc_data()

    for t in 1:AP.max_iter_alm

        # Update SSN parameters dynamically
        update_tol!(SP, t)
        update_iter!(SP, t)
        update_proxmethod!(almvar)

        # Compute ssn
        ssn_stats = @timed ssn!(t,almlog, almvar, ssnvar, proxvar, PI, SP, LS)
        almlog.ssn_times[t] = ssn_stats.time

        #Extract optimal solutions for w and y 
        almvar.w .= ssnvar.w_ssn
        almvar.y .= ssnvar.y_ssn
        almvar.w_D .= ssnvar.w_ssn_D

        #Compute this iteration's residuals
        almvar.cons_condition .= (1 / length(PI.K)) .* (almvar.y .- almvar.w_D)

        if norm(almvar.cons_condition, Inf) <= AP.tol_alm
            almlog.alm_iter = t
            almlog.L_final = ssnvar.L_obj/length(PI.K)
            break
        else
            #Update sigma and lambda
            update_sigma_gamma!(almvar, AP)
            almvar.lambda .+= almvar.sigma * (1 / length(PI.K)) .* (almvar.y .- almvar.w_D)
        end 

    end

    print("L_final",almlog.L_final)
    almlog.alm_time = time() - almlog.alm_time  # End timing

    # INITIAL plot
    #plot_init, f_init = prox_aprox_graph(almvar.delta, "initial")  # fresh plot
    #prox_all_points_graph!(plot_init, PI.w0_shifted, PI.lambda0, almvar.sigma, PI, f_init, "initial")
    #Plots.display(plot_init)  # display separately and immediately
    
    # FINAL plot
    #plot_final, f_final = prox_aprox_graph(almvar.delta, "final")  # fresh plot
    #prox_all_points_graph!(plot_final, almvar.w, almvar.lambda, almvar.sigma, PI, f_final, "final")
    #Plots.display(plot_final)

    #savefig(plot_init, "initial_plot.png")
    #savefig(plot_final, "final_plot.png")


    #open("ssn_profile.txt", "w") do s
        #Profile.print(IOContext(s, :displaysize => (24, 500)), groupby=:task)
    #end

    return almvar, almlog

end