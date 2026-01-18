function prox_aprox_graph(delta, label::String)
    f(x) = min(1, max(0, x - delta))
    x_values = -50:0.01:50
    y_values = [f(x) for x in x_values]

    plot = Plots.plot(x_values, y_values,
                      xlabel="x", ylabel="ℓ(x)", 
                      title="Linear Indicator Approximation ($label)",
                      legend=false)
    return plot, f
end


function prox_all_points_graph!(plot, w, lambda_vec, sigma, PI, f, label_prefix)
    N = length(PI.K)
    x_vals = Vector{Float64}(undef, N)
    y_vals = Vector{Float64}(undef, N)

    for idx in 1:N
        Dij = view(PI.D, :, idx)
        w_dot_dij = dot(w, Dij)
        lambda_ij = lambda_vec[idx]
        x_vals[idx] = w_dot_dij - lambda_ij / sigma
        y_vals[idx] = f(x_vals[idx])
    end

    Plots.scatter!(plot, x_vals, y_vals, markersize=3,
                   label="", color=(label_prefix == "initial" ? :blue : :red),
                   alpha=0.3)
end

function plot_projected_data_with_directions(dataset::DataSet, w_svm::Vector, w0_shifted::Vector, w_star::Vector)
    X = dataset.X
    y = dataset.y

    u1 = normalize(w_svm)
    temp = w0_shifted - dot(w0_shifted, u1) * u1
    u2 = normalize(temp)

    # Project data
    m = size(X, 2)
    x1_proj = [dot(u1, X[:, i]) for i in 1:m]
    x2_proj = [dot(u2, X[:, i]) for i in 1:m]

    # Project direction vectors
    arrow_scale = 10.0
    w_svm_proj = arrow_scale * normalize([dot(u1, w_svm), dot(u2, w_svm)])
    w0_proj = arrow_scale * normalize([dot(u1, w0_shifted), dot(u2, w0_shifted)])
    w_star_proj = arrow_scale * normalize([dot(u1, w_star), dot(u2, w_star)])

    # Plot data points
    class0 = findall(y .== 0)
    class1 = findall(y .== 1)

    p = scatter(x1_proj[class0], x2_proj[class0], label="Class 0", color=:blue, alpha=0.5, legend=:topright)
    scatter!(p, x1_proj[class1], x2_proj[class1], label="Class 1", color=:red, alpha=0.5)

    # Dummy points for legend entries
    scatter!([NaN], [NaN], color=:green, marker=:circle, label="w_svm (true)")
    scatter!([NaN], [NaN], color=:orange, marker=:circle, label="w₀_shifted (ALM init)")
    scatter!([NaN], [NaN], color=:black, marker=:circle, label="w* (ALM result)")

    # Plot arrows from origin
    quiver!([0.0], [0.0], quiver=([w_svm_proj[1]], [w_svm_proj[2]]), color=:green, label="")
    quiver!([0.0], [0.0], quiver=([w0_proj[1]], [w0_proj[2]]), color=:orange, label="")
    quiver!([0.0], [0.0], quiver=([w_star_proj[1]], [w_star_proj[2]]), color=:black, label="")

    xlabel!("u₁ (along w_svm)")
    ylabel!("u₂ (orthogonal)")
    title!("Projected Data with Direction Vectors")
    return p
end

function diagnose_w0_shifted_alignment(PI::ProblemInstance; tol::Float64 = 1e-3)
    println("----- Alignment Diagnostics: w₀_shifted vs w_svm -----")

    # Normalize vectors
    w0 = normalize(PI.w0_shifted)
    ws = normalize(PI.w_svm)

    # Angle between w0_shifted and w_svm
    angle_deg = acosd(dot(w0, ws))
    println("Angle between w₀_shifted and w_svm: $(round(angle_deg, digits=2)) degrees")

    # Norms
    println("‖w₀_shifted‖ = $(round(norm(PI.w0_shifted), digits=4))")
    println("‖w_svm‖ = $(round(norm(PI.w_svm), digits=4))")

    # Pairwise projections w₀' D_ij
    D = PI.D
    num_pairs = length(PI.K)
    projections = [dot(PI.w0_shifted, view(D, :, k)) for k in 1:num_pairs]

    mean_abs_proj = mean(abs.(projections))
    near_zero_count = count(abs.(projections) .< tol)
    percent_near_zero = 100 * near_zero_count / num_pairs

    println("Mean |w₀′ Dᵢⱼ| = $(round(mean_abs_proj, digits=5))")
    println("Pairs with |w₀′ Dᵢⱼ| < $tol: $near_zero_count / $num_pairs ($(round(percent_near_zero, digits=2))%)")

    return nothing
end

function test_threads()
    K = 10
    nt = Threads.nthreads()
    blk = div(K + nt - 1, nt)
    output = zeros(Int, K)

    Threads.@threads for t in 1:nt
        lo = (t-1)*blk + 1
        hi = min(t*blk, K)
        if lo > hi
            continue
        end
        for i in lo:hi
            output[i] = t
        end
    end

    println("Thread assignment:", output)
end

test_threads()

