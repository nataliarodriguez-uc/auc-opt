function build_pairwise_batch(X::Matrix{Float64}, y::Vector{Int}, n_pos::Int, n_neg::Int)
    pos_idx = findall(y .== 1)
    neg_idx = findall(y .== 0)

    p_sample = rand(pos_idx, n_pos)
    n_sample = rand(neg_idx, n_neg)

    K = Tuple{Int,Int}[]
    for i in p_sample
        for j in n_sample
            push!(K, (i, j))
        end
    end

    num_pairs = length(K)
    d = size(X, 1)
    D = Matrix{Float64}(undef, d, num_pairs)

    Threads.@threads for k in 1:num_pairs
        i, j = K[k]
        D[:, k] .= X[:, j] .- X[:, i]  # negative - positive
    end

    return K, D
end

function batched_alm_epochs!(
    PIs::Vector{ProblemInstance},
    AP0::ALMParameters, SP0::SSNParameters, LS0::LineSearchParameters;
    sigma0=0.5, tau0=1e-5, alpha0=1.0
)
    # Initialization from first ProblemInstance
    w = copy(PIs[1].w0)
    λ = copy(PIs[1].lambda0)

    for epoch in 1:length(PIs)
        PI = PIs[epoch]

        # Inject current state
        PI.w0 .= w
        PI.lambda0 .= λ

        # Batch-specific shallow copy
        AP = deepcopy(AP0)
        AP.max_iter_alm = 2  # short update per batch

        almvar, _ = alm!(sigma0, tau0, alpha0, PI, AP, SP0, LS0)

        w .= almvar.w
        λ .= almvar.lambda

        # Optional decay (if desired):
        sigma0 *= 0.95
        tau0 *= 0.95
    end

    return w, λ
end

mutable struct PairwiseDifferencesBatch
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    w::Vector{Float64}

    function PairwiseDifferencesBatch(X::Matrix{Float64}, y::Vector{Int}, n_pos::Int, n_neg::Int)
        pos_idx = findall(y .== 1)
        neg_idx = findall(y .== 0)

        p_sample = rand(pos_idx, n_pos)
        n_sample = rand(neg_idx, n_neg)

        K = Tuple{Int, Int}[]
        for i in p_sample, j in n_sample
            push!(K, (i, j))
        end

        num_pairs = length(K)
        d = size(X, 2)
        D = Matrix{Float64}(undef, d, num_pairs)

        Threads.@threads for k in 1:num_pairs
            i, j = K[k]
            D[:, k] .= X[j, :] .- X[i, :]
        end

        w = randn(d)
        return new(K, D, w)
    end
end

mutable struct ProblemInstanceBatch
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    m::Int
    n::Int
    w_svm::Vector{Float64}
    w0::Vector{Float64}
    w0_shifted::Vector{Float64}
    lambda0::Vector{Float64}
end

function ProblemInstanceBatch(pd::PairwiseDifferencesBatch)
    d = size(pd.D, 1)
    w0 = copy(pd.w)
    lambda0 = zeros(length(pd.K))
    w_svm = zeros(d)

    return ProblemInstanceBatch(
        pd.K,
        pd.D,
        -1,
        d,
        w_svm,
        w0,
        copy(w0),
        lambda0
    )
end



