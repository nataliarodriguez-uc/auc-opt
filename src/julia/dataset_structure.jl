using Random, LinearAlgebra, DataFrames, Base.Threads

struct DataSet
    m::Int
    n::Int
    num_classes::Int
    class_ratios::Vector{Float64}
    sep_distance::Float64
    X::Matrix{Float64}
    y::Vector{Int}
    w_svm::Vector{Float64}

    function DataSet(
        m::Int, n::Int, num_classes::Int, class_ratios::Vector{Float64},
        sep_distance::Float64;
        feature_noise::Float64 = 0.0,
        flip_ratio::Float64 = 0.0
    )   
        @assert sum(class_ratios) ≈ 1.0 "Class ratios must sum to 1."
        Random.seed!(1034)
        

        class_sizes = round.(Int, class_ratios * m)
        while sum(class_sizes) != m
            class_sizes[1] += (m - sum(class_sizes))
        end

        hyperplane_normal = normalize(randn(n))
        X = Matrix{Float64}(undef, n, m)
        y = Int[]

        start_idx = 1
        for class_id in 0:(num_classes - 1)  # Now labels are 0 and 1
            num_samples_class = class_sizes[class_id + 1]
            points = randn(n, num_samples_class)
            shift_direction = (2 * class_id - 1) * sep_distance * hyperplane_normal
            X[:, start_idx:(start_idx + num_samples_class - 1)] = points .+ shift_direction
            append!(y, fill(class_id, num_samples_class))
            start_idx += num_samples_class
        end

        if feature_noise > 0
            X .+= feature_noise * randn(size(X))
        end

        if flip_ratio > 0
            num_to_flip = round(Int, flip_ratio * length(y))
            indices = randperm(length(y))[1:num_to_flip]
            for i in indices
                y[i] = 1 - y[i]  # Flip 0 ↔ 1
            end
        end

        return new(m, n, num_classes, class_ratios, sep_distance, X, y, hyperplane_normal)
    end
end

mutable struct PairwiseDifferences
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    S::Int
    w::Vector{Float64}

    function PairwiseDifferences(dataset::DataSet)
        Random.seed!(1034)

        num_samples = dataset.m
        num_features = dataset.n
        X = dataset.X
        y = dataset.y

        K = Vector{Tuple{Int, Int}}(undef, num_samples * num_samples)
        num_pairs = 0

        for i in 1:num_samples
            for j in 1:num_samples
                if y[i] > y[j]  # i = positive, j = negative
                    num_pairs += 1
                    K[num_pairs] = (i, j)
                end
            end
        end

        resize!(K, num_pairs)
        D = Matrix{Float64}(undef, num_features, num_pairs)

        @inbounds @threads for k in 1:num_pairs
            i, j = K[k]
            D[:, k] .= X[:, j] .- X[:, i]  # negative - positive
        end

        w = randn(num_features)
        return new(K, D, num_samples, w)
    end
end

struct ProblemInstance
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    m::Int
    n::Int
    w_svm::Vector{Float64}
    w0::Vector{Float64}
    w0_shifted::Vector{Float64}
    lambda0::Vector{Float64} 

    function ProblemInstance(dataset::DataSet)
        
        pairwise_diff = PairwiseDifferences(dataset)
        rng = MersenneTwister(1234)
        w0_shifted = randn(rng, size(pairwise_diff.D, 1))  # New w₀ with seed
        return new(pairwise_diff.K, pairwise_diff.D, pairwise_diff.S, size(pairwise_diff.D, 1), dataset.w_svm, pairwise_diff.w, w0_shifted, zeros(length(pairwise_diff.K)))
    end
end
