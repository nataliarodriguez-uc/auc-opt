mutable struct DataSet_TrainTest
    m_train::Int
    n_train::Int
    m_test::Int
    n_test::Int
    num_classes::Int
    class_ratios::Vector{Float64}
    sep_distance::Float64
    X_train::Matrix{Float64}
    y_train::Vector{Int}
    X_test::Matrix{Float64}
    y_test::Vector{Int}
end

function DataSet_TrainTest(
    dataset::DataSet;
    ratio::Float64=0.7,
    stratified::Bool=true,
    balance_train::Bool=false,
    balance_test::Bool=false,
    pos_ratio::Float64=0.5  # 0.5 = balanced, 0.1 = imbalanced
)
    X, y = dataset.X, dataset.y
    m, n = dataset.m, dataset.n
    num_classes = dataset.num_classes
    class_ratios = dataset.class_ratios
    sep_distance = dataset.sep_distance

    # --- Split the data ---
    all_inds = collect(1:m)
    train_inds, test_inds = Int[], Int[]

    if stratified
        for cls in unique(y)
            cls_inds = findall(i -> y[i] == cls, all_inds)
            n_train_cls = Int(round(ratio * length(cls_inds)))
            perm = randperm(length(cls_inds))
            append!(train_inds, cls_inds[perm[1:n_train_cls]])
            append!(test_inds, cls_inds[perm[n_train_cls+1:end]])
        end
    else
        perm = randperm(m)
        n_train = Int(round(ratio * m))
        train_inds = perm[1:n_train]
        test_inds  = perm[n_train+1:end]
    end

    X_train, y_train = X[:, train_inds], y[train_inds]
    X_test,  y_test  = X[:, test_inds],  y[test_inds]

    # --- Balance helper ---
    function balance_xy(X, y, pos_ratio)
        pos_inds = findall(y .== 1)
        neg_inds = findall(y .== 0)
        total = min(length(pos_inds) / pos_ratio, length(neg_inds) / (1 - pos_ratio))
        n_pos = Int(round(pos_ratio * total))
        n_neg = Int(round((1 - pos_ratio) * total))
        pos_sample = pos_inds[randperm(length(pos_inds))[1:n_pos]]
        neg_sample = neg_inds[randperm(length(neg_inds))[1:n_neg]]
        selected = vcat(pos_sample, neg_sample)
        return X[:, selected], y[selected]
    end

    if balance_train
        X_train, y_train = balance_xy(X_train, y_train, pos_ratio)
    end
    if balance_test
        X_test, y_test = balance_xy(X_test, y_test, pos_ratio)
    end

    return DataSet_TrainTest(
        size(X_train, 2), size(X_train, 1),
        size(X_test, 2), size(X_test, 1),
        num_classes,
        class_ratios,
        sep_distance,
        X_train, y_train,
        X_test, y_test
    )
end

mutable struct PairwiseDifferencesTrainTest
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    S::Int
    w::Vector{Float64}

    function PairwiseDifferencesTrainTest(split::DataSet_TrainTest)
        Random.seed!(1034)

        X = split.X_train
        y = split.y_train
        num_samples = size(X, 2)
        num_features = size(X, 1)

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
            D[:, k] .= X[:, j] .- X[:, i]
        end

        w = randn(num_features)
        return new(K, D, num_samples, w)
    end
end

struct ProblemInstanceTrainTest
    K::Vector{Tuple{Int, Int}}
    D::Matrix{Float64}
    m::Int
    n::Int
    w0::Vector{Float64}
    lambda0::Vector{Float64} 

    function ProblemInstanceTrainTest(split::DataSet_TrainTest)
        pairwise_diff = PairwiseDifferencesTrainTest(split)
        return new(
            pairwise_diff.K,
            pairwise_diff.D,
            pairwise_diff.S,
            size(pairwise_diff.D, 1),
            pairwise_diff.w,
            zeros(length(pairwise_diff.K))
        )
    end
end
