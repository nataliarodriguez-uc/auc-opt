# Project Overview

**Goal**: Develop efficient optimization algorithms for contrastive learning and ranking objectives that traditional methods cannot handle.

---

## The Problem: Beyond Empirical Risk Minimization

### What Works Well Today

Most machine learning models are trained using **Empirical Risk Minimization (ERM)**, where we minimize the average loss over individual data points:

$$
\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \ell(w, x_i)
$$

This formulation is powerful because:
- Gradients are straightforward to compute
- Stochastic gradient descent (SGD) applies directly
- Well-understood convergence theory
- Scales efficiently to large datasets

### What Breaks Down

However, many important ML objectives **cannot** be expressed this way because they fundamentally depend on **comparing pairs or sets of samples** rather than evaluating samples independently.

**Examples where ERM fails:**

| Objective | Why ERM Doesn't Work |
|-----------|---------------------|
| **AUC (Area Under ROC)** | Performance measured by ranking positive vs. negative samples |
| **Ranking Metrics** (MAP, NDCG) | Depends on relative ordering, not individual predictions |
| **Contrastive Learning** | Minimizes distance between similar pairs, maximizes for dissimilar pairs |
| **Precision@K** | Only top-K predictions matter, not pointwise loss |

The challenge: These objectives involve:
- **Non-differentiable indicator functions** like $\mathbb{1}(\text{score}_1 > \text{score}_2)$
- **Quadratic scaling**: $O(n^2)$ pairwise comparisons
- **Cannot be decomposed** into independent per-sample losses

**This is where X-Risk comes in.**

---

## The Solution: X-Risk Framework

### Core Idea

Instead of evaluating each sample independently, **evaluate each sample relative to a set of relevant comparisons**.

The X-Risk objective generalizes ERM to pairwise structures:

$$
\min_{w \in \mathcal{X}} \frac{1}{|S|} \sum_{z_i \in S} f_i\Big(g(w, z_i, S_i)\Big)
$$

**Components:**
- $S$: Full dataset
- $S_i \subseteq S$: Subset of samples relevant to $z_i$ (e.g., same class for positives, different class for negatives)
- $g(w, z_i, S_i)$: Aggregates pairwise comparisons between $z_i$ and samples in $S_i$
- $f_i(\cdot)$: Risk measure defining the final objective

### Why This Matters

This single formulation **unifies** seemingly different problems:

**1. AUC Optimization**
- $S^+ =$ positive class, $S^- =$ negative class
- Compare every positive sample against every negative sample
- Objective: Maximize probability that positive scores higher than negative

**2. Ranking Metrics**
- Given query $q$, compare relevant documents against irrelevant ones
- Objective: Relevant items should rank higher

**3. Contrastive Learning**
- For sample $z_i$, compare against same-class samples (pull together) and different-class samples (push apart)
- Objective: Minimize intra-class distance, maximize inter-class distance

By recognizing this common structure, we can develop **general-purpose optimization methods** that work across all these problems.

---

## Key Insight: Contrastive Learning as AUC Maximization

### The Connection

When the embedding space is **one-dimensional** (or can be projected to a line), contrastive learning reduces to a **ranking problem**:
- Samples from class A should score lower than samples from class B
- This is exactly what AUC measures!

**For binary classification:**
- **Contrastive objective**: Make similar pairs close, dissimilar pairs far
- **AUC objective**: Rank positive class higher than negative class
- **They are equivalent** when embeddings are scalar-valued

This observation is powerful because:
1. AUC has well-defined optimization theory
2. We can leverage ranking methods for contrastive learning
3. Existing AUC algorithms inform our approach

### Practical Implication

Instead of treating contrastive learning as a separate problem, we can:
- Formulate it as **pairwise AUC optimization**
- Apply **proximal methods** designed for ranking objectives
- Extend to multi-class settings through pairwise decompositions

---

## The Computational Challenge

### Scalability Bottleneck

Naively evaluating all pairwise interactions leads to:

$$
\text{Cost} = O(n^2) \text{ comparisons}
$$

For $n = 10{,}000$ samples:
- Number of pairs: $10{,}000^2 = 100{,}000{,}000$
- Clearly infeasible for large datasets!

### Exploiting Sparsity

**Key insight**: We don't need to evaluate all $O(n^2)$ pairs. The optimization problem exhibits natural sparsity that can be exploited:

1. **Many pairs are already correctly classified** and contribute zero gradient
   - If $w^\top(z_j - z_i) < \delta$, the pair is correctly ranked
   - The proximal operator maps these to a flat region with zero derivative
   - No update needed for these pairs

2. **Auxiliary variable formulation creates sparse structure**
   - By introducing $y_{ij} = w^\top(z_j - z_i)$ as constraints
   - The augmented Lagrangian only updates variables for "active" pairs
   - Inactive pairs (correctly classified) remain fixed

3. **Controlled pairwise sampling**
   - Rather than form the full $n^2$ matrix, sample a fixed number of pairs per batch
   - For example: 50 positive × 50 negative = 2,500 pairs (vs. potentially millions)
   - Sampling strategy ensures diverse coverage without exhaustive enumeration

**Result**: The effective complexity becomes $O(n \cdot k)$ where $k$ is the number of active pairs per sample, which is typically much smaller than $n$.

### Implementation via Sparse Representations

The pairwise difference vectors $(z_j - z_i)$ can be represented using:
- **Sparse index sets**: Only track pairs $(i,j)$ that need updates
- **Batch-wise construction**: Build small dense blocks rather than full matrix
- **Block Coordinate Descent**: Update only subsets of pairs per iteration

This sparse structure is critical for making the method computationally tractable on real datasets.

### Non-Smoothness Challenge

Most pairwise objectives use **indicator functions**:

$$
\mathbb{1}(w^\top z_i > w^\top z_j) = 
\begin{cases}
1 & \text{if } w^\top z_i > w^\top z_j \\
0 & \text{otherwise}
\end{cases}
$$

Problems:
- **Not differentiable** at the decision boundary
- Standard gradient descent fails
- Approximations (like sigmoid) lose theoretical guarantees

---

## Our Approach

This project addresses both challenges through:

### 1. **Piecewise Linear Surrogates**

Replace the indicator function with a **continuous approximation** $\ell_\delta(t)$ that:
- Is differentiable almost everywhere
- Retains theoretical properties (consistency, calibration)
- Enables efficient proximal operators

### 2. **Proximal Optimization**

Use **proximal gradient methods** that:
- Handle non-smooth objectives directly
- Have closed-form solutions for our surrogate loss
- Provide convergence guarantees

### 3. **Augmented Lagrangian Framework**

Decompose the problem using:
- **ALM (Augmented Lagrangian Method)** for constraint handling
- **SSN (Semi-Smooth Newton)** for efficient subproblem solving
- **Controlled sampling** to reduce computational cost

### 4. **Dual-Language Implementation**

- **Python**: High-level experimentation, data loading, visualization
- **Julia**: Performance-critical optimization routines
- Combines flexibility with computational efficiency

---

## What This Repository Provides

### Theory
- Reduction of contrastive learning to AUC optimization
- Piecewise linear surrogate construction
- Proximal operator derivations (γ-dependent cases)
- Convergence analysis

### Algorithms
- ALM + SSN optimization framework
- Controlled pairwise sampling strategies
- Adaptive hyperparameter updates

### Experiments
- **Synthetic SVM datasets**: Controlled geometry to study convergence
- **CIFAR-10 binary classification**: Real-world validation
- **Baseline comparisons**: BCE, LibAUC benchmarks
- **Sensitivity analysis**: Impact of σ, sampling configuration

### Implementation
- Modular Python/Julia codebase
- Jupyter notebooks with step-by-step examples
- Visualization tools for understanding algorithm behavior

---

## Document Roadmap

This is the first of five technical documents:

1. **[Overview](1_overview.md)** ← *You are here*  
   Problem motivation, X-Risk framework, connection to contrastive learning

2. **[Algorithm](2_algorithm.md)**  
   Mathematical formulation, surrogate loss construction, constraint formulation

3. **[Optimization](3_optimization.md)**  
   Proximal operators, ALM framework, SSN solver, convergence properties

4. **[Experiments](4_experiments.md)**  
   Synthetic and real-world results, hyperparameter studies, baseline comparisons

5. **[Mathematical Appendix](5_math_appendix.md)**  
   Complete derivations, proofs, technical details

---

## Key Takeaways

- Many important ML objectives cannot be optimized with standard ERM
- X-Risk provides a unifying framework for pairwise/ranking objectives
- Contrastive learning = AUC optimization in the binary case 
- Sparse structure and controlled sampling avoid O(n²) complexity
- Proximal methods handle non-smooth losses efficiently  
- This repository bridges theory and practical implementation

**Next**: Read [Algorithm Formulation](2_algorithm.md) to see how we formulate the optimization problem.

---

## References

- Yang, T. (2023). *Algorithmic foundations of empirical X-risk minimization*
- Khanh, P. D., Mordukhovich, B. S., & Phat, V. T. (2022). *A generalized Newton method for subgradient systems*

For complete bibliography, see [Mathematical Appendix](5_math_appendix.md).