# Project Overview

This document reviews the novel X-Risk framework and its adaptations of standard machine learning problems, followed by the conceptual foundation for pairwise ranking objectives. This project scope explains how pairwise ranking objectives capture AUC maximization for new efficient optimization approaches.

---

## Background and Motivation

### Empirical Risk Formulation Baseline

Nearly all supervised learning methods are built on **Empirical Risk Minimization (ERM)**, which can be formulated as

$$
\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \ell(w, x_i).
$$

where we minimize the expectation (average) of each loss function $\ell (\dot)$ evaluation between the prediction and its true value by optimizing for the parameter $w$. 

This framework assumes:
- The loss can be computed independently for each sample $x_i$
- The total risk is the average of individual risks
- Gradients decompose across samples

The ERM formulation works well because:
- Enables stochastic gradient descent with mini-batches
- Each sample contributes independently to gradient estimates
- Convergence theory is well-documented
- Scales to massive datasets through sampling

### Frameworks Beyond Empirical Risk Formulations

Consider measuring classification problem where it is necessary to measure how well a classifier ranks samples from one class with respect to samples from the other classes. If the problem is reduced to two classes (one labeled the positive class and the other a negative class), then the metric is the AUC (Area Under the ROC Curve). 

$$
\text{AUC} = P(f(x^+) > f(x^-)) = \mathbb{E}_{x^+ \sim P^+, x^- \sim P^-}\left[\mathbb{1}(f(x^+) > f(x^-))\right]
$$

or, equivalently, 

$$
\text{AUC} = \frac{1}{n^+ n^-} \sum_{i \in I^+} \sum_{j \in I^-} \mathbb{1}(w^\top x_i > w^\top x_j)
$$

where the linear model class $f(x) = w^\top x$ usually dominates AUC optimization. As mentioned before, the ERM objective decomposes into a sum of independent per-sample terms. The AUC objective function depends on **pairs of samples** $(x^+, x^-)$ and its relative ordering. This dependency makes the loss depend on the dataset composition, which is not captured by ERM. 

AUC as a performance metric becomes necessary in several classification modeling scenarios: 

**Imbalanced Data** (1-5% positive class):
- Medical diagnosis, fraud detection, equipment failure prediction
- Accuracy is uninformative (99% by always predicting negative)

**Unknown Operating Point**:
- Threshold set by end users (doctors, analysts) with varying risk tolerance
- Business conditions change (cost of false positives varies over time)
- Need model that performs well across all possible thresholds

**Resource-Constrained Ranking**:
- Limited investigation capacity (fraud teams)
- Limited intervention budget (customer retention)
- Must prioritize: who to screen first, who to contact, what to inspect

In these settings, optimizing cross-entropy (which targets a fixed threshold) can produce models with high accuracy but poor ranking quality—missing the actual objective.

Similar to the structure of AUC metrics, pairwise ranking objectives can expand to other applications. The following problems also depend on the relationships between samples rather than independent evaluations. 

**Ranking metrics** (information retrieval):
- Precision@K: Only the top K predictions matter
- NDCG: Depends on the relative ordering of all items
- MAP: Averages precision at each relevant item's rank

**Contrastive learning** (representation learning):
- Pull together samples from the same class
- Push apart samples from different classes
- Defined over sets of positive/negative pairs, not individual samples

**Fairness constraints**:
- Equal opportunity: TPR should be equal across groups
- Equalized odds: Both TPR and FPR equal across groups
- These compare **subgroup statistics**, not individual predictions

---

## Adapting to Relational Objectives

### Introducing AUC as a Pairwise Ranking Objective

Yang (2023) introduced **X-Risk** to generalize ERM to objectives defined over sample relationships:

$$
\min_{w \in \mathcal{X}} \frac{1}{|S|} \sum_{z_i \in S} f_i\Big(g(w, z_i, S_i)\Big)
$$

which is composed by the full dataset $S$ and $S_i \subseteq S$ as the reference set of data points dependent or independent of the sample $z_i$. The mapping function $g(w, z_i, S_i)$ aggregates the comparisons between $z_i$ and samples in $S_i$ while $f_i(\cdot)$ is the risk measure applied to the aggregated result. X-Risk refers to a family of compositional measures in which the loss functions contrast one data point against the rest of the points in a defined set. 

For the binary classification setting with positive class $\mathcal{S}^+$ and negative class $\mathcal{S}^-$, the X-Risk becomes 

$$
\min_w \frac{1}{|\mathcal{S}^+|} \sum_{z_i \in \mathcal{S}^+} \left(1 - \frac{1}{|\mathcal{S}^-|}\sum_{z_j \in \mathcal{S}^-} \mathbb{1}(w^\top z_i > w^\top z_j)\right)
$$

where $S = \mathcal{S}^+ \cup \mathcal{S}^-$ (all samples) and, for each positive sample $z_i \in \mathcal{S}^+$, define $S_i = \mathcal{S}^-$ (all negative samples). The loss function becomes $g(w, z_i, S_i) = \frac{1}{|S_i|}\sum_{z_j \in S_i} \mathbb{1}(w^\top z_i > w^\top z_j)$ and $f_i(g) = 1 - g$. 

### Challenges of implementing AUC as a Pairwise Ranking Objective

- **Gradient Computation:** Because the X-Risk objectives involves sampling pairs (or sets), standard stochastic gradient descent (SGD) methods are not applicable. In other words, 

$$
\nabla \sum_{i,j} \ell(w, x_i, x_j) \neq \sum_i \nabla\ell(w, x_i).
$$


- **Quadratic Scaling:** Individual losses require $n$ evaluations, while pairwise losses require $n^2$ evaluations (or even $n^+ \times n^-$ for AUC objectives). There is a tradeoff between the computational complexity of pairwise ranking objectives and using industry standard losses (i.e. cross entropy). 


- **Non-Smoothness:** AUC uses the indicator function:

$
\mathbb{1}(w^\top x_i > w^\top x_j) = 
\begin{cases}
1 & \text{if } w^\top x_i > w^\top x_j \\
0 & \text{otherwise}
\end{cases}
$

which is both non-differentiable $w^\top x_i = w^\top x_j$ and non-continuous (small changes in $w$ cause discrete jumps).

### Two Approaches to Handling its Challenges

**Smooth Approximations:** (e.g., LibAUC, standard deep learning)
- Replace $\mathbb{1}(t)$ with smooth surrogates: $\sigma(\beta t)$ (sigmoid) or $\max(0, 1-t)$ (hinge)
- Makes the objective compatible with standard optimizers (Adam, SGD)
- **When it excels**: Deep learning with many layers, where end-to-end differentiability is crucial
- **Trade-off**: Smooth surrogates may not be perfectly calibrated to true AUC

**Our Approach: Piecewise Linear Surrogates:**
- Replace $\mathbb{1}(t)$ with $\ell_\delta(t) = \min(1, \max(0, t - \delta))$
- Still non-smooth at the breakpoints but piecewise linear
- Enables proximal operators with closed-form solutions that handle non-smoothness directly
- Fisher consistent: Converges to true AUC as new parameter $\delta \to 0$
- **When it excels**: Linear models, direct feature optimization, settings where theoretical guarantees matter
- - **Trade-off**: Requires parameter studies to assure algorithm convergence. 

Smooth approaches eliminate non-smoothness to fit standard methods. Piecewise linear approaches preserve the non-smooth structure and require specialized methods (proximal operators, ALM) designed to handle it directly. Both approaches are valid, since they optimize different trade-offs between computational convenience and theoretical precision.

---

## Our Approach: Proximal Methods for Non-Smooth AUC

Rather than adapting these problems to fit standard gradient-based frameworks, we develop methods specifically designed for **non-smooth pairwise ranking objectives in AUC applications**. 

### The Strategies

**1. Piecewise Linear Surrogate with Proximal Operators**

Replace the indicator $\mathbb{1}(t)$ with tractable surrogate:
$$
\ell_\delta(t) = \min(1, \max(0, t - \delta))
$$

- **Piecewise linear** (not smooth!), but enables closed-form proximal operators
- **Fisher consistent**: Converges to true indicator as $\delta \to 0$
- **Proximal operator** $\text{prox}_{\gamma \ell_\delta}(x)$ has **explicit solutions** depending on $\gamma = 1/\sigma$
  - Handles non-smoothness directly without smoothing
  - γ-dependent analysis reveals $\sigma = 1.0$ is optimal across problem geometries

**2. Augmented Lagrangian Decomposition**

Introduce auxiliary variables to separate the non-smooth objective from pairwise constraints:
$$
\min_{w,y} \sum_{i,j} \ell_\delta(y_{ij}) \quad \text{subject to} \quad y_{ij} = w^\top(z_j - z_i)
$$

- **ALM** handles constraints via adaptive penalty methods
- **Semi-smooth Newton (SSN)** exploits piecewise structure for efficient subproblem solving
- Proximal operators computed on $y_{ij}$, Newton updates on $w$

**3. Controlled Sampling and Sparsity Exploitation**

Reduce computational cost from $O(n^2)$ to $O(n \cdot k)$:
- **Controlled sampling**: Fix batch size (e.g., 50 pos × 50 neg = 2,500 pairs per iteration)
- **Sparsity exploitation**: Correctly classified pairs contribute zero gradient → skip updates
- **Effective complexity**: $O(n \cdot k)$ where $k \ll n$ is the number of active pairs per sample

### Connection to Other Pairwise Objectives

The piecewise linear surrogate and proximal operator framework is **problem-agnostic**—it handles the non-smoothness arising from any indicator-based pairwise objective. All these objectives evaluate **relationships between samples** rather than individual sample properties. Examples include the following. 

**Contrastive Learning**:
- Compare each sample to positive examples (same class) vs. negative examples (different classes)
- **Binary case**: Equivalent to AUC optimization when embeddings are 1-dimensional
- **Multi-class case**: Decompose via one-vs-rest or pairwise class comparisons

**Average Precision (AP)**:
- Ranking metric for information retrieval
- For each relevant document, measure precision at its rank position
- Optimization requires pairwise comparisons to establish ranking order

**Precision@K**:
- Only top-K predictions matter
- Requires ranking all samples, then selecting top-K
- Naturally formulated as pairwise ranking constraints

**Learning to Rank (LTR)**:
- NDCG, MRR, and other ranking metrics
- All depend on relative ordering of items
- Pairwise or listwise comparison structure

---

## Contributions

We provide rigorous optimization theory for the most widely deployed class of AUC models (**linear scoring functions**), with principled extensions to kernel methods. This complements recent deep learning approaches (LibAUC) which sacrifice theoretical guarantees for representation power.

Linear models dominate AUC applications due to ranking being fundemantally ordinal, i.e. AUC only cares whether $f(x^+) > f(x^-)$ in any magnitude. Additionally, *linear functions provide sufficient ranking capacity for many problems* and complex nonlinearities don't necessarily improve ranking quality proportionally to the computational cost. Many medical problems feature response variables that act as linear combinations of biomarker elevations, as well as fair lending laws where FICO scores are linearly dependent on credit factors. AUC acts as a key metric in linear models where teams need actionable rules that are easy to interpret provide flexibility. 

### What We Provide

**Theoretical Contributions:**
- Formulation of pairwise ranking objectives under the X-Risk framework
- Piecewise linear surrogate construction for indicator functions with Fisher consistency guarantees
- Proximal operator derivations with γ-dependent analysis 
- Convergence analysis for ALM + SSN framework applied to non-smooth pairwise objectives

**Algorithmic Contributions:**
- Augmented Lagrangian Method (ALM) with Semi-Smooth Newton (SSN) subsolvers for non-smooth AUC
- Controlled pairwise sampling strategies with sparsity exploitation
- Adaptive hyperparameter updates (σ, τ scheduling)
- Efficient implementation exploiting sparse structure of correctly classified pairs

**Empirical Validation:**
- **Synthetic SVM datasets**: Controlled geometry experiments
  - Low/high separation scenarios
  - Varying dimensionality: $m \gg n$ and $m \ll n$
  - Hyperparameter sensitivity: $\sigma \in \{0.1, 1.0, 2.0\}$
  - **Key finding**: Method excels in high-dimensional, low-sample regimes where baselines fail
  
- **CIFAR-10 binary classification**: Real-world validation
  - Balanced and imbalanced class distributions (1:1 and 1:9 ratios)
  - Comparison against BCE and LibAUC baselines
  - Competitive performance: 97.75% AUC (vs. 98.08% LibAUC baseline) on imbalanced data

---

### Current Scope and Extensions

**Current Implementation:**
- **Linear scoring functions**: $f(x) = w^\top x$
- Binary AUC optimization validated on synthetic and real datasets
- Controlled experimental settings (SVM, CIFAR-10 binary)

**Extensions:**
- **Kernel methods**: $f(x) = w^\top \phi(x)$ with nonlinear feature maps
  - Still linear in $w$, proximal methods apply directly
  - Maintains convexity and convergence guarantees
  - Can use kernel trick for computational efficiency
  
- **Engineered features**: Polynomial expansions, interaction terms, domain transformations
  - Transform $x \to \phi(x)$, then apply linear model
  - Framework unchanged, just operates in transformed space
  
- **Multi-class AUC**: One-vs-rest or pairwise class decompositions
  - Binary solver applied $C$ times for $C$ classes
  - Or solve $\binom{C}{2}$ pairwise problems
  
- **Other pairwise objectives**: Average Precision (AP), Precision@K, ranking metrics
  - Same pairwise comparison structure
  - Proximal operators adapt to different indicator-based losses

**Future Directions (require new analysis):**
- **Neural networks**: $f_\theta(x)$ with deep architectures
  - Introduces nonconvex subproblems (nonlinear in $\theta$)
  - Would need modified convergence analysis
  - Proximal operators still apply to output layer, but backpropagation complicates ALM framework
  
- **Large-scale deployment**: Integration with PyTorch/JAX as custom loss functions
- **Self-supervised contrastive learning**: Extension beyond supervised binary/multi-class settings
- **Real-world medical/financial datasets**: Application domains requiring interpretability

---

## Document Roadmap

This overview establishes **why** specialized methods are needed for pairwise objectives. The remaining documents cover:

2. **[Algorithm Formulation](2_algorithm.md)** - Mathematical details of the optimization problem
3. **[Optimization Methods](3_optimization.md)** - Proximal operators, ALM, SSN solver
4. **[Experiments](4_experiments.md)** - Empirical validation and results
5. **[Mathematical Appendix](5_math_appendix.md)** - Complete derivations and proofs
