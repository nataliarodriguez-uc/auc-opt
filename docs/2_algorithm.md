# Algorithmic Formulation

This section presents the mathematical formulation underlying the
contrastive learning objectives studied in this repository. We derive
the learning objective from first principles, beginning with pairwise
comparisons and showing how contrastive learning can be expressed as a
ranking and AUC-style optimization problem.

The goal is to make explicit the assumptions, reductions, and modeling
choices that motivate the optimization algorithms implemented later.

---

## Pairwise Contrastive Objective

Consider a dataset of samples
$\{(z_i, y_i)\}_{i=1}^n$, where $z_i \in \mathbb{R}^d$ denotes a feature
representation (or embedding) and $y_i$ denotes a class label. In
contrastive learning, model performance is determined not by individual
predictions, but by *relative comparisons* between samples.

To formalize this, we define a scoring function
$h_w(z): \mathbb{R}^d \to \mathbb{R}$ parameterized by $w$. The objective
is to ensure that samples from the same class are assigned similar
scores, while samples from different classes are separated.

This naturally leads to a pairwise objective of the form

$$
\min_w \; \frac{1}{|S|} \sum_{z_i \in S}
\frac{1}{|S_i|} \sum_{z_j \in S_i}
\ell\big(h_w(z_j) - h_w(z_i)\big),
$$

where:
- $S$ is the set of reference samples,
- $S_i$ is a set of samples contrasted against $z_i$,
- $\ell(\cdot)$ is a loss function penalizing incorrect ordering.

This formulation emphasizes that learning occurs through *pairwise
differences*, rather than pointwise errors.

---

## Reduction to AUC in the Binary Case

To gain intuition, we first consider the binary classification setting,
where $y_i \in \{-1, +1\}$. Let $S^+$ denote the set of positive samples
and $S^-$ the set of negative samples.

In this case, the contrastive objective reduces to comparing every
positive sample against negative samples. Using the indicator loss
$\ell(t) = \mathbb{1}(t > 0)$, the objective becomes

$$
\min_w \;
\frac{1}{|S^+||S^-|}
\sum_{z_i \in S^+}
\sum_{z_j \in S^-}
\mathbb{1}\big(h_w(z_j) - h_w(z_i) > 0\big).
$$

Minimizing this quantity is equivalent to maximizing the probability
that positive samples are ranked above negative samples, which is
precisely the definition of the Area Under the ROC Curve (AUC):

$$
\text{AUC}(w) = \mathbb{P}\big(h_w(z^+) > h_w(z^-)\big).
$$

Thus, supervised contrastive learning in the binary setting can be
interpreted as an AUC maximization problem. This connection allows
ranking-based objectives to be treated within a unified mathematical
framework.

---

## Linear Model Assumption

To simplify the analysis and enable efficient optimization, we assume a
linear scoring function

$$
h_w(z) = w^\top z.
$$

Under this assumption, each pairwise comparison depends only on the
projection of the difference vector $(z_j - z_i)$ onto $w$. The loss can
be written as

$$
\ell\big(w^\top(z_j - z_i)\big).
$$

Geometrically, each pair $(z_i, z_j)$ defines a direction in feature
space, and the optimization objective encourages $w$ to orient the
decision boundary such that positive–negative differences are correctly
ordered.

While the linear assumption is restrictive, it isolates the core
optimization challenges inherent to contrastive objectives and serves
as a foundation for extending the approach to nonlinear embeddings.

---

## Surrogate Loss Approximation

Direct optimization with indicator losses is computationally intractable
due to non-differentiability. To address this, we replace the indicator
function with a continuous piecewise-linear surrogate loss.

Introducing auxiliary variables
$y_{ij} = w^\top(z_i - z_j)$, the objective is approximated as

$$
\min_{w, y} \;
\frac{1}{|S|}
\sum_{i \in S}
\frac{1}{|S_i|}
\sum_{j \in S_i}
\ell_{b,\delta}(y_{ij})
$$

subject to the consistency constraints

$$
y_{ij} = w^\top(z_i - z_j).
$$

Here, $\ell_{b,\delta}(\cdot)$ denotes a bounded piecewise-linear
approximation to the indicator loss, parameterized by a margin
$\delta$. This approximation preserves the ranking structure of the
original problem while enabling the use of proximal and constrained
optimization techniques.

---

## Contrastive Interpretation

From a contrastive learning perspective, each pair $(z_i, z_j)$ defines
a positive or negative interaction depending on their labels. The
surrogate loss penalizes violations of the desired ordering while
allowing correctly classified pairs to contribute sparse or zero
gradients.

This interpretation highlights a key advantage of the formulation:
only *informative pairs* drive updates, which can significantly reduce
computational cost when many pairs are already correctly ranked.

---

## Summary of the Learning Objective

In summary, the algorithmic formulation in this project proceeds as
follows:

1. Express contrastive learning as a pairwise ranking problem.
2. Reduce the binary case to AUC maximization.
3. Assume a linear scoring function to expose geometric structure.
4. Replace non-differentiable indicator losses with continuous
   surrogates.
5. Introduce auxiliary variables to enable constrained optimization.

This formulation lays the foundation for the optimization methods
described next, where proximal and augmented Lagrangian techniques are
used to efficiently solve the resulting problem.
