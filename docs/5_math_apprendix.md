# Mathematical Appendix

This appendix contains detailed mathematical derivations and algorithmic
subcomponents that are omitted from the main documentation for clarity.
The material presented here supports the optimization methodology
described in the main text and corresponds directly to the implementation
in the Julia codebase.

---

## Indicator Loss Approximation

Many ranking and contrastive objectives rely on indicator loss functions
of the form

$$
\ell(t) = \mathbb{1}(t > 0),
$$

which are non-differentiable and unsuitable for gradient-based
optimization. To enable tractable optimization, we replace the indicator
function with a continuous piecewise-linear surrogate
$\ell_{b,\delta}(\cdot)$.

This surrogate preserves the ordering structure of the original loss
while introducing regions of zero or constant gradient, which induce
sparsity in updates.

---

## Constrained Reformulation

Introducing auxiliary variables
$y_{ij} = w^\top(z_i - z_j)$ allows the pairwise objective to be written
as

$$
\min_{w, y}
\frac{1}{|S|}
\sum_{i \in S}
\frac{1}{|S_i|}
\sum_{j \in S_i}
\ell_{b,\delta}(y_{ij})
\quad \text{subject to} \quad
y_{ij} = w^\top(z_i - z_j).
$$

This constrained formulation decouples the non-smooth loss from the
model parameters, enabling the use of proximal and augmented Lagrangian
methods.

---

## Augmented Lagrangian Formulation

The augmented Lagrangian associated with the constrained problem is

$$
\mathcal{L}_\sigma(w, y, \lambda)
=
\frac{1}{|S|}
\sum_{i \in S}
\frac{1}{|S_i|}
\sum_{j \in S_i}
\left(
\ell_{b,\delta}(y_{ij})
+ \lambda_{ij}(y_{ij} - w^\top(z_i - z_j))
+ \frac{\sigma}{2}
\|y_{ij} - w^\top(z_i - z_j)\|^2
\right)
+ \tau \|w\|_2^2.
$$

The quadratic penalty term enforces constraint feasibility, while the
dual variables $\lambda_{ij}$ adaptively correct violations.

---

## Proximal Mapping of the Surrogate Loss

The update for each auxiliary variable $y_{ij}$ requires solving the
proximal subproblem

$$
\operatorname{prox}_{\gamma \ell_{b,\delta}}(x)
=
\arg\min_{y}
\left\{
\ell_{b,\delta}(y)
+
\frac{1}{2\gamma}\|y - x\|^2
\right\},
$$

where $x = w^\top(z_i - z_j) - \lambda_{ij}/\sigma$ and
$\gamma = \sigma^{-1}$.

Because $\ell_{b,\delta}$ is piecewise linear, the proximal operator
admits a closed-form solution whose structure depends on the value of
$\gamma$ relative to the breakpoints induced by $\delta$.

---

## Case Analysis of the Proximal Operator

The proximal mapping exhibits different regimes depending on the value
of $\gamma$:

### Case 1: $\gamma < 2$

For $\gamma < 2$, the proximal mapping produces a unique solution, and
the derivative is well-defined almost everywhere. In this regime, the
mapping exhibits aggressive shrinkage, leading to sparse gradient
updates.

### Case 2: $\gamma = 2$

At $\gamma = 2$, the proximal operator admits multiple solutions at
boundary points. These correspond to subdifferential regions where the
gradient is set-valued.

### Case 3: $\gamma > 2$

For $\gamma > 2$, the zero-gradient region expands, and the proximal
mapping becomes increasingly flat. This behavior can slow convergence
but provides numerical stability.

These regimes directly influence optimization dynamics and motivate
adaptive scheduling of $\sigma$.

---

## Derivative of the Proximal Mapping

The derivative (or generalized Jacobian) of the proximal operator plays
a critical role in second-order optimization methods such as
semi-smooth Newton (SSN).

For $\gamma < 2$, the proximal mapping is differentiable almost
everywhere, with derivative equal to the identity outside flat regions.
For $\gamma \ge 2$, the mapping admits a set-valued derivative
corresponding to the subdifferential.

This semi-smooth structure justifies the use of SSN methods for solving
the $w$-subproblem efficiently.

---

## Semi-Smooth Newton Subproblem

With $y$ fixed, the $w$-update involves minimizing a smooth objective of
the form

$$
\sum_{i,j}
\|y_{ij} - w^\top(z_i - z_j) + \lambda_{ij}/\sigma\|^2
+ \tau \|w\|_2^2.
$$

The semi-smooth Newton method computes a search direction by solving a
linearized system involving the generalized Hessian of the objective.
Line search is applied to ensure sufficient descent.

---

## Computational Considerations

The proximal and SSN updates must be computed repeatedly over large
collections of pairwise differences. Exploiting sparsity induced by the
surrogate loss is critical for scalability.

These considerations motivate:
- Pairwise subsampling strategies,
- Block-coordinate updates,
- High-performance implementations in Julia.

---

**Supervised Contrastive Learning:**

For multi-class classification where each sample has a class label:

- **Dataset**: $S = \{(z_1, y_1), \ldots, (z_n, y_n)\}$ (all labeled samples)
- **Positive set**: For sample $z_i$ with label $y_i$, define $S_i^+ = \{z_j \in S : y_j = y_i, j \neq i\}$ (same class)
- **Negative set**: Define $S_i^- = \{z_j \in S : y_j \neq y_i\}$ (different classes)
- **Aggregation**: $g(w, z_i, S_i^+, S_i^-) = -\log \frac{\sum_{z_j \in S_i^+} \exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{z_k \in S_i^+ \cup S_i^-} \exp(\text{sim}(z_i, z_k)/\tau)}$
- **Risk measure**: $f_i(g) = g$ (minimize negative log-likelihood of correct grouping)

where $\text{sim}(z_i, z_j) = w^\top(z_i \cdot z_j)$ or other similarity measures.

---

**Ranking (Mean Average Precision):**

For information retrieval with query $q$ and document set $\mathcal{D}$:

- **Dataset**: $S = \{(d_1, r_1), \ldots, (d_n, r_n)\}$ (documents with relevance scores for query $q$)
- **Relevant set**: Define $\mathcal{R} = \{d_i : r_i > 0\}$ (relevant documents)
- **Comparison set**: For each relevant document $d_i \in \mathcal{R}$, define $S_i = \mathcal{D}$ (all documents)
- **Aggregation**: $g(w, d_i, S_i) = \text{rank}(d_i | w) = |\{d_j \in S_i : w^\top d_i < w^\top d_j\}|$ (position in ranking)
- **Risk measure**: $f_i(g) = \frac{|\{d_j \in \mathcal{R} : \text{rank}(d_j | w) \leq g\}|}{g}$ (precision at position $g$)

The X-Risk objective:
$$
\min_w -\frac{1}{|\mathcal{R}|} \sum_{d_i \in \mathcal{R}} \text{Precision}@\text{rank}(d_i | w)
$$

---

**Common Structure:**

In all three cases, the loss for sample $z_i$ is defined by:
1. Identifying a **comparison set** $S_i$ of related samples
2. **Aggregating** information across these comparisons: $g(w, z_i, S_i)$
3. Applying a **risk measure** $f_i(\cdot)$ to the aggregated result

The optimization objective has the unified form:
$$
\min_{w \in \mathcal{X}} \frac{1}{|S|} \sum_{z_i \in S} f_i\Big(g(w, z_i, S_i)\Big)
$$

This structure is **fundamentally different from ERM**, which would require:
$$
\min_{w \in \mathcal{X}} \frac{1}{|S|} \sum_{z_i \in S} \ell(w, z_i)
$$

where each sample's loss is computed **independently**.

---
