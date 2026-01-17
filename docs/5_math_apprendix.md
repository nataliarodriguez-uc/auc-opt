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

Readers interested in full implementation details are encouraged to
consult the source code alongside this appendix.
