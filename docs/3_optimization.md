# Optimization Methodology

This section describes the optimization strategies used to solve the
constrained contrastive learning problem introduced in the algorithmic
formulation. The objective involves non-smooth surrogate losses and
pairwise constraints, which makes direct gradient-based optimization
ineffective.

To address these challenges, we adopt a combination of **proximal
methods** and the **Augmented Lagrangian Method (ALM)**, enabling stable
and efficient optimization of large collections of pairwise
interactions.

---

## From Constrained Objective to Augmented Lagrangian

Recall the constrained surrogate formulation:

$$
\min_{w, y} \;
\frac{1}{|S|}
\sum_{i \in S}
\frac{1}{|S_i|}
\sum_{j \in S_i}
\ell_{b,\delta}(y_{ij})
\quad \text{subject to} \quad
y_{ij} = w^\top(z_i - z_j).
$$

The equality constraints couple the auxiliary variables $y_{ij}$ to the
model parameters $w$. Enforcing these constraints directly is
computationally expensive due to the large number of pairwise terms.

To decouple the objective while maintaining constraint feasibility, we
introduce an augmented Lagrangian formulation.

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
+ \tau \|w\|_2^2,
$$

where:
- $\lambda_{ij}$ are Lagrange multipliers,
- $\sigma > 0$ is a penalty parameter,
- $\tau > 0$ controls $\ell_2$ regularization of $w$.

The quadratic penalty term stabilizes optimization by discouraging
constraint violations, while the dual variables adaptively enforce
consistency between $y$ and $w$.

---

## Decoupling via Proximal Updates

A key advantage of the augmented Lagrangian formulation is that it
separates the optimization over $y$ and $w$.

For fixed $w$ and $\lambda$, each auxiliary variable $y_{ij}$ is updated
by solving a **proximal subproblem** of the form:

$$
y_{ij}^{\star}
=
\arg\min_{y}
\left\{
\ell_{b,\delta}(y)
+
\frac{1}{2\gamma}
\|y - (w^\top(z_i - z_j) - \lambda_{ij}/\sigma)\|^2
\right\},
$$

where $\gamma = \sigma^{-1}$.

Because $\ell_{b,\delta}$ is piecewise linear, this proximal mapping
admits a closed-form solution that can be computed efficiently and
independently for each pair $(i,j)$. This step handles the non-smoothness
of the surrogate loss and induces sparsity in gradient contributions.

---

## Optimization Over Model Parameters

With $y$ fixed, optimization over $w$ reduces to minimizing a smooth
quadratic objective:

$$
\min_w \;
\sum_{i,j}
\|y_{ij} - w^\top(z_i - z_j) + \lambda_{ij}/\sigma\|^2
+ \tau \|w\|_2^2.
$$

This structure allows the use of second-order methods. In particular,
we employ a **semi-smooth Newton (SSN)** approach to efficiently solve
the resulting system, leveraging the structure induced by the proximal
mapping.

Line search and regularization are used to ensure numerical stability
and convergence.

---

## Alternating Optimization Scheme

The overall algorithm follows an alternating minimization strategy:

1. **Primal update**:  
   Solve for $(w^{t+1}, y^{t+1})$ by minimizing
   $\mathcal{L}_\sigma(w, y, \lambda^t)$.
2. **Dual update**:  
   Update Lagrange multipliers
   $$
   \lambda_{ij}^{t+1}
   =
   \lambda_{ij}^t
   + \sigma (y_{ij}^{t+1} - w^{t+1\top}(z_i - z_j)).
   $$
3. **Parameter update**:  
   Adjust penalty parameters $(\sigma, \tau)$ as needed.

This process is repeated until convergence criteria on primal and dual
residuals are satisfied.

---

## Role of the Penalty Parameter

The penalty parameter $\sigma$ plays a critical role in optimization.
Since $\gamma = \sigma^{-1}$, changing $\sigma$ alters the shape of the
proximal mapping and the size of the zero-gradient regions induced by
the surrogate loss.

- Small $\sigma$ (large $\gamma$) increases flexibility but may slow
  convergence.
- Large $\sigma$ enforces constraints more aggressively but risks
  stagnation.

This tradeoff motivates adaptive scheduling strategies for $\sigma$,
which are explored empirically.

---

## Implementation Considerations

The optimization procedure involves repeated evaluation of pairwise
differences and proximal updates, which can be computationally
intensive.

To balance flexibility and performance:
- Python is used for data handling, experimentation, and evaluation.
- Julia is used for implementing the proximal mappings, ALM updates,
  and Newton steps, where numerical performance is critical.
- Note: Additional .py files have been added to support notebooks running on only python languages.

This separation allows rapid experimentation without sacrificing
optimization efficiency.

---

## Summary

The optimization strategy in this project combines proximal methods and
augmented Lagrangian techniques to address the non-smooth, constrained
nature of contrastive learning objectives. By decoupling pairwise
variables and exploiting closed-form proximal updates, the method
achieves scalable and stable optimization while preserving the
structure of the original learning problem.
