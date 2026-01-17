# AUC-Optimized Contrastive Learning via Proximal Methods

This repository explores **contrastive learning and ranking objectives**
through **AUC optimization**, focusing on efficient methods for
objectives that cannot be expressed using standard empirical risk
minimization (ERM).

The project formulates contrastive learning as a **pairwise optimization
problem** and applies **proximal and augmented Lagrangian methods** to
handle non-smooth losses and large collections of pairwise comparisons.

---

## Motivation

Many practical ML objectives—such as AUC, ranking metrics, and
contrastive learning—depend on **comparisons between samples** rather
than pointwise losses. These objectives are difficult to optimize using
standard gradient-based methods due to non-differentiability and
quadratic scaling in the number of pairs.

This project addresses these challenges by combining:
- Pairwise surrogate losses
- Proximal optimization
- Augmented Lagrangian methods

---

## Key Ideas

- Reformulation of contrastive learning as **AUC maximization**
- Continuous surrogates for indicator-based losses
- Proximal and ALM-based optimization for constrained objectives
- Controlled pairwise sampling to reduce computational cost

---

## Documentation

The repository includes structured technical documentation covering the
full lifecycle of the project, from problem motivation to optimization
details and empirical evaluation:

- **[Project Overview](docs/1_overview.md)**  
  Background, motivation, and framing of contrastive learning as a
  ranking and AUC optimization problem.

- **[Algorithmic Formulation](docs/2_algorithm.md)**  
  Derivation of the pairwise contrastive objective, reduction to AUC in
  the binary case, and surrogate loss construction.

- **[Optimization Methodology](docs/3_optimization.md)**  
  Proximal updates, augmented Lagrangian formulation, and solver design
  used to handle non-smooth, constrained objectives.

- **[Experimental Evaluation](docs/4_experiments.md)**  
  Synthetic SVM-style experiments and contrastive learning experiments
  on CIFAR-10, analyzing convergence behavior, sampling strategies, and
  AUC performance.

- **[Mathematical Appendix](docs/5_math_appendix.md)**  
  Full derivations, proximal case analysis, and semi-smooth Newton
  details supporting the optimization methods.


---

## Experiments

The repository includes both synthetic and real-data experiments to
evaluate optimization behavior and ranking performance.

Synthetic SVM-style experiments are used to study convergence,
stability, and sensitivity to hyperparameters under controlled geometric
settings. In addition, contrastive learning experiments on subsets of
CIFAR-10 evaluate the method in a realistic, high-dimensional setting,
including both balanced and imbalanced class distributions.

Across experiments, results highlight the impact of proximal parameters,
pairwise sampling strategies, and constraint enforcement on AUC-based
performance.

---

## Status

This repository is under active development. Current work focuses on:
- Cleaning and integrating CIFAR-10 contrastive experiments
- Modularizing optimization routines
- Extending methods to more general contrastive settings

---

## Author

**Natalia A. Rodriguez Figueroa**  
Optimization · Machine Learning · Contrastive Learning
University of California, Berkeley - Department of Industrial Engineering and Operations Research
This work is supported by the advisory of Dr. Ying Cui. 

### References

- P. D. Khanh, B. S. Mordukhovich, and V. T. Phat.  
  *A generalized Newton method for subgradient systems*, 2022.  
  arXiv:2009.10551.

- X. Li, D. Sun, and K.-C. Toh.  
  *A highly efficient semismooth Newton augmented Lagrangian method for solving LASSO problems*.  
  **SIAM Journal on Optimization**, 28(1):433–458, 2018.

- L. Tian and A. M.-C. So.  
  *Computing d-stationary points of ρ-margin loss SVM*.  
  **Proceedings of AISTATS (PMLR)**, vol. 151, 2022.

- Y. Wang, W. Yin, and J. Zeng.  
  *Global convergence of ADMM in nonconvex nonsmooth optimization*, 2015.  
  arXiv [math.OC].

- T. Yang.  
  *Algorithmic foundations of empirical X-risk minimization*, 2023.



