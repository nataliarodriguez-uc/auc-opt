# AUC-Optimized Contrastive Learning via Proximal Methods

> Rethinking contrastive learning as an AUC optimization problem, solved with proximal and augmented Lagrangian methods.

---

## About the Project

Many machine learning objectives—like AUC, ranking metrics, and contrastive learning—fundamentally depend on **pairwise comparisons** rather empirical loss function. These objectives are challenging to optimize because they involve:

- Non-differentiable indicator functions
- Quadratic scaling in pairwise comparisons  
- Cannot be expressed as standard empirical risk minimization (ERM)

This project introduces a **proximal optimization framework** for AUC-based contrastive learning that:
- Replaces indicator functions with tractable piecewise linear surrogates
- Uses augmented Lagrangian methods (ALM) with semi-smooth Newton subsolvers
- Achieves competitive performance with controlled pairwise sampling strategies

**Key Result**: Achieved **97.75% AUC** on imbalanced CIFAR-10 binary classification (vs. 98.08% baseline) with 80% less features than LibAUC Deep Learning Library. 

---

## Quick Start

### Installation

```bash
git clone https://github.com/nataliarodriguez-uc/auc-opt.git
cd auc-opt
pip install -r requirements.txt
```

### Run Demo



---

## Key Results

### CIFAR-10 Binary Classification

| Dataset | Configuration | Prox AUC | LibAUC Baseline |
|---------|--------------|----------|-----------------|
| Balanced | σ=1.0, 25 pairs, 10 batches | **95.27%** | 97.56% |
| Imbalanced (1:9) | σ=1.0, 25 pairs, 10 batches | **97.75%** | 98.08% |

### Synthetic SVM Experiments

| Scenario | Dimensions | Prox AUC | BCE | LibAUC |
|----------|-----------|----------|-----|--------|
| Low separation, m≫n | 1000×50 | 99.12% | 99.88% | 99.30% |
| High separation, m≪n | 50×500 | **100%** | 100% | 0%* |
| High separation, m≫n | 1000×50 | 99.94% | 100% | 100% |

*LibAUC fails completely in high-dimensional settings

**Finding**: σ=1.0 provides stable performance across scenarios; method excels when features >> samples.

---

## Repository Structure

```
auc-opt/
├── demos/                      # Demo notebooks and examples
│   └── svm_example.ipynb       # SMV example notebook
├── docs/                       # Detailed documentation
│   ├── 1_overview.md           # Problem motivation & X-risk background
│   ├── 2_algorithm.md          # Mathematical formulation
│   ├── 3_optimization.md       # Proximal methods & ALM details
│   ├── 4_experiments.md        # Experimental design & analysis
│   └── 5_math_appendix.md      # Derivations & proofs
├── src/                        # Core implementation
│   ├── julia/                  # Julia optimization routines
│   └── python/                 # Python implementation
│       └── aucopt/             # Main package
│           ├── __pycache__/
│           ├── data/           # Data loading utilities
│           ├── eval/           # Evaluation metrics
│           ├── optim/          # Optimization algorithms (ALM, SSN)
│           └── __init__.py
└── requirements.txt
```

---

## Documentation

**New to the project?** Start here:
- **[Overview](docs/1_overview.md)** - Problem motivation and background on X-risk minimization
- **[Algorithm](docs/2_algorithm.md)** - How contrastive learning reduces to AUC optimization

**Implementation details:**
- **[Optimization Methods](docs/3_optimization.md)** - Proximal operators, ALM, and SSN solver
- **[Experiments](docs/4_experiments.md)** - Full experimental setup and results
- **[Math Appendix](docs/5_math_appendix.md)** - Complete derivations and proofs

---

## Methodology Highlights

### Proximal Framework
- **Surrogate Loss**: Piecewise linear approximation ℓ_δ(t) of indicator function
- **Closed-form Proximal Operators**: γ-dependent solutions (γ = 1/σ)
- **ALM Decomposition**: Splits problem into tractable subproblems

### Why This Approach?
- Direct AUC optimization (not a proxy loss)
- Handles non-smooth objectives efficiently
- Extends to ranking, precision@K, and other pairwise metrics
- Computationally efficient with controlled sampling

See [docs/3_optimization.md](docs/3_optimization.md) for mathematical details.

---

## Current Status

**Completed:**
-  Proximal operator implementation with ALM + SSN framework
-  Synthetic SVM validation experiments
-  CIFAR-10 binary classification experiments

**In Progress:**
- Uploading complete experiment structure
- Advanced sampling strategies (stochastic descent via batching)
- Integration with modern architectures

**Future Work:**
- Self-supervised learning formulation
- Healthcare dataset applications
- Julia coding demo files

---

## References

Key theoretical foundations:
- Yang, T. (2023). *Algorithmic foundations of empirical X-risk minimization*
- Khanh et al. (2022). *A generalized Newton method for subgradient systems*
- Li, Sun & Toh (2018). *Semismooth Newton augmented Lagrangian method for LASSO*
- Tian & So (2022). *Computing d-stationary points of ρ-margin loss SVM*

---

## Author

**Natalia A. Rodriguez Figueroa**  
PhD Student, Industrial Engineering & Operations Research  
University of California, Berkeley  
Advisor: Dr. Ying Cui

[GitHub Profile](https://github.com/nataliarodriguez-uc)
Contact Information: natalia_rodriguezuc@berkeley.edu




