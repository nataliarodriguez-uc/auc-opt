# Efficient Proximal Optimization for Non-Smooth AUC Maximization

> This project develops efficient training optimization algorithms for direct AUC maximization using proximal gradients and augmented Lagrangian methods.

---

## Overview

### The Problem

In binary classification, **Area Under the ROC Curve (AUC)** is often the metric that matters especially in imbalanced settings like medical diagnosis, fraud detection, and information retrieval. However, most deep learning frameworks optimize cross-entropy loss (empirical loss functions), which doesn't directly correspond to AUC performance.

**Why standard methods struggle with AUC:**
- AUC measures ranking quality: all positive samples should score higher than negative samples
- This requires evaluating **pairwise comparisons** between classes, not individual predictions
- The true AUC objective uses non-differentiable indicator functions: $\mathbb{1}(\text{score}_{\text{pos}} > \text{score}_{\text{neg}})$
- Naively computing all pairs leads to $O(n^2)$ complexity

### Our Approach

This project develops **proximal optimization methods** specifically designed for direct AUC maximization. We introduce:

1. **Piecewise linear surrogate loss** that approximates the indicator function while remaining tractable
2. **Closed-form proximal operators** with γ-dependent solutions that exploit problem structure for computation efficiency
3. **Augmented Lagrangian Method (ALM)** + **Semi-Smooth Newton (SSN)** framework for efficient solving
4. **Controlled pairwise sampling** that reduces complexity from $O(n^2)$ to $O(n \cdot k)$ where k is a subset of pairs per sample used

**Key innovation**: Our methodology and analysis reveals that proximal operator parameters are proportional to the Augmented Lagrangian parameters ($\sigma = 1 / \gamma$). (Need more here...)

### Connection to Contrastive Learning


**Binary Contrastive Learning** (current implementation):
- Pull similar samples (same class) together, push dissimilar samples (different classes) apart
- When embeddings are 1-dimensional, this reduces to AUC optimization
- **Validated** on CIFAR-10 binary classification

**Multi-Class Extension** (theoretical framework):
- **One-vs-Rest**: Apply binary AUC optimization for each class vs. all others
- **Pairwise Decomposition**: Compare all class pairs $(C_i, C_j)$ separately
- **Supervised Contrastive Loss**: Same pairwise structure—positive pairs (same class) vs. negative pairs (different classes)

Our formulation **naturally handles multi-class** settings since:
1. The pairwise difference $w^\top(z_j - z_i)$ works for any pair of classes
2. The proximal operators don't depend on number of classes
3. The ALM framework scales to multiple comparison sets $S_i$

---

## Key Results

### CIFAR-10 Binary Classification

**Setup**: Binary classification on CIFAR-10 (2 classes), comparing Prox-SGD against standard baselines.

| Dataset | Configuration | Prox AUC | LibAUC Baseline | Gap |
|---------|--------------|----------|-----------------|-----|
| Balanced | σ=1.0, 25 pairs, 10 batches | **95.27%** | 97.56% | -2.29% |
| Imbalanced (1:9) | σ=1.0, 25 pairs, 10 batches | **97.75%** | 98.08% | -0.33% |

**Observations**:
- Method performs competitively on imbalanced data (within 0.33% of state-of-the-art)
- Controlled sampling (50 pos + 50 neg pairs per batch) enables efficient computation
- Room for improvement through hyperparameter tuning and sampling strategies

### Synthetic SVM Experiments

**Setup**: Controlled geometric configurations to isolate algorithm behavior.

| Scenario | Dimensions (m×n) | Prox AUC | BCE AUC | LibAUC AUC |
|----------|-----------------|----------|---------|------------|
| Low separation, many samples | 1000×50 | **99.12%** | 99.88% | 99.30% |
| High separation, few samples | 50×500 | **100%** | 100% | 0%* |
| High separation, many samples | 1000×50 | **99.94%** | 100% | 100% |

*LibAUC completely fails in high-dimensional, low-sample regime where our method achieves perfect AUC.

**Key Finding**: $\sigma = 1.0$ provides robust performance across all scenarios. Method excels when features >> samples, a challenging regime for standard approaches.

---

## Quick Start

### Installation

```bash
git clone https://github.com/nataliarodriguez-uc/auc-opt.git
cd auc-opt
pip install -r requirements.txt
```

### Run Demo

tbd 

### Basic Usage

tbd

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
│           ├── data/           # Data loading utilities
│           ├── eval/           # Evaluation metrics
│           ├── optim/          # Optimization algorithms (ALM, SSN)
│           └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Documentation

**Methodology Details** 
- **[Overview](docs/1_overview.md)** - Problem motivation and the X-risk framework
- **[Algorithm](docs/2_algorithm.md)** - From pairwise objectives to AUC formulation

**Implementation details:**
- **[Optimization Methods](docs/3_optimization.md)** - Proximal operators, ALM, and SSN solver details
- **[Experiments](docs/4_experiments.md)** - Full experimental setup, results, and analysis
- **[Math Appendix](docs/5_math_appendix.md)** - Complete derivations and proofs

---

## Optimization Highlights

### Proximal Optimization Framework

**Surrogate Loss Construction**:
- Replace non-differentiable indicator $\mathbb{1}(t > 0)$ with piecewise linear $\ell_\delta(t) = \min(1, \max(0, t - \delta))$
- Retains theoretical properties (Fisher consistency) while enabling efficient computation

**γ-Dependent Proximal Operators**:
- Closed-form solutions for proximal mapping: $\text{prox}_{\gamma \ell_\delta}(x)$
- Three regimes based on $\gamma = 1/\sigma$:
  - $\gamma < 2$: Sharp transitions, stable convergence
  - $\gamma = 2$: Subdifferential at boundary
  - $\gamma > 2$: Wider non-differentiable region
- Empirical finding: $\sigma = 1.0$ ($\gamma = 1.0$) optimal across problem types

**Augmented Lagrangian Decomposition**:
- Introduce auxiliary variables $y_{ij} = w^\top(z_j - z_i)$ for each pair
- ALM framework: alternate between primal (SSN) and dual (Lagrange multiplier) updates
- Exploits sparsity: only update pairs in active regions (not correctly classified)

### Computational Efficiency

Rather than evaluating all $O(n^2)$ pairs:
1. **Controlled sampling**: Fix batch size (e.g., 25 pos × 25 neg = 625 pairs)
2. **Sparse structure**: Many pairs are correctly classified → zero gradient → skip updates
3. **Block updates**: Only recompute active pairs each iteration

**Result**: Effective complexity $O(n \cdot k)$ where $k \ll n$.

See [docs/3_optimization.md](docs/3_optimization.md) for complete mathematical details.

---

## Applications

This methodology applies to:

**AUC-Based Classification** (current validation):
- Medical diagnosis: Binary or multi-class with imbalanced classes
- Fraud detection: Rare positive class vs. normal transactions  
- Information retrieval: Binary or graded relevance judgments
- Any classification problem where AUC is the target metric

**Contrastive Learning** (framework supports, validation in progress):
- Binary contrastive learning: Validated on CIFAR-10
- Multi-class contrastive learning: Pairwise formulation naturally extends
- Supervised contrastive loss: Same pairwise comparison structure

**Ranking Objectives**:
- Learning to rank with pairwise preferences
- Precision@K optimization
- Multi-class AUC via one-vs-rest or pairwise decomposition

**TBD...**:
- Large-scale self-supervised pretraining (SimCLR, MoCo scale)
- High-dimensional embeddings beyond linear projections
- Production deployment as PyTorch/TensorFlow loss function

---

## Current Status

**Completed:**
- Proximal operator derivation and implementation (γ-dependent cases)
- ALM + SSN optimization framework
- Synthetic SVM validation experiments
- CIFAR-10 binary classification experiments
- Comprehensive technical documentation

**In Progress:**
- Advanced sampling strategies (hard negative mining, curriculum sampling)
- Hyperparameter sensitivity analysis
- Extension to larger-scale experiments

**Future Directions:**
- Multi-class AUC via one-vs-rest
- Integration with PyTorch as a custom loss function
- Application to medical imaging datasets
- Self-supervised contrastive learning formulation

---

## Citation

```bibtex
@software{rodriguez2026aucopt,
  author = {Rodriguez Figueroa, Natalia A.},
  title = {Proximal Methods for AUC Optimization},
  year = {2026},
  url = {https://github.com/nataliarodriguez-uc/auc-opt}
}
```

---

## References

**Theoretical Foundations:**
- Yang, T. (2023). *Algorithmic foundations of empirical X-risk minimization*
- Khanh, P. D., Mordukhovich, B. S., & Phat, V. T. (2022). *A generalized Newton method for subgradient systems*
- Li, X., Sun, D., & Toh, K.-C. (2018). *A highly efficient semismooth Newton augmented Lagrangian method for solving LASSO problems*
- Tian, L., & So, A. M.-C. (2022). *Computing d-stationary points of ρ-margin loss SVM*
  
---

## Authors

**Natalia A. Rodriguez Figueroa**  
Industrial Engineering & Operations Research  
University of California, Berkeley  
Advisor: Dr. Ying Cui

📧 Email: natalia_rodriguezuc@berkeley.edu  
🔗 [GitHub Profile](https://github.com/nataliarodriguez-uc)