# Pairwise Proximal Methods for Direct AUC Optimization

Efficient optimization algorithms for direct AUC maximization using piecewise linear approximations.

## The Problem

Most deep learning optimizes cross-entropy loss, but this doesn't align with what we actually care about in many real applications.

**Why AUC matters:**

In imbalanced classification, accuracy is misleading. A cancer screening model that labels everyone "healthy" achieves 99% accuracy if only 1% of patients have cancer—but it's useless. What matters is **ranking**: can the model reliably score true positives higher than false positives?

AUC (Area Under the ROC Curve) directly measures this ranking quality. It's the probability that a randomly chosen positive example scores higher than a randomly chosen negative example. This is exactly what we want in:

- **Medical diagnosis**: Rank high-risk patients above low-risk ones, regardless of threshold
- **Fraud detection**: Flag suspicious transactions at the top of the review queue
- **Credit scoring**: Order loan applicants by default risk for manual review
- **Information retrieval**: Return relevant documents before irrelevant ones

**The optimization challenge:**

True AUC requires comparing all positive-negative pairs:
$$\text{AUC} = \frac{1}{n_+ n_-} \sum_{i: y_i=1} \sum_{j: y_j=0} \mathbb{1}(f(x_i) > f(x_j))$$

This has two problems:
1. $O(n^2)$ pairwise comparisons (computationally expensive)
2. Non-differentiable indicator function $\mathbb{1}(\cdot)$ (can't use gradient descent)

## Our Approach

We develop proximal optimization methods that:
1. Replace the indicator function with a Fisher-consistent piecewise linear surrogate
2. Derive explicit proximal operators with $\gamma$-dependent formulas
3. Use Augmented Lagrangian Method (ALM) + Semi-Smooth Newton (SSN) for efficient solving
4. Reduce complexity from $O(n^2)$ to $O(n \cdot k)$ through controlled sampling

**Scope**: Linear models $f(x) = w^\top x$, which are interpretable, theoretically tractable, and widely deployed in medical/financial applications. Framework extends naturally to kernel methods.

**Key finding**: $\sigma = 1.0$ provides robust performance across different problem geometries.

## Results

### CIFAR-10 Binary Classification

| Dataset | Config | Prox AUC | LibAUC | Gap |
|---------|--------|----------|--------|-----|
| Balanced | σ=1.0, 25 pairs/batch | **95.27%** | 97.56% | -2.29% |
| Imbalanced (1:9) | σ=1.0, 25 pairs/batch | **97.75%** | 98.08% | -0.33% |

### Synthetic SVM Experiments

| Scenario | Dimensions | Prox | BCE | LibAUC |
|----------|-----------|------|-----|--------|
| Low sep, many samples | 1000×50 | **99.12%** | 99.88% | 99.30% |
| High sep, few samples | 50×500 | **100%** | 100% | 0%* |
| High sep, many samples | 1000×50 | **99.94%** | 100% | 100% |

*LibAUC fails completely in high-dimensional, low-sample regimes where our method achieves perfect AUC.

## Quick Start

```bash
git clone https://github.com/nataliarodriguez-uc/auc-opt.git
cd auc-opt
pip install -r requirements.txt
```

See `demos/` for working examples.

## Repository Structure

```
auc-opt/
├── demos/          # Jupyter notebooks and examples
├── docs/           # Technical documentation
├── src/
│   ├── julia/      # Julia implementation
│   └── python/     # Python implementation
│       └── aucopt/ # Main package
└── requirements.txt
```

## Documentation

- [Overview](docs/1_overview.md) - Problem motivation
- [Algorithm](docs/2_algorithm.md) - Mathematical formulation  
- [Optimization](docs/3_optimization.md) - Proximal operators and ALM details
- [Experiments](docs/4_experiments.md) - Full results and analysis
- [Math Appendix](docs/5_math_appendix.md) - Derivations and proofs

## Applications

Works for any pairwise ranking objective:
- Medical diagnosis (imbalanced datasets)
- Fraud detection (rare positives)
- Credit scoring (interpretable models required)
- Average Precision, Precision@K
- Binary contrastive learning

Not yet addressed: deep neural networks, large-scale pretraining, PyTorch/TensorFlow integration.

## Status

**Done**: Core algorithm, synthetic validation, CIFAR-10 experiments, documentation

**In progress**: Multi-class AUC, advanced sampling strategies, hyperparameter analysis

**Future**: Kernel extensions, deep network integration, production deployment

## Citation

```bibtex
@misc{rodriguez2026aucopt,
  author = {Rodriguez Figueroa, Natalia A.},
  title = {Pairwise Proximal Methods for Direct AUC Optimization},
  year = {2026},
  howpublished = {\url{https://github.com/nataliarodriguez-uc/auc-opt}},
  note = {Python package for direct AUC optimization}
}
```

## Author

**Natalia A. Rodriguez Figueroa**  
PhD Student, IEOR, UC Berkeley  
Advisor: Dr. Ying Cui  
📧 natalia_rodriguezuc@berkeley.edu