# Project Overview

This project studies efficient optimization algorithms for contrastive
learning objectives that cannot be naturally expressed through standard
empirical risk minimization (ERM). In particular, we focus on objectives
defined over *pairs or sets of samples*, such as AUC, ranking metrics,
and supervised contrastive learning.

The goal of this repository is to bridge the gap between the theoretical
formulation of these objectives and practical, scalable optimization
methods that can be implemented efficiently in modern machine learning
pipelines.

---

## From Empirical Risk Minimization to X-Risk

Most machine learning algorithms are formulated under the empirical risk
minimization (ERM) framework, where a model is trained by minimizing the
average loss over individual data points:

$$
\min_{w \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n \ell(w, x_i).
$$

While ERM has been extensively studied and is well-supported by
stochastic gradient methods, it fails to capture a broad class of
objectives where performance depends on *comparisons between samples*
rather than independent evaluations.

Examples include:
- Area Under the ROC Curve (AUC)
- Ranking and information retrieval metrics
- Contrastive learning objectives

In these settings, the loss function depends on how a data point relates
to a *set* of other points, making gradients difficult or impossible to
compute directly.

To address this limitation, we adopt the **X-Risk** framework, where the
objective is defined over interactions between samples rather than
individual observations.

---

## X-Risk as a Unifying Framework

Under the X-Risk formulation, each data point is evaluated relative to a
set of relevant samples. The general objective takes the form:

$$
\min_{w \in \mathcal{X}} \frac{1}{|S|} \sum_{z_i \in S} f_i\big(g(w, z_i, S_i)\big),
$$

where:
- $S$ is the full dataset,
- $S_i$ is a subset of samples relevant to $z_i$,
- $g(\cdot)$ aggregates pairwise losses,
- $f_i(\cdot)$ defines the risk measure of interest.

This formulation naturally encompasses:
- **AUC optimization**, where positive samples are compared against
  negative samples,
- **Ranking metrics**, where relevance is defined relative to a query,
- **Contrastive learning**, where samples are pulled together or pushed
  apart in an embedding space.

By casting these objectives under a common framework, X-Risk enables the
development of general-purpose optimization algorithms that extend
beyond ERM.

---

## Connection to Contrastive Learning

Contrastive learning aims to learn representations by minimizing the
distance between similar samples while maximizing separation between
dissimilar ones. In supervised settings, this often involves comparing
samples across class boundaries.

When the embedding space is one-dimensional or ordered, contrastive
learning reduces to a *ranking problem*, where correct ordering of
samples becomes the primary objective. In the binary case, this
reduction leads directly to **AUC maximization**.

This observation motivates the approach taken in this project: we treat
contrastive learning as a special case of AUC-style pairwise
optimization, allowing us to leverage well-defined ranking objectives
and optimization techniques.

---

## Why Efficient Optimization Matters

A major challenge in contrastive and ranking-based objectives is their
computational cost. Naively evaluating all pairwise interactions leads
to quadratic complexity, making direct optimization infeasible for large
datasets.

Additionally, many objectives rely on indicator functions or
non-smooth losses, which:
- prevent direct application of gradient-based methods,
- introduce numerical instability,
- complicate convergence analysis.

This project addresses these challenges by:
- introducing continuous surrogate losses,
- exploiting sparsity in pairwise interactions,
- applying proximal and augmented Lagrangian methods,
- separating high-level experimentation (Python) from
  performance-critical optimization (Julia).

---

## Scope of This Repository

This repository presents a unified treatment of contrastive learning
and ranking-based objectives under the X-Risk framework, along with
optimization algorithms designed for non-smooth, pairwise loss
functions. It includes experimental analyses on synthetic datasets to
study convergence behavior and algorithmic stability, as well as
multi-language implementations that balance rapid experimentation with
high-performance optimization.

The accompanying documentation is structured to reflect this
progression—from motivation and problem formulation to optimization
methods and empirical behavior—while full mathematical derivations are
provided separately for reference.
