# AUC Optimization via Scenario Reduction and Contrastive Sampling

## Overview
This project investigates **AUC optimization** using a **Proximal Stochastic Gradient Descent (Prox-SGD)** method based on **Augmented Lagrangian Methods (ALM)** and Proximal Algorithm execution on a surrogate model that approximates the indicator function in AUC metrics. We compare this custom approach against baseline optimizers including PyTorch BCE and **LibAUC PESG**, with a focus on **sample efficiency** and **generalization performance**.

Key highlights:
- Uses **disjoint positive-negative pairs** in SGD updates.
- Evaluates performance across **synthetic** and **CIFAR-10-based** datasets.
- Provides automatic **AUC evaluation**, **ROC plotting**, and **comparison exports**.

## Core Concepts
- **Pairwise ranking loss**: Transforms AUC into an optimization-friendly form.
- **Scenario reduction**: Efficient learning using a subset of meaningful pairs.
- **Augmented Lagrangian Method**: Solves constrained pairwise optimization problems.
- **Sample efficiency**: Competes with full-data methods using fewer training points.

## Technologies
- Python 3.9+
- Proximal Algorithm, Augmented Lagrangian Method, Semi-smooth Newton Descent
- PyTorch (for Prox-SGD and LibAUC baselines)
- torchvision (for feature extraction on CIFAR-10)
- numpy, matplotlib, scikit-learn (for evaluation and plotting)
- LibAUC (https://github.com/Optimization-AI/LibAUC)
