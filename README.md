# XGBoost — From Scratch

A complete, research-grade implementation of **XGBoost from scratch** in pure Python/NumPy, extended with **in-place incremental tree updates** for adapting to concept drift in streaming data — without retraining from scratch.

---

## Overview

Standard XGBoost retrains all trees from scratch when new data arrives. This repository implements two components:

1. **`XGBoostModel`** — A full from-scratch XGBoost implementation: histogram-binned split finding, sparsity-aware splits with learned default directions, second-order gradient boosting, and early stopping.

2. **`IncrementalXGBoostClassifier`** — Extends the base model with in-place tree adaptation. When new data arrives (potentially with distribution shift), each existing tree is updated by routing new samples, rechecking split stability, and **selectively retraining only the subtrees that have drifted** — leaving stable branches unchanged.

---

## Key Algorithms

### 1 — Regularized Objective & Leaf Value

XGBoost minimises the regularized loss at each boosting round. The optimal leaf weight for a node with gradient sum $G$ and hessian sum $H$ is:

$$w^* = -\frac{G}{H + \lambda}$$

The improvement in objective from splitting a node into left and right children is given by:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right]$$

A split is only accepted if $\text{Gain} \ge \gamma$, where $\gamma$ serves as the minimum gain threshold for tree regularization.

### 2 — In-Place Tree Update (Algorithm 1)

When a new batch $\mathcal{B}$ arrives:

1. For each tree $T_i$, route every sample in $\mathcal{B}$ through the tree.
2. At each internal node $v$ reached by new samples:
   - Recompute $G_v = \sum_{j \in v} g_j$, $H_v = \sum_{j \in v} h_j$ over all samples (old + new).
   - Re-run split finding to get the new best $(f^*, \theta^*)$.
   - Call `check_retrain(v, f^*, \theta^*)`.
3. If `check_retrain` returns True: rebuild the subtree at $v$.  
   Otherwise: continue routing to children.
4. Update leaf weights $w = -G/(H + \lambda)$ for all reachable leaves.

### 3 — Split Stability Check (Algorithm 2)

`check_retrain` at node $v$ with current split $(f_v, \theta_v)$:

- **Feature change**: if $f^* \neq f_v$ → retrain.
- **Threshold rank shift**: sort all split candidates for feature $f_v$ by gain descending. If $\theta_v$'s rank > $\lceil K \cdot \sigma \rceil$ (where $K$ = number of candidates, $\sigma \in (0,1)$) → retrain.
- Otherwise → stable, just route.

$\sigma$ controls aggressiveness: lower $\sigma$ retrains more freely.

---

## Results

### Base Model — Parity with Official XGBoost

Evaluated on standard benchmarks with $T=50$ estimators, $\text{depth}=4$, $\eta=0.05$, $\lambda=1.0$, $80/20$ train/test split:

| Dataset | Model | Train Acc | Test Acc | Train F1 | Test F1 |
|---|---|---|---|---|---|
| Breast Cancer (569 samples, 30 features) | **From-Scratch** | 0.9934 | **0.9474** | 0.9934 | 0.9476 |
| Breast Cancer | Official XGBoost | 0.9934 | 0.9561 | 0.9934 | 0.9560 |
| Wine Binary (178 samples, 13 features) | **From-Scratch** | 1.0000 | **0.9444** | 1.0000 | 0.9444 |
| Wine Binary | Official XGBoost | 1.0000 | 0.9444 | 1.0000 | 0.9444 |

### Incremental Update — Concept Drift Adaptation Benchmark

Setup: 2,000 synthetic samples (20 features, 8 informative, 10% label noise), 90/10 split. The streaming batch experiences feature mean shifts drawn from $\mathcal{N}(0, 0.5)$ to simulate concept drift. Evaluated on the drifted stream ($T=10$ trees, $\eta=0.05$, $\sigma=0.5$):

| Model | Training Regimen | Streaming Drift Handling | Accuracy | F1 |
|---|---|---|---|---|
| **Static Baseline** | 1,800 clean | None (frozen model) | 0.7850 | 0.7849 |
| **Full Retrain** | 1,800 + 200 drifted | Retrained from scratch on pooled data ($O(N)$ rebuild) | 0.8100 | 0.8100 |
| **Incremental Update** | 1,800 clean | **In-place adaptation** on 200 drifted ($O(1)$ on stable nodes) | **0.8300** | **0.8300** |

#### Multi-Run Statistical Distribution (20 Independent Seeds)
Across 20 independent trials with varying synthetic distributions and drift perturbations:
- **Mean accuracy improvement over frozen baseline**: **+5.7%**
- **Adaptation reliability**: **95.0%** of runs show equal or superior accuracy vs frozen baseline
- **Median gap closed relative to full retrain**: **100.0%** (in several trials, selective in-place adaptation outperforms pooled retraining because pooled retrain is dominated by the 90% historical distribution).

---

## Repository Structure

```
XGBoost/
├── src/
│   ├── tree.py               # TreeNode and Tree classes (histogram split finding, sparsity-aware)
│   ├── xgboost_model.py      # XGBoostModel — base classifier/regressor
│   ├── incremental_model.py  # IncrementalXGBoostClassifier — in-place drift adaptation
│   ├── objective.py          # Gradient/hessian computation (logloss, RMSE)
│   ├── metric.py             # Evaluation metrics (AUC, F1, RMSE, R²)
│   └── analysis.py           # Post-training analysis utilities
├── examples/
│   ├── train_breast_cancer.py   # Base model demo + comparison with official xgboost
│   └── incremental_drift.py     # Concept drift benchmark (3-model comparison)
├── tests/
│   ├── test_base_model.py       # Unit tests for XGBoostModel
│   └── test_incremental.py      # Unit tests for IncrementalXGBoostClassifier
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/tejask-42/XGBoost.git
cd XGBoost
pip install -r requirements.txt
```

---

## Quickstart

### Running Examples & Benchmarks

```bash
# Compare base from-scratch model against official C++ XGBoost
python examples/train_breast_cancer.py

# Run the 3-model concept drift adaptation benchmark
python examples/incremental_drift.py
```

### Python API

```python
from src.incremental_model import IncrementalXGBoostClassifier
from src.objective import Objective

# 1. Train on historical data
model = IncrementalXGBoostClassifier(params, Objective())
model.fit(X_train, y_train)

# 2. Adapt in-place when a drifted stream batch arrives (no full retrain needed)
model.update(X_stream, y_stream, sigma=0.5)

# 3. Predict adapted probabilities
preds = model.predict(X_test)
```

### Run Tests

```bash
python tests/test_base_model.py
python tests/test_incremental.py
```

---

## Implementation Details

| Feature | Details |
|---|---|
| Split finding | Histogram binning with configurable `n_bins` |
| Sparsity | Learned default directions for missing/zero values |
| Objectives | Binary cross-entropy (classification), RMSE (regression) |
| Class imbalance | `scale_pos_weight` computed automatically |
| Regularization | L2 leaf penalty $\lambda$, minimum gain $\gamma$ |
| Early stopping | Validation-set metric with configurable patience |
| Incremental update | In-place gradient/leaf update, selective subtree retrain |
| Drift detection | Split stability via sigma-rank criterion |

---

## In Progress

- **Multi-batch streaming benchmark**: A sequential online stream simulation with multiple concept drift events injected at different timepoints — evaluating the incremental model continuously across a full stream, rather than a single-batch update scenario. This will produce proper prequential evaluation curves (accuracy vs stream position) and compare against periodic full retraining.

---

## License

MIT
