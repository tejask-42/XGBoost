"""
examples/incremental_drift.py
==============================
Demonstrates IncrementalXGBoostClassifier adapting to concept drift.

Three models are compared on a held-out drifted stream:
  1. Baseline      — trained on clean data, never updated
  2. Full Retrain  — trained from scratch on clean + drifted data (upper bound)
  3. Incremental   — trained on clean data, updated in-place on drifted data

Run from repo root:
    python examples/incremental_drift.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import copy
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from incremental_model import IncrementalXGBoostClassifier
from objective import Objective


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_params():
    return {
        "n_estimators": 10,
        "max_depth": 4,
        "learning_rate": 0.05,
        "lambda_": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "objective_type": "classification",
        "n_bins": 64,
        "use_sparsity_split": False,
    }


def add_feature_drift(X, drift_scale=0.5, random_state=42):
    """Shift feature means to simulate concept drift."""
    rng = np.random.RandomState(random_state)
    shifts = rng.normal(0, drift_scale, size=X.shape[1])
    return X + shifts


def evaluate(model, X, y, name):
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='weighted')
    print(f"  {name:<30}  Accuracy: {acc:.4f}   F1: {f1:.4f}")
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("InplaceXGB — Single-Batch Concept Drift Benchmark")
    print("=" * 65)

    # Generate dataset
    print("\n[1] Generating synthetic classification dataset (2000 samples, 20 features)...")
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=8,
        n_redundant=12, n_clusters_per_class=3,
        flip_y=0.10, class_sep=1.0, random_state=42
    )

    # 90% train, 10% streaming
    X_train, X_stream_clean, y_train, y_stream = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Inject concept drift into the stream
    X_stream = add_feature_drift(X_stream_clean, drift_scale=0.5)

    # Combined dataset for full retrain baseline
    X_full = np.vstack([X_train, X_stream])
    y_full = np.concatenate([y_train, y_stream])

    print(f"    Train : {X_train.shape}  (no drift)")
    print(f"    Stream: {X_stream.shape}  (feature mean shift, drift_scale=0.5)")

    # ── Model 1: Static Baseline (no update) ──────────────────────────────────
    print("\n[2] Training Static Baseline (trained on clean historical data only)...")
    params = make_params()
    baseline = IncrementalXGBoostClassifier(params, Objective())
    baseline.fit(X_train, y_train)

    # ── Model 2: Full Retrain (pooled historical + stream data) ───────────────
    print("\n[3] Training Full Retrain (rebuilding from scratch on pooled data)...")
    retrain = IncrementalXGBoostClassifier(make_params(), Objective())
    retrain.fit(X_full, y_full)

    # ── Model 3: Incremental update ───────────────────────────────────────────
    print("\n[4] Running Incremental Update on drifted stream...")
    incremental = copy.deepcopy(baseline)
    incremental.update(X_stream, y_stream, sigma=0.5)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Results on drifted stream set:")
    print("-" * 65)
    base_acc, _ = evaluate(baseline, X_stream, y_stream, "Static Baseline (no update)")
    ret_acc, _  = evaluate(retrain,  X_stream, y_stream, "Full Retrain (pooled data)")
    incr_acc, _ = evaluate(incremental, X_stream, y_stream, "Incremental In-Place Update")

    if ret_acc > base_acc:
        gap_closed = (incr_acc - base_acc) / (ret_acc - base_acc) * 100
        print(f"\n  Gap closed relative to Full Retrain: {gap_closed:.1f}%")
        print(f"  (100% = matches full retrain; >100% = adapts better than pooled retrain)")
    print("=" * 65)


if __name__ == "__main__":
    main()
