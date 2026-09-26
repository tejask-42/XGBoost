"""
examples/train_breast_cancer.py
================================
Demonstrates XGBoostModel (base, from-scratch) on:
  - Breast Cancer Wisconsin dataset
  - Wine dataset (binarized)

Compares against the official xgboost library to validate implementation parity.

Run from repo root:
    python examples/train_breast_cancer.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from xgboost_model import XGBoostModel
from objective import Objective

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[NOTE] Official xgboost not installed — skipping library comparison.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_params(n_estimators=50):
    return {
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.05,
        "lambda_": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "objective_type": "classification",
        "n_bins": 64,
        "use_sparsity_split": False,
    }


def run_dataset(X, y, dataset_name, n_estimators=50):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Samples: {X.shape[0]}   Features: {X.shape[1]}")
    print(f"  Classes: {np.unique(y)}   Distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(float), test_size=0.2, random_state=42, stratify=y
    )

    # ── From-scratch model ─────────────────────────────────────────────────────
    params = make_params(n_estimators)
    model = XGBoostModel(params, Objective())
    model.fit(X_train, y_train)

    train_preds = (model.predict(X_train) >= 0.5).astype(int)
    test_preds  = (model.predict(X_test)  >= 0.5).astype(int)

    scratch_train_acc = accuracy_score(y_train, train_preds)
    scratch_test_acc  = accuracy_score(y_test,  test_preds)
    scratch_train_f1  = f1_score(y_train, train_preds, average='weighted')
    scratch_test_f1   = f1_score(y_test,  test_preds,  average='weighted')

    print(f"\n  [From-Scratch XGBoost]")
    print(f"    Train Accuracy: {scratch_train_acc:.4f}   Train F1: {scratch_train_f1:.4f}")
    print(f"    Test  Accuracy: {scratch_test_acc:.4f}   Test  F1: {scratch_test_f1:.4f}")

    # ── Official xgboost comparison ────────────────────────────────────────────
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=1.0,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train.astype(int))

        xgb_train_preds = xgb_model.predict(X_train)
        xgb_test_preds  = xgb_model.predict(X_test)

        xgb_train_acc = accuracy_score(y_train, xgb_train_preds)
        xgb_test_acc  = accuracy_score(y_test,  xgb_test_preds)
        xgb_train_f1  = f1_score(y_train, xgb_train_preds, average='weighted')
        xgb_test_f1   = f1_score(y_test,  xgb_test_preds,  average='weighted')

        print(f"\n  [Official XGBoost library]")
        print(f"    Train Accuracy: {xgb_train_acc:.4f}   Train F1: {xgb_train_f1:.4f}")
        print(f"    Test  Accuracy: {xgb_test_acc:.4f}   Test  F1: {xgb_test_f1:.4f}")

        delta_test = scratch_test_acc - xgb_test_acc
        print(f"\n  Delta (scratch - library) Test Accuracy: {delta_test:+.4f}")

    print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print("From-Scratch XGBoost — Parity Benchmark")
    print("=" * 60)

    # Breast Cancer
    data = load_breast_cancer()
    run_dataset(data.data, data.target, "Breast Cancer Wisconsin", n_estimators=50)

    # Wine (binarize: class 0 vs classes 1+2)
    wine = load_wine()
    y_wine = (wine.target == 0).astype(int)
    run_dataset(wine.data, y_wine, "Wine (binary: class 0 vs rest)", n_estimators=50)


if __name__ == "__main__":
    main()
