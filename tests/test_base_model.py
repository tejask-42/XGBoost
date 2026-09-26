"""
tests/test_base_model.py
=========================
Unit tests for XGBoostModel (from-scratch base implementation).

Run:
    python -m pytest tests/test_base_model.py -v
or:
    python tests/test_base_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost_model import XGBoostModel
from objective import Objective


def make_params(n_estimators=30):
    return {
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.1,
        "lambda_": 1.0,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "objective_type": "classification",
        "n_bins": 64,
        "use_sparsity_split": False,
    }


def test_breast_cancer_accuracy():
    """Model should achieve >= 90% test accuracy on Breast Cancer dataset."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target.astype(float),
        test_size=0.2, random_state=42, stratify=data.target
    )
    model = XGBoostModel(make_params(30), Objective())
    model.fit(X_train, y_train)
    preds = (model.predict(X_test) >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"  Breast Cancer test accuracy: {acc:.4f}")
    assert acc >= 0.90, f"Expected >= 0.90, got {acc:.4f}"


def test_predict_output_shape():
    """predict() should return a 1-D array of length n_samples."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 10)
    y = (X[:, 0] > 0).astype(float)
    model = XGBoostModel(make_params(5), Objective())
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,), f"Expected (200,), got {preds.shape}"


def test_predict_probabilities_in_range():
    """Classification predict() should return values in [0, 1]."""
    rng = np.random.RandomState(1)
    X = rng.randn(100, 5)
    y = (X[:, 0] > 0).astype(float)
    model = XGBoostModel(make_params(5), Objective())
    model.fit(X, y)
    probs = model.predict(X)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), \
        "Predicted probabilities outside [0, 1]"


def test_no_trees_after_init():
    """A freshly constructed model should have no trees."""
    model = XGBoostModel(make_params(), Objective())
    assert len(model.trees) == 0


def test_correct_number_of_trees():
    """After fit, model should have exactly n_estimators trees (or fewer if early-stopped)."""
    rng = np.random.RandomState(2)
    X = rng.randn(300, 8)
    y = (X[:, 0] > 0).astype(float)
    n_est = 10
    params = make_params(n_est)
    model = XGBoostModel(params, Objective())
    model.fit(X, y)
    assert len(model.trees) <= n_est, \
        f"Expected <= {n_est} trees, got {len(model.trees)}"


if __name__ == "__main__":
    tests = [
        test_predict_output_shape,
        test_predict_probabilities_in_range,
        test_no_trees_after_init,
        test_correct_number_of_trees,
        test_breast_cancer_accuracy,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            print(f"  PASSED\n")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
    print(f"Results: {passed}/{len(tests)} passed")
