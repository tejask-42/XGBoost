"""
tests/test_incremental.py
==========================
Unit tests for IncrementalXGBoostClassifier.

Key assertions:
  1. After .update(), accuracy on drifted data is >= baseline (adaptation is real).
  2. Gap closed >= 50% (conservative threshold; paper shows ~88.9%).
  3. Predict output is still valid probabilities after update.
  4. Leaf values change after update (model actually changed).

Run:
    python -m pytest tests/test_incremental.py -v
or:
    python tests/test_incremental.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import copy
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from incremental_model import IncrementalXGBoostClassifier
from objective import Objective


# ── Shared fixtures ────────────────────────────────────────────────────────────

def make_params(n_estimators=10):
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


def make_drift_scenario(n_train=1800, n_stream=200, drift_scale=0.5, seed=42):
    """Generate base dataset, split into train and drifted stream."""
    X, y = make_classification(
        n_samples=n_train + n_stream, n_features=20, n_informative=8,
        n_redundant=12, n_clusters_per_class=3,
        flip_y=0.10, class_sep=1.0, random_state=seed
    )
    X_train, X_stream_clean, y_train, y_stream = train_test_split(
        X, y, test_size=n_stream / (n_train + n_stream),
        random_state=seed, stratify=y
    )
    rng = np.random.RandomState(seed)
    shifts = rng.normal(0, drift_scale, size=X_stream_clean.shape[1])
    X_stream = X_stream_clean + shifts
    return X_train, X_stream, y_train, y_stream


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_update_improves_accuracy_after_drift():
    """
    After concept drift, incremental update should improve accuracy
    over the non-updated baseline.
    """
    X_train, X_stream, y_train, y_stream = make_drift_scenario()

    baseline = IncrementalXGBoostClassifier(make_params(), Objective())
    baseline.fit(X_train, y_train)

    incremental = copy.deepcopy(baseline)
    incremental.update(X_stream, y_stream, sigma=0.5)

    base_acc = accuracy_score(y_stream, (baseline.predict(X_stream) >= 0.5).astype(int))
    incr_acc = accuracy_score(y_stream, (incremental.predict(X_stream) >= 0.5).astype(int))

    print(f"  Baseline accuracy on drifted stream : {base_acc:.4f}")
    print(f"  Incremental accuracy on drifted stream: {incr_acc:.4f}")
    assert incr_acc >= base_acc, \
        f"Incremental ({incr_acc:.4f}) should be >= Baseline ({base_acc:.4f})"


def test_gap_closed_relative_to_retrain():
    """
    Incremental update should close a significant fraction of the gap
    between the frozen baseline and full retrain on the drifted distribution.
    """
    X_train, X_stream, y_train, y_stream = make_drift_scenario(seed=0)
    X_full = np.vstack([X_train, X_stream])
    y_full = np.concatenate([y_train, y_stream])

    baseline = IncrementalXGBoostClassifier(make_params(), Objective())
    baseline.fit(X_train, y_train)

    retrain = IncrementalXGBoostClassifier(make_params(), Objective())
    retrain.fit(X_full, y_full)

    incremental = copy.deepcopy(baseline)
    incremental.update(X_stream, y_stream, sigma=0.5)

    base_acc  = accuracy_score(y_stream, (baseline.predict(X_stream)    >= 0.5).astype(int))
    ret_acc   = accuracy_score(y_stream, (retrain.predict(X_stream)     >= 0.5).astype(int))
    incr_acc  = accuracy_score(y_stream, (incremental.predict(X_stream) >= 0.5).astype(int))

    print(f"  Baseline accuracy : {base_acc:.4f}")
    print(f"  Retrain accuracy  : {ret_acc:.4f}")
    print(f"  Incremental accuracy: {incr_acc:.4f}")

    # Accuracy must improve over frozen baseline
    assert incr_acc > base_acc, f"Incremental ({incr_acc:.4f}) did not improve over baseline ({base_acc:.4f})"
    if ret_acc > base_acc:
        gap_closed = (incr_acc - base_acc) / (ret_acc - base_acc) * 100
        print(f"  Gap closed: {gap_closed:.1f}%")
        assert gap_closed >= 50.0, f"Expected gap closed >= 50%, got {gap_closed:.1f}%"


def test_predict_probabilities_valid_after_update():
    """After update, predict() should still return valid probabilities in [0, 1]."""
    X_train, X_stream, y_train, y_stream = make_drift_scenario(n_train=500, n_stream=100)
    model = IncrementalXGBoostClassifier(make_params(5), Objective())
    model.fit(X_train, y_train)
    model.update(X_stream, y_stream)

    probs = model.predict(X_train)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), \
        "Predicted probabilities outside [0, 1] after update"


def test_leaf_values_change_after_update():
    """Leaf values in at least one tree should change after update — model is not frozen."""
    X_train, X_stream, y_train, y_stream = make_drift_scenario(n_train=500, n_stream=100)
    model = IncrementalXGBoostClassifier(make_params(5), Objective())
    model.fit(X_train, y_train)

    def collect_leaf_values(model):
        leaves = []
        def walk(node):
            if node is None:
                return
            if node.is_leaf:
                leaves.append(node.value)
            else:
                walk(node.left)
                walk(node.right)
        for tree in model.trees:
            walk(tree.root)
        return np.array(leaves)

    leaves_before = collect_leaf_values(model).copy()
    model.update(X_stream, y_stream)
    leaves_after = collect_leaf_values(model)

    min_len = min(len(leaves_before), len(leaves_after))
    changed = not np.allclose(leaves_before[:min_len], leaves_after[:min_len], atol=1e-6)
    assert changed, "Leaf values did not change after update — model may not be updating"


def test_multiple_updates_do_not_crash():
    """Calling update() multiple times should not raise any errors."""
    X_train, X_stream, y_train, y_stream = make_drift_scenario(n_train=500, n_stream=50)
    model = IncrementalXGBoostClassifier(make_params(5), Objective())
    model.fit(X_train, y_train)
    for _ in range(3):
        X_next, X_stream, y_next, y_stream = train_test_split(
            X_stream, y_stream, test_size=0.5, random_state=0
        )
        model.update(X_next, y_next)
    preds = model.predict(X_stream)
    assert preds.shape[0] == X_stream.shape[0]


if __name__ == "__main__":
    tests = [
        test_update_improves_accuracy_after_drift,
        test_gap_closed_relative_to_retrain,
        test_predict_probabilities_valid_after_update,
        test_leaf_values_change_after_update,
        test_multiple_updates_do_not_crash,
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
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print()
    print(f"Results: {passed}/{len(tests)} passed")
