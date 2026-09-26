"""
incremental_model.py
====================
IncrementalXGBoostClassifier — extends XGBoostModel with in-place tree updates
for adapting to concept drift without full retraining.

After a normal .fit(), calling .update(X_new, y_new) routes new samples through
each tree, recalculates gradients, and selectively retrains only the subtrees
whose splits have drifted — leaving stable parts of the tree unchanged.
Usage:
    from incremental_model import IncrementalXGBoostClassifier
    from objective import Objective

    model = IncrementalXGBoostClassifier(params, Objective())
    model.fit(X_train, y_train)

    # When new data arrives with potential distribution shift:
    model.update(X_new, y_new, sigma=0.5)
    preds = model.predict(X_test)
"""

import numpy as np
from math import ceil
from xgboost_model import XGBoostModel
from tree import TreeNode


class IncrementalXGBoostClassifier(XGBoostModel):
    """
    XGBoost classifier with in-place incremental tree updates.

    Inherits all training and prediction logic from XGBoostModel.
    Adds:
        - _attach_indices(X, y): post-fit pass to store sample indices,
          gradient sums G and hessian sums H on every tree node.
        - update(X_new, y_new): in-place adaptation to new data / concept drift.

    Parameters
    ----------
    params : dict
        Same hyperparameter dict as XGBoostModel. Additional keys:
        (none required — update() accepts them as arguments)
    objective : Objective
        Loss function object (same as XGBoostModel).
    """

    def __init__(self, params, objective):
        super().__init__(params, objective)
        # Store full training data for gradient recalculation during updates.
        # Each .update() call appends to these.
        self._X_all = None
        self._y_all = None
        # Per-tree gradient/hessian arrays (shape: [n_samples] each), grown on update.
        self._tree_grads = []
        self._tree_hess = []
        self._batch_num = 0

    # ------------------------------------------------------------------
    # fit — train normally, then attach indices/G/H to every node
    # ------------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=10,
            scale_pos_weight=None):
        """Train model, then run post-processing to attach node statistics."""
        super().fit(X, y, X_val=X_val, y_val=y_val,
                    early_stopping_rounds=early_stopping_rounds,
                    scale_pos_weight=scale_pos_weight)
        self._X_all = X.copy()
        self._y_all = y.copy()
        self._attach_indices(X, y)

    # ------------------------------------------------------------------
    # _attach_indices — post-processing: walk every tree, store indices/G/H
    # ------------------------------------------------------------------

    def _attach_indices(self, X, y):
        """
        Walk every tree node and attach:
            node.indices  — np.ndarray of sample row indices reaching this node
            node.G        — sum of gradients at this node
            node.H        — sum of hessians at this node

        This is called once after fit() and after each update(). It is O(n * T)
        where n = number of samples and T = number of trees.
        """
        n = X.shape[0]
        all_indices = np.arange(n)

        # Rebuild cumulative raw predictions up to each tree to get per-tree grads
        self._tree_grads = []
        self._tree_hess = []

        # Raw predictions before each tree (log-odds space)
        raw_preds = np.zeros(n)  # base score absorbed into first tree's context

        for tree in self.trees:
            # Gradients/hessians at this boosting round
            grad, hess = self.objective.gradient_hessian(y, raw_preds, "logloss")
            self._tree_grads.append(grad.copy())
            self._tree_hess.append(hess.copy())

            # Walk tree BFS, assigning indices and computing G/H at each node
            queue = [(tree.root, all_indices)]
            while queue:
                node, idx = queue.pop(0)
                node.indices = idx
                node.G = float(np.sum(grad[idx]))
                node.H = float(np.sum(hess[idx]))

                if node.is_leaf or node.feature_index is None:
                    continue

                feat_vals = X[idx, node.feature_index]
                left_mask = feat_vals <= node.threshold

                # Handle sparse/missing the same way predict does
                if node.default_direction == 'left':
                    sparse_mask = np.isnan(feat_vals) | (feat_vals == 0)
                    left_mask = left_mask | sparse_mask
                elif node.default_direction == 'right':
                    sparse_mask = np.isnan(feat_vals) | (feat_vals == 0)
                    left_mask = left_mask & ~sparse_mask

                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]

                if node.left is not None and len(left_idx) > 0:
                    queue.append((node.left, left_idx))
                if node.right is not None and len(right_idx) > 0:
                    queue.append((node.right, right_idx))

            # Advance raw predictions by this tree's contribution
            raw_preds += self.params["learning_rate"] * self._predict_tree(X, tree)

    # ------------------------------------------------------------------
    # update — in-place adaptation to new batch
    # ------------------------------------------------------------------

    def update(self, X_new, y_new, sigma=0.5, update_sample_ratio=0.05):
        """
        Adapt the model in-place to a new batch of data.

        For each tree:
          1. New samples are routed through the existing structure.
          2. At each node, if new samples are a significant fraction, we
             re-find the best split on the combined data.
          3. check_retrain() decides whether the split has drifted enough
             to warrant rebuilding the subtree at this node.
          4. If not retraining, leaf weights are updated directly (O(1) per leaf).

        Parameters
        ----------
        X_new : np.ndarray, shape (m, n_features)
        y_new : np.ndarray, shape (m,)
        sigma : float
            Top-k fraction for retrain decision. Lower = more aggressive retraining.
            Default 0.5 (retrain if old threshold falls outside top-50% by gain).
        update_sample_ratio : float
            Minimum fraction of node samples that must be new to trigger split check.
            Default 0.05.
        """
        self._batch_num += 1
        n_old = self._X_all.shape[0]
        batch_len = X_new.shape[0]
        new_indices = np.arange(n_old, n_old + batch_len)

        # Append new data
        self._X_all = np.concatenate([self._X_all, X_new], axis=0)
        self._y_all = np.concatenate([self._y_all, y_new], axis=0)
        X_all = self._X_all
        y_all = self._y_all

        # Raw predictions up to (not including) tree i
        raw_preds_cumul = np.zeros(n_old + batch_len)

        for i, tree in enumerate(self.trees):
            # ── Recompute gradients for all samples (old + new) at this round ──
            grad_all, hess_all = self.objective.gradient_hessian(
                y_all, raw_preds_cumul, "logloss"
            )
            self._tree_grads[i] = grad_all
            self._tree_hess[i] = hess_all

            # Expand root's index set
            tree.root.indices = np.arange(n_old + batch_len)
            tree.root.G = float(np.sum(grad_all))
            tree.root.H = float(np.sum(hess_all))

            # BFS walk: route new samples, check retrain at each internal node
            queue = [(tree.root, new_indices)]
            affected_leaves = []

            while queue:
                node, node_new = queue.pop(0)

                # Update node stats with new samples
                node.indices = np.union1d(node.indices, node_new) if node.indices is not None else node_new
                node.G = float(np.sum(grad_all[node.indices]))
                node.H = float(np.sum(hess_all[node.indices]))

                if node.is_leaf:
                    affected_leaves.append(node)
                    continue

                # ── Check whether split has drifted ──────────────────────────
                n_node = len(node.indices)
                n_new_at_node = len(node_new)

                if n_new_at_node > n_node * update_sample_ratio:
                    should_retrain, new_feat, new_thresh = self._check_retrain(
                        node, X_all, grad_all, hess_all, sigma
                    )
                    if should_retrain:
                        # Rebuild the subtree from this node downward
                        self._rebuild_subtree(node, X_all, grad_all, hess_all)
                        continue  # subtree rebuilt; don't route further

                # ── Route new samples to children ────────────────────────────
                feat_vals = X_all[node_new, node.feature_index]
                left_mask = feat_vals <= node.threshold

                if node.default_direction == 'left':
                    sparse = np.isnan(feat_vals) | (feat_vals == 0)
                    left_mask = left_mask | sparse
                elif node.default_direction == 'right':
                    sparse = np.isnan(feat_vals) | (feat_vals == 0)
                    left_mask = left_mask & ~sparse

                left_new = node_new[left_mask]
                right_new = node_new[~left_mask]

                if node.left is not None:
                    if len(left_new) > 0:
                        queue.append((node.left, left_new))
                    elif node.left.is_leaf:
                        affected_leaves.append(node.left)

                if node.right is not None:
                    if len(right_new) > 0:
                        queue.append((node.right, right_new))
                    elif node.right.is_leaf:
                        affected_leaves.append(node.right)

            # ── Update leaf weights for unchanged leaves ──────────────────────
            for leaf in affected_leaves:
                if leaf.indices is not None and len(leaf.indices) > 0:
                    g_sum = np.sum(grad_all[leaf.indices])
                    h_sum = np.sum(hess_all[leaf.indices])
                    leaf.value = -g_sum / (h_sum + self.params["lambda_"])
                    leaf.G = float(g_sum)
                    leaf.H = float(h_sum)

            # Advance cumulative raw predictions
            raw_preds_cumul += self.params["learning_rate"] * self._predict_tree(
                X_all, tree
            )

    # ------------------------------------------------------------------
    # _check_retrain — Algorithm 2: split stability check
    # ------------------------------------------------------------------

    def _check_retrain(self, node, X, grad, hess, sigma):
        """
        Determine whether the optimal split at this node has drifted enough
        to require retraining the subtree.

        Returns
        -------
        should_retrain : bool
        new_feat : int  — best feature on current data
        new_thresh : float — best threshold on current data
        """
        idx = node.indices
        best_gain = -np.inf
        best_feat = None
        best_thresh = None
        best_gains_for_feat = None  # gain array over all thresholds of best feature

        g = grad[idx]
        h = hess[idx]
        G_total = g.sum()
        H_total = h.sum()
        lambda_ = self.params["lambda_"]
        n_node = len(idx)

        for feat in range(X.shape[1]):
            feat_vals = X[idx, feat]
            feat_min = np.nanmin(feat_vals)
            feat_max = np.nanmax(feat_vals)
            if feat_min == feat_max or np.isnan(feat_min):
                continue

            # Use same binning approach as base model
            n_bins = self.params.get("n_bins", 64)
            bin_edges = np.linspace(feat_min, feat_max, n_bins)
            bins = np.clip(np.digitize(feat_vals, bin_edges) - 1, 0, n_bins - 1)

            bin_g = np.zeros(n_bins)
            bin_h = np.zeros(n_bins)
            np.add.at(bin_g, bins, g)
            np.add.at(bin_h, bins, h)

            # Vectorised gain across all split points
            cum_g = np.cumsum(bin_g)[:-1]
            cum_h = np.cumsum(bin_h)[:-1]
            cum_g_r = G_total - cum_g
            cum_h_r = H_total - cum_h

            min_hw = self.params.get("min_child_weight", 1.0)
            valid = (cum_h >= min_hw) & (cum_h_r >= min_hw)

            gains = np.zeros(n_bins - 1)
            gains[valid] = 0.5 * (
                cum_g[valid] ** 2 / (cum_h[valid] + lambda_)
                + cum_g_r[valid] ** 2 / (cum_h_r[valid] + lambda_)
                - G_total ** 2 / (H_total + lambda_)
            )

            feat_best_idx = np.argmax(gains)
            feat_best_gain = gains[feat_best_idx]

            if feat_best_gain > best_gain:
                best_gain = feat_best_gain
                best_feat = feat
                best_thresh = bin_edges[feat_best_idx + 1]
                best_gains_for_feat = gains

        if best_feat is None:
            # Can't find any valid split → no retrain
            return False, None, None

        # ── Feature change check ──────────────────────────────────────────────
        if node.feature_index is None or best_feat != node.feature_index:
            return True, best_feat, best_thresh

        # ── Threshold rank check (sigma criterion) ────────────────────────────
        # Sort thresholds by gain descending; check where the current threshold ranks
        sorted_indices = np.argsort(best_gains_for_feat)[::-1]
        n_bins_feat = len(best_gains_for_feat)
        top_k = ceil(n_bins_feat * sigma)

        # Map current threshold to a bin index
        feat_vals_all = X[idx, best_feat]
        feat_min = np.nanmin(feat_vals_all)
        feat_max = np.nanmax(feat_vals_all)
        bin_edges = np.linspace(feat_min, feat_max, self.params.get("n_bins", 64))

        # Find which bin index the current threshold falls into
        old_bin_idx = np.searchsorted(bin_edges, node.threshold, side='right') - 1
        old_bin_idx = int(np.clip(old_bin_idx, 0, n_bins_feat - 1))

        # Rank of old bin among sorted gains (0 = best)
        rank_arr = np.where(sorted_indices == old_bin_idx)[0]
        old_rank = int(rank_arr[0]) if len(rank_arr) > 0 else n_bins_feat

        should_retrain = old_rank > top_k
        return should_retrain, best_feat, best_thresh

    # ------------------------------------------------------------------
    # _rebuild_subtree — retrain a subtree on current combined data
    # ------------------------------------------------------------------

    def _rebuild_subtree(self, node, X, grad, hess):
        """
        Rebuild the subtree rooted at `node` using the current gradient/hessian
        arrays and the node's current index set. Updates node in-place,
        strictly respecting the existing node.depth so total tree depth <= max_depth.
        """
        idx = node.indices
        depth = getattr(node, 'depth', 0)
        self._build_node(node, X, grad, hess, idx, depth)

    def _build_node(self, node, X, grad, hess, idx, depth):
        """
        Recursively rebuild the subtree at `node` with samples `idx`.
        Mirrors the base Tree._build_tree logic while strictly tracking depth.
        """
        lambda_ = self.params["lambda_"]
        gamma = self.params.get("gamma", 0.0)
        min_child_weight = self.params.get("min_child_weight", 1.0)
        max_depth = self.params["max_depth"]
        n_bins = self.params.get("n_bins", 64)

        # Update node stats & depth
        node.indices = idx
        node.depth = depth
        node.G = float(np.sum(grad[idx]))
        node.H = float(np.sum(hess[idx]))

        # Stopping conditions
        if depth >= max_depth or len(idx) < 2 or node.H < min_child_weight:
            node.is_leaf = True
            node.feature_index = None
            node.threshold = None
            node.left = None
            node.right = None
            node.value = -node.G / (node.H + lambda_)
            return

        # Find best split
        best_gain = gamma
        best_feat = None
        best_thresh = None
        best_left_idx = None
        best_right_idx = None

        G_total = node.G
        H_total = node.H
        g = grad[idx]
        h = hess[idx]

        for feat in range(X.shape[1]):
            feat_vals = X[idx, feat]
            feat_min = np.nanmin(feat_vals)
            feat_max = np.nanmax(feat_vals)
            if feat_min == feat_max or np.isnan(feat_min):
                continue

            bin_edges = np.linspace(feat_min, feat_max, n_bins)
            bins = np.clip(np.digitize(feat_vals, bin_edges) - 1, 0, n_bins - 1)

            bin_g = np.zeros(n_bins)
            bin_h = np.zeros(n_bins)
            bin_cnt = np.zeros(n_bins)
            np.add.at(bin_g, bins, g)
            np.add.at(bin_h, bins, h)
            np.add.at(bin_cnt, bins, 1)

            cum_g = 0.0
            cum_h = 0.0
            for b in range(n_bins - 1):
                if bin_cnt[b] == 0:
                    continue
                cum_g += bin_g[b]
                cum_h += bin_h[b]
                rem_g = G_total - cum_g
                rem_h = H_total - cum_h
                if cum_h < min_child_weight or rem_h < min_child_weight:
                    continue
                gain = 0.5 * (
                    cum_g ** 2 / (cum_h + lambda_)
                    + rem_g ** 2 / (rem_h + lambda_)
                    - G_total ** 2 / (H_total + lambda_)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = bin_edges[b + 1]

        if best_feat is None:
            node.is_leaf = True
            node.feature_index = None
            node.threshold = None
            node.left = None
            node.right = None
            node.value = -G_total / (H_total + lambda_)
            return

        # Apply split
        feat_vals = X[idx, best_feat]
        left_mask = feat_vals <= best_thresh
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            node.is_leaf = True
            node.feature_index = None
            node.threshold = None
            node.left = None
            node.right = None
            node.value = -G_total / (H_total + lambda_)
            return

        node.is_leaf = False
        node.feature_index = best_feat
        node.threshold = best_thresh
        node.value = 0.0
        node.gain = best_gain

        node.left = TreeNode(depth=depth + 1)
        node.right = TreeNode(depth=depth + 1)

        self._build_node(node.left, X, grad, hess, left_idx, depth + 1)
        self._build_node(node.right, X, grad, hess, right_idx, depth + 1)
