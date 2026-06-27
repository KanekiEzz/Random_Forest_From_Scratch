from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

try:
    from .utils import (
        candidate_thresholds,
        entropy,
        information_gain,
        majority_label,
        resolve_max_features,
        split_indices,
    )
except ImportError:
    from utils import (
        candidate_thresholds,
        entropy,
        information_gain,
        majority_label,
        resolve_max_features,
        split_indices,
    )

Label = Hashable


@dataclass
class _TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    value: Label | None = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def _validate_training_data(X: Sequence[Sequence[float]], y: Sequence[Label]) -> None:
    if not X:
        raise ValueError("X cannot be empty.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    n_features = len(X[0])
    if n_features == 0:
        raise ValueError("X must contain at least one feature.")

    for row_index, row in enumerate(X, start=1):
        if len(row) != n_features:
            raise ValueError(
                f"Inconsistent number of features at row {row_index}: "
                f"expected {n_features}, got {len(row)}."
            )


def _validate_prediction_data(X: Sequence[Sequence[float]], n_features: int) -> None:
    for row_index, row in enumerate(X, start=1):
        if len(row) != n_features:
            raise ValueError(
                f"Prediction row {row_index} has {len(row)} features; expected {n_features}."
            )


class DecisionTreeClassifier:
    """Entropy-based decision tree classifier for numeric features."""

    def __init__(
        self,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        min_gain: float = 1e-7,
        max_thresholds: int = 32,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be at least 1 or None.")
        if min_gain < 0:
            raise ValueError("min_gain must be non-negative.")
        if max_thresholds < 1:
            raise ValueError("max_thresholds must be at least 1.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_gain = min_gain
        self.max_thresholds = max_thresholds
        self._random = random.Random(random_state)
        self.root: _TreeNode | None = None
        self.n_features_: int | None = None
        self.max_features_: int | None = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Label]) -> "DecisionTreeClassifier":
        X_rows = [list(row) for row in X]
        y_values = list(y)
        _validate_training_data(X_rows, y_values)

        self.n_features_ = len(X_rows[0])
        self.max_features_ = self.max_features or self.n_features_
        self.root = self._grow_tree(X_rows, y_values, depth=0)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[Label]:
        if self.root is None or self.n_features_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict().")

        X_rows = [list(row) for row in X]
        _validate_prediction_data(X_rows, self.n_features_)
        return [self._predict_row(row, self.root) for row in X_rows]

    def _grow_tree(self, X: list[list[float]], y: list[Label], depth: int) -> _TreeNode:
        leaf_value = majority_label(y)
        depth_limit_reached = self.max_depth is not None and depth >= self.max_depth

        if depth_limit_reached or len(set(y)) == 1 or len(y) < self.min_samples_split:
            return _TreeNode(value=leaf_value)

        split = self._best_split(X, y)
        if split is None:
            return _TreeNode(value=leaf_value)

        feature_index, threshold, left_indices, right_indices, gain = split
        if gain <= self.min_gain:
            return _TreeNode(value=leaf_value)

        X_left = [X[index] for index in left_indices]
        y_left = [y[index] for index in left_indices]
        X_right = [X[index] for index in right_indices]
        y_right = [y[index] for index in right_indices]

        return _TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=self._grow_tree(X_left, y_left, depth + 1),
            right=self._grow_tree(X_right, y_right, depth + 1),
        )

    def _best_split(
        self,
        X: list[list[float]],
        y: list[Label],
    ) -> tuple[int, float, list[int], list[int], float] | None:
        n_features = len(X[0])
        feature_indices = list(range(n_features))

        if self.max_features_ is not None and self.max_features_ < n_features:
            feature_indices = self._random.sample(feature_indices, self.max_features_)

        parent_entropy = entropy(y)
        best_gain = 0.0
        best_split: tuple[int, float, list[int], list[int], float] | None = None

        for feature_index in feature_indices:
            values = [row[feature_index] for row in X]
            for threshold in candidate_thresholds(values, self.max_thresholds):
                left_indices, right_indices = split_indices(X, feature_index, threshold)
                if (
                    len(left_indices) < self.min_samples_leaf
                    or len(right_indices) < self.min_samples_leaf
                ):
                    continue

                y_left = [y[index] for index in left_indices]
                y_right = [y[index] for index in right_indices]
                gain = information_gain(parent_entropy, y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (
                        feature_index,
                        threshold,
                        left_indices,
                        right_indices,
                        gain,
                    )

        return best_split

    def _predict_row(self, row: Sequence[float], node: _TreeNode) -> Label:
        current = node
        while not current.is_leaf:
            if current.feature_index is None or current.threshold is None:
                raise RuntimeError("Encountered an invalid internal tree node.")

            if row[current.feature_index] <= current.threshold:
                current = current.left  # type: ignore[assignment]
            else:
                current = current.right  # type: ignore[assignment]

            if current is None:
                raise RuntimeError(
                    "Encountered an invalid tree structure during prediction."
                )

        if current.value is None:
            raise RuntimeError("Encountered a leaf node without a prediction value.")
        return current.value


class RandomForestClassifier:
    """Random Forest classifier implemented from scratch using decision trees."""

    def __init__(
        self,
        n_trees: int = 50,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | None = "sqrt",
        max_thresholds: int = 32,
        bootstrap: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_trees < 1:
            raise ValueError("n_trees must be at least 1.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if max_thresholds < 1:
            raise ValueError("max_thresholds must be at least 1.")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.bootstrap = bootstrap
        self._random = random.Random(random_state)
        self.trees: list[DecisionTreeClassifier] = []
        self.n_features_: int | None = None

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[Label],
    ) -> "RandomForestClassifier":
        X_rows = [list(row) for row in X]
        y_values = list(y)
        _validate_training_data(X_rows, y_values)

        self.n_features_ = len(X_rows[0])
        max_features = resolve_max_features(self.max_features, self.n_features_)
        self.trees = []

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X_rows, y_values)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                max_thresholds=self.max_thresholds,
                random_state=self._random.randint(0, 1_000_000_000),
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[Label]:
        if not self.trees or self.n_features_ is None:
            raise RuntimeError(
                "RandomForestClassifier must be fitted before predict()."
            )

        X_rows = [list(row) for row in X]
        _validate_prediction_data(X_rows, self.n_features_)

        tree_predictions = [tree.predict(X_rows) for tree in self.trees]
        return [majority_label(votes) for votes in zip(*tree_predictions)]

    def _bootstrap_sample(
        self,
        X: list[list[float]],
        y: list[Label],
    ) -> tuple[list[list[float]], list[Label]]:
        if not self.bootstrap:
            return list(X), list(y)

        indices = [self._random.randrange(len(X)) for _ in range(len(X))]
        return [X[index] for index in indices], [y[index] for index in indices]


__all__ = ["DecisionTreeClassifier", "RandomForestClassifier"]


