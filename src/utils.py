from __future__ import annotations

from cProfile import label
import math
from collections import Counter
from collections.abc import Hashable, Sequence

Label = Hashable


def entropy(labels: Sequence[Label]) -> float:
    """Return the entropy of a label distribution."""
    if not labels:
        return 0.0

    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def information_gain(
    parent_entropy: float,
    parent_labels: Sequence[Label],
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
) -> float:
    """Return the information gain from splitting a parent node."""
    if not parent_labels:
        return 0.0

    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    child_entropy = left_weight * entropy(left_labels) + right_weight * entropy(
        right_labels
    )
    return parent_entropy - child_entropy


def split_indices( X: Sequence[Sequence[float]],  feature_index: int, threshold: float,) -> tuple[list[int], list[int]]:
    """Split row indices using a numeric threshold."""
    left_indices: list[int] = []
    right_indices: list[int] = []

    for index, row in enumerate(X):
        if row[feature_index] <= threshold:
            left_indices.append(index)
        else:
            right_indices.append(index)

    return left_indices, right_indices


def candidate_thresholds(values: Sequence[float], max_thresholds: int) -> list[float]:
    """Return candidate thresholds between sorted unique values."""
    if max_thresholds < 1:
        raise ValueError("max_thresholds must be at least 1.")

    unique_values = sorted(set(values))
    if len(unique_values) < 2:
        return []

    thresholds = [
        (unique_values[index] + unique_values[index + 1]) / 2.0
        for index in range(len(unique_values) - 1)
    ]
    if len(thresholds) <= max_thresholds:
        return thresholds

    if max_thresholds == 1:
        return [thresholds[len(thresholds) // 2]]

    step = (len(thresholds) - 1) / (max_thresholds - 1)
    selected: list[float] = []
    seen: set[float] = set()

    for index in range(max_thresholds):
        threshold = thresholds[round(index * step)]
        if threshold not in seen:
            selected.append(threshold)
            seen.add(threshold)

    return selected


def resolve_max_features(max_features: str | int | None, n_features: int) -> int:
    """Resolve max_features into a concrete feature count."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1.")

    if max_features is None:
        return n_features

    if isinstance(max_features, int):
        return max(1, min(max_features, n_features))

    value = max_features.strip().lower()
    if value.isdigit():
        return max(1, min(int(value), n_features))
    if value == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    if value == "log2":
        return max(1, int(math.log2(n_features)))

    raise ValueError("max_features must be None, an int, 'sqrt', or 'log2'.")


def majority_label(labels: Sequence[Label]) -> Label:
    """Return the most common label with stable first-seen tie-breaking."""
    if not labels:
        raise ValueError("labels cannot be empty.")

    counts = Counter(labels)
    highest_count = max(counts.values())

    for label in labels:
        if counts[label] == highest_count:
            return label

    raise RuntimeError("Failed to determine the majority label.")


__all__ = [
    "Label",
    "entropy",
    "information_gain",
    "split_indices",
    "candidate_thresholds",
    "resolve_max_features",
    "majority_label",
]
