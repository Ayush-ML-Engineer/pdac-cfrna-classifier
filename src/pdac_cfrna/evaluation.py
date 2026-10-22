"""Metrics for the PDAC score: AUC and a bootstrap confidence interval for it."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve. Raises ValueError if y_true contains only one class."""
    if len(np.unique(np.asarray(y_true))) < 2:  # newer scikit-learn returns nan instead of raising
        raise ValueError("AUC is undefined: y_true contains only one class")

    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the AUC of one fixed set of scores.

    Re-draws the test samples with replacement n_boot times and recomputes the AUC each
    time. The model is NOT retrained: only the existing test labels and scores are resampled.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if len(y) != len(s):
        raise ValueError(f"y_true and y_score differ in length: {len(y)} vs {len(s)}")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    n = len(y)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # n row positions, repeats allowed
        if len(np.unique(y[idx])) < 2:  # AUC is undefined with a single class: skip this draw
            continue
        aucs.append(auc(y[idx], s[idx]))  # the SAME idx for labels and scores keeps pairs matched

    if not aucs:
        raise ValueError("No bootstrap sample contained both classes; cannot compute an AUC")

    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(aucs, [tail, 100 - tail])
    return float(low), float(high)
