"""Metrics computed from a run's cached ``[N, C]`` logits (DESIGN Sec 5).

NumPy only - no scikit-learn. All four metrics are read off the same cached
logits on the frozen test mask so they match exactly what the trainer reported.
"""

from __future__ import annotations

import numpy as np

METRIC_NAMES: tuple[str, ...] = ("test_accuracy", "precision", "recall", "f1")


def _predictions(logits: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    return np.asarray(logits)[m].argmax(axis=1), m


def accuracy(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Argmax accuracy of ``logits`` against ``labels`` on ``mask``.
    ``nan`` if ``mask`` selects nothing."""
    pred, m = _predictions(logits, mask)
    if pred.size == 0:
        return float("nan")
    return float((pred == np.asarray(labels)[m]).mean())


def macro_prf(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(macro_precision, macro_recall, macro_f1)`` averaged over the
    classes that actually occur in ``labels[mask]``.

    A class with no predicted positives contributes precision 0; a class with no
    true instances is not in the average. ``(nan, nan, nan)`` if ``mask`` is
    empty.
    """
    pred, m = _predictions(logits, mask)
    if pred.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    true = np.asarray(labels)[m]
    classes = np.unique(true)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for c in classes:
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
    )


def all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """``{"test_accuracy", "precision", "recall", "f1"}`` in one pass."""
    p, r, f = macro_prf(logits, labels, mask)
    return {
        "test_accuracy": accuracy(logits, labels, mask),
        "precision": p,
        "recall": r,
        "f1": f,
    }
