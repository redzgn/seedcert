"""Metric helpers (DESIGN Sec 5, WP1)."""

from __future__ import annotations

import numpy as np
import pytest

from seedcert.verifiers import metrics


def test_metric_names_are_the_four_reported() -> None:
    assert metrics.METRIC_NAMES == ("test_accuracy", "precision", "recall", "f1")


def test_accuracy_matches_argmax() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    labels = np.array([0, 1, 1])
    mask = np.array([True, True, True])
    assert metrics.accuracy(logits, labels, mask) == pytest.approx(2 / 3)


def test_accuracy_empty_mask_is_nan() -> None:
    logits = np.zeros((3, 2))
    out = metrics.accuracy(logits, np.zeros(3, int), np.zeros(3, bool))
    assert np.isnan(out)


def test_macro_prf_perfect_case() -> None:
    logits = np.array([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0], [0.0, 9.0]])
    labels = np.array([0, 0, 1, 1])
    mask = np.ones(4, dtype=bool)
    assert metrics.macro_prf(logits, labels, mask) == pytest.approx((1.0, 1.0, 1.0))


def test_macro_prf_hand_worked_confusion() -> None:
    # pred = [0, 0, 1, 1, 0], labels = [0, 0, 0, 1, 1]
    # class 0: tp=2 fp=1 fn=1 -> p=r=f=2/3
    # class 1: tp=1 fp=1 fn=1 -> p=r=f=1/2
    logits = np.array(
        [[9.0, 0.0], [9.0, 0.0], [0.0, 9.0], [0.0, 9.0], [9.0, 0.0]]
    )
    labels = np.array([0, 0, 0, 1, 1])
    mask = np.ones(5, dtype=bool)
    p, r, f = metrics.macro_prf(logits, labels, mask)
    assert p == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert r == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert f == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_all_metrics_keys_and_agreement() -> None:
    logits = np.array([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0], [0.0, 9.0]])
    labels = np.array([0, 1, 0, 1])
    mask = np.ones(4, dtype=bool)
    out = metrics.all_metrics(logits, labels, mask)
    assert set(out) == set(metrics.METRIC_NAMES)
    assert out["test_accuracy"] == pytest.approx(1.0)
