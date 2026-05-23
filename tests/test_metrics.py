"""Tests for the segmentation metrics."""

from __future__ import annotations

import numpy as np

from terratrace.training.metrics import compute_segmentation_metrics


def test_perfect_prediction_scores_all_ones() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    metrics = compute_segmentation_metrics(target, target)
    assert metrics.iou == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0


def test_completely_wrong_prediction_scores_zero_iou() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred = 1 - target
    metrics = compute_segmentation_metrics(pred, target)
    assert metrics.iou == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_weights_exclude_pixels() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    # Without mask: pred has 1 TP, 1 FN, 0 FP, so IoU = 1/2 = 0.5.
    metrics_full = compute_segmentation_metrics(pred, target)
    assert metrics_full.iou == 0.5
    # Mask out the FN; only the matching TP remains and accuracy on the kept pixels is 100%.
    weights = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    metrics_masked = compute_segmentation_metrics(pred, target, weights)
    assert metrics_masked.iou == 1.0


def test_handles_all_negative_target_gracefully() -> None:
    target = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4), dtype=np.uint8)
    metrics = compute_segmentation_metrics(pred, target)
    # No positives anywhere: IoU/precision/recall/F1 conventionally 0; accuracy 1.
    assert metrics.iou == 0.0
    assert metrics.accuracy == 1.0
