"""Segmentation metrics for the forest-loss class.

We headline IoU/precision/recall/F1 — *not* pixel accuracy. With ~5% positive pixels, a "predict
0 everywhere" baseline already scores 95% accuracy and 0 IoU, so accuracy alone is misleading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SegMetrics:
    iou: float
    precision: float
    recall: float
    f1: float
    accuracy: float  # included only for completeness; never headline this alone.

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def compute_segmentation_metrics(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None = None,
) -> SegMetrics:
    """Compute the standard binary-segmentation metrics.

    All three arguments may be numpy arrays or torch tensors of any shape; we flatten and
    treat them as 1-D. ``weights == 0`` excludes the pixel from every metric.
    """
    pred_arr = _to_bool(pred)
    target_arr = _to_bool(target)
    if weights is not None:
        mask = _to_numpy(weights).astype(bool).ravel()
        pred_arr = pred_arr[mask]
        target_arr = target_arr[mask]

    tp = int(np.logical_and(pred_arr, target_arr).sum())
    fp = int(np.logical_and(pred_arr, ~target_arr).sum())
    fn = int(np.logical_and(~pred_arr, target_arr).sum())
    tn = int(np.logical_and(~pred_arr, ~target_arr).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    return SegMetrics(
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
    )


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_bool(x: np.ndarray | torch.Tensor) -> np.ndarray:
    return _to_numpy(x).astype(bool).ravel()
