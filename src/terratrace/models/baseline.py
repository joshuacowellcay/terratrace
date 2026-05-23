"""NDVI-difference baseline. The non-ML benchmark the deep model is measured against.

NDVI ((NIR - Red) / (NIR + Red)) is high for healthy vegetation and low for bare soil or
water. If a pixel was vegetation in ``before`` (NDVI_before high) and isn't in ``after``
(NDVI_after low), it's a deforestation candidate. The detector thresholds the NDVI drop:

    pred = (NDVI_before - NDVI_after) > threshold

The threshold is calibrated on the validation set (we sweep candidate values and pick the
one with highest F1 for the forest-loss class). Any improvement the deep model shows over
this baseline then represents the value of learning context the threshold can't see.

Band indices are passed explicitly so this module stays reusable if a different stack is
ever swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def ndvi(image: np.ndarray, *, red_idx: int = 0, nir_idx: int = 3) -> np.ndarray:
    """Compute NDVI for a (H, W, C) image. Output shape (H, W), range ~[-1, 1]."""
    if image.ndim != 3:
        raise ValueError(f"expected 3D (H,W,C), got shape {image.shape}")
    red = image[..., red_idx].astype(np.float32)
    nir = image[..., nir_idx].astype(np.float32)
    return (nir - red) / (nir + red + EPS)


def ndvi_difference(
    before: np.ndarray,
    after: np.ndarray,
    *,
    red_idx: int = 0,
    nir_idx: int = 3,
) -> np.ndarray:
    """NDVI(before) - NDVI(after). Positive values indicate vegetation loss."""
    return ndvi(before, red_idx=red_idx, nir_idx=nir_idx) - ndvi(
        after, red_idx=red_idx, nir_idx=nir_idx
    )


def predict(
    before: np.ndarray,
    after: np.ndarray,
    *,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict forest-loss mask via NDVI difference threshold.

    Pixels outside ``valid_mask`` (typically cloudy or out-of-frame) are forced to 0.
    """
    diff = ndvi_difference(before, after)
    pred = (diff > threshold).astype(np.uint8)
    if valid_mask is not None:
        pred = pred * valid_mask.astype(np.uint8)
    return pred


@dataclass(frozen=True)
class ThresholdSearchResult:
    best_threshold: float
    best_f1: float
    sweep: list[tuple[float, float]]


def calibrate_threshold(
    before: np.ndarray,
    after: np.ndarray,
    label: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
) -> ThresholdSearchResult:
    """Pick the NDVI-difference threshold that maximises F1 on the forest-loss class.

    ``before``, ``after``: stacked across all tiles in the calibration split, shape
    (N, H, W, C). ``label``: (N, H, W) uint8 ground truth. ``valid_mask``: same shape as
    label, 1 where the pixel should count. Returns the best threshold, its F1, and the
    full sweep for plotting.
    """
    if candidates is None:
        candidates = np.linspace(0.05, 0.6, 23, dtype=np.float32)

    if before.shape != after.shape or before.shape[:-1] != label.shape:
        raise ValueError(
            f"shape mismatch: before={before.shape}, after={after.shape}, label={label.shape}"
        )
    n, h, w, c = before.shape
    flat_label = label.reshape(-1).astype(bool)
    flat_valid = (
        valid_mask.reshape(-1).astype(bool) if valid_mask is not None else np.ones_like(flat_label)
    )

    ndvi_before = ndvi(before.reshape(n * h, w, c))
    ndvi_after = ndvi(after.reshape(n * h, w, c))
    diff = (ndvi_before - ndvi_after).reshape(-1)

    sweep: list[tuple[float, float]] = []
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for thr in candidates:
        pred = (diff > thr) & flat_valid
        tp = int(np.logical_and(pred, flat_label).sum())
        fp = int(np.logical_and(pred, ~flat_label).sum())
        fn = int(np.logical_and(~pred, flat_label).sum())
        if tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        sweep.append((float(thr), float(f1)))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    return ThresholdSearchResult(best_threshold=best_threshold, best_f1=best_f1, sweep=sweep)
