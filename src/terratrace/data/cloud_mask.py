"""Cloud-mask utilities.

For Sentinel-2 L2A we read the Scene Classification Layer (SCL) and treat the classes that
represent cloud, cirrus, or cloud shadow as invalid. The synthetic path already emits a
boolean cloud mask, which we just convert to the same dtype.

SCL classes (Sentinel-2 L2A documentation):
    0  no data
    1  saturated / defective
    2  dark area pixels
    3  cloud shadow
    4  vegetation
    5  not vegetated
    6  water
    7  unclassified
    8  cloud medium probability
    9  cloud high probability
    10 thin cirrus
    11 snow / ice

We treat 0, 1, 3, 8, 9, 10 as invalid by default. ``11`` (snow) is kept; the Amazon AOI
never sees it but other regions might need to mask it too.
"""

from __future__ import annotations

import numpy as np

INVALID_SCL_CLASSES: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})


def cloud_mask_from_scl(
    scl: np.ndarray, invalid_classes: frozenset[int] = INVALID_SCL_CLASSES
) -> np.ndarray:
    """Return a uint8 mask where 1 = pixel is cloudy/invalid and must not be trained on.

    Accepts ``scl`` as a 2D integer array (any integer dtype is fine).
    """
    if scl.ndim != 2:
        raise ValueError(f"SCL must be 2D, got shape {scl.shape}")
    flat = scl.astype(np.int32, copy=False)
    mask = np.zeros_like(flat, dtype=np.uint8)
    for cls in invalid_classes:
        mask |= (flat == cls).astype(np.uint8)
    return mask


def cloud_fraction(mask: np.ndarray) -> float:
    """Fraction of pixels in ``mask`` flagged as cloudy (mask == 1). Range [0, 1]."""
    if mask.size == 0:
        return 0.0
    return float(mask.astype(bool).mean())


def valid_pixel_weights(cloud_before: np.ndarray, cloud_after: np.ndarray) -> np.ndarray:
    """Per-pixel weights for use during training and evaluation.

    A pixel that is cloudy in either timestamp can't be reliably labelled as forest-loss, so
    we down-weight it to zero. Returns a float32 array of the same shape as the inputs.
    """
    if cloud_before.shape != cloud_after.shape:
        raise ValueError(f"shape mismatch: before={cloud_before.shape}, after={cloud_after.shape}")
    invalid = (cloud_before.astype(bool)) | (cloud_after.astype(bool))
    return (~invalid).astype(np.float32)
