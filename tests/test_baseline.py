"""Tests for the NDVI-difference baseline detector."""

from __future__ import annotations

import numpy as np

from terratrace.data.config import SyntheticConfig
from terratrace.data.synthetic import generate_world, tile_world
from terratrace.models.baseline import (
    calibrate_threshold,
    ndvi,
    ndvi_difference,
    predict,
)


def test_ndvi_shape_and_range() -> None:
    img = np.random.default_rng(0).random((16, 16, 4), dtype=np.float32)
    out = ndvi(img)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    # NDVI is bounded in [-1, 1]; with our +EPS divisor it can asymptote close to +/-1.
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_ndvi_difference_zero_for_identical_inputs() -> None:
    img = np.random.default_rng(0).random((8, 8, 4), dtype=np.float32)
    diff = ndvi_difference(img, img)
    np.testing.assert_allclose(diff, 0.0, atol=1e-6)


def test_predict_respects_valid_mask() -> None:
    # Construct a case where NDVI difference is high but a pixel is masked invalid.
    h, w = 4, 4
    before = np.zeros((h, w, 4), dtype=np.float32)
    after = np.zeros((h, w, 4), dtype=np.float32)
    before[..., 3] = 0.8  # high NIR -> high NDVI before
    after[..., 0] = 0.4  # bumped red, low NIR -> low NDVI after -> large positive diff

    pred_all_valid = predict(before, after, threshold=0.1)
    assert pred_all_valid.sum() > 0

    mask = np.zeros((h, w), dtype=np.uint8)
    pred_masked = predict(before, after, threshold=0.1, valid_mask=mask)
    assert pred_masked.sum() == 0


def test_calibrate_threshold_picks_a_better_score_than_extremes_on_synthetic_data() -> None:
    cfg = SyntheticConfig(seed=2024, tile_size=32, grid_rows=4, grid_cols=4, cloud_fraction=0.0)
    world = generate_world(cfg)
    tiles = tile_world(world, cfg)
    before_stack = np.stack([t.before for t in tiles])
    after_stack = np.stack([t.after for t in tiles])
    label_stack = np.stack([t.label for t in tiles])

    result = calibrate_threshold(before_stack, after_stack, label_stack)

    # Picked threshold lies in the sensible range we sweep.
    assert 0.05 <= result.best_threshold <= 0.6

    # The best threshold's F1 should beat both extremes of the sweep; otherwise the
    # calibration isn't doing anything useful.
    sweep = dict(result.sweep)
    assert result.best_f1 > sweep[min(sweep)]
    assert result.best_f1 > sweep[max(sweep)]
    # On a generator that puts ~5-20% of pixels in the loss class with a high NIR drop, F1
    # should also be meaningfully positive.
    assert result.best_f1 > 0.3, f"baseline F1 too low: {result.best_f1:.3f}"
