"""Tests for cloud-mask helpers."""

from __future__ import annotations

import numpy as np
import pytest

from terratrace.data.cloud_mask import (
    INVALID_SCL_CLASSES,
    cloud_fraction,
    cloud_mask_from_scl,
    valid_pixel_weights,
)


def test_invalid_classes_include_clouds_and_shadow() -> None:
    # Sanity check: standard SCL cloud/cirrus/shadow codes are in the invalid set.
    assert {3, 8, 9, 10}.issubset(INVALID_SCL_CLASSES)


def test_cloud_mask_from_scl_basic() -> None:
    # 0 = nodata, 4 = veg, 8 = cloud, 9 = cloud, 10 = cirrus, 6 = water
    scl = np.array(
        [
            [0, 4, 4, 6],
            [8, 9, 10, 4],
            [4, 4, 3, 4],
            [4, 5, 4, 4],
        ],
        dtype=np.int32,
    )
    mask = cloud_mask_from_scl(scl)
    assert mask.dtype == np.uint8
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, expected)


def test_cloud_mask_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        cloud_mask_from_scl(np.zeros((4, 4, 4), dtype=np.int32))


def test_cloud_fraction_zero_and_full() -> None:
    assert cloud_fraction(np.zeros((10, 10), dtype=np.uint8)) == 0.0
    assert cloud_fraction(np.ones((10, 10), dtype=np.uint8)) == 1.0


def test_valid_pixel_weights_zeros_invalid_pixels() -> None:
    cb = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    ca = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    w = valid_pixel_weights(cb, ca)
    # Pixel (0,1) is cloudy in before, (1,0) cloudy in after → both must be zeroed.
    np.testing.assert_array_equal(w, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_valid_pixel_weights_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        valid_pixel_weights(np.zeros((4, 4), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint8))
