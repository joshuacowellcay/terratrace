"""Tests for raster tiling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from terratrace.data.tiling import (
    compute_tile_indices,
    extract_tile,
    tile_coverage_fraction,
)


def test_tile_indices_exact_fit() -> None:
    out = compute_tile_indices(raster_height=256, raster_width=384, tile_size=128)
    assert len(out) == 2 * 3
    assert {(i.grid_row, i.grid_col) for i in out} == {(r, c) for r in range(2) for c in range(3)}


def test_tile_indices_drops_partial_edge() -> None:
    out = compute_tile_indices(raster_height=300, raster_width=300, tile_size=128)
    # floor(300/128) = 2; we keep only the 2x2 fully-covered grid, no partial edge tiles.
    assert len(out) == 4
    for idx in out:
        assert idx.y_start + idx.height <= 300
        assert idx.x_start + idx.width <= 300


def test_tile_indices_rejects_zero_size() -> None:
    with pytest.raises(ValueError):
        compute_tile_indices(100, 100, 0)


def test_extract_tile_2d_and_3d() -> None:
    arr2 = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    arr3 = np.broadcast_to(arr2[..., None], (16, 16, 3)).copy()
    indices = compute_tile_indices(16, 16, 8)
    out2 = extract_tile(arr2, indices[0])
    out3 = extract_tile(arr3, indices[0])
    assert out2.shape == (8, 8)
    assert out3.shape == (8, 8, 3)
    np.testing.assert_array_equal(out2, arr2[:8, :8])


def test_extract_tile_rejects_bad_ndim() -> None:
    arr = np.zeros((8, 8, 4, 2))
    idx = compute_tile_indices(8, 8, 4)[0]
    with pytest.raises(ValueError):
        extract_tile(arr, idx)


def test_coverage_fraction() -> None:
    assert tile_coverage_fraction(256, 256, 128) == 1.0
    # 300/128 = 2 tiles wide → 256/300 ≈ 0.85 per side → 0.728 area
    assert 0.7 < tile_coverage_fraction(300, 300, 128) < 0.75
