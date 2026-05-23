"""Tests for the geographic-block train/val/test splitter.

The core invariant (no spatial block appears in more than one split) is the property that
defends against the most common silent bug in remote-sensing ML pipelines.
"""

from __future__ import annotations

import pytest

from terratrace.data.splits import (
    SPLITS,
    TileLocation,
    assert_no_spatial_overlap,
    split_tiles,
)


def _grid(rows: int, cols: int) -> list[TileLocation]:
    return [
        TileLocation(tile_id=f"r{r:03d}_c{c:03d}", grid_row=r, grid_col=c)
        for r in range(rows)
        for c in range(cols)
    ]


def test_split_returns_all_three_splits_nonempty() -> None:
    tiles = _grid(8, 8)
    out = split_tiles(tiles, grid_size=4, val_blocks=2, test_blocks=2, seed=0)
    for s in SPLITS:
        assert out[s], f"split {s} is empty"
    assert sum(len(out[s]) for s in SPLITS) == len(tiles)


def test_split_preserves_every_tile_exactly_once() -> None:
    tiles = _grid(12, 12)
    out = split_tiles(tiles, grid_size=4, val_blocks=3, test_blocks=3, seed=42)
    flat_ids = [t.tile_id for s in SPLITS for t in out[s]]
    assert len(flat_ids) == len(set(flat_ids)) == len(tiles)


def test_split_is_geographically_disjoint() -> None:
    """The headline invariant: a macro-block's tiles all share a split."""
    tiles = _grid(12, 12)
    out = split_tiles(tiles, grid_size=4, val_blocks=3, test_blocks=3, seed=7)

    rows_per_block = 12 // 4
    cols_per_block = 12 // 4

    block_to_split: dict[tuple[int, int], str] = {}
    for split_name, items in out.items():
        for tile in items:
            block = (tile.grid_row // rows_per_block, tile.grid_col // cols_per_block)
            existing = block_to_split.setdefault(block, split_name)
            assert (
                existing == split_name
            ), f"block {block} appears in both '{existing}' and '{split_name}' (spatial leakage)"


def test_assert_no_spatial_overlap_passes_for_valid_split() -> None:
    tiles = _grid(8, 8)
    out = split_tiles(tiles, grid_size=4, val_blocks=2, test_blocks=2, seed=1)
    assert_no_spatial_overlap(out)


def test_assert_no_spatial_overlap_catches_duplicate() -> None:
    a = TileLocation(tile_id="a", grid_row=0, grid_col=0)
    splits = {
        "train": [a],
        "val": [a],  # same tile in two splits, must fail
        "test": [TileLocation(tile_id="b", grid_row=1, grid_col=1)],
    }
    with pytest.raises(AssertionError):
        assert_no_spatial_overlap(splits)


def test_split_handles_non_divisible_grid() -> None:
    """Real Sentinel-2 rasters rarely tile into a perfect square. The splitter must cope."""
    tiles = _grid(25, 8)  # the exact shape that broke the build the first time real data ran
    out = split_tiles(tiles, grid_size=4, val_blocks=3, test_blocks=3, seed=0)
    for s in SPLITS:
        assert out[s], f"split {s} empty on non-divisible grid"
    assert_no_spatial_overlap(out)


def test_split_rejects_grid_smaller_than_block_count() -> None:
    """If the tile grid is finer than ``grid_size``, the split is ill-defined."""
    tiles = _grid(3, 3)
    with pytest.raises(ValueError):
        split_tiles(tiles, grid_size=4, val_blocks=1, test_blocks=1, seed=0)


def test_split_is_deterministic_per_seed() -> None:
    tiles = _grid(8, 8)
    out_a = split_tiles(tiles, grid_size=4, val_blocks=2, test_blocks=2, seed=99)
    out_b = split_tiles(tiles, grid_size=4, val_blocks=2, test_blocks=2, seed=99)
    for s in SPLITS:
        assert [t.tile_id for t in out_a[s]] == [t.tile_id for t in out_b[s]]
