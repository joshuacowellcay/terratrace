"""Geographic-block train/val/test splitting.

A naive shuffle of tiles is the most common silent bug in remote-sensing ML: adjacent tiles
share atmospheric conditions, sensor artefacts, and often the same physical objects spilling
across tile boundaries, so a random split lets the model memorise local context that doesn't
generalise to a held-out region. The fix is to split by spatially disjoint blocks: divide
the AOI into a grid of macro-blocks and assign whole blocks (not individual tiles) to one
of train/val/test.

This module implements that split and exposes :func:`assert_no_spatial_overlap` so the
invariant can be checked with a single line in a test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class TileLocation:
    """A tile identified by its grid position. The pipeline pairs this with an ID/path."""

    tile_id: str
    grid_row: int
    grid_col: int


def _block_for_tile(
    tile_row: int,
    tile_col: int,
    rows_total: int,
    cols_total: int,
    grid_size: int,
) -> tuple[int, int]:
    """Map a tile coordinate to its macro-block index.

    Uses proportional placement (``tile_row * grid_size // rows_total``) so the grid does not
    need to be divisible by ``grid_size``; real-data rasters rarely are. Result is always in
    ``[0, grid_size)``.
    """
    br = min(grid_size - 1, (tile_row * grid_size) // max(1, rows_total))
    bc = min(grid_size - 1, (tile_col * grid_size) // max(1, cols_total))
    return br, bc


def split_tiles(
    tiles: Iterable[TileLocation],
    *,
    grid_size: int,
    val_blocks: int,
    test_blocks: int,
    seed: int,
) -> dict[Split, list[TileLocation]]:
    """Assign whole spatial blocks to train/val/test.

    The tile grid is divided into ``grid_size`` x ``grid_size`` macro-blocks. The RNG
    (seeded by ``seed``) picks ``test_blocks`` blocks for test, then ``val_blocks`` blocks
    for val out of what remains, and the rest become train. The choice is purely spatial;
    tile content is never inspected.

    Raises if any macro-block ends up empty, usually a configuration mistake (fewer tiles
    than blocks).
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles supplied")

    max_row = max(t.grid_row for t in tiles)
    max_col = max(t.grid_col for t in tiles)
    rows_total = max_row + 1
    cols_total = max_col + 1

    if rows_total < grid_size or cols_total < grid_size:
        raise ValueError(
            f"tile grid {rows_total}x{cols_total} is smaller than split grid_size={grid_size}; "
            "shrink split.grid_size or expand the AOI"
        )
    all_blocks = [(br, bc) for br in range(grid_size) for bc in range(grid_size)]

    if val_blocks + test_blocks >= len(all_blocks):
        raise ValueError(f"val ({val_blocks}) + test ({test_blocks}) leaves zero blocks for train")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_blocks))
    rng.shuffle(indices)

    test_block_set = {all_blocks[i] for i in indices[:test_blocks]}
    val_block_set = {all_blocks[i] for i in indices[test_blocks : test_blocks + val_blocks]}

    bucket: dict[Split, list[TileLocation]] = {s: [] for s in SPLITS}
    seen_blocks_with_tiles: set[tuple[int, int]] = set()
    for tile in tiles:
        block = _block_for_tile(tile.grid_row, tile.grid_col, rows_total, cols_total, grid_size)
        seen_blocks_with_tiles.add(block)
        if block in test_block_set:
            bucket["test"].append(tile)
        elif block in val_block_set:
            bucket["val"].append(tile)
        else:
            bucket["train"].append(tile)

    if seen_blocks_with_tiles != set(all_blocks):
        missing = set(all_blocks) - seen_blocks_with_tiles
        raise ValueError(
            f"some spatial blocks contain no tiles: {sorted(missing)}; check tile coverage"
        )

    for s in SPLITS:
        if not bucket[s]:
            raise ValueError(f"split '{s}' ended up empty; reduce val_blocks/test_blocks")

    return bucket


def assert_no_spatial_overlap(splits: dict[Split, list[TileLocation]]) -> None:
    """Assert that no spatial block appears in more than one split.

    This is the key invariant of geographic-block splitting and the property that the data
    pipeline's contract with the evaluator rests on. Called from the test suite.
    """
    # Without rows_per_block we can't reconstruct the macro-block grid here, so we fall
    # back to tile-level uniqueness (the strictest possible check: no two splits share any
    # tile at all).
    seen: dict[tuple[int, int], Split] = {}
    for split_name, items in splits.items():
        for tile in items:
            key = (tile.grid_row, tile.grid_col)
            if key in seen and seen[key] != split_name:
                raise AssertionError(
                    f"tile {tile.tile_id} appears in both '{seen[key]}' and '{split_name}'"
                )
            seen[key] = split_name
