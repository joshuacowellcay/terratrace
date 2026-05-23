"""Tile a 2D raster (or stack of rasters) into fixed-size patches.

Used by the real-data path when slicing Sentinel-2 mosaics. The synthetic path has its own
``tile_world`` because it generates the world and the tiles together; this module operates
on arrays that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileIndex:
    """Identifies a tile by its grid position and the pixel window it covers."""

    grid_row: int
    grid_col: int
    y_start: int
    x_start: int
    height: int
    width: int

    @property
    def tile_id(self) -> str:
        return f"r{self.grid_row:03d}_c{self.grid_col:03d}"


def compute_tile_indices(
    raster_height: int,
    raster_width: int,
    tile_size: int,
) -> list[TileIndex]:
    """Return non-overlapping tile windows covering the raster.

    Edge tiles are dropped if the remainder is smaller than ``tile_size``. This guarantees
    every emitted tile has identical dimensions, which keeps the downstream Dataset simple
    (no per-tile padding logic).
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    rows = raster_height // tile_size
    cols = raster_width // tile_size
    out: list[TileIndex] = []
    for r in range(rows):
        for c in range(cols):
            out.append(
                TileIndex(
                    grid_row=r,
                    grid_col=c,
                    y_start=r * tile_size,
                    x_start=c * tile_size,
                    height=tile_size,
                    width=tile_size,
                )
            )
    return out


def extract_tile(arr: np.ndarray, idx: TileIndex) -> np.ndarray:
    """Slice a tile out of a 2D or 3D (H, W, C) array."""
    y0, x0 = idx.y_start, idx.x_start
    y1, x1 = y0 + idx.height, x0 + idx.width
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1].copy()
    if arr.ndim == 3:
        return arr[y0:y1, x0:x1, :].copy()
    raise ValueError(f"expected 2D or 3D array, got {arr.ndim}D")


def tile_coverage_fraction(
    raster_height: int,
    raster_width: int,
    tile_size: int,
) -> float:
    """How much of the raster ends up inside emitted tiles. Useful for sanity-checking
    that an arbitrary input resolution isn't silently losing most of the data."""
    rows = raster_height // tile_size
    cols = raster_width // tile_size
    covered = rows * cols * tile_size * tile_size
    total = raster_height * raster_width
    return covered / total if total else 0.0
