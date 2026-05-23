"""The :class:`Tile` payload — a single training sample, source-agnostic.

A Tile is the unit emitted by both the synthetic generator and the real-data STAC fetcher. The
pipeline orchestrator writes Tiles to disk in a uniform on-disk layout and records each one in
the manifest. Downstream code never needs to know whether a tile came from satellite imagery
or the procedural generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    """One spatial patch: before/after imagery, ground-truth loss mask, and cloud masks.

    Shapes:
        before, after  : (H, W, C) float32, reflectance-like in [0, 1]
        label          : (H, W)    uint8, 1 = forest-loss
        cloud_before   : (H, W)    uint8, 1 = cloudy / invalid
        cloud_after    : (H, W)    uint8, 1 = cloudy / invalid
    """

    grid_row: int
    grid_col: int
    before: np.ndarray
    after: np.ndarray
    label: np.ndarray
    cloud_before: np.ndarray
    cloud_after: np.ndarray

    @property
    def tile_id(self) -> str:
        return f"r{self.grid_row:03d}_c{self.grid_col:03d}"

    @property
    def height(self) -> int:
        return int(self.label.shape[0])

    @property
    def width(self) -> int:
        return int(self.label.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.before.shape[-1])
