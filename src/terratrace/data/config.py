"""Configuration models for the data pipeline.

Single source of truth for study region, time windows, tiling parameters, and split layout.
Loaded from YAML in `configs/pipeline.yaml` and consumed by the pipeline orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """WGS84 bounding box, west/south/east/north."""

    model_config = ConfigDict(frozen=True)

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.east <= self.west:
            raise ValueError(f"east ({self.east}) must be greater than west ({self.west})")
        if self.north <= self.south:
            raise ValueError(f"north ({self.north}) must be greater than south ({self.south})")
        return self

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south


class DateWindow(BaseModel):
    """Inclusive date range as ISO strings, e.g. "2019-06-01" to "2019-09-30"."""

    model_config = ConfigDict(frozen=True)

    start: str
    end: str


class SplitConfig(BaseModel):
    """Geographic block-split configuration.

    The study region (or synthetic grid) is divided into ``grid_size`` x ``grid_size`` blocks.
    Each block is assigned wholly to one of train/val/test. Block-level assignment is what
    prevents the spatial leakage that random tile-shuffling would introduce.
    """

    model_config = ConfigDict(frozen=True)

    grid_size: int = Field(ge=2, le=16, default=4)
    val_blocks: int = Field(ge=1, default=3)
    test_blocks: int = Field(ge=1, default=3)
    seed: int = 42

    @model_validator(mode="after")
    def _check_block_counts(self) -> SplitConfig:
        total = self.grid_size * self.grid_size
        if self.val_blocks + self.test_blocks >= total:
            raise ValueError(
                f"val_blocks ({self.val_blocks}) + test_blocks ({self.test_blocks}) "
                f"must leave at least one block for train (total={total})"
            )
        return self


class SyntheticConfig(BaseModel):
    """Synthetic-data generator configuration.

    The generator produces a single large "world" array, then slices it into
    ``grid_rows`` x ``grid_cols`` tiles of size ``tile_size``. Tile coordinates carry
    over so the geographic-block split sees spatially coherent neighbourhoods.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 1337
    tile_size: int = 128
    grid_rows: int = 12
    grid_cols: int = 12
    cloud_fraction: float = Field(ge=0.0, le=0.5, default=0.08)
    seasonal_shift: float = Field(ge=0.0, le=0.2, default=0.04)

    @property
    def num_tiles(self) -> int:
        return self.grid_rows * self.grid_cols


class RealDataConfig(BaseModel):
    """Real-data fetch limits; bounded so a misconfigured run cannot fill the disk."""

    model_config = ConfigDict(frozen=True)

    region_name: str = "rondonia"
    bbox: BoundingBox
    before_window: DateWindow
    after_window: DateWindow
    max_cloud_pct: float = Field(ge=0.0, le=100.0, default=20.0)
    max_tiles: int = Field(ge=4, le=2000, default=200)
    max_download_mb: int = Field(ge=10, le=8000, default=600)
    tile_size: int = 128
    bands: list[str] = Field(default_factory=lambda: ["B02", "B03", "B04", "B08"])


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration loaded from YAML."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path = Path("data/processed")
    sample_dir: Path = Path("data/sample")
    sample_tiles: int = Field(ge=2, le=32, default=6)
    fetch_mode: Literal["auto", "real", "synthetic"] = "auto"
    real: RealDataConfig
    synthetic: SyntheticConfig = SyntheticConfig()
    split: SplitConfig = SplitConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
