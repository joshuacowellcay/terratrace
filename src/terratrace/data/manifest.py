"""Dataset manifest: the contract between the data pipeline and everything downstream.

The manifest is a JSON document that lists every tile, its split assignment, its on-disk
paths, and provenance (real vs synthetic, region, seed, dates). Training, evaluation, and
inference scripts read this file rather than re-globbing the filesystem so that every run
sees a deterministic snapshot of the dataset.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["real", "synthetic"]


class TileEntry(BaseModel):
    """One tile in the manifest."""

    tile_id: str
    split: Literal["train", "val", "test"]
    grid_row: int
    grid_col: int
    before_path: str
    after_path: str
    label_path: str
    cloud_before_path: str
    cloud_after_path: str
    cloud_fraction_before: float = Field(ge=0.0, le=1.0)
    cloud_fraction_after: float = Field(ge=0.0, le=1.0)
    loss_pixel_count: int = Field(ge=0)
    valid_pixel_count: int = Field(ge=0)


class DatasetManifest(BaseModel):
    """Top-level manifest schema."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    source: Source
    created_at: str
    region_name: str
    bbox: dict[str, float] | None = None
    before_window: dict[str, str] | None = None
    after_window: dict[str, str] | None = None
    tile_size: int
    bands: list[str]
    seed: int
    tiles: list[TileEntry]
    notes: str = ""

    def by_split(self, split: Literal["train", "val", "test"]) -> list[TileEntry]:
        return [t for t in self.tiles if t.split == split]

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        return cls.model_validate(json.loads(Path(path).read_text()))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
