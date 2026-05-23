"""End-to-end pipeline tests (synthetic path, deterministic, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from terratrace.data.config import (
    BoundingBox,
    DateWindow,
    PipelineConfig,
    RealDataConfig,
    SplitConfig,
    SyntheticConfig,
)
from terratrace.data.dataset import ChangeDetectionDataset
from terratrace.data.manifest import DatasetManifest
from terratrace.data.pipeline import build_dataset
from terratrace.data.splits import TileLocation, assert_no_spatial_overlap


def _tmp_config(tmp_path: Path) -> PipelineConfig:
    """A small, fast pipeline config that always uses the synthetic generator."""
    return PipelineConfig(
        output_dir=tmp_path / "processed",
        sample_dir=tmp_path / "sample",
        sample_tiles=4,
        fetch_mode="synthetic",
        real=RealDataConfig(
            region_name="testregion",
            bbox=BoundingBox(west=-1, south=-1, east=1, north=1),
            before_window=DateWindow(start="2020-01-01", end="2020-03-31"),
            after_window=DateWindow(start="2022-01-01", end="2022-03-31"),
        ),
        synthetic=SyntheticConfig(
            seed=99, tile_size=32, grid_rows=4, grid_cols=4, cloud_fraction=0.05
        ),
        split=SplitConfig(grid_size=2, val_blocks=1, test_blocks=1, seed=0),
    )


def test_pipeline_writes_manifest_and_tiles(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)
    manifest = build_dataset(cfg)

    assert manifest.source == "synthetic"
    assert len(manifest.tiles) == cfg.synthetic.num_tiles == 16

    # Manifest file exists and is loadable.
    loaded = DatasetManifest.read(cfg.output_dir / "manifest.json")
    assert loaded.source == "synthetic"
    assert {t.tile_id for t in loaded.tiles} == {t.tile_id for t in manifest.tiles}

    # Every tile's npz files were written and are readable.
    for entry in manifest.tiles:
        for key in (
            entry.before_path,
            entry.after_path,
            entry.label_path,
            entry.cloud_before_path,
            entry.cloud_after_path,
        ):
            assert (cfg.output_dir / key).is_file()


def test_pipeline_split_has_no_spatial_overlap(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)
    manifest = build_dataset(cfg)

    # Convert manifest entries back into TileLocation per split and run the invariant check.
    splits: dict[str, list[TileLocation]] = {"train": [], "val": [], "test": []}
    for entry in manifest.tiles:
        splits[entry.split].append(
            TileLocation(tile_id=entry.tile_id, grid_row=entry.grid_row, grid_col=entry.grid_col)
        )
    assert_no_spatial_overlap(splits)

    # All three splits must be non-empty.
    for split in ("train", "val", "test"):
        assert splits[split], f"split '{split}' is empty"


def test_pipeline_materialises_sample(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)
    manifest = build_dataset(cfg)

    sample_manifest = DatasetManifest.read(cfg.sample_dir / "manifest.json")
    assert len(sample_manifest.tiles) <= cfg.sample_tiles
    assert len(sample_manifest.tiles) > 0
    assert all(t.split == "test" for t in sample_manifest.tiles)

    # Sample tile arrays load cleanly.
    for entry in sample_manifest.tiles:
        before = np.load(cfg.sample_dir / entry.before_path)["arr"]
        assert before.shape[-1] == len(manifest.bands)


def test_dataset_loader_returns_correct_shapes(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)
    manifest = build_dataset(cfg)

    ds = ChangeDetectionDataset.from_manifest_path(cfg.output_dir / "manifest.json", "train")
    sample = ds[0]
    c = len(manifest.bands)
    h = w = cfg.synthetic.tile_size

    assert sample["before"].shape == (c, h, w)
    assert sample["after"].shape == (c, h, w)
    assert sample["label"].shape == (h, w)
    assert sample["weights"].shape == (h, w)
    assert isinstance(sample["tile_id"], str)
