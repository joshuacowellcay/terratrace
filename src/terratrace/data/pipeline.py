"""Pipeline orchestrator: build a dataset, write it to disk, emit a manifest.

The orchestrator is the only place that decides whether to use real Sentinel-2 data or the
synthetic fallback. Everything downstream consumes the manifest and is source-agnostic.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from .cloud_mask import cloud_fraction
from .config import PipelineConfig
from .manifest import DatasetManifest, Source, TileEntry, now_iso
from .splits import Split, TileLocation, split_tiles
from .stac import fetch_real_tiles
from .synthetic import BAND_NAMES, generate_world, tile_world
from .tile import Tile

log = logging.getLogger(__name__)


def _save_tile_to_disk(
    tile: Tile,
    split: Split,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Save a tile's arrays under ``output_dir/{split}/{tile_id}/``. Returns the five paths."""
    tile_dir = output_dir / split / tile.tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "before": tile_dir / "before.npz",
        "after": tile_dir / "after.npz",
        "label": tile_dir / "label.npz",
        "cloud_before": tile_dir / "cloud_before.npz",
        "cloud_after": tile_dir / "cloud_after.npz",
    }
    np.savez_compressed(paths["before"], arr=tile.before.astype(np.float32))
    np.savez_compressed(paths["after"], arr=tile.after.astype(np.float32))
    np.savez_compressed(paths["label"], arr=tile.label.astype(np.uint8))
    np.savez_compressed(paths["cloud_before"], arr=tile.cloud_before.astype(np.uint8))
    np.savez_compressed(paths["cloud_after"], arr=tile.cloud_after.astype(np.uint8))
    return (
        paths["before"],
        paths["after"],
        paths["label"],
        paths["cloud_before"],
        paths["cloud_after"],
    )


def _build_tiles(config: PipelineConfig) -> tuple[list[Tile], Source]:
    """Decide source, produce tiles. Returns (tiles, source) where source ∈ {'real','synthetic'}."""
    if config.fetch_mode == "synthetic":
        log.info("fetch_mode=synthetic: generating procedural dataset.")
        return _synthesise(config), "synthetic"

    if config.fetch_mode == "real":
        log.info("fetch_mode=real: requiring real Sentinel-2 data.")
        tiles = fetch_real_tiles(config.real)
        if tiles is None:
            raise RuntimeError(
                "fetch_mode=real but real-data fetch failed. Set fetch_mode=auto to enable "
                "the synthetic fallback, or check network / Planetary Computer status."
            )
        return tiles, "real"

    # auto: try real, fall back on any failure.
    log.info("fetch_mode=auto: trying real-data path first.")
    tiles = fetch_real_tiles(config.real)
    if tiles is not None and len(tiles) >= 16:
        return tiles, "real"
    log.warning("Real-data path unavailable or returned too few tiles; using synthetic.")
    return _synthesise(config), "synthetic"


def _synthesise(config: PipelineConfig) -> list[Tile]:
    world = generate_world(config.synthetic)
    return tile_world(world, config.synthetic)


def build_dataset(config: PipelineConfig) -> DatasetManifest:
    """Build the full dataset on disk and return the manifest.

    Steps:
      1. Generate or fetch raw tiles.
      2. Split the tile grid into train/val/test by spatial blocks.
      3. Write each tile to ``output_dir/{split}/{tile_id}/`` as compressed npz.
      4. Write the manifest to ``output_dir/manifest.json``.
      5. Write a tiny ``data/sample/`` copy with a handful of train + test tiles.
    """
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles, source = _build_tiles(config)
    if not tiles:
        raise RuntimeError("pipeline produced zero tiles")

    locations = [
        TileLocation(tile_id=t.tile_id, grid_row=t.grid_row, grid_col=t.grid_col) for t in tiles
    ]
    splits = split_tiles(
        locations,
        grid_size=config.split.grid_size,
        val_blocks=config.split.val_blocks,
        test_blocks=config.split.test_blocks,
        seed=config.split.seed,
    )

    tile_to_split: dict[str, Split] = {}
    for split_name, items in splits.items():
        for loc in items:
            tile_to_split[loc.tile_id] = split_name

    # Determine bands list. Synthetic uses the canonical (R, G, B, NIR) order. Real path uses
    # the configured `bands` list, which already corresponds to the band stack.
    bands = list(BAND_NAMES) if source == "synthetic" else list(config.real.bands)
    tile_size = config.synthetic.tile_size if source == "synthetic" else config.real.tile_size

    entries: list[TileEntry] = []
    for tile in tiles:
        split = tile_to_split[tile.tile_id]
        before_p, after_p, label_p, cb_p, ca_p = _save_tile_to_disk(tile, split, output_dir)
        # Paths in the manifest are relative to the manifest's own directory, so a manifest
        # at `data/processed/manifest.json` references `train/r000_c000/before.npz` etc. The
        # Dataset class joins them with the manifest's parent path.
        entries.append(
            TileEntry(
                tile_id=tile.tile_id,
                split=split,
                grid_row=tile.grid_row,
                grid_col=tile.grid_col,
                before_path=str(before_p.relative_to(output_dir)),
                after_path=str(after_p.relative_to(output_dir)),
                label_path=str(label_p.relative_to(output_dir)),
                cloud_before_path=str(cb_p.relative_to(output_dir)),
                cloud_after_path=str(ca_p.relative_to(output_dir)),
                cloud_fraction_before=cloud_fraction(tile.cloud_before),
                cloud_fraction_after=cloud_fraction(tile.cloud_after),
                loss_pixel_count=int(tile.label.astype(bool).sum()),
                valid_pixel_count=int(
                    ((~tile.cloud_before.astype(bool)) & (~tile.cloud_after.astype(bool))).sum()
                ),
            )
        )

    bbox_dict = config.real.bbox.model_dump() if source == "real" else None
    before_window = config.real.before_window.model_dump() if source == "real" else None
    after_window = config.real.after_window.model_dump() if source == "real" else None

    notes = (
        "Synthetic data was used because the real Sentinel-2 + Hansen GFC fetch was "
        "unavailable, disabled by TERRATRACE_FORCE_SYNTHETIC, or produced too few tiles. "
        "Re-run with network access and `fetch_mode: real` to use real imagery."
        if source == "synthetic"
        else "Real Sentinel-2 L2A imagery, Hansen GFC {} labels.".format("v1.11")
    )

    manifest = DatasetManifest(
        version=1,
        source=source,
        created_at=now_iso(),
        region_name=config.real.region_name,
        bbox=bbox_dict,
        before_window=before_window,
        after_window=after_window,
        tile_size=tile_size,
        bands=bands,
        seed=(config.synthetic.seed if source == "synthetic" else config.split.seed),
        tiles=entries,
        notes=notes,
    )
    manifest.write(output_dir / "manifest.json")
    log.info(
        "Wrote manifest with %d tiles (train=%d, val=%d, test=%d) from source=%s.",
        len(entries),
        len(manifest.by_split("train")),
        len(manifest.by_split("val")),
        len(manifest.by_split("test")),
        source,
    )

    _materialise_sample(config, manifest, output_dir)
    return manifest


def _materialise_sample(
    config: PipelineConfig, manifest: DatasetManifest, output_dir: Path
) -> None:
    """Copy a small, deterministic subset of tiles + the manifest into ``config.sample_dir``.

    This is what gets committed to the repo so a fresh clone can run inference and tests
    without any data build.
    """
    sample_dir = Path(config.sample_dir)
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Pick the first N tiles from the test split (sorted by tile_id for determinism). This
    # gives the sample data a clean spatial origin and never leaks into the test split for
    # any future evaluation that re-uses these tiles.
    test_tiles = sorted(manifest.by_split("test"), key=lambda t: t.tile_id)
    chosen = test_tiles[: config.sample_tiles]
    if not chosen:
        log.warning("No test-split tiles to sample; skipping data/sample materialisation.")
        return

    sample_entries: list[TileEntry] = []
    for entry in chosen:
        src_dir = output_dir / entry.split / entry.tile_id
        dst_dir = sample_dir / entry.tile_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "before.npz",
            "after.npz",
            "label.npz",
            "cloud_before.npz",
            "cloud_after.npz",
        ):
            shutil.copy2(src_dir / name, dst_dir / name)
        sample_entries.append(
            entry.model_copy(
                update={
                    "split": "test",
                    "before_path": str((dst_dir / "before.npz").relative_to(sample_dir)),
                    "after_path": str((dst_dir / "after.npz").relative_to(sample_dir)),
                    "label_path": str((dst_dir / "label.npz").relative_to(sample_dir)),
                    "cloud_before_path": str(
                        (dst_dir / "cloud_before.npz").relative_to(sample_dir)
                    ),
                    "cloud_after_path": str((dst_dir / "cloud_after.npz").relative_to(sample_dir)),
                }
            )
        )

    sample_manifest = manifest.model_copy(
        update={
            "tiles": sample_entries,
            "notes": (
                "Committed sample subset: a handful of tiles drawn from the test split of the "
                "build that produced data/processed/manifest.json. Used by inference tests so a "
                "fresh clone needs no data build to run them."
            ),
        }
    )
    sample_manifest.write(sample_dir / "manifest.json")
    log.info("Wrote sample (%d tiles) to %s.", len(sample_entries), sample_dir)
