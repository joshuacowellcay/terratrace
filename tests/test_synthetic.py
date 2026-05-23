"""Tests for the procedural synthetic data generator."""

from __future__ import annotations

import numpy as np

from terratrace.data.config import SyntheticConfig
from terratrace.data.synthetic import (
    BAND_NAMES,
    N_BANDS,
    generate_world,
    tile_world,
)


def _small_config(**overrides) -> SyntheticConfig:
    base = dict(seed=1337, tile_size=32, grid_rows=4, grid_cols=4, cloud_fraction=0.1)
    base.update(overrides)
    return SyntheticConfig(**base)


def test_world_has_expected_shape() -> None:
    cfg = _small_config()
    world = generate_world(cfg)
    expected_h = cfg.grid_rows * cfg.tile_size
    expected_w = cfg.grid_cols * cfg.tile_size
    assert world.before.shape == (expected_h, expected_w, N_BANDS)
    assert world.after.shape == (expected_h, expected_w, N_BANDS)
    assert world.label.shape == (expected_h, expected_w)
    assert world.cloud_before.shape == (expected_h, expected_w)
    assert world.cloud_after.shape == (expected_h, expected_w)


def test_world_is_deterministic() -> None:
    cfg = _small_config()
    a = generate_world(cfg)
    b = generate_world(cfg)
    np.testing.assert_array_equal(a.before, b.before)
    np.testing.assert_array_equal(a.after, b.after)
    np.testing.assert_array_equal(a.label, b.label)


def test_different_seeds_produce_different_worlds() -> None:
    a = generate_world(_small_config(seed=1))
    b = generate_world(_small_config(seed=2))
    assert not np.array_equal(a.label, b.label)


def test_labels_are_binary_and_only_in_forest() -> None:
    cfg = _small_config(grid_rows=6, grid_cols=6)
    world = generate_world(cfg)
    # Label is uint8 with values in {0, 1}.
    assert world.label.dtype == np.uint8
    unique = set(np.unique(world.label).tolist())
    assert unique.issubset({0, 1})
    # Every loss pixel must have been forest in the "before" classification (class 1).
    loss = world.label.astype(bool)
    assert (world.classes_before[loss] == 1).all()
    # And must no longer be forest in "after".
    assert (world.classes_after[loss] != 1).all()


def test_loss_fraction_is_nontrivial_but_minority() -> None:
    """The synthetic task should be class-imbalanced (mirrors real forest-loss prevalence)."""
    cfg = _small_config(grid_rows=8, grid_cols=8)
    world = generate_world(cfg)
    loss_fraction = float(world.label.mean())
    assert 0.005 < loss_fraction < 0.4, f"loss fraction {loss_fraction:.3f} outside expected range"


def test_clouds_are_present_and_bounded() -> None:
    cfg = _small_config(cloud_fraction=0.15)
    world = generate_world(cfg)
    frac = float(world.cloud_before.mean())
    # Threshold-based generator is approximate but should land in the right ballpark.
    assert 0.05 < frac < 0.30


def test_tile_world_produces_expected_count_and_shapes() -> None:
    cfg = _small_config(grid_rows=3, grid_cols=5, tile_size=16)
    world = generate_world(cfg)
    tiles = tile_world(world, cfg)
    assert len(tiles) == cfg.num_tiles == 15
    for tile in tiles:
        assert tile.before.shape == (cfg.tile_size, cfg.tile_size, N_BANDS)
        assert tile.label.shape == (cfg.tile_size, cfg.tile_size)
    # Tile IDs are unique.
    assert len({t.tile_id for t in tiles}) == len(tiles)


def test_band_names_match_channel_count() -> None:
    assert len(BAND_NAMES) == N_BANDS
