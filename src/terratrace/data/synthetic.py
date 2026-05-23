"""Procedural synthetic data generator.

Produces before/after multi-band image pairs with a known forest-loss ground-truth mask.
The goal is not photorealism but to give every downstream component (loss, training loop,
baseline, evaluation, inference) the same data shape and statistical structure as the real
Sentinel-2 path, so nothing in the pipeline secretly depends on real data being available.

Land-cover classes:
    0 = water           (low NIR, low overall brightness)
    1 = forest          (high NIR, low red, high NDVI)
    2 = bare / cleared  (moderate NIR, moderate red, low NDVI)

The "after" image keeps the class-2 areas, removes forest in randomly shaped polygons
(which become bare, the deforestation signal), applies a small global seasonal brightness
shift, and re-rolls cloud cover. The forest-loss label is exactly
``(forest_before & ~forest_after)``.

All randomness is driven by a single ``numpy.random.Generator`` seeded from
``SyntheticConfig.seed`` so the same config always yields the same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SyntheticConfig
from .tile import Tile

# Index conventions: 0 R, 1 G, 2 B, 3 NIR. Reflectance-like in [0, 1].
BAND_NAMES: tuple[str, ...] = ("red", "green", "blue", "nir")
N_BANDS = len(BAND_NAMES)

# Per-class mean spectra. Values picked so that NDVI separates forest from non-forest, but
# bare and water still have non-trivial overlap; the task is meant to reward learning
# spatial context, not just thresholding a single band.
_CLASS_MEANS: dict[int, np.ndarray] = {
    0: np.array([0.04, 0.06, 0.10, 0.02], dtype=np.float32),  # water
    1: np.array([0.05, 0.10, 0.05, 0.42], dtype=np.float32),  # forest
    2: np.array([0.20, 0.17, 0.14, 0.24], dtype=np.float32),  # bare
}
_CLOUD_SPECTRUM = np.array([0.85, 0.85, 0.85, 0.90], dtype=np.float32)
_SHADOW_DARKENING = 0.35  # multiplicative factor applied to pixels under cloud shadow


@dataclass(frozen=True)
class SyntheticWorld:
    """A complete synthetic landscape, before tiling.

    Shapes are (H, W, C) for images and (H, W) for masks. ``height``/``width`` are
    ``grid_rows * tile_size`` / ``grid_cols * tile_size``.
    """

    before: np.ndarray  # float32 (H, W, 4)
    after: np.ndarray  # float32 (H, W, 4)
    label: np.ndarray  # uint8 (H, W), 1 = forest-loss
    cloud_before: np.ndarray  # uint8 (H, W), 1 = cloudy / invalid
    cloud_after: np.ndarray  # uint8 (H, W), 1 = cloudy / invalid
    classes_before: np.ndarray  # uint8 (H, W), land-cover class index
    classes_after: np.ndarray  # uint8 (H, W)


def _smoothed_noise(rng: np.random.Generator, shape: tuple[int, int], scale: int) -> np.ndarray:
    """Generate a 2D field with characteristic correlation length ``scale``.

    Implementation: draw a coarse white-noise grid sized so its block-expansion fully covers
    ``shape``, expand by nearest-neighbour repeat, and crop. The coarse grid uses *ceil*
    division so we never under-allocate when ``scale`` exceeds an image dimension.
    """
    h, w = shape
    coarse_h = max(1, -(-h // scale))  # ceil division
    coarse_w = max(1, -(-w // scale))
    coarse = rng.standard_normal((coarse_h, coarse_w)).astype(np.float32)
    expanded = np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)
    return expanded[:h, :w]


def _multioctave(
    rng: np.random.Generator,
    shape: tuple[int, int],
    octaves: tuple[int, ...] = (32, 16, 8, 4),
) -> np.ndarray:
    """Sum of multiple smoothed-noise octaves at decreasing weight."""
    field = np.zeros(shape, dtype=np.float32)
    weight = 1.0
    for scale in octaves:
        field += weight * _smoothed_noise(rng, shape, scale)
        weight *= 0.5
    # Normalise to roughly [0, 1].
    field -= field.min()
    if field.max() > 0:
        field /= field.max()
    return field


def _classify(elevation: np.ndarray, moisture: np.ndarray) -> np.ndarray:
    """Derive a discrete class map from two noise fields.

    Low elevation + high moisture gives water. The remaining land splits into forest where
    "forest potential" (a mix of moisture and mid-elevation) is high, and bare otherwise.
    """
    classes = np.full(elevation.shape, fill_value=2, dtype=np.uint8)  # default: bare
    water_mask = (elevation < 0.32) & (moisture > 0.55)
    forest_potential = moisture * 0.7 + (1.0 - np.abs(elevation - 0.55) * 1.8) * 0.3
    forest_mask = (~water_mask) & (forest_potential > 0.45)
    classes[water_mask] = 0
    classes[forest_mask] = 1
    return classes


def _render(
    classes: np.ndarray,
    cloud_mask: np.ndarray,
    shadow_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    per_pixel_noise: float = 0.025,
    brightness_shift: float = 0.0,
) -> np.ndarray:
    """Render a class map to a multi-band image, then apply clouds + shadows + noise."""
    h, w = classes.shape
    img = np.empty((h, w, N_BANDS), dtype=np.float32)
    for cls_idx, mean in _CLASS_MEANS.items():
        sel = classes == cls_idx
        if not sel.any():
            continue
        img[sel] = mean

    # Per-pixel Gaussian noise, kept small so classes remain separable.
    img += rng.standard_normal(img.shape).astype(np.float32) * per_pixel_noise

    # Seasonal brightness shift, multiplicative on visible bands and small on NIR.
    if brightness_shift != 0.0:
        img[..., :3] *= 1.0 + brightness_shift
        img[..., 3] *= 1.0 + brightness_shift * 0.3

    # Apply shadow first (clouds overwrite shadow where the two overlap).
    if shadow_mask.any():
        img[shadow_mask] *= _SHADOW_DARKENING

    if cloud_mask.any():
        img[cloud_mask] = (
            _CLOUD_SPECTRUM
            + rng.standard_normal((cloud_mask.sum(), N_BANDS)).astype(np.float32) * 0.02
        )

    return np.clip(img, 0.0, 1.0)


def _generate_clouds(
    rng: np.random.Generator, shape: tuple[int, int], fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cloud_mask, shadow_mask) as uint8 arrays of the same shape.

    Clouds are blob-shaped (high quantile of a smoothed noise field). Shadows are offset
    copies of the clouds, shifted by a small fixed displacement to mimic sun-angle
    projection. ``fraction`` is the target cloudy fraction (approximate).
    """
    if fraction <= 0.0:
        empty = np.zeros(shape, dtype=bool)
        return empty.astype(np.uint8), empty.astype(np.uint8)

    cloud_field = _multioctave(rng, shape, octaves=(48, 24, 12))
    threshold = float(np.quantile(cloud_field, 1.0 - fraction))
    cloud_mask = cloud_field > threshold

    # Shadows are the cloud mask shifted by ~6% of the image diagonal.
    h, w = shape
    dy, dx = max(1, h // 18), max(1, w // 18)
    shadow_mask = np.zeros(shape, dtype=bool)
    shadow_mask[dy:, dx:] = cloud_mask[:-dy, :-dx]
    # Strip shadow pixels that fall on top of their own cloud.
    shadow_mask &= ~cloud_mask

    return cloud_mask.astype(np.uint8), shadow_mask.astype(np.uint8)


def _random_deforestation(
    rng: np.random.Generator,
    forest_mask: np.ndarray,
    *,
    n_polygons: int = 14,
    min_radius_frac: float = 0.02,
    max_radius_frac: float = 0.08,
) -> np.ndarray:
    """Build a boolean mask of "deforested" pixels: random ellipses constrained to forest.

    Returned mask has the same shape as ``forest_mask``. Outside-forest pixels are never
    set, because you cannot deforest non-forest.
    """
    h, w = forest_mask.shape
    diag = float(np.hypot(h, w))
    deforested = np.zeros_like(forest_mask, dtype=bool)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(n_polygons):
        cy = float(rng.uniform(0, h))
        cx = float(rng.uniform(0, w))
        a = float(rng.uniform(min_radius_frac, max_radius_frac)) * diag
        b = float(rng.uniform(min_radius_frac, max_radius_frac)) * diag
        # Random axis rotation
        theta = float(rng.uniform(0, np.pi))
        ct, st = np.cos(theta), np.sin(theta)
        dy = yy - cy
        dx = xx - cx
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        inside = (u * u) / (a * a) + (v * v) / (b * b) < 1.0
        deforested |= inside

    return deforested & forest_mask.astype(bool)


def generate_world(config: SyntheticConfig) -> SyntheticWorld:
    """Generate a complete synthetic landscape for the given config.

    The returned object holds the full world arrays; callers slice them into tiles via
    :func:`tile_world`.
    """
    rng = np.random.default_rng(config.seed)
    h = config.grid_rows * config.tile_size
    w = config.grid_cols * config.tile_size

    elevation = _multioctave(rng, (h, w), octaves=(64, 32, 16, 8))
    moisture = _multioctave(rng, (h, w), octaves=(48, 24, 12, 6))

    classes_before = _classify(elevation, moisture)

    forest_before_mask = classes_before == 1
    deforested = _random_deforestation(rng, forest_before_mask)
    classes_after = classes_before.copy()
    classes_after[deforested] = 2  # forest -> bare

    cloud_before, shadow_before = _generate_clouds(rng, (h, w), config.cloud_fraction)
    cloud_after, shadow_after = _generate_clouds(rng, (h, w), config.cloud_fraction)

    season_shift = float(rng.uniform(-config.seasonal_shift, config.seasonal_shift))

    before = _render(
        classes_before,
        cloud_before.astype(bool),
        shadow_before.astype(bool),
        rng=rng,
    )
    after = _render(
        classes_after,
        cloud_after.astype(bool),
        shadow_after.astype(bool),
        rng=rng,
        brightness_shift=season_shift,
    )

    label = (forest_before_mask & (classes_after != 1)).astype(np.uint8)

    return SyntheticWorld(
        before=before,
        after=after,
        label=label,
        cloud_before=cloud_before,
        cloud_after=cloud_after,
        classes_before=classes_before,
        classes_after=classes_after,
    )


def tile_world(world: SyntheticWorld, config: SyntheticConfig) -> list[Tile]:
    """Slice a SyntheticWorld into tiles preserving grid coordinates.

    Tiles are emitted in row-major order; the grid coordinates are what the geographic
    splitter uses to assign blocks.
    """
    tiles: list[Tile] = []
    ts = config.tile_size
    for r in range(config.grid_rows):
        for c in range(config.grid_cols):
            ys, xs = r * ts, c * ts
            ye, xe = ys + ts, xs + ts
            tiles.append(
                Tile(
                    grid_row=r,
                    grid_col=c,
                    before=world.before[ys:ye, xs:xe].copy(),
                    after=world.after[ys:ye, xs:xe].copy(),
                    label=world.label[ys:ye, xs:xe].copy(),
                    cloud_before=world.cloud_before[ys:ye, xs:xe].copy(),
                    cloud_after=world.cloud_after[ys:ye, xs:xe].copy(),
                )
            )
    return tiles
