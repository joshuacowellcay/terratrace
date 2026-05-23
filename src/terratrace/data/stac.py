"""Fetch Sentinel-2 L2A imagery and Hansen GFC labels from the Microsoft Planetary Computer.

This is the real-data path. The pipeline tries it first; any failure (no network, item not
found, repeated HTTP errors, slow download, missing band, etc.) is converted to a ``None``
return so the orchestrator can fall back to the synthetic generator. The contract: this
function either returns a fully usable ``list[Tile]`` or returns ``None``. It never half-fails.

Hansen Global Forest Change is **not** indexed by the Planetary Computer STAC API, so we fetch
its ``lossyear`` granule directly from the public Google Cloud Storage bucket published by the
University of Maryland / Google Earth Engine partners. The version is pinned for
reproducibility.

NOTE: pystac-client and planetary-computer are imported lazily so that the synthetic-only
test path doesn't import any networking stack.
"""

from __future__ import annotations

import importlib
import logging
import os
import socket
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np

from .cloud_mask import cloud_mask_from_scl
from .config import RealDataConfig
from .tile import Tile
from .tiling import compute_tile_indices, extract_tile

if TYPE_CHECKING:
    import rasterio  # noqa: F401

log = logging.getLogger(__name__)

HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_GCS_TEMPLATE = (
    "https://storage.googleapis.com/earthenginepartners-hansen/"
    f"{HANSEN_VERSION}/Hansen_{HANSEN_VERSION}_lossyear_{{tile}}.tif"
)
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION = "sentinel-2-l2a"
DEFAULT_TIMEOUT_S = 30
MAX_RETRIES = 3


def _hansen_tile_id(west: float, north: float) -> str:
    """Build the Hansen granule identifier for the 10° tile covering (west, north).

    Hansen tiles are named e.g. "10S_070W" — the latitude/longitude of the **upper-left
    corner**, rounded **up** in absolute value to the next multiple of 10. The northern
    hemisphere uses ``N``, southern ``S``; eastern ``E``, western ``W``.
    """
    # Snap northern edge UP to the next multiple of 10.
    if north >= 0:
        lat_int = int(np.ceil(north / 10.0)) * 10
        lat_hemi = "N"
    else:
        lat_int = int(np.ceil(abs(north) / 10.0)) * 10
        lat_hemi = "S"
    if west >= 0:
        lon_int = int(np.floor(west / 10.0)) * 10
        lon_hemi = "E"
    else:
        lon_int = int(np.floor(abs(west) / 10.0) + 10) * 10
        # the western-edge tile UL is `west` rounded DOWN (more negative) to multiple of 10
        if west % 10 == 0:
            lon_int = int(abs(west))
        lon_hemi = "W"
    return f"{lat_int:02d}{lat_hemi}_{lon_int:03d}{lon_hemi}"


def _can_reach_internet(host: str = "planetarycomputer.microsoft.com", port: int = 443) -> bool:
    try:
        with socket.create_connection((host, port), timeout=4):
            return True
    except OSError:
        return False


def fetch_real_tiles(config: RealDataConfig) -> list[Tile] | None:
    """Attempt to build a tile list from real Sentinel-2 + Hansen GFC data.

    Returns ``None`` if anything goes wrong. The caller is responsible for falling back to
    the synthetic generator.
    """
    if os.environ.get("TERRATRACE_FORCE_SYNTHETIC", "").strip() not in ("", "0", "false", "False"):
        log.info("TERRATRACE_FORCE_SYNTHETIC set; skipping real-data fetch.")
        return None

    if not _can_reach_internet():
        log.warning("No outbound connectivity to Planetary Computer; falling back to synthetic.")
        return None

    try:
        # Lazy imports so a synthetic-only environment doesn't need rasterio/pystac etc.
        import planetary_computer as pc
        import pystac_client

        importlib.import_module("rasterio")  # availability check for later use in helpers
    except ImportError as exc:
        log.warning("Real-data deps unavailable (%s); falling back to synthetic.", exc)
        return None

    try:
        catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC, modifier=pc.sign_inplace)
        bbox = (
            config.bbox.west,
            config.bbox.south,
            config.bbox.east,
            config.bbox.north,
        )

        def _least_cloudy_item(window_start: str, window_end: str):
            search = catalog.search(
                collections=[STAC_COLLECTION],
                bbox=bbox,
                datetime=f"{window_start}/{window_end}",
                query={"eo:cloud_cover": {"lt": config.max_cloud_pct}},
                max_items=20,
            )
            items = list(search.items())
            if not items:
                return None
            return min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))

        before_item = _retry(
            lambda: _least_cloudy_item(config.before_window.start, config.before_window.end)
        )
        after_item = _retry(
            lambda: _least_cloudy_item(config.after_window.start, config.after_window.end)
        )
        if before_item is None or after_item is None:
            log.warning(
                "No Sentinel-2 scenes matching the bbox/cloud filter; falling back to synthetic."
            )
            return None

        # Read just the requested bands + SCL for each timestamp.
        before_stack, scl_before, _transform = _read_bands(
            before_item, config.bands, bbox, target_size_hint=config.tile_size * 32
        )
        if before_stack is None:
            return None
        after_stack, scl_after, _ = _read_bands(
            after_item, config.bands, bbox, target_size_hint=config.tile_size * 32
        )
        if after_stack is None:
            return None

        if before_stack.shape != after_stack.shape:
            # Align by cropping to the smaller dimensions — STAC items can disagree by ~1px.
            h = min(before_stack.shape[0], after_stack.shape[0])
            w = min(before_stack.shape[1], after_stack.shape[1])
            before_stack = before_stack[:h, :w]
            after_stack = after_stack[:h, :w]
            scl_before = scl_before[:h, :w]
            scl_after = scl_after[:h, :w]

        # Hansen GFC label: lossyear in [1, 23] gives the year (2001 + n). Filter for losses
        # that fall strictly between the before and after windows.
        before_year = int(config.before_window.start[:4])
        after_year = int(config.after_window.end[:4])
        label = _build_hansen_label(
            west=config.bbox.west,
            south=config.bbox.south,
            east=config.bbox.east,
            north=config.bbox.north,
            target_shape=before_stack.shape[:2],
            min_loss_year=before_year + 1,
            max_loss_year=after_year,
        )
        if label is None:
            return None

        cloud_before_full = cloud_mask_from_scl(scl_before)
        cloud_after_full = cloud_mask_from_scl(scl_after)

        # Tile everything.
        tiles: list[Tile] = []
        indices = compute_tile_indices(
            raster_height=before_stack.shape[0],
            raster_width=before_stack.shape[1],
            tile_size=config.tile_size,
        )
        if not indices:
            log.warning("Real raster smaller than tile_size; falling back to synthetic.")
            return None
        for idx in indices[: config.max_tiles]:
            before_tile = extract_tile(before_stack, idx)
            after_tile = extract_tile(after_stack, idx)
            label_tile = extract_tile(label, idx)
            cb_tile = extract_tile(cloud_before_full, idx)
            ca_tile = extract_tile(cloud_after_full, idx)
            tiles.append(
                Tile(
                    grid_row=idx.grid_row,
                    grid_col=idx.grid_col,
                    before=before_tile,
                    after=after_tile,
                    label=label_tile,
                    cloud_before=cb_tile,
                    cloud_after=ca_tile,
                )
            )
        log.info(
            "Fetched %d tiles from real Sentinel-2 (before=%s, after=%s).",
            len(tiles),
            before_item.id,
            after_item.id,
        )
        return tiles

    except Exception as exc:  # broad: ANY failure should produce graceful fallback
        log.warning(
            "Real-data fetch failed (%s: %s); falling back to synthetic.", type(exc).__name__, exc
        )
        return None


def _retry(fn, *, tries: int = MAX_RETRIES):
    last_exc: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            log.warning("Retry %d/%d after %s: %s", attempt, tries, type(exc).__name__, exc)
    if last_exc:
        raise last_exc
    return None


def _read_bands(
    item, bands: list[str], bbox: tuple[float, float, float, float], *, target_size_hint: int
) -> tuple[np.ndarray, np.ndarray, object] | tuple[None, None, None]:
    """Read the requested bands + SCL for a STAC item, cropped to the bbox.

    The STAC item's assets are COGs; we open each, window-read the bbox, and stack channel-last.
    Reflectance is divided by 10_000 to land in roughly [0, 1].
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError:
        return None, None, None

    arrays: list[np.ndarray] = []
    transform_ref = None
    for band in bands:
        asset = item.assets.get(band)
        if asset is None:
            log.warning("Item %s missing band %s; aborting real fetch.", item.id, band)
            return None, None, None
        with rasterio.open(asset.href) as ds:
            window_bbox = transform_bounds("EPSG:4326", ds.crs, *bbox, densify_pts=21)
            window = (
                from_bounds(*window_bbox, transform=ds.transform).round_offsets().round_lengths()
            )
            data = ds.read(1, window=window, out_dtype="float32") / 10_000.0
            arrays.append(np.clip(data, 0.0, 1.5))
            if transform_ref is None:
                transform_ref = (ds.transform, ds.crs)

    scl_asset = item.assets.get("SCL")
    scl_data: np.ndarray
    if scl_asset is None:
        # If SCL is missing the imagery is unusable for cloud masking.
        log.warning("Item %s has no SCL; aborting real fetch.", item.id)
        return None, None, None
    with rasterio.open(scl_asset.href) as ds:
        window_bbox = transform_bounds("EPSG:4326", ds.crs, *bbox, densify_pts=21)
        window = from_bounds(*window_bbox, transform=ds.transform).round_offsets().round_lengths()
        scl_data = ds.read(1, window=window, out_dtype="int32")
        # SCL is typically at 20 m; the bands above at 10 m. We resample SCL by nearest-neighbour
        # to match the first band's shape.
        if scl_data.shape != arrays[0].shape:
            with suppress(Exception):
                from PIL import Image

                pil = Image.fromarray(scl_data.astype(np.int32), mode="I")
                pil = pil.resize((arrays[0].shape[1], arrays[0].shape[0]), Image.NEAREST)
                scl_data = np.array(pil, dtype=np.int32)

    stack = np.stack(arrays, axis=-1).astype(np.float32)
    return stack, scl_data, transform_ref


def _build_hansen_label(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    target_shape: tuple[int, int],
    min_loss_year: int,
    max_loss_year: int,
) -> np.ndarray | None:
    """Download and crop the Hansen GFC lossyear granule covering the bbox, then resample.

    Output: a (H, W) uint8 mask where 1 = a Hansen "loss" event in the requested year window.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError:
        return None

    granule_id = _hansen_tile_id(west=west, north=north)
    url = HANSEN_GCS_TEMPLATE.format(tile=granule_id)
    try:
        with rasterio.open(url) as ds:
            window_bbox = transform_bounds(
                "EPSG:4326", ds.crs, west, south, east, north, densify_pts=21
            )
            window = (
                from_bounds(*window_bbox, transform=ds.transform).round_offsets().round_lengths()
            )
            data = ds.read(1, window=window, out_dtype="int32")
    except Exception as exc:
        log.warning("Hansen download failed for %s (%s); falling back.", url, exc)
        return None

    # `data` values are years since 2000; e.g. 19 = 2019.
    min_n = min_loss_year - 2000
    max_n = max_loss_year - 2000
    label = ((data >= min_n) & (data <= max_n)).astype(np.uint8)

    if label.shape != target_shape:
        with suppress(Exception):
            from PIL import Image

            pil = Image.fromarray(label, mode="L")
            pil = pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
            label = np.array(pil, dtype=np.uint8)
    return label
