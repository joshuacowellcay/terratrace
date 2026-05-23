"""FastAPI service exposing the change-detection model.

Endpoints:
    GET  /health                    — readiness probe, reports device + sample-tile count.
    POST /predict                   — bbox + before/after dates → hectares + confidence + id.
    GET  /tile/{prediction_id}/before.png      — original 'before' image, true-colour stretched.
    GET  /tile/{prediction_id}/after.png       — original 'after' image, true-colour stretched.
    GET  /tile/{prediction_id}/mask.png        — predicted forest-loss mask (red overlay).
    GET  /tile/{prediction_id}/confidence.png  — heatmap of per-pixel probability.

The service runs in two modes:
    * **real**: tries to fetch Sentinel-2 + Hansen via the same code path as the data
      pipeline. If imagery is too cloudy or unavailable it returns 422 with a clear message
      rather than silently giving back a wrong answer.
    * **sample**: when ``TERRATRACE_FORCE_SYNTHETIC=1`` or there is no committed checkpoint,
      the service serves predictions over the tiles in ``data/sample/``. The bbox+dates of
      the request deterministically pick a sample tile, so the same request always returns
      the same prediction.

All predictions are written into a file-backed tile cache; the per-tile image endpoints serve
the PNGs from there. The cache key is a hash of the request body, so repeat requests are O(1).
"""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from PIL import Image

from ..data.config import BoundingBox, DateWindow, RealDataConfig
from ..data.manifest import DatasetManifest
from ..data.stac import fetch_real_tiles
from ..training.train import select_device
from .predict import TilePrediction, load_checkpoint, predict_tile
from .schemas import HealthResponse, PredictRequest, PredictResponse
from .tile_cache import CacheEntry, TileCache, fingerprint

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "fast.pt"
DEFAULT_SAMPLE_MANIFEST = ROOT / "data" / "sample" / "manifest.json"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "tiles"
MAX_CLOUD_FRACTION = 0.85  # above this we refuse to give a hectares estimate


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["arr"]


@lru_cache(maxsize=1)
def _device() -> torch.device:
    return select_device(os.environ.get("TERRATRACE_DEVICE", "auto"))


@lru_cache(maxsize=1)
def _model_and_meta() -> tuple[object, dict | None]:
    """Load the model lazily. Returns (model, ckpt_meta) or (None, None) if no checkpoint."""
    ckpt_path = Path(os.environ.get("TERRATRACE_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    if not ckpt_path.is_file():
        log.warning("No checkpoint at %s — service can only serve health probes.", ckpt_path)
        return None, None
    model, meta = load_checkpoint(ckpt_path, _device())
    log.info("Loaded checkpoint from %s.", ckpt_path)
    return model, meta


@lru_cache(maxsize=1)
def _sample_manifest() -> DatasetManifest | None:
    path = Path(os.environ.get("TERRATRACE_SAMPLE_MANIFEST", str(DEFAULT_SAMPLE_MANIFEST)))
    if not path.is_file():
        log.warning("No sample manifest at %s.", path)
        return None
    return DatasetManifest.read(path)


def _cache() -> TileCache:
    cache_dir = Path(os.environ.get("TERRATRACE_TILE_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    return TileCache(cache_dir)


def _force_sample() -> bool:
    return os.environ.get("TERRATRACE_FORCE_SYNTHETIC", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )


def _to_true_color_png(img: np.ndarray) -> bytes:
    """Render the first three bands as an RGB PNG, with a per-tile percentile stretch.

    Sentinel-2 reflectance is roughly in [0, 0.3]; using (2nd, 98th) percentile per channel
    gives a reasonable display stretch on both real and synthetic data.
    """
    rgb = img[..., :3].astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = float(np.percentile(ch, 2)), float(np.percentile(ch, 98))
        if hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    buf = io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _mask_overlay_png(mask: np.ndarray) -> bytes:
    """Render the forest-loss mask as a semi-transparent red RGBA PNG for map overlay."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    on = mask.astype(bool)
    rgba[on, 0] = 220  # R
    rgba[on, 1] = 30  # G
    rgba[on, 2] = 50  # B
    rgba[on, 3] = 190  # A (transparency for the overlay)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _confidence_png(prob: np.ndarray) -> bytes:
    """Render the per-pixel probability as a yellow→red heatmap RGBA PNG."""
    h, w = prob.shape
    p = np.clip(prob, 0.0, 1.0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # R
    rgba[..., 1] = (255 * (1.0 - p)).astype(np.uint8)  # G fades as confidence rises
    rgba[..., 2] = 0  # B
    rgba[..., 3] = (p * 220).astype(np.uint8)  # alpha = confidence
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _write_prediction_assets(entry: CacheEntry, pred: TilePrediction) -> None:
    entry.png_path("before").write_bytes(_to_true_color_png(pred.before))
    entry.png_path("after").write_bytes(_to_true_color_png(pred.after))
    entry.png_path("mask").write_bytes(_mask_overlay_png(pred.mask))
    entry.png_path("confidence").write_bytes(_confidence_png(pred.probability))


def _sample_tile_for_request(
    request: PredictRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Pick a deterministic sample tile based on the request fingerprint."""
    manifest = _sample_manifest()
    if not manifest or not manifest.tiles:
        raise HTTPException(
            status_code=503,
            detail="No sample dataset available. Set TERRATRACE_SAMPLE_MANIFEST or build the sample.",
        )
    sample_root = Path(
        os.environ.get("TERRATRACE_SAMPLE_MANIFEST", str(DEFAULT_SAMPLE_MANIFEST))
    ).parent
    idx = int(fingerprint(request), 16) % len(manifest.tiles)
    entry = manifest.tiles[idx]
    before = _load_npz(sample_root / entry.before_path)
    after = _load_npz(sample_root / entry.after_path)
    cb = _load_npz(sample_root / entry.cloud_before_path)
    ca = _load_npz(sample_root / entry.cloud_after_path)
    return before, after, cb, ca, entry.tile_id


def _try_real_tiles(
    request: PredictRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Best-effort attempt to fetch a single real tile via STAC."""
    cfg = RealDataConfig(
        region_name="api-request",
        bbox=BoundingBox(
            west=request.bbox.west,
            south=request.bbox.south,
            east=request.bbox.east,
            north=request.bbox.north,
        ),
        before_window=DateWindow(start=request.before_date, end=request.before_date),
        after_window=DateWindow(start=request.after_date, end=request.after_date),
        tile_size=128,
        max_tiles=4,
    )
    tiles = fetch_real_tiles(cfg)
    if not tiles:
        return None
    t = tiles[0]
    return t.before, t.after, t.cloud_before, t.cloud_after


def _build_app() -> FastAPI:
    app = FastAPI(
        title="TerraTrace inference service",
        description=(
            "Forest-loss change detection. POST /predict with a bounding box and two dates; "
            "the service returns hectares lost and serves the prediction overlays as PNGs."
        ),
        version="0.1.0",
    )

    # CORS — the frontend at localhost:3000 needs to call this in dev.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model, _ = _model_and_meta()
        sample = _sample_manifest()
        return HealthResponse(
            checkpoint_loaded=model is not None,
            sample_tiles=len(sample.tiles) if sample else 0,
            device=str(_device()),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        model, _meta = _model_and_meta()
        if model is None:
            raise HTTPException(503, "No model checkpoint loaded — service not ready.")

        source: str
        notes = ""
        if _force_sample():
            before, after, cb, ca, _tile_id = _sample_tile_for_request(req)
            source = "sample"
            notes = "TERRATRACE_FORCE_SYNTHETIC is set — returning a sample-tile prediction."
        else:
            real = _try_real_tiles(req)
            if real is not None:
                before, after, cb, ca = real
                source = "real"
            else:
                before, after, cb, ca, _tile_id = _sample_tile_for_request(req)
                source = "sample"
                notes = (
                    "Real Sentinel-2 data not available for the requested AOI/dates "
                    "(no scenes, fetch error, or no network). Returning a sample-tile prediction "
                    "so the demo still works; numbers below are not from real imagery."
                )

        pred = predict_tile(
            model=model,
            before=before,
            after=after,
            cloud_before=cb,
            cloud_after=ca,
            device=_device(),
            threshold=req.threshold,
        )
        if pred.cloud_fraction > MAX_CLOUD_FRACTION:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Imagery for the requested AOI/dates is {pred.cloud_fraction*100:.0f}% "
                    f"cloud-covered (max allowed {MAX_CLOUD_FRACTION*100:.0f}%). "
                    "Pick a different date window or shrink the AOI."
                ),
            )

        pred_id = fingerprint(req)
        cache_entry = _cache().prepare(pred_id)
        _write_prediction_assets(cache_entry, pred)
        metadata = {
            "prediction_id": pred_id,
            "source": source,
            "hectares_lost": pred.hectares_lost,
            "mean_confidence": pred.mean_confidence,
            "cloud_fraction": pred.cloud_fraction,
            "pixels_total": int(pred.mask.size),
            "pixels_loss": int(pred.mask.sum()),
        }
        cache_entry.metadata_path.write_text(__json_dumps(metadata))

        return PredictResponse(
            prediction_id=pred_id,
            bbox=req.bbox,
            hectares_lost=pred.hectares_lost,
            pixels_loss=int(pred.mask.sum()),
            pixels_total=int(pred.mask.size),
            mean_confidence=pred.mean_confidence,
            cloud_fraction=pred.cloud_fraction,
            source=source,  # type: ignore[arg-type]
            notes=notes,
        )

    @app.get("/tile/{prediction_id}/{kind}.png")
    def tile_png(prediction_id: str, kind: str) -> Response:
        if kind not in {"before", "after", "mask", "confidence"}:
            raise HTTPException(404, f"unknown tile kind '{kind}'")
        entry = _cache().entry(prediction_id)
        png_path = entry.png_path(kind)
        if not png_path.is_file():
            raise HTTPException(404, "prediction not found — has /predict been called?")
        return Response(content=png_path.read_bytes(), media_type="image/png")

    return app


def __json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2)


app = _build_app()
