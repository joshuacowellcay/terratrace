"""Single-tile prediction pipeline shared by the FastAPI service and tests.

Given a tile (before / after / cloud masks), runs the model and computes the headline numbers
(hectares lost, mean confidence, cloud fraction). Keeps the heavy lifting outside the HTTP
layer so unit tests can call it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.siamese_unet import SiameseUNet, SiameseUNetConfig

# Sentinel-2 L2A bands are at 10 m ground resolution by default; the synthetic generator's
# tiles match, so one pixel is ~100 m^2. A future revision should derive this from the
# manifest if a different pixel size is recorded there.
DEFAULT_PIXEL_AREA_M2 = 100.0


@dataclass(frozen=True)
class TilePrediction:
    probability: np.ndarray  # (H, W) float32 in [0, 1]
    mask: np.ndarray  # (H, W) uint8, thresholded at the requested cutoff
    before: np.ndarray  # (H, W, C) float32
    after: np.ndarray  # (H, W, C) float32
    cloud_mask: np.ndarray  # (H, W) uint8, 1 = invalid in either timestamp
    pixel_area_m2: float

    @property
    def hectares_lost(self) -> float:
        """Convert predicted loss pixels to hectares (1 ha = 10 000 m^2)."""
        return float(self.mask.sum() * self.pixel_area_m2 / 10_000.0)

    @property
    def mean_confidence(self) -> float:
        """Mean probability over pixels predicted as loss; 0 when none are predicted."""
        sel = self.mask.astype(bool)
        if not sel.any():
            return 0.0
        return float(self.probability[sel].mean())

    @property
    def cloud_fraction(self) -> float:
        if self.cloud_mask.size == 0:
            return 0.0
        return float(self.cloud_mask.astype(bool).mean())


def load_checkpoint(path: Path, device: torch.device) -> tuple[SiameseUNet, dict[str, object]]:
    """Reconstruct the model from its embedded config + state dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = SiameseUNetConfig(**ckpt["model_config"])
    model = SiameseUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_tile(
    *,
    model: SiameseUNet,
    before: np.ndarray,
    after: np.ndarray,
    cloud_before: np.ndarray,
    cloud_after: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    pixel_area_m2: float = DEFAULT_PIXEL_AREA_M2,
) -> TilePrediction:
    """Run a single forward pass and assemble the post-processed output."""
    if before.shape != after.shape:
        raise ValueError(f"before/after shape mismatch: {before.shape} vs {after.shape}")
    h, w, _c = before.shape
    if cloud_before.shape != (h, w) or cloud_after.shape != (h, w):
        raise ValueError("cloud masks must match the spatial dimensions of before/after")

    cloud = ((cloud_before.astype(bool)) | (cloud_after.astype(bool))).astype(np.uint8)

    with torch.no_grad():
        b = (
            torch.from_numpy(np.ascontiguousarray(before.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(device)
        )
        a = torch.from_numpy(np.ascontiguousarray(after.transpose(2, 0, 1))).unsqueeze(0).to(device)
        logits = model(b.float(), a.float()).squeeze(0).squeeze(0)
        prob = torch.sigmoid(logits).cpu().numpy()

    # Zero out confidence on cloud-corrupted pixels; we don't trust the model there.
    prob_masked = prob.copy()
    prob_masked[cloud.astype(bool)] = 0.0
    mask = (prob_masked > threshold).astype(np.uint8)

    return TilePrediction(
        probability=prob_masked.astype(np.float32),
        mask=mask,
        before=before,
        after=after,
        cloud_mask=cloud,
        pixel_area_m2=pixel_area_m2,
    )
