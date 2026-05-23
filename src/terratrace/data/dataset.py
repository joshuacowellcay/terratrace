"""PyTorch :class:`Dataset` wrapper over a built dataset directory.

Reads the manifest, lazily loads each tile's arrays from disk, and yields tensors in the
shape the training loop expects: (C, H, W) for images, (H, W) for label and weight masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .cloud_mask import valid_pixel_weights
from .manifest import DatasetManifest, TileEntry

Split = Literal["train", "val", "test"]


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["arr"]


class ChangeDetectionDataset(Dataset[dict[str, torch.Tensor]]):
    """A dataset of (before, after) → forest-loss-mask tuples.

    Returns a dict per sample:
        before:   float32 tensor (C, H, W)
        after:    float32 tensor (C, H, W)
        label:    float32 tensor (H, W)  — 0 or 1
        weights:  float32 tensor (H, W)  — 1 where the pixel is valid, 0 under cloud
        tile_id:  str (Python string, not a tensor)
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        split: Split,
        *,
        root: Path | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest = manifest
        self.split = split
        self.entries: list[TileEntry] = manifest.by_split(split)
        # Tile paths in the manifest are relative to the manifest's directory; ``root`` is
        # that directory. Pass it explicitly, or use :meth:`from_manifest_path` below.
        self.root = Path(root) if root is not None else Path(".")
        self.augment = augment
        if not self.entries:
            raise ValueError(f"split '{split}' contains zero tiles")

    @classmethod
    def from_manifest_path(
        cls,
        manifest_path: Path | str,
        split: Split,
        *,
        augment: bool = False,
    ) -> ChangeDetectionDataset:
        manifest_path = Path(manifest_path)
        manifest = DatasetManifest.read(manifest_path)
        return cls(manifest, split, root=manifest_path.parent, augment=augment)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[idx]
        before = _load_npz(self.root / entry.before_path)
        after = _load_npz(self.root / entry.after_path)
        label = _load_npz(self.root / entry.label_path).astype(np.float32)
        cb = _load_npz(self.root / entry.cloud_before_path)
        ca = _load_npz(self.root / entry.cloud_after_path)
        weights = valid_pixel_weights(cb, ca)

        if self.augment:
            before, after, label, weights = self._augment(before, after, label, weights)

        # Channels-last → channels-first.
        before_t = torch.from_numpy(np.ascontiguousarray(before.transpose(2, 0, 1)))
        after_t = torch.from_numpy(np.ascontiguousarray(after.transpose(2, 0, 1)))
        return {
            "before": before_t.float(),
            "after": after_t.float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).float(),
            "weights": torch.from_numpy(np.ascontiguousarray(weights)).float(),
            "tile_id": entry.tile_id,
        }

    @staticmethod
    def _augment(
        before: np.ndarray, after: np.ndarray, label: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Spatial-symmetry augmentation only — no colour jitter (it would distort NDVI)."""
        if np.random.random() < 0.5:
            before = before[:, ::-1, :]
            after = after[:, ::-1, :]
            label = label[:, ::-1]
            weights = weights[:, ::-1]
        if np.random.random() < 0.5:
            before = before[::-1, :, :]
            after = after[::-1, :, :]
            label = label[::-1, :]
            weights = weights[::-1, :]
        # Random rotation in {0, 90, 180, 270}
        k = int(np.random.randint(0, 4))
        if k:
            before = np.rot90(before, k=k, axes=(0, 1))
            after = np.rot90(after, k=k, axes=(0, 1))
            label = np.rot90(label, k=k, axes=(0, 1))
            weights = np.rot90(weights, k=k, axes=(0, 1))
        return before, after, label, weights
