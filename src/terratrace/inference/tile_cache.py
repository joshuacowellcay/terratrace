"""Process-local cache for prediction results, keyed by request fingerprint.

Stored on disk so the cache survives restarts. Each entry is a directory containing the four
PNGs the frontend fetches plus a small ``metadata.json`` with the headline numbers. This keeps
the FastAPI handlers thin: build the prediction once, then GET endpoints serve files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .schemas import PredictRequest


@dataclass(frozen=True)
class CacheEntry:
    prediction_id: str
    directory: Path
    metadata_path: Path

    def png_path(self, name: str) -> Path:
        return self.directory / f"{name}.png"


def fingerprint(request: PredictRequest) -> str:
    """Deterministic short identifier from the request body. 12-hex chars is plenty for collisions."""
    payload = json.dumps(request.model_dump(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


class TileCache:
    """Tiny on-disk cache. Not eviction-aware — operators clear the directory periodically."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def entry(self, prediction_id: str) -> CacheEntry:
        directory = self.root / prediction_id
        return CacheEntry(
            prediction_id=prediction_id,
            directory=directory,
            metadata_path=directory / "metadata.json",
        )

    def has(self, prediction_id: str) -> bool:
        return self.entry(prediction_id).metadata_path.is_file()

    def prepare(self, prediction_id: str, *, overwrite: bool = True) -> CacheEntry:
        entry = self.entry(prediction_id)
        if entry.directory.exists() and overwrite:
            shutil.rmtree(entry.directory)
        entry.directory.mkdir(parents=True, exist_ok=True)
        return entry

    def clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
