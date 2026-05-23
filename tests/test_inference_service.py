"""Tests for the FastAPI inference service.

Runs entirely against the committed data/sample/ tiles (TERRATRACE_FORCE_SYNTHETIC=1) so no
network is involved. Reaches into the service module to repoint env vars at a temp cache
directory so tests don't pollute the user's workspace.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    os.environ["TERRATRACE_FORCE_SYNTHETIC"] = "1"
    os.environ["TERRATRACE_TILE_CACHE_DIR"] = str(tmp_path / "cache")
    os.environ["TERRATRACE_CHECKPOINT"] = str(ROOT / "checkpoints" / "fast.pt")
    os.environ["TERRATRACE_SAMPLE_MANIFEST"] = str(ROOT / "data" / "sample" / "manifest.json")

    # Re-import the service module so the lru_cache picks up the env vars set above.
    import terratrace.inference.service as svc

    importlib.reload(svc)
    with TestClient(svc.app) as c:
        yield c


def test_health_endpoint_reports_loaded_state(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checkpoint_loaded"] is True
    assert body["sample_tiles"] > 0


def test_predict_returns_hectares_and_overlay(client: TestClient) -> None:
    payload = {
        "bbox": {"west": -63.1, "south": -10.4, "east": -63.0, "north": -10.3},
        "before_date": "2019-07-15",
        "after_date": "2023-07-15",
        "threshold": 0.5,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "sample"
    assert body["hectares_lost"] >= 0.0
    assert 0 <= body["pixels_loss"] <= body["pixels_total"]
    assert body["pixels_total"] > 0
    pid = body["prediction_id"]
    assert len(pid) == 12

    for kind in ("before", "after", "mask", "confidence"):
        png = client.get(f"/tile/{pid}/{kind}.png")
        assert png.status_code == 200, kind
        assert png.headers["content-type"] == "image/png"
        assert len(png.content) > 64


def test_unknown_tile_kind_returns_404(client: TestClient) -> None:
    # First make a prediction so a prediction_id exists in the cache.
    payload = {
        "bbox": {"west": -63.1, "south": -10.4, "east": -63.0, "north": -10.3},
        "before_date": "2019-07-15",
        "after_date": "2023-07-15",
    }
    pid = client.post("/predict", json=payload).json()["prediction_id"]
    assert client.get(f"/tile/{pid}/wrongkind.png").status_code == 404


def test_unknown_prediction_id_returns_404(client: TestClient) -> None:
    assert client.get("/tile/deadbeef0000/mask.png").status_code == 404


def test_predict_is_deterministic_for_same_request(client: TestClient) -> None:
    payload = {
        "bbox": {"west": -63.1, "south": -10.4, "east": -63.0, "north": -10.3},
        "before_date": "2019-07-15",
        "after_date": "2023-07-15",
    }
    a = client.post("/predict", json=payload).json()
    b = client.post("/predict", json=payload).json()
    assert a["prediction_id"] == b["prediction_id"]
    assert a["hectares_lost"] == b["hectares_lost"]
