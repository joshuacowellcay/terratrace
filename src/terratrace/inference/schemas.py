"""Request and response schemas for the inference service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PredictionSource = Literal["real", "sample"]


class BBoxModel(BaseModel):
    """WGS84 bounding box."""

    model_config = ConfigDict(extra="forbid")

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bbox: BBoxModel
    before_date: str = Field(description="ISO date or YYYY-MM-DD for the 'before' image.")
    after_date: str = Field(description="ISO date or YYYY-MM-DD for the 'after' image.")
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(description="Identifier for fetching per-pixel PNGs back.")
    bbox: BBoxModel
    hectares_lost: float = Field(ge=0.0)
    pixels_loss: int = Field(ge=0)
    pixels_total: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    cloud_fraction: float = Field(ge=0.0, le=1.0)
    source: PredictionSource
    notes: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    checkpoint_loaded: bool
    sample_tiles: int
    device: str
