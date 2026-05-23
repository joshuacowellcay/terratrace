"""Training-run configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..models.siamese_unet import SiameseUNetConfig


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(gt=0, default=1e-3)
    weight_decay: float = Field(ge=0, default=1e-4)


class LossConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    dice_weight: float = Field(ge=0, default=0.5)
    bce_weight: float = Field(ge=0, default=0.5)
    pos_weight: float = Field(gt=0, default=5.0)


class TrainConfig(BaseModel):
    """Top-level training config."""

    model_config = ConfigDict(frozen=True)

    name: str = "fast"
    manifest_path: Path = Path("data/processed/manifest.json")
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "model.pt"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "terratrace"

    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42

    epochs: int = Field(ge=1, default=20)
    batch_size: int = Field(ge=1, default=8)
    num_workers: int = Field(ge=0, default=0)  # 0 = stay in main process; safer on macOS

    model: SiameseUNetConfig = SiameseUNetConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()

    augment_train: bool = True
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
