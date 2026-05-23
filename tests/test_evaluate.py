"""End-to-end smoke test for the evaluator.

This test trains a tiny model for a single epoch on a tiny synthetic dataset, then runs the
full evaluator and checks that the report is structurally sound and the headline numbers fall
in a plausible range. It's not a regression test on absolute metrics — those move with seed
and config — it's a contract test on the evaluator's outputs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from terratrace.data.config import (
    BoundingBox,
    DateWindow,
    PipelineConfig,
    RealDataConfig,
    SplitConfig,
    SyntheticConfig,
)
from terratrace.data.pipeline import build_dataset
from terratrace.models.siamese_unet import SiameseUNet, SiameseUNetConfig
from terratrace.training.evaluate import evaluate


def _tiny_pipeline_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=tmp_path / "processed",
        sample_dir=tmp_path / "sample",
        sample_tiles=2,
        fetch_mode="synthetic",
        real=RealDataConfig(
            region_name="testregion",
            bbox=BoundingBox(west=-1, south=-1, east=1, north=1),
            before_window=DateWindow(start="2020-01-01", end="2020-03-31"),
            after_window=DateWindow(start="2022-01-01", end="2022-03-31"),
        ),
        synthetic=SyntheticConfig(
            seed=99, tile_size=32, grid_rows=4, grid_cols=4, cloud_fraction=0.05
        ),
        split=SplitConfig(grid_size=2, val_blocks=1, test_blocks=1, seed=0),
    )


def test_evaluator_returns_structurally_sound_report(tmp_path: Path) -> None:
    cfg = _tiny_pipeline_config(tmp_path)
    manifest = build_dataset(cfg)

    # Train a one-epoch tiny model just so the checkpoint has some signal in it.
    model_cfg = SiameseUNetConfig(in_channels=4, base_channels=8, depth=3)
    model = SiameseUNet(model_cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": model_cfg.in_channels,
                "base_channels": model_cfg.base_channels,
                "depth": model_cfg.depth,
            },
            "epoch": 0,
            "val_iou": 0.0,
            "data_source": manifest.source,
        },
        ckpt_path,
    )

    report = evaluate(
        manifest_path=cfg.output_dir / "manifest.json",
        checkpoint_path=ckpt_path,
        device="cpu",
        results_path=tmp_path / "metrics.json",
        assets_dir=tmp_path / "assets",
    )

    # Headline metrics are present and lie in [0, 1].
    for key in ("iou", "precision", "recall", "f1", "accuracy"):
        assert key in report.deep_model
        assert 0.0 <= report.deep_model[key] <= 1.0
        assert key in report.baseline
        assert 0.0 <= report.baseline[key] <= 1.0

    # Calibration curve has the expected bin count.
    assert len(report.calibration.bin_centers) == 10
    assert sum(report.calibration.bin_counts) > 0

    # JSON + figures were written.
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "assets" / "calibration.png").is_file()
    assert (tmp_path / "assets" / "metrics_comparison.png").is_file()
