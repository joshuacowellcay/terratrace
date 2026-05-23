#!/usr/bin/env python3
"""Train the Siamese U-Net change detector.

Usage:
    python scripts/train.py --config configs/fast.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from terratrace.training.config import TrainConfig
from terratrace.training.train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Siamese U-Net change detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "fast.yaml",
        help="Path to the training YAML config (default: configs/fast.yaml).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = TrainConfig.from_yaml(args.config)
    outcome = train(cfg)
    print(
        f"\nTraining complete. best_val_iou={outcome.best_val_iou:.4f} "
        f"params={outcome.parameter_count:,} device={outcome.device} "
        f"checkpoint={outcome.checkpoint_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
