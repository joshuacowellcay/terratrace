#!/usr/bin/env python3
"""Run the full evaluation: model + baseline on the test split, with metrics + figures.

Usage:
    python scripts/evaluate.py [--checkpoint checkpoints/fast.pt] [--manifest data/processed/manifest.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from terratrace.training.evaluate import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate model + baseline on the test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "fast.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "manifest.json",
        help="Path to the dataset manifest.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "metrics.json",
        help="Path to write the metrics JSON.",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=ROOT / "assets",
        help="Directory for generated figures (calibration.png, metrics_comparison.png).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = evaluate(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        device=args.device,
        results_path=args.results,
        assets_dir=args.assets,
    )

    print("\n=== TerraTrace evaluation ===")
    print(
        f"data source: {report.data_source}    region: {report.region_name}    "
        f"test tiles: {report.n_test_tiles}"
    )
    print(f"\nSiamese U-Net:  {json.dumps(report.deep_model, indent=2)}")
    print(
        f"NDVI baseline (t={report.baseline_threshold:.3f}):  "
        f"{json.dumps(report.baseline, indent=2)}"
    )
    print(f"\nFailure breakdown:  {json.dumps(report.failure.__dict__, indent=2)}")
    print(f"\nMetrics → {report.metrics_path}")
    print(f"Figures → {', '.join(report.figures.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
