#!/usr/bin/env python3
"""Build the TerraTrace dataset.

Usage:
    python scripts/build_dataset.py [--config configs/pipeline.yaml]

The script picks the real-data path or the synthetic fallback automatically (configurable via
``fetch_mode`` in the YAML, or by setting ``TERRATRACE_FORCE_SYNTHETIC=1``). Output goes to
``data/processed/`` and a small committed-friendly subset goes to ``data/sample/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_dataset.py` from the repo root without an install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from terratrace.data.config import PipelineConfig
from terratrace.data.pipeline import build_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the TerraTrace dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline.yaml",
        help="Path to the pipeline YAML config (default: configs/pipeline.yaml).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig.from_yaml(args.config)
    manifest = build_dataset(config)
    print(
        f"\nDataset built. Source: {manifest.source}. "
        f"Tiles total={len(manifest.tiles)} "
        f"(train={len(manifest.by_split('train'))}, "
        f"val={len(manifest.by_split('val'))}, "
        f"test={len(manifest.by_split('test'))})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
