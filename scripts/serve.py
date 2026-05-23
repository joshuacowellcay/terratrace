#!/usr/bin/env python3
"""Start the TerraTrace inference service.

Usage:
    python scripts/serve.py [--host 0.0.0.0] [--port 8000]

For production-style runs use uvicorn directly:
    uvicorn terratrace.inference.service:app --host 0.0.0.0 --port 8000
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

import uvicorn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TerraTrace inference service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev only).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(
        "terratrace.inference.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
