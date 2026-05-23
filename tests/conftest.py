"""Test-suite configuration.

Adds ``src/`` to ``sys.path`` so tests work even when the editable install isn't picked up
(notably on macOS where setuptools' editable .pth file sometimes inherits the UF_HIDDEN flag
from the parent directory, which causes ``site.py`` to skip it). The package install is still
the supported workflow; this is belt-and-braces.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
