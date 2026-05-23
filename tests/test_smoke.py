"""Smoke test: the package imports and exposes a version."""

from __future__ import annotations

import terratrace


def test_package_imports() -> None:
    assert terratrace.__version__
