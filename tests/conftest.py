"""Shared fixtures for the OHTTP / TEE-verification tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import _ohttp_recipient as _recipient  # tests/ is on sys.path under pytest
import pytest


@pytest.fixture(scope="session")
def recipient():
    """A self-contained OHTTP recipient so crypto is exercised in CI without an
    external tee-gateway checkout."""
    return _recipient


@pytest.fixture
def real_tee_gateway():
    """The real ``tee_gateway.ohttp`` module, for cross-checking wire compatibility.

    Skips when no tee-gateway checkout is present (``OG_TEE_GATEWAY`` or a sibling
    ``../tee-gateway``). Restores ``sys.path`` afterwards so it can't perturb
    import resolution in later tests.
    """
    override = os.getenv("OG_TEE_GATEWAY")
    candidates = [Path(override)] if override else []
    candidates.append(Path(__file__).resolve().parents[2] / "tee-gateway")
    root = next((p for p in candidates if (p / "tee_gateway" / "ohttp.py").exists()), None)
    if root is None:
        pytest.skip("tee-gateway checkout not found (set OG_TEE_GATEWAY)")

    inserted = str(root)
    sys.path.insert(0, inserted)
    try:
        import tee_gateway.ohttp as srv

        yield srv
    finally:
        try:
            sys.path.remove(inserted)
        except ValueError:
            pass
        for name in [n for n in sys.modules if n == "tee_gateway" or n.startswith("tee_gateway.")]:
            del sys.modules[name]
