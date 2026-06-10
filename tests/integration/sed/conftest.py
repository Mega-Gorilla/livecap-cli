"""Shared fixtures for Phase 2 SED evaluation tests (Issue #305 PR-D0).

The EfficientAT smoke test requires a manual clone (see
``benchmarks/sed/README.md``); this conftest exposes the path-detection
fixture so test modules can skip cleanly when the clone is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def efficientat_path() -> Path | None:
    """Resolve the EfficientAT clone path, or return ``None`` if unavailable.

    Resolution order:
    1. ``LIVECAP_SED_EFFICIENTAT_PATH`` env var
    2. ``.tmp/EfficientAT/`` under the repository root (the README default)

    Returns ``None`` when neither path exists, signalling that smoke tests
    should be skipped.
    """

    env_value = os.environ.get("LIVECAP_SED_EFFICIENTAT_PATH")
    if env_value:
        candidate = Path(env_value)
        if candidate.is_dir():
            return candidate
        return None

    # Fall back to the README-documented default path.
    repo_root = Path(__file__).resolve().parents[3]
    fallback = repo_root / ".tmp" / "EfficientAT"
    if fallback.is_dir():
        return fallback
    return None
