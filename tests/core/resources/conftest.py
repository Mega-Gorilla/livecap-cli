"""Keep resource tests out of the developer's real cache.

``FFmpegManager`` and ``ModelManager`` fall back to the user cache directory
(``%LOCALAPPDATA%/PineLab/LiveCap`` and friends). Tests that construct them
without overriding that wrote fake binaries into the real cache - and once the
managed cache learned to repair itself (Issue #398), that turned into an actual
268 MB download during a unit-test run.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_resource_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
