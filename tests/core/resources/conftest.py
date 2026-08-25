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
def isolated_resource_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))

    # env を差し替えるだけでは足りない (Issue #375)。configuration は最初のアクセス
    # で **freeze** され、以後 env の変更を無視する — 意図した設計だが、その結果
    # 1 つ目のテストの root を全テストが共有してしまう。前後で完全 reset する。
    from livecap_cli.resources import _reset_resources_for_tests

    _reset_resources_for_tests()
    yield
    _reset_resources_for_tests()
