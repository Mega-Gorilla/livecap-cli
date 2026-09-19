"""engine テスト共通 fixture — ModelRoot 契約の sentinel (Issue #456)。

``model_root_sentinels`` は「設定した ModelRoot の外へモデル資産を書かない」ことを
**実 path で**固定する。

* ``HF_HUB_CACHE`` を空の tmp、``HF_HUB_OFFLINE=1``、``HF_HOME`` を sentinel へ (#428 方式)。
  production が ``cache_dir=`` を落として既定 cache へ silent fallback したら必ず落ちる
* ``LIVECAP_CORE_MODELS_DIR`` / ``LIVECAP_CORE_CACHE_DIR`` を tmp へ
* 実 ``%LOCALAPPDATA%\\whisper_s2t\\whisper_s2t\\Cache\\models`` と ``~/.cache/huggingface/hub``
  の**ファイル**一覧を before / after で比較する。**空 dir は違反にしない** —
  ``import whisper_s2t`` は upstream の ``os.makedirs`` で空 dir を必ず作る (#430)。
  ``platformdirs`` は Windows で ``SHGetKnownFolderPath`` を使い ``LOCALAPPDATA`` env を
  見ないので、env を差し替える方式では捕まえられない (#430 実測) — だから実 path を見る
* teardown で models_root 内に transient (``.cache`` / ``.locks`` / ``*.lock`` /
  ``*.incomplete`` / ``*.metadata`` / ``*.part``) が無いこと、``HF_HUB_CACHE`` が空のままで
  あることを assert する
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from livecap_cli.engines.model_store import TRANSIENT_MARKERS, TRANSIENT_SUFFIXES
from livecap_cli.resources import _reset_resources_for_tests


def _external_model_dirs() -> list[Path]:
    """設定した root の外で、モデルが落ちてはならない実 path。存在するものだけ。"""
    candidates = [Path.home() / ".cache" / "huggingface" / "hub"]
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        candidates.append(Path(local_app) / "whisper_s2t" / "whisper_s2t" / "Cache" / "models")
    return [p for p in candidates if p.is_dir()]


def _file_listing(root: Path) -> set[str]:
    try:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    except OSError:
        return set()


def _transient_leftovers(root: Path) -> list[str]:
    hits = []
    if not root.is_dir():
        return hits
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in TRANSIENT_MARKERS for part in rel.parts) or p.name.endswith(TRANSIENT_SUFFIXES):
            hits.append(str(rel))
    return hits


@pytest.fixture
def model_root_sentinels(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    default_hub = tmp_path / "sentinel-hf-hub"
    default_hub.mkdir()
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(default_hub))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sentinel-hf-home"))
    _reset_resources_for_tests()

    external = _external_model_dirs()
    before = {root: _file_listing(root) for root in external}

    yield types.SimpleNamespace(
        models_root=models_root,
        cache_root=cache_root,
        default_hub=default_hub,
        staging_root=cache_root / "downloads",
        hub_root=cache_root / "huggingface" / "hub",
    )

    _reset_resources_for_tests()
    assert not any(default_hub.iterdir()), "既定 HF cache (HF_HUB_CACHE sentinel) に何か書かれた"
    for root in external:
        new_files = _file_listing(root) - before[root]
        assert not new_files, f"設定した root の外にモデル関連ファイルが作られた: {root}: {sorted(new_files)[:5]}"
    leftovers = _transient_leftovers(models_root)
    assert not leftovers, f"models_root に transient が残っている: {leftovers[:5]}"
