"""WhisperS2T の CTranslate2 モデルが**管理 HF cache**へ落ち、ローカル dir で load されること (Issue #430)。

以前は ``whisper_s2t.load_model(model_identifier="base")`` が内部で
``snapshot_download(cache_dir=platformdirs.user_cache_dir("whisper_s2t")/models)`` を呼び、
``%LOCALAPPDATA%\\whisper_s2t\\...`` に固定されていた (``LOCALAPPDATA`` 差し替えも受け口も無い)。

固定する契約:

* 本 repo が repo id (``MODEL_REPOS``) を決め、``snapshot_download(cache_dir=<管理 cache>)``
  で解決する (``hf_cache.resolve_snapshot``)
* ``whisper_s2t.load_model`` へ渡るのは **size 文字列ではなくローカル snapshot dir**
  (``WhisperModelCT2.__init__`` の ``os.path.isdir`` 分岐 → 内部 download が走らない)
* cuDNN fallback の再ロードも同じ dir
* marker / cache hit / self-heal の規則は Qwen3-ASR (#428) と同じ

``whisper_s2t`` と ``snapshot_download`` は差し替える。ネットワークもモデルも使わない。
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines.hf_cache import write_marker
from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.resources import _reset_resources_for_tests

REPO_ID = "Systran/faster-whisper-base"
REPO_DIR = "models--Systran--faster-whisper-base"
SHA = "b" * 40
SNAPSHOT_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


def _make_snapshot(hub: Path, files=SNAPSHOT_FILES) -> Path:
    repo = hub / REPO_DIR
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(SHA, encoding="utf-8")
    snapshot = repo / "snapshots" / SHA
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snapshot / name).write_text(name, encoding="utf-8")
    return snapshot


class _FakeSnapshotDownload:
    def __init__(self, *, fail: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail

    def __call__(self, repo_id, **kwargs):
        self.calls.append((repo_id, dict(kwargs)))
        if self.fail is not None:
            raise self.fail
        return str(_make_snapshot(Path(kwargs["cache_dir"])))


def _point_roots(monkeypatch, models_root: Path, cache_root: Path) -> None:
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


@pytest.fixture
def managed(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    default_hub = tmp_path / "default-hf-hub"
    default_hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sentinel-hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(default_hub))
    _point_roots(monkeypatch, models_root, cache_root)

    # whisper_s2t は差し替える: load_model が受け取った引数を記録する。
    fake = types.ModuleType("whisper_s2t")
    loaded = MagicMock(name="WhisperModelCT2")
    fake.load_model = MagicMock(name="load_model", return_value=loaded)
    monkeypatch.setitem(sys.modules, "whisper_s2t", fake)

    yield types.SimpleNamespace(
        tmp_path=tmp_path,
        models_root=models_root,
        cache_root=cache_root,
        managed_hub=cache_root / "huggingface" / "hub",
        default_hub=default_hub,
        marker=models_root / "Systran--faster-whisper-base.marker",
        load_model=fake.load_model,
        monkeypatch=monkeypatch,
    )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine(**kwargs):
    from livecap_cli.engines.whispers2t_engine import WhisperS2TEngine

    return WhisperS2TEngine(device="cpu", model_size="base", language="en", **kwargs)


def _load_with(fake: _FakeSnapshotDownload, engine=None):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = engine or _engine()
        engine.load_model()
    return engine


class TestModelRepos:
    def test_every_size_maps_to_a_ct2_repo(self):
        from livecap_cli.engines.whispers2t_engine import MODEL_REPOS, VALID_MODEL_SIZES

        assert VALID_MODEL_SIZES == frozenset(MODEL_REPOS)
        assert all("/" in repo for repo in MODEL_REPOS.values()), "repo id (org/name) でなければならない"
        assert MODEL_REPOS["base"] == REPO_ID
        assert MODEL_REPOS["large-v3-turbo"] == "deepdml/faster-whisper-large-v3-turbo-ct2"


class TestColdCache:
    def test_snapshot_is_resolved_into_managed_cache(self, managed):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        (repo_id, kwargs), = fake.calls
        assert repo_id == REPO_ID
        assert Path(kwargs["cache_dir"]) == managed.managed_hub, "管理 cache (get_huggingface_cache_dir) へ"
        assert kwargs["max_workers"] == 1
        assert "model.bin" in kwargs["allow_patterns"] and "config.json" in kwargs["allow_patterns"]
        assert not any(managed.default_hub.iterdir()), "既定 HF cache には何も書かれない"

    def test_load_model_receives_local_snapshot_dir_not_size(self, managed):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        managed.load_model.assert_called_once()
        kwargs = managed.load_model.call_args.kwargs
        assert kwargs["model_identifier"] != "base", "size 文字列を渡すと whisper_s2t が %LOCALAPPDATA% へ落とす"
        assert Path(kwargs["model_identifier"]) == (managed.managed_hub / REPO_DIR / "snapshots" / SHA).resolve()
        assert Path(kwargs["model_identifier"]).is_dir(), "os.path.isdir 分岐に入る"
        assert kwargs["backend"] == "CTranslate2" and kwargs["n_mels"] == 80

    def test_marker_records_relative_snapshot_and_manifest(self, managed):
        _load_with(_FakeSnapshotDownload())

        payload = json.loads(managed.marker.read_text(encoding="utf-8"))
        assert payload["snapshot"] == f"{REPO_DIR}/snapshots/{SHA}"
        assert payload["files"] == sorted(SNAPSHOT_FILES)

    def test_environment_is_not_rewritten(self, managed, tmp_path):
        _load_with(_FakeSnapshotDownload())

        assert os.environ["HF_HOME"] == str(tmp_path / "sentinel-hf-home")
        assert os.environ["HF_HUB_CACHE"] == str(managed.default_hub)


class TestCacheHit:
    def test_marker_with_existing_snapshot_skips_download(self, managed):
        snapshot = _make_snapshot(managed.managed_hub)
        write_marker(managed.marker, managed.managed_hub, snapshot)
        fake = _FakeSnapshotDownload(fail=AssertionError("cache hit なので呼ばれない"))

        _load_with(fake)

        assert fake.calls == []
        assert Path(managed.load_model.call_args.kwargs["model_identifier"]) == snapshot.resolve()

    def test_marker_from_previous_cache_root_is_not_a_hit(self, managed):
        cache_a = managed.tmp_path / "cache-a"
        hub_a = cache_a / "huggingface" / "hub"
        _point_roots(managed.monkeypatch, managed.models_root, cache_a)
        write_marker(managed.marker, hub_a, _make_snapshot(hub_a))
        assert _engine()._is_model_cached(managed.marker)

        _point_roots(managed.monkeypatch, managed.models_root, managed.cache_root)
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        target = Path(managed.load_model.call_args.kwargs["model_identifier"])
        assert target.is_relative_to(managed.managed_hub.resolve())
        assert not target.is_relative_to(hub_a.resolve())

    def test_snapshot_missing_weights_is_not_a_hit(self, managed):
        snapshot = _make_snapshot(managed.managed_hub)
        write_marker(managed.marker, managed.managed_hub, snapshot)
        (snapshot / "model.bin").unlink()
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1


class TestFallbackAndFailure:
    def test_cudnn_fallback_reloads_from_the_same_snapshot(self, managed):
        """cuDNN 失敗時の CPU 再ロードも size 文字列ではなく同じローカル dir を渡す。"""
        snapshot = _make_snapshot(managed.managed_hub)
        write_marker(managed.marker, managed.managed_hub, snapshot)
        managed.load_model.side_effect = [RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"), MagicMock(name="cpu_model")]
        engine = _engine()
        engine.device = "cuda"  # fallback 分岐の条件を作る (実 GPU は使わない)

        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownload(fail=AssertionError("hit"))):
            engine.load_model()

        first, second = managed.load_model.call_args_list
        assert Path(first.kwargs["model_identifier"]) == snapshot.resolve()
        assert Path(second.kwargs["model_identifier"]) == snapshot.resolve()
        assert second.kwargs["device"] == "cpu" and second.kwargs["compute_type"] == "int8"
        assert engine.device == "cpu"
        assert managed.marker.exists(), "fallback で成功したので marker は残す"

    def test_load_failure_invalidates_marker_for_self_heal(self, managed):
        snapshot = _make_snapshot(managed.managed_hub)
        write_marker(managed.marker, managed.managed_hub, snapshot)
        managed.load_model.side_effect = OSError("corrupt model.bin")

        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownload(fail=AssertionError("hit"))):
            with pytest.raises(OSError, match="corrupt"):
                _engine().load_model()

        assert not managed.marker.exists(), "marker が残ると永久に skip して落ち続ける"

    def test_failed_resolution_leaves_no_marker(self, managed):
        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownload(fail=RuntimeError("network down"))):
            with pytest.raises(RuntimeError, match="network down"):
                _engine().load_model()

        assert not managed.marker.exists()
        managed.load_model.assert_not_called()
