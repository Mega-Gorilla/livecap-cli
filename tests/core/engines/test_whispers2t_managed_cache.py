"""WhisperS2T の CTranslate2 モデルが **models_root の flattened dir** に置かれ、ローカル dir で
load されること (Issue #430 → #456)。

以前は ``whisper_s2t.load_model(model_identifier="base")`` が内部で
``snapshot_download(cache_dir=platformdirs.user_cache_dir("whisper_s2t")/models)`` を呼び、
``%LOCALAPPDATA%\\whisper_s2t\\...`` に固定されていた (``LOCALAPPDATA`` 差し替えも受け口も無い)。

固定する契約:

* 本 repo が repo id (``MODEL_REPOS``) を決め、``hf_cache.fetch_repo_dir`` で
  ``<models_root>/Systran--faster-whisper-<size>/`` へ配置する (``SNAPSHOT_ALLOW_PATTERNS`` で絞る)
* ``whisper_s2t.load_model`` へ渡るのは **size 文字列ではなくその dir**
  (``WhisperModelCT2.__init__`` の ``os.path.isdir`` 分岐 → 内部 download が走らない)
* cuDNN fallback の再ロードも同じ dir
* cache hit / adopt / migration / self-heal の規則は Qwen3-ASR と同じ
  (``variant=model_size`` も照合する)

``whisper_s2t`` と ``snapshot_download`` は差し替える。ネットワークもモデルも使わない。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines import model_store as ms
from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.resources import _reset_resources_for_tests
from tests.core.engines.conftest import FakeSnapshotDownloadLocalDir, write_hub_snapshot, write_repo_dir

REPO_ID = "Systran/faster-whisper-base"
DEST_NAME = "Systran--faster-whisper-base"
REPO_FILES = {
    "config.json": b"{}",
    "model.bin": b"w" * 256,
    "tokenizer.json": b"{}",
    "vocabulary.txt": b"a\nb\n",
    "README.md": b"# readme",
    ".gitattributes": b"",
}
MODEL_FILES = {k: v for k, v in REPO_FILES.items() if k not in ("README.md", ".gitattributes")}


@pytest.fixture
def managed(model_root_sentinels, monkeypatch):
    roots = model_root_sentinels
    ModelMemoryCache.clear()

    # whisper_s2t は差し替える: load_model が受け取った引数を記録する。
    fake = types.ModuleType("whisper_s2t")
    loaded = MagicMock(name="WhisperModelCT2")
    fake.load_model = MagicMock(name="load_model", return_value=loaded)
    monkeypatch.setitem(sys.modules, "whisper_s2t", fake)

    yield types.SimpleNamespace(
        models_root=roots.models_root,
        cache_root=roots.cache_root,
        default_hub=roots.default_hub,
        staging_root=roots.staging_root,
        hub_root=roots.hub_root,
        destination=roots.models_root / DEST_NAME,
        load_model=fake.load_model,
    )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine(**kwargs):
    from livecap_cli.engines.whispers2t_engine import WhisperS2TEngine

    return WhisperS2TEngine(device="cpu", model_size="base", language="en", **kwargs)


def _fake(**kwargs) -> FakeSnapshotDownloadLocalDir:
    return FakeSnapshotDownloadLocalDir(files=REPO_FILES, **kwargs)


def _load_with(fake, engine=None):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = engine or _engine()
        engine.load_model()
    return engine


def _manifest(managed) -> ms.Manifest:
    manifest = ms.validate_repo_dir(managed.destination, repo_id=REPO_ID, variant="base")
    assert manifest is not None
    return manifest


class TestModelRepos:
    def test_every_size_maps_to_a_ct2_repo(self):
        from livecap_cli.engines.whispers2t_engine import MODEL_REPOS, VALID_MODEL_SIZES

        assert VALID_MODEL_SIZES == frozenset(MODEL_REPOS)
        assert all("/" in repo for repo in MODEL_REPOS.values()), "repo id (org/name) でなければならない"
        assert MODEL_REPOS["base"] == REPO_ID
        assert MODEL_REPOS["large-v3-turbo"] == "deepdml/faster-whisper-large-v3-turbo-ct2"


class TestColdCache:
    def test_downloads_via_staging_into_models_root(self, managed):
        fake = _fake()

        _load_with(fake)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID
        assert Path(call["local_dir"]) == managed.staging_root / DEST_NAME / "download"
        assert Path(call["cache_dir"]) == managed.hub_root, "管理 cache (get_huggingface_cache_dir) へ"
        assert call["max_workers"] == 1
        assert "model.bin" in call["allow_patterns"] and "config.json" in call["allow_patterns"]
        assert not any(managed.default_hub.iterdir()), "既定 HF cache には何も書かれない"
        assert not (managed.staging_root / DEST_NAME).exists()

    def test_load_model_receives_models_root_dir_not_size(self, managed):
        _load_with(_fake())

        managed.load_model.assert_called_once()
        kwargs = managed.load_model.call_args.kwargs
        assert kwargs["model_identifier"] != "base", "size 文字列を渡すと whisper_s2t が %LOCALAPPDATA% へ落とす"
        assert Path(kwargs["model_identifier"]) == managed.destination
        assert Path(kwargs["model_identifier"]).is_dir(), "os.path.isdir 分岐に入る"
        assert kwargs["backend"] == "CTranslate2" and kwargs["n_mels"] == 80

    def test_manifest_records_variant_and_model_files(self, managed):
        _load_with(_fake())

        manifest = _manifest(managed)
        assert manifest.variant == "base" and manifest.source == "download"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES), "allow_patterns の外は取らない"
        assert not (managed.destination / ".cache").exists()


class TestCacheHit:
    def test_valid_manifest_skips_download(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="base")
        fake = _fake(fail=AssertionError("cache hit なので呼ばれない"))

        _load_with(fake)

        assert fake.calls == []
        assert Path(managed.load_model.call_args.kwargs["model_identifier"]) == managed.destination

    def test_manifest_with_other_variant_is_not_a_hit(self, managed):
        """同じ dir 名でも variant (model_size) が違えば別物として扱う。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="small")

        assert not _engine()._is_model_cached(managed.destination)

    def test_missing_weights_is_not_a_hit(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="base")
        (managed.destination / "model.bin").unlink()
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1
        assert _manifest(managed) is not None

    def test_complete_dir_without_manifest_is_adopted(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="base", with_manifest=False)

        _load_with(_fake(fail=AssertionError("adopt できるので呼ばれない")))

        assert _manifest(managed).source == "adopted"


class TestLegacyMigration:
    def test_hub_snapshot_and_marker_are_migrated_and_removed(self, managed):
        """0.2.0 (#430) の配置: ``<cache_root>/huggingface/hub/models--Systran--…`` + marker。"""
        snapshot = write_hub_snapshot(managed.hub_root, REPO_ID, REPO_FILES)
        marker = managed.models_root / f"{DEST_NAME}.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")

        _load_with(_fake(fail=AssertionError("旧配置から取り込めるので再ダウンロードしない")))

        manifest = _manifest(managed)
        assert manifest.source == "migrated" and manifest.variant == "base"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES)
        assert not snapshot.exists() and not marker.exists()
        assert Path(managed.load_model.call_args.kwargs["model_identifier"]) == managed.destination


class TestFallbackAndFailure:
    def test_cudnn_fallback_reloads_from_the_same_dir(self, managed):
        """cuDNN 失敗時の CPU 再ロードも size 文字列ではなく同じローカル dir を渡す。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="base")
        managed.load_model.side_effect = [RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"), MagicMock(name="cpu_model")]
        engine = _engine()
        engine.device = "cuda"  # fallback 分岐の条件を作る (実 GPU は使わない)

        _load_with(_fake(fail=AssertionError("hit")), engine)

        first, second = managed.load_model.call_args_list
        assert Path(first.kwargs["model_identifier"]) == managed.destination
        assert Path(second.kwargs["model_identifier"]) == managed.destination
        assert second.kwargs["device"] == "cpu" and second.kwargs["compute_type"] == "int8"
        assert engine.device == "cpu"
        assert _manifest(managed) is not None, "fallback で成功したので manifest は残す"

    def test_load_failure_invalidates_manifest_for_self_heal(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, variant="base")
        managed.load_model.side_effect = OSError("corrupt model.bin")

        with patch("huggingface_hub.snapshot_download", _fake(fail=AssertionError("hit"))):
            with pytest.raises(OSError, match="corrupt"):
                _engine().load_model()

        assert ms.read_manifest(managed.destination).source == ms.INVALIDATED_SOURCE
        assert not _engine()._is_model_cached(managed.destination)

    def test_failed_download_creates_no_destination(self, managed):
        with patch("huggingface_hub.snapshot_download", _fake(fail=RuntimeError("network down"))):
            with pytest.raises(RuntimeError, match="network down"):
                _engine().load_model()

        assert not managed.destination.exists()
        managed.load_model.assert_not_called()
