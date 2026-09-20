"""ReazonSpeech (sherpa-onnx) の正本が **models_root の flattened dir 1 部だけ**で、
必要 4 ファイルだけを HF から取ること (Issue #456)。

以前は float32 が repo 全体 (775 MB、int8 の encoder 込み) を ``<cache_root>/huggingface`` へ
落としてから ``<models_root>`` へ copy (二重保持)、int8 は GitHub の tarball (713 MB、float32
encoder + test_wavs 込み) を ``<cache_root>/downloads`` に**残したまま**展開していた。さらに
``load_model()`` の override が root 側 dir を ``<models_root>/reazonspeech/`` へ移してから
template を呼ぶので、template が root 側で miss して**再ダウンロード**していた。

固定する契約:

* int8 / float32 とも ``reazon-research/reazonspeech-k2-v2`` から ``required_files()`` の
  4 ファイルだけを ``fetch_repo_dir(allow_patterns=…)`` で取る (tarball 経路は無い)
* dir 名は #456 以前と同じ (float32: ``reazon-research--reazonspeech-k2-v2``、
  int8: ``sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01``)。manifest の ``variant`` で区別
* ``load_model()`` の override (inverted workaround) は無い。engine subdir の重複
  (``<models_root>/reazonspeech/<name>``) と 0.1.0 の ``<cache_root>/huggingface/models--…`` は
  取り込んで消す
* ``from_transducer`` 失敗で manifest を無効化する (self-heal)

``sherpa_onnx`` と ``snapshot_download`` は差し替える。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines import model_store as ms
from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.engines.reazonspeech_cache import required_files
from livecap_cli.resources import _reset_resources_for_tests
from tests.core.model_root_fixtures import FakeSnapshotDownloadLocalDir, write_hub_snapshot, write_repo_dir

REPO_ID = "reazon-research/reazonspeech-k2-v2"
FLOAT32_DIR = "reazon-research--reazonspeech-k2-v2"
INT8_DIR = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"

#: 実 repo のファイル (int8 / float32 の両方 + 不要な test_wavs 相当)。
REPO_FILES = {
    "tokens.txt": b"<blk>\n",
    "encoder-epoch-99-avg-1.onnx": b"f" * 640,
    "decoder-epoch-99-avg-1.onnx": b"f" * 32,
    "joiner-epoch-99-avg-1.onnx": b"f" * 16,
    "encoder-epoch-99-avg-1.int8.onnx": b"i" * 160,
    "joiner-epoch-99-avg-1.int8.onnx": b"i" * 8,
    "README.md": b"#",
    "test_wavs/1.wav": b"RIFF",
}


def _model_files(use_int8: bool) -> dict:
    return {name: REPO_FILES[name] for name in required_files(use_int8=use_int8).values()}


@pytest.fixture
def managed(model_root_sentinels, monkeypatch):
    roots = model_root_sentinels
    ModelMemoryCache.clear()

    fake_sherpa = types.ModuleType("sherpa_onnx")
    fake_sherpa.__version__ = "1.12.9"
    fake_sherpa.OfflineRecognizer = MagicMock(name="OfflineRecognizer")
    fake_sherpa.OfflineRecognizer.from_transducer.return_value = MagicMock(name="recognizer")
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)

    yield types.SimpleNamespace(
        models_root=roots.models_root,
        cache_root=roots.cache_root,
        staging_root=roots.staging_root,
        hub_root=roots.hub_root,
        from_transducer=fake_sherpa.OfflineRecognizer.from_transducer,
    )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine(use_int8: bool = False):
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

    return ReazonSpeechEngine(device="cpu", use_int8=use_int8)


def _fake(**kwargs):
    return FakeSnapshotDownloadLocalDir(files=REPO_FILES, **kwargs)


def _load_with(fake, use_int8: bool = False):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = _engine(use_int8)
        engine.load_model()
    return engine


@pytest.mark.parametrize("use_int8,dest_name", [(False, FLOAT32_DIR), (True, INT8_DIR)], ids=["float32", "int8"])
class TestColdCache:
    def test_fetches_only_required_files_from_hf(self, managed, use_int8, dest_name):
        fake = _fake()

        _load_with(fake, use_int8)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID, "int8 も tarball ではなく HF repo から"
        assert sorted(call["allow_patterns"]) == sorted(required_files(use_int8=use_int8).values())
        assert Path(call["local_dir"]) == managed.staging_root / dest_name / "download"
        destination = managed.models_root / dest_name
        manifest = ms.validate_repo_dir(destination, repo_id=REPO_ID, variant="int8" if use_int8 else "float32")
        assert manifest is not None
        assert sorted(f.path for f in manifest.files) == sorted(_model_files(use_int8))
        assert not list(managed.cache_root.glob("downloads/*.tar.bz2"))

    def test_from_transducer_receives_files_in_models_root(self, managed, use_int8, dest_name):
        _load_with(_fake(), use_int8)

        kwargs = managed.from_transducer.call_args.kwargs
        names = required_files(use_int8=use_int8)
        for key in ("tokens", "encoder", "decoder", "joiner"):
            assert Path(kwargs[key]) == (managed.models_root / dest_name / names[key]).resolve()


class TestCacheHitAndAdopt:
    def test_valid_manifest_skips_download(self, managed):
        write_repo_dir(managed.models_root / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, variant="float32")

        _load_with(_fake(fail=AssertionError("hit")))

        managed.from_transducer.assert_called_once()

    def test_pre_456_root_dir_without_manifest_is_adopted(self, managed):
        write_repo_dir(managed.models_root / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, variant="float32", with_manifest=False)

        _load_with(_fake(fail=AssertionError("adopt できるので呼ばれない")))

        manifest = ms.validate_repo_dir(managed.models_root / FLOAT32_DIR, repo_id=REPO_ID, variant="float32")
        assert manifest is not None and manifest.source == "adopted"

    def test_int8_and_float32_manifests_do_not_cross_hit(self, managed):
        """variant が違えば同じ repo でも hit にしない。"""
        write_repo_dir(managed.models_root / INT8_DIR, _model_files(True), repo_id=REPO_ID, variant="float32")

        assert not _engine(use_int8=True)._is_model_cached(managed.models_root / INT8_DIR)


class TestLegacyMigration:
    def test_engine_subdir_duplicate_is_migrated_and_removed(self, managed):
        """旧 ``load_model()`` override が作った ``<models_root>/reazonspeech/<name>``。"""
        legacy = write_repo_dir(managed.models_root / "reazonspeech" / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, with_manifest=False)

        _load_with(_fake(fail=AssertionError("旧配置から取り込めるので呼ばれない")))

        manifest = ms.validate_repo_dir(managed.models_root / FLOAT32_DIR, repo_id=REPO_ID, variant="float32")
        assert manifest is not None and manifest.source == "migrated"
        assert not legacy.exists()
        assert not (managed.models_root / "reazonspeech").exists(), "空になった engine subdir も消す"

    def test_010_huggingface_snapshot_is_migrated_and_removed(self, managed):
        """0.1.0 の ``<cache_root>/huggingface/models--reazon-research--…``。"""
        snapshot = write_hub_snapshot(managed.cache_root / "huggingface", REPO_ID, REPO_FILES)

        _load_with(_fake(fail=AssertionError("hit")))

        manifest = ms.validate_repo_dir(managed.models_root / FLOAT32_DIR, repo_id=REPO_ID, variant="float32")
        assert manifest is not None and manifest.source == "migrated"
        assert sorted(f.path for f in manifest.files) == sorted(_model_files(False)), "int8 側と test_wavs は取り込まない"
        assert not snapshot.exists()

    def test_root_dir_plus_subdir_duplicate_keeps_root_and_removes_subdir(self, managed):
        """実測の形: root 側 748 MB + ``reazonspeech/`` 748 MB の二重保持。"""
        write_repo_dir(managed.models_root / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, with_manifest=False)
        dup = write_repo_dir(managed.models_root / "reazonspeech" / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, with_manifest=False)

        _load_with(_fake(fail=AssertionError("hit")))

        assert ms.validate_repo_dir(managed.models_root / FLOAT32_DIR, repo_id=REPO_ID, variant="float32").source == "adopted"
        assert not dup.exists()


class TestLegacyInt8Archive:
    """#456 PR 1 手順: `<cache_root>/downloads/*.tar.bz2` (旧 int8 経路の 713 MB) は初回起動時に削除。
    **int8 の正本が validator を通った後にだけ**消す (float32 だけの利用では触らない)。"""

    def _archive(self, managed) -> Path:
        downloads = managed.cache_root / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloads / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
        archive.write_bytes(b"t" * 32)
        return archive

    def test_removed_after_int8_cache_hit(self, managed):
        archive = self._archive(managed)
        write_repo_dir(managed.models_root / INT8_DIR, _model_files(True), repo_id=REPO_ID, variant="int8")

        _load_with(_fake(fail=AssertionError("hit")), use_int8=True)

        assert not archive.exists()

    def test_removed_after_int8_cold_download(self, managed):
        archive = self._archive(managed)

        _load_with(_fake(), use_int8=True)

        assert not archive.exists()
        assert ms.validate_repo_dir(managed.models_root / INT8_DIR, repo_id=REPO_ID, variant="int8") is not None

    def test_kept_when_int8_download_fails(self, managed):
        archive = self._archive(managed)

        with patch("huggingface_hub.snapshot_download", _fake(fail=RuntimeError("network down"))):
            with pytest.raises(RuntimeError, match="network down"):
                _engine(use_int8=True).load_model()

        assert archive.exists(), "正本が validator を通る前には消さない"

    def test_kept_for_float32_only_usage(self, managed):
        archive = self._archive(managed)
        write_repo_dir(managed.models_root / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, variant="float32")

        _load_with(_fake(fail=AssertionError("hit")), use_int8=False)

        assert archive.exists(), "int8 の正本が無いうちは触らない (info の残骸一覧に出る)"


class TestFailure:
    def test_load_failure_invalidates_manifest(self, managed):
        destination = write_repo_dir(managed.models_root / FLOAT32_DIR, _model_files(False), repo_id=REPO_ID, variant="float32")
        managed.from_transducer.side_effect = RuntimeError("onnx protobuf parse failed")

        with patch("huggingface_hub.snapshot_download", _fake(fail=AssertionError("hit"))):
            with pytest.raises(RuntimeError, match="protobuf"):
                _engine().load_model()

        assert ms.read_manifest(destination).source == ms.INVALIDATED_SOURCE
        assert not _engine()._is_model_cached(destination)

    def test_no_load_model_override(self):
        from livecap_cli.engines.base_engine import BaseEngine
        from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine

        assert ReazonSpeechEngine.load_model is BaseEngine.load_model, (
            "root → engine subdir へ移してから template を呼ぶ inverted workaround は再ダウンロードの原因 (#456)"
        )
