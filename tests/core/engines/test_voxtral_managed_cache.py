"""Voxtral の正本が **models_root の flattened dir 1 部だけ**になること (Issue #456)。

以前は ``from_pretrained(repo, cache_dir=<cache_root>/huggingface/hub/transformers)`` で
snapshot (hub 階層、8.8 GB) を落とした後 ``save_pretrained()`` で ``<models_root>/mistralai--…/``
へ**もう 1 部**書いていた (合計 17.6 GB)。さらに repo には transformers が使わない
``consolidated.safetensors`` (9.3 GB) があり、絞らずに取ると 1 モデルで 18.7 GB になる。

固定する契約:

* ``fetch_repo_dir(allow_patterns=ALLOW_PATTERNS)`` で ``config / generation_config /
  preprocessor_config / tekken.json / model.safetensors.index.json / model-*.safetensors`` だけを
  ``<models_root>/mistralai--Voxtral-Mini-3B-2507/`` へ置く。``consolidated.safetensors`` は取らない
* ``from_pretrained`` / ``AutoProcessor.from_pretrained`` はその dir を受ける (``save_pretrained`` は呼ばない)
* 既存の ``save_pretrained()`` 出力 (manifest 無し) は adopt、``<cache_root>/huggingface/hub/transformers``
  / ``<cache_root>/huggingface/transformers`` (0.1.0) の snapshot は取り込んで消す
* load 失敗で manifest を無効化する (self-heal)

``transformers`` と ``snapshot_download`` は差し替える。ネットワークもモデルも使わない。
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
from tests.core.model_root_fixtures import FakeSnapshotDownloadLocalDir, write_hub_snapshot, write_repo_dir

REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
DEST_NAME = "mistralai--Voxtral-Mini-3B-2507"
REPO_FILES = {
    "config.json": b"{}",
    "generation_config.json": b"{}",
    "preprocessor_config.json": b"{}",
    "tekken.json": b"{}",
    "model.safetensors.index.json": b"{}",
    "model-00001-of-00002.safetensors": b"w" * 64,
    "model-00002-of-00002.safetensors": b"w" * 32,
    "consolidated.safetensors": b"m" * 96,  # mistral 形式 (transformers は使わない、9.3 GB)
    "params.json": b"{}",
    "README.md": b"#",
}
MODEL_FILES = {k: v for k, v in REPO_FILES.items() if k not in ("consolidated.safetensors", "params.json", "README.md")}


@pytest.fixture
def managed(model_root_sentinels, monkeypatch):
    roots = model_root_sentinels
    ModelMemoryCache.clear()

    fake_tf = types.ModuleType("transformers")
    fake_tf.VoxtralForConditionalGeneration = MagicMock(name="VoxtralForConditionalGeneration")
    fake_tf.AutoProcessor = MagicMock(name="AutoProcessor")
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "mistral_common", types.ModuleType("mistral_common"))

    with patch("livecap_cli.engines.voxtral_engine.check_transformers_availability", return_value=True):
        yield types.SimpleNamespace(
            models_root=roots.models_root,
            cache_root=roots.cache_root,
            default_hub=roots.default_hub,
            staging_root=roots.staging_root,
            hub_root=roots.hub_root,
            destination=roots.models_root / DEST_NAME,
            from_pretrained=fake_tf.VoxtralForConditionalGeneration.from_pretrained,
            processor_from_pretrained=fake_tf.AutoProcessor.from_pretrained,
        )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine():
    from livecap_cli.engines.voxtral_engine import VoxtralEngine

    return VoxtralEngine(device="cpu", language="en")


def _fake(**kwargs):
    return FakeSnapshotDownloadLocalDir(files=REPO_FILES, **kwargs)


def _load_with(fake):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = _engine()
        engine.load_model()
    return engine


def _manifest(managed) -> ms.Manifest:
    manifest = ms.validate_repo_dir(managed.destination, repo_id=REPO_ID)
    assert manifest is not None
    return manifest


class TestColdCache:
    def test_fetches_only_transformers_files_into_models_root(self, managed):
        fake = _fake()

        _load_with(fake)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID
        assert Path(call["local_dir"]) == managed.staging_root / DEST_NAME / "download"
        assert Path(call["cache_dir"]) == managed.hub_root
        assert call["max_workers"] == 1
        files = sorted(f.path for f in _manifest(managed).files)
        assert files == sorted(MODEL_FILES)
        assert "consolidated.safetensors" not in files, "9.3 GB の mistral 形式は取らない"
        assert not (managed.cache_root / "huggingface" / "hub" / "transformers").exists()
        assert not (managed.staging_root / DEST_NAME).exists()

    def test_from_pretrained_receives_models_root_dir_and_no_save_pretrained(self, managed):
        _load_with(_fake())

        (target,), kwargs = managed.from_pretrained.call_args
        assert Path(target) == managed.destination
        assert "cache_dir" not in kwargs, "hub 階層へは落とさない"
        (ptarget,), _ = managed.processor_from_pretrained.call_args
        assert Path(ptarget) == managed.destination
        model = managed.from_pretrained.return_value.to.return_value
        model.save_pretrained.assert_not_called()
        managed.processor_from_pretrained.return_value.save_pretrained.assert_not_called()


class TestCacheHitAndAdopt:
    def test_valid_manifest_skips_download(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)

        _load_with(_fake(fail=AssertionError("hit")))

        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == managed.destination

    def test_existing_save_pretrained_output_is_adopted(self, managed):
        """#456 以前の ``save_pretrained()`` 出力 (manifest 無し、必要ファイルは揃う) を再取得しない。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, with_manifest=False)

        _load_with(_fake(fail=AssertionError("adopt できるので呼ばれない")))

        assert _manifest(managed).source == "adopted"

    def test_missing_shard_is_not_a_hit(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        (managed.destination / "model-00002-of-00002.safetensors").unlink()
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1


class TestLegacyMigration:
    @pytest.mark.parametrize(
        "hub_subpath",
        [("huggingface", "hub", "transformers"), ("huggingface", "transformers")],
        ids=["0.2.0-hub-transformers", "0.1.0-transformers"],
    )
    def test_transformers_cache_snapshot_is_migrated_and_removed(self, managed, hub_subpath):
        hub = managed.cache_root.joinpath(*hub_subpath)
        snapshot = write_hub_snapshot(hub, REPO_ID, REPO_FILES)

        _load_with(_fake(fail=AssertionError("旧配置から取り込めるので再ダウンロードしない")))

        manifest = _manifest(managed)
        assert manifest.source == "migrated"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES), "consolidated は取り込まない"
        assert not snapshot.exists() and not (hub / f"models--{DEST_NAME}").exists()

    def test_adopted_flattened_dir_removes_duplicate_transformers_cache(self, managed):
        """save_pretrained 出力 + transformers cache の二重保持 → adopt して cache 側を消す。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, with_manifest=False)
        hub = managed.cache_root / "huggingface" / "hub" / "transformers"
        snapshot = write_hub_snapshot(hub, REPO_ID, REPO_FILES)

        _load_with(_fake(fail=AssertionError("hit")))

        assert _manifest(managed).source == "adopted"
        assert not snapshot.exists()


class TestFailure:
    def test_load_failure_invalidates_manifest(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        managed.from_pretrained.side_effect = OSError("corrupt shard")

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
        managed.from_pretrained.assert_not_called()
