"""Qwen3-ASR の正本が **models_root の flattened dir** に置かれ、そこからロードされること
(Issue #428 → #456)。

production の経路::

    <models_root>/Qwen--Qwen3-ASR-0.6B/          # 正本 (重み + tokenizer + livecap-manifest.json)
      ← legacy_model_layouts.migrate_dir()        # 旧配置 (0.2.0 hub snapshot + marker) があれば取り込む
      ← hf_cache.fetch_repo_dir()                 # 無ければ staging (<cache_root>/downloads/…) 経由で取得
      → Qwen3ASRModel.from_pretrained(<正本 dir>)

固定する契約:

* ``snapshot_download`` は ``local_dir=<cache_root>/downloads/<name>/download`` +
  ``cache_dir=<cache_root>/huggingface/hub`` (transient lookup) + ``max_workers=1`` で呼ぶ。
  既定 HF cache (``HF_HUB_CACHE``) には何も書かない
* ``from_pretrained`` へ渡るのは **repo ID ではなく正本 dir** — qwen-asr の ``AutoProcessor`` は
  ``cache_dir`` を受けないので、repo ID を渡すと processor 側だけ既定 cache へ行く
* cache hit は **manifest の全ファイルがサイズ一致で実在**するときだけ。「非空 dir」は hit にしない。
  manifest 無しでも必要ファイルが揃う dir はその場で採用 (adopt) して再取得しない
* 旧 0.2.0 配置 (``<cache_root>/huggingface/hub/models--…`` + ``*.marker``) は cold load で
  取り込み、取り込み後に消す (二重保持の解消、再ダウンロード無し)
* 取得失敗時に正本を作らない (staging は resume 用に残す)。ロード失敗時は manifest を無効化
  して次回再取得する (self-heal)
* ``models_root`` に transient (``.cache`` / ``*.metadata`` / ``*.lock`` / ``*.incomplete``) を
  残さない (``model_root_sentinels`` の teardown)
* ``HF_HOME`` / ``HF_HUB_CACHE`` を書き換えない
* 0.1.0 が既定 HF cache (root の**外**) に落とした snapshot は cold load で **copy** して取り込み、
  外は 1 byte も変えない (#453)

ネットワークもモデルも使わない。``snapshot_download`` と ``qwen_asr`` は差し替える。
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines import model_store as ms
from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.resources import _reset_resources_for_tests
from tests.core.model_root_fixtures import FakeSnapshotDownloadLocalDir, file_fingerprints, write_hub_snapshot, write_repo_dir

REPO_ID = "Qwen/Qwen3-ASR-0.6B"
DEST_NAME = "Qwen--Qwen3-ASR-0.6B"
#: 実 repo と同じ顔ぶれ (README / .gitattributes は取らない)。
REPO_FILES = {
    "config.json": b'{"model_type": "qwen3_asr"}',
    "model.safetensors": b"w" * 512,
    "tokenizer_config.json": b"{}",
    "tokenizer.json": b"{}",
    "preprocessor_config.json": b"{}",
    "README.md": b"# readme",
    ".gitattributes": b"*.safetensors filter=lfs",
}
MODEL_FILES = {k: v for k, v in REPO_FILES.items() if k not in ("README.md", ".gitattributes")}


@pytest.fixture
def managed(model_root_sentinels, monkeypatch):
    """root を tmp へ (sentinel 付き)、qwen_asr を差し替える。"""
    roots = model_root_sentinels
    ModelMemoryCache.clear()

    fake_qwen = types.ModuleType("qwen_asr")
    fake_qwen.Qwen3ASRModel = MagicMock(name="Qwen3ASRModel")
    fake_qwen.Qwen3ASRModel.from_pretrained.return_value = MagicMock(name="loaded_model")
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen)

    with patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability", return_value=True):
        yield types.SimpleNamespace(
            models_root=roots.models_root,
            cache_root=roots.cache_root,
            default_hub=roots.default_hub,
            staging_root=roots.staging_root,
            hub_root=roots.hub_root,
            external_hub=roots.external_hub,
            destination=roots.models_root / DEST_NAME,
            from_pretrained=fake_qwen.Qwen3ASRModel.from_pretrained,
        )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine():
    from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

    return Qwen3ASREngine(device="cpu")


def _fake(**kwargs) -> FakeSnapshotDownloadLocalDir:
    return FakeSnapshotDownloadLocalDir(files=REPO_FILES, **kwargs)


def _load_with(fake):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = _engine()
        engine.load_model()
    return engine


def _manifest(managed) -> ms.Manifest:
    manifest = ms.validate_repo_dir(managed.destination, repo_id=REPO_ID)
    assert manifest is not None, "正本 dir が manifest 込みで揃っていない"
    return manifest


class TestColdCache:
    def test_downloads_via_staging_into_models_root(self, managed):
        fake = _fake()

        _load_with(fake)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID
        assert Path(call["local_dir"]) == managed.staging_root / DEST_NAME / "download", "staging は cache_root 側"
        assert Path(call["cache_dir"]) == managed.hub_root, "lookup 先も管理 cache (既定 HF cache へ落とさない)"
        assert call["max_workers"] == 1, "hf_hub の symlink 判定 race (huggingface_hub#4915) の回避"
        assert "README.md" in call["ignore_patterns"] and ".gitattributes" in call["ignore_patterns"]
        assert "local_files_only" not in call, "offline は HF_HUB_OFFLINE に任せる"
        assert managed.destination.is_dir()
        assert not (managed.staging_root / DEST_NAME).exists(), "成功後は staging を消す"
        assert not any(managed.default_hub.iterdir())

    def test_manifest_lists_model_files_only(self, managed):
        _load_with(_fake())

        manifest = _manifest(managed)
        assert manifest.source == "download"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES)
        assert manifest.commit_sha == "c" * 40
        assert not (managed.destination / ".cache").exists(), "HF の管理メタデータを正本へ持ち込まない"

    def test_from_pretrained_receives_models_root_dir_not_repo_id(self, managed):
        _load_with(_fake())

        managed.from_pretrained.assert_called_once()
        (target,), kwargs = managed.from_pretrained.call_args
        assert target != REPO_ID, "repo ID を渡すと processor 側が既定 cache へ行く"
        assert Path(target) == managed.destination
        assert Path(target).is_relative_to(managed.models_root), "正本は models_root 配下"
        assert not Path(target).is_relative_to(managed.cache_root), "cache_root は一時 (消しても良い) 領域"
        assert kwargs == {"device_map": "cpu"}

    def test_environment_is_not_rewritten(self, managed):
        _load_with(_fake())

        assert os.environ["HF_HUB_CACHE"] == str(managed.default_hub)
        assert os.environ["HF_HUB_OFFLINE"] == "1"


class TestCacheHit:
    def test_valid_manifest_skips_download(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        fake = _fake(fail=AssertionError("cache hit なので呼ばれてはならない"))

        _load_with(fake)

        assert fake.calls == []
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == managed.destination

    def test_complete_dir_without_manifest_is_adopted(self, managed):
        """manifest だけ無い (別プロセス / 手動配置) → その場で採用し、再取得しない。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID, with_manifest=False)
        fake = _fake(fail=AssertionError("adopt できるので呼ばれてはならない"))

        _load_with(fake)

        assert fake.calls == []
        assert _manifest(managed).source == "adopted"

    def test_missing_weights_is_not_a_hit(self, managed):
        """config.json はあるが manifest に記録した重みが欠ける → miss → 再取得。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        (managed.destination / "model.safetensors").unlink()
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1
        assert (managed.destination / "model.safetensors").stat().st_size == 512
        assert _manifest(managed).source == "download"

    def test_size_mismatch_is_not_a_hit(self, managed):
        """ファイルは全部あるが size が manifest と違う (途中で切れた) → miss。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        (managed.destination / "model.safetensors").write_bytes(b"w" * 100)
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1

    def test_non_empty_dir_without_required_files_is_not_a_hit(self, managed):
        """「非空 dir」では hit にしない (#456 の中心的な規則)。"""
        managed.destination.mkdir(parents=True)
        (managed.destination / "config.json").write_text("{}", encoding="utf-8")
        assert not _engine()._is_model_cached(managed.destination)
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1
        assert _manifest(managed) is not None

    def test_manifest_for_another_repo_is_not_a_hit(self, managed):
        write_repo_dir(managed.destination, MODEL_FILES, repo_id="Qwen/Qwen3-ASR-1.7B")

        assert not _engine()._is_model_cached(managed.destination)


class TestLegacyMigration:
    """0.2.0 (#428) の配置 = ``<cache_root>/huggingface/hub/models--…`` + ``<models_root>/*.marker``。"""

    def _legacy_020(self, managed) -> tuple[Path, Path]:
        snapshot = write_hub_snapshot(managed.hub_root, REPO_ID, REPO_FILES)
        marker = managed.models_root / f"{DEST_NAME}.marker"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"snapshot": "x", "files": []}', encoding="utf-8")
        return snapshot, marker

    def test_hub_snapshot_is_migrated_without_download(self, managed):
        snapshot, marker = self._legacy_020(managed)
        fake = _fake(fail=AssertionError("旧配置から取り込めるので再ダウンロードしない"))

        _load_with(fake)

        assert fake.calls == []
        manifest = _manifest(managed)
        assert manifest.source == "migrated"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES), "README 等は取り込まない"
        assert (managed.destination / "model.safetensors").read_bytes() == REPO_FILES["model.safetensors"]
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == managed.destination

    def test_legacy_copies_are_removed_after_successful_publish(self, managed):
        snapshot, marker = self._legacy_020(managed)

        _load_with(_fake(fail=AssertionError("hit")))

        assert not snapshot.exists() and not (managed.hub_root / f"models--{DEST_NAME}").exists()
        assert not marker.exists(), "marker は正本が確定したら不要"

    def test_cache_hit_still_removes_duplicate_legacy_copies(self, managed):
        """正本が既に valid (hit) でも、同じ repo の旧配置 (二重保持) は消す — 旧配置の整理は
        download phase ではなく cache 判定の前 (``_reconcile_legacy_layouts``) で毎回行う。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        snapshot, marker = self._legacy_020(managed)

        _load_with(_fake(fail=AssertionError("hit")))

        assert _manifest(managed).source == "download", "正本はそのまま"
        assert not snapshot.exists() and not marker.exists()

    def test_incomplete_legacy_snapshot_is_left_alone_and_download_runs(self, managed):
        """旧 snapshot に重みが無い (中断) → 取り込まず download。検証できない旧側は消さず、
        ``scan_legacy_layouts`` (``livecap-cli info``) に残骸として見える。"""
        from livecap_cli.engines.legacy_model_layouts import scan_legacy_layouts

        files = {k: v for k, v in REPO_FILES.items() if k != "model.safetensors"}
        snapshot = write_hub_snapshot(managed.hub_root, REPO_ID, files)
        fake = _fake()

        _load_with(fake)

        assert len(fake.calls) == 1
        assert _manifest(managed).source == "download"
        assert snapshot.is_dir(), "取り込めなかった旧配置は消さない (検証していないものは消さない)"
        assert [p for p, _ in scan_legacy_layouts(managed.models_root, managed.cache_root)] == [
            managed.hub_root / f"models--{DEST_NAME}"
        ]

    def test_load_failure_leaves_broken_dir_quarantined_on_refetch(self, managed):
        """self-heal 後の再取得で、壊れた旧正本は消さず ``<name>.invalid-*`` へ隔離される。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        ms.invalidate_manifest(managed.destination, reason="test")

        _load_with(_fake())

        quarantined = [p for p in managed.models_root.iterdir() if p.name.startswith(f"{DEST_NAME}.invalid-")]
        assert len(quarantined) == 1 and (quarantined[0] / "model.safetensors").is_file()
        assert _manifest(managed).source == "download"


class TestExternalCacheAdoption:
    """0.1.0 の配置 = 既定 HF cache ``~/.cache/huggingface/hub/models--Qwen--…`` (root の外、#453)。"""

    def test_default_hf_cache_snapshot_is_copied_without_download(self, managed):
        snapshot = write_hub_snapshot(managed.external_hub, REPO_ID, REPO_FILES)
        before = file_fingerprints(managed.external_hub)
        fake = _fake(fail=AssertionError("root の外の snapshot から取り込めるので再ダウンロードしない"))

        _load_with(fake)

        assert fake.calls == []
        manifest = _manifest(managed)
        assert manifest.source == "migrated"
        assert sorted(f.path for f in manifest.files) == sorted(MODEL_FILES)
        assert snapshot.is_dir() and file_fingerprints(managed.external_hub) == before, "外は消さない・変えない"
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == managed.destination

    def test_in_root_legacy_wins_over_the_external_cache(self, managed):
        inside = write_hub_snapshot(managed.hub_root, REPO_ID, {**REPO_FILES, "model.safetensors": b"inside" * 64})
        write_hub_snapshot(managed.external_hub, REPO_ID, {**REPO_FILES, "model.safetensors": b"outside" * 64})
        before = file_fingerprints(managed.external_hub)

        _load_with(_fake(fail=AssertionError("hit")))

        assert (managed.destination / "model.safetensors").read_bytes() == b"inside" * 64
        assert not inside.exists() and file_fingerprints(managed.external_hub) == before


class TestFailure:
    def test_failed_download_creates_no_destination(self, managed):
        fake = _fake(fail=RuntimeError("network down"))

        with patch("huggingface_hub.snapshot_download", fake):
            with pytest.raises(RuntimeError, match="network down"):
                _engine().load_model()

        assert not managed.destination.exists(), "取得に失敗した正本を作らない"
        assert (managed.staging_root / DEST_NAME / "download").is_dir(), "staging は resume 用に残す"
        managed.from_pretrained.assert_not_called()

    def test_snapshot_without_required_files_is_rejected(self, managed):
        fake = FakeSnapshotDownloadLocalDir(files={"config.json": b"{}", "README.md": b"#"})

        with patch("huggingface_hub.snapshot_download", fake):
            with pytest.raises(RuntimeError, match="必要ファイルが無い"):
                _engine().load_model()

        assert not managed.destination.exists()

    def test_load_failure_invalidates_manifest_for_self_heal(self, managed):
        """manifest に無い形で dir が壊れて from_pretrained が落ちたら manifest を消し、
        次回 load_model() で再取得できるようにする。"""
        write_repo_dir(managed.destination, MODEL_FILES, repo_id=REPO_ID)
        managed.from_pretrained.side_effect = OSError("corrupt safetensors header")

        with patch("huggingface_hub.snapshot_download", _fake(fail=AssertionError("hit なので呼ばれない"))):
            with pytest.raises(OSError, match="corrupt"):
                _engine().load_model()

        assert ms.validate_repo_dir(managed.destination) is None, "manifest が残ると永久に skip して落ち続ける"
        assert ms.read_manifest(managed.destination).source == ms.INVALIDATED_SOURCE
        assert not _engine()._is_model_cached(managed.destination)

        managed.from_pretrained.side_effect = None
        fake = _fake()
        _load_with(fake)
        assert len(fake.calls) == 1, "次回は再取得する (壊れた内容を adopt で再採用しない)"
        assert _manifest(managed).source == "download"


class TestRemovedApi:
    def test_huggingface_cache_context_manager_is_gone(self):
        from livecap_cli.resources.model_manager import ModelManager

        assert not hasattr(ModelManager, "huggingface_cache"), (
            "HF_HOME を実行時に書き換える旧 API は効かないので削除した (#428)"
        )

    def test_marker_api_is_gone(self):
        from livecap_cli.engines import hf_cache

        for name in ("resolve_snapshot", "read_marker", "write_marker", "invalidate_marker"):
            assert not hasattr(hf_cache, name), f"{name}: marker 方式は #456 で manifest + flattened dir に置き換えた"
