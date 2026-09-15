"""Qwen3-ASR の重みが**管理 HF cache**へ落ち、そこからロードされること (Issue #428)。

production の経路::

    ModelManager.get_huggingface_cache_dir()          # <cache_root>/huggingface/hub
      → snapshot_download(repo_id, cache_dir=<上記>, max_workers=1)
      → marker に snapshot path を書く (成功後のみ)
      → Qwen3ASRModel.from_pretrained(<ローカル snapshot path>)

固定する契約:

* ``snapshot_download`` へ渡る ``cache_dir`` は管理 cache であり、``huggingface_hub`` の
  既定 cache (``HF_HUB_CACHE``) ではない。**既定 cache への silent fallback はしない**
* ``from_pretrained`` へ渡るのは **repo ID ではなくローカル snapshot path** —
  qwen-asr の ``AutoProcessor`` は ``cache_dir`` を受けないので、repo ID を渡すと
  processor 側だけ既定 cache へ行く
* marker は「解決済み snapshot の記録」であり、**marker だけでは cache hit にしない**。
  snapshot が実在して初めて hit。旧形式 (``model=...``) も hit にしない
* ダウンロード失敗時に marker を残さない (以前は実ダウンロードの前に書いていた)
* ``HF_HOME`` / ``HF_HUB_CACHE`` を書き換えない

ネットワークもモデルも使わない。``snapshot_download`` と ``qwen_asr`` は差し替える。
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.resources import _reset_resources_for_tests

REPO_ID = "Qwen/Qwen3-ASR-0.6B"
REPO_DIR = "models--Qwen--Qwen3-ASR-0.6B"
SHA = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"


def _make_snapshot(hub: Path, sha: str = SHA) -> Path:
    """管理 cache に ``refs/main`` + ``snapshots/<sha>/config.json`` を置く。"""
    repo = hub / REPO_DIR
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(sha, encoding="utf-8")
    snapshot = repo / "snapshots" / sha
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3_asr"}', encoding="utf-8")
    return snapshot


class _FakeSnapshotDownload:
    """``huggingface_hub.snapshot_download`` の代役。呼び出し引数を記録し、``cache_dir``
    へ snapshot を書いて返す (実 library と同じ戻り値の形)。"""

    def __init__(self, *, fail: Exception | None = None):
        self.calls: list[tuple[tuple, dict]] = []
        self.fail = fail

    def __call__(self, repo_id, **kwargs):
        self.calls.append(((repo_id,), dict(kwargs)))
        if self.fail is not None:
            raise self.fail
        cache_dir = Path(kwargs["cache_dir"])
        return str(_make_snapshot(cache_dir))


@pytest.fixture
def managed(tmp_path, monkeypatch):
    """管理 root を tmp へ向け、既定 HF cache を**空の別 dir** に固定する。"""
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    default_hub = tmp_path / "default-hf-hub"  # 空のまま — ここに落ちたら fallback
    default_hub.mkdir()
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sentinel-hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(default_hub))
    _reset_resources_for_tests()
    ModelMemoryCache.clear()

    # qwen_asr は engines-qwen3asr extra。unit test では差し替える。
    fake_qwen = types.ModuleType("qwen_asr")
    fake_qwen.Qwen3ASRModel = MagicMock(name="Qwen3ASRModel")
    fake_qwen.Qwen3ASRModel.from_pretrained.return_value = MagicMock(name="loaded_model")
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen)

    with patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability", return_value=True):
        yield types.SimpleNamespace(
            models_root=models_root,
            cache_root=cache_root,
            managed_hub=cache_root / "huggingface" / "hub",
            default_hub=default_hub,
            marker=models_root / "Qwen--Qwen3-ASR-0.6B.marker",
            from_pretrained=fake_qwen.Qwen3ASRModel.from_pretrained,
        )
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _engine():
    from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

    return Qwen3ASREngine(device="cpu")


def _load_with(fake: _FakeSnapshotDownload):
    with patch("huggingface_hub.snapshot_download", fake):
        engine = _engine()
        engine.load_model()
    return engine


class TestColdCache:
    def test_snapshot_is_resolved_into_managed_cache(self, managed):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        (repo_id,), kwargs = fake.calls[0]
        assert repo_id == REPO_ID
        assert Path(kwargs["cache_dir"]) == managed.managed_hub, (
            "cache_dir= は ModelManager.get_huggingface_cache_dir() (管理 cache) でなければならない"
        )
        assert Path(kwargs["cache_dir"]) != managed.default_hub, "既定 HF cache へ落としてはならない"
        assert (managed.managed_hub / REPO_DIR / "snapshots" / SHA / "config.json").is_file()
        assert not any(managed.default_hub.iterdir()), "既定 HF cache に何も書かれていない"

    def test_single_worker_avoids_hf_hub_symlink_probe_race(self, managed):
        """hf_hub 0.36.0: fresh な cache dir へ複数 worker で落とすと symlink 可否判定が
        thread 間で競合し Windows (Developer Mode 無し) で WinError 1314 になる (実測)。"""
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        _, kwargs = fake.calls[0]
        assert kwargs.get("max_workers") == 1

    def test_offline_is_left_to_huggingface_hub(self, managed):
        """HF_HUB_OFFLINE=1 なら library が管理 cache だけから解決し、無ければ fail loud
        (LocalEntryNotFoundError) になる。engine 側で local_files_only を上書きしない。"""
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        _, kwargs = fake.calls[0]
        assert "local_files_only" not in kwargs

    def test_from_pretrained_receives_local_snapshot_not_repo_id(self, managed):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        managed.from_pretrained.assert_called_once()
        (target,), kwargs = managed.from_pretrained.call_args
        assert target != REPO_ID, "repo ID を渡すと processor 側が既定 cache へ行く"
        assert Path(target) == managed.managed_hub / REPO_DIR / "snapshots" / SHA
        assert kwargs == {"device_map": "cpu"}

    def test_marker_records_snapshot_path(self, managed):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert managed.marker.is_file()
        recorded = Path(managed.marker.read_text(encoding="utf-8").strip())
        assert recorded == (managed.managed_hub / REPO_DIR / "snapshots" / SHA).resolve()

    def test_environment_is_not_rewritten(self, managed, tmp_path):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert os.environ["HF_HOME"] == str(tmp_path / "sentinel-hf-home")
        assert os.environ["HF_HUB_CACHE"] == str(managed.default_hub)


class TestCacheHit:
    def test_marker_with_existing_snapshot_skips_download(self, managed):
        snapshot = _make_snapshot(managed.managed_hub)
        managed.models_root.mkdir(parents=True, exist_ok=True)
        managed.marker.write_text(str(snapshot.resolve()), encoding="utf-8")
        fake = _FakeSnapshotDownload(fail=AssertionError("cache hit なので呼ばれてはならない"))

        _load_with(fake)

        assert fake.calls == []
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == snapshot.resolve()

    def test_marker_without_snapshot_is_not_a_hit(self, managed):
        """marker だけ残っている (以前の「ダウンロード前に marker を書く」実装の残骸や
        cache を消した後) 状態を cache hit にしない。"""
        managed.models_root.mkdir(parents=True, exist_ok=True)
        managed.marker.write_text(
            str(managed.managed_hub / REPO_DIR / "snapshots" / "deadbeef"), encoding="utf-8"
        )
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1, "snapshot が無ければ再解決する"
        assert Path(managed.marker.read_text(encoding="utf-8").strip()).is_dir()

    def test_legacy_marker_is_not_a_hit(self, managed):
        """#428 以前の marker (`model=...` / `device=...`) は snapshot path として読めない
        → 再解決される。これが Migration の挙動 (既定 cache からは移設しない)。"""
        managed.models_root.mkdir(parents=True, exist_ok=True)
        managed.marker.write_text(f"model={REPO_ID}\ndevice=cpu", encoding="utf-8")
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        recorded = managed.marker.read_text(encoding="utf-8")
        assert "model=" not in recorded and Path(recorded.strip()).is_dir()

    def test_snapshot_without_marker_is_resolved_offline_without_network_download(self, managed):
        """管理 cache に snapshot はあるが marker が無い (別プロセスが落とした / marker を
        消した) → snapshot_download が呼ばれるが、library は refs/main から解決するだけで
        ネットワークへ出ない (HF_HUB_OFFLINE=1 でも成功する: 実 Qwen で実測、#428)。
        ここでは「marker が無いだけで再ダウンロードを強制しない」形 (cache_dir= が同じ)
        を固定する。"""
        snapshot = _make_snapshot(managed.managed_hub)
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        _, kwargs = fake.calls[0]
        assert Path(kwargs["cache_dir"]) == managed.managed_hub
        assert Path(managed.marker.read_text(encoding="utf-8").strip()) == snapshot.resolve()


class TestFailure:
    def test_failed_resolution_leaves_no_marker(self, managed):
        """以前は実ダウンロードの**前**に marker を書いていたので、失敗後も「cached」と
        判定されて次回のロードで落ち続けた。"""
        fake = _FakeSnapshotDownload(fail=RuntimeError("network down"))

        with patch("huggingface_hub.snapshot_download", fake):
            engine = _engine()
            with pytest.raises(RuntimeError, match="network down"):
                engine.load_model()

        assert not managed.marker.exists()
        managed.from_pretrained.assert_not_called()

    def test_snapshot_without_config_is_rejected(self, managed):
        def broken(repo_id, **kwargs):
            empty = Path(kwargs["cache_dir"]) / REPO_DIR / "snapshots" / SHA
            empty.mkdir(parents=True, exist_ok=True)
            return str(empty)

        with patch("huggingface_hub.snapshot_download", broken):
            engine = _engine()
            with pytest.raises(RuntimeError, match="config.json"):
                engine.load_model()

        assert not managed.marker.exists()


class TestRemovedApi:
    def test_huggingface_cache_context_manager_is_gone(self):
        from livecap_cli.resources.model_manager import ModelManager

        assert not hasattr(ModelManager, "huggingface_cache"), (
            "HF_HOME を実行時に書き換える旧 API は効かないので削除した (#428)"
        )
