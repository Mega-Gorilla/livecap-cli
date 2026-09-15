"""Qwen3-ASR の重みが**管理 HF cache**へ落ち、そこからロードされること (Issue #428)。

production の経路::

    ModelManager.get_huggingface_cache_dir()          # <cache_root>/huggingface/hub
      → snapshot_download(repo_id, cache_dir=<上記>, max_workers=1)
      → marker に「hub root からの相対 path + ファイル一覧」を書く (成功後のみ)
      → Qwen3ASRModel.from_pretrained(<ローカル snapshot path>)

固定する契約:

* ``snapshot_download`` へ渡る ``cache_dir`` は管理 cache であり、``huggingface_hub`` の
  既定 cache (``HF_HUB_CACHE``) ではない。**既定 cache への silent fallback はしない**
* ``from_pretrained`` へ渡るのは **repo ID ではなくローカル snapshot path** —
  qwen-asr の ``AutoProcessor`` は ``cache_dir`` を受けないので、repo ID を渡すと
  processor 側だけ既定 cache へ行く
* marker は「解決済み snapshot の記録」であり、**marker だけでは cache hit にしない**。
  **現在の**管理 cache 配下に snapshot が実在し、記録した全ファイルが揃って初めて hit。
  旧形式 (``model=...``) も、cache root 変更前の root を指す marker も hit にしない
* ダウンロード失敗時に marker を残さない (以前は実ダウンロードの前に書いていた)。
  ロード失敗時は marker を無効化して次回再解決する (self-heal)
* ``HF_HOME`` / ``HF_HUB_CACHE`` を書き換えない

ネットワークもモデルも使わない。``snapshot_download`` と ``qwen_asr`` は差し替える。
"""

from __future__ import annotations

import json
import os
import shutil
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
#: 実 snapshot と同じ形 (config + 重み shard + tokenizer)。完全性確認の対象。
SNAPSHOT_FILES = {
    "config.json": '{"model_type": "qwen3_asr"}',
    "model.safetensors": "weights",
    "tokenizer.json": "{}",
}


def _make_snapshot(hub: Path, sha: str = SHA, files=None) -> Path:
    """管理 cache に ``refs/main`` + ``snapshots/<sha>/{config, weights, tokenizer}`` を置く。"""
    repo = hub / REPO_DIR
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(sha, encoding="utf-8")
    snapshot = repo / "snapshots" / sha
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, body in (files or SNAPSHOT_FILES).items():
        (snapshot / name).write_text(body, encoding="utf-8")
    return snapshot


def _write_marker(marker: Path, hub: Path, snapshot: Path) -> None:
    """production と同じ形式で marker を書く (形式は engine 側に単一ソース化)。"""
    from livecap_cli.engines.hf_cache import write_marker

    write_marker(marker, hub, snapshot)


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


def _point_roots(monkeypatch, models_root: Path, cache_root: Path) -> None:
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


@pytest.fixture
def managed(tmp_path, monkeypatch):
    """管理 root を tmp へ向け、既定 HF cache を**空の別 dir** に固定する。"""
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    default_hub = tmp_path / "default-hf-hub"  # 空のまま — ここに落ちたら fallback
    default_hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "sentinel-hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(default_hub))
    _point_roots(monkeypatch, models_root, cache_root)

    # qwen_asr は engines-qwen3asr extra。unit test では差し替える。
    fake_qwen = types.ModuleType("qwen_asr")
    fake_qwen.Qwen3ASRModel = MagicMock(name="Qwen3ASRModel")
    fake_qwen.Qwen3ASRModel.from_pretrained.return_value = MagicMock(name="loaded_model")
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen)

    with patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability", return_value=True):
        yield types.SimpleNamespace(
            tmp_path=tmp_path,
            models_root=models_root,
            cache_root=cache_root,
            managed_hub=cache_root / "huggingface" / "hub",
            default_hub=default_hub,
            marker=models_root / "Qwen--Qwen3-ASR-0.6B.marker",
            from_pretrained=fake_qwen.Qwen3ASRModel.from_pretrained,
            monkeypatch=monkeypatch,
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


def _marker_payload(marker: Path) -> dict:
    return json.loads(marker.read_text(encoding="utf-8"))


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
        """hf_hub 0.36.0 / 1.31.0: fresh な cache dir へ複数 worker で落とすと symlink 可否判定が
        thread 間で競合し Windows (Developer Mode 無し) で WinError 1314 になる
        (huggingface/huggingface_hub#4915)。"""
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
        assert Path(target) == (managed.managed_hub / REPO_DIR / "snapshots" / SHA).resolve()
        assert kwargs == {"device_map": "cpu"}

    def test_marker_records_relative_snapshot_and_manifest(self, managed):
        """marker は hub root からの**相対** path + ファイル一覧。絶対 path を書くと
        cache root 変更後に旧 root の snapshot を使い続ける (#446 レビュー)。"""
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        payload = _marker_payload(managed.marker)
        assert payload["snapshot"] == f"{REPO_DIR}/snapshots/{SHA}"
        assert not Path(payload["snapshot"]).is_absolute()
        assert payload["files"] == sorted(SNAPSHOT_FILES)

    def test_environment_is_not_rewritten(self, managed, tmp_path):
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert os.environ["HF_HOME"] == str(tmp_path / "sentinel-hf-home")
        assert os.environ["HF_HUB_CACHE"] == str(managed.default_hub)


class TestCacheHit:
    def test_marker_with_existing_snapshot_skips_download(self, managed):
        snapshot = _make_snapshot(managed.managed_hub)
        _write_marker(managed.marker, managed.managed_hub, snapshot)
        fake = _FakeSnapshotDownload(fail=AssertionError("cache hit なので呼ばれてはならない"))

        _load_with(fake)

        assert fake.calls == []
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target) == snapshot.resolve()

    def test_marker_without_snapshot_is_not_a_hit(self, managed):
        """marker だけ残っている (以前の「ダウンロード前に marker を書く」実装の残骸や
        cache を消した後) 状態を cache hit にしない。"""
        snapshot = _make_snapshot(managed.managed_hub)
        _write_marker(managed.marker, managed.managed_hub, snapshot)
        shutil.rmtree(snapshot)
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1, "snapshot が無ければ再解決する"
        assert (managed.managed_hub / _marker_payload(managed.marker)["snapshot"]).is_dir()

    def test_legacy_marker_is_not_a_hit(self, managed):
        """#428 以前の marker (`model=...` / `device=...`) は JSON として読めない
        → 再解決される。これが Migration の挙動 (既定 cache からは移設しない)。"""
        managed.models_root.mkdir(parents=True, exist_ok=True)
        managed.marker.write_text(f"model={REPO_ID}\ndevice=cpu", encoding="utf-8")
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        assert "snapshot" in _marker_payload(managed.marker)

    def test_marker_from_previous_cache_root_is_not_a_hit(self, managed):
        """**cache root を A → B へ変えたら、A の有効な snapshot を指す marker は miss**
        (#446 レビュー HIGH)。models root は同一。B へ再解決され、`info` が示す root と
        実際にロードする root が一致する。"""
        cache_a = managed.tmp_path / "cache-a"
        hub_a = cache_a / "huggingface" / "hub"
        _point_roots(managed.monkeypatch, managed.models_root, cache_a)
        snapshot_a = _make_snapshot(hub_a)
        _write_marker(managed.marker, hub_a, snapshot_a)
        assert _engine()._is_model_cached(managed.marker), "前提: A では hit"

        # 同じ models root のまま cache root だけ B へ
        _point_roots(managed.monkeypatch, managed.models_root, managed.cache_root)
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1, "A の snapshot は現在の root 配下でないので miss"
        _, kwargs = fake.calls[0]
        assert Path(kwargs["cache_dir"]) == managed.managed_hub
        (target,), _ = managed.from_pretrained.call_args
        assert Path(target).is_relative_to(managed.managed_hub.resolve())
        assert not Path(target).is_relative_to(hub_a.resolve())

    def test_marker_escaping_hub_root_is_not_a_hit(self, managed):
        """相対 path が `..` で hub root の外へ出る marker は hit にしない。"""
        outside = _make_snapshot(managed.tmp_path / "elsewhere")
        managed.models_root.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(outside, managed.managed_hub)
        managed.marker.write_text(
            json.dumps({"snapshot": Path(rel).as_posix(), "files": sorted(SNAPSHOT_FILES)}),
            encoding="utf-8",
        )
        assert Path(rel).parts[0] == "..", "前提: hub root の外を指している"

        assert not _engine()._is_model_cached(managed.marker)

    def test_snapshot_missing_weights_is_not_a_hit(self, managed):
        """config.json はあるが重み (manifest に記録したファイル) が欠ける → miss
        (#446 レビュー MEDIUM: config 1 個の確認では重み欠損を見逃す)。"""
        snapshot = _make_snapshot(managed.managed_hub)
        _write_marker(managed.marker, managed.managed_hub, snapshot)
        (snapshot / "model.safetensors").unlink()
        assert (snapshot / "config.json").is_file()
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1, "重みが欠けていれば再解決する"

    def test_snapshot_without_marker_is_resolved_offline_without_network_download(self, managed):
        """管理 cache に snapshot はあるが marker が無い (別プロセスが落とした / marker を
        消した) → snapshot_download が呼ばれるが、library は refs/main から解決するだけで
        ネットワークへ出ない (HF_HUB_OFFLINE=1 でも成功する: 実 Qwen で実測、#428)。
        ここでは「marker が無いだけで再ダウンロードを強制しない」形 (cache_dir= が同じ)
        を固定する。"""
        _make_snapshot(managed.managed_hub)
        fake = _FakeSnapshotDownload()

        _load_with(fake)

        assert len(fake.calls) == 1
        _, kwargs = fake.calls[0]
        assert Path(kwargs["cache_dir"]) == managed.managed_hub
        assert _marker_payload(managed.marker)["snapshot"] == f"{REPO_DIR}/snapshots/{SHA}"


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

    def test_snapshot_outside_managed_cache_is_rejected(self, managed):
        """library が cache_dir の外の path を返したら (既定 cache への fallback 等)
        marker を書かずに落とす。"""
        def elsewhere(repo_id, **kwargs):
            return str(_make_snapshot(managed.default_hub))

        with patch("huggingface_hub.snapshot_download", elsewhere):
            engine = _engine()
            with pytest.raises(RuntimeError, match="管理 cache の外"):
                engine.load_model()

        assert not managed.marker.exists()

    def test_load_failure_invalidates_marker_for_self_heal(self, managed):
        """manifest に無い形で snapshot が壊れて from_pretrained が落ちたら marker を
        無効化し、次回 load_model() で再解決できるようにする (#446 レビュー MEDIUM)。"""
        snapshot = _make_snapshot(managed.managed_hub)
        _write_marker(managed.marker, managed.managed_hub, snapshot)
        managed.from_pretrained.side_effect = OSError("corrupt safetensors header")
        fake = _FakeSnapshotDownload(fail=AssertionError("hit なので呼ばれない"))

        with patch("huggingface_hub.snapshot_download", fake):
            engine = _engine()
            with pytest.raises(OSError, match="corrupt"):
                engine.load_model()

        assert not managed.marker.exists(), "marker が残ると永久に skip して落ち続ける"
        assert not _engine()._is_model_cached(managed.marker)


class TestRemovedApi:
    def test_huggingface_cache_context_manager_is_gone(self):
        from livecap_cli.resources.model_manager import ModelManager

        assert not hasattr(ModelManager, "huggingface_cache"), (
            "HF_HOME を実行時に書き換える旧 API は効かないので削除した (#428)"
        )
