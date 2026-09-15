"""``livecap_cli.engines.hf_cache`` — HF Hub からの取得を管理 cache に閉じる共通 helper。

Qwen3-ASR (#428) / WhisperS2T (#430) / NeMo (#447) が共有する。engine 側のテストは
「helper へ正しい引数が渡る」ことを見るので、helper 自体の契約はここで固定する。
``snapshot_download`` / ``hf_hub_download`` は差し替え、ネットワークは使わない。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines import hf_cache

REPO = "org/model"
REPO_DIR = "models--org--model"
SHA = "a" * 40


def _make_snapshot(hub: Path, files=("config.json", "model.bin")) -> Path:
    snapshot = hub / REPO_DIR / "snapshots" / SHA
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snapshot / name).write_text(name, encoding="utf-8")
    return snapshot


class TestResolveSnapshot:
    def test_passes_managed_cache_dir_single_worker_and_patterns(self, tmp_path):
        hub = tmp_path / "hub"
        calls = []

        def fake(repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            return str(_make_snapshot(Path(kwargs["cache_dir"])))

        with patch("huggingface_hub.snapshot_download", fake):
            snapshot = hf_cache.resolve_snapshot(
                REPO, hub_root=hub, marker=tmp_path / "m.marker", allow_patterns=["config.json", "*.bin"]
            )

        (repo_id, kwargs), = calls
        assert repo_id == REPO
        assert Path(kwargs["cache_dir"]) == hub
        assert kwargs["max_workers"] == 1, "hf_hub の symlink 判定 race (huggingface_hub#4915) の回避"
        assert kwargs["allow_patterns"] == ["config.json", "*.bin"]
        assert snapshot == (hub / REPO_DIR / "snapshots" / SHA).resolve()

    def test_no_allow_patterns_means_whole_repo(self, tmp_path):
        hub = tmp_path / "hub"
        calls = []

        def fake(repo_id, **kwargs):
            calls.append(kwargs)
            return str(_make_snapshot(Path(kwargs["cache_dir"])))

        with patch("huggingface_hub.snapshot_download", fake):
            hf_cache.resolve_snapshot(REPO, hub_root=hub, marker=tmp_path / "m.marker")

        assert "allow_patterns" not in calls[0]

    def test_writes_marker_only_after_success(self, tmp_path):
        hub = tmp_path / "hub"
        marker = tmp_path / "m.marker"

        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("down")):
            with pytest.raises(RuntimeError, match="down"):
                hf_cache.resolve_snapshot(REPO, hub_root=hub, marker=marker)
        assert not marker.exists()

        with patch("huggingface_hub.snapshot_download", lambda r, **k: str(_make_snapshot(Path(k["cache_dir"])))):
            hf_cache.resolve_snapshot(REPO, hub_root=hub, marker=marker)
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload == {"snapshot": f"{REPO_DIR}/snapshots/{SHA}", "files": ["config.json", "model.bin"]}

    def test_rejects_snapshot_outside_hub_root(self, tmp_path):
        hub = tmp_path / "hub"
        elsewhere = _make_snapshot(tmp_path / "default-cache")

        with patch("huggingface_hub.snapshot_download", lambda r, **k: str(elsewhere)):
            with pytest.raises(RuntimeError, match="管理 cache の外"):
                hf_cache.resolve_snapshot(REPO, hub_root=hub, marker=tmp_path / "m.marker")
        assert not (tmp_path / "m.marker").exists()

    def test_rejects_snapshot_without_config(self, tmp_path):
        hub = tmp_path / "hub"

        with patch("huggingface_hub.snapshot_download", lambda r, **k: str(_make_snapshot(hub, files=("model.bin",)))):
            with pytest.raises(RuntimeError, match="config.json"):
                hf_cache.resolve_snapshot(REPO, hub_root=hub, marker=tmp_path / "m.marker")


class TestMarker:
    def test_read_requires_current_root_manifest_and_config(self, tmp_path):
        hub_a, hub_b = tmp_path / "a", tmp_path / "b"
        snapshot = _make_snapshot(hub_a)
        marker = tmp_path / "m.marker"
        hf_cache.write_marker(marker, hub_a, snapshot)

        assert hf_cache.read_marker(marker, hub_a) == snapshot.resolve()
        assert hf_cache.read_marker(marker, hub_b) is None, "別の root からは解決しない (A→B)"

        (snapshot / "model.bin").unlink()
        assert hf_cache.read_marker(marker, hub_a) is None, "manifest のファイルが欠けたら miss"

    def test_read_rejects_escaping_relative_path_and_legacy_text(self, tmp_path):
        hub = tmp_path / "hub"
        outside = _make_snapshot(tmp_path / "elsewhere")
        marker = tmp_path / "m.marker"
        marker.write_text(json.dumps({"snapshot": "../elsewhere/" + f"{REPO_DIR}/snapshots/{SHA}", "files": []}), encoding="utf-8")
        assert outside.is_dir()
        assert hf_cache.read_marker(marker, hub) is None

        marker.write_text("model=org/model\ndevice=cpu", encoding="utf-8")
        assert hf_cache.read_marker(marker, hub) is None

    def test_invalidate_removes_marker(self, tmp_path):
        marker = tmp_path / "m.marker"
        marker.write_text("{}", encoding="utf-8")
        hf_cache.invalidate_marker(marker, reason="test")
        assert not marker.exists()
        hf_cache.invalidate_marker(marker, reason="idempotent")


class _FakeHfHubDownload:
    """``hf_hub_download(local_dir=...)`` の代役: local_dir へ本体と metadata を書く。"""

    def __init__(self, *, fail: Exception | None = None, body: bytes = b"nemo-bytes"):
        self.calls: list[dict] = []
        self.fail = fail
        self.body = body

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        local_dir = Path(kwargs["local_dir"])
        meta = local_dir / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        if self.fail is not None:
            (local_dir / (kwargs["filename"] + ".incomplete")).write_bytes(b"partial")
            raise self.fail
        target = local_dir / kwargs["filename"]
        target.write_bytes(self.body)
        (meta / (kwargs["filename"] + ".metadata")).write_text("etag", encoding="utf-8")
        return str(target)


class TestDownloadFile:
    def test_downloads_into_staging_then_moves_and_cleans(self, tmp_path):
        staging = tmp_path / "cache" / "downloads" / "models--org--model"
        destination = tmp_path / "models" / "org--model.nemo"
        fake = _FakeHfHubDownload()

        with patch("huggingface_hub.hf_hub_download", fake):
            result = hf_cache.download_file(REPO, "model.nemo", staging_dir=staging, destination=destination)

        assert result == destination
        assert destination.read_bytes() == b"nemo-bytes"
        (call,) = fake.calls
        assert call["repo_id"] == REPO and call["filename"] == "model.nemo"
        assert Path(call["local_dir"]) == staging, "既定 HF cache ではなく管理 staging へ取る"
        assert "cache_dir" not in call
        assert not staging.exists(), "staging (.cache/huggingface の metadata ごと) は消す — 保持は 1 部だけ"

    def test_failure_leaves_no_destination_and_keeps_staging_for_resume(self, tmp_path):
        staging = tmp_path / "cache" / "downloads" / "models--org--model"
        destination = tmp_path / "models" / "org--model.nemo"
        fake = _FakeHfHubDownload(fail=ConnectionError("network down"))

        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(ConnectionError):
                hf_cache.download_file(REPO, "model.nemo", staging_dir=staging, destination=destination)

        assert not destination.exists()
        assert (staging / "model.nemo.incomplete").exists(), "resume 用に staging は残す"

    def test_offline_miss_propagates_as_local_entry_not_found(self, tmp_path):
        from huggingface_hub.errors import LocalEntryNotFoundError

        fake = _FakeHfHubDownload(fail=LocalEntryNotFoundError("offline and not cached"))
        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(LocalEntryNotFoundError):
                hf_cache.download_file(
                    REPO, "model.nemo", staging_dir=tmp_path / "s", destination=tmp_path / "d.nemo"
                )
        assert not (tmp_path / "d.nemo").exists()
