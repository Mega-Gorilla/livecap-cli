"""``livecap_cli.engines.hf_cache`` — HF Hub からの取得を管理 cache に閉じる共通 helper。

Qwen3-ASR (#428) / WhisperS2T (#430) / NeMo (#447) が共有する。engine 側のテストは
「helper へ正しい引数が渡る」ことを見るので、helper 自体の契約はここで固定する。
``snapshot_download`` / ``hf_hub_download`` は差し替え、ネットワークは使わない。
"""

from __future__ import annotations

import errno
import json
import threading
import time
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
    def _roots(self, tmp_path):
        return {
            "hub_root": tmp_path / "cache" / "huggingface" / "hub",
            "staging_dir": tmp_path / "cache" / "downloads" / "models--org--model",
            "destination": tmp_path / "models" / "org--model.nemo",
        }

    def test_downloads_into_staging_then_publishes_and_cleans(self, tmp_path):
        roots = self._roots(tmp_path)
        fake = _FakeHfHubDownload()

        with patch("huggingface_hub.hf_hub_download", fake):
            result = hf_cache.download_file(REPO, "model.nemo", **roots)

        assert result == roots["destination"]
        assert roots["destination"].read_bytes() == b"nemo-bytes"
        (call,) = fake.calls
        assert call["repo_id"] == REPO and call["filename"] == "model.nemo"
        assert Path(call["local_dir"]) == roots["staging_dir"], "既定 HF cache ではなく管理 staging へ取る"
        assert Path(call["cache_dir"]) == roots["hub_root"], (
            "local_dir モードでも hf_hub は cache_dir を try_to_load_from_cache で探す。"
            "省略すると既定 HF_HUB_CACHE から silent fallback する (#448 レビュー)"
        )
        assert not roots["staging_dir"].exists(), "staging (.cache/huggingface の metadata ごと) は消す — 保持は 1 部だけ"
        assert not list(roots["destination"].parent.glob(".*.part")), "publish 用の temp を残さない"

    def test_failure_leaves_no_destination_and_keeps_staging_for_resume(self, tmp_path):
        roots = self._roots(tmp_path)
        fake = _FakeHfHubDownload(fail=ConnectionError("network down"))

        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(ConnectionError):
                hf_cache.download_file(REPO, "model.nemo", **roots)

        assert not roots["destination"].exists()
        assert (roots["staging_dir"] / "model.nemo.incomplete").exists(), "resume 用に staging は残す"

    def test_offline_miss_propagates_as_local_entry_not_found(self, tmp_path):
        from huggingface_hub.errors import LocalEntryNotFoundError

        roots = self._roots(tmp_path)
        fake = _FakeHfHubDownload(fail=LocalEntryNotFoundError("offline and not cached"))
        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(LocalEntryNotFoundError):
                hf_cache.download_file(REPO, "model.nemo", **roots)
        assert not roots["destination"].exists()

    def test_interrupted_publish_leaves_no_destination(self, tmp_path):
        """cross-volume の move は copy → 削除になり途中で落ち得る。destination に途中までの
        ファイルを残すと、BaseEngine の完全性確認 (先頭数 byte) を通って cache hit に
        固定される (#448 レビュー HIGH)。同一 volume の temp → os.replace で publish する。"""
        roots = self._roots(tmp_path)
        real_copy2 = hf_cache.shutil.copy2

        def cross_volume_rename(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        def interrupted_copy2(src, dst, *args, **kwargs):
            Path(dst).write_bytes(b"nemo-")  # 途中まで書いて落ちる
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload()):
            with patch.object(hf_cache.os, "rename", cross_volume_rename):
                with patch.object(hf_cache.shutil, "copy2", interrupted_copy2):
                    with pytest.raises(OSError, match="No space"):
                        hf_cache.download_file(REPO, "model.nemo", **roots)

        assert not roots["destination"].exists(), "途中までの .nemo を最終位置に残さない"
        assert not list(roots["destination"].parent.glob(".*.part")), "temp も残さない"
        assert (roots["staging_dir"] / "model.nemo").is_file(), "staging の完了済みファイルは resume 用に残す"
        assert hf_cache.shutil.copy2 is real_copy2

    @pytest.mark.parametrize("same_volume", [True, False], ids=["rename", "copy"])
    def test_replace_failure_keeps_completed_download_in_staging(self, tmp_path, same_volume):
        """rename / copy のどちらで temp を作った場合も、`os.replace` が失敗したら
        destination は作られず、**完了済みの download は staging に残る** (#448 再レビュー)。"""
        roots = self._roots(tmp_path)
        real_rename, real_replace = hf_cache.os.rename, hf_cache.os.replace

        def failing_replace(src, dst):
            if Path(dst) == roots["destination"]:
                raise PermissionError(errno.EACCES, "destination locked by another process")
            return real_replace(src, dst)  # temp → source の復元は通す

        def cross_volume_rename(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload()):
            with patch.object(hf_cache.os, "replace", failing_replace):
                if same_volume:
                    ctx = patch.object(hf_cache.os, "rename", real_rename)
                else:
                    ctx = patch.object(hf_cache.os, "rename", cross_volume_rename)
                with ctx:
                    with pytest.raises(PermissionError):
                        hf_cache.download_file(REPO, "model.nemo", **roots)

        assert hf_cache.os.rename is real_rename and hf_cache.os.replace is real_replace
        assert not roots["destination"].exists()
        assert not list(roots["destination"].parent.glob(".*.part")), "temp を残さない"
        staged = roots["staging_dir"] / "model.nemo"
        assert staged.is_file() and staged.read_bytes() == b"nemo-bytes", "完了済み download を失わない"

    def test_cross_volume_copy_publishes_and_removes_source(self, tmp_path):
        """rename できない (別 volume) 場合は copy → replace → **publish 成功後に** source 削除。
        (`download_file` の rmtree に隠れないよう helper を直接呼ぶ)"""
        source = tmp_path / "staging" / "model.nemo"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"nemo-bytes")
        destination = tmp_path / "models" / "org--model.nemo"

        def cross_volume_rename(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        with patch.object(hf_cache.os, "rename", cross_volume_rename):
            hf_cache._publish_atomically(source, destination)

        assert destination.read_bytes() == b"nemo-bytes"
        assert not source.exists(), "publish 成功後は source を消す (2 部にしない)"
        assert not list(destination.parent.glob(".*.part"))

    def test_concurrent_downloads_of_same_repo_are_serialized(self, tmp_path):
        """同じ repo を 2 worker が同時に cold load しても、staging の move / rmtree が競合せず、
        取得は 1 回で済む (#448 レビュー MEDIUM)。"""
        roots = self._roots(tmp_path)
        started = threading.Event()

        class SlowDownload(_FakeHfHubDownload):
            def __call__(self, repo_id, **kwargs):
                started.set()
                time.sleep(0.3)
                return super().__call__(repo_id, **kwargs)

        fake = SlowDownload()
        results, errors = [], []

        def worker():
            try:
                results.append(hf_cache.download_file(REPO, "model.nemo", **roots))
            except BaseException as exc:  # noqa: BLE001 - テストで捕まえて assert する
                errors.append(exc)

        import huggingface_hub

        original = huggingface_hub.hf_hub_download
        # **patch は main thread で 1 回だけ、start / join 全体を包む。** worker ごとに
        # 重ねて patch すると restore 順が入れ替わり、終了後も fake が残る (#448 再レビュー)
        with patch("huggingface_hub.hf_hub_download", fake):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            threads[0].start()
            started.wait(5)
            threads[1].start()
            for t in threads:
                t.join(30)
        assert huggingface_hub.hf_hub_download is original, "mock がプロセスに漏れている"

        assert errors == [], errors
        assert results == [roots["destination"]] * 2
        assert roots["destination"].read_bytes() == b"nemo-bytes"
        assert len(fake.calls) == 1, "後続は lock 取得後に destination の実在を見て取得を skip する"
        assert not roots["staging_dir"].exists()


# ---------------------------------------------------------------------------
# fetch_repo_dir (flattened dir + manifest、#456)
# ---------------------------------------------------------------------------

from livecap_cli.engines import model_store as ms  # noqa: E402

REPO_FILES = {"config.json": b'{"model_type": "x"}', "model.bin": b"w" * 256, "README.md": b"# readme"}
COMMIT = "c" * 40


class _FakeSnapshotDownloadLocalDir:
    """``snapshot_download(local_dir=...)`` の代役: 本体と ``.cache/huggingface/download/*.metadata``
    (commit_hash / etag / timestamp の 3 行) を local_dir に書く。実 library と同じ配置。"""

    def __init__(self, *, fail: Exception | None = None, files=None):
        self.calls: list[dict] = []
        self.fail = fail
        self.files = dict(files or REPO_FILES)

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        local_dir = Path(kwargs["local_dir"])
        meta = local_dir / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        (local_dir / ".cache" / "huggingface" / ".gitignore").write_text("*", encoding="utf-8")
        allow = kwargs.get("allow_patterns")
        ignore = kwargs.get("ignore_patterns") or []
        import fnmatch

        for name, body in self.files.items():
            if allow is not None and not any(fnmatch.fnmatch(name, p) for p in allow):
                continue
            if any(fnmatch.fnmatch(name, p) for p in ignore):
                continue
            if self.fail is not None:
                (local_dir / f"{name}.incomplete").write_bytes(body[: len(body) // 2])
                raise self.fail
            (local_dir / name).write_bytes(body)
            (meta / f"{name}.metadata").write_text(f"{COMMIT}\n\"etag-{name}\"\n0.0\n", encoding="utf-8")
        return str(local_dir)


class TestFetchRepoDir:
    def _roots(self, tmp_path):
        return {
            "hub_root": tmp_path / "cache" / "huggingface" / "hub",
            "staging_root": tmp_path / "cache" / "downloads",
            "destination": tmp_path / "models" / "org--model",
        }

    def test_fetches_into_staging_then_publishes_flattened_dir(self, tmp_path):
        roots = self._roots(tmp_path)
        fake = _FakeSnapshotDownloadLocalDir()

        with patch("huggingface_hub.snapshot_download", fake):
            result = hf_cache.fetch_repo_dir(REPO, variant="base", ignore_patterns=["README.md"], **roots)

        (call,) = fake.calls
        assert Path(call["local_dir"]) == roots["staging_root"] / "org--model" / "download"
        assert Path(call["cache_dir"]) == roots["hub_root"], "local_dir モードでも cache_dir を明示 (#448)"
        assert call["max_workers"] == 1
        assert call["ignore_patterns"] == ["README.md"]
        assert result == roots["destination"]
        manifest = ms.validate_repo_dir(result, repo_id=REPO, variant="base")
        assert manifest is not None and manifest.source == "download"
        assert [f.path for f in manifest.files] == ["config.json", "model.bin"]
        assert manifest.commit_sha == COMMIT and manifest.files[1].etag == '"etag-model.bin"'
        assert not (result / ".cache").exists(), "HF の管理メタデータを ModelRoot へ持ち込まない"
        assert not list(result.rglob("*.metadata")) and not list(result.rglob("*.lock"))
        assert not (roots["staging_root"] / "org--model").exists(), "成功後は staging を消す"

    def test_required_files_are_enforced(self, tmp_path):
        roots = self._roots(tmp_path)
        fake = _FakeSnapshotDownloadLocalDir()

        with patch("huggingface_hub.snapshot_download", fake):
            with pytest.raises(RuntimeError, match="必要ファイルが無い"):
                hf_cache.fetch_repo_dir(REPO, allow_patterns=["config.json"], required=["model.bin"], **roots)

        assert not roots["destination"].exists()

    def test_no_match_fails_loud(self, tmp_path):
        roots = self._roots(tmp_path)
        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownloadLocalDir()):
            with pytest.raises(RuntimeError, match="取得したファイルが無い"):
                hf_cache.fetch_repo_dir(REPO, allow_patterns=["nothing-*"], **roots)
        assert not roots["destination"].exists()

    def test_download_failure_keeps_staging_and_creates_no_destination(self, tmp_path):
        roots = self._roots(tmp_path)
        fake = _FakeSnapshotDownloadLocalDir(fail=ConnectionError("network down"))

        with patch("huggingface_hub.snapshot_download", fake):
            with pytest.raises(ConnectionError):
                hf_cache.fetch_repo_dir(REPO, **roots)

        assert not roots["destination"].exists()
        download = roots["staging_root"] / "org--model" / "download"
        assert download.is_dir() and list(download.glob("*.incomplete")), "resume 用に staging を残す"

    def test_offline_miss_propagates(self, tmp_path):
        from huggingface_hub.errors import LocalEntryNotFoundError

        roots = self._roots(tmp_path)
        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownloadLocalDir(fail=LocalEntryNotFoundError("offline"))):
            with pytest.raises(LocalEntryNotFoundError):
                hf_cache.fetch_repo_dir(REPO, **roots)
        assert not roots["destination"].exists()

    def test_valid_destination_skips_download(self, tmp_path):
        roots = self._roots(tmp_path)
        dest = roots["destination"]
        dest.mkdir(parents=True)
        (dest / "config.json").write_bytes(b"{}")
        ms.write_manifest(dest, ms.build_manifest_from_dir(dest, repo_id=REPO))
        fake = _FakeSnapshotDownloadLocalDir(fail=AssertionError("hit なので呼ばれない"))

        with patch("huggingface_hub.snapshot_download", fake):
            assert hf_cache.fetch_repo_dir(REPO, **roots) == dest
        assert fake.calls == []

    def test_invalid_destination_is_quarantined_not_reused(self, tmp_path):
        roots = self._roots(tmp_path)
        dest = roots["destination"]
        dest.mkdir(parents=True)
        (dest / "junk.txt").write_bytes(b"old")  # 非空だが manifest 無し = 旧来なら hit だった形

        with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownloadLocalDir()):
            hf_cache.fetch_repo_dir(REPO, **roots)

        assert ms.validate_repo_dir(dest, repo_id=REPO) is not None
        assert list(dest.parent.glob("org--model.invalid-*")), "非空 dir は隔離され、hit にはならない"

    def test_concurrent_fetches_are_serialized(self, tmp_path):
        roots = self._roots(tmp_path)
        started = threading.Event()

        class Slow(_FakeSnapshotDownloadLocalDir):
            def __call__(self, repo_id, **kwargs):
                started.set()
                time.sleep(0.3)
                return super().__call__(repo_id, **kwargs)

        fake = Slow()
        results, errors = [], []

        def worker():
            try:
                results.append(hf_cache.fetch_repo_dir(REPO, **roots))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        import huggingface_hub

        original = huggingface_hub.snapshot_download
        with patch("huggingface_hub.snapshot_download", fake):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            threads[0].start()
            started.wait(5)
            threads[1].start()
            for t in threads:
                t.join(30)
        assert huggingface_hub.snapshot_download is original
        assert errors == [] and results == [roots["destination"]] * 2
        assert len(fake.calls) == 1, "後続は lock 取得後に destination が valid なので取得を skip"
