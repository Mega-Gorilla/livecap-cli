"""``livecap_cli.engines.legacy_model_layouts`` — 旧配置から ModelRoot の正本への取り込み (Issue #456)。

engine 側のテスト (``test_*_managed_cache.py`` / ``test_nemo_download.py``) は「engine が
正しい引数で helper を呼ぶ」ことを見る。ここでは helper 自体の規則を固定する:

* 候補の順序 (0.2.0 hub → hub/transformers → 0.1.0 transformers → 0.1.0 huggingface → engine subdir)
* marker が指す snapshot > ``refs/main`` > 唯一の snapshot。特定できなければ触らない
* 実体化 (symlink は dereference) → manifest (``source="migrated"``) → publish の**後にだけ**旧側を消す
* 取り込みに失敗した候補は消さない。root の外は消さない
* ``scan_legacy_layouts`` は削除せず列挙する (``livecap-cli info``)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines import legacy_model_layouts as legacy
from livecap_cli.engines import model_store as ms
from tests.core.engines.conftest import write_hub_snapshot, write_repo_dir

REPO = "org/model"
DEST = "org--model"
FILES = {"config.json": b"{}", "model.bin": b"w" * 64, "README.md": b"#"}
REQUIRED = ("config.json", "model.bin")


@pytest.fixture
def roots(tmp_path):
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    models_root.mkdir()
    cache_root.mkdir()
    return models_root, cache_root


def _migrate(models_root, cache_root, **kwargs):
    return legacy.migrate_dir(
        models_root / DEST,
        repo_id=REPO,
        models_root=models_root,
        cache_root=cache_root,
        staging_root=cache_root / "downloads",
        required=REQUIRED,
        **kwargs,
    )


class TestFindLegacyDirs:
    def test_orders_newest_layout_first_then_engine_subdirs(self, roots):
        models_root, cache_root = roots
        hf = cache_root / "huggingface"
        s_020 = write_hub_snapshot(hf / "hub", REPO, FILES)
        s_020t = write_hub_snapshot(hf / "hub" / "transformers", REPO, FILES)
        s_010t = write_hub_snapshot(hf / "transformers", REPO, FILES)
        s_010 = write_hub_snapshot(hf, REPO, FILES)
        sub = write_repo_dir(models_root / "eng" / DEST, FILES, repo_id=REPO, with_manifest=False)

        found = legacy.find_legacy_dirs(repo_id=REPO, models_root=models_root, cache_root=cache_root, destination_name=DEST, engine_subdirs=("eng",))

        assert [c.source for c in found] == [s_020, s_020t, s_010t, s_010, sub]
        assert [c.kind for c in found] == ["hub_snapshot"] * 4 + ["flattened_dir"]

    def test_marker_snapshot_wins_over_refs_main(self, roots):
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        write_hub_snapshot(hub, REPO, FILES, sha="a" * 40)
        marked = write_hub_snapshot(hub, REPO, FILES, sha="b" * 40)
        (hub / f"models--{DEST}" / "refs" / "main").write_text("a" * 40, encoding="utf-8")
        marker = models_root / f"{DEST}.marker"
        marker.write_text(json.dumps({"snapshot": f"models--{DEST}/snapshots/{'b' * 40}", "files": []}), encoding="utf-8")

        (found,) = legacy.find_legacy_dirs(repo_id=REPO, models_root=models_root, cache_root=cache_root, destination_name=DEST)

        assert found.source == marked.resolve()
        assert marker in found.cleanup and hub / f"models--{DEST}" in found.cleanup

    def test_ambiguous_snapshot_is_skipped(self, roots):
        """refs/main 無し + snapshot が 2 つ → 特定できないので触らない。"""
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        write_hub_snapshot(hub, REPO, FILES, sha="a" * 40)
        write_hub_snapshot(hub, REPO, FILES, sha="b" * 40)
        (hub / f"models--{DEST}" / "refs" / "main").unlink()

        assert legacy.find_legacy_dirs(repo_id=REPO, models_root=models_root, cache_root=cache_root, destination_name=DEST) == []


class TestMigrateDir:
    def test_migrates_hub_snapshot_and_removes_all_legacy_copies(self, roots):
        models_root, cache_root = roots
        hf = cache_root / "huggingface"
        s_020 = write_hub_snapshot(hf / "hub", REPO, FILES)
        s_010 = write_hub_snapshot(hf, REPO, FILES)
        marker = models_root / f"{DEST}.marker"
        marker.write_text("{}", encoding="utf-8")

        manifest = _migrate(models_root, cache_root, ignore_patterns=["README.md"])

        assert manifest is not None and manifest.source == "migrated"
        assert [f.path for f in manifest.files] == ["config.json", "model.bin"]
        assert (models_root / DEST / "model.bin").read_bytes() == FILES["model.bin"]
        assert not s_020.exists() and not s_010.exists() and not marker.exists(), "同じ repo の旧配置は全部消す"
        leftovers = [p for p in (cache_root / "downloads").iterdir()] if (cache_root / "downloads").exists() else []
        assert not [p for p in leftovers if p.suffix != ".lock"], f"staging payload / migration temp を残さない: {leftovers}"
        # `<destination>.lock` は cache_root に許可された transient (Unix では release 後も残る)
        assert all(p.suffix == ".lock" for p in leftovers)

    def test_required_generator_is_materialized_once(self, roots):
        """`required` が generator でも adopt 判定と候補の必須ファイル検査の両方で使える。"""
        models_root, cache_root = roots
        write_hub_snapshot(cache_root / "huggingface" / "hub", REPO, FILES)
        manifest = legacy.migrate_dir(
            models_root / DEST, repo_id=REPO, models_root=models_root, cache_root=cache_root,
            staging_root=cache_root / "downloads", required=(n for n in REQUIRED), ignore_patterns=["README.md"],
        )
        assert manifest is not None and [f.path for f in manifest.files] == ["config.json", "model.bin"]

    def test_adopts_destination_and_removes_duplicates(self, roots):
        models_root, cache_root = roots
        write_repo_dir(models_root / DEST, FILES, repo_id=REPO, with_manifest=False)
        dup = write_repo_dir(models_root / "eng" / DEST, FILES, repo_id=REPO, with_manifest=False)

        manifest = _migrate(models_root, cache_root, engine_subdirs=("eng",))

        assert manifest is not None and manifest.source == "adopted"
        assert not dup.exists() and not (models_root / "eng").exists()

    def test_returns_none_and_touches_nothing_when_no_candidate_is_complete(self, roots):
        models_root, cache_root = roots
        snapshot = write_hub_snapshot(cache_root / "huggingface" / "hub", REPO, {"config.json": b"{}"})

        assert _migrate(models_root, cache_root) is None
        assert snapshot.is_dir() and not (models_root / DEST).exists()

    def test_publish_failure_keeps_legacy_and_tries_next_candidate(self, roots):
        models_root, cache_root = roots
        broken = write_hub_snapshot(cache_root / "huggingface" / "hub", REPO, FILES)
        good = write_repo_dir(models_root / "eng" / DEST, FILES, repo_id=REPO, with_manifest=False)
        real_materialize = ms.materialize_files
        calls = []

        def flaky(src, dst, names):
            calls.append(src)
            if len(calls) == 1:
                raise OSError("disk full")
            return real_materialize(src, dst, names)

        with patch("livecap_cli.engines.legacy_model_layouts.materialize_files", flaky):
            manifest = _migrate(models_root, cache_root, engine_subdirs=("eng",))

        assert manifest is not None and manifest.source == "migrated"
        assert calls == [broken, good]
        assert not good.exists(), "取り込んだ候補は消す"
        assert not broken.exists(), "正本が確定したので、失敗した候補も同じ repo の旧配置として消す"
        assert not list((cache_root / "downloads").glob("*.migrate-*")), "失敗した payload を残さない"

    def test_never_deletes_outside_roots(self, roots, tmp_path):
        models_root, cache_root = roots
        outside = tmp_path / "elsewhere" / "x"
        outside.mkdir(parents=True)
        (outside / "f").write_bytes(b"1")

        legacy._remove_legacy([outside], roots=(models_root, cache_root))

        assert outside.exists()

    @pytest.mark.skipif(os.name == "nt", reason="symlink には特権が要る (Windows)")
    def test_dereferences_hub_symlinks(self, roots):
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        repo = hub / f"models--{DEST}"
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "h1").write_bytes(b"w" * 64)
        snapshot = repo / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"{}")
        os.symlink(Path("..") / ".." / "blobs" / "h1", snapshot / "model.bin")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("c" * 40, encoding="utf-8")

        manifest = _migrate(models_root, cache_root)

        assert manifest is not None
        target = models_root / DEST / "model.bin"
        assert not target.is_symlink() and target.read_bytes() == b"w" * 64
        assert not repo.exists()


def _nemo_valid(path: Path) -> bool:
    """本物の validator と同じ規則 (`BaseEngine._verify_model_integrity`): 先頭が `./.` か PK。"""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    return head[:3] == b"./." or head == b"PK\x03\x04"


def _migrate_nemo(dest: Path, models_root: Path, cache_root: Path | None = None, **kw) -> bool:
    return legacy.migrate_nemo_file(
        dest,
        models_root=models_root,
        staging_root=models_root.parent / "cache" / "downloads",
        validate=_nemo_valid,
        cache_root=cache_root,
        **kw,
    )


class TestMigrateNemoFile:
    def test_unnests_valid_inner_file_and_drops_siblings(self, roots):
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dest.mkdir()
        (dest / "org--m.nemo").write_bytes(b"./.nemo")
        (dest / "org--m.bin").write_bytes(b"stale")

        assert _migrate_nemo(dest, models_root) is True
        assert dest.is_file() and dest.read_bytes() == b"./.nemo"
        assert list(models_root.iterdir()) == [dest]

    def test_unnest_failure_restores_the_original_layout(self, roots):
        """`os.replace` が失敗したら退避 dir を元名へ戻す — 正規 path が消えて元データが hidden な
        parked dir に取り残されない (PR #458 再レビュー HIGH)。"""
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dest.mkdir()
        (dest / "org--m.nemo").write_bytes(b"./.nemo")
        (dest / "org--m.bin").write_bytes(b"sidecar")
        real_replace = os.replace

        def failing_replace(src, dst, *a, **k):
            if Path(dst) == dest:
                raise OSError("replace blocked")
            return real_replace(src, dst, *a, **k)

        with patch("livecap_cli.engines.legacy_model_layouts.os.replace", failing_replace):
            with pytest.raises(RuntimeError, match="復元した"):
                _migrate_nemo(dest, models_root)

        assert dest.is_dir(), "元の nested dir が正規の名前に戻る"
        assert (dest / "org--m.nemo").read_bytes() == b"./.nemo" and (dest / "org--m.bin").read_bytes() == b"sidecar"
        assert list(models_root.iterdir()) == [dest], "hidden な退避 dir を残さない"

    def test_invalid_candidate_survives_another_candidates_success(self, roots):
        """invalid な hub `.nemo` + valid な engine subdir 複製 → subdir を採用した後も、validator を
        通らなかった hub 側は**消さない** (PR #458 再レビュー MEDIUM: 全候補を一括 cleanup していた)。"""
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        bad_snapshot = write_hub_snapshot(hub, "org/m", {"m.nemo": b"corrupt"})
        dest = models_root / "org--m.nemo"
        dup = models_root / "eng" / "org--m.nemo"
        dup.parent.mkdir()
        dup.write_bytes(b"./.good")

        assert _migrate_nemo(dest, models_root, cache_root, repo_id="org/m", engine_subdirs=("eng",)) is True
        assert dest.read_bytes() == b"./.good"
        assert not dup.exists(), "採用した valid な複製は消す"
        assert (bad_snapshot / "m.nemo").read_bytes() == b"corrupt", "invalid な候補は残す (info に出る)"
        assert (hub / "models--org--m") in [p for p, _ in legacy.scan_legacy_layouts(models_root, cache_root)]

    def test_nested_dir_with_invalid_inner_file_is_quarantined(self, roots):
        """中の同名ファイルが validator を通らない → 動かさず dir ごと隔離 (削除しない)。"""
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dest.mkdir()
        (dest / "org--m.nemo").write_bytes(b"garbage")

        assert _migrate_nemo(dest, models_root) is False
        assert not dest.exists(), "download が publish できるよう path を空ける"
        (quarantined,) = [p for p in models_root.iterdir() if ".invalid-" in p.name]
        assert (quarantined / "org--m.nemo").read_bytes() == b"garbage"

    def test_dir_without_inner_file_is_quarantined_so_download_can_publish(self, roots):
        """inner 無しの `.nemo/` dir を残すと `_publish_atomically` の `os.replace` が dir 上で失敗し続ける
        (PR #458 レビュー HIGH) → 隔離して cold download を通す。"""
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dest.mkdir()
        (dest / "other").write_bytes(b"?")

        assert _migrate_nemo(dest, models_root) is False
        assert not dest.exists()
        assert any(".invalid-" in p.name for p in models_root.iterdir())
        # cold download の publish (file → 空いた path) が通る
        dest.write_bytes(b"./.fresh")
        assert _nemo_valid(dest)

    def test_corrupt_root_file_is_quarantined_and_valid_duplicate_becomes_the_root(self, roots):
        """root 側が truncated + engine subdir に valid な複製 → root を隔離し、複製を正本にする。
        以前は `destination.is_file()` だけで正本扱いし、valid な複製の方を消していた (レビュー HIGH)。"""
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dest.write_bytes(b"trunc")
        dup = models_root / "eng" / "org--m.nemo"
        dup.parent.mkdir()
        dup.write_bytes(b"./.good")

        assert _migrate_nemo(dest, models_root, engine_subdirs=("eng",)) is True
        assert dest.read_bytes() == b"./.good"
        assert not dup.exists() and not dup.parent.exists(), "正本が valid になった後に複製を消す"
        (quarantined,) = [p for p in models_root.iterdir() if ".invalid-" in p.name]
        assert quarantined.read_bytes() == b"trunc", "壊れた旧 root は削除ではなく隔離"

    def test_invalid_duplicate_is_left_alone_and_nothing_is_published(self, roots):
        """候補が validator を通らない → 何も配置せず、旧側も消さない (info の残骸一覧に出る)。"""
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dup = models_root / "eng" / "org--m.nemo"
        dup.parent.mkdir()
        dup.write_bytes(b"bad")

        assert _migrate_nemo(dest, models_root, engine_subdirs=("eng",)) is False
        assert not dest.exists() and dup.read_bytes() == b"bad"

    def test_subdir_duplicate_moves_or_is_dropped(self, roots):
        models_root, _ = roots
        dest = models_root / "org--m.nemo"
        dup = models_root / "eng" / "org--m.nemo"
        dup.parent.mkdir()
        dup.write_bytes(b"./.from-subdir")

        assert _migrate_nemo(dest, models_root, engine_subdirs=("eng",)) is True
        assert dest.read_bytes() == b"./.from-subdir" and not dup.parent.exists()
        assert not [p for p in models_root.iterdir() if p.name.startswith(".")], "temp を残さない"

        dup.parent.mkdir()
        dup.write_bytes(b"./.dup-again")
        assert _migrate_nemo(dest, models_root, engine_subdirs=("eng",)) is False
        assert dest.read_bytes() == b"./.from-subdir" and not dup.exists()

    def test_hub_snapshot_nemo_is_materialized_and_repo_removed(self, roots):
        """0.1.0 の NeMo `from_pretrained` が落とした `<hub>/models--org--m/snapshots/<sha>/m.nemo`。"""
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        snapshot = write_hub_snapshot(hub, "org/m", {"m.nemo": b"./.hub"})
        dest = models_root / "org--m.nemo"

        assert _migrate_nemo(dest, models_root, cache_root, repo_id="org/m") is True
        assert dest.read_bytes() == b"./.hub"
        assert not snapshot.exists() and not (hub / "models--org--m").exists()
        assert not [p for p in models_root.iterdir() if p.name.startswith(".")], "temp を残さない"

        # 正本があるときは hub 側を消すだけ
        write_hub_snapshot(hub, "org/m", {"m.nemo": b"./.dup"})
        assert _migrate_nemo(dest, models_root, cache_root, repo_id="org/m") is False
        assert dest.read_bytes() == b"./.hub" and not (hub / "models--org--m").exists()

    def test_hub_snapshot_without_nemo_is_left_alone(self, roots):
        models_root, cache_root = roots
        hub = cache_root / "huggingface" / "hub"
        snapshot = write_hub_snapshot(hub, "org/m", {"config.json": b"{}"})

        assert _migrate_nemo(models_root / "org--m.nemo", models_root, cache_root, repo_id="org/m") is False
        assert snapshot.exists()

    def test_two_workers_migrating_the_same_duplicate_do_not_race(self, roots):
        """2 process (ここでは 2 thread + 共有 FileLock) が同じ旧配置を同時に取り込んでも例外にならず、
        正本は 1 つ、旧側は消える (PR #458 レビュー MEDIUM: migration が download の lock の外にあった)。"""
        import threading

        models_root, cache_root = roots
        dest = models_root / "org--m.nemo"
        dup = models_root / "eng" / "org--m.nemo"
        dup.parent.mkdir()
        dup.write_bytes(b"./.shared")
        write_hub_snapshot(cache_root / "huggingface" / "hub", "org/m", {"m.nemo": b"./.shared"})
        errors: list = []
        results: list = []

        def worker():
            try:
                results.append(_migrate_nemo(dest, models_root, cache_root, repo_id="org/m", engine_subdirs=("eng",)))
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)

        assert errors == [], errors
        assert results.count(True) == 1 and results.count(False) == 3
        assert dest.read_bytes() == b"./.shared"
        assert not dup.exists() and not (cache_root / "huggingface" / "hub" / "models--org--m").exists()


class TestSharedLock:
    def test_migration_and_download_use_the_same_lock_file(self, roots):
        """migrate_dir / migrate_nemo_file と fetch_repo_dir / download_file が同じ lock を取る。"""
        from livecap_cli.engines.model_store import model_lock_path

        models_root, cache_root = roots
        staging_root = cache_root / "downloads"
        assert model_lock_path(staging_root, models_root / DEST) == staging_root / f"{DEST}.lock"
        assert model_lock_path(staging_root, models_root / "org--m.nemo") == staging_root / "org--m.nemo.lock"

    def test_migrate_dir_holds_the_destination_lock(self, roots):
        """lock を別スレッドが握っている間は migrate_dir が進まない。"""
        import threading
        import time

        from livecap_cli.engines.model_store import model_lock

        models_root, cache_root = roots
        write_hub_snapshot(cache_root / "huggingface" / "hub", REPO, FILES)
        done = threading.Event()
        started = threading.Event()

        def hold():
            with model_lock(cache_root / "downloads", models_root / DEST):
                started.set()
                time.sleep(0.6)
            done.set()

        holder = threading.Thread(target=hold)
        holder.start()
        started.wait(5)
        t0 = time.time()
        manifest = _migrate(models_root, cache_root)
        holder.join(5)
        assert manifest is not None
        assert done.is_set() and time.time() - t0 >= 0.4, "lock 保持中は待つ"

    def test_two_workers_migrating_the_same_hub_snapshot_do_not_race(self, roots):
        import threading

        models_root, cache_root = roots
        write_hub_snapshot(cache_root / "huggingface" / "hub", REPO, FILES)
        errors: list = []
        results: list = []

        def worker():
            try:
                results.append(_migrate(models_root, cache_root, ignore_patterns=["README.md"]))
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)

        assert errors == [], errors
        assert all(m is not None for m in results)
        sources = sorted(m.source for m in results)
        assert sources.count("migrated") >= 1, "1 worker が取り込む"
        assert ms.validate_repo_dir(models_root / DEST, repo_id=REPO) is not None
        assert not (cache_root / "huggingface" / "hub" / f"models--{DEST}").exists()


class TestRemoveLegacyArchives:
    def test_removes_named_archive_and_empty_extract_dir(self, roots):
        models_root, cache_root = roots
        downloads = cache_root / "downloads"
        downloads.mkdir()
        archive = downloads / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
        archive.write_bytes(b"t" * 16)
        other = downloads / "other.tar.bz2"
        other.write_bytes(b"o")
        (cache_root / "reazonspeech-extract").mkdir()

        removed = legacy.remove_legacy_archives(cache_root, [archive.name])

        assert removed == [archive] and not archive.exists()
        assert other.exists(), "名前指定したものだけ消す"
        assert not (cache_root / "reazonspeech-extract").exists(), "空になった展開先も消す"
        assert legacy.remove_legacy_archives(cache_root, [archive.name]) == [], "冪等"


class TestScan:
    def test_refs_only_transient_hub_repo_is_not_legacy(self, roots, caplog):
        """新方式の `snapshot_download(local_dir=, cache_dir=)` が cache_dir 側に残す
        `models--<repo>/refs/main` だけの metadata は旧配置ではない (PR #458 レビュー MEDIUM):
        scan は 0 件、find_legacy_dirs も warning を出さない。"""
        import logging

        models_root, cache_root = roots
        repo = cache_root / "huggingface" / "hub" / f"models--{DEST}"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("c" * 40, encoding="utf-8")
        (repo / ".no_exist").mkdir()

        assert legacy.scan_legacy_layouts(models_root, cache_root) == []
        with caplog.at_level(logging.WARNING):
            assert legacy.find_legacy_dirs(repo_id=REPO, models_root=models_root, cache_root=cache_root, destination_name=DEST) == []
        assert not [r for r in caplog.records if "特定できない" in r.getMessage()]
        assert repo.exists(), "触らない"

    def test_lists_every_legacy_shape_with_sizes(self, roots):
        models_root, cache_root = roots
        hf = cache_root / "huggingface"
        write_hub_snapshot(hf / "hub", REPO, FILES)
        write_hub_snapshot(hf / "transformers", "o/v", FILES)
        (cache_root / "downloads").mkdir()
        (cache_root / "downloads" / "x.tar.bz2").write_bytes(b"t" * 10)
        (models_root / f"{DEST}.marker").write_text("{}", encoding="utf-8")
        nested = models_root / "org--m.nemo"
        nested.mkdir()
        (nested / "org--m.nemo").write_bytes(b"n" * 5)
        write_repo_dir(models_root / "reazonspeech" / "r", {"a": b"1"}, repo_id="r/r", with_manifest=False)
        write_repo_dir(models_root / DEST, FILES, repo_id=REPO)  # 正本は列挙しない
        (models_root / f"{DEST}.invalid-20260101-000000-abc123").mkdir()
        (models_root / f"{DEST}.invalid-20260101-000000-abc123" / "model.bin").write_bytes(b"q" * 7)
        (models_root / "org--m2.nemo.invalid-20260101-000000-def456").write_bytes(b"n" * 9)  # 隔離された **file**

        hits = dict(legacy.scan_legacy_layouts(models_root, cache_root))

        assert hits[hf / "hub" / f"models--{DEST}"] == sum(len(v) for v in FILES.values()) + 40
        assert hits[hf / "transformers" / "models--o--v"] > 0
        assert hits[cache_root / "downloads" / "x.tar.bz2"] == 10
        assert hits[models_root / f"{DEST}.marker"] == 2
        assert hits[nested] == 5
        assert hits[models_root / "reazonspeech"] == 1
        assert hits[models_root / f"{DEST}.invalid-20260101-000000-abc123"] == 7
        assert hits[models_root / "org--m2.nemo.invalid-20260101-000000-def456"] == 9, "隔離された .nemo file も列挙 (数 GB が不可視にならない)"
        assert models_root / DEST not in hits
        assert all(p.exists() for p in hits), "scan は消さない"
