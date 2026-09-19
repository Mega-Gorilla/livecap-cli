"""``livecap_cli.engines.model_store`` — ModelRoot 契約の共通実装 (Issue #456)。

固定する契約:

* dir の cache hit は manifest の全ファイルがサイズ一致で実在するときだけ (「非空 dir」は hit ではない)
* symlink が dir の外を指す dir は hit にしない (旧 HF cache の ``blobs/`` を指したまま移した形)
* ``adopt_dir`` は required が揃う既存 dir にだけ manifest を書く
* ``materialize_files`` は symlink を dereference する
* ``publish_dir`` は valid な destination を skip / invalid を隔離 / 失敗時に destination を作らず
  payload と旧 destination を復元 / temp を残さない
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines import model_store as ms

REPO = "org/model"


def _make_dir(root: Path, files: dict, *, manifest: bool = True, repo_id: str = REPO, variant=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    if manifest:
        ms.write_manifest(root, ms.build_manifest_from_dir(root, repo_id=repo_id, variant=variant))
    return root


FILES = {"config.json": b'{"x": 1}', "model.bin": b"w" * 128, "sub/tokenizer.json": b"{}"}


class TestManifest:
    def test_round_trip_and_transient_exclusion(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES, manifest=False)
        (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (d / ".cache" / "huggingface" / "download" / "model.bin.metadata").write_text("x", encoding="utf-8")
        (d / "model.bin.incomplete").write_bytes(b"partial")
        (d / "stale.lock").write_bytes(b"")

        manifest = ms.build_manifest_from_dir(
            d, repo_id=REPO, variant="base", revision="main", commit_sha="a" * 40, etags={"model.bin": '"e"'}
        )
        ms.write_manifest(d, manifest)
        loaded = ms.read_manifest(d)

        assert loaded == manifest
        assert [f.path for f in loaded.files] == ["config.json", "model.bin", "sub/tokenizer.json"]
        assert loaded.files[1].size == 128 and loaded.files[1].etag == '"e"'
        assert loaded.commit_sha == "a" * 40 and loaded.variant == "base" and loaded.source == "download"

    @pytest.mark.parametrize(
        "text",
        ["not json", "{}", '{"schema_version": 99, "repo_id": "x", "files": []}',
         '{"schema_version": 1, "repo_id": "x", "files": [{"path": "a"}]}'],
    )
    def test_read_rejects_malformed(self, tmp_path, text):
        d = tmp_path / "m"
        d.mkdir()
        (d / ms.MANIFEST_NAME).write_text(text, encoding="utf-8")
        assert ms.read_manifest(d) is None
        assert ms.validate_repo_dir(d) is None

    def test_invalidate_removes_manifest_and_is_idempotent(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES)
        ms.invalidate_manifest(d, reason="test")
        assert not (d / ms.MANIFEST_NAME).exists()
        ms.invalidate_manifest(d, reason="again")


class TestValidate:
    def test_valid_dir(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES, variant="base")
        assert ms.validate_repo_dir(d, repo_id=REPO, variant="base") is not None

    def test_non_empty_dir_without_manifest_is_not_a_hit(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES, manifest=False)
        assert ms.validate_repo_dir(d) is None, "非空 dir を hit にしない (#456)"

    def test_missing_file_is_a_miss(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES)
        (d / "model.bin").unlink()
        assert ms.validate_repo_dir(d) is None

    def test_size_mismatch_is_a_miss(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES)
        (d / "model.bin").write_bytes(b"w" * 64)  # truncated copy
        assert ms.validate_repo_dir(d) is None

    def test_repo_id_or_variant_mismatch_is_a_miss(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES, variant="base")
        assert ms.validate_repo_dir(d, repo_id="other/model") is None
        assert ms.validate_repo_dir(d, variant="large") is None
        assert ms.validate_repo_dir(d, repo_id=REPO, variant="base") is not None

    def test_empty_manifest_is_a_miss(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        ms.write_manifest(d, ms.Manifest(repo_id=REPO, files=()))
        assert ms.validate_repo_dir(d) is None

    def test_symlink_pointing_outside_dir_is_a_miss(self, tmp_path):
        outside = tmp_path / "blobs" / "abc"
        outside.parent.mkdir()
        outside.write_bytes(b"w" * 128)
        d = _make_dir(tmp_path / "m", {"config.json": b"{}"}, manifest=False)
        try:
            os.symlink(outside, d / "model.bin")
        except OSError as exc:  # Windows without Developer Mode
            pytest.skip(f"symlink を作れない環境: {exc}")
        ms.write_manifest(d, ms.build_manifest_from_dir(d, repo_id=REPO))
        assert (d / "model.bin").is_file()
        assert ms.validate_repo_dir(d) is None, "旧 HF cache の blobs/ を指したままの symlink は正本ではない"


class TestAdopt:
    def test_adopts_when_required_present(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES, manifest=False)
        manifest = ms.adopt_dir(d, repo_id=REPO, required=["config.json", "model.bin"], variant="base")
        assert manifest is not None and manifest.source == "adopted"
        assert ms.validate_repo_dir(d, repo_id=REPO, variant="base") is not None

    def test_refuses_when_required_missing(self, tmp_path):
        d = _make_dir(tmp_path / "m", {"config.json": b"{}"}, manifest=False)
        assert ms.adopt_dir(d, repo_id=REPO, required=["config.json", "model.bin"]) is None
        assert not (d / ms.MANIFEST_NAME).exists(), "揃っていない dir に manifest を書かない"

    def test_returns_existing_valid_manifest_untouched(self, tmp_path):
        d = _make_dir(tmp_path / "m", FILES)
        before = (d / ms.MANIFEST_NAME).read_text(encoding="utf-8")
        manifest = ms.adopt_dir(d, repo_id=REPO, required=["config.json"])
        assert manifest is not None and manifest.source == "download"
        assert (d / ms.MANIFEST_NAME).read_text(encoding="utf-8") == before


class TestMaterialize:
    def test_dereferences_symlinks(self, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "h1").write_bytes(b"w" * 128)
        snap = tmp_path / "snapshots" / "sha"
        snap.mkdir(parents=True)
        (snap / "config.json").write_bytes(b"{}")
        try:
            os.symlink(Path("..") / ".." / "blobs" / "h1", snap / "model.bin")
        except OSError as exc:
            pytest.skip(f"symlink を作れない環境: {exc}")

        mechanisms = ms.materialize_files(snap, tmp_path / "payload", ["config.json", "model.bin"])

        out = tmp_path / "payload" / "model.bin"
        assert out.is_file() and not out.is_symlink() and out.read_bytes() == b"w" * 128
        assert set(mechanisms) == {"config.json", "model.bin"}
        assert set(mechanisms.values()) <= {"hardlink", "copy"}

    def test_missing_source_fails_loud(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        with pytest.raises(FileNotFoundError):
            ms.materialize_files(src, tmp_path / "dst", ["nope.bin"])

    def test_falls_back_to_copy_when_hardlink_fails(self, tmp_path):
        src = _make_dir(tmp_path / "src", {"a.bin": b"x" * 16}, manifest=False)

        def no_link(*a, **k):
            raise OSError(errno.EXDEV, "cross-device")

        with patch.object(ms.os, "link", no_link):
            mechanisms = ms.materialize_files(src, tmp_path / "dst", ["a.bin"])
        assert mechanisms == {"a.bin": "copy"}
        assert (tmp_path / "dst" / "a.bin").read_bytes() == b"x" * 16


class TestPublishDir:
    def _valid(self, path: Path) -> bool:
        return ms.validate_repo_dir(path, repo_id=REPO) is not None

    def test_publishes_payload_and_leaves_no_temp(self, tmp_path):
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)
        dest = tmp_path / "models" / "org--model"

        result = ms.publish_dir(payload, dest, validate=self._valid)

        assert result == dest and self._valid(dest)
        assert not payload.exists(), "同一 volume では rename されるので payload は残らない"
        assert not list(dest.parent.glob(".*.part")), "temp を残さない"

    def test_skips_when_destination_already_valid(self, tmp_path):
        dest = _make_dir(tmp_path / "models" / "org--model", FILES)
        marker = (dest / "config.json").read_bytes()
        payload = _make_dir(tmp_path / "staging" / "payload", {**FILES, "config.json": b'{"new": 1}'})

        ms.publish_dir(payload, dest, validate=self._valid)

        assert (dest / "config.json").read_bytes() == marker, "valid な destination は触らない"
        assert payload.exists(), "skip したので payload は呼び出し側に残す"

    def test_quarantines_invalid_destination(self, tmp_path):
        dest = _make_dir(tmp_path / "models" / "org--model", {"junk.txt": b"old"}, manifest=False)
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)

        ms.publish_dir(payload, dest, validate=self._valid)

        assert self._valid(dest)
        quarantined = list(dest.parent.glob("org--model.invalid-*"))
        assert len(quarantined) == 1 and (quarantined[0] / "junk.txt").read_bytes() == b"old", "invalid は消さず隔離"

    def test_cross_volume_copies_and_removes_payload_after_success(self, tmp_path):
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)
        dest = tmp_path / "models" / "org--model"

        def cross_volume_rename(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        with patch.object(ms.os, "rename", cross_volume_rename):
            ms.publish_dir(payload, dest, validate=self._valid)

        assert self._valid(dest)
        assert not payload.exists(), "copy 経路でも publish 成功後は payload を消す (2 部にしない)"
        assert not list(dest.parent.glob(".*.part"))

    def test_failed_validation_restores_payload_and_creates_no_destination(self, tmp_path):
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)
        (payload / "model.bin").write_bytes(b"w" * 64)  # manifest と size が合わない = 壊れた payload
        dest = tmp_path / "models" / "org--model"

        with pytest.raises(RuntimeError, match="validate"):
            ms.publish_dir(payload, dest, validate=self._valid)

        assert not dest.exists()
        assert payload.is_dir() and (payload / "config.json").exists(), "完了済み payload を失わない (resume 用)"
        assert not list(dest.parent.glob(".*.part"))

    def test_replace_failure_restores_quarantined_destination(self, tmp_path):
        dest = _make_dir(tmp_path / "models" / "org--model", {"junk.txt": b"old"}, manifest=False)
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)
        real_replace = ms.os.replace

        def failing_replace(src, dst):
            if Path(dst) == dest:
                raise PermissionError(errno.EACCES, "destination locked")
            return real_replace(src, dst)

        with patch.object(ms.os, "replace", failing_replace):
            with pytest.raises(PermissionError):
                ms.publish_dir(payload, dest, validate=self._valid)

        assert dest.is_dir() and (dest / "junk.txt").read_bytes() == b"old", "隔離した旧 destination を戻す"
        assert not list(dest.parent.glob("org--model.invalid-*"))
        assert payload.is_dir() and (payload / "model.bin").exists(), "payload も staging に戻す"
        assert not list(dest.parent.glob(".*.part"))

    def test_cross_volume_copy_failure_keeps_payload(self, tmp_path):
        payload = _make_dir(tmp_path / "staging" / "payload", FILES)
        dest = tmp_path / "models" / "org--model"

        def cross_volume_rename(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        def interrupted_copytree(src, dst, **kw):
            Path(dst).mkdir()
            (Path(dst) / "config.json").write_bytes(b"{")
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch.object(ms.os, "rename", cross_volume_rename):
            with patch.object(ms.shutil, "copytree", interrupted_copytree):
                with pytest.raises(OSError, match="No space"):
                    ms.publish_dir(payload, dest, validate=self._valid)

        assert not dest.exists()
        assert self._valid(payload), "payload は無傷"
        assert not list(dest.parent.glob(".*.part"))


class TestExemptAssets:
    def test_exempt_assets_are_install_owned_and_documented(self):
        assert set(ms.MODEL_STORE_EXEMPT_ASSETS) == {"silero_vad", "ten_vad"}
        for reason in ms.MODEL_STORE_EXEMPT_ASSETS.values():
            assert "runtime の書き込み無し" in reason
