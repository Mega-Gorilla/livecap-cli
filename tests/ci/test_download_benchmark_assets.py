"""``scripts/download_benchmark_assets.py`` の corpus 配置 (#470)。

self-hosted runner の永続 cache を作り直した直後、LibriSpeech の展開物を ``%TEMP%`` から
``shutil.move`` したところ、ウイルス対策の走査が掴んでいて ``os.rename`` が WinError 5 に
なり、copy に fallback した後の temp 側の削除で例外になった。存在判定は「speaker dir が
1 つでもあるか」なので、途中で落ちた dir を次回は完成品として扱ってしまう。
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_benchmark_assets.py"


def _load_module():
    """Import by path: ``scripts`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_download_benchmark_assets", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = _load_module()


def _corpus(root: Path) -> Path:
    for speaker in ("1089", "1188"):
        d = root / speaker / "134686"
        d.mkdir(parents=True)
        (d / f"{speaker}-134686-0000.flac").write_bytes(b"fLaC" + speaker.encode())
        (d / f"{speaker}-134686.trans.txt").write_text("HELLO\n", encoding="utf-8")
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_installs_a_complete_copy(tmp_path):
    extracted = _corpus(tmp_path / "extract" / "test-clean")
    dest = tmp_path / "source" / "librispeech" / "test-clean"

    assets.install_extracted_dir(extracted, dest)

    assert _files(dest) == _files(extracted)
    assert not dest.with_name("test-clean.part").exists()


def test_replaces_an_existing_destination(tmp_path):
    extracted = _corpus(tmp_path / "extract" / "test-clean")
    dest = tmp_path / "dest" / "test-clean"
    (dest / "stale").mkdir(parents=True)
    (dest / "stale" / "old.flac").write_bytes(b"old")

    assets.install_extracted_dir(extracted, dest)

    assert _files(dest) == _files(extracted)


def test_a_failed_copy_never_leaves_a_destination(tmp_path, monkeypatch):
    """copy が途中で落ちても dest_dir は現れない — 次回の存在判定が途中状態を完成品と見なさない。"""
    extracted = _corpus(tmp_path / "extract" / "test-clean")
    dest = tmp_path / "dest" / "test-clean"
    real_copytree = shutil.copytree
    calls = {"n": 0}

    def flaky_copy2(src, dst, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # 1 ファイル目は書けた後に落ちる (途中状態を作る)
            raise PermissionError(5, "Access is denied")
        return shutil.copy2(src, dst, *args, **kwargs)

    def copytree_with_flaky_copy(src, dst, symlinks=False, ignore=None, copy_function=None, *args, **kwargs):
        # copytree は copy_function の既定値 (copy2) を定義時に束縛し、sub dir へは位置引数で
        # 再帰するので、copytree 自体を包んで copy_function を差し替える
        return real_copytree(src, dst, symlinks, ignore, flaky_copy2, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", copytree_with_flaky_copy)

    with pytest.raises(shutil.Error):
        assets.install_extracted_dir(extracted, dest)

    assert not dest.exists()


def test_a_leftover_staging_dir_is_discarded(tmp_path):
    extracted = _corpus(tmp_path / "extract" / "test-clean")
    dest = tmp_path / "dest" / "test-clean"
    staging = dest.with_name("test-clean.part")
    staging.mkdir(parents=True)
    (staging / "partial.flac").write_bytes(b"partial")

    assets.install_extracted_dir(extracted, dest)

    assert _files(dest) == _files(extracted)
    assert not staging.exists()


def test_temp_cleanup_errors_do_not_fail_the_download():
    """temp 側の削除失敗 (展開物を走査中のロック等) は download の失敗にしない。"""
    source = SCRIPT.read_text(encoding="utf-8")

    assert "shutil.move(" not in source
    assert source.count("tempfile.TemporaryDirectory(ignore_cleanup_errors=True)") == 2
