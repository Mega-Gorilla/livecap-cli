"""アーカイブ展開の境界プローブ (Issue #378)。

**2 軸で測る**: 展開先ディレクトリ名が非 ASCII の場合と、アーカイブ内の
メンバ名が非 ASCII の場合。前者は本 epic の主題だが、後者は独立した
failure family (アーカイブのエンコーディングフラグの問題) であり、
ASCII staging では直らない。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from ..record import ProbeContext
from . import probe

_MEMBERS = {
    "ascii_member.txt": b"ascii payload",
    # メンバ名も非 ASCII にして 2 軸目を測る
    "日本語メンバ.txt": "非 ASCII メンバの中身".encode("utf-8"),
}


def _extract_observation(dest: Path) -> dict:
    """パスに依存しない観測 (名前は dest からの相対で見る)。"""
    found = sorted(
        str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file()
    )
    sizes = {name: (dest / name).stat().st_size for name in found}
    return {"members": found, "sizes": sizes}


@probe("zipfile.extractall")
def zipfile_extractall(ctx: ProbeContext) -> dict:
    """``ffmpeg_manager`` の ``zipfile.ZipFile(...).extractall(temp_dir)`` 相当。"""
    archive = ctx.root / "bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in _MEMBERS.items():
            zf.writestr(name, data)
    ctx.stage("create_archive")

    dest = ctx.root / "extracted_zip"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    ctx.stage("extractall")

    return {"archive_bytes_nonzero": archive.stat().st_size > 0, **_extract_observation(dest)}
