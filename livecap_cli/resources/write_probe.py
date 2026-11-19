"""ディレクトリに実際に書けるかを確かめる probe (Issue #468)。

Windows では ``os.access(W_OK)`` が ACL を見ないので、確かめるには実際に作るしかない。
作るのに ``tempfile.mkstemp`` を使うと、ACL で書き込みを拒否されたディレクトリでは
返らなくなる。理由と、代わりに使う上限つきの作成は
:mod:`livecap_cli.resources.bounded_tempfile` を参照。
"""
from __future__ import annotations

import os
from pathlib import Path

from .bounded_tempfile import mkstemp

__all__ = ["probe_write_access"]


def probe_write_access(directory: Path, *, prefix: str) -> None:
    """``directory`` に新しいファイルを作れるかを、実際に作って確かめる。

    作った probe ファイルは消す。消すのは ``O_EXCL`` で**この呼び出しが原子的に
    作った**ファイルだけで、既存のファイルには触れない (同じ名前があれば別の名前を
    試す)。名前を固定しないのは、同名のファイルを壊さないためと、複数プロセスが
    同時に probe しても奪い合わないため。

    Args:
        directory: 確かめるディレクトリ。存在している必要がある。
        prefix: probe ファイル名の接頭辞 (後ろに乱数を付ける)。

    Raises:
        OSError: 作れないとき。``PermissionError`` などは 1 回目でそのまま送出する
            (:func:`livecap_cli.resources.bounded_tempfile.mkstemp`)。
    """
    handle, probe = mkstemp(dir=directory, prefix=prefix)
    try:
        os.close(handle)
    finally:
        try:
            os.unlink(probe)
        except OSError:
            # 書けたのに消せないのは稀。書けるかという問いには答えが出ている。
            pass
