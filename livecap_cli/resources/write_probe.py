"""ディレクトリに実際に書けるかを確かめる probe (Issue #468)。

なぜ ``tempfile.mkstemp`` を使わないのか
-------------------------------------
CPython の ``tempfile._mkstemp_inner`` は、Windows で ``PermissionError`` を受けても
``os.path.isdir(dir) and os.access(dir, os.W_OK)`` が真なら「同名のディレクトリが
既にある」とみなして別の名前で再試行する (bpo-22107)。ところが Windows の
``os.access(W_OK)`` は読み取り専用属性しか見ず **ACL を見ない**。ACL で書き込みを
拒否されたディレクトリでは真を返し続けるので、再試行は ``os.TMP_MAX``
(Windows では 2,147,483,647) 回まで続く。呼び出し側の ``except OSError`` には
届かず、CPU を 1 コア使い切ったまま返らない (Windows 11 / CPython 3.11.13 で再現)。

ここでは ``os.open(O_CREAT | O_EXCL)`` を直接呼び、再試行するのは
``FileExistsError`` (名前の衝突) のときだけにする。名前は 64 bit の乱数なので、
``PermissionError`` を「同名のディレクトリがある」と読み替える必要は無い。
"""
from __future__ import annotations

import errno
import os
import secrets
from pathlib import Path

__all__ = ["PROBE_ATTEMPTS", "probe_write_access"]

#: ``FileExistsError`` のときに別の名前を試す回数の上限。
#:
#: 名前は 64 bit の乱数なので、正常な filesystem で 2 回目が要ることはまず無い。
#: 上限は、``O_EXCL`` がいつも ``EEXIST`` を返すような異常な filesystem で
#: 回り続けないためにある。
PROBE_ATTEMPTS = 8

_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)


def probe_write_access(directory: Path, *, prefix: str) -> None:
    """``directory`` に新しいファイルを作れるかを、実際に作って確かめる。

    作った probe ファイルは消す。消すのは ``O_EXCL`` で**この呼び出しが原子的に
    作った**ファイルだけで、既存のファイルには触れない (同じ名前があれば
    ``FileExistsError`` になり、別の名前を試す)。名前を固定しないのは、同名の
    ファイルを壊さないためと、複数プロセスが同時に probe しても奪い合わないため。

    Args:
        directory: 確かめるディレクトリ。存在している必要がある。
        prefix: probe ファイル名の接頭辞 (後ろに乱数を付ける)。

    Raises:
        OSError: 作れないとき。``PermissionError`` など ``FileExistsError`` 以外は
            1 回目でそのまま送出する。``FileExistsError`` が :data:`PROBE_ATTEMPTS`
            回続いたときは ``FileExistsError`` を送出する。
    """
    for _ in range(PROBE_ATTEMPTS):
        probe = os.path.join(directory, prefix + secrets.token_hex(8))
        try:
            handle = os.open(probe, _FLAGS, 0o600)
        except FileExistsError:
            continue
        try:
            os.close(handle)
        finally:
            try:
                os.unlink(probe)
            except OSError:
                # 書けたのに消せないのは稀。書けるかという問いには答えが出ている。
                pass
        return
    raise FileExistsError(
        errno.EEXIST,
        f"no unused probe file name after {PROBE_ATTEMPTS} attempts",
        str(directory),
    )
