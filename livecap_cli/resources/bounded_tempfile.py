"""再試行に上限のある一時ファイル / 一時ディレクトリの作成 (Issue #468 / #504)。

なぜ ``tempfile`` をそのまま使わないのか
-------------------------------------
CPython の ``tempfile.mkstemp`` / ``tempfile.mkdtemp`` (``TemporaryDirectory`` も内部で
使う) は、Windows で ``PermissionError`` を受けても
``os.path.isdir(dir) and os.access(dir, os.W_OK)`` が真なら「同名のディレクトリが既に
ある」とみなして別の名前で再試行する (bpo-22107)。ところが Windows の
``os.access(W_OK)`` は **ACL を見ず**、ディレクトリに対しては読み取り専用属性があっても
真を返す。ACL で書き込みを拒否されたディレクトリでも真を返し続けるので、再試行は
``os.TMP_MAX`` (Windows では 2,147,483,647) 回まで続き、CPU を 1 コア使い切ったまま
返らない (Windows 11 / CPython 3.11.13 で再現)。

ここでは再試行を**名前の衝突**に限り、回数に上限を置く。衝突とみなすのは
``FileExistsError`` と、Windows で**候補の名前に実際にディレクトリがある**ときの
``PermissionError`` だけで、``os.access`` には頼らない。それ以外の ``OSError``
(ACL で拒否されたときの ``PermissionError`` を含む) は 1 回目でそのまま送出する。
"""
from __future__ import annotations

import errno
import os
import secrets
import shutil
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Tuple, TypeVar, Union

__all__ = ["NAME_ATTEMPTS", "mkstemp", "mkdtemp", "temporary_directory"]

#: 名前が衝突したときに別の名前を試す回数の上限。
#:
#: 名前は 32 bit の乱数 (``tempfile`` と同じ 8 文字) なので、正常な filesystem で
#: 2 回目が要ることはまず無い。上限は、``O_EXCL`` がいつも ``EEXIST`` を返すような
#: 異常な filesystem で回り続けないためにある。
NAME_ATTEMPTS = 8

#: ``tempfile.mkstemp`` と同じ flag (``O_NOINHERIT`` は ``os.open`` が既定で付ける)。
_FILE_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_BINARY", 0)
)

_PathLike = Union[str, "os.PathLike[str]"]
_T = TypeVar("_T")


def _create(
    directory: _PathLike, prefix: str, suffix: str, make: Callable[[str], _T]
) -> Tuple[_T, str]:
    """``make(path)`` が成功する名前を、上限つきで探す。``(make の戻り値, 絶対 path)``。"""
    base = os.path.abspath(directory)
    for _ in range(NAME_ATTEMPTS):
        path = os.path.join(base, prefix + secrets.token_hex(4) + suffix)
        try:
            return make(path), path
        except FileExistsError:
            continue
        except PermissionError:
            # Windows では、作ろうとした名前に既にディレクトリがあると FileExistsError
            # ではなく PermissionError になる。そのときだけ衝突として扱う。
            if os.name == "nt" and os.path.isdir(path):
                continue
            raise
    raise FileExistsError(
        errno.EEXIST,
        f"no unused temporary name after {NAME_ATTEMPTS} attempts",
        base,
    )


def mkstemp(*, dir: _PathLike, prefix: str = "tmp", suffix: str = "") -> Tuple[int, str]:
    """``tempfile.mkstemp(dir=..., prefix=..., suffix=...)`` の代わり。

    ``(読み書き用の fd, 絶対 path)`` を返す。ファイルは ``O_EXCL`` でこの呼び出しが
    作ったもので、権限は ``0o600``。閉じて消すのは呼び出し側の責任。

    Raises:
        OSError: 作れないとき。名前の衝突が :data:`NAME_ATTEMPTS` 回続いたときは
            ``FileExistsError``。
    """
    return _create(dir, prefix, suffix, lambda path: os.open(path, _FILE_FLAGS, 0o600))


def mkdtemp(*, dir: _PathLike, prefix: str = "tmp", suffix: str = "") -> str:
    """``tempfile.mkdtemp(dir=..., prefix=..., suffix=...)`` の代わり。

    この呼び出しが作った空のディレクトリ (権限 ``0o700``) の絶対 path を返す。

    Raises:
        OSError: 作れないとき。名前の衝突が :data:`NAME_ATTEMPTS` 回続いたときは
            ``FileExistsError``。
    """
    _, path = _create(dir, prefix, suffix, lambda path: os.mkdir(path, 0o700))
    return path


@contextmanager
def temporary_directory(
    *, dir: _PathLike, prefix: str = "tmp", suffix: str = ""
) -> Iterator[str]:
    """``with tempfile.TemporaryDirectory(dir=...) as name:`` の代わり。

    :func:`mkdtemp` で作ったディレクトリの絶対 path を渡し、抜けるときに中身ごと
    消す。後片付けは ``TemporaryDirectory`` と同じく、権限のせいで消せないもの
    (読み取り専用のファイルなど) は権限を戻してから消し、既に無いものは無視する。
    それ以外の失敗は送出する (``ignore_cleanup_errors=False`` と同じ)。
    """
    path = mkdtemp(dir=dir, prefix=prefix, suffix=suffix)
    try:
        yield path
    finally:
        _remove_tree(path)


def _remove_tree(root: str) -> None:
    def on_error(func, target: str, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if not isinstance(error, PermissionError):
            raise error
        if target != root:
            _make_writable(os.path.dirname(target))
        _make_writable(target)
        # やり直しは 1 回だけ。権限を戻しても消せなければ、その例外を送出する。
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(root, onexc=on_error)
    else:  # 3.11 には onexc が無い (3.12 からは onerror が非推奨)
        shutil.rmtree(root, onerror=lambda func, target, info: on_error(func, target, info[1]))


def _make_writable(target: str) -> None:
    # Windows の os.chmod は書き込みビットで読み取り専用属性を外すだけ。
    try:
        os.chmod(target, 0o700)
    except FileNotFoundError:
        pass
