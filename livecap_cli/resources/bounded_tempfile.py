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
import stat
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Tuple, TypeVar, Union

__all__ = ["NAME_ATTEMPTS", "mkstemp", "mkdtemp", "reserve_file", "temporary_directory"]

#: 名前が衝突したときに別の名前を試す回数の上限。
#:
#: 名前は既定で 32 bit の乱数 (``tempfile`` と同じ 8 文字) なので、正常な filesystem で
#: 2 回目が要ることはまず無い。上限は、``O_EXCL`` がいつも ``EEXIST`` を返すような
#: 異常な filesystem で回り続けないためにある。
NAME_ATTEMPTS = 8

#: 名前の乱数のバイト数の既定 (16 進で 8 文字)。
_TOKEN_BYTES = 4

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
    directory: _PathLike,
    prefix: str,
    suffix: str,
    make: Callable[[str], _T],
    *,
    token_bytes: int = _TOKEN_BYTES,
) -> Tuple[_T, str]:
    """``make(path)`` が成功する名前を、上限つきで探す。``(make の戻り値, 絶対 path)``。

    名前は ``<prefix><token_bytes バイトの乱数の 16 進><suffix>``。排他的な作成と有限回の
    再試行は、このモジュールの外も含めてここだけに置く (#506)。
    """
    base = os.path.abspath(directory)
    for _ in range(NAME_ATTEMPTS):
        path = os.path.join(base, prefix + secrets.token_hex(token_bytes) + suffix)
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


def reserve_file(
    *, dir: _PathLike, prefix: str = "tmp", suffix: str = "", token_bytes: int = _TOKEN_BYTES
) -> str:
    """:func:`mkstemp` と同じ規則で空のファイルを作り、閉じてから絶対 path を返す。

    名前を確保して、あとで別の処理 (ネイティブのライブラリなど) に path で書かせる用途
    (発話ごとの一時 wav、#419 / #506)。中身を書いて消すのは呼び出し側の責任。
    ``token_bytes`` は名前の乱数のバイト数 (16 進で 2 倍の文字数)。

    Raises:
        OSError: 作れないとき。名前の衝突が :data:`NAME_ATTEMPTS` 回続いたときは
            ``FileExistsError``。
    """
    handle, path = _create(
        dir,
        prefix,
        suffix,
        lambda path: os.open(path, _FILE_FLAGS, 0o600),
        token_bytes=token_bytes,
    )
    os.close(handle)
    return path


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
    (読み取り専用のファイルや、開けないディレクトリ) は権限を戻してから消し、既に
    無いものは無視する。権限を戻すときにリンクの先には触れない。それ以外の失敗は
    送出する (``ignore_cleanup_errors=False`` と同じ)。
    """
    path = mkdtemp(dir=dir, prefix=prefix, suffix=suffix)
    try:
        yield path
    finally:
        _remove_tree(path)


def _remove_tree(root: str, *, repeated: bool = False) -> None:
    """``root`` を中身ごと消す。手順は CPython の ``TemporaryDirectory._rmtree`` と同じ。

    - 権限のせいで消せないものは、親と自身の権限を戻してから消し直す
    - **権限を戻すときにリンクを辿らない** (:func:`_reset_permissions`)。作業
      ディレクトリの中の symlink / junction の先にある、外のファイルの権限を変えない
      (CPython 側の CVE-2023-6597 と同じ扱い)
    - ディレクトリは、この関数で再帰的に消し直す。入れ子のディレクトリの権限拒否も
      同じ手順で回復する
    - 同じディレクトリを 2 回目も消せなければ送出する (``repeated``)。権限を戻せない
      場合に、無限に再試行しない。CPython は ``IsADirectoryError`` の経路でこの
      上限を付けていないが、ここでは両方の経路に付ける
    """

    def on_error(func, path: str, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if not isinstance(error, PermissionError):
            raise error
        if repeated and path == root:
            raise error
        try:
            if path != root:
                _reset_permissions(os.path.dirname(path))
            _reset_permissions(path)
            try:
                os.unlink(path)
            except IsADirectoryError:
                _remove_tree(path, repeated=(path == root))
            except PermissionError:
                # Windows (と FreeBSD) では、ディレクトリの unlink も PermissionError に
                # なる。本物のディレクトリ (リンクや junction ではない) のときだけ
                # 消し直し、それ以外は元の PermissionError を送出する (bpo-43153)。
                if not _is_plain_directory(path):
                    raise
                _remove_tree(path, repeated=(path == root))
        except FileNotFoundError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(root, onexc=on_error)
    else:  # 3.11 には onexc が無い (3.12 からは onerror が非推奨)
        shutil.rmtree(root, onerror=lambda func, path, info: on_error(func, path, info[1]))


def _without_following_links(func: Callable[..., object], path: str, *args: object) -> None:
    """``func(path, *args)`` を、リンクを辿らずに呼ぶ (CPython の ``tempfile`` と同じ)。

    ``follow_symlinks=False`` が使えない platform では、リンクには何もしない。
    Windows (3.12 まで) の ``os.chmod`` は元からリンク自身に作用するので、そのまま呼ぶ。
    """
    if func in os.supports_follow_symlinks:
        func(path, *args, follow_symlinks=False)
    elif os.name == "nt" or not os.path.islink(path):
        func(path, *args)


def _reset_permissions(path: str) -> None:
    """消せるように権限を戻す。Windows の ``os.chmod`` は読み取り専用属性を外すだけ。"""
    chflags = getattr(os, "chflags", None)
    if chflags is not None:
        _without_following_links(chflags, path, 0)
    _without_following_links(os.chmod, path, 0o700)


def _is_plain_directory(path: str) -> bool:
    """リンクでも junction でもない、本物のディレクトリか。"""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return False
    is_junction = (
        getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        and getattr(st, "st_reparse_tag", None) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", object())
    )
    return not is_junction
