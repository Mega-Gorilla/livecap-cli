"""staging entry の所有権マーカー兼 使用中 lease (Issue #375 PR 2、契約は #378 §6.6)。

**1 つのファイルが 2 つの役割を持つ。**

============  ==========================================  ==========================
役割          意味                                        寿命
============  ==========================================  ==========================
**所有権**    この entry は LiveCap が作った                entry と同じ (個別に消さない)
**lease**     いま使っている                              スコープの間だけ (開いている)
============  ==========================================  ==========================

なぜ所有権マーカーが要るのか
--------------------------
明示 staging root (``LIVECAP_CORE_ASCII_STAGING_DIR``) には運用者が**既存の
ディレクトリ**を指定できる。その配下に無関係なデータがあっても、TTL だけで回収すると
**それを消してしまう** — #386 のデータ消失そのものである。

したがって reaper は **自分が作った印のある entry にしか触らない**。マーカーは entry と
運命を共にする (個別に unlink しない) ので、スコープを抜けても所有権の印は残る。

なぜ「開いたまま」が lease なのか
------------------------------
「14 日経過していて ``rmtree`` が通る」は生存判定ではない。使用中のプロセスがその瞬間
ハンドルを開いていなければ消せてしまう。**スコープの全期間 開いたままにする**ことで、
開いていること自体が「まだ使っている」の証明になる。

判定はプラットフォームごとに OS へ任せる:

- **Windows**: 開いたハンドルが削除を阻む。reaper の ``rmtree`` が ``PermissionError``
  になれば使用中 (判定と削除の間に状態が変わる隙が無い)
- **POSIX**: 開いていても削除できるので ``flock(LOCK_EX | LOCK_NB)`` で確認する

**マーカーを unlink しない**のは所有権のためだけではない。POSIX で別の holder が lock を
持っている状況で path を消すと、holder は inode を保持したままでも**次の reaper からは
「マーカー無し」に見え**、使用中の entry が消される。

なぜ PID を使わないのか
--------------------
子プロセスは親の ``%TEMP%`` を継承するがディレクトリ名は親 pid のままで、**pid は
再利用される**。名前に埋めた pid の生死は根拠にならない。

支えない範囲 (v1)
----------------
**親のスコープより長生きする子プロセスは保護しない。** Python のハンドルは既定で
非継承 (PEP 446) であり、スコープを抜ければ lease は解放される。公開 API の契約は
「**スコープ内で完了する同期境界**」であり、spawn した子はスコープを抜ける前に
終了 / join させること。支えるなら子側も lease を握る別プロトコルが要る。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

from .errors import AsciiPathError

logger = logging.getLogger(__name__)

__all__ = ["MARKER_NAME", "marker_path", "hold_lease", "is_owned", "is_leased"]

#: 所有権マーカー兼 lease のファイル名。
MARKER_NAME = ".livecap-entry"

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]


def marker_path(entry: Path) -> Path:
    """``entry`` の所有権マーカー。**entry の中に置く。**

    **内側であることが Windows の保護そのもの。** reaper は ``rmtree(entry)`` を試すので、
    開いたハンドルが中にあれば削除が失敗して使用中だと分かる。隣に置くと
    ``rmtree(entry)`` を妨げず、**lease を持っていても消されてしまう**。

    「空のディレクトリを返す」契約 (:func:`~livecap_cli.paths.workspace.ascii_safe_workspace`)
    とは、**reaper の単位 (entry) と消費側に見せるディレクトリを分ける**ことで両立させる。
    """
    return entry / MARKER_NAME


@contextmanager
def hold_lease(entry: Path, *, boundary: str) -> Iterator[None]:
    """``entry`` に所有権マーカーを置き、スコープの間 lease として保持する。

    **確立できなければ送出する。** 保護なしで進めると、reaper から見て使用中と区別の
    つかない entry が生まれ、まさにこの module が防いでいるデータ消失に戻る。lease は
    「唯一の使用中証明」なので、無いまま進むのは契約違反である (TTL は猶予であって
    安全性ではない — 14 日を超えて動く境界は現に存在し得る)。

    Args:
        entry: reaper の単位となるディレクトリ。
        boundary: どのネイティブ境界のためか。失敗メッセージに必ず出すので必須。

    Raises:
        AsciiPathError: マーカーを作れない、または既に他者が lease を保持しているとき。
    """
    marker = marker_path(entry)
    try:
        # "ab" で開くのは、**既存マーカーを truncate しない**ため。所有権の印を
        # 引き継ぎつつ lease を取り直す。
        handle: IO[bytes] = open(marker, "ab")
    except OSError as error:
        raise AsciiPathError(
            f"{boundary}: could not create the staging ownership marker at "
            f"{ascii(str(marker))}: {error}. Without it the reaper cannot tell this "
            "directory apart from unrelated data, so continuing would risk deleting "
            "it (and would leave this entry unprotected while in use).",
            boundary=boundary,
        ) from error

    if fcntl is not None:  # pragma: no branch - POSIX のみ
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            # entry は uuid で固有なので、ここに来るのは異常 (衝突 / 再利用)。
            # **マーカーは消さない** — 保持している側のものだから。
            handle.close()
            raise AsciiPathError(
                f"{boundary}: the staging entry {ascii(str(entry))} is already leased "
                f"by another holder: {error}. Refusing to share it — the reaper relies "
                "on the lease to tell live entries from stale ones.",
                boundary=boundary,
            ) from error

    try:
        yield
    finally:
        # **マーカーは unlink しない。** entry と運命を共にすることで所有権の印が残り、
        # reaper が無関係なディレクトリと区別できる。また POSIX で他者が lock を
        # 持っている場合に path を消してしまう事故も起きない。
        try:
            handle.close()
        except OSError:  # pragma: no cover - close はまず失敗しない
            pass


def is_owned(entry: Path) -> bool:
    """``entry`` を LiveCap が作ったか。

    **reaper はこれが ``True`` の entry にしか触らない。** 明示 staging root には運用者が
    既存ディレクトリを指定でき、その配下の無関係なデータを消してはならない (#386)。
    """
    return marker_path(entry).is_file()


def is_leased(entry: Path) -> bool:
    """``entry`` がいま使われているか。

    **Windows では常に ``False`` を返す。** あちらは ``rmtree`` の失敗が正確な答えなので、
    ここで ``True`` を返すと**クラッシュで残ったマーカーが entry を永久に回収不能にする**
    (ハンドルは閉じているのにマーカーは残っている)。判定を OS に任せる方が、判定と削除の
    間に状態が変わる隙も無い。

    POSIX で判定できない場合は **``True`` (使用中) を返す** — 消せないより消してしまう方が
    害が大きい。#386 のデータ消失がまさにそれだった。
    """
    marker = marker_path(entry)
    if not marker.is_file():
        return False
    if fcntl is None:
        # Windows: rmtree の PermissionError が権威。
        return False

    try:
        handle = open(marker, "rb")
    except OSError:
        return True  # 開けない = 判断材料が無い -> 安全側

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True  # 誰かが持っている
    except OSError:
        return True  # 判定不能 -> 安全側
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()
