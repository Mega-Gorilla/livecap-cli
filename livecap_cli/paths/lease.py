"""staging entry の使用中マーカー (Issue #375 PR 2、契約は #378 §6.6)。

なぜ TTL だけでは足りないのか
--------------------------
「14 日経過していて ``rmtree`` が通る」は**生存判定ではない**。

- 使用中のプロセスが path を保持していても、reaper が走った**その瞬間に**ハンドルを
  開いていなければ削除できてしまう
- ``%TEMP%`` を継承した子プロセスが**後から**同じ path を使う場合を判別できない

#378 §6.6 は「reaper は**プロセス内 refcount またはプロセス間 lease が生きている
entry に触れない**」ことを要求している。そこで **スコープの全期間にわたって開いた
ままのファイル**を lease とする — 開いていること自体が「まだ使っている」の証明になる。

なぜ PID を使わないのか
--------------------
子プロセスは親の ``%TEMP%`` を継承するがディレクトリ名は親 pid のままで、**pid は
再利用される**。名前に埋めた pid の生死は根拠にならない。

判定はプラットフォームごとに OS へ任せる:

- **Windows**: 開いたハンドルが削除を阻む。reaper の ``rmtree`` が ``PermissionError``
  になれば使用中
- **POSIX**: ハンドルを開いていても削除できるので、``flock(LOCK_EX | LOCK_NB)`` を
  試して ``BlockingIOError`` なら使用中
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Optional

logger = logging.getLogger(__name__)

__all__ = ["lease_path", "hold_lease", "is_leased", "LEASE_NAME"]

#: lease ファイル名。
LEASE_NAME = ".livecap-inuse"


def lease_path(entry: Path) -> Path:
    """``entry`` に対応する lease ファイル。**entry の中に置く。**

    **内側であることが Windows の保護そのもの。** reaper は ``rmtree(entry)`` を
    試すので、開いたハンドルが中にあれば削除が失敗して使用中だと分かる。隣に
    置くと ``rmtree(entry)`` を妨げず、**lease を持っていても消されてしまう**。

    「空のディレクトリを返す」契約 (:func:`~livecap_cli.paths.workspace.ascii_safe_workspace`)
    とは、**reaper の単位と消費側に見せるディレクトリを分ける**ことで両立させる —
    lease は entry 側に置き、消費側にはその子を渡す。
    """
    return entry / LEASE_NAME

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]


@contextmanager
def hold_lease(entry: Path) -> Iterator[None]:
    """``entry`` を使用中としてマークし、スコープの間ずっと保持する。

    **開いたまま保持することが lease の実体である。** 一瞬開いて閉じるのでは、
    reaper がその隙に消せてしまい何も守れない。

    lease を取れなくても**本筋を止めない** — 使用中マークが無いだけで、entry 自体は
    使える。TTL (既定 14 日) があるので即座に消されるわけでもない。
    """
    handle: Optional[IO[bytes]] = None
    try:
        handle = open(lease_path(entry), "wb")
        if fcntl is not None:  # pragma: no branch - POSIX のみ
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # 既に他プロセスが持っている。entry の共有自体は起こらない設計
                # (uuid で固有) なので、ここに来るのは異常だが致命ではない。
                logger.debug("Could not flock the lease for %s", ascii(str(entry)))
    except OSError as error:
        logger.debug("Could not create a lease for %s: %s", ascii(str(entry)), error)

    try:
        yield
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:  # pragma: no cover - close はまず失敗しない
                pass
            # 自分の lease は自分で片付ける。残しても reaper が回収するが、
            # 消えていないと「使用中かもしれない」と読める余地を残すことになる。
            try:
                lease_path(entry).unlink()
            except OSError:
                pass


def is_leased(entry: Path) -> bool:
    """``entry`` が使用中か。

    **POSIX 専用の判定である。** Windows では開いたハンドルが ``rmtree`` を阻むので、
    reaper は削除の失敗そのもので判定する (OS に仕事をさせる方が確実で、判定と削除の
    間に状態が変わる隙も無い)。

    **Windows では常に ``False`` を返す。** あちらは ``rmtree`` の失敗が正確な答え
    なので、ここで ``True`` を返すと**クラッシュで残ったオーファン lease が entry を
    永久に回収不能にする** (ハンドルは閉じているのに「使用中」と読んでしまう)。
    判定を OS に任せる方が、判定と削除の間に状態が変わる隙も無い。

    POSIX で判定できない場合は **``True`` (使用中) を返す** — 消せないより消して
    しまう方が害が大きい。#386 のデータ消失がまさにそれだった。
    """
    lease = lease_path(entry)
    if not lease.is_file():
        return False
    if fcntl is None:
        # Windows: rmtree の PermissionError が権威。ここで止めると
        # オーファン lease が永久に残る。
        return False

    try:
        handle = open(lease, "rb")
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
