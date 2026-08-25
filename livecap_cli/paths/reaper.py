"""staging root に残った孤児ディレクトリの回収 (Issue #375 PR 2、契約は #378 §6.6)。

なぜ回収が要るのか
----------------
:func:`~livecap_cli.paths.temp_env.ascii_safe_temp_environment` は**退出時に自分の
ディレクトリを消さない**。プロセス全体の TEMP を向けている間は無関係なスレッドの
``NamedTemporaryFile()`` もそこへ落ちるためで、消すと #386 のデータ消失が再発する。
その代わりに残骸が積み上がるので、ここで TTL 回収する。

なぜ PID を見ないのか
-------------------
「別 pid かつ N 時間経過」は lease の代わりにならない — 子プロセスは親の TEMP を
継承するがディレクトリ名は親 pid のままで、**pid は再利用され**、複数プロセスが
併存し得る。**PID 生存判定は使わない。**

代わりに **OS に判定させる**: Windows では開いたハンドルが削除を阻むので、
``PermissionError`` が返れば使用中とみなしてそのエントリを飛ばす。POSIX では
ディレクトリを消せてしまうが、そちらは元々非 ASCII パス問題が起きない環境である。

**best-effort。** 回収は本筋ではないので、失敗しても例外にしない。
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

__all__ = ["reap_staging_root", "DEFAULT_TTL_HOURS"]

#: 既定の保持時間 (#378 §6.5 の ``_TTL_HOURS``)。
DEFAULT_TTL_HOURS = 336  # 14 日

_lock = threading.Lock()
_reaped: Set[Path] = set()


def reap_staging_root(
    root: Path, *, ttl_hours: float = DEFAULT_TTL_HOURS, force: bool = False
) -> int:
    """``root`` 配下の古い孤児ディレクトリを消す。消せた数を返す。

    root ごとに**プロセス内で 1 回だけ**走る (``force=True`` で再実行できる) —
    境界呼び出しのたびに ``scandir`` するのは無駄で、回収は緊急性が無い。

    Args:
        root: staging root。
        ttl_hours: これより古いエントリを対象にする。
        force: 1 回きりの制限を外す (テスト用)。

    Returns:
        削除できたディレクトリ数。**例外は送出しない。**
    """
    with _lock:
        if not force and root in _reaped:
            return 0
        _reaped.add(root)

    cutoff = time.time() - ttl_hours * 3600
    removed = 0
    try:
        with os.scandir(root) as entries:
            purposes = [e for e in entries if e.is_dir()]
    except OSError:
        return 0

    for purpose in purposes:
        try:
            with os.scandir(purpose.path) as entries:
                children = list(entries)
        except OSError:
            continue

        for child in children:
            try:
                if not child.is_dir() or child.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue

            try:
                # ignore_errors=False にするのが要点。**使用中を検出したい**ので、
                # Windows が返す PermissionError を握り潰さない。
                shutil.rmtree(child.path)
                removed += 1
            except PermissionError:
                # 誰かが掴んでいる = 使用中。OS に判定させている箇所。
                logger.debug("Staging entry is in use, skipping: %s", ascii(child.path))
            except OSError as error:
                logger.debug(
                    "Could not reap staging entry %s: %s", ascii(child.path), error
                )

    if removed:
        logger.info("Reaped %d stale staging director%s under the ASCII staging root.",
                    removed, "y" if removed == 1 else "ies")
    return removed


def reset_reaper_state() -> None:
    """「root ごとに 1 回」の記録を捨てる。**テスト専用。**"""
    with _lock:
        _reaped.clear()
