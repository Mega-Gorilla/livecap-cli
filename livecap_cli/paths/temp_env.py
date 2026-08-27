"""プロセス全体の TEMP を ASCII 保証された場所へ向ける (Issue #375 PR 2)。

契約は #378 §6.7 / §6.10。実装の本体は `livecap_cli/utils/__init__.py` にあったものを
移設した — **ロック実装を 2 つ保守しない**ため (#378 §6.11)。旧 helper
``unicode_safe_download_directory()`` は本 module へ委譲する薄い層になったのち、
**#375 PR 3 で削除された** (名前に反して ASCII 保証が無く、誤読を招くため)。

なぜ退出時にディレクトリを消さないのか
------------------------------------
``TEMP`` / ``TMP`` / ``TMPDIR`` / ``tempfile.tempdir`` は**プロセス全体**の状態なので、
スコープが開いている間は**無関係なスレッドの ``NamedTemporaryFile()`` もそこへ落ちる**。
「自分が作ったディレクトリだから」と退出時に ``rmtree`` すると、共有ディレクトリの
場合とまったく同じく別処理のファイルを消す — これが #386 で修正したデータ消失そのもの。

残骸の回収は :mod:`livecap_cli.paths.reaper` が TTL で行う。

支えない範囲: **スコープ内で完了する同期境界だけ**
------------------------------------------------
**親のスコープより長生きする子プロセスは支えない。** 子は ``%TEMP%`` を継承するが、
lease のハンドルは既定で非継承 (PEP 446) であり、親がスコープを抜けた時点で解放される。
その後 TTL を超えれば、まだ使っている子の足元を reaper が回収し得る。

したがって**この context の中で spawn した子プロセスは、抜ける前に終了 / join させること**。
支えるには子側も lease を握る別プロトコルが必要で、v1 の範囲外である。

**「我々が作るファイル」には使わないこと。** 発話ごとの wav のような用途は
:func:`livecap_cli.paths.workspace.ascii_safe_workspace` が正解で、プロセスグローバル
状態を発話ごとに書き換えるのは現行バグの縮小再生産になる (#378 §6.10)。

明示的な非保証
------------
- **単一スレッド上の複数 async task から使わないこと。** 排他は ``threading.RLock``
  (スレッド単位の再入ロック) なので、**同じ event loop スレッド上で複数の task が
  ``await`` を跨いでこの同期 context manager を交差利用すると、字句的なネストと
  区別できない**。内側の退出が外側の深度を下げ、環境が早すぎるタイミングで復元され得る。
  非同期からは、**enter・境界処理・exit を同じ同期関数にまとめ、その関数全体を 1 回の
  ``asyncio.to_thread()`` で実行する**こと。**enter と exit を別々の呼び出しへ分割しない** —
  別の worker スレッドで走り得るが、``RLock`` はスレッド所有権を持つので取得したスレッド
  以外からは解放できない (``RuntimeError: cannot release un-acquired lock``)。
- **別スレッドの呼び出しは ``purpose`` に関係なくすべて直列化される。** 排他をスコープの
  全期間保持するためで、待たされるのであって ``TempEnvironmentConflictError`` にはならない
  (同エラーは**同一スレッドで別 purpose をネスト**したときに出る)。
- ``os.environ`` / ``tempfile.tempdir`` はプロセスグローバルなので、**移設ウィンドウ中に
  読む並行コードは移設後の値を見る**。ロックは変更を*一貫*させられるが*スレッドスコープ*
  にはできない。**ウィンドウは最小に**すること (境界呼び出しだけを包む)。
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .errors import TempEnvironmentConflictError
from .lease import hold_lease
from .roots import log_staging_use, select_staging_root, validate_purpose

logger = logging.getLogger(__name__)

__all__ = ["ascii_safe_temp_environment"]

# ``TEMP`` / ``TMP`` / ``TMPDIR`` / ``tempfile.tempdir`` は**プロセス全体**の状態
# なので、変更は直列化しなければ一貫させられない。並行スコープが同時に書き換えると、
# 内側の snapshot が外側の**上書き済み**の値を掴み、外側の復元がそれを恒久的に
# 書き戻してしまう。
_TEMP_ENV_LOCK = threading.RLock()

# ``_TEMP_ENV_LOCK`` を保持している間だけ触ってよい。
_TEMP_ENV_STATE: dict = {"depth": 0, "purpose": None, "path": None, "saved": None}


def _override(temp_dir: Path) -> dict:
    saved = {
        "TEMP": os.environ.get("TEMP"),
        "TMP": os.environ.get("TMP"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "tempdir": tempfile.tempdir,
    }
    text = str(temp_dir)
    os.environ["TEMP"] = text
    os.environ["TMP"] = text
    os.environ["TMPDIR"] = text
    tempfile.tempdir = text
    return saved


def _restore(saved: dict) -> None:
    for key in ("TEMP", "TMP", "TMPDIR"):
        value = saved.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = saved.get("tempdir")


@contextmanager
def temp_environment(
    purpose: str,
    *,
    base: Path,
    boundary: Optional[str] = None,
) -> Iterator[Path]:
    """内部コア。ロック / ネスト / lease / env 復元を持つ。

    ``base`` の ASCII 保証は**呼び出し側の責任**である。唯一の呼び出しは
    :func:`ascii_safe_temp_environment` で、そちらが候補 ladder で ASCII root を
    選び、``purpose`` を検証し、staging ログを出してから ``base`` を渡す。

    Args:
        purpose: ``base`` 配下のサブディレクトリ名。
        base: 親ディレクトリ。**必須** — 既定値を持たせると「ASCII 保証の無い
            場所へ黙って移設する」経路が復活する (旧 ``unicode_safe_download_directory``
            がそれだった。#375 PR 3 で削除済み)。
        boundary: 失敗メッセージに出す境界名。**診断契約の 1 番目**なので、
            公開 API からは必ず渡す。

    Raises:
        TempEnvironmentConflictError: 別 purpose のスコープが既に開いている場合。
    """
    # RLock は同一スレッドでは再入できるので、ネストではここで待たない。
    if not _TEMP_ENV_LOCK.acquire(blocking=False):
        logger.info(
            "Waiting for another temp-environment scope to finish (purpose=%s).", purpose
        )
        _TEMP_ENV_LOCK.acquire()

    try:
        active = _TEMP_ENV_STATE["purpose"]
        if _TEMP_ENV_STATE["depth"] > 0 and active != purpose:
            raise TempEnvironmentConflictError(
                f"{boundary or purpose}: temp environment already redirected for "
                f"purpose {active!r}; cannot nest purpose {purpose!r}. "
                "Nested scopes must share the same purpose, otherwise the yielded "
                "path would not match where tempfile actually writes.",
                boundary=boundary,
            )

        if _TEMP_ENV_STATE["depth"] > 0:
            # ネストは**外側のディレクトリを再利用し、同じ path を返す**。
            # 内側が自前のディレクトリを作っても環境は外側を指したままなので、
            # 別 path を返すと呼び出し側に嘘をつくことになる。
            _TEMP_ENV_STATE["depth"] += 1
            try:
                yield _TEMP_ENV_STATE["path"]
            finally:
                _TEMP_ENV_STATE["depth"] -= 1
            return

        base = base / purpose
        base.mkdir(parents=True, exist_ok=True)

        # **スコープごとに固有のディレクトリを作る。** 12 hex で衝突は事実上
        # 起こらない。短くしているのは MAX_PATH の余裕を残すため。万一衝突したら
        # exist_ok=False で **黙って共有せず落ちる**。
        target = base / uuid.uuid4().hex[:12]
        target.mkdir(parents=True)

        # **env を書き換える*前*に lease を確立し、復元し終わるまで保持する。**
        # 逆順にすると「プロセス全体の TEMP が target を指しているのに lease が
        # 無い」区間が生まれ、その隙に別プロセスの reaper が消せてしまう。
        with hold_lease(target, boundary=boundary or purpose):
            saved = _override(target)
            _TEMP_ENV_STATE.update(depth=1, purpose=purpose, path=target, saved=saved)
            try:
                yield target
            finally:
                _TEMP_ENV_STATE.update(depth=0, purpose=None, path=None, saved=None)
                _restore(saved)
                # ascii() で包むのは、base がユーザー名を含み得るため。日本語 Windows
                # では stderr がリダイレクトされると cp932 + strict になるので、素の
                # パスを出すとログ自体が UnicodeEncodeError で落ちる。
                logger.debug("Temp environment restored (was %s).", ascii(str(target)))
    finally:
        _TEMP_ENV_LOCK.release()


@contextmanager
def ascii_safe_temp_environment(
    *, boundary: str, purpose: str = "runtime"
) -> Iterator[Path]:
    """``%TEMP%`` を **ASCII 保証された**ディレクトリへ向ける。

    非 ASCII な ``%TEMP%`` の中でネイティブライブラリが自前でアーカイブを展開する、
    といった境界のためのもの。**渡す path ではなく展開先が壊れている**ケースに効く
    (#379 の NeMo が実例)。

    Args:
        boundary: どのネイティブ境界のためか。失敗メッセージに必ず出すので必須。
        purpose: staging root 配下のサブディレクトリ名。

    Raises:
        AsciiStagingUnavailableError: ASCII 保証された root を用意できないとき。
        AsciiPathError: 所有権マーカー兼 lease を確立できないとき。
        TempEnvironmentConflictError: 別 purpose のスコープが開いているとき。

    Example:
        >>> with ascii_safe_temp_environment(boundary="parakeet.nemo.restore_from.untar"):
        ...     model = ASRModel.restore_from(restore_path=str(model_path))
    """
    validate_purpose(purpose, boundary=boundary)
    selection = select_staging_root(boundary=boundary)
    # **staging 発生を 1 行で観測できるようにする** (Issue #375 の AC)。root が
    # cache hit でも出す — 「なぜこの root か」は 2 回目以降こそ分からなくなる。
    log_staging_use(selection, boundary=boundary, mechanism="temp-environment")
    with temp_environment(purpose, base=selection.path, boundary=boundary) as target:
        yield target
