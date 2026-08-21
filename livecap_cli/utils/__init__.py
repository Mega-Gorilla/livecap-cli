from __future__ import annotations

"""Shared engine utilities (device detection, temp dirs, model paths)."""

import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from livecap_cli.resources import get_model_manager

__all__ = [
    "get_models_dir",
    "get_temp_dir",
    "detect_device",
    "unicode_safe_temp_directory",
    "unicode_safe_download_directory",
    "TempEnvironmentConflictError",
    "get_available_vram",
    "can_fit_on_gpu",
]


class TempEnvironmentConflictError(RuntimeError):
    """異なる purpose の temp environment スコープを入れ子にしようとした。

    ``OSError`` ではなく ``RuntimeError`` 派生にしているのは、呼び出し側の
    ``except OSError`` に握り潰されないようにするため (#378 §6.8)。
    #375 PR 2 で ``livecap_cli/paths/errors.py`` へ移動する。
    """


def _model_manager():
    return get_model_manager()


def get_models_dir(engine_name: Optional[str] = None) -> Path:
    """Return the shared models directory (optionally scoped per engine)."""
    return _model_manager().get_models_dir(engine_name)


def get_temp_dir(purpose: str = "runtime") -> Path:
    """Return a cache-backed temp directory for the given purpose."""
    return _model_manager().get_temp_dir(purpose)


def detect_device(requested_device: Optional[str], engine_name: str) -> str:
    """
    デバイスを検出して返す。

    Args:
        requested_device: 要求されたデバイス ("cuda", "cpu", None=auto)
        engine_name: エンジン名（ログ用）

    Returns:
        使用するデバイス ("cuda" または "cpu")
    """
    logger = logging.getLogger(__name__)

    if requested_device == "cpu":
        logger.info("Using CPU for %s (explicitly requested).", engine_name)
        return "cpu"

    try:
        import torch

        version = torch.__version__
        if torch.cuda.is_available():
            logger.info("Using CUDA for %s (PyTorch %s).", engine_name, version)
            return "cuda"

        if "+cpu" in version:
            logger.warning("PyTorch CPU build detected (%s); falling back to CPU for %s.", version, engine_name)
        else:
            logger.warning("CUDA unavailable (PyTorch %s); falling back to CPU for %s.", version, engine_name)
    except ImportError:
        logger.warning("PyTorch not installed; using CPU for %s.", engine_name)

    return "cpu"


def _override_temp_environment(temp_dir: Path):
    saved = {
        "TEMP": os.environ.get("TEMP"),
        "TMP": os.environ.get("TMP"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "tempdir": tempfile.tempdir,
    }

    temp_dir_str = str(temp_dir)
    os.environ["TEMP"] = temp_dir_str
    os.environ["TMP"] = temp_dir_str
    os.environ["TMPDIR"] = temp_dir_str
    tempfile.tempdir = temp_dir_str

    return saved


def _restore_temp_environment(saved):
    for key in ("TEMP", "TMP", "TMPDIR"):
        value = saved.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = saved.get("tempdir")


# ``TEMP`` / ``TMP`` / ``TMPDIR`` / ``tempfile.tempdir`` は**プロセス全体**の状態
# なので、変更は直列化しなければ一貫させられない。並行スコープが同時に書き換えると、
# 内側の snapshot が外側の**上書き済み**の値を掴み、外側の復元がそれを恒久的に
# 書き戻してしまう。
_TEMP_ENV_LOCK = threading.RLock()

# ``_TEMP_ENV_LOCK`` を保持している間だけ触ってよい。
_TEMP_ENV_STATE: dict = {"depth": 0, "purpose": None, "path": None, "saved": None}


@contextmanager
def _temp_environment(purpose: str, *, unique: bool) -> Iterator[Path]:
    """プロセス全体の TEMP を cache 配下へ一時的に向ける (Issue #386)。

    **このスコープは自分が作ったディレクトリを削除しない。** TEMP がプロセス全体
    なので、スコープが開いている間は**無関係なスレッドの ``NamedTemporaryFile()``
    もそこへ落ちる**。「自分が作ったディレクトリだから」と退出時に ``rmtree`` すると、
    共有ディレクトリの場合とまったく同じく別処理のファイルを消す (これが #386 で
    修正したデータ消失そのもの)。

    回収 (reaper) もここでは行わない。「別 pid かつ N 時間経過」は lease の代わりに
    ならない — 子プロセスは親の TEMP を継承するがディレクトリ名は親 pid のままで、
    pid は再利用され、複数プロセスが併存し得る。**生存判定はロックであって経過時間
    ではない。** 所有権マーカーとプロセスロックを持つ正式な回収は #375 PR 2 が担当し、
    それまでは ``cache_root`` 配下にディレクトリが残る (意図的なトレードオフ)。

    Args:
        purpose: ``cache_root`` 配下のサブディレクトリ名。
        unique: 最外周スコープごとに固有のサブディレクトリを作るか。

    Raises:
        TempEnvironmentConflictError: 別 purpose のスコープが既に開いている場合。
    """
    logger = logging.getLogger(__name__)

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
                f"temp environment already redirected for purpose {active!r}; "
                f"cannot nest purpose {purpose!r}. "
                "Nested scopes must share the same purpose, otherwise the yielded "
                "path would not match where tempfile actually writes."
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

        base = get_temp_dir(purpose)
        if unique:
            # 12 hex で衝突は事実上起こらない。短くしているのは MAX_PATH の余裕を
            # 残すため。万一衝突したら exist_ok=False で **黙って共有せず落ちる**。
            target = base / uuid.uuid4().hex[:12]
            target.mkdir(parents=True)
        else:
            target = base
            target.mkdir(parents=True, exist_ok=True)

        saved = _override_temp_environment(target)
        _TEMP_ENV_STATE.update(depth=1, purpose=purpose, path=target, saved=saved)
        try:
            yield target
        finally:
            _TEMP_ENV_STATE.update(depth=0, purpose=None, path=None, saved=None)
            _restore_temp_environment(saved)
            # ascii() で包むのは、cache_root がユーザー名を含み得るため。
            # 日本語 Windows では stderr がリダイレクトされると cp932 + strict に
            # なるので、素のパスを出すとログ自体が UnicodeEncodeError で落ちる。
            logger.debug(
                "Retained temp directory (reclaimed by #375 reaper): %s",
                ascii(str(target)),
            )
    finally:
        _TEMP_ENV_LOCK.release()


@contextmanager
def unicode_safe_temp_directory():
    """
    Temporarily point tempfile + env vars to a Unicode-safe cache directory.

    .. deprecated::
        **名前に反して ASCII 安全ではない** (``cache_root`` は appdirs 既定では
        ユーザー名を含む)。呼び出しはゼロで、#375 PR 3 で削除される。
    """
    with _temp_environment("runtime", unique=False) as temp_dir:
        yield temp_dir


@contextmanager
def unicode_safe_download_directory():
    """
    Point tempfile + env vars at a per-scope directory under the download cache.

    **共有ディレクトリを再帰削除しない** (Issue #386)。以前はスコープ退出時に
    ``cache_root/downloads`` を丸ごと ``rmtree`` しており、スコープ中に別スレッドが
    作った一時ファイル (発話ごとの wav を含む) まで削除していた。

    .. note::
        **ASCII 安全性は保証しない。** ``cache_root`` はユーザー名を含み得る。
        ASCII 保証は #375 PR 2 の ``ascii_safe_temp_environment()`` の契約であり、
        本ヘルパは #375 PR 3 で置き換えて削除する。
    """
    with _temp_environment("downloads", unique=True) as temp_dir:
        yield temp_dir


def get_available_vram() -> Optional[int]:
    """
    利用可能な VRAM（MB）を返す。

    Returns:
        VRAM（MB）。GPU なしまたは torch 未インストールの場合は None

    Note:
        torch がインストールされていない場合でも CTranslate2 は
        CUDA を使用可能。この関数は便利機能であり、必須ではない。
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except ImportError:
        pass
    return None


def can_fit_on_gpu(required_mb: int, safety_margin: float = 0.9) -> bool:
    """
    指定サイズが GPU に収まるか確認。

    Args:
        required_mb: 必要な VRAM（MB）
        safety_margin: 安全マージン（デフォルト 0.9 = 90%）

    Returns:
        収まる場合 True。GPU なしまたは torch なしの場合は False
    """
    available = get_available_vram()
    if available is None:
        return False
    return available * safety_margin >= required_mb
