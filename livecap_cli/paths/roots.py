"""ASCII 保証された staging root の選定 (Issue #375 PR 2、契約は #378 §6.5)。

なぜ root を選ぶ必要があるのか
----------------------------
既定の ``cache_root`` は ``appdirs.user_cache_dir()`` 由来で**ユーザー名を含む**。
Windows のユーザー名が非 ASCII だと、そこへ ``%TEMP%`` を向けても
「ASCII が必要なネイティブ境界」の役に立たない。ASCII だと**保証できる**場所を
別に選ぶ必要がある。

明示指定は PR 1 が freeze 時に検証済み
------------------------------------
``configure_resources(staging_root=...)`` と ``LIVECAP_CORE_ASCII_STAGING_DIR`` は
**PR 1 が freeze 時に ASCII / 長さ / 書き込み可能を検証**し、不正なら
``AsciiStagingUnavailableError`` を送出している。したがってここでは readback を
読むだけでよく、**env を直読みしない** — manager が env を読む構図は PR 1 が
潰したものである。
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from livecap_cli.resources import get_resource_configuration
from livecap_cli.resources.configuration import (
    ENV_ASCII_STAGING_DIR,
    STAGING_ROOT_MAX_LEN,
    normalize_path,
)

from .errors import AsciiStagingUnavailableError

logger = logging.getLogger(__name__)

__all__ = ["select_staging_root", "is_ascii_safe", "reset_staging_root_cache"]

#: 全候補が使う ASCII リテラルのディレクトリ名。ユーザー名やロケール由来の文字列を
#: 混ぜないこと — それをすると候補自身が非 ASCII になる。
_DIR_NAME = "LiveCap"
_STAGING_LEAF = "staging"

_lock = threading.RLock()
_cached_root: Optional[Path] = None


def is_ascii_safe(path: os.PathLike[str] | str) -> bool:
    """ネイティブ境界へ渡して安全な path か (ASCII のみか)。"""
    return str(path).isascii()


def _anonymous_user_tag() -> str:
    """``%ProgramData%`` 配下でユーザーを分けるための短いタグ。

    **ユーザー名そのものは絶対に使わない** (#378 §6.5)。非 ASCII なユーザー名を
    候補 path に混ぜたら、ASCII 保証という目的そのものを壊す。共有領域なので
    分離は要るが、識別子は hash で足りる。
    """
    raw = os.environ.get("USERNAME") or os.environ.get("USER") or "anonymous"
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:8]


def _candidates(source_volume: Optional[str]) -> List[Tuple[str, Optional[Path]]]:
    """``(説明, path)`` の順序付き候補。先勝ち。

    **順序は契約である。** 後から先頭へ差し込むと、既存環境の staging root が
    黙って移動する。``source_volume`` 候補は現状の 2 API (temp environment /
    workspace) では使わない (source が無い) が、**将来 file / dir staging を足す
    ときに hardlink 段を生かすための席**として今のうちに確保しておく。
    """
    out: List[Tuple[str, Optional[Path]]] = []

    # 0. 明示指定 (API / env)。PR 1 が freeze 時に検証済み。
    policy = get_resource_configuration().staging_policy
    if policy.configured_root is not None:
        out.append(
            (f"explicit staging root ({policy.source})", normalize_path(policy.configured_root))
        )

    # 1. ソースと同一ボリューム。hardlink 段を生かすため最上位に置く。
    if source_volume:
        out.append(("source volume", normalize_path(Path(source_volume) / f"{_DIR_NAME}Staging")))

    # 2-4. OS 提供の共有領域。いずれもドライブレター + ASCII リテラルで構成される。
    program_data = os.environ.get("ProgramData")
    if program_data:
        out.append((
            "%ProgramData%",
            normalize_path(Path(program_data) / _DIR_NAME / _STAGING_LEAF / _anonymous_user_tag()),
        ))
    system_drive = os.environ.get("SystemDrive")
    if system_drive:
        # "C:" だけだとカレントディレクトリ相対になるので区切りを足す。
        out.append((
            "%SystemDrive%",
            normalize_path(Path(system_drive + os.sep) / _DIR_NAME / _STAGING_LEAF),
        ))
    public = os.environ.get("PUBLIC")
    if public:
        out.append((
            "%PUBLIC%",
            normalize_path(Path(public) / _DIR_NAME / _STAGING_LEAF),
        ))

    # 5-6. ASCII 保証が無い候補。述語を通った場合のみ採用される。
    out.append(("cache root", normalize_path(get_resource_configuration().cache_root / "ascii-staging")))
    out.append(("system temp", normalize_path(Path(tempfile.gettempdir()) / "livecap-ascii")))
    return out


def _reject_reason(path: Path) -> Optional[str]:
    """述語を満たさない理由。満たすなら ``None``。

    順序が意味を持つ: **filesystem を触る前に安い判定を済ませる**。
    """
    text = str(path)
    if not text.isascii():
        return "not ASCII"
    if len(text) > STAGING_ROOT_MAX_LEN:
        return f"too long ({len(text)} > {STAGING_ROOT_MAX_LEN})"

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return f"cannot create: {error}"

    # **書き込み probe。** Windows の ACL 検査は当てにならないので実際に書く。
    # 固定名にしないのは、同名ファイルがあったときに truncate してから消すことに
    # なるため (PR 1 のレビューで実際に指摘された経路)。
    try:
        handle, probe = tempfile.mkstemp(dir=path, prefix=".livecap-probe-")
    except OSError as error:
        return f"not writable: {error}"
    os.close(handle)
    try:
        os.unlink(probe)
    except OSError:
        pass
    return None


def select_staging_root(
    *, boundary: str, source_volume: Optional[str] = None
) -> Path:
    """ASCII 保証された staging root を返す。

    Args:
        boundary: どのネイティブ境界のために必要か。**失敗メッセージに必ず出す**
            ため必須にしている。
        source_volume: staging 元のボリューム (``"D:"`` 等)。同一ボリューム候補を
            最優先にして hardlink を効かせるためのもの。現行 caller は ``None``。

    Raises:
        AsciiStagingUnavailableError: 候補が全滅したとき。**元の非 ASCII path へ
            黙って fallback することはしない。**
    """
    global _cached_root
    with _lock:
        if _cached_root is not None and source_volume is None:
            return _cached_root

        attempts: List[Tuple[str, str]] = []
        for label, candidate in _candidates(source_volume):
            if candidate is None:
                continue
            reason = _reject_reason(candidate)
            if reason is None:
                logger.debug(
                    "ASCII staging root for %s: %s (%s)", boundary, ascii(str(candidate)), label
                )
                _record_selected_root(candidate, label)
                # root の**初回使用時に 1 回だけ**残骸を回収する。
                # ascii_safe_temp_environment() は自分のディレクトリを消さない
                # (#386) ので、放っておくと積み上がる。best-effort。
                _reap_once(candidate)
                if source_volume is None:
                    _cached_root = candidate
                return candidate
            attempts.append((f"{label}: {ascii(str(candidate))}", reason))

        tried = "; ".join(f"{where} -> {why}" for where, why in attempts) or "no candidate"
        raise AsciiStagingUnavailableError(
            f"{boundary}: no ASCII-safe staging root is available. Tried: {tried}. "
            f"Set {ENV_ASCII_STAGING_DIR} to a writable ASCII path "
            f"(at most {STAGING_ROOT_MAX_LEN} characters).",
            boundary=boundary,
            attempts=attempts,
        )


def _reap_once(root: Path) -> None:
    """孤児回収。**失敗しても本筋を止めない。**"""
    from .reaper import reap_staging_root

    try:
        reap_staging_root(root)
    except Exception as error:  # pragma: no cover - reaper 側で握っているはず
        logger.debug("Staging reaper failed (ignored): %s", error)


def _record_selected_root(path: Path, mechanism: str) -> None:
    """選んだ root を readback へ載せる。

    記録の実体は ``resources`` 側に置く。``paths`` が書き ``resources`` が読む —
    逆向きにすると循環 import になる。
    """
    from livecap_cli.resources.configuration import record_staging_root

    record_staging_root(
        path=path,
        source_volume=os.path.splitdrive(str(path))[0] or None,
        mechanism=mechanism,
        selected_at=time.time(),
    )


def reset_staging_root_cache() -> None:
    """選定結果のキャッシュを捨てる。**テスト専用。**

    root は 1 プロセス内で動かない前提なのでキャッシュしている。env や
    configuration を差し替えるテストはこれを呼ぶ。
    """
    global _cached_root
    with _lock:
        _cached_root = None
