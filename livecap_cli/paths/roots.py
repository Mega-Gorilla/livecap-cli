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

なぜ選定結果をまるごとキャッシュするのか
------------------------------------
**path だけをキャッシュすると「なぜその root になったか」が 2 回目以降失われる。**
運用者にとって重要なのは「cache root が選ばれた」ことではなく「``%ProgramData%``
が長すぎたので cache root へ降りた」ことである。したがって :class:`RootSelection`
として**選択元と拒否された候補の理由まで**保持し、staging のたびに境界名と併せて
ログへ出す (Issue #375 の AC)。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from livecap_cli.resources import freeze_and_snapshot
from livecap_cli.resources.configuration import (
    ENV_ASCII_STAGING_DIR,
    STAGING_ROOT_MAX_LEN,
    normalize_path,
)

from .errors import AsciiStagingUnavailableError

logger = logging.getLogger(__name__)

__all__ = [
    "RootSelection",
    "resolve_staging_root",
    "select_staging_root",
    "log_staging_use",
    "is_ascii_safe",
    "reset_staging_root_cache",
    "validate_purpose",
    "PURPOSE_MAX_LEN",
]

#: ``purpose`` に許す最大長。``STAGING_ROOT_MAX_LEN`` の予算計算がこの値を前提に
#: している (``<root>\<purpose>\<uuid12>``)。**契約として強制する** — 計算の
#: 前提を呼び出し側の善意に委ねない。
PURPOSE_MAX_LEN = 16

#: ``purpose`` は **ASCII の basename ひとつ**。separator も ``.`` / ``..`` も許さない。
_PURPOSE_RE = re.compile(r"\A[A-Za-z0-9_-]{1,%d}\Z" % PURPOSE_MAX_LEN)

#: 全候補が使う ASCII リテラルのディレクトリ名。ユーザー名やロケール由来の文字列を
#: 混ぜないこと — それをすると候補自身が非 ASCII になる。
_DIR_NAME = "LiveCap"
_STAGING_LEAF = "staging"

_lock = threading.RLock()
#: ``(configuration の同一性キー, 選定結果)``。
#:
#: **キーを持つのが要点。** 素のキャッシュだと、``_reset_resources_for_tests()``
#: などで configuration が入れ替わったときに古い root を返し続ける。キーが変われば
#: 自動的に選び直す。
#:
#: **path ではなく :class:`RootSelection` を持つ。** 拒否された候補の理由は後続候補が
#: 成功した時点で失われる情報なので、2 回目以降の staging でもログへ出せるよう
#: 選定結果ごと保持する。
_cached: Optional[Tuple[Tuple, "RootSelection"]] = None

#: 既に INFO を出した ``(boundary, mechanism, root)``。``_lock`` の下でだけ触る。
_logged: Set[Tuple[str, str, str]] = set()


def validate_purpose(purpose: str, *, boundary: str) -> str:
    """``purpose`` が staging root の中に留まり、保証を壊さないことを確かめる。

    公開 API が受け取った値をそのまま path へ連結するので、**検証しないと保証が
    3 つとも破れる**:

    - ``"日本語"`` -> 完成した path が**非 ASCII になる** (この API の存在意義が消える)
    - ``"../outside"`` -> **staging root の外へ出る**
    - 長い文字列 -> ``STAGING_ROOT_MAX_LEN`` の予算計算が崩れる

    Raises:
        ValueError: slug 契約 (``[A-Za-z0-9_-]``、1..16 文字) を満たさないとき。
    """
    if not isinstance(purpose, str) or not _PURPOSE_RE.match(purpose):
        raise ValueError(
            f"{boundary}: purpose must be an ASCII slug matching "
            f"[A-Za-z0-9_-]{{1,{PURPOSE_MAX_LEN}}} (got {purpose!r}). "
            "It is joined onto the ASCII staging root, so anything else would "
            "break the ASCII guarantee, escape the root, or blow the path budget."
        )
    return purpose


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


def _config_key(config) -> Tuple:
    """キャッシュの同一性キー。**staging 指定と cache root で決まる。**"""
    policy = config.staging_policy
    return (
        str(policy.configured_root) if policy.configured_root else None,
        policy.source,
        str(config.cache_root),
    )


def _candidates(config, source_volume: Optional[str]) -> List[Tuple[str, Optional[Path]]]:
    """``(説明, path)`` の順序付き候補。先勝ち。

    **順序は契約である。** 後から先頭へ差し込むと、既存環境の staging root が
    黙って移動する。``source_volume`` 候補は現状の 2 API (temp environment /
    workspace) では使わない (source が無い) が、**将来 file / dir staging を足す
    ときに hardlink 段を生かすための席**として今のうちに確保しておく。
    """
    out: List[Tuple[str, Optional[Path]]] = []

    # 0. 明示指定 (API / env)。configure_resources() が freeze 時に検証済み。
    policy = config.staging_policy
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
    out.append(("cache root", normalize_path(config.cache_root / "ascii-staging")))
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


@dataclass(frozen=True, slots=True)
class RootSelection:
    """選ばれた staging root と、**そこへ至った経緯**。

    ``path`` だけでは「なぜその root なのか」が残らない。優先候補を落とした理由は、
    後続候補が成功した時点で失われる情報なので、**選定と同時に captureする**。
    """

    #: 採用された root。
    path: Path
    #: どの候補が採用されたか (``"%ProgramData%"`` / ``"cache root"`` 等)。
    root_source: str
    #: 拒否された候補と理由。``(候補の説明, 理由)`` の順で ladder 順に並ぶ。
    fallbacks: Tuple[Tuple[str, str], ...]
    #: **staging 元**のボリューム — 呼び出し側が渡した入力そのもの。
    #:
    #: 採用された root の drive **ではない**。``D:`` から staging しようとして
    #: 同一ボリューム候補が拒否され ``C:\\ProgramData\\...`` へ降りた場合、ここは
    #: ``"D:"`` のまま残る。**そうでないと fallback の関係が説明できない**
    #: (どこから来てどこへ降りたのかが分からなくなる)。採用先の drive が要るなら
    #: ``path`` から求められるので、両方を持つ必要はない。
    source_volume: Optional[str]


def select_staging_root(*, boundary: str, source_volume: Optional[str] = None) -> Path:
    """ASCII 保証された staging root の path を返す。

    経緯まで要るなら :func:`resolve_staging_root` を使うこと。
    """
    return resolve_staging_root(boundary=boundary, source_volume=source_volume).path


def resolve_staging_root(
    *, boundary: str, source_volume: Optional[str] = None
) -> RootSelection:
    """ASCII 保証された staging root を、**選択元と拒否理由つきで**返す。

    Args:
        boundary: どのネイティブ境界のために必要か。**失敗メッセージに必ず出す**
            ため必須にしている。
        source_volume: staging 元のボリューム (``"D:"`` 等)。同一ボリューム候補を
            最優先にして hardlink を効かせるためのもの。現行 caller は ``None``。

    Raises:
        AsciiStagingUnavailableError: 候補が全滅したとき。**元の非 ASCII path へ
            黙って fallback することはしない。**
    """
    global _cached
    with _lock:
        # **ここで freeze する。** preview を読むと、この呼び出しの後に
        # configure_resources(staging_root=...) が成功してしまい、**既に配った
        # root と食い違う設定が黙って受け入れられる**。resolved 値を配る操作は
        # configuration を確定させなければならない。
        config = freeze_and_snapshot()
        key = _config_key(config)

        if _cached is not None and _cached[0] == key and source_volume is None:
            # **経緯ごと返す。** path だけを cache していた頃は、2 回目以降の
            # staging で「なぜこの root か」が観測できなかった。
            return _cached[1]

        candidates = _candidates(config, source_volume)
        explicit = config.staging_policy.configured_root is not None

        attempts: List[Tuple[str, str]] = []
        for index, (label, candidate) in enumerate(candidates):
            if candidate is None:
                continue
            reason = _reject_reason(candidate)

            if reason is not None:
                # **明示指定が使えなくなったら候補へ降りない** (R2)。configure 時は
                # 有効でも、その後 ACL 変更・削除・容量で使えなくなり得る。降りると
                # 「運用者が指定した場所を黙って使わない」ことになる。
                if explicit and index == 0:
                    raise AsciiStagingUnavailableError(
                        f"{boundary}: the configured ASCII staging root is no longer "
                        f"usable ({reason}): {ascii(str(candidate))}. "
                        f"Fix it or change {ENV_ASCII_STAGING_DIR} — "
                        "falling back to another location would silently ignore "
                        "an explicit setting.",
                        boundary=boundary,
                        attempts=[(f"{label}: {ascii(str(candidate))}", reason)],
                    )
                attempts.append((f"{label}: {ascii(str(candidate))}", reason))
                continue

            selection = RootSelection(
                path=candidate,
                root_source=label,
                fallbacks=tuple(attempts),
                source_volume=source_volume,
            )
            _record_selected_root(selection)
            # root の**初回使用時に 1 回だけ**残骸を回収する。
            # ascii_safe_temp_environment() は自分のディレクトリを消さない
            # (#386) ので、放っておくと積み上がる。best-effort。
            _reap_once(candidate)
            if source_volume is None:
                _cached = (key, selection)
            return selection

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


def _record_selected_root(selection: RootSelection) -> None:
    """選んだ root を readback へ載せる。

    記録の実体は ``resources`` 側に置く。``paths`` が書き ``resources`` が読む —
    逆向きにすると循環 import になる。
    """
    from livecap_cli.resources.configuration import record_staging_root

    record_staging_root(
        path=selection.path,
        # **入力をそのまま載せる。** ここで ``splitdrive(selection.path)`` を計算すると
        # 「staging 元」ではなく「採用先の drive」になり、field の定義と食い違う。
        # 現行 2 API は source を持たないので ``None`` が入る (それが正しい)。
        source_volume=selection.source_volume,
        root_source=selection.root_source,
        fallbacks=selection.fallbacks,
        selected_at=time.time(),
    )


def log_staging_use(selection: RootSelection, *, boundary: str, mechanism: str) -> None:
    """staging 発生を 1 行の構造化ログへ出す (Issue #375 の AC)。

    **境界・mechanism・root の組み合わせごとに 1 回だけ INFO**、以降は DEBUG。

    毎回 INFO にしない理由: :func:`~livecap_cli.paths.workspace.ascii_safe_workspace`
    は**発話ごと**に呼ばれる (PR 4 で 5 engine が移行する)。realtime 転写で 1 発話 1 行
    出すとログが埋まり、**肝心の 1 行が読めなくなる**。一方 DEBUG だけだと通常の
    CLI / GUI ログで観測できない — AC が求めているのは「運用者が見える」ことなので、
    **初回を INFO**にして両立させる。

    ``mechanism`` は**どの staging API を通ったか** (``temp-environment`` /
    ``workspace``)。root の**選択元**は ``root_source`` で別に出す — 本 repo では
    "mechanism" を hardlink / copy の materialization の意味で使っており
    (``tests/nonascii/artifacts.py``)、混ぜると読み手が誤解する。
    """
    fallbacks = (
        "[" + "; ".join(f"{where} -> {why}" for where, why in selection.fallbacks) + "]"
        if selection.fallbacks
        else "[]"
    )
    message = (
        "ASCII staging: boundary=%s mechanism=%s resolved_root=%s "
        "root_source=%s fallbacks=%s"
    )
    args = (
        boundary,
        mechanism,
        # ascii() で包むのは、root が非 ASCII な cache root 由来のこともあるため。
        # 日本語 Windows では stderr がリダイレクトされると cp932 + strict になり、
        # 素の path を出すとログ自体が UnicodeEncodeError で落ちる。
        ascii(str(selection.path)),
        selection.root_source,
        fallbacks,
    )

    seen = (boundary, mechanism, str(selection.path))
    with _lock:
        first = seen not in _logged
        if first:
            _logged.add(seen)

    if first:
        logger.info(message, *args)
    else:
        logger.debug(message, *args)


def reset_staging_root_cache() -> None:
    """選定結果のキャッシュを捨てる。**テスト専用。**

    root は 1 プロセス内で動かない前提なのでキャッシュしている。env や
    configuration を差し替えるテストはこれを呼ぶ。
    """
    global _cached
    with _lock:
        _cached = None
        _logged.clear()
