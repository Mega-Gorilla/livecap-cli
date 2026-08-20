"""ASCII 保証された base root の探索 (Issue #378、レビュー指摘 1)。

**なぜ専用の探索が要るのか**: 本ハーネスが最も測りたいのは
「Windows ユーザー名が非 ASCII の環境」である。ところがその環境では
``tempfile.mkdtemp()`` が ``C:\\Users\\<非ASCII>\\AppData\\Local\\Temp`` 配下に
落ちるため、base root 自体が非 ASCII になる。base root が非 ASCII だと
「variant セグメントだけを非 ASCII にする」という差分設計が成立せず、
session ごと skip される — **検証したい実環境でハーネスが動かない**。

そこで env override の有無に関わらず ASCII かつ書き込み可能な候補を探索する。
候補の並びは、実装側 (`ascii_safe_path()` の staging root、棚卸し表 §6.5) と
意図的に同じ考え方にしてある — ユーザー名由来のパスを避け、
ボリューム root と ``%ProgramData%`` / ``%PUBLIC%`` を優先する。

**候補は「共有される親」であって session root ではない。** 候補パスは固定名なので、
2 つの run が同時に走ると同じ probe パスを読み書きし、片方の teardown が
もう片方の実行中データを消してしまう。これは本調査が問題視している
``unicode_safe_download_directory`` の「共有ディレクトリを rmtree する」欠陥と
同じ構造である。したがって親の下に **PID + UUID の session 固有 root** を作り、
後始末はその session root だけに限定する。

**削除するのは所有権マーカーを持つものだけ。** ``LIVECAP_NONASCII_ROOT`` には
利用者が任意の既存ディレクトリを指定できるので、「``run-*`` という名前で古いもの」
だけを条件に再帰削除すると無関係な ``run-backup`` を消しかねない。
session 作成時にマーカーを書き、reaper は**厳密な名前形式**と**有効なマーカー**の
両方を満たすものだけを削除する。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

#: session root の下にハーネスが作る最長サフィックスの見積り。
#: 実測 (2026-08-20): ``test folder (1)/onnxruntime.InferenceSession.str_path/
#: model/encoder-epoch-99-avg-1.int8.onnx`` = 93 文字。余裕を見て 100 とする。
MAX_PROBE_SUFFIX_LEN = 100

#: session root に許す最大長。MAX_PATH (260) からプローブ側の予算を引いたもの。
MAX_SESSION_ROOT_LEN = 260 - MAX_PROBE_SUFFIX_LEN

#: ``/run-<pid>-<uuid8>`` の最大長 (pid は 10 桁まで見込む)。
SESSION_SUFFIX_LEN = len("/run-") + 10 + 1 + 8

#: **共有される親**に許す最大長。session suffix 分を先に予約しておかないと、
#: 「親は上限以内」を満たしても実際の base root がその分だけ超過し、
#: MAX_PATH の予算保証が成立しない。
#:
#: 名前を session root 側と分け、``is_usable()`` には「上限そのもの」を渡す形に
#: してある — **二重に予約してしまう事故**を防ぐため (実際に一度やらかした:
#: 定数に suffix を織り込んだうえで呼び出し側でも引いてしまい、実効上限が 112 に
#: なって CI が 113 文字の親で落ちた)。
MAX_PARENT_ROOT_LEN = MAX_SESSION_ROOT_LEN - SESSION_SUFFIX_LEN

_LEAF = "livecap-nonascii-probe"

#: 異常終了した run が残した session root を掃除する閾値。
STALE_SESSION_HOURS = 6.0

#: session root の所有権マーカー。**これが無いディレクトリは絶対に削除しない。**
SESSION_MARKER_NAME = ".livecap-nonascii-session.json"
SESSION_MAGIC = "livecap-nonascii-probe-session"
SESSION_MARKER_SCHEMA = 1

#: 厳密な session root 名。マーカーと**両方**を満たすものだけが削除対象になる。
#: glob の ``run-*`` だけでは ``run-backup`` や ``run-2025`` も引っかかる。
_SESSION_NAME_RE = re.compile(r"^run-[0-9]{1,10}-[0-9a-f]{8}$")


@dataclass(frozen=True)
class RootCandidate:
    label: str
    path: Path | None
    ephemeral: bool = False


def _volume_of(path: Path | None) -> Path | None:
    if path is None:
        return None
    anchor = Path(path).anchor
    return Path(anchor) if anchor else None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def candidates(models_root: Path | None, repo_root: Path) -> list[RootCandidate]:
    """先勝ちの候補列。

    ``models_root`` と同一ボリュームの候補を上位に置くのは、実モデル tier で
    ``os.link`` を効かせるため (8.8 GB のモデルでも 0 バイト・ミリ秒で実体化できる)。
    """
    out: list[RootCandidate] = []

    model_volume = _volume_of(models_root)
    if model_volume is not None:
        out.append(RootCandidate("model volume", model_volume / _LEAF))

    # repo 直下は開発時に最も確実に書ける。repo path が非 ASCII なら述語で落ちる。
    out.append(RootCandidate("repo .tmp", repo_root / ".tmp" / _LEAF))

    program_data = _env_path("ProgramData")
    if program_data is not None:
        out.append(RootCandidate("%ProgramData%", program_data / "LiveCap" / _LEAF))

    system_drive = os.environ.get("SystemDrive")
    if system_drive:
        out.append(RootCandidate("%SystemDrive%", Path(system_drive + os.sep) / _LEAF))

    public = _env_path("PUBLIC")
    if public is not None:
        out.append(RootCandidate("%PUBLIC%", public / "LiveCap" / _LEAF))

    if sys.platform != "win32":
        out.append(RootCandidate("/tmp", Path("/tmp") / _LEAF))

    # 最後の手段。ユーザー名が ASCII なら通るが、非 ASCII なら述語で落ちる。
    out.append(
        RootCandidate("system temp", Path(tempfile.gettempdir()) / _LEAF, ephemeral=True)
    )
    return out


def is_usable(path: Path, *, limit: int = MAX_SESSION_ROOT_LEN) -> tuple[bool, str]:
    """ASCII かつ十分短く、実際に書き込めるか。

    ``limit`` は**上限そのもの**を渡す (差分ではない)。共有親を判定するときは
    ``MAX_PARENT_ROOT_LEN`` を渡すこと — 親が上限内でも、後から付く
    ``/run-<pid>-<uuid>`` の分だけ実際の base root が超過してしまうため。

    Windows の ACL 検査は当てにならないので**書き込みプローブ**で判定する。
    """
    text = str(path)
    if not text.isascii():
        return False, "非 ASCII"
    if len(text) > limit:
        return False, f"長すぎる ({len(text)} > {limit})"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"mkdir 失敗: {type(exc).__name__} errno={exc.errno}"
    probe = path / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"書き込み不可: {type(exc).__name__} errno={exc.errno}"
    return True, ""


def resolve_base_root(
    *,
    override: str | None,
    models_root: Path | None,
    repo_root: Path,
) -> tuple[Path, str, list[tuple[str, str]]]:
    """(**共有される親** root, 採用した候補ラベル, 落ちた候補と理由) を返す。

    返るのは session root ではない。呼び出し側は ``create_session_root()`` で
    session 固有のサブディレクトリを作ること。

    ``override`` (``LIVECAP_NONASCII_ROOT``) が指定されていて述語を満たさない場合は
    **黙って fallback せず** ``RuntimeError`` を投げる — 運用者の明示指示を
    無視するのは、本調査が問題視している silent degradation そのものだから。
    """
    rejected: list[tuple[str, str]] = []

    if override:
        path = Path(override)
        ok, reason = is_usable(path, limit=MAX_PARENT_ROOT_LEN)
        if not ok:
            raise RuntimeError(
                f"LIVECAP_NONASCII_ROOT={override!r} が使えない: {reason}。"
                "明示指定を黙って無視すると測定対象が変わってしまうため中断する。"
            )
        return path, "LIVECAP_NONASCII_ROOT", rejected

    for candidate in candidates(models_root, repo_root):
        if candidate.path is None:
            continue
        ok, reason = is_usable(candidate.path, limit=MAX_PARENT_ROOT_LEN)
        if ok:
            return candidate.path, candidate.label, rejected
        rejected.append((candidate.label, reason))

    raise RuntimeError(
        "ASCII かつ書き込み可能な base root が見つからない。"
        "LIVECAP_NONASCII_ROOT で明示指定すること。候補と理由: "
        + " / ".join(f"{label}: {reason}" for label, reason in rejected)
    )


def write_session_marker(session: Path) -> dict:
    """session root に**所有権マーカー**を書く。

    reaper はこのマーカーがあるディレクトリしか削除しない。「ハーネスが作った
    ものだけを消す」という保証をファイル側に持たせるための仕組みである。
    """
    payload = {
        "magic": SESSION_MAGIC,
        "schema": SESSION_MARKER_SCHEMA,
        "session_id": session.name,
        "pid": os.getpid(),
        "created_at": time.time(),
    }
    (session / SESSION_MARKER_NAME).write_text(
        json.dumps(payload, ensure_ascii=True), encoding="utf-8"
    )
    return payload


def read_session_marker(session: Path) -> dict | None:
    """有効な所有権マーカーを返す。無効 / 不在なら ``None``。"""
    try:
        payload = json.loads((session / SESSION_MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("magic") != SESSION_MAGIC:
        return None
    if payload.get("schema") != SESSION_MARKER_SCHEMA:
        return None
    return payload


def create_session_root(parent: Path) -> Path:
    """共有親の下に **この run 専用** のディレクトリを作って返す。

    PID + UUID で名前を作るので、同時に走る 2 つの run が同じパスを掴むことはない。
    後始末はこの session root だけを対象にすればよく、実行中の他 run を壊さない。
    所有権マーカーを書くので、reaper が他人のディレクトリを消すこともない。
    """
    session = parent / f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if len(str(session)) > MAX_SESSION_ROOT_LEN:
        # 親の述語で予約しているので通常ここには来ない。来たら黙って進まない。
        raise RuntimeError(
            f"session root が長すぎる ({len(str(session))} > {MAX_SESSION_ROOT_LEN}): "
            f"{session}。より短い LIVECAP_NONASCII_ROOT を指定すること。"
        )
    session.mkdir(parents=True, exist_ok=False)
    write_session_marker(session)
    return session


def reap_stale_sessions(
    parent: Path, *, max_age_hours: float = STALE_SESSION_HOURS
) -> list[str]:
    """異常終了した run が残した session root を best-effort で掃除する。

    残骸を放置すると共有親が際限なく育つうえ、古い hardlink が残っていると
    ``materialize_file()`` が ``existing`` として**古いモデルを再利用**してしまい、
    証拠の再現性が損なわれる。

    **削除するのは (1) 厳密な名前形式 と (2) 有効な所有権マーカー の両方を満たす
    ディレクトリだけ。** ``LIVECAP_NONASCII_ROOT`` に利用者が既存ディレクトリを
    指定している可能性があるため、名前と mtime だけを条件に再帰削除してはならない。

    **生存中の run は消さない** — PID 生存判定は PID 再利用があるので使わず、
    マーカーの ``created_at`` による経過時間だけで判断する
    (実装側 reaper と同じ方針、棚卸し表 §6.6)。失敗しても例外にしない。
    """
    reaped: list[str] = []
    cutoff = time.time() - max_age_hours * 3600.0
    try:
        children = list(parent.glob("run-*"))
    except OSError:
        return reaped

    for child in children:
        if not child.is_dir():
            continue
        # (1) 厳密な名前形式。glob の "run-*" だけでは run-backup 等が引っかかる。
        if not _SESSION_NAME_RE.match(child.name):
            continue
        # (2) 所有権マーカー。無ければハーネスの生成物ではないので絶対に触らない。
        marker = read_session_marker(child)
        if marker is None:
            continue
        # (3) 経過時間はマーカーの created_at で見る (我々が書いた値なので信頼できる)。
        created_at = marker.get("created_at")
        if not isinstance(created_at, (int, float)) or created_at >= cutoff:
            continue
        shutil.rmtree(child, ignore_errors=True)
        if not child.exists():
            reaped.append(child.name)
    return reaped


__all__ = [
    "MAX_PROBE_SUFFIX_LEN",
    "MAX_PARENT_ROOT_LEN",
    "MAX_SESSION_ROOT_LEN",
    "SESSION_MAGIC",
    "SESSION_MARKER_NAME",
    "SESSION_SUFFIX_LEN",
    "STALE_SESSION_HOURS",
    "RootCandidate",
    "candidates",
    "create_session_root",
    "is_usable",
    "read_session_marker",
    "reap_stale_sessions",
    "resolve_base_root",
    "write_session_marker",
]
