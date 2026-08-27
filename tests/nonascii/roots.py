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
旧 ``unicode_safe_download_directory`` の「共有ディレクトリを rmtree する」欠陥
([#386](https://github.com/Mega-Gorilla/livecap-cli/issues/386)) と
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


#: 使用中ロック。**保持している間はその session root を誰にも消させない。**
#: 経過時間だけで「もう終わった run」と判断すると、heavy / real_model tier や
#: 低速環境で 6 時間を超えて**実行中**の session を消してしまう。生存判定は
#: 時間ではなくロックで行う (棚卸し表 §6.7 の in-use lease と同じ考え方)。
SESSION_LOCK_NAME = ".livecap-nonascii-session.lock"

#: このプロセスが保持しているロックのハンドル。session root path -> file object。
_HELD_LOCKS: dict[str, object] = {}


def _lock_exclusive(fileno: int) -> bool:
    """ファイル記述子に**非ブロッキングの排他ロック**を掛ける。取れたら True。

    Windows は ``msvcrt.locking``、POSIX は ``fcntl.flock`` を使う。
    「ロックファイルを削除できるか」で判定する手もあるが、**判定自体が破壊的**に
    なり 2 回目の答えが変わってしまうので採らない。
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fileno, 0, os.SEEK_SET)
            msvcrt.locking(fileno, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fileno, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError, ValueError):
        return False
    return True


def _unlock(fileno: int) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fileno, 0, os.SEEK_SET)
            msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fileno, fcntl.LOCK_UN)
    except (ImportError, OSError, ValueError):
        pass


def _acquire_session_lock(session: Path) -> None:
    """session root の使用中ロックを取得し、プロセス寿命の間保持する。

    ロックを握っている間、他 run の reaper はこの session を「実行中」と判定して
    絶対に削除しない。プロセスが異常終了すれば OS がハンドルを閉じるので、
    ロックは自然に解放され、次の run が残骸として回収できる。
    """
    handle = open(session / SESSION_LOCK_NAME, "w+", encoding="utf-8")
    handle.write(str(os.getpid()))
    handle.flush()
    _lock_exclusive(handle.fileno())
    _HELD_LOCKS[str(session)] = handle


def release_session_root(session: Path) -> None:
    """使用中ロックを手放す。**session root を削除する前に必ず呼ぶこと。**

    Windows ではロックを握ったままだと自分自身の ``rmtree`` も失敗する。
    """
    handle = _HELD_LOCKS.pop(str(session), None)
    if handle is None:
        return
    try:
        _unlock(handle.fileno())
    except Exception:  # noqa: BLE001 - 後始末で落とさない
        pass
    try:
        handle.close()
    except OSError:
        pass


def is_session_in_use(session: Path) -> bool:
    """その session root を保持しているプロセスがまだ生きているか。

    **PID 生存判定は使わない** — PID 再利用があるので不健全。代わりに
    「排他ロックを掴めるか」で判断する。掴めない = 所有プロセスが生存中。
    **判定は非破壊**なので、何度呼んでも同じ答えになる。

    ロックファイルが無い場合は判断材料が無いので**安全側に倒して True** を返す
    (この機構が入る前に作られた session や、作成途中のものを消さないため)。
    """
    if str(session) in _HELD_LOCKS:
        return True

    lock_path = session / SESSION_LOCK_NAME
    if not lock_path.exists():
        return True

    try:
        with open(lock_path, "r+", encoding="utf-8") as handle:
            if not _lock_exclusive(handle.fileno()):
                return True
            _unlock(handle.fileno())
    except OSError:
        # 開けない = 所有プロセスが掴んでいる可能性が高い。安全側に倒す。
        return True
    return False


def create_session_root(parent: Path) -> Path:
    """共有親の下に **この run 専用** のディレクトリを作って返す。

    PID + UUID で名前を作るので、同時に走る 2 つの run が同じパスを掴むことはない。
    後始末はこの session root だけを対象にすればよく、実行中の他 run を壊さない。
    所有権マーカーを書くので、reaper が他人のディレクトリを消すこともない。
    さらに**使用中ロック**を取得するので、実行が長引いても他 run に消されない。

    削除する前に ``release_session_root()`` を呼ぶこと。
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
    _acquire_session_lock(session)
    return session


def reap_stale_sessions(
    parent: Path, *, max_age_hours: float = STALE_SESSION_HOURS
) -> list[str]:
    """異常終了した run が残した session root を best-effort で掃除する。

    目的は**ディスクの衛生**である。session root は UUID で分離されているので、
    古い残骸が新しい run に混入することはない (``materialize_file()`` が参照する
    のは常に自分の session root 配下なので、古い hardlink を ``existing`` として
    再利用することもない)。したがって回収は「あれば嬉しい」程度の位置づけであり、
    **少しでも危ないなら消さない**方に倒す。

    削除するのは以下を**すべて**満たすものだけ:

    1. 厳密な session 名形式 — glob の ``run-*`` だけでは ``run-backup`` も拾う
    2. 有効な所有権マーカー — ``LIVECAP_NONASCII_ROOT`` には利用者の既存
       ディレクトリが指定され得るので、我々の生成物であることを確認する
    3. **使用中ロックを掴める** = 所有プロセスが終了済み
    4. マーカーの ``created_at`` が閾値より古い (保守的な追加条件)

    3 が生存判定の本体である。経過時間だけで判断すると、heavy / real_model tier や
    低速環境で **6 時間を超えて実行中**の session を消してしまう。失敗しても
    例外にしない。
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
        # (1) 厳密な名前形式。
        if not _SESSION_NAME_RE.match(child.name):
            continue
        # (2) 所有権マーカー。無ければハーネスの生成物ではないので絶対に触らない。
        marker = read_session_marker(child)
        if marker is None:
            continue
        # (3) 経過時間 (保守的な追加条件)。
        created_at = marker.get("created_at")
        if not isinstance(created_at, (int, float)) or created_at >= cutoff:
            continue
        # (4) **生存判定**。所有プロセスが生きているなら絶対に消さない。
        if is_session_in_use(child):
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
    "SESSION_LOCK_NAME",
    "SESSION_MARKER_NAME",
    "SESSION_SUFFIX_LEN",
    "STALE_SESSION_HOURS",
    "RootCandidate",
    "candidates",
    "create_session_root",
    "is_session_in_use",
    "is_usable",
    "read_session_marker",
    "reap_stale_sessions",
    "release_session_root",
    "resolve_base_root",
    "write_session_marker",
]
