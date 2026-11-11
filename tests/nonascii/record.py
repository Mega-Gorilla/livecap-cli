"""結果レコードと測定メタデータ (Issue #378)。

schema_version 1。``benchmark_results/nonascii/<date>/results.json`` に
このスキーマで書き出し、``report.py`` が registry と突き合わせて
棚卸し表 (docs) をレンダリングする。
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class ProbeSkipped(Exception):
    """依存未導入などで測定不能。**バグの証拠ではない**。

    probe 実装はこれを raise することで ``verdict="skipped"`` を要求できる。
    ``worker`` との循環 import を避けるためここに置く。
    """


class Verdict(str, Enum):
    """境界 1 つ × variant 1 つの判定。

    ``fail_silent`` が epic #380 の中核関心事 — 「壊れているのに兆候ゼロ」。
    """

    PASS = "pass"
    FAIL_LOUD = "fail_loud"
    FAIL_SILENT = "fail_silent"
    SKIPPED = "skipped"
    ERROR_HARNESS = "error_harness"


class EvidenceKind(str, Enum):
    RUNTIME = "runtime"
    SOURCE_CHECK = "source_check"
    NOT_APPLICABLE = "not_applicable"


class Tier(str, Enum):
    CHEAP = "cheap"
    REAL_MODEL = "real_model"
    HEAVY = "heavy"
    #: CUDA が要るがモデルは要らない (#422 の Jiterator kernel cache)。
    #: real_model / heavy と分けるのは、**実モデルの所在も NeMo も要求しない**ため —
    #: それらの tier に混ぜると source が見つからず黙って skip する。
    GPU = "gpu"
    NETWORK = "network"
    NONE = "none"


@dataclass(frozen=True)
class ProbeContext:
    """worker がプローブ実装に渡す実行文脈。"""

    probe_id: str
    variant: str
    root: Path
    is_control: bool
    payload: dict = field(default_factory=dict)
    # 完了したステージ名。遅延失敗 (ロードは通ったが後段で落ちる) の判定に使う。
    # frozen dataclass だが list 自体は可変なので probe から append できる。
    stages: list = field(default_factory=list)

    def stage(self, name: str) -> None:
        """ステージ完了を記録する。probe は節目ごとに必ず呼ぶこと。"""
        self.stages.append(name)


@dataclass
class ProbeResult:
    boundary_id: str
    probe_id: str
    variant: str
    apply_to: str
    tier: str
    evidence_kind: str
    verdict: str
    control_verdict: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    exception_type: str | None = None
    exception_message: str | None = None
    error_mentions_path: bool = False
    observation: Any = None
    control_observation: Any = None
    silent_criteria_hit: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    notes: str = ""
    # worker が JSON を返せなかったときだけ埋まる。**制限付き環境で「何も分からない」
    # 状態を避けるための診断**であり、正常時は None のまま (差分比較を汚さない)。
    worker_diagnostics: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _active_code_page() -> int | None:
    """Windows ANSI code page。narrow path 変換で使われるのはこれ。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetACP())
    except Exception:
        return None


def _oem_code_page() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetOEMCP())
    except Exception:
        return None


def _long_paths_enabled() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "sherpa-onnx",
        "onnxruntime",
        "soundfile",
        "librosa",
        "torch",
        "safetensors",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "whisper-s2t",
        "ffmpeg-python",
        "sentencepiece",
        "nemo-toolkit",
        "qwen-asr",
        "appdirs",
        "numpy",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "(not installed)"
    return out


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str, cwd: Path | None = None) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd or _REPO_ROOT,
        )
    except Exception:
        return None
    return result.stdout if result.returncode == 0 else None


def _git_commit(cwd: Path | None = None) -> str:
    out = _run_git("rev-parse", "HEAD", cwd=cwd)
    return (out or "").strip() or "(unknown)"


def _git_dirty(cwd: Path | None = None) -> bool | None:
    """記録した commit を checkout しても**この結果を再現できない**か。

    ハーネスを未コミットの working tree で実行すると、``git_commit`` は 1 つ前の
    commit を指したまま、実際には手元の変更で測ることになる。証拠だけを見ても
    実行コードを特定できず、**再現不能な evidence** が commit される (#377 で実際に
    起きた: 強化後の probe が出した ``token_count`` を、その機能が入る前の commit
    の測定結果として記録していた)。

    ``None`` は「git が使えず判定不能」。**判定不能と clean を混同しない。**

    ``--untracked-files=all`` が要る。素の ``--porcelain`` は
    ``status.showUntrackedFiles`` 設定を尊重するため、``no`` を設定した環境では
    **未追跡の probe / helper / test が実行に使われても出力が空になり、
    ``git_dirty=False`` と記録される**。防ぎたいのはまさに「記録した commit に
    存在しないコードで測ったのに clean と表示する」ことなので、未追跡は必ず拾う。

    Note:
        evidence ファイル自体の変更も dirty として拾う (probe コードの再現性には
        影響しない)。**保守的に倒している** — 偽の dirty は目に見えて直せるが、
        偽の clean は本 issue で直したバグそのものだから。運用は「コードを commit
        -> clean な tree で測定 -> 証拠を commit」の順で回す。
    """
    out = _run_git("status", "--porcelain", "--untracked-files=all", cwd=cwd)
    if out is None:
        return None
    return bool(out.strip())


@dataclass
class RunMetadata:
    """§0 測定メタデータ。どのホストで測ったかを必ず残す。

    CI runner は ACP=cp1252、日本語開発機は cp932 で、検出できる失敗の
    部分集合が異なる。どちらも単独では権威ではない。
    """

    run_id: str
    measured_at: str
    git_commit: str = field(default_factory=_git_commit)
    #: True なら ``git_commit`` を checkout してもこの結果は再現できない。
    git_dirty: bool | None = field(default_factory=_git_dirty)
    os: str = field(default_factory=platform.platform)
    machine: str = field(default_factory=platform.machine)
    python: str = field(default_factory=lambda: platform.python_version())
    fs_encoding: str = field(default_factory=sys.getfilesystemencoding)
    active_code_page: int | None = field(default_factory=_active_code_page)
    oem_code_page: int | None = field(default_factory=_oem_code_page)
    console_encoding: str | None = field(
        default_factory=lambda: getattr(sys.stdout, "encoding", None)
    )
    preferred_encoding: str = field(
        default_factory=lambda: __import__("locale").getpreferredencoding(False)
    )
    python_utf8_mode: bool = field(default_factory=lambda: bool(sys.flags.utf8_mode))
    long_paths_enabled: bool | None = field(default_factory=_long_paths_enabled)
    eight_dot_three_state: str = "(not queried)"
    system_temp: str = field(default_factory=tempfile.gettempdir)
    system_temp_is_ascii: bool = field(
        default_factory=lambda: tempfile.gettempdir().isascii()
    )
    username_is_ascii: bool = field(
        default_factory=lambda: os.environ.get("USERNAME", os.environ.get("USER", "")).isascii()
    )
    nonascii_root: str = ""
    # どの候補が採用され、どれがなぜ落ちたか。base root が実環境と違う場所に
    # 落ち着いた run を、後から見分けられるようにする。
    root_label: str = ""
    # 共有される親 root。nonascii_root はその下の run 固有 session root。
    root_parent: str = ""
    rejected_roots: dict = field(default_factory=dict)
    # 異常終了した過去 run から回収した session root。
    reaped_stale_sessions: list = field(default_factory=list)
    root_volume: str = ""
    materialization: str = "n/a"
    packages: dict[str, str] = field(default_factory=_package_versions)
    tiers_enabled: list[str] = field(default_factory=list)
    variants_supported: list[str] = field(default_factory=list)
    variants_skipped: dict[str, str] = field(default_factory=dict)
    normalization_preserved: bool | None = None
    leftover_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


#: 証拠 JSON の中でホームディレクトリを置き換える文字列。
HOME_PLACEHOLDER = "~"

#: ホームの直後に来てよい文字 (区切り・空白・引用符・括弧などの区切り記号) か末尾。
#: これ以外が続くなら、より長い別の名前 (``C:\\Users\\abc`` に対する ``C:\\Users\\ab``) の一部。
_HOME_END = r"""(?=[\\/\s'"`)\]},;:<>|]|$)"""


def _home_spellings(home: str) -> set[str]:
    """``home`` が文字列の中に現れうる表記。

    - そのまま / ``/`` 区切り
    - ``repr()`` の表記: ``\\`` が 2 重になる。``str(OSError(...))`` のファイル名はこれ
    - ``ascii()`` の表記 (:func:`safe_path_repr`): 2 重の ``\\`` に加えて、非 ASCII は ``\\uXXXX``
    """
    spellings = set()
    for spelling in (home, home.replace("\\", "/")):
        spellings |= {spelling, repr(spelling)[1:-1], ascii(spelling)[1:-1]}
    return spellings


def redact_home(value: Any, home: str | None = None) -> Any:
    """``value`` (str / dict / list) に含まれるホームディレクトリを ``~`` に置き換える。

    証拠 JSON は commit して公開するので、ユーザー名を含むパス
    (``system_temp`` の ``C:\\Users\\<name>\\AppData\\Local\\Temp``、例外メッセージの
    ``[Errno 2] ...: 'C:\\\\Users\\\\<name>\\\\...'`` など) を残さない。
    ユーザー名が非 ASCII かどうかは ``username_is_ascii`` / ``system_temp_is_ascii`` が残す。

    表記は :func:`_home_spellings` のすべてを見て、大文字小文字は区別しない。直後が
    区切り・空白・引用符などか末尾のときだけ置き換える (``C:\\Users\\ab`` が
    ``C:\\Users\\abc`` の先頭に当たらない)。dict は key も置き換える。
    """
    home = os.path.expanduser("~") if home is None else home
    if not home or home == HOME_PLACEHOLDER or Path(home).parent == Path(home):
        return value
    variants = sorted(_home_spellings(home), key=len, reverse=True)
    pattern = re.compile("(?:" + "|".join(re.escape(v) for v in variants) + ")" + _HOME_END, re.IGNORECASE)

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub(lambda _m: HOME_PLACEHOLDER, item)
        if isinstance(item, dict):
            return {walk(k): walk(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(v) for v in item]
        return item

    return walk(value)


def write_results(path: Path, run: RunMetadata, results: list[ProbeResult]) -> None:
    """証拠 JSON を書き出す。

    ``ensure_ascii=True``: 非 ASCII パスを含む JSON をどんなロケールの
    コンソール/エディタでも安全に扱えるようにするため (本ハーネス自体が
    非 ASCII 由来のクラッシュを起こしては本末転倒)。

    ホームディレクトリは :func:`redact_home` で ``~`` に伏せてから書く。
    """
    payload = redact_home(
        {
            "schema_version": SCHEMA_VERSION,
            "run": run.to_dict(),
            "results": [r.to_dict() for r in results],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def read_results(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version が一致しない: {payload.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return payload["run"], payload["results"]


def safe_path_repr(p: Any) -> str:
    """非 ASCII パスを cp932 コンソールへ出しても落ちない表現。

    日本語 Windows では stderr をファイル/パイプにリダイレクトすると
    locale (cp932) + errors='strict' になり、素の f-string ログが
    ``UnicodeEncodeError`` を投げる。
    """
    return ascii(os.fspath(p) if hasattr(p, "__fspath__") else str(p))


__all__ = [
    "HOME_PLACEHOLDER",
    "SCHEMA_VERSION",
    "EvidenceKind",
    "ProbeContext",
    "ProbeSkipped",
    "ProbeResult",
    "RunMetadata",
    "Tier",
    "Verdict",
    "read_results",
    "redact_home",
    "safe_path_repr",
    "write_results",
]
