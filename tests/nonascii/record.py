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


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip() or "(unknown)"
    except Exception:
        return "(unknown)"


@dataclass
class RunMetadata:
    """§0 測定メタデータ。どのホストで測ったかを必ず残す。

    CI runner は ACP=cp1252、日本語開発機は cp932 で、検出できる失敗の
    部分集合が異なる。どちらも単独では権威ではない。
    """

    run_id: str
    measured_at: str
    git_commit: str = field(default_factory=_git_commit)
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
    rejected_roots: dict = field(default_factory=dict)
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


def write_results(path: Path, run: RunMetadata, results: list[ProbeResult]) -> None:
    """証拠 JSON を書き出す。

    ``ensure_ascii=True``: 非 ASCII パスを含む JSON をどんなロケールの
    コンソール/エディタでも安全に扱えるようにするため (本ハーネス自体が
    非 ASCII 由来のクラッシュを起こしては本末転倒)。
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run": run.to_dict(),
        "results": [r.to_dict() for r in results],
    }
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
    "SCHEMA_VERSION",
    "EvidenceKind",
    "ProbeContext",
    "ProbeSkipped",
    "ProbeResult",
    "RunMetadata",
    "Tier",
    "Verdict",
    "read_results",
    "safe_path_repr",
    "write_results",
]
