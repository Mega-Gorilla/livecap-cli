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
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: base root に許す最大長。variant / probe / digest を足しても MAX_PATH に
#: 余裕があるようにする (実装側の staging root 述語と同じ考え方)。
MAX_ROOT_LEN = 120

_LEAF = "livecap-nonascii-probe"


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


def is_usable(path: Path) -> tuple[bool, str]:
    """ASCII かつ十分短く、実際に書き込めるか。

    Windows の ACL 検査は当てにならないので**書き込みプローブ**で判定する。
    """
    text = str(path)
    if not text.isascii():
        return False, "非 ASCII"
    if len(text) > MAX_ROOT_LEN:
        return False, f"長すぎる ({len(text)} > {MAX_ROOT_LEN})"
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
    """(base root, 採用した候補ラベル, 落ちた候補と理由) を返す。

    ``override`` (``LIVECAP_NONASCII_ROOT``) が指定されていて述語を満たさない場合は
    **黙って fallback せず** ``RuntimeError`` を投げる — 運用者の明示指示を
    無視するのは、本調査が問題視している silent degradation そのものだから。
    """
    rejected: list[tuple[str, str]] = []

    if override:
        path = Path(override)
        ok, reason = is_usable(path)
        if not ok:
            raise RuntimeError(
                f"LIVECAP_NONASCII_ROOT={override!r} が使えない: {reason}。"
                "明示指定を黙って無視すると測定対象が変わってしまうため中断する。"
            )
        return path, "LIVECAP_NONASCII_ROOT", rejected

    for candidate in candidates(models_root, repo_root):
        if candidate.path is None:
            continue
        ok, reason = is_usable(candidate.path)
        if ok:
            return candidate.path, candidate.label, rejected
        rejected.append((candidate.label, reason))

    raise RuntimeError(
        "ASCII かつ書き込み可能な base root が見つからない。"
        "LIVECAP_NONASCII_ROOT で明示指定すること。候補と理由: "
        + " / ".join(f"{label}: {reason}" for label, reason in rejected)
    )


__all__ = ["MAX_ROOT_LEN", "RootCandidate", "candidates", "is_usable", "resolve_base_root"]
