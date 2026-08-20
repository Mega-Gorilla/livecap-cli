"""非 ASCII path variant の語彙と、ホスト FS が受理できるかの判定 (Issue #378)。

**各 variant は「日本語を混ぜたもの」ではなく、別々の失敗機構を切り分ける。**
ある行が ``cjk_kana`` を通るのに ``outside_acp`` で落ちる場合と、
``space_paren`` で落ちる場合とでは、必要な修正が根本的に異なる。
"""

from __future__ import annotations

import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

CONTROL = "control"


@dataclass(frozen=True)
class PathVariant:
    id: str
    segment: str
    mechanism: str
    default_enabled: bool = True

    @property
    def is_control(self) -> bool:
        return self.id == CONTROL


VARIANTS: tuple[PathVariant, ...] = (
    PathVariant(
        CONTROL,
        "ascii_control",
        "差分の基準。これが落ちたら結果は error_harness であってバグの証拠ではない。",
    ),
    PathVariant(
        "cjk_kana",
        "ユーザー",
        "実世界ケース (JP ユーザー名)。cp932 の内側 / cp1252 の外側 → "
        "「UTF-8 で書いたものを Win32 narrow API が ACP として解釈」を切り分ける。"
        "8.3 短縮名が生成されない実測済みケースでもある。",
    ),
    PathVariant(
        "outside_acp",
        "한국어Ω",
        "cp932 と cp1252 の両方の外側 → JP 開発機でも en-US CI でも "
        "「ACP で表現不能」モードを強制し、両ホストの結果を比較可能にする。",
    ),
    PathVariant(
        "space_paren",
        "テスト フォルダ (1)",
        "空白 + 括弧 = 別の failure family (subprocess / ffmpeg-python の argv "
        "quoting であってエンコーディングではない)。ASCII staging では直らない"
        "バグを捕まえる。",
    ),
    PathVariant(
        "nfd",
        "がんだん",  # NFD: か+゙ / た+゙ (合成前)
        "NFD 分解形。正規化を仮定するライブラリはファイルを見失う。"
        "ascii_safe_path() の契約が NFC 入力を仮定してはいけない根拠。",
    ),
    PathVariant(
        "emoji_astral",
        "音楽\U0001F3B5",
        "astral 面 → UTF-16 でサロゲートペア / UTF-8 で 4 バイト。"
        "BMP(UCS-2) 前提と len() ベースのバッファ計算を突く。",
        default_enabled=False,
    ),
    PathVariant(
        "long_mixed",
        ("ながいなまえ テスト\U0001F3B5" * 12)[:200],
        "MAX_PATH との相互作用 (別軸・別 issue)。",
        default_enabled=False,
    ),
)

VARIANTS_BY_ID = {v.id: v for v in VARIANTS}
DEFAULT_VARIANT_IDS = tuple(v.id for v in VARIANTS if v.default_enabled)


def variant(variant_id: str) -> PathVariant:
    try:
        return VARIANTS_BY_ID[variant_id]
    except KeyError:  # pragma: no cover - 呼び出し側のバグ
        raise KeyError(f"未知の variant: {variant_id!r}") from None


def make_variant_root(base: Path, variant_id: str, probe_id: str) -> Path:
    """``<base>/<variant>/<probe>/`` を作って返す。

    probe_id はファイル名に使えない文字を含み得ないが、念のため sanitise する。
    """
    leaf = "".join(c if (c.isalnum() or c in "._-") else "_" for c in probe_id)
    root = base / variant(variant_id).segment / leaf
    root.mkdir(parents=True, exist_ok=True)
    return root


def _roundtrip_ok(parent: Path, segment: str) -> tuple[bool, str]:
    """作成した名前が**同一コードポイント列**で読み戻せるかを検査する。

    macOS APFS や一部のネットワーク共有は NFC/NFD 正規化を行うため、
    「非 ASCII を試したつもりが別の文字列を試していた」を防ぐ。
    """
    try:
        listed = os.listdir(parent)
    except OSError as exc:
        return False, f"listdir 失敗: {exc}"
    if segment in listed:
        return True, ""
    normalized = {unicodedata.normalize("NFC", n): n for n in listed}
    if unicodedata.normalize("NFC", segment) in normalized:
        return False, (
            "FS がファイル名を正規化した "
            f"(要求={ascii(segment)} / 実際={ascii(normalized[unicodedata.normalize('NFC', segment)])})"
        )
    return False, f"作成した名前が listdir に現れない (要求={ascii(segment)})"


def probe_variant_support(base: Path, variant_id: str) -> tuple[bool, str]:
    """この FS が variant を素通しできるか。

    (1) mkdir できる → (2) 同一コードポイントで読み戻せる → (3) 中に
    ファイルを書いて読み戻せる、の 3 段階。落ちた variant は
    ``verdict="skipped"`` + 理由として記録され、「非 ASCII が通った」と
    「非 ASCII を試していない」が混同されないようにする。
    """
    v = variant(variant_id)
    base.mkdir(parents=True, exist_ok=True)
    target = base / v.segment
    try:
        target.mkdir(parents=True, exist_ok=True)
    except (OSError, UnicodeEncodeError, ValueError) as exc:
        return False, f"filesystem が variant を拒否: {type(exc).__name__}: {exc}"

    ok, reason = _roundtrip_ok(base, v.segment)
    if not ok:
        return False, reason

    probe_file = target / "probe.txt"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        if probe_file.read_text(encoding="utf-8") != "ok":
            return False, "書き込んだ内容が読み戻せない"
    except (OSError, UnicodeEncodeError, ValueError) as exc:
        return False, f"variant ディレクトリ内にファイルを作れない: {type(exc).__name__}: {exc}"
    finally:
        try:
            probe_file.unlink(missing_ok=True)
        except OSError:
            pass
    return True, ""


def supported_variants(
    base: Path, variant_ids: tuple[str, ...] = DEFAULT_VARIANT_IDS
) -> tuple[list[str], dict[str, str]]:
    """(使える variant, 使えない variant -> 理由)。"""
    ok: list[str] = []
    skipped: dict[str, str] = {}
    for vid in variant_ids:
        supported, reason = probe_variant_support(base, vid)
        if supported:
            ok.append(vid)
        else:
            skipped[vid] = reason
    return ok, skipped


def short_path_name(path: Path) -> str | None:
    """Windows 8.3 短縮名。別名が無ければ入力がそのまま返る。

    8.3 を ASCII staging の代替にできない理由を、散文ではなく機械記録として
    残すためのプローブ。``ユーザー`` は 8.3 に収まるので別名が生成されない。
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathNameW.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = GetShortPathNameW(str(path), buf, 1024)
        if n == 0:
            return None
        return buf.value
    except Exception:
        return None


def eight_dot_three_state(volume: str) -> str:
    """``fsutil 8dot3name query <vol>`` の結果 (read-only)。"""
    if sys.platform != "win32":
        return "n/a (not windows)"
    import subprocess

    try:
        proc = subprocess.run(
            ["fsutil", "8dot3name", "query", volume],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (proc.stdout or proc.stderr or "").strip().replace("\n", " / ")
        return out or f"(exit {proc.returncode})"
    except Exception as exc:  # pragma: no cover
        return f"(query failed: {type(exc).__name__})"


__all__ = [
    "CONTROL",
    "DEFAULT_VARIANT_IDS",
    "VARIANTS",
    "PathVariant",
    "eight_dot_three_state",
    "make_variant_root",
    "probe_variant_support",
    "short_path_name",
    "supported_variants",
    "variant",
]
