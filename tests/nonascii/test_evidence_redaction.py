"""証拠 JSON にホームディレクトリ (ユーザー名) を残さないこと。

``benchmark_results/nonascii/<date>/results.json`` は commit して公開する。以前は
``system_temp`` に ``C:\\Users\\<name>\\AppData\\Local\\Temp`` をそのまま書き、測定者の
ユーザー名が公開されていた。ユーザー名が非 ASCII かどうかは ``username_is_ascii`` /
``system_temp_is_ascii`` が残すので、パスそのものは要らない。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from .record import (
    HOME_PLACEHOLDER,
    ProbeResult,
    RunMetadata,
    read_results,
    redact_home,
    safe_path_repr,
    write_results,
)

pytestmark = pytest.mark.nonascii_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "benchmark_results" / "nonascii"
# Windows / macOS / Linux のホームの形。``USER`` / ``user`` / ``runner`` は伏せ字・CI の既定なので許す
HOME_LIKE = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/Users/|/home/)(?!(?:USER|user|runner)(?![^\\/]))[^\\/\s\"']+",
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


class TestRedactHome:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"C:\Users\Taro\AppData\Local\Temp", r"~\AppData\Local\Temp"),
            ("C:/Users/Taro/AppData/Local/Temp", "~/AppData/Local/Temp"),
            # 大文字小文字だけが違う (Windows は区別しない)
            (r"c:\users\taro\AppData", r"~\AppData"),
            (r"C:\Users\Taro", "~"),
            # 文中に現れる場合も
            (r"cached at: C:\Users\Taro\x.bin (ok)", r"cached at: ~\x.bin (ok)"),
        ],
    )
    def test_windows_home(self, text, expected):
        assert redact_home(text, home=r"C:\Users\Taro") == expected

    def test_posix_home(self):
        assert redact_home("/home/taro/.cache/x", home="/home/taro") == "~/.cache/x"

    def test_a_longer_name_is_not_a_match(self):
        """``C:\\Users\\Taro`` は ``C:\\Users\\Taro2`` の先頭に当たらない。"""
        text = r"C:\Users\Taro2\AppData"
        assert redact_home(text, home=r"C:\Users\Taro") == text

    def test_nested_values_and_keys(self):
        home = r"C:\Users\Taro"
        payload = {
            "run": {"system_temp": rf"{home}\AppData\Local\Temp", "rejected_roots": {rf"{home}\x": "no space"}},
            "results": [{"detail": [rf"{home}\y", 3, None, True]}],
        }
        assert redact_home(payload, home=home) == {
            "run": {"system_temp": r"~\AppData\Local\Temp", "rejected_roots": {r"~\x": "no space"}},
            "results": [{"detail": [r"~\y", 3, None, True]}],
        }

    @pytest.mark.parametrize("home", ["", "/", "C:\\", HOME_PLACEHOLDER])
    def test_degenerate_home_changes_nothing(self, home):
        """ホームが取れない / root のときに、全部のパスを ``~`` にしない。"""
        text = r"C:\livecap-nonascii-probe\run-1"
        assert redact_home(text, home=home) == text


class TestRedactEscapedSpellings:
    """例外メッセージや repr に入ったホームも伏せる (PR #486 レビュー F1)。

    証拠には ``worker.py`` の ``str(exc)`` (``exception_message``) が入る。``OSError`` の
    ファイル名は repr で ``\\`` が 2 重になり、:func:`safe_path_repr` (``ascii()``) は非 ASCII も
    ``\\uXXXX`` にする。期待値は「ホームを ``~`` にして同じ表現を作ったもの」で書く。
    名前はすべて架空 (失敗時にも実在のユーザー名をログへ出さない)。
    """

    @pytest.mark.parametrize("home", [r"C:\Users\Taro", "C:\\Users\\太郎", "/home/taro"])
    def test_os_error_message(self, home):
        sep = "/" if home.startswith("/") else "\\"

        message = str(FileNotFoundError(2, "missing", home + sep + "missing.bin"))

        assert redact_home(message, home=home) == str(FileNotFoundError(2, "missing", "~" + sep + "missing.bin"))

    @pytest.mark.parametrize("home", [r"C:\Users\Taro", "C:\\Users\\太郎", "/home/taro"])
    def test_safe_path_repr(self, home):
        sep = "/" if home.startswith("/") else "\\"

        text = f"open failed: {safe_path_repr(home + sep + 'x.bin')}"

        assert redact_home(text, home=home) == f"open failed: {safe_path_repr('~' + sep + 'x.bin')}"

    @pytest.mark.parametrize(
        ("home", "text", "expected"),
        [
            ("/home/taro", "[Errno 2] missing: '/home/taro'", "[Errno 2] missing: '~'"),
            (r"C:\Users\Taro", r"""cwd="C:\Users\Taro" (home)""", 'cwd="~" (home)'),
            (r"C:\Users\Taro", r"[Errno 13] denied: 'C:\\Users\\Taro'", "[Errno 13] denied: '~'"),
        ],
    )
    def test_quoted_home_alone(self, home, text, expected):
        """ホームの直後が引用符・空白でも置き換える。"""
        assert redact_home(text, home=home) == expected

    @pytest.mark.parametrize(
        "text",
        [
            str(FileNotFoundError(2, "missing", r"C:\Users\Taro2\x.bin")),
            safe_path_repr(r"C:\Users\Taro2\x.bin"),
            "'/home/taro2'",
        ],
    )
    def test_a_longer_name_is_not_a_match_in_escaped_forms(self, text):
        home = "/home/taro" if text.startswith("'/home") else r"C:\Users\Taro"
        assert redact_home(text, home=home) == text


def test_write_results_round_trip_redacts_every_spelling(tmp_path: Path, monkeypatch):
    """``write_results`` → ``read_results`` で、run の値も例外メッセージも伏せてある。

    ホームは環境変数で架空の値にする (``os.path.expanduser`` は Windows で ``USERPROFILE``、
    POSIX で ``HOME`` を読む)。
    """
    home = r"C:\Users\Taro" if os.name == "nt" else "/home/taro"
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.setenv("HOME", home)
    assert os.path.expanduser("~") == home  # 前提
    missing = os.path.join(home, "missing.bin")
    run = RunMetadata(
        run_id="run-test",
        measured_at="2026-11-10",
        system_temp=os.path.join(home, "AppData", "Local", "Temp"),
        leftover_paths=[os.path.join(home, "left")],
    )
    result = ProbeResult(
        boundary_id="boundary",
        probe_id="probe",
        variant="cjk",
        apply_to="model",
        tier="cheap",
        evidence_kind="runtime",
        verdict="fail_loud",
        exception_type="FileNotFoundError",
        exception_message=str(FileNotFoundError(2, "missing", missing)),
        observation={"path": safe_path_repr(missing), "cwd": home},
    )
    out = tmp_path / "results.json"

    write_results(out, run, [result])

    run_dict, results = read_results(out)
    assert run_dict["system_temp"] == os.path.join(HOME_PLACEHOLDER, "AppData", "Local", "Temp")
    assert run_dict["leftover_paths"] == [os.path.join(HOME_PLACEHOLDER, "left")]
    redacted_missing = os.path.join(HOME_PLACEHOLDER, "missing.bin")
    assert results[0]["exception_message"] == str(FileNotFoundError(2, "missing", redacted_missing))
    assert results[0]["observation"] == {"path": safe_path_repr(redacted_missing), "cwd": HOME_PLACEHOLDER}
    assert not [s for s in _strings({"run": run_dict, "results": results}) if "taro" in s.lower()]


def test_committed_evidence_has_no_home_paths():
    """commit 済みの証拠に、ユーザー名を含むホームのパスが無い。

    失敗メッセージにはファイルと件数だけを出す (値を出すと CI のログにユーザー名が残る)。
    """
    files = sorted(EVIDENCE_DIR.glob("*/results.json"))
    assert files, "commit 済みの results.json が無い"
    leaks = {}
    for path in files:
        run, results = read_results(path)
        count = sum(len(HOME_LIKE.findall(s)) for s in _strings({"run": run, "results": results}))
        if count:
            leaks[path.relative_to(REPO_ROOT).as_posix()] = count
    assert not leaks, f"ホームのパスが残っている (redact_home を通していない): {leaks}"
