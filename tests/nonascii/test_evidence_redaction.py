"""証拠 JSON にホームディレクトリ (ユーザー名) を残さないこと。

``benchmark_results/nonascii/<date>/results.json`` は commit して公開する。以前は
``system_temp`` に ``C:\\Users\\<name>\\AppData\\Local\\Temp`` をそのまま書き、測定者の
ユーザー名が公開されていた。ユーザー名が非 ASCII かどうかは ``username_is_ascii`` /
``system_temp_is_ascii`` が残すので、パスそのものは要らない。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .record import HOME_PLACEHOLDER, RunMetadata, read_results, redact_home, write_results

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


def test_write_results_redacts_the_real_home(tmp_path: Path):
    """実際のホーム (``os.path.expanduser("~")``) で書いた値が、ファイルに残らない。"""
    home = Path.home()
    run = RunMetadata(
        run_id="run-test",
        measured_at="2026-11-10",
        system_temp=str(home / "AppData" / "Local" / "Temp"),
        leftover_paths=[str(home / "left")],
    )
    out = tmp_path / "results.json"

    write_results(out, run, [])

    run_dict, _results = read_results(out)
    assert run_dict["system_temp"] == str(Path(HOME_PLACEHOLDER) / "AppData" / "Local" / "Temp")
    assert run_dict["leftover_paths"] == [str(Path(HOME_PLACEHOLDER) / "left")]
    home_lower = str(home).lower()
    assert not [s for s in _strings(run_dict) if home_lower in s.lower()]


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
