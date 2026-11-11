"""PATH に ffmpeg が無く ``LIVECAP_FFMPEG_BIN`` だけのホストで、whisper_s2t の import 時の確認が通る (#474)。

livecap-gui (Steam 配布) は同梱の ffmpeg を ``LIVECAP_FFMPEG_BIN`` で渡し、PATH には入れない。
``whisper_s2t`` は import 時に ``os.system('ffmpeg -version')`` で PATH を確かめるので、import の前に
PATH へ反映しないと ``Seems 'ffmpeg' is not installed`` が出て、resampler も確かめられない。

PATH の書き換えは Windows だけ (``FFmpegManager`` の方針) なので Windows で回す。実の ffmpeg
(``LIVECAP_FFMPEG_BIN`` または ``./ffmpeg-bin``) と ``whisper_s2t`` が無ければ skip。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WARNING = "Seems 'ffmpeg' is not installed"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="PATH を書き換えるのは Windows だけ")


def _real_ffmpeg_dir() -> Path | None:
    for candidate in (os.environ.get("LIVECAP_FFMPEG_BIN"), REPO_ROOT / "ffmpeg-bin"):
        if candidate and (Path(candidate) / "ffmpeg.exe").is_file():
            return Path(candidate)
    return None


def _run(snippet: str, ffmpeg_dir: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """PATH を Windows の system dir だけにし、ffmpeg は ``LIVECAP_FFMPEG_BIN`` だけで渡す。"""
    drop = {"PATH", "LIVECAP_FFMPEG_BIN", "FFMPEG_BINARY", "PYTHONPATH"}
    env = {key: value for key, value in os.environ.items() if key.upper() not in drop}
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([str(Path(system_root) / "System32"), system_root])
    env["LIVECAP_FFMPEG_BIN"] = str(ffmpeg_dir)
    env["LIVECAP_CORE_CACHE_DIR"] = str(tmp_path / "cache")
    env["LIVECAP_CORE_MODELS_DIR"] = str(tmp_path / "models")
    env["PYTHONPATH"] = str(REPO_ROOT)  # この checkout のコードを読む (editable install が別の場所を指していても)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )


def test_import_time_check_passes_with_ffmpeg_only_in_livecap_ffmpeg_bin(tmp_path):
    ffmpeg_dir = _real_ffmpeg_dir()
    if ffmpeg_dir is None:
        pytest.skip("実の ffmpeg (LIVECAP_FFMPEG_BIN / ./ffmpeg-bin) が無い")
    if importlib.util.find_spec("whisper_s2t") is None:
        pytest.skip("whisper_s2t が入っていない")

    # 前提: 修正前と同じ素の import では確認が失敗する (= PATH から ffmpeg を外せている)
    plain = _run("import whisper_s2t.audio", ffmpeg_dir, tmp_path)
    assert plain.returncode == 0, plain.stderr[-2000:]
    if WARNING not in plain.stdout:
        pytest.fail(f"前提が崩れた: PATH を絞っても ffmpeg が見つかっている\n{plain.stdout[-2000:]}")

    fixed = _run(
        "from livecap_cli.engines.whisper_s2t_import import import_whisper_s2t\n"
        "import_whisper_s2t()\n"
        "import whisper_s2t.audio as audio\n"
        "print('RESAMPLING_ENGINE=' + audio.RESAMPLING_ENGINE)\n",
        ffmpeg_dir,
        tmp_path,
    )

    assert fixed.returncode == 0, fixed.stderr[-2000:]
    assert WARNING not in fixed.stdout, fixed.stdout[-2000:]
    assert "RESAMPLING_ENGINE=" in fixed.stdout
