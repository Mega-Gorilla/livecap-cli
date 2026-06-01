from pathlib import Path
from io import StringIO
import sys

import pytest

from livecap_cli import cli


@pytest.mark.parametrize("ensure_ffmpeg", [False])
def test_cli_diagnose_reports_i18n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensure_ffmpeg: bool) -> None:
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))

    report = cli.diagnose(ensure_ffmpeg=ensure_ffmpeg)

    assert report.models_root
    assert report.cache_root
    assert report.i18n.fallback_count >= 0
    assert report.i18n.translator.registered in (True, False)
    assert isinstance(report.available_engines, list)
    # Phase 2: New diagnostic fields
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.vad_backends, list)
    # cuda_device can be None or str
    assert report.cuda_device is None or isinstance(report.cuda_device, str)


class TestCLISubcommands:
    """Tests for CLI subcommand structure (Issue #74 Phase 6B)."""

    def test_cli_no_command_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """No command shows help and returns 0."""
        result = cli.main([])
        captured = capsys.readouterr()
        assert result == 0
        assert "livecap-cli" in captured.out
        assert "info" in captured.out
        assert "devices" in captured.out
        assert "levels" in captured.out
        assert "engines" in captured.out
        assert "translators" in captured.out
        assert "transcribe" in captured.out

    def test_cli_info_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """info command shows diagnostics."""
        monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))

        result = cli.main(["info"])
        captured = capsys.readouterr()

        assert result == 0
        assert "livecap-cli diagnostics" in captured.out
        assert "FFmpeg:" in captured.out
        assert "Models root:" in captured.out

    def test_cli_info_as_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """info --as-json outputs valid JSON."""
        import json

        monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))

        result = cli.main(["info", "--as-json"])
        captured = capsys.readouterr()

        assert result == 0
        data = json.loads(captured.out)
        assert "models_root" in data
        assert "ffmpeg_path" in data
        assert "available_engines" in data

    def test_cli_engines_command(self, capsys: pytest.CaptureFixture) -> None:
        """engines command lists available engines."""
        result = cli.main(["engines"])
        captured = capsys.readouterr()

        assert result == 0
        # At least whispers2t should be available
        assert "whispers2t" in captured.out

    def test_cli_translators_command(self, capsys: pytest.CaptureFixture) -> None:
        """translators command lists available translators."""
        result = cli.main(["translators"])
        captured = capsys.readouterr()

        assert result == 0
        # At least google translator should be listed
        assert "google" in captured.out

    def test_cli_transcribe_requires_input(self, capsys: pytest.CaptureFixture) -> None:
        """transcribe without input shows error."""
        result = cli.main(["transcribe"])
        captured = capsys.readouterr()

        assert result == 1
        assert "Error:" in captured.err

    def test_cli_transcribe_realtime_requires_mic(self, capsys: pytest.CaptureFixture) -> None:
        """transcribe --realtime without --mic shows error."""
        result = cli.main(["transcribe", "--realtime"])
        captured = capsys.readouterr()

        assert result == 1
        assert "--mic" in captured.err


class TestCLINoiseGateOptions:
    """--noise-gate CLI オプションの parse テスト。"""

    def test_noise_gate_args_parsed(self) -> None:
        """--noise-gate 関連オプションが正しく parse される。"""
        import argparse

        # main() の parser を再構築せず、直接 parse_args を検証
        # transcribe コマンドに --noise-gate オプションが存在することを確認
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        # cli.main() と同等の構造を簡易再現
        from livecap_cli.cli import main

        # parse が通ることを確認（実行はしない）
        # SystemExit を避けるため、help 表示で確認
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                main(["transcribe", "--help"])
            except SystemExit:
                pass
        help_text = buf.getvalue()
        assert "--noise-gate" in help_text
        assert "--noise-gate-threshold" in help_text
        assert "--noise-gate-attack" in help_text
        assert "--noise-gate-release" in help_text
        # PR B additions (Issue #280 C-1/C-2)
        assert "--noise-gate-close-threshold" in help_text
        assert "--noise-gate-floor" in help_text

    def test_levels_command_in_help(self, capsys: pytest.CaptureFixture) -> None:
        """levels コマンドが help に表示される。"""
        result = cli.main([])
        captured = capsys.readouterr()
        assert "levels" in captured.out

    def test_levels_has_json_and_duration_options(self) -> None:
        """levels --help に --json / --duration が含まれる (Issue #280 C-4)。"""
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                cli.main(["levels", "--help"])
            except SystemExit:
                pass
        help_text = buf.getvalue()
        assert "--json" in help_text
        assert "--duration" in help_text


class TestLevelsBehavior:
    """levels コマンドの E2E 挙動テスト (Issue #280 C-4)。

    MicrophoneSource を fake 化することで、実マイク不要で
    --json 出力と --duration 自動停止の挙動を検証する。
    """

    @staticmethod
    def _install_fake_mic(monkeypatch: pytest.MonkeyPatch) -> None:
        """livecap_cli.MicrophoneSource を定常ノイズを返す fake に置換。"""
        import numpy as np

        class FakeMic:
            def __init__(self, device: int = 0) -> None:
                self.device = device

            def __enter__(self) -> "FakeMic":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def start(self) -> None:
                pass

            def read(self, timeout: float | None = None) -> "np.ndarray":
                # ambient -60 dB 相当の定常ノイズ (rms = 0.001 → 20*log10 = -60)
                return np.full(1600, 0.001, dtype=np.float32)

        import livecap_cli

        # `livecap_cli.MicrophoneSource` は __getattr__ 経由の遅延 import のため、
        # setattr が既存値確認で __getattr__ を呼び出し、PortAudio を読み込もうとする。
        # CI (Linux, PortAudio 無し) では OSError になるので、__dict__ に直接差し込んで
        # __getattr__ を回避する。
        monkeypatch.setitem(livecap_cli.__dict__, "MicrophoneSource", FakeMic)

    def test_levels_json_output_is_parseable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """levels --json の出力が JSON として parse でき、NoiseAnalysis 構造を持つ。"""
        import json

        self._install_fake_mic(monkeypatch)

        result = cli.main(
            ["levels", "--mic", "0", "--duration", "0.3", "--json"]
        )
        captured = capsys.readouterr()

        assert result == 0
        data = json.loads(captured.out)

        # NoiseAnalysis の全フィールドが存在する (issue #291 新 schema)
        expected_fields = {
            "noise_floor_db",
            "noise_rms_p95_db",
            "peak_p95_db",
            "suggested_threshold_db",
            "danger_zone",
            "sample_count",
            "duration_s",
        }
        assert expected_fields.issubset(set(data.keys()))
        # 旧 schema の field は削除されている
        assert "noise_peak_db" not in data
        assert "safe_zone_min_db" not in data

        # 値の整合性
        assert data["sample_count"] > 0
        assert data["duration_s"] > 0
        assert isinstance(data["danger_zone"], list) and len(data["danger_zone"]) == 2
        # suggested = peak_p95 + 6 (PEAK_SAFETY_MARGIN_DB)
        assert data["suggested_threshold_db"] == pytest.approx(
            data["peak_p95_db"] + 6.0
        )

    def test_levels_duration_stops_within_time_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """--duration 指定時に有限時間で終了する。"""
        import time

        self._install_fake_mic(monkeypatch)

        start = time.monotonic()
        result = cli.main(
            ["levels", "--mic", "0", "--duration", "0.3", "--json"]
        )
        elapsed = time.monotonic() - start

        assert result == 0
        # 0.3s 指定で 2s 以内に終了 (マージン込み)
        assert elapsed < 2.0, f"--duration did not stop in time: {elapsed:.2f}s"
