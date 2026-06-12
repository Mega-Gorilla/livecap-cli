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

        # NoiseAnalysis の全フィールドが存在する (#291 + #292 schema)
        expected_fields = {
            "noise_floor_db",
            "noise_rms_p95_db",
            "peak_p95_db",
            "suggested_threshold_db",
            "suggested_engine_min_rms_dbfs",  # 🆕 #292
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
        # #292: suggested_engine_min_rms = noise_rms_p95 + 6 (ENGINE_MIN_RMS_SAFETY_MARGIN_DB)
        assert data["suggested_engine_min_rms_dbfs"] == pytest.approx(
            data["noise_rms_p95_db"] + 6.0
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

    def test_levels_custom_engine_min_rms_margin(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """--engine-min-rms-margin で suggested_engine_min_rms_dbfs を任意調整可能。"""
        import json

        self._install_fake_mic(monkeypatch)

        # margin=10 で suggested = noise_rms_p95 + 10
        result = cli.main(
            [
                "levels", "--mic", "0", "--duration", "0.3", "--json",
                "--engine-min-rms-margin", "10",
            ]
        )
        captured = capsys.readouterr()
        assert result == 0
        data = json.loads(captured.out)
        assert data["suggested_engine_min_rms_dbfs"] == pytest.approx(
            data["noise_rms_p95_db"] + 10.0
        )


class TestEnergyGateFlags:
    """#292 EnergyGate CLI flag の parse テスト。"""

    @staticmethod
    def _parse_transcribe(*extra: str) -> "argparse.Namespace":
        """transcribe parser を再現して args を返す。

        cli.main() は parse 後すぐ実行に入るため、内部の add_argument を
        ミラーした最小 parser を組んで parse のみを検証する。
        """
        import argparse

        from livecap_cli.cli import (
            _parse_engine_energy_frame_ms,
            _parse_engine_min_rms,
        )

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--engine-min-rms",
            type=_parse_engine_min_rms,
            default=-45.0,
        )
        parser.add_argument(
            "--engine-energy-metric",
            choices=("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"),
            default="max_frame_rms",
        )
        parser.add_argument(
            "--engine-energy-frame-ms",
            type=_parse_engine_energy_frame_ms,
            default=32.0,
        )
        return parser.parse_args(list(extra))

    # === --engine-min-rms parse ===

    def test_engine_min_rms_numeric_parse(self) -> None:
        """数値: --engine-min-rms=-45 → -45.0"""
        a = self._parse_transcribe("--engine-min-rms=-45")
        assert a.engine_min_rms == -45.0

    def test_engine_min_rms_off_parse(self) -> None:
        """文字列 'off' → -inf"""
        a = self._parse_transcribe("--engine-min-rms", "off")
        assert a.engine_min_rms == float("-inf")

    def test_engine_min_rms_disabled_parse(self) -> None:
        """文字列 'disabled' → -inf"""
        a = self._parse_transcribe("--engine-min-rms", "disabled")
        assert a.engine_min_rms == float("-inf")

    def test_engine_min_rms_equals_minus_inf_parse(self) -> None:
        """equals 形式 --engine-min-rms=-inf → -inf"""
        a = self._parse_transcribe("--engine-min-rms=-inf")
        assert a.engine_min_rms == float("-inf")

    def test_engine_min_rms_default(self) -> None:
        """default は -45.0"""
        a = self._parse_transcribe()
        assert a.engine_min_rms == -45.0

    def test_engine_min_rms_invalid_raises(self) -> None:
        """不正値で argparse error (SystemExit)"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-min-rms", "abc")

    # === --engine-energy-metric choices ===

    def test_engine_energy_metric_default(self) -> None:
        a = self._parse_transcribe()
        assert a.engine_energy_metric == "max_frame_rms"

    def test_engine_energy_metric_choices_all_valid(self) -> None:
        for m in ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"):
            a = self._parse_transcribe("--engine-energy-metric", m)
            assert a.engine_energy_metric == m

    def test_engine_energy_metric_invalid_raises(self) -> None:
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-energy-metric", "bogus")

    # === --engine-energy-frame-ms parse ===

    def test_engine_energy_frame_ms_parse(self) -> None:
        a = self._parse_transcribe("--engine-energy-frame-ms", "50")
        assert a.engine_energy_frame_ms == 50.0

    def test_engine_energy_frame_ms_default(self) -> None:
        a = self._parse_transcribe()
        assert a.engine_energy_frame_ms == 32.0

    # === nan / inf rejection (codex-review followup) ===

    def test_engine_min_rms_nan_rejected(self) -> None:
        """--engine-min-rms nan は silent disable を防ぐため argparse error。"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-min-rms", "nan")

    def test_engine_min_rms_positive_inf_rejected(self) -> None:
        """--engine-min-rms=inf も argparse error (全 segment drop の sanity 防止)。"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-min-rms", "inf")

    def test_engine_min_rms_neg_inf_via_equals_accepted(self) -> None:
        """--engine-min-rms=-inf は引き続き opt-out として受け入れる。"""
        a = self._parse_transcribe("--engine-min-rms=-inf")
        assert a.engine_min_rms == float("-inf")

    def test_engine_energy_frame_ms_nan_rejected(self) -> None:
        """--engine-energy-frame-ms nan は <=0 check をすり抜けるため reject。"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-energy-frame-ms", "nan")

    def test_engine_energy_frame_ms_inf_rejected(self) -> None:
        """--engine-energy-frame-ms inf は int() overflow を防ぐため reject。"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-energy-frame-ms", "inf")

    def test_engine_energy_frame_ms_negative_rejected(self) -> None:
        """--engine-energy-frame-ms negative も reject (既存 positive check)。"""
        with pytest.raises(SystemExit):
            self._parse_transcribe("--engine-energy-frame-ms", "-1")

    # === help text を通じての可視性確認 ===

    def test_transcribe_help_lists_energy_gate_flags(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """transcribe --help に 3 つの --engine-* flag が表示される。"""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                cli.main(["transcribe", "--help"])
            except SystemExit:
                pass
        help_text = buf.getvalue()
        assert "--engine-min-rms" in help_text
        assert "--engine-energy-metric" in help_text
        assert "--engine-energy-frame-ms" in help_text
        # help text に物理量警告が含まれる
        assert "--noise-gate-threshold" in help_text  # cross-reference
        # default が conservative であることと levels calibration 推奨の明示
        assert "conservative" in help_text.lower()
        assert "levels" in help_text  # `livecap-cli levels` を案内


class TestConfidenceFilterFlag:
    """PR-A.1: ``--confidence-filter`` flag + ``LIVECAP_CONFIDENCE_FILTER`` env var の parse test."""

    @staticmethod
    def _parse_transcribe(*extra: str) -> "argparse.Namespace":
        """transcribe parser の最小 mirror で parse を検証。"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--confidence-filter",
            choices=("off", "observe", "on"),
            default="on",
        )
        return parser.parse_args(list(extra))

    def test_default_is_on(self) -> None:
        """v3.1: production default は ``on`` (filter 適用)。"""
        args = self._parse_transcribe()
        assert args.confidence_filter == "on"

    def test_off_parses(self) -> None:
        args = self._parse_transcribe("--confidence-filter", "off")
        assert args.confidence_filter == "off"

    def test_observe_parses(self) -> None:
        args = self._parse_transcribe("--confidence-filter", "observe")
        assert args.confidence_filter == "observe"

    def test_invalid_raises_systemexit(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            self._parse_transcribe("--confidence-filter", "invalid")

    def test_env_var_overrides_cli_flag(self, monkeypatch) -> None:
        """``LIVECAP_CONFIDENCE_FILTER`` が CLI flag より優先される。"""
        import argparse

        from livecap_cli.cli import _create_filter_config

        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "off")
        args = argparse.Namespace(confidence_filter="on")
        cfg = _create_filter_config(args)
        assert cfg.mode == "off", "env var が CLI flag より優先"

    def test_env_var_observe_overrides_cli_on(self, monkeypatch) -> None:
        import argparse

        from livecap_cli.cli import _create_filter_config

        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "observe")
        args = argparse.Namespace(confidence_filter="on")
        cfg = _create_filter_config(args)
        assert cfg.mode == "observe"

    def test_empty_env_var_falls_through_to_cli(self, monkeypatch) -> None:
        """env var 未設定 / 空文字 → CLI flag を使う。"""
        import argparse

        from livecap_cli.cli import _create_filter_config

        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "")
        args = argparse.Namespace(confidence_filter="on")
        cfg = _create_filter_config(args)
        assert cfg.mode == "on"

    def test_invalid_env_var_falls_through_to_cli(
        self, monkeypatch, capsys
    ) -> None:
        """invalid env var は warning 出力 + CLI flag を使う。"""
        import argparse

        from livecap_cli.cli import _create_filter_config

        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "garbage")
        args = argparse.Namespace(confidence_filter="off")
        cfg = _create_filter_config(args)
        assert cfg.mode == "off", "invalid env var は無視され CLI flag を使う"
        captured = capsys.readouterr()
        assert "LIVECAP_CONFIDENCE_FILTER" in captured.err
        assert "invalid" in captured.err.lower()

    def test_env_var_case_insensitive(self, monkeypatch) -> None:
        """env var は case-insensitive (lower 化して whitelist 検証)。"""
        import argparse

        from livecap_cli.cli import _create_filter_config

        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "OFF")
        args = argparse.Namespace(confidence_filter="on")
        cfg = _create_filter_config(args)
        assert cfg.mode == "off"

    def test_help_text_mentions_filter(self) -> None:
        """transcribe --help に flag と env var override の説明が含まれる。"""
        import sys
        from io import StringIO

        from livecap_cli.cli import main

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            try:
                main(["transcribe", "--help"])
            except SystemExit:
                pass
            help_text = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "--confidence-filter" in help_text
        assert "LIVECAP_CONFIDENCE_FILTER" in help_text


class TestBuildEngineKwargs:
    """PR-A.5.2 codex Point 1 regression: ``_build_engine_kwargs`` で
    ``--language`` が qwen3asr engine に確実に渡ることを pin。

    渡らない場合 ``Qwen3ASREngine._asr_language`` が None になり、
    ``transcribe()`` が wrapper fallback path (engine_confidence 全 None) に入り、
    PR の主目的 (avg_logprob populate → confidence filter) が CLI 経路で無効化される。
    """

    def _make_args(self, **overrides):
        import argparse
        ns = argparse.Namespace(
            engine="whispers2t",
            language="ja",
            model_size="base",
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_qwen3asr_receives_language_ja(self) -> None:
        from livecap_cli.cli import _build_engine_kwargs

        args = self._make_args(engine="qwen3asr", language="ja")
        kwargs = _build_engine_kwargs(args)

        assert kwargs.get("language") == "ja", (
            "qwen3asr requires --language pass-through (else wrapper "
            "fallback fail-open). See PR-A.5.2 codex Point 1."
        )

    def test_qwen3asr_receives_language_en(self) -> None:
        from livecap_cli.cli import _build_engine_kwargs

        args = self._make_args(engine="qwen3asr", language="en")
        kwargs = _build_engine_kwargs(args)

        assert kwargs.get("language") == "en"

    def test_qwen3asr_receives_language_auto(self) -> None:
        """``--language auto`` でも literal を pass-through (engine 側で None に
        resolve、auto-detect fail-open path に乗る)。CLI 層は decision を
        engine に委譲する。"""
        from livecap_cli.cli import _build_engine_kwargs

        args = self._make_args(engine="qwen3asr", language="auto")
        kwargs = _build_engine_kwargs(args)

        assert kwargs.get("language") == "auto"

    def test_whispers2t_does_not_receive_language(self) -> None:
        """language pass-through は qwen3asr 専用 (whispers2t は VAD 経由で扱う)。"""
        from livecap_cli.cli import _build_engine_kwargs

        args = self._make_args(engine="whispers2t", language="ja", model_size="base")
        kwargs = _build_engine_kwargs(args)

        assert "language" not in kwargs
        assert kwargs.get("model_size") == "base"

    def test_voxtral_does_not_receive_language(self) -> None:
        """Voxtral は default ``language='auto'`` で auto-detect mode に
        従って動作するため、CLI 層で明示的に渡さない (既存 behavior 維持)。"""
        from livecap_cli.cli import _build_engine_kwargs

        args = self._make_args(engine="voxtral", language="ja")
        kwargs = _build_engine_kwargs(args)

        assert "language" not in kwargs

    def test_qwen3asr_realtime_e2e_engine_factory_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--realtime --engine qwen3asr --language ja`` で
        EngineFactory.create_engine() が ``language='ja'`` 付きで呼ばれることを
        実 CLI flow で pin。"""
        # GitHub Actions Linux runner 等 PortAudio 未 install の環境では
        # ``livecap_cli.MicrophoneSource`` への access が sounddevice import を
        # trigger し OSError で fail する。上の 5 件で helper の behavior は
        # pin 済のため、e2e 経路 verify はオーディオ環境が揃った場合のみ実行。
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as e:
            pytest.skip(f"sounddevice/PortAudio unavailable: {e}")

        from unittest.mock import MagicMock

        from livecap_cli.cli import main
        from livecap_cli.engines import EngineFactory

        captured: dict = {}

        def fake_create_engine(engine_name, **kwargs):
            captured["engine_name"] = engine_name
            captured["kwargs"] = kwargs
            mock = MagicMock()
            mock.load_model.return_value = None
            return mock

        monkeypatch.setattr(EngineFactory, "create_engine", fake_create_engine)

        class _FailingMic:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                raise RuntimeError("test-early-exit")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr("livecap_cli.MicrophoneSource", _FailingMic)

        rc = main([
            "transcribe", "--realtime", "--mic", "0",
            "--engine", "qwen3asr", "--language", "ja",
        ])

        assert rc == 1  # early-exit RuntimeError
        assert captured.get("engine_name") == "qwen3asr"
        assert captured.get("kwargs", {}).get("language") == "ja", (
            "EngineFactory.create_engine() must receive language='ja' for "
            "qwen3asr; otherwise confidence filter is disabled in CLI path."
        )
