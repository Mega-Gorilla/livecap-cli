"""Tests for --export-preset functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip all tests if optuna is not installed
pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.optimization.vad_optimizer import OptimizationResult


class TestSplitParams:
    """Test _split_params separates prefixed Optuna params correctly."""

    def test_silero_params(self):
        """Silero: all params are vad_config, no backend."""
        result = OptimizationResult(
            vad_type="silero",
            language="ja",
            best_params={
                "vad_config_threshold": 0.294,
                "vad_config_neg_threshold": 0.123,
                "vad_config_min_speech_ms": 450,
                "vad_config_min_silence_ms": 190,
                "vad_config_speech_pad_ms": 150,
            },
            best_score=0.082,
            n_trials=60,
            engine_id="parakeet_ja",
        )
        vad_config, backend = result._split_params()

        assert backend == {}
        assert vad_config == {
            "threshold": 0.294,
            "neg_threshold": 0.123,
            "min_speech_ms": 450,
            "min_silence_ms": 190,
            "speech_pad_ms": 150,
        }

    def test_tenvad_params(self):
        """TenVAD: hop_size goes to backend, rest to vad_config."""
        result = OptimizationResult(
            vad_type="tenvad",
            language="ja",
            best_params={
                "backend_hop_size": 256,
                "vad_config_threshold": 0.204,
                "vad_config_neg_threshold": 0.488,
                "vad_config_min_speech_ms": 400,
                "vad_config_min_silence_ms": 180,
                "vad_config_speech_pad_ms": 90,
            },
            best_score=0.072,
            n_trials=115,
            engine_id="parakeet_ja",
        )
        vad_config, backend = result._split_params()

        assert backend == {"hop_size": 256}
        assert vad_config["threshold"] == 0.204
        assert vad_config["min_speech_ms"] == 400

    def test_webrtc_params(self):
        """WebRTC: mode and frame_duration_ms go to backend, no threshold."""
        result = OptimizationResult(
            vad_type="webrtc",
            language="ja",
            best_params={
                "backend_mode": 1,
                "backend_frame_duration_ms": 30,
                "vad_config_min_speech_ms": 300,
                "vad_config_min_silence_ms": 150,
                "vad_config_speech_pad_ms": 80,
            },
            best_score=0.077,
            n_trials=145,
            engine_id="parakeet_ja",
        )
        vad_config, backend = result._split_params()

        assert backend == {"mode": 1, "frame_duration_ms": 30}
        assert "threshold" not in vad_config
        assert vad_config["min_speech_ms"] == 300


class TestBuildPresetDict:
    """Test build_preset_dict produces valid preset JSON."""

    def _make_result(self, vad_type: str, language: str) -> OptimizationResult:
        """Create a minimal OptimizationResult for testing."""
        params: dict = {
            "vad_config_min_speech_ms": 300,
            "vad_config_min_silence_ms": 150,
            "vad_config_speech_pad_ms": 80,
        }
        if vad_type in ("silero", "tenvad"):
            params["vad_config_threshold"] = 0.3
            params["vad_config_neg_threshold"] = 0.2
        if vad_type == "tenvad":
            params["backend_hop_size"] = 256
        if vad_type == "webrtc":
            params["backend_mode"] = 1
            params["backend_frame_duration_ms"] = 30

        return OptimizationResult(
            vad_type=vad_type,
            language=language,
            best_params=params,
            best_score=0.08,
            n_trials=50,
            engine_id="parakeet_ja" if language == "ja" else "parakeet",
            created_at="2026-01-01T00:00:00",
        )

    @pytest.mark.parametrize("vad_type", ["silero", "tenvad", "webrtc"])
    def test_valid_schema(self, vad_type: str):
        """build_preset_dict should produce a schema-valid preset."""
        result = self._make_result(vad_type, "ja")
        preset = result.build_preset_dict()

        assert preset["vad_type"] == vad_type
        assert preset["language"] == "ja"
        assert isinstance(preset["vad_config"], dict)
        assert isinstance(preset["backend"], dict)
        assert preset["metadata"]["metric"] == "cer"
        assert preset["metadata"]["engine"] == "parakeet_ja"
        assert preset["metadata"]["trials"] == 50

    def test_english_metric(self):
        """English should use 'wer' metric."""
        result = self._make_result("silero", "en")
        preset = result.build_preset_dict()

        assert preset["metadata"]["metric"] == "wer"

    def test_tenvad_backend(self):
        """TenVAD preset should include hop_size in backend."""
        result = self._make_result("tenvad", "ja")
        preset = result.build_preset_dict()

        assert preset["backend"]["hop_size"] == 256

    def test_webrtc_backend(self):
        """WebRTC preset should include mode and frame_duration_ms."""
        result = self._make_result("webrtc", "ja")
        preset = result.build_preset_dict()

        assert preset["backend"]["mode"] == 1
        assert preset["backend"]["frame_duration_ms"] == 30
        assert "threshold" not in preset["vad_config"]


class TestExportPreset:
    """Test export_preset writes valid JSON atomically."""

    def _make_result(self, vad_type: str = "silero") -> OptimizationResult:
        params: dict = {
            "vad_config_threshold": 0.3,
            "vad_config_neg_threshold": 0.2,
            "vad_config_min_speech_ms": 300,
            "vad_config_min_silence_ms": 150,
            "vad_config_speech_pad_ms": 80,
        }
        return OptimizationResult(
            vad_type=vad_type,
            language="ja",
            best_params=params,
            best_score=0.08,
            n_trials=50,
            engine_id="parakeet_ja",
            created_at="2026-01-01T00:00:00",
        )

    def test_writes_json_file(self, tmp_path: Path):
        """export_preset should create a JSON file."""
        result = self._make_result()
        path = result.export_preset(preset_dir=tmp_path)

        assert path.exists()
        assert path.name == "silero_ja_parakeet_ja.json"
        assert path.parent == tmp_path

    def test_json_content_valid(self, tmp_path: Path):
        """Written JSON should be loadable and match the preset schema."""
        result = self._make_result()
        path = result.export_preset(preset_dir=tmp_path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["vad_type"] == "silero"
        assert data["language"] == "ja"
        assert data["vad_config"]["threshold"] == 0.3
        assert data["metadata"]["score"] == 0.08
        assert data["metadata"]["metric"] == "cer"

    def test_overwrites_existing(self, tmp_path: Path):
        """export_preset should overwrite an existing file."""
        result1 = self._make_result()
        result1.best_score = 0.10
        result1.export_preset(preset_dir=tmp_path)

        result2 = self._make_result()
        result2.best_score = 0.07
        path = result2.export_preset(preset_dir=tmp_path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["score"] == 0.07

    def test_no_temp_files_on_success(self, tmp_path: Path):
        """No .tmp files should remain after successful export."""
        result = self._make_result()
        result.export_preset(preset_dir=tmp_path)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_json_ends_with_newline(self, tmp_path: Path):
        """Written file should end with a newline."""
        result = self._make_result()
        path = result.export_preset(preset_dir=tmp_path)

        assert path.read_text(encoding="utf-8").endswith("\n")


class TestCLIExportPresetArg:
    """Test --export-preset CLI argument parsing."""

    def test_parse_export_preset_flag(self):
        from benchmarks.optimization.__main__ import parse_args

        args = parse_args(["--vad", "silero", "--language", "ja", "--export-preset"])
        assert args.export_preset is True

    def test_default_no_export(self):
        from benchmarks.optimization.__main__ import parse_args

        args = parse_args(["--vad", "silero", "--language", "ja"])
        assert args.export_preset is False

    def test_parse_preset_dir(self):
        from benchmarks.optimization.__main__ import parse_args

        args = parse_args([
            "--vad", "silero", "--language", "ja",
            "--export-preset", "--preset-dir", "/tmp/presets",
        ])
        assert args.preset_dir == Path("/tmp/presets")

    def test_preset_dir_default_none(self):
        from benchmarks.optimization.__main__ import parse_args

        args = parse_args(["--vad", "silero", "--language", "ja"])
        assert args.preset_dir is None
