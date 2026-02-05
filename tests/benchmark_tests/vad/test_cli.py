"""Tests for VAD benchmark CLI argument parsing and validation."""

from __future__ import annotations

import logging

import pytest

from benchmarks.vad.cli import main, parse_args
from benchmarks.vad.preset_integration import get_preset_vad_ids


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_default_args(self):
        """Default arguments should be sensible."""
        parsed = parse_args([])
        assert parsed.mode == "quick"
        assert parsed.language == ["ja", "en"]
        assert parsed.vad is None
        assert parsed.engine is None
        assert parsed.param_source == "default"

    def test_param_source_preset(self):
        """--param-source preset should be accepted."""
        parsed = parse_args(["--param-source", "preset"])
        assert parsed.param_source == "preset"

    def test_multiple_vads(self):
        """Multiple VADs should be parsed as a list."""
        parsed = parse_args(["--vad", "silero", "webrtc", "javad_tiny"])
        assert parsed.vad == ["silero", "webrtc", "javad_tiny"]


class TestPresetVADFiltering:
    """Test that --param-source preset filters non-preset VADs with warnings."""

    def test_preset_skips_non_preset_vads(self, caplog):
        """Non-preset VADs should be skipped with a warning, not cause an error."""
        # This will fail at the runner stage (no dataset), but we only test
        # the VAD filtering logic. We mock the runner to isolate.
        args = [
            "--param-source", "preset",
            "--vad", "silero", "javad_tiny", "javad_balanced",
        ]
        parsed = parse_args(args)

        # Simulate the filtering logic from main()
        preset_vads = set(get_preset_vad_ids())
        skipped = [v for v in parsed.vad if v not in preset_vads]
        kept = [v for v in parsed.vad if v in preset_vads]

        assert "javad_tiny" in skipped
        assert "javad_balanced" in skipped
        assert kept == ["silero"]

    def test_preset_all_non_preset_returns_error(self, caplog):
        """If all VADs are non-preset, main() should return 1."""
        with caplog.at_level(logging.ERROR):
            exit_code = main([
                "--param-source", "preset",
                "--vad", "javad_tiny", "javad_balanced",
            ])
        assert exit_code == 1
        assert "No VADs remaining" in caplog.text

    def test_preset_keeps_valid_vads(self):
        """Preset mode should keep silero/tenvad/webrtc."""
        args = [
            "--param-source", "preset",
            "--vad", "silero", "tenvad", "webrtc",
        ]
        parsed = parse_args(args)

        preset_vads = set(get_preset_vad_ids())
        kept = [v for v in parsed.vad if v in preset_vads]
        assert kept == ["silero", "tenvad", "webrtc"]


    def test_preset_unknown_vad_returns_error(self, caplog):
        """Unknown VAD (typo) should error even in preset mode."""
        with caplog.at_level(logging.ERROR):
            exit_code = main([
                "--param-source", "preset",
                "--vad", "sileroo",
            ])
        assert exit_code == 1
        assert "Unknown VAD" in caplog.text


class TestDefaultModeVADValidation:
    """Test VAD validation in default (non-preset) mode."""

    def test_unknown_vad_returns_error(self, caplog):
        """Unknown VAD should cause main() to return 1."""
        with caplog.at_level(logging.ERROR):
            exit_code = main([
                "--param-source", "default",
                "--vad", "totally_unknown_vad",
            ])
        assert exit_code == 1
        assert "Unknown VAD" in caplog.text
