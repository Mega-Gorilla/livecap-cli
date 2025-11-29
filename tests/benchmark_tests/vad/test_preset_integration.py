"""Tests for VAD benchmark preset integration."""

from __future__ import annotations

import pytest

from benchmarks.vad.preset_integration import (
    OPTIMIZABLE_VADS,
    create_vad_with_preset,
    get_preset_vad_ids,
    get_preset_config,
    is_preset_available,
)


class TestGetPresetVadIds:
    """Test get_preset_vad_ids function."""

    def test_returns_list(self):
        """Should return a list of VAD IDs."""
        result = get_preset_vad_ids()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_contains_expected_vads(self):
        """Should contain silero, tenvad, webrtc."""
        result = get_preset_vad_ids()
        assert "silero" in result
        assert "tenvad" in result
        assert "webrtc" in result

    def test_excludes_javad(self):
        """Should not contain javad variants."""
        result = get_preset_vad_ids()
        assert "javad_tiny" not in result
        assert "javad_balanced" not in result
        assert "javad_precise" not in result


class TestIsPresetAvailable:
    """Test is_preset_available function."""

    def test_available_for_silero_ja(self):
        """Should return True for silero/ja."""
        assert is_preset_available("silero", "ja") is True

    def test_available_for_webrtc_en(self):
        """Should return True for webrtc/en."""
        assert is_preset_available("webrtc", "en") is True

    def test_unavailable_for_unknown_vad(self):
        """Should return False for unknown VAD."""
        assert is_preset_available("unknown_vad", "ja") is False

    def test_unavailable_for_unknown_language(self):
        """Should return False for unknown language."""
        assert is_preset_available("silero", "zh") is False

    def test_unavailable_for_javad(self):
        """Should return False for javad (not optimizable)."""
        assert is_preset_available("javad_balanced", "ja") is False


class TestGetPresetConfig:
    """Test get_preset_config function."""

    def test_returns_config_for_valid_combination(self):
        """Should return config dict for valid combination."""
        config = get_preset_config("silero", "ja")
        assert config is not None
        assert isinstance(config, dict)
        assert "vad_config" in config
        assert "metadata" in config

    def test_vad_config_has_required_keys(self):
        """VAD config should have required parameter keys."""
        config = get_preset_config("silero", "ja")
        vad_config = config["vad_config"]
        assert "threshold" in vad_config
        assert "min_speech_ms" in vad_config
        assert "min_silence_ms" in vad_config
        assert "speech_pad_ms" in vad_config

    def test_returns_none_for_unknown_vad(self):
        """Should return None for unknown VAD."""
        config = get_preset_config("unknown_vad", "ja")
        assert config is None

    def test_returns_none_for_unknown_language(self):
        """Should return None for unknown language."""
        config = get_preset_config("silero", "zh")
        assert config is None

    def test_webrtc_has_backend_config(self):
        """WebRTC preset should have backend config."""
        config = get_preset_config("webrtc", "ja")
        assert "backend" in config
        assert "mode" in config["backend"]

    def test_tenvad_has_backend_config(self):
        """TenVAD preset should have backend config."""
        config = get_preset_config("tenvad", "ja")
        assert "backend" in config
        assert "hop_size" in config["backend"]


class TestCreateVadWithPreset:
    """Test create_vad_with_preset function."""

    def test_creates_silero_vad_for_ja(self):
        """Should create Silero VAD for Japanese."""
        try:
            vad = create_vad_with_preset("silero", "ja")
            # Check that it has the expected method (Protocol check)
            assert hasattr(vad, "process_audio")
            assert callable(vad.process_audio)
        except ImportError as e:
            pytest.skip(f"silero_vad not installed: {e}")

    def test_creates_silero_vad_for_en(self):
        """Should create Silero VAD for English."""
        try:
            vad = create_vad_with_preset("silero", "en")
            # Check that it has the expected method (Protocol check)
            assert hasattr(vad, "process_audio")
            assert callable(vad.process_audio)
        except ImportError as e:
            pytest.skip(f"silero_vad not installed: {e}")

    def test_raises_for_unknown_vad(self):
        """Should raise ValueError for unknown VAD."""
        with pytest.raises(ValueError) as exc_info:
            create_vad_with_preset("unknown_vad", "ja")
        assert "No preset for unknown_vad/ja" in str(exc_info.value)

    def test_raises_for_unknown_language(self):
        """Should raise ValueError for unknown language."""
        with pytest.raises(ValueError) as exc_info:
            create_vad_with_preset("silero", "zh")
        assert "No preset for silero/zh" in str(exc_info.value)


class TestOptimizableVads:
    """Test OPTIMIZABLE_VADS constant."""

    def test_is_list(self):
        """Should be a list."""
        assert isinstance(OPTIMIZABLE_VADS, list)

    def test_contains_expected_vads(self):
        """Should contain expected VADs."""
        assert "silero" in OPTIMIZABLE_VADS
        assert "tenvad" in OPTIMIZABLE_VADS
        assert "webrtc" in OPTIMIZABLE_VADS

    def test_excludes_javad(self):
        """Should not contain javad."""
        assert "javad_tiny" not in OPTIMIZABLE_VADS
        assert "javad_balanced" not in OPTIMIZABLE_VADS
        assert "javad_precise" not in OPTIMIZABLE_VADS
