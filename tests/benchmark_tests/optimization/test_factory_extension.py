"""Tests for factory.py extension with backend_params and vad_config support."""

from __future__ import annotations

import pytest

from benchmarks.vad.factory import create_vad, get_all_vad_ids, VADConfig


# Check which backends are available
def _check_vad_available(vad_id: str) -> bool:
    """Check if a VAD backend is available."""
    try:
        create_vad(vad_id)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


# Skip markers for optional backends
requires_silero = pytest.mark.skipif(
    not _check_vad_available("silero"),
    reason="Silero VAD not available"
)
requires_tenvad = pytest.mark.skipif(
    not _check_vad_available("tenvad"),
    reason="TenVAD not available"
)
requires_javad = pytest.mark.skipif(
    not _check_vad_available("javad_tiny"),
    reason="JaVAD not available"
)


class TestCreateVadWithParams:
    """Test create_vad with custom parameters."""

    @requires_silero
    def test_create_silero_default(self):
        """create_vad should work with default parameters for Silero."""
        vad = create_vad("silero")
        assert vad is not None
        assert hasattr(vad, "process_audio")

    @requires_tenvad
    def test_create_tenvad_default(self):
        """create_vad should work with default parameters for TenVAD."""
        vad = create_vad("tenvad")
        assert vad is not None
        assert hasattr(vad, "process_audio")

    @requires_silero
    def test_create_silero_with_custom_backend_params(self):
        """Silero should accept custom threshold."""
        vad = create_vad("silero", backend_params={"threshold": 0.7})
        assert vad is not None
        # Verify the backend got the custom threshold (uses _threshold)
        assert vad._backend._threshold == 0.7

    @requires_silero
    def test_create_silero_with_custom_vad_config(self):
        """Silero should accept custom VADConfig."""
        config = VADConfig(
            min_speech_ms=300,
            min_silence_ms=200,
            speech_pad_ms=150,
        )
        vad = create_vad("silero", vad_config=config)
        assert vad is not None
        assert vad._config.min_speech_ms == 300
        assert vad._config.min_silence_ms == 200
        assert vad._config.speech_pad_ms == 150

    @requires_silero
    def test_create_silero_with_both_params(self):
        """Silero should accept both backend_params and vad_config."""
        config = VADConfig(min_speech_ms=400)
        vad = create_vad(
            "silero",
            backend_params={"threshold": 0.6},
            vad_config=config,
        )
        assert vad is not None
        assert vad._backend._threshold == 0.6
        assert vad._config.min_speech_ms == 400

    @requires_tenvad
    def test_create_tenvad_with_hop_size(self):
        """TenVAD should accept custom hop_size."""
        vad = create_vad("tenvad", backend_params={"hop_size": 160})
        assert vad is not None

    @requires_silero
    def test_backend_params_override_registry(self):
        """Custom backend_params should override registry defaults."""
        # Default threshold for silero is 0.5
        vad_default = create_vad("silero")
        vad_custom = create_vad("silero", backend_params={"threshold": 0.3})

        assert vad_default._backend._threshold == 0.5
        assert vad_custom._backend._threshold == 0.3


class TestCreateJavadWithParams:
    """Test create_vad for JaVAD variants."""

    @requires_javad
    @pytest.mark.parametrize(
        "vad_id,expected_model",
        [
            ("javad_tiny", "tiny"),
            ("javad_balanced", "balanced"),
            ("javad_precise", "precise"),
        ],
    )
    def test_javad_uses_registry_model(self, vad_id: str, expected_model: str):
        """JaVAD should use model from registry."""
        vad = create_vad(vad_id)
        assert vad is not None
        # JaVAD model is stored in _model attribute
        assert vad._model == expected_model

    @requires_javad
    def test_javad_backend_params_can_override_model(self):
        """JaVAD backend_params can override model."""
        vad = create_vad("javad_tiny", backend_params={"model": "precise"})
        assert vad._model == "precise"

    @requires_javad
    def test_javad_ignores_vad_config(self):
        """JaVAD should work even with vad_config (it's ignored)."""
        # JaVAD doesn't support VADConfig, but shouldn't crash
        config = VADConfig(min_speech_ms=300)
        vad = create_vad("javad_tiny", vad_config=config)
        assert vad is not None


class TestVADConfigExport:
    """Test that VADConfig is exported from factory."""

    def test_vadconfig_importable(self):
        """VADConfig should be importable from factory."""
        from benchmarks.vad.factory import VADConfig as FactoryVADConfig
        assert FactoryVADConfig is VADConfig
