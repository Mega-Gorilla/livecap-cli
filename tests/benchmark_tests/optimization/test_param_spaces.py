"""Tests for parameter space definitions and suggestion functions."""

from __future__ import annotations

import pytest

# Skip all tests if optuna is not installed
optuna = pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.optimization.param_spaces import (
    PARAM_SPACES,
    suggest_params,
    get_supported_vad_types,
)
from livecap_core.vad.config import VADConfig


class TestParamSpaces:
    """Test parameter space definitions."""

    def test_param_spaces_has_required_vads(self):
        """PARAM_SPACES should include silero, tenvad, webrtc."""
        assert "silero" in PARAM_SPACES
        assert "tenvad" in PARAM_SPACES
        assert "webrtc" in PARAM_SPACES

    def test_param_spaces_structure(self):
        """Each VAD should have backend and/or vad_config groups."""
        for vad_type, space in PARAM_SPACES.items():
            assert isinstance(space, dict), f"{vad_type} should be a dict"
            assert "backend" in space or "vad_config" in space, (
                f"{vad_type} should have backend or vad_config"
            )

    def test_webrtc_excludes_threshold(self):
        """WebRTC should not have threshold in backend params."""
        webrtc_backend = PARAM_SPACES["webrtc"]["backend"]
        assert "threshold" not in webrtc_backend, (
            "WebRTC outputs binary, threshold should be excluded"
        )
        assert "mode" in webrtc_backend
        assert "frame_duration_ms" in webrtc_backend

    def test_silero_threshold_in_vad_config(self):
        """Silero should have threshold in vad_config (not backend)."""
        silero_backend = PARAM_SPACES["silero"]["backend"]
        silero_vad_config = PARAM_SPACES["silero"]["vad_config"]
        # threshold is in vad_config because state machine uses VADConfig.threshold
        assert "threshold" not in silero_backend
        assert "threshold" in silero_vad_config

    def test_tenvad_includes_hop_size(self):
        """TenVAD should have hop_size in backend params."""
        tenvad_backend = PARAM_SPACES["tenvad"]["backend"]
        assert "hop_size" in tenvad_backend
        # NOTE: threshold is in vad_config, NOT backend_params
        # (state machine uses VADConfig.threshold for detection)


class TestSuggestParams:
    """Test parameter suggestion function."""

    def test_suggest_silero_params(self):
        """suggest_params should return valid Silero parameters."""
        study = optuna.create_study()
        trial = study.ask()

        backend_params, vad_config = suggest_params(trial, "silero")

        # Check backend params (empty for Silero - threshold moved to vad_config)
        # Silero backend threshold is NOT used for detection
        assert "threshold" not in backend_params

        # Check VADConfig - threshold IS here (used by state machine)
        assert vad_config is not None
        assert isinstance(vad_config, VADConfig)
        assert 0.2 <= vad_config.threshold <= 0.8
        assert 100 <= vad_config.min_speech_ms <= 500
        assert 30 <= vad_config.min_silence_ms <= 300
        assert 30 <= vad_config.speech_pad_ms <= 200

    def test_suggest_tenvad_params(self):
        """suggest_params should return valid TenVAD parameters."""
        study = optuna.create_study()
        trial = study.ask()

        backend_params, vad_config = suggest_params(trial, "tenvad")

        # Check backend params (only hop_size, threshold moved to vad_config)
        assert "hop_size" in backend_params
        assert backend_params["hop_size"] in [160, 256]
        assert "threshold" not in backend_params  # Moved to vad_config

        # Check VADConfig - threshold IS here (used by state machine)
        assert vad_config is not None
        assert isinstance(vad_config, VADConfig)
        assert 0.2 <= vad_config.threshold <= 0.8

    def test_suggest_webrtc_params(self):
        """suggest_params should return valid WebRTC parameters."""
        study = optuna.create_study()
        trial = study.ask()

        backend_params, vad_config = suggest_params(trial, "webrtc")

        # Check backend params (no threshold)
        assert "threshold" not in backend_params
        assert "mode" in backend_params
        assert backend_params["mode"] in [0, 1, 2, 3]
        assert "frame_duration_ms" in backend_params
        assert backend_params["frame_duration_ms"] in [10, 20, 30]

        # Check VADConfig
        assert vad_config is not None
        assert isinstance(vad_config, VADConfig)

    def test_suggest_unknown_vad_raises(self):
        """suggest_params should raise ValueError for unknown VAD type."""
        study = optuna.create_study()
        trial = study.ask()

        with pytest.raises(ValueError, match="Unknown VAD type"):
            suggest_params(trial, "unknown_vad")

    def test_suggest_params_deterministic_with_seed(self):
        """Same seed should produce same parameters."""
        sampler1 = optuna.samplers.TPESampler(seed=42)
        sampler2 = optuna.samplers.TPESampler(seed=42)

        study1 = optuna.create_study(sampler=sampler1)
        study2 = optuna.create_study(sampler=sampler2)

        trial1 = study1.ask()
        trial2 = study2.ask()

        params1, config1 = suggest_params(trial1, "silero")
        params2, config2 = suggest_params(trial2, "silero")

        # threshold is in vad_config (not backend_params)
        assert config1.threshold == config2.threshold
        assert config1.min_speech_ms == config2.min_speech_ms


class TestGetSupportedVadTypes:
    """Test get_supported_vad_types function."""

    def test_returns_list(self):
        """Should return a list of VAD types."""
        vad_types = get_supported_vad_types()
        assert isinstance(vad_types, list)
        assert len(vad_types) > 0

    def test_excludes_javad(self):
        """JaVAD should not be in the list (uses Grid Search)."""
        vad_types = get_supported_vad_types()
        assert "javad" not in vad_types
        assert "javad_tiny" not in vad_types

    def test_includes_bayesian_vads(self):
        """Should include VADs that use Bayesian optimization."""
        vad_types = get_supported_vad_types()
        assert "silero" in vad_types
        assert "tenvad" in vad_types
        assert "webrtc" in vad_types
