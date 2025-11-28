"""Tests for VAD factory."""

from __future__ import annotations

import pytest

from benchmarks.vad.factory import (
    VAD_REGISTRY,
    create_vad,
    get_all_vad_ids,
    get_vad_config,
)


class TestVADRegistry:
    """Tests for VAD_REGISTRY."""

    def test_registry_has_expected_vads(self) -> None:
        """Test registry contains all expected VADs."""
        expected = {
            "silero",
            "webrtc_mode0",
            "webrtc_mode1",
            "webrtc_mode2",
            "webrtc_mode3",
            "tenvad",
            "javad_tiny",
            "javad_balanced",
            "javad_precise",
        }
        assert set(VAD_REGISTRY.keys()) == expected

    def test_protocol_vads_have_required_fields(self) -> None:
        """Test protocol VADs have required configuration fields."""
        protocol_vads = [
            "silero",
            "webrtc_mode0",
            "webrtc_mode1",
            "webrtc_mode2",
            "webrtc_mode3",
            "tenvad",
        ]
        for vad_id in protocol_vads:
            config = VAD_REGISTRY[vad_id]
            assert config["type"] == "protocol"
            assert "backend_class" in config
            assert "module" in config
            assert "params" in config

    def test_javad_vads_have_required_fields(self) -> None:
        """Test JaVAD VADs have required configuration fields."""
        javad_vads = ["javad_tiny", "javad_balanced", "javad_precise"]
        for vad_id in javad_vads:
            config = VAD_REGISTRY[vad_id]
            assert config["type"] == "javad"
            assert "model" in config


class TestGetAllVadIds:
    """Tests for get_all_vad_ids()."""

    def test_returns_all_ids(self) -> None:
        """Test returns all VAD IDs from registry."""
        ids = get_all_vad_ids()
        assert len(ids) == len(VAD_REGISTRY)
        assert set(ids) == set(VAD_REGISTRY.keys())

    def test_returns_list(self) -> None:
        """Test returns a list."""
        ids = get_all_vad_ids()
        assert isinstance(ids, list)


class TestGetVadConfig:
    """Tests for get_vad_config()."""

    def test_returns_config_copy(self) -> None:
        """Test returns a copy of config (not original)."""
        config1 = get_vad_config("silero")
        config2 = get_vad_config("silero")
        assert config1 == config2
        assert config1 is not config2  # Should be different objects

    def test_silero_config(self) -> None:
        """Test silero configuration."""
        config = get_vad_config("silero")
        assert config["type"] == "protocol"
        assert config["backend_class"] == "SileroVAD"
        assert "threshold" in config["params"]

    def test_webrtc_mode3_config(self) -> None:
        """Test webrtc_mode3 configuration."""
        config = get_vad_config("webrtc_mode3")
        assert config["type"] == "protocol"
        assert config["params"]["mode"] == 3
        assert config["params"]["frame_duration_ms"] == 20

    def test_javad_balanced_config(self) -> None:
        """Test javad_balanced configuration."""
        config = get_vad_config("javad_balanced")
        assert config["type"] == "javad"
        assert config["model"] == "balanced"

    def test_unknown_vad_raises(self) -> None:
        """Test unknown VAD raises ValueError."""
        with pytest.raises(ValueError, match="Unknown VAD"):
            get_vad_config("unknown_vad")


class TestCreateVad:
    """Tests for create_vad()."""

    def test_unknown_vad_raises(self) -> None:
        """Test unknown VAD raises ValueError."""
        with pytest.raises(ValueError, match="Unknown VAD"):
            create_vad("unknown_vad")

    @pytest.mark.parametrize("vad_id", ["webrtc_mode0", "webrtc_mode1", "webrtc_mode2", "webrtc_mode3"])
    def test_create_webrtc_vads(self, vad_id: str) -> None:
        """Test creating WebRTC VAD backends."""
        try:
            vad = create_vad(vad_id)
            assert hasattr(vad, "process_audio")
            assert hasattr(vad, "name")
            assert hasattr(vad, "config")
            assert vad_id in vad.name
        except ImportError:
            pytest.skip("webrtcvad not installed")

    def test_create_silero_vad(self) -> None:
        """Test creating Silero VAD backend."""
        try:
            vad = create_vad("silero")
            assert hasattr(vad, "process_audio")
            assert hasattr(vad, "name")
            assert hasattr(vad, "config")
            assert "silero" in vad.name
        except ImportError:
            pytest.skip("silero-vad or torch not installed")

    def test_create_tenvad(self) -> None:
        """Test creating TenVAD backend."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                vad = create_vad("tenvad")
            assert hasattr(vad, "process_audio")
            assert hasattr(vad, "name")
            assert hasattr(vad, "config")
            assert "tenvad" in vad.name
        except ImportError:
            pytest.skip("ten-vad not installed")

    @pytest.mark.parametrize("vad_id", ["javad_tiny", "javad_balanced", "javad_precise"])
    def test_create_javad_vads(self, vad_id: str) -> None:
        """Test creating JaVAD backends."""
        try:
            vad = create_vad(vad_id)
            assert hasattr(vad, "process_audio")
            assert hasattr(vad, "name")
            assert hasattr(vad, "config")
            assert vad_id == vad.name
        except ImportError:
            pytest.skip("javad not installed")

    def test_create_vad_returns_new_instance(self) -> None:
        """Test create_vad returns new instance each time (no caching)."""
        try:
            vad1 = create_vad("webrtc_mode3")
            vad2 = create_vad("webrtc_mode3")
            assert vad1 is not vad2
        except ImportError:
            pytest.skip("webrtcvad not installed")
