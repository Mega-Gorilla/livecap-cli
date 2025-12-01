"""Tests for VAD optimized presets."""

from __future__ import annotations

import pytest

from livecap_core.vad import VADConfig
from livecap_core.vad.presets import (
    VAD_OPTIMIZED_PRESETS,
    get_available_presets,
    get_best_vad_for_language,
    get_optimized_preset,
)


class TestVADOptimizedPresets:
    """Test VAD_OPTIMIZED_PRESETS structure."""

    def test_presets_have_required_vad_types(self):
        """All expected VAD types should be present."""
        expected_vads = {"silero", "tenvad", "webrtc"}
        assert set(VAD_OPTIMIZED_PRESETS.keys()) == expected_vads

    def test_presets_have_both_languages(self):
        """Each VAD should have both JA and EN presets."""
        for vad_type, languages in VAD_OPTIMIZED_PRESETS.items():
            assert "ja" in languages, f"{vad_type} missing 'ja' preset"
            assert "en" in languages, f"{vad_type} missing 'en' preset"

    def test_preset_structure(self):
        """Each preset should have vad_config and metadata."""
        for vad_type, languages in VAD_OPTIMIZED_PRESETS.items():
            for lang, preset in languages.items():
                assert "vad_config" in preset, f"{vad_type}/{lang} missing vad_config"
                assert "metadata" in preset, f"{vad_type}/{lang} missing metadata"
                assert "score" in preset["metadata"], f"{vad_type}/{lang} missing score"

    def test_vad_config_can_be_created(self):
        """VADConfig should be creatable from preset vad_config."""
        for vad_type, languages in VAD_OPTIMIZED_PRESETS.items():
            for lang, preset in languages.items():
                config = VADConfig.from_dict(preset["vad_config"])
                assert isinstance(config, VADConfig)
                assert config.min_speech_ms > 0
                assert config.min_silence_ms > 0
                assert config.speech_pad_ms >= 0


class TestGetOptimizedPreset:
    """Test get_optimized_preset function."""

    def test_get_existing_preset(self):
        """Should return preset for valid vad_type and language."""
        preset = get_optimized_preset("silero", "ja")
        assert preset is not None
        assert "vad_config" in preset
        assert preset["vad_config"]["threshold"] == pytest.approx(0.294, rel=0.01)

    def test_get_nonexistent_vad_type(self):
        """Should return None for unknown VAD type."""
        preset = get_optimized_preset("unknown_vad", "ja")
        assert preset is None

    def test_get_nonexistent_language(self):
        """Should return None for unknown language."""
        preset = get_optimized_preset("silero", "zh")
        assert preset is None

    def test_get_all_combinations(self):
        """Should return presets for all known combinations."""
        for vad_type in ["silero", "tenvad", "webrtc"]:
            for lang in ["ja", "en"]:
                preset = get_optimized_preset(vad_type, lang)
                assert preset is not None, f"Missing preset: {vad_type}/{lang}"


class TestGetAvailablePresets:
    """Test get_available_presets function."""

    def test_returns_all_combinations(self):
        """Should return all 6 combinations (3 VADs x 2 languages)."""
        presets = get_available_presets()
        assert len(presets) == 6

    def test_returns_tuples(self):
        """Should return list of (vad_type, language) tuples."""
        presets = get_available_presets()
        for item in presets:
            assert isinstance(item, tuple)
            assert len(item) == 2
            vad_type, language = item
            assert isinstance(vad_type, str)
            assert isinstance(language, str)

    def test_contains_expected_combinations(self):
        """Should contain all expected combinations."""
        presets = get_available_presets()
        assert ("silero", "ja") in presets
        assert ("silero", "en") in presets
        assert ("webrtc", "ja") in presets
        assert ("tenvad", "en") in presets


class TestGetBestVadForLanguage:
    """Test get_best_vad_for_language function."""

    def test_best_vad_for_japanese(self):
        """TenVAD should be best for Japanese (lowest CER)."""
        result = get_best_vad_for_language("ja")
        assert result is not None
        vad_type, preset = result
        assert vad_type == "tenvad"
        assert preset["metadata"]["score"] == pytest.approx(0.072, rel=0.01)

    def test_best_vad_for_english(self):
        """WebRTC should be best for English (lowest WER)."""
        result = get_best_vad_for_language("en")
        assert result is not None
        vad_type, preset = result
        assert vad_type == "webrtc"
        assert preset["metadata"]["score"] == pytest.approx(0.033, rel=0.01)

    def test_unknown_language(self):
        """Should return None for unknown language."""
        result = get_best_vad_for_language("zh")
        assert result is None


class TestPresetMetadata:
    """Test preset metadata correctness."""

    def test_ja_tenvad_is_best(self):
        """TenVAD should have lowest score for JA."""
        silero = get_optimized_preset("silero", "ja")
        tenvad = get_optimized_preset("tenvad", "ja")
        webrtc = get_optimized_preset("webrtc", "ja")

        assert tenvad["metadata"]["score"] < silero["metadata"]["score"]
        assert tenvad["metadata"]["score"] < webrtc["metadata"]["score"]

    def test_en_webrtc_is_best(self):
        """WebRTC should have lowest score for EN."""
        silero = get_optimized_preset("silero", "en")
        tenvad = get_optimized_preset("tenvad", "en")
        webrtc = get_optimized_preset("webrtc", "en")

        assert webrtc["metadata"]["score"] < tenvad["metadata"]["score"]
        assert webrtc["metadata"]["score"] < silero["metadata"]["score"]

    def test_all_scores_are_valid(self):
        """All scores should be between 0 and 1."""
        for vad_type, languages in VAD_OPTIMIZED_PRESETS.items():
            for lang, preset in languages.items():
                score = preset["metadata"]["score"]
                assert 0 < score < 1, f"{vad_type}/{lang} has invalid score: {score}"
