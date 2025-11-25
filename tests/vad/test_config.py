"""Unit tests for VADConfig."""

import pytest

from livecap_core.vad import VADConfig


class TestVADConfigBasics:
    """VADConfig 基本機能テスト"""

    def test_default_values(self):
        """デフォルト値"""
        config = VADConfig()
        assert config.threshold == 0.5
        assert config.neg_threshold is None
        assert config.min_speech_ms == 250
        assert config.min_silence_ms == 100
        assert config.speech_pad_ms == 100
        assert config.max_speech_ms == 0
        assert config.interim_min_duration_ms == 2000
        assert config.interim_interval_ms == 1000

    def test_custom_values(self):
        """カスタム値"""
        config = VADConfig(
            threshold=0.6,
            neg_threshold=0.4,
            min_speech_ms=300,
            min_silence_ms=150,
            speech_pad_ms=50,
            max_speech_ms=10000,
            interim_min_duration_ms=3000,
            interim_interval_ms=500,
        )
        assert config.threshold == 0.6
        assert config.neg_threshold == 0.4
        assert config.min_speech_ms == 300
        assert config.min_silence_ms == 150
        assert config.speech_pad_ms == 50
        assert config.max_speech_ms == 10000
        assert config.interim_min_duration_ms == 3000
        assert config.interim_interval_ms == 500

    def test_frozen(self):
        """frozen dataclass"""
        config = VADConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.threshold = 0.7


class TestVADConfigNegThreshold:
    """VADConfig neg_threshold テスト"""

    def test_get_neg_threshold_default(self):
        """neg_threshold がNoneの場合"""
        config = VADConfig(threshold=0.5)
        assert config.get_neg_threshold() == 0.35  # 0.5 - 0.15

    def test_get_neg_threshold_explicit(self):
        """neg_threshold が明示的に設定されている場合"""
        config = VADConfig(threshold=0.5, neg_threshold=0.3)
        assert config.get_neg_threshold() == 0.3

    def test_get_neg_threshold_clamped(self):
        """neg_threshold が負になる場合は0にクランプ"""
        config = VADConfig(threshold=0.1)
        assert config.get_neg_threshold() == 0.0  # max(0, 0.1 - 0.15)


class TestVADConfigFromDict:
    """VADConfig from_dict テスト"""

    def test_from_dict_empty(self):
        """空の辞書からデフォルト設定を作成"""
        config = VADConfig.from_dict({})
        assert config.threshold == 0.5
        assert config.min_speech_ms == 250

    def test_from_dict_partial(self):
        """一部の設定を指定"""
        config = VADConfig.from_dict({"threshold": 0.7, "min_speech_ms": 400})
        assert config.threshold == 0.7
        assert config.min_speech_ms == 400
        assert config.min_silence_ms == 100  # デフォルト

    def test_from_dict_full(self):
        """全設定を指定"""
        config = VADConfig.from_dict(
            {
                "threshold": 0.6,
                "neg_threshold": 0.4,
                "min_speech_ms": 300,
                "min_silence_ms": 150,
                "speech_pad_ms": 50,
                "max_speech_ms": 10000,
                "interim_min_duration_ms": 3000,
                "interim_interval_ms": 500,
            }
        )
        assert config.threshold == 0.6
        assert config.neg_threshold == 0.4
        assert config.max_speech_ms == 10000


class TestVADConfigToDict:
    """VADConfig to_dict テスト"""

    def test_to_dict_default(self):
        """デフォルト設定を辞書に変換"""
        config = VADConfig()
        d = config.to_dict()

        assert d["threshold"] == 0.5
        assert d["neg_threshold"] is None
        assert d["min_speech_ms"] == 250
        assert d["min_silence_ms"] == 100
        assert d["speech_pad_ms"] == 100
        assert d["max_speech_ms"] == 0

    def test_to_dict_custom(self):
        """カスタム設定を辞書に変換"""
        config = VADConfig(threshold=0.6, min_speech_ms=300)
        d = config.to_dict()

        assert d["threshold"] == 0.6
        assert d["min_speech_ms"] == 300

    def test_roundtrip(self):
        """from_dict -> to_dict のラウンドトリップ"""
        original = {
            "threshold": 0.6,
            "neg_threshold": 0.4,
            "min_speech_ms": 300,
            "min_silence_ms": 150,
            "speech_pad_ms": 50,
            "max_speech_ms": 10000,
            "interim_min_duration_ms": 3000,
            "interim_interval_ms": 500,
        }
        config = VADConfig.from_dict(original)
        result = config.to_dict()

        assert result == original
