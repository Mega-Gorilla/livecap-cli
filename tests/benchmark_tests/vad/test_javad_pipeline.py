"""Unit tests for JaVADPipeline."""

from __future__ import annotations

import pytest


class TestJaVADPipeline:
    """JaVADPipeline テスト"""

    @pytest.fixture
    def javad_pipeline(self):
        """JaVADPipeline インスタンス"""
        try:
            from benchmarks.vad.backends import JaVADPipeline

            return JaVADPipeline(model="balanced")
        except ImportError:
            pytest.skip("javad not installed")

    def test_name(self, javad_pipeline):
        """バックエンド名"""
        assert javad_pipeline.name == "javad_balanced"

    def test_window_size_ms(self, javad_pipeline):
        """ウィンドウサイズ（balanced = 1920ms）"""
        assert javad_pipeline.window_size_ms == 1920

    def test_valid_models(self):
        """有効なモデル"""
        try:
            from benchmarks.vad.backends import JaVADPipeline
        except ImportError:
            pytest.skip("javad not installed")

        assert JaVADPipeline.VALID_MODELS == ("tiny", "balanced", "precise")
        assert JaVADPipeline.WINDOW_SIZES == {
            "tiny": 640,
            "balanced": 1920,
            "precise": 3840,
        }

    def test_invalid_model_raises(self):
        """無効なモデルでエラー"""
        try:
            from benchmarks.vad.backends import JaVADPipeline
        except ImportError:
            pytest.skip("javad not installed")

        with pytest.raises(ValueError, match="model must be one of"):
            JaVADPipeline(model="invalid")


class TestJaVADPipelineModels:
    """JaVADPipeline モデル別テスト"""

    @pytest.fixture
    def tiny_pipeline(self):
        """tiny モデル"""
        try:
            from benchmarks.vad.backends import JaVADPipeline

            return JaVADPipeline(model="tiny")
        except ImportError:
            pytest.skip("javad not installed")

    @pytest.fixture
    def precise_pipeline(self):
        """precise モデル"""
        try:
            from benchmarks.vad.backends import JaVADPipeline

            return JaVADPipeline(model="precise")
        except ImportError:
            pytest.skip("javad not installed")

    def test_tiny_model(self, tiny_pipeline):
        """tiny モデル"""
        assert tiny_pipeline.name == "javad_tiny"
        assert tiny_pipeline.window_size_ms == 640

    def test_precise_model(self, precise_pipeline):
        """precise モデル"""
        assert precise_pipeline.name == "javad_precise"
        assert precise_pipeline.window_size_ms == 3840
