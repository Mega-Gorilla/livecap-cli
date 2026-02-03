"""Qwen3-ASR エンジンのユニットテスト"""
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.engines.metadata import EngineInfo, EngineMetadata


class TestQwen3ASREngineMetadata:
    """Qwen3-ASR エンジンメタデータのテスト"""

    def test_qwen3asr_engine_registered(self):
        """qwen3asr エンジンが EngineMetadata に登録されていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert info.id == "qwen3asr"
        assert info.display_name == "Qwen3-ASR 0.6B"
        assert info.module == ".qwen3asr_engine"
        assert info.class_name == "Qwen3ASREngine"

    def test_qwen3asr_supported_languages(self):
        """qwen3asr エンジンが 30 言語をサポートしていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        # 30 languages supported
        assert len(info.supported_languages) == 30
        # Key languages are supported
        assert "ja" in info.supported_languages  # Japanese
        assert "en" in info.supported_languages  # English
        assert "zh" in info.supported_languages  # Chinese
        assert "ko" in info.supported_languages  # Korean
        assert "yue" in info.supported_languages  # Cantonese

    def test_qwen3asr_device_support(self):
        """qwen3asr エンジンが CPU と CUDA をサポートしていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert "cpu" in info.device_support
        assert "cuda" in info.device_support

    def test_qwen3asr_default_params(self):
        """qwen3asr エンジンのデフォルトパラメータを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert info.default_params["model_name"] == "Qwen/Qwen3-ASR-0.6B"
        assert info.default_params["engine_id"] == "qwen3asr"

    def test_qwen3asr_in_available_engines(self):
        """qwen3asr が利用可能なエンジンリストに含まれていることを確認"""
        engines = EngineFactory.get_available_engines()
        assert "qwen3asr" in engines

    def test_qwen3asr_in_japanese_engines(self):
        """qwen3asr が日本語対応エンジンリストに含まれていることを確認"""
        ja_engines = EngineFactory.get_engines_for_language("ja")
        assert "qwen3asr" in ja_engines

    def test_qwen3asr_in_chinese_engines(self):
        """qwen3asr が中国語対応エンジンリストに含まれていることを確認"""
        zh_engines = EngineFactory.get_engines_for_language("zh")
        assert "qwen3asr" in zh_engines

    def test_qwen3asr_in_cantonese_engines(self):
        """qwen3asr が広東語対応エンジンリストに含まれていることを確認"""
        yue_engines = EngineFactory.get_engines_for_language("yue")
        assert "qwen3asr" in yue_engines


class TestQwen3ASREngineCreation:
    """Qwen3-ASR エンジン作成のテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_creation_with_defaults(self, mock_availability):
        """デフォルトパラメータでエンジンを作成できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        assert engine.engine_name == "qwen3asr"
        assert engine.model_name == "Qwen/Qwen3-ASR-0.6B"
        assert engine.language is None  # 自動検出
        assert engine.torch_device == "cpu"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_creation_with_custom_params(self, mock_availability):
        """カスタムパラメータでエンジンを作成できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(
            device="cpu",
            language="ja",
            model_name="Qwen/Qwen3-ASR-1.7B",
            engine_id="qwen3asr_large",
        )

        assert engine.engine_name == "qwen3asr_large"
        assert engine.model_name == "Qwen/Qwen3-ASR-1.7B"
        assert engine.language == "ja"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_get_supported_languages(self, mock_availability):
        """サポート言語リストを取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        languages = engine.get_supported_languages()

        assert len(languages) == 30
        assert "ja" in languages
        assert "en" in languages
        assert "zh" in languages

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_get_required_sample_rate(self, mock_availability):
        """要求サンプルレートが 16kHz であることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        assert engine.get_required_sample_rate() == 16000

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_get_engine_name_0_6b(self, mock_availability):
        """0.6B モデルのエンジン名を取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", model_name="Qwen/Qwen3-ASR-0.6B")
        assert engine.get_engine_name() == "Qwen3-ASR 0.6B"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_get_engine_name_1_7b(self, mock_availability):
        """1.7B モデルのエンジン名を取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", model_name="Qwen/Qwen3-ASR-1.7B")
        assert engine.get_engine_name() == "Qwen3-ASR 1.7B"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_engine_get_model_metadata(self, mock_availability):
        """モデルメタデータを取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        metadata = engine.get_model_metadata()

        assert metadata["name"] == "Qwen/Qwen3-ASR-0.6B"
        assert metadata["version"] == "0.6B"
        assert metadata["format"] == "transformers"


class TestQwen3ASRAvailabilityCheck:
    """Qwen3-ASR 可用性チェックのテスト"""

    def test_check_qwen_asr_availability_not_installed(self):
        """qwen-asr がインストールされていない場合 False を返すことを確認"""
        import livecap_cli.engines.qwen3asr_engine as module

        # Reset the global state
        original_value = module.QWEN_ASR_AVAILABLE
        module.QWEN_ASR_AVAILABLE = None

        try:
            # 非 frozen 環境でインポートが ImportError を発生させる場合をシミュレート
            def mock_import(name, *args, **kwargs):
                if name == "qwen_asr" or (args and args[0] == "qwen_asr"):
                    raise ImportError("No module named 'qwen_asr'")
                return original_import(name, *args, **kwargs)

            import builtins
            original_import = builtins.__import__

            with patch.object(builtins, "__import__", side_effect=mock_import):
                # Force re-check
                module.QWEN_ASR_AVAILABLE = None
                result = module.check_qwen_asr_availability()
                assert result is False
        finally:
            # Reset for other tests
            module.QWEN_ASR_AVAILABLE = original_value

    def test_check_qwen_asr_availability_caches_result(self):
        """可用性チェックの結果がキャッシュされることを確認"""
        import livecap_cli.engines.qwen3asr_engine as module

        # Save original value
        original_value = module.QWEN_ASR_AVAILABLE

        try:
            # Set cached value
            module.QWEN_ASR_AVAILABLE = True

            # Should return cached value without checking
            result = module.check_qwen_asr_availability()
            assert result is True
        finally:
            # Reset for other tests
            module.QWEN_ASR_AVAILABLE = original_value


class TestQwen3ASRDependencyCheck:
    """Qwen3-ASR 依存関係チェックのテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_check_dependencies_raises_when_not_available(self, mock_availability):
        """qwen-asr が利用不可の場合 ImportError が発生することを確認"""
        mock_availability.return_value = False

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        with pytest.raises(ImportError, match="qwen-asr is not installed"):
            engine._check_dependencies()

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_check_dependencies_passes_when_available(self, mock_availability):
        """qwen-asr が利用可能な場合、依存関係チェックが成功することを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        # Should not raise
        engine._check_dependencies()


class TestQwen3ASRTranscribeValidation:
    """Qwen3-ASR transcribe メソッドのバリデーションテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen_asr_availability")
    def test_transcribe_raises_when_not_initialized(self, mock_availability):
        """初期化前に transcribe を呼び出すと RuntimeError が発生することを確認"""
        mock_availability.return_value = True

        import numpy as np
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)
