"""
エンジンメタデータの定義

移動元：gui/dialogs/settings/constants.py: ENGINE_METADATA
このモジュールは、ASRエンジンのメタデータを一元管理します。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class EngineInfo:
    """エンジン情報"""
    id: str
    display_name: str
    description: str
    supported_languages: List[str]
    requires_download: bool = False
    model_size: Optional[str] = None
    device_support: List[str] = field(default_factory=lambda: ["cpu"])
    streaming: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    module: Optional[str] = None  # エンジンモジュールのパス
    class_name: Optional[str] = None  # エンジンクラス名


class EngineMetadata:
    """
    エンジンメタデータの中央管理

    移動元：gui/dialogs/settings/constants.py: ENGINE_METADATA
    参照元：12箇所（engine_factory, lazy_loader, basic_tab等）
    """

    _ENGINES: Dict[str, EngineInfo] = {
        "reazonspeech": EngineInfo(
            id="reazonspeech",
            display_name="ReazonSpeech K2 v2",
            description="Japanese-specialized high-accuracy ASR engine optimized for real-time transcription",
            supported_languages=["ja"],
            requires_download=True,
            model_size="159.34MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".reazonspeech_engine",
            class_name="ReazonSpeechEngine",
            default_params={
                "temperature": 0.0,
                "beam_size": 10,
            }
        ),
        "parakeet": EngineInfo(
            id="parakeet",
            display_name="NVIDIA Parakeet TDT 0.6B v2",
            description="English-optimized high-accuracy ASR with WER 6.05%",
            supported_languages=["en"],
            requires_download=True,
            model_size="1.2GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".parakeet_engine",
            class_name="ParakeetEngine",
        ),
        "parakeet_ja": EngineInfo(
            id="parakeet_ja",
            display_name="NVIDIA Parakeet TDT CTC 0.6B JA",
            description="Japanese-specialized high-accuracy streaming ASR model",
            supported_languages=["ja"],
            requires_download=True,
            model_size="600MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".parakeet_engine",
            class_name="ParakeetEngine",
            default_params={
                "model_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
            }
        ),
        "canary": EngineInfo(
            id="canary",
            display_name="NVIDIA Canary 1B Flash",
            description="Fast multilingual ASR supporting EN, DE, FR, ES",
            supported_languages=["en", "de", "fr", "es"],
            requires_download=True,
            model_size="1.5GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".canary_engine",
            class_name="CanaryEngine",
        ),
        "voxtral": EngineInfo(
            id="voxtral",
            display_name="MistralAI Voxtral Mini 3B",
            description="Advanced multilingual ASR with auto language detection",
            supported_languages=["en", "es", "fr", "pt", "hi", "de", "nl", "it"],
            requires_download=True,
            model_size="3GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".voxtral_engine",
            class_name="VoxtralEngine",
            default_params={
                "temperature": 0.0,
                "do_sample": False,
                "max_new_tokens": 448,
            }
        ),
        # Whisper S2T variants
        "whispers2t_base": EngineInfo(
            id="whispers2t_base",
            display_name="WhisperS2T Base",
            description="Lightweight multilingual ASR model with good balance",
            supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
            requires_download=True,
            model_size="74MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            default_params={
                "model_size": "base",
            }
        ),
        "whispers2t_tiny": EngineInfo(
            id="whispers2t_tiny",
            display_name="WhisperS2T Tiny",
            description="Ultra-lightweight multilingual ASR model for fastest processing",
            supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
            requires_download=True,
            model_size="39MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            default_params={
                "model_size": "tiny",
            }
        ),
        "whispers2t_small": EngineInfo(
            id="whispers2t_small",
            display_name="WhisperS2T Small",
            description="Small multilingual ASR model with improved accuracy",
            supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
            requires_download=True,
            model_size="244MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            default_params={
                "model_size": "small",
            }
        ),
        "whispers2t_medium": EngineInfo(
            id="whispers2t_medium",
            display_name="WhisperS2T Medium",
            description="Medium multilingual ASR model with higher accuracy",
            supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
            requires_download=True,
            model_size="769MB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            default_params={
                "model_size": "medium",
            }
        ),
        "whispers2t_large_v3": EngineInfo(
            id="whispers2t_large_v3",
            display_name="WhisperS2T Large-v3",
            description="Large multilingual ASR model with best accuracy",
            supported_languages=["ja", "en", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "ru", "ar", "pt", "it", "hi"],
            requires_download=True,
            model_size="1.55GB",
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            default_params={
                "model_size": "large-v3",
            }
        ),
    }

    @classmethod
    def get(cls, engine_id: str) -> Optional[EngineInfo]:
        """
        エンジン情報を取得

        Args:
            engine_id: エンジンID

        Returns:
            EngineInfo オブジェクト、またはNone
        """
        return cls._ENGINES.get(engine_id)

    @classmethod
    def get_all(cls) -> Dict[str, EngineInfo]:
        """
        全エンジン情報を取得

        Returns:
            エンジンID -> EngineInfo の辞書
        """
        return cls._ENGINES.copy()

    @classmethod
    def get_display_name(cls, engine_id: str) -> str:
        """
        表示名を取得

        Args:
            engine_id: エンジンID

        Returns:
            表示名（見つからない場合はエンジンIDを返す）
        """
        info = cls.get(engine_id)
        return info.display_name if info else engine_id

    @classmethod
    def get_engines_for_language(cls, lang_code: str) -> List[str]:
        """
        指定言語をサポートするエンジンリストを取得

        Args:
            lang_code: 言語コード

        Returns:
            エンジンIDのリスト
        """
        # 言語コードを正規化
        from livecap_core.languages import Languages  # type: ignore

        normalized = Languages.normalize(lang_code) or lang_code
        if not normalized:
            return []

        result = []
        for engine_id, info in cls._ENGINES.items():
            if normalized in info.supported_languages:
                result.append(engine_id)
        return result

    @classmethod
    def get_module_info(cls, engine_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        エンジンのモジュール情報を取得

        Args:
            engine_id: エンジンID

        Returns:
            (module_path, class_name) のタプル
        """
        info = cls.get(engine_id)
        if info:
            return info.module, info.class_name
        return None, None
