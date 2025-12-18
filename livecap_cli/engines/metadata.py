"""
エンジンメタデータの定義

移動元：gui/dialogs/settings/constants.py: ENGINE_METADATA
このモジュールは、ASRエンジンのメタデータを一元管理します。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import langcodes

from .whisper_languages import WHISPER_LANGUAGES


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
    available_model_sizes: Optional[List[str]] = None  # 選択可能なモデルサイズ一覧


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
                # カテゴリA パラメータ（エンジンの__init__で使用）
                "use_int8": False,
                "num_threads": 4,
                "decoding_method": "greedy_search",
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
            default_params={
                "model_name": "nvidia/parakeet-tdt-0.6b-v2",
                "decoding_strategy": "greedy",
            }
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
                "decoding_strategy": "greedy",
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
            default_params={
                "model_name": "nvidia/canary-1b-flash",
                "beam_size": 1,
            }
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
                "model_name": "mistralai/Voxtral-Mini-3B-2507",
            }
        ),
        # WhisperS2T - Unified multilingual ASR engine
        "whispers2t": EngineInfo(
            id="whispers2t",
            display_name="WhisperS2T",
            description="Multilingual ASR model with selectable model sizes (tiny to large-v3-turbo)",
            supported_languages=list(WHISPER_LANGUAGES),  # 99 languages
            requires_download=True,
            model_size=None,  # Multiple sizes available
            device_support=["cpu", "cuda"],
            streaming=True,
            module=".whispers2t_engine",
            class_name="WhisperS2TEngine",
            available_model_sizes=[
                # Standard models
                "tiny", "base", "small", "medium",
                # Large models
                "large-v1", "large-v2", "large-v3",
                # High-speed models
                "large-v3-turbo", "distil-large-v3",
            ],
            default_params={
                "model_size": "large-v3",  # Benchmark compatibility (was whispers2t_large_v3)
                "compute_type": "auto",
                "batch_size": 24,
                "use_vad": True,
            },
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
            lang_code: 言語コード（"ja", "zh-CN", "en" など）

        Returns:
            エンジンIDのリスト

        Note:
            BCP-47 形式（zh-CN, zh-TW, pt-BR など）は ISO 639-1（zh, pt）に
            変換してから比較する。これにより WhisperS2T の100言語サポートが
            正しく機能する。
        """
        # BCP-47 → ISO 639-1 変換（自己完結）
        iso_code = cls.to_iso639_1(lang_code)

        result = []
        for engine_id, info in cls._ENGINES.items():
            if iso_code in info.supported_languages:
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

    @classmethod
    def to_iso639_1(cls, code: str) -> str:
        """
        BCP-47 言語コードを ISO 639-1 に変換

        Args:
            code: 言語コード（"ja", "zh-CN", "ZH-TW" など）

        Returns:
            ISO 639-1 言語コード（"ja", "zh" など）

        Raises:
            langcodes.LanguageTagError: 無効な言語コード形式の場合

        Examples:
            >>> EngineMetadata.to_iso639_1("zh-CN")
            'zh'
            >>> EngineMetadata.to_iso639_1("pt-BR")
            'pt'
            >>> EngineMetadata.to_iso639_1("ja")
            'ja'
            >>> EngineMetadata.to_iso639_1("ZH-CN")  # 大文字も自動正規化
            'zh'
            >>> EngineMetadata.to_iso639_1("yue")  # ISO 639-3 はパススルー
            'yue'
            >>> EngineMetadata.to_iso639_1("auto")  # パススルー
            'auto'
        """
        return langcodes.Language.get(code).language
