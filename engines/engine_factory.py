"""音声認識エンジンファクトリー"""
from typing import Optional, Dict, Any
import logging
from copy import deepcopy

from .base_engine import BaseEngine
from config.core_config_builder import build_core_config
from livecap_core.i18n import translate

logger = logging.getLogger(__name__)

# エンジンメタデータをインポート（新しい中央定義から）
from .metadata import EngineMetadata


class EngineFactory:
    """音声認識エンジンを作成するファクトリークラス"""

    # 利用可能なエンジン（EngineMetadataから構築）
    _ENGINES = None  # キャッシュ

    @classmethod
    def _get_engines(cls):
        """EngineMetadataからENGINES辞書を構築（キャッシュ付き）"""
        if cls._ENGINES is None:
            cls._ENGINES = {}
            for engine_id, info in EngineMetadata.get_all().items():
                cls._ENGINES[engine_id] = {
                    "module": info.module,
                    "class_name": info.class_name,
                    "name": info.display_name,
                    "description": info.description,
                    "supported_languages": info.supported_languages,
                    "model_name": getattr(info, 'model_name', None),  # パラメータがある場合
                    "model_size": info.default_params.get('model_size') if info.default_params else None,  # WhisperS2T用
                }
        return cls._ENGINES

    @classmethod
    def ENGINES(cls):
        """後方互換性のためのクラスメソッド"""
        return cls._get_engines()
    
    @classmethod
    def _get_engine_class(cls, engine_type: str):
        """エンジンクラスを遅延インポートで取得"""
        try:
            engine_info = cls._get_engines()[engine_type]
            module_name = engine_info["module"]
            class_name = engine_info["class_name"]
            
            # 動的インポート
            import importlib
            module = importlib.import_module(module_name, package="engines")
            engine_class = getattr(module, class_name)
            
            return engine_class
            
        except (ImportError, AttributeError, KeyError) as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(
                "エンジンクラス '%s' の動的インポートに失敗しました: %s\n%s",
                engine_type,
                e,
                error_details,
            )
            # エラーを再送出し、呼び出し元で処理できるようにする
            raise ValueError(f"Failed to load engine class for '{engine_type}'. Check logs for details.") from e

    @classmethod
    def _prepare_config(cls, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalise user/GUI config into the core schema."""
        return build_core_config(config)
    
    @classmethod
    def _configure_engine_specific_settings(cls, engine_type: str, engine_info: Dict, config: Dict) -> None:
        """エンジン固有の設定を構成"""
        # WhisperS2Tの場合、モデルサイズを設定に追加
        if engine_type.startswith("whispers2t_"):
            if "whispers2t" not in config:
                config["whispers2t"] = {}
            config["whispers2t"]["model_size"] = engine_info.get("model_size", "base")
        
        # Parakeetの場合、モデル名を設定に追加
        elif engine_type.startswith("parakeet"):
            # parakeet_jaの場合は別の設定キーを使用
            if engine_type == "parakeet_ja":
                if "parakeet_ja" not in config:
                    config["parakeet_ja"] = {}
                # 日本語モデルを設定
                config["parakeet_ja"]["model_name"] = engine_info.get("model_name", "nvidia/parakeet-tdt_ctc-0.6b-ja")
            else:
                if "parakeet" not in config:
                    config["parakeet"] = {}
                # 英語モデルを設定
                config["parakeet"]["model_name"] = engine_info.get("model_name", "nvidia/parakeet-tdt-0.6b-v3")
        
        # Voxtralの場合、モデル名を設定に追加
        elif engine_type == "voxtral":
            if "voxtral" not in config:
                config["voxtral"] = {}
            config["voxtral"]["model_name"] = "mistralai/Voxtral-Mini-3B-2507"
    
    @classmethod
    def create_engine(cls, engine_type: str, device: Optional[str] = None, 
                     config: Optional[Dict[str, Any]] = None) -> BaseEngine:
        """
        指定されたタイプのエンジンを作成
        
        Args:
            engine_type: エンジンタイプ（"reazonspeech" or "parakeet"）
            device: 使用するデバイス（cuda/cpu/null）
            config: 設定辞書
            
            Returns:
                BaseEngineのインスタンス
                
            Raises:
                ValueError: 不明なエンジンタイプ
        """
        prepared_config = cls._prepare_config(config)

        resolved_engine = engine_type
        if resolved_engine == "auto":
            resolved_engine = cls.resolve_auto_engine(prepared_config)
            input_language = prepared_config.get("transcription", {}).get("input_language", "ja")
            logger.info("Auto-selected engine '%s' for language '%s'", resolved_engine, input_language)

        if resolved_engine not in cls.ENGINES():
            raise ValueError(
                f"Unknown engine type: {resolved_engine}. "
                f"Available engines: {list(cls.ENGINES().keys())}"
            )

        engine_info = cls.ENGINES()[resolved_engine]
        engine_config = deepcopy(prepared_config)
        cls._configure_engine_specific_settings(resolved_engine, engine_info, engine_config)

        if resolved_engine == "reazonspeech":
            reazonspeech_config = engine_config.get("transcription", {}).get("reazonspeech_config", {})
            engine_specific_config = {
                "use_int8": reazonspeech_config.get("use_int8", False),
                "num_threads": reazonspeech_config.get("num_threads", 4),
                "decoding_method": reazonspeech_config.get("decoding_method", "greedy_search"),
            }
        else:
            engine_specific_config = engine_config

        # エンジンクラスを遅延インポート
        engine_class = cls._get_engine_class(resolved_engine)
        
        # エンジンのインスタンスを作成
        return engine_class(
            device=device,
            config=engine_specific_config if resolved_engine == "reazonspeech" else engine_config,
        )
        
    @classmethod
    def get_available_engines(cls) -> Dict[str, Dict[str, str]]:
        """
        利用可能なエンジンの情報を取得
        
        Returns:
            エンジン情報の辞書
        """
        engines: Dict[str, Dict[str, str]] = {}
        for engine_id, info in EngineMetadata.get_all().items():
            engines[engine_id] = {
                "name": translate(
                    f"engines.{engine_id}.name",
                    default=info.display_name,
                ),
                "description": translate(
                    f"engines.{engine_id}.description",
                    default=info.description,
                ),
            }
        return engines
        
    @classmethod
    def get_engine_info(cls, engine_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        特定のエンジンの情報を取得
        
        Args:
            engine_type: エンジンタイプ
            config: 設定辞書（表示名などを上書きする場合に使用）
            
        Returns:
            エンジン情報またはNone
        """
        # まずEngineMetadataから取得を試みる
        prepared_config = cls._prepare_config(config) if config is not None else None

        engine_info = EngineMetadata.get(engine_type)
        if engine_info:
            name = translate(
                f"engines.{engine_type}.name",
                default=engine_info.display_name,
            )
            description = translate(
                f"engines.{engine_type}.description",
                default=engine_info.description,
            )

            if prepared_config:
                override = prepared_config.get("engines", {}).get(engine_type, {})
                name = override.get("display_name", name)
                description = override.get("description", description)

            return {
                "name": name,
                "description": description,
                "supported_languages": engine_info.supported_languages
            }
        
        fallback_registry = cls.ENGINES()
        if engine_type in fallback_registry:
            engine_data = fallback_registry[engine_type]
            display_name = engine_data.get("display_name", engine_type)
            description = engine_data.get("description", "")
            if prepared_config:
                override = prepared_config.get("engines", {}).get(engine_type, {})
                display_name = override.get("display_name", display_name)
                description = override.get("description", description)
            return {
                "name": display_name,
                "description": description,
                "supported_languages": []
            }
        return None
    
    @classmethod
    def get_engines_for_language(cls, language_code: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        特定の言語に対応したエンジンのリストを取得
        
        Args:
            language_code: 言語コード（例: "ja", "en"）
            config: 設定辞書（任意）
            
        Returns:
            言語に対応したエンジン情報の辞書
        """
        prepared_config = None
        if config is not None:
            prepared_config = cls._prepare_config(config)

        result: Dict[str, Dict[str, Any]] = {}

        for engine_key, metadata in EngineMetadata.get_all().items():
            if language_code in metadata.supported_languages:
                info = cls.get_engine_info(engine_key, config=prepared_config)
                if info:
                    result[engine_key] = info

        if prepared_config:
            language_engines = prepared_config.get("transcription", {}).get("language_engines", {})
            for engine_key in language_engines.values():
                metadata = EngineMetadata.get(engine_key)
                if metadata:
                    info = cls.get_engine_info(engine_key, config=prepared_config)
                    if info:
                        result.setdefault(engine_key, info)

        return result
    
    @classmethod
    def get_default_engine_for_language(cls, language_code: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        特定の言語のデフォルトエンジンを取得
        
        Args:
            language_code: 言語コード（例: "ja", "en"）
            config: 設定辞書
            
        Returns:
            デフォルトエンジンタイプ
        """
        if config:
            language_engines = config.get("transcription", {}).get("language_engines", {})
            return language_engines.get(language_code, 
                                       language_engines.get("default", "whispers2t_base"))
        
        # 設定がない場合のデフォルト
        default_mapping = {
            "ja": "reazonspeech",
            "en": "canary",
            "de": "canary",
            "fr": "canary",
            "es": "canary",
        }
        return default_mapping.get(language_code, "whispers2t_base")
    
    @classmethod
    def resolve_auto_engine(cls, config: Dict[str, Any]) -> str:
        """
        'auto'エンジンタイプを実際のエンジンタイプに解決
        
        Args:
            config: 設定辞書
            
        Returns:
            実際のエンジンタイプ
        """
        input_language = config.get("transcription", {}).get("input_language", "ja")
        language_engines = config.get("transcription", {}).get("language_engines", {})
        
        # 言語に対応するエンジンを選択
        engine_type = language_engines.get(input_language, 
                                          language_engines.get("default", "whispers2t_base"))
        return engine_type
