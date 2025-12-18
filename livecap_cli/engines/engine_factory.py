"""音声認識エンジンファクトリー"""
from typing import Optional, Dict, Any
import logging

from .base_engine import BaseEngine
from livecap_cli.i18n import translate

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
            module = importlib.import_module(module_name, package="livecap_cli.engines")
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
    def create_engine(
        cls,
        engine_type: str,
        device: Optional[str] = None,
        **engine_options,
    ) -> BaseEngine:
        """
        指定されたタイプのエンジンを作成

        Args:
            engine_type: エンジンタイプ（必須）
                利用可能なエンジン: reazonspeech, parakeet, parakeet_ja,
                canary, voxtral, whispers2t_base, whispers2t_tiny,
                whispers2t_small, whispers2t_medium, whispers2t_large_v3
            device: 使用するデバイス（cuda/cpu/None=auto）
            **engine_options: エンジン固有のパラメータ（default_paramsを上書き）

        Returns:
            BaseEngineのインスタンス

        Raises:
            ValueError: 不明なエンジンタイプまたは"auto"が指定された場合

        Examples:
            # 基本的な使用法
            engine = EngineFactory.create_engine("reazonspeech")

            # デバイス指定
            engine = EngineFactory.create_engine("whispers2t_base", device="cuda")

            # パラメータ上書き
            engine = EngineFactory.create_engine(
                "reazonspeech",
                use_int8=True,
                num_threads=8
            )

            # 多言語エンジンの言語指定
            engine = EngineFactory.create_engine(
                "canary",
                language="de"
            )
        """
        # "auto"は非推奨
        if engine_type == "auto":
            raise ValueError(
                "engine_type='auto' is deprecated. "
                "Use EngineMetadata.get_engines_for_language() to discover available engines, "
                "then specify the engine explicitly."
            )

        # メタデータから情報を取得
        metadata = EngineMetadata.get(engine_type)
        if metadata is None:
            available = list(cls._get_engines().keys())
            raise ValueError(
                f"Unknown engine type: {engine_type}. "
                f"Available engines: {available}"
            )

        # default_params と engine_options をマージ
        # engine_options が優先される
        params = {**metadata.default_params, **engine_options}

        # Parakeetの場合、engine_nameパラメータを追加
        if engine_type in ("parakeet", "parakeet_ja"):
            params["engine_name"] = engine_type

        # エンジンクラスを遅延インポート
        engine_class = cls._get_engine_class(engine_type)

        # エンジンのインスタンスを作成
        return engine_class(device=device, **params)

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
    def get_engine_info(cls, engine_type: str) -> Optional[Dict[str, Any]]:
        """
        特定のエンジンの情報を取得

        Args:
            engine_type: エンジンタイプ

        Returns:
            エンジン情報またはNone
        """
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
            result = {
                "name": name,
                "description": description,
                "supported_languages": engine_info.supported_languages,
                "default_params": engine_info.default_params,
            }
            # available_model_sizes が設定されている場合のみ追加
            if engine_info.available_model_sizes:
                result["available_model_sizes"] = engine_info.available_model_sizes
            return result
        return None

    @classmethod
    def get_engines_for_language(cls, language_code: str) -> Dict[str, Dict[str, Any]]:
        """
        特定の言語に対応したエンジンのリストを取得

        Args:
            language_code: 言語コード（例: "ja", "en"）

        Returns:
            言語に対応したエンジン情報の辞書
        """
        result: Dict[str, Dict[str, Any]] = {}

        for engine_key in EngineMetadata.get_engines_for_language(language_code):
            info = cls.get_engine_info(engine_key)
            if info:
                result[engine_key] = info

        return result
