"""
翻訳エンジンのメタデータ管理

翻訳エンジンの登録情報とファクトリー生成用メタデータを管理する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TranslatorInfo:
    """翻訳エンジンのメタデータ"""

    translator_id: str
    display_name: str
    description: str
    module: str  # e.g., ".impl.google"
    class_name: str  # e.g., "GoogleTranslator"
    supported_pairs: List[Tuple[str, str]]  # [("ja", "en"), ("en", "ja"), ...]
    requires_model_load: bool = False  # モデルロードが必要か
    requires_gpu: bool = False  # GPU 必須か
    default_context_sentences: int = 2  # デフォルト文脈数
    default_params: Dict[str, Any] = field(default_factory=dict)


class TranslatorMetadata:
    """翻訳エンジンのメタデータ管理"""

    _TRANSLATORS: Dict[str, TranslatorInfo] = {
        "google": TranslatorInfo(
            translator_id="google",
            display_name="Google Translate",
            description="Google Cloud Translation API (via deep-translator)",
            module=".impl.google",
            class_name="GoogleTranslator",
            supported_pairs=[],  # 動的に取得（ほぼ全言語対応）
            requires_model_load=False,
            requires_gpu=False,
            default_context_sentences=2,
        ),
        "opus_mt": TranslatorInfo(
            translator_id="opus_mt",
            display_name="OPUS-MT",
            description="Helsinki-NLP OPUS-MT models via CTranslate2",
            module=".impl.opus_mt",
            class_name="OpusMTTranslator",
            supported_pairs=[("ja", "en"), ("en", "ja")],  # Phase 1: ja↔en のみ
            requires_model_load=True,
            requires_gpu=False,
            default_context_sentences=0,  # Issue #190: 文脈抽出が不安定なため無効化
            default_params={"device": "cpu", "compute_type": "int8"},
        ),
        "riva_instruct": TranslatorInfo(
            translator_id="riva_instruct",
            display_name="Riva Translate 4B Instruct",
            description="NVIDIA Riva-Translate-4B-Instruct LLM",
            module=".impl.riva_instruct",
            class_name="RivaInstructTranslator",
            supported_pairs=[
                ("ja", "en"),
                ("en", "ja"),
                ("zh", "en"),
                ("en", "zh"),
                ("ko", "en"),
                ("en", "ko"),
                ("de", "en"),
                ("en", "de"),
                ("fr", "en"),
                ("en", "fr"),
                ("es", "en"),
                ("en", "es"),
                ("pt", "en"),
                ("en", "pt"),
                ("ru", "en"),
                ("en", "ru"),
                ("ar", "en"),
                ("en", "ar"),
            ],
            requires_model_load=True,
            requires_gpu=True,
            default_context_sentences=5,  # LLM なので多めに活用可
            default_params={"device": "cuda", "max_new_tokens": 256},
        ),
    }

    @classmethod
    def get(cls, translator_id: str) -> Optional[TranslatorInfo]:
        """
        翻訳エンジンのメタデータを取得

        Args:
            translator_id: 翻訳エンジンID

        Returns:
            TranslatorInfo、見つからない場合は None
        """
        return cls._TRANSLATORS.get(translator_id)

    @classmethod
    def get_all(cls) -> Dict[str, TranslatorInfo]:
        """
        全ての翻訳エンジンメタデータを取得

        Returns:
            翻訳エンジンID をキーとした TranslatorInfo の辞書
        """
        return cls._TRANSLATORS.copy()

    @classmethod
    def get_translators_for_pair(cls, source: str, target: str) -> List[str]:
        """
        指定された言語ペアをサポートする翻訳エンジンを取得

        Args:
            source: ソース言語コード
            target: ターゲット言語コード

        Returns:
            翻訳エンジンIDのリスト
        """
        result = []
        for tid, info in cls._TRANSLATORS.items():
            # Google は全ペア対応
            if tid == "google":
                result.append(tid)
            elif (source, target) in info.supported_pairs:
                result.append(tid)
        return result

    @classmethod
    def list_translator_ids(cls) -> List[str]:
        """
        利用可能な翻訳エンジンIDのリストを取得

        Returns:
            翻訳エンジンIDのリスト
        """
        return list(cls._TRANSLATORS.keys())
