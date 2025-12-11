"""
OPUS-MT 翻訳エンジン実装

Helsinki-NLP の OPUS-MT モデルを CTranslate2 で高速推論する翻訳エンジン。
ローカルで動作し、CPU でも十分高速。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

# Import dependencies at module level to enable conditional import in __init__.py
# This allows `impl/__init__.py` to catch ImportError when deps are missing
import ctranslate2
import transformers

from ..base import BaseTranslator
from ..exceptions import TranslationModelError, UnsupportedLanguagePairError
from ..lang_codes import get_opus_mt_model_name, to_iso639_1
from ..result import TranslationResult

if TYPE_CHECKING:
    pass  # Type-only imports go here if needed

logger = logging.getLogger(__name__)


class OpusMTTranslator(BaseTranslator):
    """
    OPUS-MT via CTranslate2

    Helsinki-NLP の OPUS-MT モデルを使用したローカル翻訳エンジン。
    CTranslate2 による INT8 量子化で高速・省メモリ推論。

    Examples:
        >>> translator = OpusMTTranslator(source_lang="ja", target_lang="en")
        >>> translator.load_model()
        >>> result = translator.translate("こんにちは", "ja", "en")
        >>> print(result.text)
        "Hello"
    """

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        model_name: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ):
        """
        OpusMTTranslator を初期化

        Args:
            source_lang: ソース言語コード（デフォルト: "ja"）
            target_lang: ターゲット言語コード（デフォルト: "en"）
            model_name: HuggingFace モデル名（省略時は言語ペアから自動生成）
            device: 推論デバイス（"cpu" or "cuda"、デフォルト: "cpu"）
            compute_type: 量子化タイプ（"int8", "float16" 等、デフォルト: "int8"）
            **kwargs: BaseTranslator に渡すパラメータ
        """
        super().__init__(**kwargs)
        self.source_lang = source_lang
        self.target_lang = target_lang

        # model_name が指定されていない場合は言語ペアから生成
        if model_name is None:
            model_name = get_opus_mt_model_name(source_lang, target_lang)
        self.model_name = model_name

        self.device = device
        self.compute_type = compute_type
        self._model: Optional[ctranslate2.Translator] = None
        self._tokenizer: Optional[transformers.PreTrainedTokenizer] = None

    def load_model(self) -> None:
        """
        モデルをロード

        CTranslate2 形式に変換されたモデルをロード。
        変換済みモデルが存在しない場合は自動変換。

        Raises:
            TranslationModelError: モデルのロードまたは変換に失敗した場合
        """
        from livecap_core.utils import get_models_dir

        # CTranslate2 形式に変換されたモデルのディレクトリ
        model_dir = get_models_dir() / "opus-mt" / self.model_name.replace("/", "--")

        if not model_dir.exists():
            self._convert_model(model_dir)

        try:
            self._model = ctranslate2.Translator(
                str(model_dir),
                device=self.device,
                compute_type=self.compute_type,
            )
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            self._initialized = True
            logger.info(
                "Loaded OPUS-MT model: %s (device=%s, compute_type=%s)",
                self.model_name,
                self.device,
                self.compute_type,
            )
        except Exception as e:
            raise TranslationModelError(f"Failed to load model: {e}") from e

    def _convert_model(self, output_dir: Path) -> None:
        """
        HuggingFace モデルを CTranslate2 形式に変換

        Args:
            output_dir: 変換後モデルの出力ディレクトリ

        Raises:
            TranslationModelError: 変換に失敗した場合
        """
        from ctranslate2.converters import TransformersConverter

        logger.info("Converting %s to CTranslate2 format...", self.model_name)

        try:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            converter = TransformersConverter(self.model_name)
            converter.convert(str(output_dir), quantization=self.compute_type)
            logger.info("Model conversion completed: %s", output_dir)
        except Exception as e:
            raise TranslationModelError(f"Failed to convert model: {e}") from e

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        テキストを翻訳

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード (BCP-47)
            target_lang: ターゲット言語コード (BCP-47)
            context: 過去の文脈（直近N文）。翻訳精度向上のため使用。

        Returns:
            TranslationResult

        Raises:
            TranslationModelError: モデル未ロード、または推論エラー
            UnsupportedLanguagePairError: 同一言語が指定された場合
        """
        # モデルロードチェック
        if not self._initialized or self._model is None or self._tokenizer is None:
            raise TranslationModelError("Model not loaded. Call load_model() first.")

        # 入力バリデーション: 空文字列
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        # 入力バリデーション: 同一言語
        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        # 文脈連結（改行区切りで段落として認識させる）
        if context:
            ctx = context[-self._default_context_sentences :]
            full_text = "\n".join(ctx) + "\n" + text
        else:
            full_text = text

        try:
            # トークナイズ
            source_tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(full_text)
            )

            # 翻訳
            results = self._model.translate_batch([source_tokens])
            target_tokens = results[0].hypotheses[0]

            # デコード
            result = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(target_tokens),
                skip_special_tokens=True,
            )
        except Exception as e:
            raise TranslationModelError(f"Translation failed: {e}") from e

        # 文脈を含めた場合、最後の段落を抽出
        if context:
            result = self._extract_relevant_part(result)

        return TranslationResult(
            text=result,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _extract_relevant_part(self, translated: str) -> str:
        """
        翻訳結果から対象部分（最後の段落）を抽出

        文脈を連結して翻訳した場合、最後の段落のみを抽出する。

        Args:
            translated: 翻訳結果テキスト

        Returns:
            最後の段落
        """
        lines = translated.strip().split("\n")
        return lines[-1] if lines else translated

    def get_translator_name(self) -> str:
        """翻訳エンジン名を取得"""
        return "opus_mt"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        サポートする言語ペアを取得

        Returns:
            初期化時に指定された言語ペア
        """
        return [(self.source_lang, self.target_lang)]

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ

        モデルとトークナイザーを解放。
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False
        logger.debug("OpusMTTranslator cleanup completed")
