"""
Riva-Translate-4B-Instruct 翻訳エンジン実装

NVIDIA Riva-Translate-4B-Instruct LLM を使用した高品質翻訳エンジン。
GPU 推奨（~8GB VRAM）、CPU でも動作可能だが低速。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

# Import dependencies at module level to enable conditional import in __init__.py
import torch
import transformers

from ..base import BaseTranslator
from ..exceptions import TranslationModelError, UnsupportedLanguagePairError
from ..lang_codes import get_language_name, to_iso639_1
from ..result import TranslationResult

if TYPE_CHECKING:
    pass  # Type-only imports go here if needed

logger = logging.getLogger(__name__)

# Riva-4B がサポートする言語
SUPPORTED_LANGUAGES = [
    "ja",  # Japanese
    "en",  # English
    "zh",  # Simplified Chinese
    "ko",  # Korean
    "de",  # German
    "fr",  # French
    "es",  # Spanish
    "pt",  # Brazilian Portuguese
    "ru",  # Russian
    "ar",  # Arabic
]


class RivaInstructTranslator(BaseTranslator):
    """
    Riva-Translate-4B-Instruct via transformers

    NVIDIA Riva-Translate-4B-Instruct LLM を使用した翻訳エンジン。
    12言語に対応し、文脈を考慮した高品質な翻訳が可能。

    Examples:
        >>> translator = RivaInstructTranslator(device="cuda")
        >>> translator.load_model()
        >>> result = translator.translate("こんにちは", "ja", "en")
        >>> print(result.text)
        "Hello"
    """

    MODEL_NAME = "nvidia/Riva-Translate-4B-Instruct"
    REQUIRED_VRAM_MB = 8000  # ~8GB for fp16

    def __init__(
        self,
        device: str = "cuda",
        max_new_tokens: int = 256,
        **kwargs,
    ):
        """
        RivaInstructTranslator を初期化

        Args:
            device: 推論デバイス（"cuda" or "cpu"、デフォルト: "cuda"）
            max_new_tokens: 生成する最大トークン数（デフォルト: 256）
            **kwargs: BaseTranslator に渡すパラメータ
        """
        super().__init__(**kwargs)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Optional[transformers.PreTrainedModel] = None
        self._tokenizer: Optional[transformers.PreTrainedTokenizer] = None

    def load_model(self) -> None:
        """
        モデルをロード

        Raises:
            TranslationModelError: モデルのロードに失敗した場合
        """
        from livecap_cli.utils import get_available_vram

        # VRAM チェック（警告のみ）
        if self.device == "cuda":
            available = get_available_vram()
            if available is not None and available < self.REQUIRED_VRAM_MB:
                logger.warning(
                    "Riva-4B requires ~%dMB VRAM. Available: %dMB. "
                    "Consider using device='cpu' or 'opus_mt' translator.",
                    self.REQUIRED_VRAM_MB,
                    available,
                )
            elif available is None:
                logger.debug("VRAM check skipped (torch.cuda not available)")

        try:
            logger.info(
                "Loading Riva-Translate-4B-Instruct model (device=%s)...",
                self.device,
            )

            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.MODEL_NAME
            )

            # GPU: device_map="auto" で自動配置、CPU: None でロード後に移動
            if self.device == "cuda":
                self._model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self._model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.float32,
                )
                self._model = self._model.to("cpu")

            self._initialized = True
            logger.info(
                "Loaded Riva-Translate-4B-Instruct model on %s",
                self.device,
            )
        except Exception as e:
            raise TranslationModelError(
                f"Failed to load Riva-Translate-4B-Instruct: {e}"
            ) from e

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

        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        # プロンプト構築
        system_content = (
            f"You are an expert at translating text from {source_name} to {target_name}."
        )

        if context:
            ctx = context[-self._default_context_sentences :]
            system_content += "\n\nPrevious context for reference:\n" + "\n".join(ctx)

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"What is the {target_name} translation of the sentence: {text}?",
            },
        ]

        try:
            tokenized = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    tokenized,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=False,  # Greedy decoding for deterministic output
                )

            # 入力部分を除いてデコード
            result = self._tokenizer.decode(
                outputs[0][tokenized.shape[1] :],
                skip_special_tokens=True,
            )
        except Exception as e:
            raise TranslationModelError(f"Translation failed: {e}") from e

        return TranslationResult(
            text=result.strip(),
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def get_translator_name(self) -> str:
        """翻訳エンジン名を取得"""
        return "riva_instruct"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """
        サポートする言語ペアを取得

        Returns:
            サポートする全言語ペアのリスト
        """
        return [
            (s, t) for s in SUPPORTED_LANGUAGES for t in SUPPORTED_LANGUAGES if s != t
        ]

    def cleanup(self) -> None:
        """
        リソースのクリーンアップ

        モデルとトークナイザーを解放し、CUDA キャッシュをクリア。
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False

        # CUDA キャッシュをクリア
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("RivaInstructTranslator cleanup completed")
