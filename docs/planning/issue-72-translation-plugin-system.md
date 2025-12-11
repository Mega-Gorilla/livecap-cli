# Issue #72: 翻訳プラグインシステム実装

## 概要

翻訳プラグインシステムを設計し、Google Translate、OPUS-MT、Riva-Translate-4B-Instruct の3つの翻訳エンジンを実装する。

## 実装ステータス

> **Note**: 本ドキュメントは **実装計画** です。以下のコンポーネントは計画段階であり、実装はこれから行われます。

### 新規作成（未実装）

| コンポーネント | ステータス | Phase | 備考 |
|---------------|-----------|-------|------|
| `livecap_core/translation/` | ❌ 未実装 | 1 | ディレクトリ作成 |
| `translation/base.py` | ❌ 未実装 | 1 | BaseTranslator ABC |
| `translation/result.py` | ❌ 未実装 | 1 | TranslationResult dataclass |
| `translation/metadata.py` | ❌ 未実装 | 1 | TranslatorMetadata |
| `translation/factory.py` | ❌ 未実装 | 1 | TranslatorFactory |
| `translation/exceptions.py` | ❌ 未実装 | 1 | 例外クラス階層 |
| `translation/retry.py` | ❌ 未実装 | 1 | リトライデコレータ |
| `translation/lang_codes.py` | ❌ 未実装 | 1 | 言語コードユーティリティ |
| `translation/impl/google.py` | ❌ 未実装 | 2 | GoogleTranslator |
| `translation/impl/opus_mt.py` | ❌ 未実装 | 3 | OpusMTTranslator |
| `translation/impl/riva_instruct.py` | ❌ 未実装 | 4 | RivaInstructTranslator |
| `utils/__init__.py` VRAM 追加 | ❌ 未実装 | 1 | get_available_vram 等 |
| `pyproject.toml` 依存追加 | ❌ 未実装 | 1 | translation-local, translation-riva |
| `livecap_core/__init__.py` エクスポート | ❌ 未実装 | 5 | TranslatorFactory 等 |
| `tests/core/translation/` | ❌ 未実装 | 2-4 | ユニットテスト |
| `tests/integration/test_translation.py` | ❌ 未実装 | 5 | 統合テスト |
| `tests/conftest.py` マーカー追加 | ❌ 未実装 | 2 | network, slow, gpu |

### 既存コード（参照のみ）

| コンポーネント | ステータス | ファイル |
|---------------|-----------|----------|
| `TranslationRequestEventDict` | ✅ 既存 | `transcription_types.py` |
| `TranslationResultEventDict` | ✅ 既存 | `transcription_types.py` |
| `create_translation_result_event()` | ✅ 既存 | `transcription_types.py` |
| `LoadPhase.TRANSLATION_MODEL` | ✅ 既存 | `model_loading_phases.py` |
| `EngineMetadata.to_iso639_1()` | ✅ 既存 | `engines/metadata.py` |

## 背景と調査結果

### 翻訳エンジン選定

調査の結果、以下の3エンジンを実装対象として選定：

| エンジン | 種別 | 文脈対応 | 要件 |
|---------|------|---------|------|
| **Google Translate** | クラウド API | パラグラフ連結 | インターネット |
| **OPUS-MT** | ローカル NMT | パラグラフ連結 | CPU / GPU |
| **Riva-Translate-4B-Instruct** | ローカル LLM | プロンプト (8K) | GPU (~8GB) |

### 除外したエンジン

- **Riva NMT**: Riva Server のセットアップが複雑（Triton + gRPC 必須）
- **NLLB-200 / M2M-100**: OPUS-MT と役割が重複、言語ペアごとのモデル管理の方が柔軟
- **Argos Translate**: OPUS-MT と同じ CTranslate2 基盤、直接 OPUS-MT を使用する方がシンプル

### 文脈挿入方式

全エンジンで文脈を活用可能：

```python
# 方式1: パラグラフ連結（Google, OPUS-MT）
history = "昨日はVRChatで友達とドライブした。彼はとてもスピードを出した。"
current = "そのせいで今日は少し疲れている。"
full_input = history + "\n" + current
translation = translator.translate(full_input)

# 方式2: プロンプト（Riva-4B-Instruct）
prompt = f"""<s>System You are an expert translator.
Previous context: {history}</s>
<s>User Translate: {current}</s>
<s>Assistant"""
```

## アーキテクチャ設計

### 設計方針: ASR エンジンとの関係

#### 検討した選択肢

| 選択肢 | 説明 | 評価 |
|--------|------|------|
| **A. 共通基底クラス** | `BaseModelEngine` → `BaseEngine` / `BaseTranslator` | ❌ 過度な結合 |
| **B. 完全分離** | `BaseTranslator` を独立設計 | ✅ **採用** |
| **C. コンポジション** | `ModelLoader` を共有 | △ 将来検討 |

#### 採用理由（選択肢 B）

1. **インターフェースの違い**:
   - ASR: `transcribe(audio, sample_rate) -> (text, confidence)`
   - 翻訳: `translate(text, source, target, context) -> str`

2. **モデルロードパターンの違い**:
   - ASR: 重い初期化、進捗報告が重要
   - 翻訳: Google は初期化不要、OPUS-MT は軽量

3. **共有する部分**:
   - `detect_device()` - 既存ユーティリティを再利用
   - `get_models_dir()` - モデルディレクトリ管理
   - `TranslatorMetadata` - `EngineMetadata` と同様の設計

### モジュール構成

```
livecap_core/
├── translation/
│   ├── __init__.py              # Public API exports
│   ├── base.py                  # BaseTranslator ABC
│   ├── factory.py               # TranslatorFactory
│   ├── metadata.py              # TranslatorMetadata
│   ├── result.py                # TranslationResult dataclass
│   └── impl/
│       ├── __init__.py
│       ├── google.py            # GoogleTranslator
│       ├── opus_mt.py           # OpusMTTranslator
│       └── riva_instruct.py     # RivaInstructTranslator
└── __init__.py                  # Add translation exports
```

### クラス設計

#### BaseTranslator

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class TranslationResult:
    """翻訳結果"""
    text: str                          # 翻訳テキスト
    original_text: str                 # 原文（イベント型との整合性）
    source_lang: str                   # ソース言語
    target_lang: str                   # ターゲット言語
    confidence: Optional[float] = None # 信頼度（LLMの場合）
    source_id: str = "default"         # ソース識別子

    def to_event_dict(self) -> "TranslationResultEventDict":
        """既存の TranslationResultEventDict に変換"""
        from livecap_core.transcription_types import create_translation_result_event
        return create_translation_result_event(
            original_text=self.original_text,
            translated_text=self.text,
            source_id=self.source_id,
            source_language=self.source_lang,
            target_language=self.target_lang,
            confidence=self.confidence,
        )

class BaseTranslator(ABC):
    """翻訳エンジンの抽象基底クラス"""

    def __init__(self, default_context_sentences: int = 2, **kwargs):
        self._initialized = False
        self._default_context_sentences = default_context_sentences

    @abstractmethod
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
            context: 過去の文脈（直近N文）

        Returns:
            TranslationResult
        """
        ...

    async def translate_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        非同期翻訳（デフォルト実装）

        同期メソッドを asyncio.to_thread でラップ。
        サブクラスで真の非同期実装が必要な場合はオーバーライド可能。
        """
        import asyncio
        return await asyncio.to_thread(
            self.translate, text, source_lang, target_lang, context
        )

    @abstractmethod
    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """サポートする言語ペアを取得"""
        ...

    @abstractmethod
    def get_translator_name(self) -> str:
        """翻訳エンジン名を取得"""
        ...

    def is_initialized(self) -> bool:
        """初期化済みかどうか"""
        return self._initialized

    def load_model(self) -> None:
        """モデルをロード（ローカルモデルの場合）"""
        pass  # クラウド API はオーバーライド不要

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        pass
```

#### TranslatorMetadata

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class TranslatorInfo:
    """翻訳エンジンのメタデータ"""
    translator_id: str
    display_name: str
    description: str
    module: str                              # e.g., ".impl.google"
    class_name: str                          # e.g., "GoogleTranslator"
    supported_pairs: List[Tuple[str, str]]   # [("ja", "en"), ("en", "ja"), ...]
    requires_model_load: bool = False        # モデルロードが必要か
    requires_gpu: bool = False               # GPU 必須か
    default_context_sentences: int = 2       # デフォルト文脈数
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
            default_context_sentences=2,
            default_params={"device": "cpu", "compute_type": "int8"},
        ),
        "riva_instruct": TranslatorInfo(
            translator_id="riva_instruct",
            display_name="Riva Translate 4B Instruct",
            description="NVIDIA Riva-Translate-4B-Instruct LLM",
            module=".impl.riva_instruct",
            class_name="RivaInstructTranslator",
            supported_pairs=[
                ("ja", "en"), ("en", "ja"),
                ("zh", "en"), ("en", "zh"),
                ("ko", "en"), ("en", "ko"),
                # ... 12言語対応
            ],
            requires_model_load=True,
            requires_gpu=True,
            default_context_sentences=5,  # LLM なので多めに活用可
            default_params={"device": "cuda", "max_new_tokens": 256},
        ),
    }

    @classmethod
    def get(cls, translator_id: str) -> Optional[TranslatorInfo]:
        return cls._TRANSLATORS.get(translator_id)

    @classmethod
    def get_all(cls) -> Dict[str, TranslatorInfo]:
        return cls._TRANSLATORS.copy()

    @classmethod
    def get_translators_for_pair(cls, source: str, target: str) -> List[str]:
        """指定された言語ペアをサポートする翻訳エンジンを取得"""
        result = []
        for tid, info in cls._TRANSLATORS.items():
            # Google は全ペア対応
            if tid == "google":
                result.append(tid)
            elif (source, target) in info.supported_pairs:
                result.append(tid)
        return result
```

#### TranslatorFactory

```python
class TranslatorFactory:
    """翻訳エンジンを作成するファクトリークラス"""

    @classmethod
    def create_translator(
        cls,
        translator_type: str,
        **translator_options,
    ) -> BaseTranslator:
        """
        指定されたタイプの翻訳エンジンを作成

        Args:
            translator_type: 翻訳エンジンタイプ
                利用可能: google, opus_mt, riva_instruct
            **translator_options: エンジン固有のパラメータ

        Returns:
            BaseTranslator のインスタンス

        Examples:
            # Google Translate
            translator = TranslatorFactory.create_translator("google")

            # OPUS-MT (CPU)
            translator = TranslatorFactory.create_translator(
                "opus_mt",
                model_name="Helsinki-NLP/opus-mt-ja-en",
                device="cpu"
            )

            # Riva 4B Instruct (GPU)
            translator = TranslatorFactory.create_translator(
                "riva_instruct",
                device="cuda"
            )
        """
        metadata = TranslatorMetadata.get(translator_type)
        if metadata is None:
            available = list(TranslatorMetadata.get_all().keys())
            raise ValueError(
                f"Unknown translator type: {translator_type}. "
                f"Available: {available}"
            )

        # default_params と options をマージ
        params = {**metadata.default_params, **translator_options}

        # default_context_sentences をメタデータから注入
        if "default_context_sentences" not in params:
            params["default_context_sentences"] = metadata.default_context_sentences

        # 動的インポート
        import importlib
        module = importlib.import_module(
            metadata.module,
            package="livecap_core.translation"
        )
        translator_class = getattr(module, metadata.class_name)

        return translator_class(**params)
```

### 各エンジンの実装

#### 1. GoogleTranslator

```python
from typing import Optional, List, Tuple
from deep_translator import GoogleTranslator as DeepGoogleTranslator
from deep_translator.exceptions import (
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

from .base import BaseTranslator, TranslationResult
from .lang_codes import normalize_for_google, to_iso639_1
from .exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)
from .retry import with_retry

class GoogleTranslator(BaseTranslator):
    """Google Translate (via deep-translator)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = True  # 初期化不要

    @with_retry(max_retries=3, base_delay=1.0)
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        # 入力バリデーション
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        # 文脈をパラグラフとして連結
        if context:
            ctx = context[-self._default_context_sentences:]
            full_text = "\n".join(ctx) + "\n" + text
        else:
            full_text = text

        try:
            translator = DeepGoogleTranslator(
                source=normalize_for_google(source_lang),
                target=normalize_for_google(target_lang),
            )
            result = translator.translate(full_text)
        except TooManyRequests as e:
            raise TranslationNetworkError(f"Rate limited: {e}") from e
        except RequestError as e:
            raise TranslationNetworkError(f"API request failed: {e}") from e
        except TranslationNotFound as e:
            raise TranslationError(f"Translation not found: {e}") from e
        except Exception as e:
            raise TranslationError(f"Unexpected error: {e}") from e

        # 文脈を含めた場合、最後の文を抽出
        if context:
            result = self._extract_last_sentence(result)

        return TranslationResult(
            text=result,
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def _extract_last_sentence(self, text: str) -> str:
        """最後の文を抽出"""
        lines = text.strip().split("\n")
        return lines[-1] if lines else text

    def get_translator_name(self) -> str:
        return "google"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        # 空リスト = 全言語ペア対応
        return []
```

#### 2. OpusMTTranslator

```python
import ctranslate2
import transformers
from typing import Optional, List, Tuple

from .base import BaseTranslator, TranslationResult
from .lang_codes import get_opus_mt_model_name
from .exceptions import TranslationModelError

class OpusMTTranslator(BaseTranslator):
    """OPUS-MT via CTranslate2"""

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        model_name: Optional[str] = None,  # 上級者向けオーバーライド
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source_lang = source_lang
        self.target_lang = target_lang
        # model_name が指定されていない場合は言語ペアから生成
        if model_name is None:
            model_name = get_opus_mt_model_name(source_lang, target_lang)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """モデルをロード"""
        from livecap_core.utils import get_models_dir

        # CTranslate2 形式に変換されたモデルをロード
        model_dir = get_models_dir() / "opus-mt" / self.model_name.replace("/", "--")

        if not model_dir.exists():
            self._convert_model(model_dir)

        self.model = ctranslate2.Translator(
            str(model_dir),
            device=self.device,
            compute_type=self.compute_type,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self._initialized = True

    def _convert_model(self, output_dir):
        """HuggingFace モデルを CTranslate2 形式に変換（Python API 使用）"""
        from ctranslate2.converters import TransformersConverter
        import logging

        logger = logging.getLogger(__name__)
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
        if not self._initialized:
            raise TranslationModelError("Model not loaded. Call load_model() first.")

        # 文脈連結（改行区切りで段落として認識させる）
        if context:
            ctx = context[-self._default_context_sentences:]
            full_text = "\n".join(ctx) + "\n" + text
        else:
            full_text = text

        # トークナイズ
        source_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(full_text)
        )

        # 翻訳
        results = self.model.translate_batch([source_tokens])
        target_tokens = results[0].hypotheses[0]

        # デコード
        result = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(target_tokens),
            skip_special_tokens=True,
        )

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
        """翻訳結果から対象部分（最後の段落）を抽出"""
        lines = translated.strip().split("\n")
        return lines[-1] if lines else translated

    def get_translator_name(self) -> str:
        return "opus_mt"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        return [(self.source_lang, self.target_lang)]

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._initialized = False
```

#### 3. RivaInstructTranslator

```python
from typing import Optional, List, Tuple
import logging

from .base import BaseTranslator, TranslationResult
from .lang_codes import get_language_name, to_iso639_1
from .exceptions import TranslationModelError, UnsupportedLanguagePairError

logger = logging.getLogger(__name__)

class RivaInstructTranslator(BaseTranslator):
    """Riva-Translate-4B-Instruct via transformers"""

    LANG_NAMES = {
        "ja": "Japanese", "en": "English", "zh": "Simplified Chinese",
        "ko": "Korean", "de": "German", "fr": "French",
        "es": "Spanish", "pt": "Brazilian Portuguese", "ru": "Russian",
        "ar": "Arabic", "zh-TW": "Traditional Chinese",
    }

    def __init__(
        self,
        device: str = "cuda",
        max_new_tokens: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """モデルをロード"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from livecap_core.utils import get_available_vram

        model_name = "nvidia/Riva-Translate-4B-Instruct"

        # VRAM チェック（警告のみ）
        if self.device == "cuda":
            available = get_available_vram()
            if available is not None and available < 8000:
                logger.warning(
                    "Riva-4B requires ~8GB VRAM. Available: %dMB. "
                    "Consider using device='cpu' or 'opus_mt' translator.",
                    available
                )
            elif available is None:
                logger.debug("VRAM check skipped (torch not installed)")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            if self.device == "cuda":
                self.model = self.model.cuda()

            self._initialized = True
        except Exception as e:
            raise TranslationModelError(f"Failed to load model: {e}") from e

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[str]] = None,
    ) -> TranslationResult:
        if not self._initialized:
            raise TranslationModelError("Model not loaded. Call load_model() first.")

        # 入力バリデーション
        if not text or not text.strip():
            return TranslationResult(
                text="",
                original_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        if to_iso639_1(source_lang) == to_iso639_1(target_lang):
            raise UnsupportedLanguagePairError(
                source_lang, target_lang, self.get_translator_name()
            )

        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        # プロンプト構築
        system_content = f"You are an expert at translating text from {source_name} to {target_name}."

        if context:
            ctx = context[-self._default_context_sentences:]
            system_content += f"\n\nPrevious context for reference:\n" + "\n".join(ctx)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"What is the {target_name} translation of the sentence: {text}?"},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            tokenized,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 入力部分を除いてデコード
        result = self.tokenizer.decode(
            outputs[0][tokenized.shape[1]:],
            skip_special_tokens=True,
        )

        return TranslationResult(
            text=result.strip(),
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def get_translator_name(self) -> str:
        return "riva_instruct"

    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        langs = list(self.LANG_NAMES.keys())
        return [(s, t) for s in langs for t in langs if s != t]

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._initialized = False
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
```

## エラーハンドリング

### 例外クラス階層

```python
# livecap_core/translation/exceptions.py

class TranslationError(Exception):
    """翻訳エラーの基底クラス"""
    pass

class TranslationNetworkError(TranslationError):
    """ネットワーク関連エラー（API 失敗、タイムアウト）"""
    pass

class TranslationModelError(TranslationError):
    """モデル関連エラー（ロード失敗、推論失敗）"""
    pass

class UnsupportedLanguagePairError(TranslationError):
    """未サポートの言語ペア"""
    def __init__(self, source: str, target: str, translator: str):
        self.source = source
        self.target = target
        self.translator = translator
        super().__init__(
            f"Language pair ({source} -> {target}) not supported by {translator}"
        )
```

### 入力バリデーション戦略

各翻訳エンジンの `translate()` メソッドで以下のバリデーションを実施：

| ケース | 動作 | 理由 |
|--------|------|------|
| 空文字列 `""` | 空の `TranslationResult` を返す | エラーより実用的、呼び出し側での追加チェック不要 |
| 同一言語 | `UnsupportedLanguagePairError` を送出 | `validate_translation_event()` と一貫、無駄な API 呼び出し防止 |
| 無効な言語コード | `langcodes` の例外をそのまま伝播 | Fail fast、早期にユーザーに通知 |
| 長文（トークン制限超過） | 将来検討 | Phase 1 では未実装、警告ログ + 切り詰めを予定 |

**実装例**（各翻訳エンジンの `translate()` 冒頭）:

```python
def translate(self, text, source_lang, target_lang, context=None):
    # 空文字列チェック
    if not text or not text.strip():
        return TranslationResult(
            text="",
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    # 同一言語チェック
    if to_iso639_1(source_lang) == to_iso639_1(target_lang):
        raise UnsupportedLanguagePairError(
            source_lang, target_lang, self.get_translator_name()
        )

    # ... 翻訳処理
```

### リトライ戦略

Google Translate のみリトライ対象（ローカルモデルは失敗時リトライ不要）：

```python
# livecap_core/translation/retry.py
import functools
import time
import logging

logger = logging.getLogger(__name__)

def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """指数バックオフリトライデコレータ"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TranslationNetworkError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Translation failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt + 1, max_retries, delay, e
                        )
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator
```

### 適用例

```python
class GoogleTranslator(BaseTranslator):
    @with_retry(max_retries=3, base_delay=1.0)
    def translate(self, text, source_lang, target_lang, context=None):
        try:
            # ... 翻訳処理
        except Exception as e:
            raise TranslationNetworkError(f"Google Translate API error: {e}") from e
```

## 言語コード正規化

### 既存実装との整合性

Issue #168 で実装された `EngineMetadata.to_iso639_1()` と `langcodes` ライブラリを活用する。

```python
# 既存実装: livecap_core/engines/metadata.py:234-261
@classmethod
def to_iso639_1(cls, code: str) -> str:
    """BCP-47 言語コードを ISO 639-1 に変換"""
    return langcodes.Language.get(code).language

# 使用例
>>> EngineMetadata.to_iso639_1("zh-CN")  # 'zh'
>>> EngineMetadata.to_iso639_1("ZH-TW")  # 'zh'（大文字も正規化）
>>> EngineMetadata.to_iso639_1("pt-BR")  # 'pt'
```

### 標準入力形式

入力言語コードは **BCP-47** を受け入れ、内部で ISO 639-1 に変換：

| 入力例 | ISO 639-1 | 言語 |
|--------|-----------|------|
| `ja`, `ja-JP` | `ja` | 日本語 |
| `en`, `en-US` | `en` | 英語 |
| `zh`, `zh-CN` | `zh` | 中国語（簡体字） |
| `zh-TW` | `zh` | 中国語（繁体字）※ |
| `ko`, `ko-KR` | `ko` | 韓国語 |

**※ 注**: `zh-TW` は ISO 639-1 では `zh` になるが、Google Translate では `zh-TW` として扱う必要がある。

### 言語コードユーティリティ

```python
# livecap_core/translation/lang_codes.py
import langcodes

# Riva-4B プロンプト用の言語名マッピング
LANGUAGE_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Brazilian Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}

def to_iso639_1(code: str) -> str:
    """
    BCP-47 言語コードを ISO 639-1 に変換

    langcodes ライブラリを使用（EngineMetadata.to_iso639_1 と同じ実装）
    """
    return langcodes.Language.get(code).language

def normalize_for_google(lang: str) -> str:
    """
    Google Translate 用に正規化

    Note: Google は zh-CN/zh-TW を区別するため、
          元の入力が zh-TW なら維持する
    """
    # 元の入力を保持（zh-TW の場合）
    if lang.lower() in ("zh-tw", "zh-hant"):
        return "zh-TW"
    # それ以外は ISO 639-1 に変換後、中国語のみ zh-CN に
    iso = to_iso639_1(lang)
    if iso == "zh":
        return "zh-CN"
    return iso

def normalize_for_opus_mt(lang: str) -> str:
    """OPUS-MT 用に正規化（ISO 639-1）"""
    return to_iso639_1(lang)

def get_language_name(lang: str) -> str:
    """Riva 用に言語名を取得"""
    iso = to_iso639_1(lang)
    # zh-TW は特別扱い
    if lang.lower() in ("zh-tw", "zh-hant"):
        return LANGUAGE_NAMES.get("zh-TW", "Traditional Chinese")
    return LANGUAGE_NAMES.get(iso, langcodes.Language.get(lang).display_name())

def get_opus_mt_model_name(source: str, target: str) -> str:
    """OPUS-MT モデル名を生成"""
    src = normalize_for_opus_mt(source)
    tgt = normalize_for_opus_mt(target)
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"
```

### 各エンジンでの使用

```python
# GoogleTranslator
def _normalize_lang(self, lang: str) -> str:
    return normalize_for_google(lang)  # zh-CN/zh-TW を区別

# OpusMTTranslator
def __init__(self, source_lang: str, target_lang: str, ...):
    self.model_name = get_opus_mt_model_name(source_lang, target_lang)

# RivaInstructTranslator
def _build_prompt(self, text, source_lang, target_lang):
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    return f"Translate from {source_name} to {target_name}: {text}"
```

## 既存コードとの統合

### 既存イベント型との連携

`livecap_core/transcription_types.py` に翻訳関連のイベント型が既に定義されている：

| 型 | 用途 |
|----|------|
| `TranslationRequestEventDict` | 翻訳リクエストイベント |
| `TranslationResultEventDict` | 翻訳結果イベント |
| `create_translation_request_event()` | リクエストイベント生成 |
| `create_translation_result_event()` | 結果イベント生成 |

`TranslationResult.to_event_dict()` メソッドにより、翻訳結果を既存のパイプラインイベントに変換可能：

```python
# 翻訳実行
result = translator.translate("こんにちは", "ja", "en")

# 既存イベント型に変換してパイプラインに流す
event = result.to_event_dict()
# -> TranslationResultEventDict として処理可能
```

### 話者識別（Speaker Diarization）について

既存の `TranslationResultEventDict` には `speaker` フィールドが定義されているが、これは**未実装のプレースホルダー**である。

| 項目 | 状態 |
|------|------|
| ASR エンジンでの話者設定 | ❌ 未実装 |
| 話者識別機能の計画 | ❌ なし |
| 本 Issue での対応 | ❌ スコープ外 |

**結論**: `TranslationResult` に `speaker` フィールドは追加しない。既存の `speaker` フィールドの削除は別 Issue (#179) で対応。

### LoadPhase.TRANSLATION_MODEL との関係

`livecap_core/engines/model_loading_phases.py` に `LoadPhase.TRANSLATION_MODEL` (進捗 75-100%) が定義済み。

#### 設計方針

| 選択肢 | 説明 | 採用 |
|--------|------|------|
| 統合ロード | ASR モデルロード後に自動で翻訳もロード | ❌ |
| **分離ロード** | 翻訳は別途 `TranslatorFactory` で管理 | ✅ |

**理由**:
- ASR と翻訳は独立したライフサイクル（片方だけ使うケースも多い）
- GPU メモリ管理を明示的に制御可能
- 既存の `LoadPhase.TRANSLATION_MODEL` は GUI 統合時のオプションとして活用

#### GUI 統合時の利用例（将来）

```python
# StreamTranscriberWithTranslation などの統合クラスで使用する場合
class StreamTranscriberWithTranslation:
    def __init__(self, engine, translator=None):
        self.engine = engine
        self.translator = translator

    def load_models(self, progress_callback=None):
        # ASR ロード (0-75%)
        self.engine.load_model(progress_callback=...)

        # 翻訳ロード (75-100%) - LoadPhase.TRANSLATION_MODEL を使用
        if self.translator:
            if progress_callback:
                progress_callback(75, LoadPhase.TRANSLATION_MODEL)
            self.translator.load_model()
            if progress_callback:
                progress_callback(100, LoadPhase.COMPLETED)
```

## GPU メモリ管理

### 問題の背景

ASR エンジンと翻訳エンジンを同時に GPU にロードする場合、VRAM の競合が発生する可能性がある。

### メモリ使用量の見積もり

| コンポーネント | VRAM 使用量 |
|---------------|------------|
| Whisper Base (ASR) | ~150MB |
| Whisper Large-v3 (ASR) | ~3GB |
| Canary 1B (ASR) | ~2.5GB |
| OPUS-MT (per pair) | ~500MB |
| Riva-Translate-4B-Instruct | ~8GB (fp16) |

**最悪ケース**: Whisper Large-v3 (3GB) + Riva-4B (8GB) = **11GB VRAM**

### デフォルトデバイス戦略

| 翻訳エンジン | デフォルト | 理由 |
|-------------|-----------|------|
| Google Translate | N/A | API ベース、GPU 不要 |
| OPUS-MT | **CPU** | 軽量（~500MB）、CPU 上でも十分高速 |
| Riva-4B-Instruct | **GPU** | LLM のため GPU 推奨、CPU では低速 |

### TranslatorMetadata の拡張

```python
@dataclass
class TranslatorInfo:
    # ... 既存フィールド ...
    default_device: Optional[str] = None     # デフォルトデバイス
    gpu_vram_required_mb: int = 0            # 必要 VRAM (MB)
    cpu_fallback_warning: bool = False       # CPU フォールバック時の警告

class TranslatorMetadata:
    _TRANSLATORS = {
        "google": TranslatorInfo(
            # ... 既存 ...
            default_device=None,
            gpu_vram_required_mb=0,
        ),
        "opus_mt": TranslatorInfo(
            # ... 既存 ...
            default_device="cpu",            # CPU 推奨
            gpu_vram_required_mb=500,
        ),
        "riva_instruct": TranslatorInfo(
            # ... 既存 ...
            default_device="cuda",
            gpu_vram_required_mb=8000,
            cpu_fallback_warning=True,       # CPU だと低速
        ),
    }
```

### VRAM 確認ユーティリティ

`livecap_core/utils/__init__.py` に追加：

```python
from typing import Optional

def get_available_vram() -> Optional[int]:
    """
    利用可能な VRAM（MB）を返す。

    Returns:
        VRAM（MB）。GPU なしまたは torch 未インストールの場合は None

    Note:
        torch がインストールされていない場合でも CTranslate2 は
        CUDA を使用可能。この関数は便利機能であり、必須ではない。
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
    except ImportError:
        pass
    return None

def can_fit_on_gpu(required_mb: int, safety_margin: float = 0.9) -> bool:
    """
    指定サイズが GPU に収まるか確認。

    Args:
        required_mb: 必要な VRAM（MB）
        safety_margin: 安全マージン（デフォルト 0.9 = 90%）

    Returns:
        収まる場合 True。GPU なしまたは torch なしの場合は False
    """
    available = get_available_vram()
    if available is None:
        return False
    return available * safety_margin >= required_mb
```

### 推奨構成パターン

| ユースケース | ASR | Translator | 必要 VRAM |
|-------------|-----|-----------|----------|
| **軽量リアルタイム** | Whisper Base (GPU) | Google (N/A) | ~150MB |
| **高品質オフライン** | Whisper Large-v3 (GPU) | OPUS-MT (CPU) | ~3GB |
| **完全ローカル高品質** | Whisper Base (GPU) | Riva-4B (GPU) | ~8.5GB |
| **CPU 専用** | ReazonSpeech (CPU) | OPUS-MT (CPU) | 0 |
| **最高品質** | Whisper Large-v3 (GPU) | Riva-4B (GPU) | ~11GB |

### 実装方針

1. **明示的なデバイス指定**: ユーザーが `device` パラメータで制御可能
   ```python
   engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
   translator = TranslatorFactory.create_translator("opus_mt", device="cpu")
   ```

2. **OPUS-MT は CPU デフォルト**: CTranslate2 の int8 量子化により CPU でも十分高速

3. **Riva-4B は警告付き GPU デフォルト**: VRAM 不足時は明確なエラーメッセージ
   ```python
   if device == "cuda" and not can_fit_on_gpu(8000):
       logger.warning(
           "Riva-4B requires ~8GB VRAM. Current available: %dMB. "
           "Consider using device='cpu' or 'opus_mt' translator.",
           get_available_vram()
       )
   ```

4. **ドキュメントで推奨構成を明記**: ユーザーの GPU 環境に応じた構成例を提供

## 依存関係

### pyproject.toml 更新

```toml
[project.optional-dependencies]
"translation" = [
    "deep-translator>=1.11.4",   # Google Translate
]
"translation-local" = [
    "ctranslate2>=4.0.0",        # OPUS-MT 推論エンジン
    "transformers>=4.40.0",      # モデルロード・トークナイザ
    # sentencepiece は transformers が自動インストール
]
"translation-riva" = [
    "transformers>=4.40.0",      # Riva-4B-Instruct（Mistral ベース）
    "torch>=2.0.0",              # GPU 推論
    # tiktoken 不要: Riva-4B は SentencePiece tokenizer を使用
]
```

**Note**: `tiktoken` は不要。Riva-Translate-4B-Instruct は Mistral ベースで、`transformers.AutoTokenizer` で十分対応可能。

### 既存依存との共有

| 依存 | 共有元 | 用途 |
|------|-------|------|
| `ctranslate2` | `whispers2t` | OPUS-MT 推論（バージョン互換） |
| `transformers` | `engines-nemo` | Riva-4B ロード |
| `torch` | `engines-torch` | GPU 推論 |
| `sentencepiece` | `transformers` (transitive) | OPUS-MT トークナイザ |

## 実装フェーズ

### Phase 1: 基盤実装

1. `livecap_core/translation/` ディレクトリ作成
2. `base.py` - BaseTranslator ABC
3. `result.py` - TranslationResult dataclass
4. `metadata.py` - TranslatorMetadata
5. `factory.py` - TranslatorFactory

### Phase 2: Google Translator

1. `impl/google.py` - GoogleTranslator 実装
2. ユニットテスト
3. 文脈連結のテスト

### Phase 3: OPUS-MT Translator

1. `impl/opus_mt.py` - OpusMTTranslator 実装
2. CTranslate2 モデル変換処理
3. ユニットテスト（モック使用）
4. 統合テスト（実モデル）

### Phase 4: Riva Instruct Translator

1. `impl/riva_instruct.py` - RivaInstructTranslator 実装
2. プロンプトテンプレート最適化
3. ユニットテスト（モック使用）
4. GPU 統合テスト

### Phase 5: 統合・ドキュメント

1. `livecap_core/__init__.py` への export 追加
2. 翻訳単体の統合テスト
3. ドキュメント作成
4. サンプルスクリプト作成（翻訳単体 + ASR 連携例）

**Note**: StreamTranscriber との密結合（`TranslatingTranscriber` 等）は本 Issue のスコープ外。

## 使用例

### 基本的な翻訳（Google Translate）

```python
from livecap_core import TranslatorFactory

# Google Translate は初期化不要
translator = TranslatorFactory.create_translator("google")

# 日本語 → 英語
result = translator.translate("こんにちは", "ja", "en")
print(result.text)  # "Hello"
print(result.original_text)  # "こんにちは"

# 英語 → 日本語
result = translator.translate("Good morning", "en", "ja")
print(result.text)  # "おはようございます"
```

### 文脈を考慮した翻訳

```python
from livecap_core import TranslatorFactory

translator = TranslatorFactory.create_translator("google")

# 会話の文脈を保持
context = [
    "昨日はVRChatで友達とドライブした。",
    "彼はとてもスピードを出した。",
]

# 文脈を渡して翻訳（代名詞の解決に有効）
result = translator.translate(
    "そのせいで今日は少し疲れている。",
    source_lang="ja",
    target_lang="en",
    context=context,
)
print(result.text)
# -> "Because of that, I'm a little tired today."
```

### ローカル翻訳（OPUS-MT）

```python
from livecap_core import TranslatorFactory

# OPUS-MT は CPU で動作（GPU 不要）
translator = TranslatorFactory.create_translator(
    "opus_mt",
    source_lang="ja",
    target_lang="en",
    device="cpu",  # デフォルト
    compute_type="int8",  # 量子化で高速化
)

# モデルロード（初回はダウンロード + 変換）
translator.load_model()

# 翻訳
result = translator.translate("今日は天気が良いですね。", "ja", "en")
print(result.text)

# リソース解放
translator.cleanup()
```

### GPU 翻訳（Riva-4B-Instruct）

```python
from livecap_core import TranslatorFactory

# Riva-4B は GPU 推奨（~8GB VRAM）
translator = TranslatorFactory.create_translator(
    "riva_instruct",
    device="cuda",
    max_new_tokens=256,
)

# モデルロード
translator.load_model()

# 文脈付き翻訳（LLM なので文脈理解が得意）
context = ["会議は10時から始まります。", "資料は事前に配布済みです。"]
result = translator.translate(
    "質問があれば遠慮なくどうぞ。",
    source_lang="ja",
    target_lang="en",
    context=context,
)
print(result.text)

translator.cleanup()
```

### 非同期翻訳

```python
import asyncio
from livecap_core import TranslatorFactory

async def translate_texts():
    translator = TranslatorFactory.create_translator("google")

    texts = [
        "おはようございます",
        "こんにちは",
        "こんばんは",
    ]

    # 並列翻訳
    tasks = [
        translator.translate_async(text, "ja", "en")
        for text in texts
    ]
    results = await asyncio.gather(*tasks)

    for original, result in zip(texts, results):
        print(f"{original} -> {result.text}")

asyncio.run(translate_texts())
```

### ASR + 翻訳の組み合わせ

```python
from livecap_core import (
    EngineFactory,
    StreamTranscriber,
    TranslatorFactory,
    FileSource,
)

# ASR エンジン（GPU）
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="base")
engine.load_model()

# 翻訳エンジン（CPU で VRAM 節約）
translator = TranslatorFactory.create_translator("opus_mt", device="cpu")
translator.load_model()

# 文脈バッファ
context = []

# 音声ファイルからリアルタイム文字起こし + 翻訳
with FileSource("audio.wav") as source:
    with StreamTranscriber(engine=engine) as transcriber:
        for result in transcriber.transcribe_sync(source):
            if result.is_final:
                # 翻訳
                translation = translator.translate(
                    result.text,
                    source_lang="ja",
                    target_lang="en",
                    context=context[-3:],  # 直近3文を文脈として使用
                )

                print(f"[JA] {result.text}")
                print(f"[EN] {translation.text}")
                print()

                # 文脈を更新
                context.append(result.text)

# クリーンアップ
engine.cleanup()
translator.cleanup()
```

### イベント型への変換

```python
from livecap_core import TranslatorFactory

translator = TranslatorFactory.create_translator("google")
result = translator.translate("こんにちは", "ja", "en")

# TranslationResultEventDict に変換
event = result.to_event_dict()

print(event["event_type"])       # "translation_result"
print(event["original_text"])    # "こんにちは"
print(event["translated_text"])  # "Hello"
print(event["source_language"])  # "ja"
print(event["target_language"])  # "en"
print(event["timestamp"])        # Unix timestamp
```

### エラーハンドリング

```python
from livecap_core import TranslatorFactory
from livecap_core.translation.exceptions import (
    TranslationError,
    TranslationNetworkError,
    UnsupportedLanguagePairError,
)

translator = TranslatorFactory.create_translator("google")

try:
    result = translator.translate("Hello", "en", "ja")
except TranslationNetworkError as e:
    # ネットワークエラー（API 失敗、タイムアウト）
    print(f"Network error: {e}")
except UnsupportedLanguagePairError as e:
    # 未サポートの言語ペア
    print(f"Unsupported: {e.source} -> {e.target}")
except TranslationError as e:
    # その他の翻訳エラー
    print(f"Translation failed: {e}")
```

## テスト計画

### テストマーカー

```python
# conftest.py に追加
def pytest_configure(config):
    config.addinivalue_line("markers", "network: requires network access (Google API)")
    config.addinivalue_line("markers", "slow: slow tests (real model loading)")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
```

### CI 実行方針

| テスト種別 | マーカー | CI 実行 |
|-----------|---------|---------|
| ユニット（モック） | なし | ✅ 常に実行 |
| Google 実 API | `@pytest.mark.network` | ❌ スキップ |
| OPUS-MT 実モデル | `@pytest.mark.slow` | ⚙️ オプション |
| Riva GPU | `@pytest.mark.gpu` | ⚙️ self-hosted のみ |

```bash
# CI デフォルト
pytest tests/core/translation -m "not network and not slow and not gpu"

# ローカル開発時（GPU なし）
pytest tests/core/translation -m "not gpu"

# フル実行（GPU 環境）
pytest tests/core/translation
```

### テストファイル構成

```
tests/core/translation/
├── __init__.py
├── test_lang_codes.py           # 言語コード正規化
├── test_retry.py                # リトライデコレータ
├── test_metadata.py             # TranslatorMetadata
├── test_exceptions.py           # 例外クラス
├── test_result.py               # TranslationResult
├── test_factory.py              # TranslatorFactory
├── test_google_translator.py    # GoogleTranslator
├── test_opus_mt_translator.py   # OpusMTTranslator
└── test_riva_instruct_translator.py  # RivaInstructTranslator

tests/integration/
└── test_translation.py          # 統合テスト
```

### ユニットテスト

```python
# tests/core/translation/test_lang_codes.py
import pytest
from livecap_core.translation.lang_codes import (
    to_iso639_1,
    normalize_for_google,
    normalize_for_opus_mt,
    get_language_name,
    get_opus_mt_model_name,
)

def test_to_iso639_1():
    assert to_iso639_1("ja") == "ja"
    assert to_iso639_1("ja-JP") == "ja"
    assert to_iso639_1("zh-CN") == "zh"
    assert to_iso639_1("ZH-TW") == "zh"  # 大文字も正規化

def test_normalize_for_google():
    assert normalize_for_google("ja") == "ja"
    assert normalize_for_google("zh") == "zh-CN"
    assert normalize_for_google("zh-TW") == "zh-TW"  # 繁体字は維持
    assert normalize_for_google("zh-Hant") == "zh-TW"

def test_get_opus_mt_model_name():
    assert get_opus_mt_model_name("ja", "en") == "Helsinki-NLP/opus-mt-ja-en"
    assert get_opus_mt_model_name("en", "ja") == "Helsinki-NLP/opus-mt-en-ja"

# tests/core/translation/test_retry.py
import pytest
from unittest.mock import MagicMock
from livecap_core.translation.retry import with_retry
from livecap_core.translation.exceptions import TranslationNetworkError

def test_retry_success_first_attempt():
    mock_func = MagicMock(return_value="success")
    decorated = with_retry(max_retries=3)(mock_func)
    assert decorated() == "success"
    assert mock_func.call_count == 1

def test_retry_success_after_failure():
    mock_func = MagicMock(side_effect=[
        TranslationNetworkError("fail1"),
        TranslationNetworkError("fail2"),
        "success",
    ])
    decorated = with_retry(max_retries=3, base_delay=0.01)(mock_func)
    assert decorated() == "success"
    assert mock_func.call_count == 3

def test_retry_exhausted():
    mock_func = MagicMock(side_effect=TranslationNetworkError("always fail"))
    decorated = with_retry(max_retries=2, base_delay=0.01)(mock_func)
    with pytest.raises(TranslationNetworkError):
        decorated()
    assert mock_func.call_count == 2

# tests/core/translation/test_metadata.py
from livecap_core.translation.metadata import TranslatorMetadata

def test_get_existing_translator():
    info = TranslatorMetadata.get("google")
    assert info is not None
    assert info.translator_id == "google"
    assert info.requires_model_load is False

def test_get_nonexistent_translator():
    info = TranslatorMetadata.get("nonexistent")
    assert info is None

def test_get_translators_for_pair():
    translators = TranslatorMetadata.get_translators_for_pair("ja", "en")
    assert "google" in translators
    assert "opus_mt" in translators

# tests/core/translation/test_google_translator.py
from unittest.mock import patch, MagicMock

def test_translate_basic_mock():
    """モックを使用した基本テスト"""
    with patch("deep_translator.GoogleTranslator") as mock_gt:
        mock_gt.return_value.translate.return_value = "こんにちは"
        translator = GoogleTranslator()
        result = translator.translate("Hello", "en", "ja")
        assert result.text == "こんにちは"
        assert result.original_text == "Hello"

def test_translate_empty_text():
    """空文字列の翻訳"""
    translator = GoogleTranslator()
    result = translator.translate("", "en", "ja")
    assert result.text == ""

def test_translate_same_language_raises():
    """同一言語でエラー"""
    from livecap_core.translation.exceptions import UnsupportedLanguagePairError
    translator = GoogleTranslator()
    with pytest.raises(UnsupportedLanguagePairError):
        translator.translate("Hello", "en", "en")

@pytest.mark.network
def test_translate_basic_real():
    """実 API を使用したテスト（CI ではスキップ）"""
    translator = GoogleTranslator()
    result = translator.translate("Hello", "en", "ja")
    assert result.text  # 何らかの翻訳が返る

# tests/core/translation/test_opus_mt_translator.py
@pytest.mark.slow
def test_opus_mt_load_model():
    """実モデルロードテスト"""
    translator = OpusMTTranslator(source_lang="en", target_lang="ja")
    translator.load_model()
    assert translator.is_initialized()

def test_opus_mt_model_name_generation():
    """モデル名生成テスト"""
    translator = OpusMTTranslator(source_lang="ja", target_lang="en")
    assert translator.model_name == "Helsinki-NLP/opus-mt-ja-en"

def test_opus_mt_model_name_override():
    """モデル名オーバーライドテスト"""
    translator = OpusMTTranslator(
        source_lang="ja",
        target_lang="en",
        model_name="custom/model-ja-en"
    )
    assert translator.model_name == "custom/model-ja-en"

# tests/core/translation/test_riva_instruct_translator.py
@pytest.mark.gpu
def test_riva_instruct_with_context():
    translator = RivaInstructTranslator(device="cuda")
    translator.load_model()
    context = ["VRChatで友達とドライブした。", "彼はとてもスピードを出した。"]
    result = translator.translate(
        "そのせいで今日は少し疲れている。",
        "ja", "en",
        context=context
    )
    assert "tired" in result.text.lower() or "fatigue" in result.text.lower()
```

### 統合テスト

```python
# tests/integration/test_translation.py

@pytest.mark.slow
def test_opus_mt_full_pipeline():
    """OPUS-MT のフルパイプラインテスト"""
    translator = OpusMTTranslator(source_lang="ja", target_lang="en")
    translator.load_model()

    # 文脈付き翻訳
    context = ["昨日は友達と遊んだ。"]
    result = translator.translate(
        "今日は疲れている。",
        "ja", "en",
        context=context
    )
    assert result.text
    assert result.original_text == "今日は疲れている。"

@pytest.mark.slow
def test_translator_factory():
    """TranslatorFactory の統合テスト"""
    # Google（初期化不要）
    google = TranslatorFactory.create_translator("google")
    assert google.is_initialized()

    # OPUS-MT（要モデルロード）
    opus = TranslatorFactory.create_translator("opus_mt", source_lang="ja", target_lang="en")
    assert not opus.is_initialized()
    opus.load_model()
    assert opus.is_initialized()

def test_translation_result_to_event_dict():
    """TranslationResult のイベント変換テスト"""
    result = TranslationResult(
        text="Hello",
        original_text="こんにちは",
        source_lang="ja",
        target_lang="en",
    )
    event = result.to_event_dict()
    assert event["event_type"] == "translation_result"
    assert event["translated_text"] == "Hello"
    assert event["original_text"] == "こんにちは"
```

**Note**: StreamTranscriber との統合テストは本 Issue のスコープ外。

## 変更ファイル一覧

| ファイル | 操作 | 説明 |
|---------|------|------|
| `livecap_core/translation/__init__.py` | 新規 | Public API |
| `livecap_core/translation/base.py` | 新規 | BaseTranslator |
| `livecap_core/translation/result.py` | 新規 | TranslationResult |
| `livecap_core/translation/metadata.py` | 新規 | TranslatorMetadata |
| `livecap_core/translation/factory.py` | 新規 | TranslatorFactory |
| `livecap_core/translation/exceptions.py` | 新規 | 例外クラス階層 |
| `livecap_core/translation/retry.py` | 新規 | リトライデコレータ |
| `livecap_core/translation/lang_codes.py` | 新規 | 言語コード正規化 |
| `livecap_core/translation/impl/__init__.py` | 新規 | Impl package |
| `livecap_core/translation/impl/google.py` | 新規 | GoogleTranslator |
| `livecap_core/translation/impl/opus_mt.py` | 新規 | OpusMTTranslator |
| `livecap_core/translation/impl/riva_instruct.py` | 新規 | RivaInstructTranslator |
| `livecap_core/__init__.py` | 更新 | Translation exports |
| `livecap_core/utils/__init__.py` | 更新 | VRAM 確認ユーティリティ追加 |
| `pyproject.toml` | 更新 | 依存関係追加 |
| `tests/core/translation/` | 新規 | ユニットテスト |
| `tests/integration/test_translation.py` | 新規 | 統合テスト |
| `tests/conftest.py` | 更新 | テストマーカー追加 |

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| Google Translate レート制限 | 高頻度使用で失敗 | リトライ + バックオフ |
| OPUS-MT モデル変換失敗 | 初回起動が遅い | 事前変換済みモデル提供 |
| Riva-4B VRAM 不足 | GPU 8GB 必要 | 明確なエラーメッセージ + 警告 |
| ASR + Riva-4B 同時ロード | VRAM 超過 | OPUS-MT CPU デフォルト、構成ガイド |
| 文脈抽出の精度 | 翻訳結果から対象文を特定困難 | 区切り文字の工夫 |

## 完了条件

- [ ] BaseTranslator ABC が定義されている
- [ ] TranslatorFactory が動作する
- [ ] GoogleTranslator が動作する
- [ ] OpusMTTranslator が動作する（モデルロード含む）
- [ ] RivaInstructTranslator が動作する（GPU 環境）
- [ ] 文脈挿入が全エンジンで機能する
- [ ] `TranslationResult.to_event_dict()` が既存イベント型に変換できる
- [ ] VRAM 確認ユーティリティが追加されている
- [ ] VRAM 不足時の警告が実装されている
- [ ] ユニットテストがパスする
- [ ] 統合テストがパスする
- [ ] `livecap_core` から export されている

## 参考資料

- [deep-translator PyPI](https://pypi.org/project/deep-translator/)
- [CTranslate2 OPUS-MT Guide](https://opennmt.net/CTranslate2/guides/opus_mt.html)
- [Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)
- [nvidia/Riva-Translate-4B-Instruct](https://huggingface.co/nvidia/Riva-Translate-4B-Instruct)
- [Google Cloud Translation](https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-translation-ai)

---

**作成日**: 2025-12-11
**Issue**: #72
**Phase**: 4 (翻訳機能)
