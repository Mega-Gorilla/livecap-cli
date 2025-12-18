# Issue #72: 使用例

翻訳プラグインシステムの使用例集。

> **Note**: 進捗状況については [`./issue-72/README.md`](./issue-72/README.md) を参照。

## 基本的な翻訳（Google Translate）

```python
from livecap_cli.translation import TranslatorFactory

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

## 文脈を考慮した翻訳

```python
from livecap_cli.translation import TranslatorFactory

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

## ローカル翻訳（OPUS-MT）

```python
from livecap_cli.translation import TranslatorFactory

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

## GPU 翻訳（Riva-4B-Instruct）

```python
from livecap_cli.translation import TranslatorFactory

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

## 非同期翻訳

```python
import asyncio
from livecap_cli.translation import TranslatorFactory

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

## ASR + 翻訳の組み合わせ

```python
from livecap_cli import (
    EngineFactory,
    StreamTranscriber,
    FileSource,
)
from livecap_cli.translation import TranslatorFactory

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

## イベント型への変換

```python
from livecap_cli.translation import TranslatorFactory

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

## エラーハンドリング

```python
from livecap_cli.translation import TranslatorFactory
from livecap_cli.translation.exceptions import (
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

## Phase 5: StreamTranscriber 統合（予定）

Phase 5 実装後は、以下のようなシンプルな使用方法が可能になる予定：

```python
from livecap_cli import StreamTranscriber, EngineFactory, MicrophoneSource
from livecap_cli.translation import TranslatorFactory

# ASR エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# Translator 初期化
translator = TranslatorFactory.create_translator("google")

# StreamTranscriber に translator を渡す
with StreamTranscriber(
    engine=engine,
    translator=translator,
    source_lang="ja",
    target_lang="en",
) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.language}] {result.text}")
            if result.translated_text:
                print(f"[{result.target_language}] {result.translated_text}")

engine.cleanup()
```
