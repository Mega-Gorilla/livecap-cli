#!/usr/bin/env python3
"""ローカル翻訳の例 (OPUS-MT).

OPUS-MT モデルを使ったローカル翻訳のサンプルです。
インターネット不要（初回ダウンロード時のみ必要）。

必要な依存関係:
    pip install livecap-core[translation-local]

使用方法:
    python examples/translation/local_translation.py

    # カスタムテキストを指定
    python examples/translation/local_translation.py "翻訳したいテキスト"

環境変数:
    LIVECAP_SOURCE_LANG: ソース言語、デフォルト: ja
    LIVECAP_TARGET_LANG: ターゲット言語、デフォルト: en
    LIVECAP_DEVICE: 推論デバイス (cpu/cuda)、デフォルト: cpu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """メイン処理."""
    # 設定を環境変数から取得
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    device = os.getenv("LIVECAP_DEVICE", "cpu")

    # テキストを取得
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "こんにちは、世界！今日はいい天気ですね。"

    print("=== Local Translation Example (OPUS-MT) ===")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Device: {device}")
    print(f"Input text: {text}")
    print()

    # インポート
    try:
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[translation-local]")
        sys.exit(1)

    # OPUS-MT で翻訳
    print("Initializing OPUS-MT Translator...")
    print(f"  (First run will download the model for {source_lang}->{target_lang})")
    print()

    try:
        translator = TranslatorFactory.create_translator(
            "opus_mt",
            source_lang=source_lang,
            target_lang=target_lang,
            device=device,
        )
    except NotImplementedError:
        print("Error: OPUS-MT dependencies not installed.")
        print("Please install: pip install livecap-core[translation-local]")
        sys.exit(1)

    # モデルをロード（初回はダウンロードが発生）
    print("Loading model...")
    translator.load_model()
    print(f"Translator ready: {translator.get_translator_name()}")
    print()

    # 翻訳実行
    print("=== Translation Result ===")
    try:
        result = translator.translate(text, source_lang, target_lang)
        print(f"Original ({source_lang}): {result.original_text}")
        print(f"Translated ({target_lang}): {result.text}")
    except Exception as e:
        print(f"Error during translation: {e}")
        translator.cleanup()
        sys.exit(1)

    print()

    # 文脈付き翻訳のデモ
    print("=== Translation with Context ===")
    context = [
        "昨日はVRChatで友達とドライブした。",
        "彼はとてもスピードを出した。",
    ]
    current_text = "そのせいで今日は少し疲れている。"

    print("Context:")
    for i, ctx in enumerate(context):
        print(f"  [{i+1}] {ctx}")
    print(f"Current: {current_text}")
    print()

    try:
        result = translator.translate(
            current_text,
            source_lang,
            target_lang,
            context=context,
        )
        print(f"Translated: {result.text}")
    except Exception as e:
        print(f"Error: {e}")

    # クリーンアップ
    translator.cleanup()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
