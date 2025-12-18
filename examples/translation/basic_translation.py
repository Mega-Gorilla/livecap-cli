#!/usr/bin/env python3
"""基本的な翻訳の例.

Google Translate を使った最小構成のサンプルです。
インターネット接続が必要です。

使用方法:
    python examples/translation/basic_translation.py

    # カスタムテキストを指定
    python examples/translation/basic_translation.py "翻訳したいテキスト"

環境変数:
    LIVECAP_SOURCE_LANG: ソース言語、デフォルト: ja
    LIVECAP_TARGET_LANG: ターゲット言語、デフォルト: en
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

    # テキストを取得
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "こんにちは、世界！今日はいい天気ですね。"

    print("=== Basic Translation Example (Google Translate) ===")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Input text: {text}")
    print()

    # インポート
    try:
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[translation]")
        sys.exit(1)

    # Google Translate で翻訳
    print("Initializing Google Translator...")
    translator = TranslatorFactory.create_translator("google")
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
        sys.exit(1)

    print()

    # 文脈付き翻訳のデモ
    print("=== Translation with Context ===")
    context = [
        "昨日はVRChatで友達とドライブした。",
        "彼はとてもスピードを出した。",
    ]
    current_text = "そのせいで今日は少し疲れている。"

    print(f"Context:")
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

    print()
    print("Done!")


if __name__ == "__main__":
    main()
