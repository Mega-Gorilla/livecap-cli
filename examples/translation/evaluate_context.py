#!/usr/bin/env python3
"""文脈挿入方式の評価スクリプト.

各翻訳エンジンで文脈が正しく機能しているか評価します。

評価ポイント:
1. 代名詞の解決（「彼」「それ」など）
2. 話題の継続性
3. 文脈なしとの比較
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from livecap_cli.translation import TranslatorFactory


def evaluate_context_translation(translator_name: str, **kwargs) -> None:
    """文脈挿入方式の評価."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {translator_name}")
    print("=" * 60)

    try:
        translator = TranslatorFactory.create_translator(translator_name, **kwargs)
        if hasattr(translator, "load_model") and not translator.is_initialized():
            print("Loading model...")
            translator.load_model()
    except NotImplementedError:
        print(f"  SKIP: {translator_name} dependencies not installed")
        return
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"Translator: {translator.get_translator_name()}")
    print()

    # テストケース: 代名詞解決が必要な文
    test_cases = [
        {
            "name": "代名詞解決テスト (彼)",
            "context": [
                "田中さんは毎日ジムに通っている。",
                "彼はマラソンの大会に出る予定だ。",
            ],
            "text": "彼は来月の東京マラソンに参加する。",
            "note": "「彼」が「田中さん」を指すことが文脈から分かるか",
        },
        {
            "name": "代名詞解決テスト (それ)",
            "context": [
                "新しいプロジェクトが始まった。",
                "チームは3ヶ月でそれを完成させる必要がある。",
            ],
            "text": "それは非常に挑戦的な目標だ。",
            "note": "「それ」が「プロジェクト/目標」を指すことが分かるか",
        },
        {
            "name": "話題継続性テスト",
            "context": [
                "昨日、渋谷で新しいカフェを見つけた。",
                "コーヒーがとても美味しかった。",
            ],
            "text": "また行きたいと思う。",
            "note": "「また行きたい」が「カフェ」に対することが分かるか",
        },
        {
            "name": "専門用語文脈テスト",
            "context": [
                "このモデルはTransformerアーキテクチャを採用している。",
                "Attention機構により長距離依存を捉えられる。",
            ],
            "text": "これにより翻訳品質が大幅に向上した。",
            "note": "技術的文脈での「これ」の解釈",
        },
    ]

    for case in test_cases:
        print(f"--- {case['name']} ---")
        print(f"Note: {case['note']}")
        print()
        print("Context:")
        for i, ctx in enumerate(case["context"]):
            print(f"  [{i+1}] {ctx}")
        print(f"Text: {case['text']}")
        print()

        # 文脈なしで翻訳
        try:
            result_no_ctx = translator.translate(case["text"], "ja", "en")
            print(f"Without context: {result_no_ctx.text}")
        except Exception as e:
            print(f"Without context: ERROR - {e}")

        # 文脈ありで翻訳
        try:
            result_with_ctx = translator.translate(
                case["text"], "ja", "en", context=case["context"]
            )
            print(f"With context:    {result_with_ctx.text}")
        except Exception as e:
            print(f"With context: ERROR - {e}")

        print()

    # クリーンアップ
    if hasattr(translator, "cleanup"):
        translator.cleanup()


def main() -> None:
    """メイン処理."""
    print("=" * 60)
    print("文脈挿入方式 評価スクリプト")
    print("=" * 60)
    print()
    print("各翻訳エンジンで文脈が翻訳品質に影響するか確認します。")
    print("特に代名詞の解決と話題の継続性に注目します。")

    # Google Translate
    evaluate_context_translation("google")

    # OPUS-MT
    evaluate_context_translation("opus_mt", source_lang="ja", target_lang="en")

    # Riva-4B (GPU必要、スキップ可能)
    import os
    if os.getenv("EVALUATE_RIVA", "0") == "1":
        evaluate_context_translation("riva_instruct", device="cuda")
    else:
        print("\n" + "=" * 60)
        print("Skipping: riva_instruct")
        print("=" * 60)
        print("  Set EVALUATE_RIVA=1 to evaluate Riva-4B (requires ~8GB VRAM)")

    print()
    print("=" * 60)
    print("評価完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
