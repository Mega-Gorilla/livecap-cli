#!/usr/bin/env python3
"""文脈挿入方式のデバッグスクリプト.

翻訳前後のテキストを詳細に表示して問題を特定します。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def debug_opus_mt():
    """OPUS-MT の文脈処理をデバッグ."""
    print("=" * 60)
    print("OPUS-MT Context Debug")
    print("=" * 60)

    try:
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: {e}")
        return

    try:
        translator = TranslatorFactory.create_translator(
            "opus_mt", source_lang="ja", target_lang="en"
        )
        translator.load_model()
    except NotImplementedError:
        print("OPUS-MT not available")
        return

    context = ["田中さんは毎日ジムに通っている。", "彼はマラソンの大会に出る予定だ。"]
    text = "彼は来月の東京マラソンに参加する。"

    # 連結されたテキストを確認
    ctx = context[-translator._default_context_sentences :]
    full_text = "\n".join(ctx) + "\n" + text

    print("\n--- Input Analysis ---")
    print(f"Context sentences: {translator._default_context_sentences}")
    print(f"Context (limited): {ctx}")
    print()
    print("Full text (repr):")
    print(repr(full_text))
    print()
    print("Full text (visual):")
    print(full_text)
    print()

    # トークナイズして翻訳
    source_tokens = translator._tokenizer.convert_ids_to_tokens(
        translator._tokenizer.encode(full_text)
    )
    print(f"Source tokens ({len(source_tokens)}): {source_tokens[:20]}...")
    print()

    results = translator._model.translate_batch([source_tokens])
    target_tokens = results[0].hypotheses[0]
    print(f"Target tokens ({len(target_tokens)}): {target_tokens[:20]}...")
    print()

    raw_result = translator._tokenizer.decode(
        translator._tokenizer.convert_tokens_to_ids(target_tokens),
        skip_special_tokens=True,
    )
    print("--- Raw Translation (before extraction) ---")
    print(f"repr: {repr(raw_result)}")
    print()
    print("visual:")
    print(raw_result)
    print()

    # _extract_relevant_part の動作確認
    lines = raw_result.strip().split("\n")
    print(f"--- Split by newlines: {len(lines)} lines ---")
    for i, line in enumerate(lines):
        print(f"  [{i}] {repr(line)}")
    print()

    extracted = lines[-1] if lines else raw_result
    print(f"Extracted (last line): {repr(extracted)}")
    print()

    # 最終結果
    result = translator.translate(text, "ja", "en", context=context)
    print("--- Final Result ---")
    print(f"result.text: {result.text}")

    translator.cleanup()


def debug_google():
    """Google Translate の文脈処理をデバッグ."""
    print("\n" + "=" * 60)
    print("Google Translate Context Debug")
    print("=" * 60)

    try:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: {e}")
        return

    translator = TranslatorFactory.create_translator("google")

    context = ["田中さんは毎日ジムに通っている。", "彼はマラソンの大会に出る予定だ。"]
    text = "彼は来月の東京マラソンに参加する。"

    ctx = context[-translator._default_context_sentences :]
    full_text = "\n".join(ctx) + "\n" + text

    print("\n--- Input Analysis ---")
    print(f"Context sentences: {translator._default_context_sentences}")
    print()
    print("Full text (repr):")
    print(repr(full_text))
    print()
    print("Full text (visual):")
    print(full_text)
    print()

    # Google Translate 直接呼び出し
    gt = DeepGoogleTranslator(source="ja", target="en")
    raw_result = gt.translate(full_text)

    print("--- Raw Translation (before extraction) ---")
    print(f"repr: {repr(raw_result)}")
    print()
    print("visual:")
    print(raw_result)
    print()

    lines = raw_result.strip().split("\n")
    print(f"--- Split by newlines: {len(lines)} lines ---")
    for i, line in enumerate(lines):
        print(f"  [{i}] {repr(line)}")
    print()

    extracted = lines[-1] if lines else raw_result
    print(f"Extracted (last line): {repr(extracted)}")


if __name__ == "__main__":
    debug_opus_mt()
    debug_google()
