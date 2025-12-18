#!/usr/bin/env python3
"""音声認識 + 翻訳パイプラインの例.

音声ファイルを文字起こしして翻訳するサンプルです。
ASR エンジンと翻訳エンジンを組み合わせて使用します。

必要な依存関係:
    pip install livecap-core[engines-torch,translation]

使用方法:
    python examples/translation/transcription_with_translation.py <audio_file>

    # 例:
    python examples/translation/transcription_with_translation.py recording.wav

環境変数:
    LIVECAP_SOURCE_LANG: ソース言語（音声の言語）、デフォルト: ja
    LIVECAP_TARGET_LANG: 翻訳先言語、デフォルト: en
    LIVECAP_ENGINE: ASR エンジン、デフォルト: whispers2t_base
    LIVECAP_DEVICE: 推論デバイス、デフォルト: cuda
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
    # コマンドライン引数のチェック
    if len(sys.argv) < 2:
        print("Usage: python transcription_with_translation.py <audio_file>")
        print()
        print("Example:")
        print("  python transcription_with_translation.py recording.wav")
        sys.exit(1)

    audio_file = Path(sys.argv[1])
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # 設定を環境変数から取得
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t_base")
    device = os.getenv("LIVECAP_DEVICE", "cuda")

    print("=== Transcription + Translation Pipeline ===")
    print(f"Audio file: {audio_file}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"ASR engine: {engine_type}")
    print(f"Device: {device}")
    print()

    # インポート
    try:
        from livecap_cli import EngineFactory, FileSource
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[engines-torch,translation]")
        sys.exit(1)

    # ASR エンジンの初期化
    print("Initializing ASR engine...")
    try:
        engine = EngineFactory.create_engine(engine_type, device=device)
        engine.load_model()
        print(f"  Engine: {engine.get_engine_name()}")
    except Exception as e:
        print(f"Error initializing ASR engine: {e}")
        sys.exit(1)

    # 翻訳エンジンの初期化（Google Translate を使用）
    print("Initializing translator...")
    try:
        translator = TranslatorFactory.create_translator("google")
        print(f"  Translator: {translator.get_translator_name()}")
    except Exception as e:
        print(f"Error initializing translator: {e}")
        engine.cleanup()
        sys.exit(1)

    print()

    # 音声ファイルを読み込み
    print("=== Processing Audio ===")
    try:
        with FileSource(str(audio_file)) as source:
            # 音声データを取得
            audio_data = []
            for chunk in source:
                audio_data.append(chunk)

            # チャンクを結合
            import numpy as np

            audio = np.concatenate(audio_data)
            sample_rate = source.sample_rate

            print(f"Audio loaded: {len(audio) / sample_rate:.2f} seconds")
            print()

            # 音声認識を実行
            print("=== Transcription ===")
            text, confidence = engine.transcribe(audio, sample_rate)
            print(f"Recognized ({source_lang}): {text}")
            print(f"Confidence: {confidence:.2%}")
            print()

            # 翻訳を実行
            if text.strip():
                print("=== Translation ===")
                result = translator.translate(text, source_lang, target_lang)
                print(f"Translated ({target_lang}): {result.text}")
            else:
                print("No speech detected in audio.")

    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # クリーンアップ
        engine.cleanup()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
