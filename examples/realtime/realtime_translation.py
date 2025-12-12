#!/usr/bin/env python3
"""リアルタイム翻訳の例.

StreamTranscriber に translator を統合し、ASR + 翻訳のパイプラインを実行します。
音声ファイルを読み込み、文字起こしと同時に翻訳を行います。

使用方法:
    python examples/realtime/realtime_translation.py [audio_file]

    # デフォルト（日本語テストファイル → 英語翻訳）
    python examples/realtime/realtime_translation.py

    # 特定のファイルを指定
    python examples/realtime/realtime_translation.py path/to/audio.wav

    # 英語 → 日本語翻訳
    LIVECAP_SOURCE_LANG=en LIVECAP_TARGET_LANG=ja python examples/realtime/realtime_translation.py

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t
    LIVECAP_SOURCE_LANG: 入力言語（翻訳元）、デフォルト: ja
    LIVECAP_TARGET_LANG: 出力言語（翻訳先）、デフォルト: en
    LIVECAP_MODEL_SIZE: WhisperS2Tのモデルサイズ、デフォルト: base
    LIVECAP_TRANSLATOR: 翻訳エンジン（google/opus_mt/riva_instruct）、デフォルト: google
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
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t")
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    model_size = os.getenv("LIVECAP_MODEL_SIZE", "base")
    translator_type = os.getenv("LIVECAP_TRANSLATOR", "google")

    # 音声ファイルのパスを取得
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        # デフォルト: 日本語テストファイル
        audio_path = ROOT / "tests" / "assets" / "audio" / "ja" / "jsut_basic5000_0001.wav"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=== Realtime Translation Example ===")
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Translator: {translator_type}")
    print()

    # インポート（遅延インポートでエラーメッセージを明確に）
    try:
        from livecap_core import FileSource, StreamTranscriber
        from livecap_core.engines.engine_factory import EngineFactory
        from livecap_core.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch,translation]")
        sys.exit(1)

    # ASR エンジン初期化
    print("Initializing ASR engine...")

    engine_options = {}
    if engine_type in ("whispers2t", "canary", "voxtral"):
        engine_options["language"] = source_lang
    if engine_type == "whispers2t":
        engine_options["model_size"] = model_size

    try:
        engine = EngineFactory.create_engine(
            engine_type=engine_type,
            device=device,
            **engine_options,
        )
        engine.load_model()
        print(f"Engine loaded: {engine.get_engine_name()}")
    except Exception as e:
        print(f"Error: Failed to initialize engine: {e}")
        print("Try running with LIVECAP_DEVICE=cpu")
        sys.exit(1)

    # 翻訳エンジン初期化
    print("Initializing translator...")

    translator_options = {}
    if translator_type == "opus_mt":
        translator_options["source_lang"] = source_lang
        translator_options["target_lang"] = target_lang
        translator_options["device"] = "cpu"  # OPUS-MT は CPU で十分高速

    try:
        translator = TranslatorFactory.create_translator(
            translator_type,
            **translator_options,
        )
        # ローカルモデルの場合はロードが必要
        if translator_type in ("opus_mt", "riva_instruct"):
            translator.load_model()
        print(f"Translator loaded: {translator.get_translator_name()}")
    except Exception as e:
        print(f"Error: Failed to initialize translator: {e}")
        sys.exit(1)

    # 文字起こし + 翻訳実行
    print()
    print("=== Transcription + Translation Results ===")
    print()

    try:
        with StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            source_id="file",
        ) as transcriber:
            with FileSource(str(audio_path)) as source:
                for result in transcriber.transcribe_sync(source):
                    print(f"[{result.start_time:6.2f}s - {result.end_time:6.2f}s]")
                    print(f"  [{source_lang.upper()}] {result.text}")
                    if result.translated_text:
                        print(f"  [{target_lang.upper()}] {result.translated_text}")
                    else:
                        print(f"  [{target_lang.upper()}] (translation unavailable)")
                    print(f"  Confidence: {result.confidence:.2f}")
                    print()
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # クリーンアップ
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        cleanup = getattr(translator, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print("Done!")


if __name__ == "__main__":
    main()
