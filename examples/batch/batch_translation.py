#!/usr/bin/env python3
"""バッチ翻訳の例 (Phase 6a).

FileTranscriptionPipeline に translator を統合し、
ファイル単位の ASR + 翻訳パイプラインを実行します。

使用方法:
    python examples/batch/batch_translation.py [audio_file ...]

    # デフォルト（日本語テストファイル → 英語翻訳）
    python examples/batch/batch_translation.py

    # 複数ファイルを指定
    python examples/batch/batch_translation.py file1.wav file2.mp3

    # 英語 → 日本語翻訳
    LIVECAP_SOURCE_LANG=en LIVECAP_TARGET_LANG=ja python examples/batch/batch_translation.py

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t
    LIVECAP_SOURCE_LANG: 入力言語（翻訳元）、デフォルト: ja
    LIVECAP_TARGET_LANG: 出力言語（翻訳先）、デフォルト: en
    LIVECAP_MODEL_SIZE: WhisperS2T のモデルサイズ、デフォルト: base
    LIVECAP_TRANSLATOR: 翻訳エンジン（google/opus_mt/riva_instruct）、デフォルト: google
    LIVECAP_WRITE_TRANSLATED_SRT: 翻訳 SRT 出力（true/false）、デフォルト: true
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
    write_translated_srt = os.getenv("LIVECAP_WRITE_TRANSLATED_SRT", "true").lower() == "true"

    # 音声ファイルのパスを取得
    if len(sys.argv) > 1:
        audio_paths = [Path(p) for p in sys.argv[1:]]
    else:
        # デフォルト: 日本語テストファイル
        audio_paths = [ROOT / "tests" / "assets" / "audio" / "ja" / "jsut_basic5000_0001.wav"]

    # ファイル存在チェック
    for audio_path in audio_paths:
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

    print("=== Batch Translation Example (Phase 6a) ===")
    print(f"Files: {len(audio_paths)}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Translator: {translator_type}")
    print(f"Write translated SRT: {write_translated_srt}")
    print()

    # インポート（遅延インポートでエラーメッセージを明確に）
    try:
        from livecap_cli import FileTranscriptionPipeline
        from livecap_cli.engines.engine_factory import EngineFactory
        from livecap_cli.translation import TranslatorFactory
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

    # segment_transcriber コールバック作成
    def segment_transcriber(audio, sample_rate):
        text, confidence = engine.transcribe(audio, sample_rate)
        return text

    # パイプライン実行
    print()
    print("=== Processing Files ===")
    print()

    try:
        pipeline = FileTranscriptionPipeline()

        for audio_path in audio_paths:
            print(f"Processing: {audio_path.name}")

            result = pipeline.process_file(
                audio_path,
                segment_transcriber=segment_transcriber,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                write_subtitles=True,
                write_translated_subtitles=write_translated_srt,
            )

            if result.success:
                print(f"  Status: Success")
                print(f"  Segments: {len(result.subtitles)}")
                if result.output_path:
                    print(f"  SRT: {result.output_path}")
                if result.metadata.get("translated_srt_path"):
                    print(f"  Translated SRT: {result.metadata['translated_srt_path']}")
                print()

                # 結果を表示
                for segment in result.subtitles:
                    print(f"  [{segment.start:6.2f}s - {segment.end:6.2f}s]")
                    print(f"    [{source_lang.upper()}] {segment.text}")
                    if segment.translated_text:
                        print(f"    [{target_lang.upper()}] {segment.translated_text}")
                    else:
                        print(f"    [{target_lang.upper()}] (translation unavailable)")
                    print()
            else:
                print(f"  Status: Failed")
                print(f"  Error: {result.error}")
                print()

        pipeline.close()

    except Exception as e:
        print(f"Error during processing: {e}")
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
