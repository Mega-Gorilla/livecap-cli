#!/usr/bin/env python3
"""音声ファイル（MP3/WAV等）の翻訳パイプライン.

FileTranscriptionPipeline を使って音声ファイルを文字起こし + 翻訳し、
SRT字幕ファイル（原文 + 翻訳版）を出力します。

必要な依存関係:
    pip install livecap-core[engines-torch,translation]

使用方法:
    # 単一ファイル
    python examples/translation/file_translation.py audio.mp3

    # 複数ファイル
    python examples/translation/file_translation.py file1.mp3 file2.wav file3.m4a

    # 英語 → 日本語翻訳
    LIVECAP_SOURCE_LANG=en LIVECAP_TARGET_LANG=ja python examples/translation/file_translation.py audio.mp3

    # 翻訳SRTを出力しない
    python examples/translation/file_translation.py --no-translated-srt audio.mp3

環境変数:
    LIVECAP_SOURCE_LANG: ソース言語（音声の言語）、デフォルト: ja
    LIVECAP_TARGET_LANG: 翻訳先言語、デフォルト: en
    LIVECAP_ENGINE: ASR エンジン、デフォルト: whispers2t
    LIVECAP_DEVICE: 推論デバイス、デフォルト: cuda
    LIVECAP_MODEL_SIZE: WhisperS2T のモデルサイズ、デフォルト: base
    LIVECAP_TRANSLATOR: 翻訳エンジン（google/opus_mt）、デフォルト: google

出力ファイル:
    audio.mp3 → audio.srt (原文) + audio_en.srt (翻訳)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Translate audio files (MP3/WAV/etc.) with SRT subtitle output"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Audio files to translate (MP3, WAV, M4A, etc.)",
    )
    parser.add_argument(
        "--no-translated-srt",
        action="store_true",
        help="Don't output translated SRT file",
    )
    parser.add_argument(
        "--no-original-srt",
        action="store_true",
        help="Don't output original language SRT file",
    )

    args = parser.parse_args()

    # ファイル存在チェック
    audio_files = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        audio_files.append(path)

    # 設定を環境変数から取得
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t")
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    model_size = os.getenv("LIVECAP_MODEL_SIZE", "base")
    translator_type = os.getenv("LIVECAP_TRANSLATOR", "google")

    print("=" * 60)
    print("File Translation Pipeline")
    print("=" * 60)
    print(f"Files: {len(audio_files)}")
    for f in audio_files:
        print(f"  - {f.name}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"ASR engine: {engine_type}")
    print(f"Translator: {translator_type}")
    print(f"Device: {device}")
    print(f"Output original SRT: {not args.no_original_srt}")
    print(f"Output translated SRT: {not args.no_translated_srt}")
    print()

    # インポート
    try:
        from livecap_core import FileTranscriptionPipeline
        from livecap_core.engines.engine_factory import EngineFactory
        from livecap_core.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[engines-torch,translation]")
        sys.exit(1)

    # ASR エンジンの初期化
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
        print(f"  Engine: {engine.get_engine_name()}")
    except Exception as e:
        print(f"Error initializing ASR engine: {e}")
        print("Try: LIVECAP_DEVICE=cpu python ...")
        sys.exit(1)

    # 翻訳エンジンの初期化
    print("Initializing translator...")
    translator_options = {}
    if translator_type == "opus_mt":
        translator_options["source_lang"] = source_lang
        translator_options["target_lang"] = target_lang
        translator_options["device"] = "cpu"

    try:
        translator = TranslatorFactory.create_translator(
            translator_type,
            **translator_options,
        )
        if translator_type in ("opus_mt", "riva_instruct"):
            translator.load_model()
        print(f"  Translator: {translator.get_translator_name()}")
    except Exception as e:
        print(f"Error initializing translator: {e}")
        engine.cleanup()
        sys.exit(1)

    print()

    # segment_transcriber コールバック
    def segment_transcriber(audio, sample_rate):
        text, confidence = engine.transcribe(audio, sample_rate)
        return text

    # パイプライン処理
    print("=" * 60)
    print("Processing Files")
    print("=" * 60)
    print()

    pipeline = FileTranscriptionPipeline()
    results = []

    try:
        for i, audio_path in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_path.name}")
            print("-" * 40)

            result = pipeline.process_file(
                audio_path,
                segment_transcriber=segment_transcriber,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                write_subtitles=not args.no_original_srt,
                write_translated_subtitles=not args.no_translated_srt,
            )

            results.append(result)

            if result.success:
                print(f"  Status: Success")
                print(f"  Segments: {len(result.subtitles)}")

                if result.output_path:
                    print(f"  Original SRT: {result.output_path}")
                if result.metadata.get("translated_srt_path"):
                    print(f"  Translated SRT: {result.metadata['translated_srt_path']}")

                # セグメント詳細を表示
                print()
                for seg in result.subtitles:
                    print(f"  [{seg.start:6.2f}s - {seg.end:6.2f}s]")
                    print(f"    [{source_lang.upper()}] {seg.text}")
                    if seg.translated_text:
                        print(f"    [{target_lang.upper()}] {seg.translated_text}")
                    else:
                        print(f"    [{target_lang.upper()}] (unavailable)")
            else:
                print(f"  Status: Failed")
                print(f"  Error: {result.error}")

            print()

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.close()
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        cleanup = getattr(translator, "cleanup", None)
        if callable(cleanup):
            cleanup()

    # サマリー表示
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"Total: {len(results)} files")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print()

    if success_count > 0:
        print("Output files:")
        for r in results:
            if r.success:
                if r.output_path:
                    print(f"  {r.output_path}")
                if r.metadata.get("translated_srt_path"):
                    print(f"  {r.metadata['translated_srt_path']}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
