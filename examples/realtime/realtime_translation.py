#!/usr/bin/env python3
"""マイク入力からのリアルタイム翻訳の例.

MicrophoneSource と StreamTranscriber + translator を使った
リアルタイム音声認識 + 翻訳のサンプルです。

使用方法:
    python examples/realtime/realtime_translation.py

    # 特定のマイクデバイスを使用
    python examples/realtime/realtime_translation.py --device 2

    # デバイス一覧を表示
    python examples/realtime/realtime_translation.py --list-devices

    # ファイルを指定（従来の動作）
    python examples/realtime/realtime_translation.py --file path/to/audio.wav

    # 英語 → 日本語翻訳
    LIVECAP_SOURCE_LANG=en LIVECAP_TARGET_LANG=ja python examples/realtime/realtime_translation.py

    # 停止: Ctrl+C

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t
    LIVECAP_SOURCE_LANG: 入力言語（翻訳元）、デフォルト: ja
    LIVECAP_TARGET_LANG: 出力言語（翻訳先）、デフォルト: en
    LIVECAP_MODEL_SIZE: WhisperS2Tのモデルサイズ、デフォルト: base
    LIVECAP_TRANSLATOR: 翻訳エンジン（google/opus_mt/riva_instruct）、デフォルト: google

必要条件:
    - PortAudio ライブラリがインストールされていること（マイク使用時）
      Ubuntu: sudo apt-get install libportaudio2
      macOS: brew install portaudio
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def list_devices() -> None:
    """利用可能なオーディオデバイスを一覧表示."""
    try:
        from livecap_cli import MicrophoneSource
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install livecap-core[vad]")
        sys.exit(1)
    except OSError as e:
        print(f"Error: PortAudio not found: {e}")
        print("Please install PortAudio:")
        print("  Ubuntu: sudo apt-get install libportaudio2")
        print("  macOS: brew install portaudio")
        sys.exit(1)

    print("=== Available Audio Devices ===")
    print()

    devices = MicrophoneSource.list_devices()
    if not devices:
        print("No input devices found.")
        return

    for dev in devices:
        default = " (default)" if dev.is_default else ""
        print(f"  [{dev.index}] {dev.name}{default}")
        print(f"      Channels: {dev.channels}, Sample Rate: {dev.sample_rate}")
        print()


async def run_microphone_translation(device_id: int | None) -> None:
    """マイクからのリアルタイム翻訳を実行."""
    # 設定を環境変数から取得
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t")
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    model_size = os.getenv("LIVECAP_MODEL_SIZE", "large-v2")
    translator_type = os.getenv("LIVECAP_TRANSLATOR", "google")

    print("=== Realtime Microphone Translation ===")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Translator: {translator_type}")
    print(f"Microphone: {device_id if device_id is not None else 'default'}")
    print()

    # インポート
    try:
        from livecap_cli import MicrophoneSource, StreamTranscriber
        from livecap_cli.engines.engine_factory import EngineFactory
        from livecap_cli.translation import TranslatorFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch,translation]")
        return
    except OSError as e:
        print(f"Error: PortAudio not found: {e}")
        print("Please install PortAudio:")
        print("  Ubuntu: sudo apt-get install libportaudio2")
        print("  macOS: brew install portaudio")
        return

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
        return

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

        # ウォームアップ: 最初のリクエストは遅いことがあるため事前に実行
        print("Warming up translator...")
        _ = translator.translate("test", source_lang, target_lang)
        print("Translator ready.")
    except Exception as e:
        print(f"Error: Failed to initialize translator: {e}")
        return

    print()
    print("=== Listening... (Press Ctrl+C to stop) ===")
    print()

    # マイクソースへの参照（シグナルハンドラから停止するため）
    mic_ref: list[MicrophoneSource | None] = [None]

    def signal_handler(sig, frame):
        print("\n\nStopping...")
        if mic_ref[0] is not None:
            mic_ref[0].stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            source_id="microphone",
        )

        mic = MicrophoneSource(device=device_id)
        mic_ref[0] = mic
        mic.start()

        try:
            async for result in transcriber.transcribe_async(mic):
                print(f"[{result.start_time:6.2f}s]")
                print(f"  [{source_lang.upper()}] {result.text}")
                if result.translated_text:
                    print(f"  [{target_lang.upper()}] {result.translated_text}")
                else:
                    print(f"  [{target_lang.upper()}] (translation unavailable)")
                print()
        finally:
            mic.stop()

        transcriber.close()

    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        cleanup = getattr(translator, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print("Done!")


def run_file_translation(audio_path: Path) -> None:
    """ファイルからの翻訳を実行（従来の動作）."""
    # 設定を環境変数から取得
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t")
    source_lang = os.getenv("LIVECAP_SOURCE_LANG", "ja")
    target_lang = os.getenv("LIVECAP_TARGET_LANG", "en")
    model_size = os.getenv("LIVECAP_MODEL_SIZE", "base")
    translator_type = os.getenv("LIVECAP_TRANSLATOR", "google")

    print("=== Realtime File Translation ===")
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    print(f"Translator: {translator_type}")
    print()

    # インポート
    try:
        from livecap_cli import FileSource, StreamTranscriber
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
        translator_options["device"] = "cpu"

    try:
        translator = TranslatorFactory.create_translator(
            translator_type,
            **translator_options,
        )
        if translator_type in ("opus_mt", "riva_instruct"):
            translator.load_model()
        print(f"Translator loaded: {translator.get_translator_name()}")

        # ウォームアップ
        print("Warming up translator...")
        _ = translator.translate("test", source_lang, target_lang)
        print("Translator ready.")
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
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        cleanup = getattr(translator, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print("Done!")


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Realtime transcription + translation (microphone or file)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone device ID (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Audio file path (if not specified, uses microphone)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.file:
        # ファイルモード
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        run_file_translation(audio_path)
    else:
        # マイクモード（デフォルト）
        asyncio.run(run_microphone_translation(args.device))


if __name__ == "__main__":
    main()
