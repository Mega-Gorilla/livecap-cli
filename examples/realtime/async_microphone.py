#!/usr/bin/env python3
"""マイク入力からの非同期文字起こしの例.

MicrophoneSource と transcribe_async() を使った非同期処理のサンプルです。
リアルタイムでマイク入力を文字起こしします。

使用方法:
    python examples/realtime/async_microphone.py

    # 特定のデバイスを使用
    python examples/realtime/async_microphone.py --device 2

    # デバイス一覧を表示
    python examples/realtime/async_microphone.py --list-devices

    # 停止: Ctrl+C

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t_base
    LIVECAP_LANGUAGE: 入力言語、デフォルト: ja

必要条件:
    - PortAudio ライブラリがインストールされていること
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
        from livecap_core import MicrophoneSource
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


async def run_transcription(device_id: int | None) -> None:
    """非同期文字起こしを実行."""
    # 設定を環境変数から取得
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t_base")
    language = os.getenv("LIVECAP_LANGUAGE", "ja")

    print(f"=== Async Microphone Transcription ===")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Language: {language}")
    print(f"Microphone: {device_id if device_id is not None else 'default'}")
    print()

    # インポート
    try:
        from livecap_core import MicrophoneSource, StreamTranscriber
        from livecap_core.config.defaults import get_default_config
        from engines.engine_factory import EngineFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch]")
        return
    except OSError as e:
        print(f"Error: PortAudio not found: {e}")
        print("Please install PortAudio:")
        print("  Ubuntu: sudo apt-get install libportaudio2")
        print("  macOS: brew install portaudio")
        return

    # エンジン初期化
    print("Initializing engine...")
    config = get_default_config()
    config["transcription"]["engine"] = engine_type
    config["transcription"]["input_language"] = language

    try:
        engine = EngineFactory.create_engine(
            engine_type=engine_type,
            device=device,
            config=config,
        )
        engine.load_model()
        print(f"Engine loaded: {engine.get_engine_name()}")
    except Exception as e:
        print(f"Error: Failed to initialize engine: {e}")
        print("Try running with LIVECAP_DEVICE=cpu")
        return

    print()
    print("=== Listening... (Press Ctrl+C to stop) ===")
    print()

    # マイクソースへの参照（シグナルハンドラから停止するため）
    mic_ref: list[MicrophoneSource | None] = [None]

    def signal_handler(sig, frame):
        print("\n\nStopping...")
        # マイクを停止して非同期イテレータを終了させる
        if mic_ref[0] is not None:
            mic_ref[0].stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        transcriber = StreamTranscriber(engine=engine, source_id="microphone")

        mic = MicrophoneSource(device=device_id)
        mic_ref[0] = mic
        mic.start()

        try:
            async for result in transcriber.transcribe_async(mic):
                print(f"[{result.start_time:6.2f}s] {result.text}")
        finally:
            mic.stop()

        transcriber.close()

    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print()
    print("Done!")


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Microphone transcription using async API"
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

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    asyncio.run(run_transcription(args.device))


if __name__ == "__main__":
    main()
