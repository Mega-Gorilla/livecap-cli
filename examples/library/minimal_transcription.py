#!/usr/bin/env python3
"""最小限の文字起こしサンプル

livecap-cli をライブラリとして使用する最小限の例です。

Usage:
    # ファイル文字起こし
    python examples/library/minimal_transcription.py

    # マイク入力（Ctrl+C で停止）
    python examples/library/minimal_transcription.py --mic

    # デバイス指定
    python examples/library/minimal_transcription.py --mic --device 0
"""

import argparse
import os
import sys

# テスト用音声ファイルのパス
DEFAULT_AUDIO = "tests/assets/audio/ja/jsut_basic5000_0001.wav"


def transcribe_file(audio_path: str, engine_type: str = "whispers2t", device: str = "auto"):
    """ファイルを文字起こし"""
    from livecap_cli import EngineFactory, FileSource, StreamTranscriber

    # エンジン作成
    print(f"Loading engine: {engine_type}...")
    device_str = None if device == "auto" else device
    engine = EngineFactory.create_engine(engine_type, device=device_str, model_size="base")
    engine.load_model()

    # ファイルソース作成
    audio_source = FileSource(audio_path)

    # 文字起こし
    print(f"Transcribing: {audio_path}\n")
    with StreamTranscriber(engine=engine) as transcriber:
        with audio_source:
            for result in transcriber.transcribe_sync(audio_source):
                print(f"[{result.start_time:.2f}s - {result.end_time:.2f}s] {result.text}")

    print("\nDone.")


def transcribe_microphone(mic_device: int | None, engine_type: str = "whispers2t", compute_device: str = "auto"):
    """マイク入力を文字起こし"""
    from livecap_cli import EngineFactory, MicrophoneSource, StreamTranscriber

    # エンジン作成
    print(f"Loading engine: {engine_type}...")
    device_str = None if compute_device == "auto" else compute_device
    engine = EngineFactory.create_engine(engine_type, device=device_str, model_size="base")
    engine.load_model()

    # マイクソース作成
    print(f"Using microphone: {mic_device if mic_device is not None else 'default'}")
    print("Press Ctrl+C to stop.\n")

    try:
        with StreamTranscriber(engine=engine) as transcriber:
            with MicrophoneSource(device=mic_device) as mic:
                for result in transcriber.transcribe_sync(mic):
                    print(f"[{result.start_time:.2f}s] {result.text}")
    except KeyboardInterrupt:
        print("\n\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Minimal transcription example")
    parser.add_argument("audio_file", nargs="?", default=None, help="Audio file to transcribe")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    parser.add_argument("--engine", default="whispers2t", help="ASR engine (default: whispers2t)")
    parser.add_argument("--compute-device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")

    args = parser.parse_args()

    if args.mic:
        transcribe_microphone(args.device, args.engine, args.compute_device)  # args.device -> mic_device
    else:
        audio_path = args.audio_file or DEFAULT_AUDIO
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        transcribe_file(audio_path, args.engine, args.compute_device)


if __name__ == "__main__":
    main()
