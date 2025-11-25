#!/usr/bin/env python3
"""基本的なファイル文字起こしの例.

FileSource と StreamTranscriber を使った最小構成のサンプルです。
音声ファイルを読み込み、VAD でセグメント検出しながら文字起こしを行います。

使用方法:
    python examples/realtime/basic_file_transcription.py [audio_file]

    # デフォルト（日本語テストファイル）
    python examples/realtime/basic_file_transcription.py

    # 特定のファイルを指定
    python examples/realtime/basic_file_transcription.py path/to/audio.wav

    # 英語ファイルを指定
    LIVECAP_LANGUAGE=en python examples/realtime/basic_file_transcription.py tests/assets/audio/librispeech_test-clean_1089-134686-0001_en.wav

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t_base
    LIVECAP_LANGUAGE: 入力言語、デフォルト: ja
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
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t_base")
    language = os.getenv("LIVECAP_LANGUAGE", "ja")

    # 音声ファイルのパスを取得
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        # デフォルト: 日本語テストファイル
        audio_path = ROOT / "tests" / "assets" / "audio" / "jsut_basic5000_0001_ja.wav"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"=== Basic File Transcription Example ===")
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Language: {language}")
    print()

    # インポート（遅延インポートでエラーメッセージを明確に）
    try:
        from livecap_core import FileSource, StreamTranscriber
        from livecap_core.config.defaults import get_default_config
        from engines.engine_factory import EngineFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch]")
        sys.exit(1)

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
        sys.exit(1)

    # 文字起こし実行
    print()
    print("=== Transcription Results ===")
    print()

    try:
        with StreamTranscriber(engine=engine, source_id="file") as transcriber:
            with FileSource(str(audio_path)) as source:
                for result in transcriber.transcribe_sync(source):
                    print(f"[{result.start_time:6.2f}s - {result.end_time:6.2f}s] {result.text}")
                    print(f"  Confidence: {result.confidence:.2f}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        # クリーンアップ
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
