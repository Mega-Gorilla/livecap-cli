#!/usr/bin/env python3
"""カスタム VAD 設定を使った文字起こしの例.

VADConfig をカスタマイズして、様々な環境に対応するサンプルです。
ノイズの多い環境や、静かな環境での最適な設定を示します。

使用方法:
    python examples/realtime/custom_vad_config.py [audio_file] [--profile PROFILE]

    # デフォルト設定
    python examples/realtime/custom_vad_config.py

    # ノイズ環境向け（厳しめの設定）
    python examples/realtime/custom_vad_config.py --profile noisy

    # 静かな環境向け（緩めの設定）
    python examples/realtime/custom_vad_config.py --profile quiet

    # 設定一覧を表示
    python examples/realtime/custom_vad_config.py --list-profiles

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t_base
    LIVECAP_LANGUAGE: 入力言語、デフォルト: ja
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


# VAD 設定プロファイル
VAD_PROFILES = {
    "default": {
        "description": "デフォルト設定（一般的な環境向け）",
        "config": {
            "threshold": 0.5,
            "min_speech_ms": 250,
            "min_silence_ms": 100,
            "speech_pad_ms": 100,
        },
    },
    "noisy": {
        "description": "ノイズ環境向け（厳しめの設定）",
        "config": {
            "threshold": 0.7,       # 高めの閾値でノイズを排除
            "min_speech_ms": 400,   # 長めの音声継続時間を要求
            "min_silence_ms": 300,  # 長めの無音で区切り
            "speech_pad_ms": 50,    # パディングを短く
        },
    },
    "quiet": {
        "description": "静かな環境向け（緩めの設定）",
        "config": {
            "threshold": 0.3,       # 低めの閾値で小さな声も拾う
            "min_speech_ms": 150,   # 短い発話も検出
            "min_silence_ms": 80,   # 短い間も継続とみなす
            "speech_pad_ms": 150,   # パディングを長めに
        },
    },
    "fast": {
        "description": "低遅延設定（リアルタイム重視）",
        "config": {
            "threshold": 0.5,
            "min_speech_ms": 150,   # 短めの最小音声時間
            "min_silence_ms": 50,   # 短い無音で即座に区切り
            "speech_pad_ms": 30,    # 最小限のパディング
        },
    },
    "accurate": {
        "description": "高精度設定（文字起こし精度重視）",
        "config": {
            "threshold": 0.5,
            "min_speech_ms": 500,   # 十分な長さの音声を収集
            "min_silence_ms": 500,  # しっかりした区切りで分割
            "speech_pad_ms": 200,   # 余裕のあるパディング
        },
    },
}


def list_profiles() -> None:
    """利用可能なプロファイルを一覧表示."""
    print("=== Available VAD Profiles ===")
    print()
    for name, profile in VAD_PROFILES.items():
        print(f"  {name}:")
        print(f"    {profile['description']}")
        print(f"    Config: {profile['config']}")
        print()


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(
        description="Custom VAD configuration example"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to audio file (default: test file)",
    )
    parser.add_argument(
        "--profile",
        choices=list(VAD_PROFILES.keys()),
        default="default",
        help="VAD profile to use",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available VAD profiles",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold value",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=None,
        help="Override min_speech_ms value",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=None,
        help="Override min_silence_ms value",
    )

    args = parser.parse_args()

    if args.list_profiles:
        list_profiles()
        return

    # 設定を環境変数から取得
    device = os.getenv("LIVECAP_DEVICE", "cuda")
    engine_type = os.getenv("LIVECAP_ENGINE", "whispers2t_base")
    language = os.getenv("LIVECAP_LANGUAGE", "ja")

    # 音声ファイルのパスを取得
    if args.audio_file:
        audio_path = Path(args.audio_file)
    else:
        audio_path = ROOT / "tests" / "assets" / "audio" / "jsut_basic5000_0001_ja.wav"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # インポート
    try:
        from livecap_core import FileSource, StreamTranscriber, VADConfig
        from livecap_core.config.defaults import get_default_config
        from engines.engine_factory import EngineFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch]")
        sys.exit(1)

    # プロファイルから VAD 設定を作成
    profile = VAD_PROFILES[args.profile]
    vad_config_dict = profile["config"].copy()

    # コマンドライン引数でオーバーライド
    if args.threshold is not None:
        vad_config_dict["threshold"] = args.threshold
    if args.min_speech_ms is not None:
        vad_config_dict["min_speech_ms"] = args.min_speech_ms
    if args.min_silence_ms is not None:
        vad_config_dict["min_silence_ms"] = args.min_silence_ms

    vad_config = VADConfig(**vad_config_dict)

    print(f"=== Custom VAD Config Example ===")
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Language: {language}")
    print(f"Profile: {args.profile} - {profile['description']}")
    print()
    print(f"VAD Configuration:")
    print(f"  threshold: {vad_config.threshold}")
    print(f"  min_speech_ms: {vad_config.min_speech_ms}")
    print(f"  min_silence_ms: {vad_config.min_silence_ms}")
    print(f"  speech_pad_ms: {vad_config.speech_pad_ms}")
    print()

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
        sys.exit(1)

    # 文字起こし実行
    print()
    print("=== Transcription Results ===")
    print()

    segment_count = 0

    try:
        with StreamTranscriber(
            engine=engine,
            vad_config=vad_config,  # カスタム VAD 設定を使用
            source_id="custom-vad",
        ) as transcriber:
            with FileSource(str(audio_path)) as source:
                for result in transcriber.transcribe_sync(source):
                    segment_count += 1
                    print(f"[Segment {segment_count}]")
                    print(f"  Time: {result.start_time:.2f}s - {result.end_time:.2f}s")
                    print(f"  Duration: {result.end_time - result.start_time:.2f}s")
                    print(f"  Text: {result.text}")
                    print(f"  Confidence: {result.confidence:.2f}")
                    print()

    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()

    print(f"=== Summary ===")
    print(f"Total segments: {segment_count}")
    print()
    print("Tips:")
    print("  - More segments = more frequent splits (shorter min_silence_ms)")
    print("  - Fewer segments = longer continuous speech (longer min_silence_ms)")
    print("  - Try different profiles: --profile noisy, --profile quiet, etc.")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
