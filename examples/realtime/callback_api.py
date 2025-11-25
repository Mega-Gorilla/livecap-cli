#!/usr/bin/env python3
"""コールバック API を使った文字起こしの例.

feed_audio() とコールバックを使った低レベル API のサンプルです。
GUI アプリケーションなど、イベント駆動型の処理に適しています。

使用方法:
    python examples/realtime/callback_api.py [audio_file]

    # デフォルト（日本語テストファイル）
    python examples/realtime/callback_api.py

環境変数:
    LIVECAP_DEVICE: 使用するデバイス（cuda/cpu）、デフォルト: cuda
    LIVECAP_ENGINE: 使用するエンジン、デフォルト: whispers2t_base
    LIVECAP_LANGUAGE: 入力言語、デフォルト: ja

注意:
    feed_audio() は VAD でセグメントを検出した際に engine.transcribe() を
    呼び出すため、ブロッキングが発生します。完全な非同期処理が必要な場合は
    async_microphone.py の transcribe_async() を参照してください。
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
        audio_path = ROOT / "tests" / "assets" / "audio" / "jsut_basic5000_0001_ja.wav"

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"=== Callback API Example ===")
    print(f"Audio file: {audio_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_type}")
    print(f"Language: {language}")
    print()

    # インポート
    try:
        from livecap_core import (
            FileSource,
            InterimResult,
            StreamTranscriber,
            TranscriptionResult,
        )
        from livecap_core.config.defaults import get_default_config
        from engines.engine_factory import EngineFactory
    except ImportError as e:
        print(f"Error: Required module not found: {e}")
        print("Please install: pip install livecap-core[vad,engines-torch]")
        sys.exit(1)

    # 結果を収集するリスト
    final_results: list[TranscriptionResult] = []
    interim_results: list[InterimResult] = []

    # コールバック関数を定義
    def on_result(result: TranscriptionResult) -> None:
        """確定結果を受け取るコールバック."""
        final_results.append(result)
        print(f"[FINAL] [{result.start_time:6.2f}s - {result.end_time:6.2f}s] {result.text}")

    def on_interim(interim: InterimResult) -> None:
        """中間結果を受け取るコールバック."""
        interim_results.append(interim)
        print(f"[INTERIM] {interim.text} (accumulated: {interim.accumulated_time:.2f}s)")

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

    # StreamTranscriber の設定
    transcriber = StreamTranscriber(
        engine=engine,
        source_id="callback-example",
    )

    # コールバックを登録
    transcriber.set_callbacks(
        on_result=on_result,
        on_interim=on_interim,
    )

    print()
    print("=== Processing Audio ===")
    print()

    try:
        # 音声ファイルを読み込み、チャンクごとに feed_audio() を呼び出す
        with FileSource(str(audio_path)) as source:
            chunk_count = 0
            for chunk in source:
                chunk_count += 1
                # feed_audio() は VAD でセグメント検出時にブロッキングする
                transcriber.feed_audio(chunk, source.sample_rate)

            print(f"Processed {chunk_count} chunks")

        # 最終セグメントを処理
        print()
        print("=== Finalizing ===")
        final = transcriber.finalize()
        if final:
            print(f"[FINAL] [{final.start_time:6.2f}s - {final.end_time:6.2f}s] {final.text}")
            final_results.append(final)

    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)
    finally:
        transcriber.close()
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()

    # サマリー表示
    print()
    print("=== Summary ===")
    print(f"Final results: {len(final_results)}")
    print(f"Interim results: {len(interim_results)}")

    if final_results:
        print()
        print("Full transcription:")
        full_text = " ".join(r.text for r in final_results)
        print(f"  {full_text}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
