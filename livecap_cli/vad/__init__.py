"""VAD (Voice Activity Detection) module for livecap_cli.

音声活動検出のためのモジュール。
複数のバックエンド（Silero, TenVAD, WebRTC）をサポート。

Usage:
    from livecap_cli.vad import VADProcessor, VADConfig

    # 言語に最適化された VAD を使用（推奨）
    processor = VADProcessor.from_language("ja")
    for chunk in audio_source:
        segments = processor.process_chunk(chunk, sample_rate=16000)
        for segment in segments:
            if segment.is_final:
                transcribe(segment.audio)

    # エンジン固有の最適化プリセットを使用
    processor = VADProcessor.from_language("ja", engine="parakeet_ja")

    # デフォルト設定（Silero VAD）
    processor = VADProcessor()

    # カスタム設定
    config = VADConfig(
        threshold=0.6,
        min_speech_ms=300,
    )
    processor = VADProcessor(config=config)

    # 別のバックエンドを使用
    from livecap_cli.vad.backends import WebRTCVAD
    processor = VADProcessor(backend=WebRTCVAD(mode=3))

Supported languages for from_language():
    - "ja" (Japanese)
    - "en" (English)
"""

from .config import VADConfig
from .presets import (
    get_available_presets,
    get_best_vad_for_language,
    get_optimized_preset,
)
from .processor import VADProcessor
from .state_machine import VADSegment, VADState, VADStateMachine

__all__ = [
    "VADConfig",
    "VADProcessor",
    "VADSegment",
    "VADState",
    "VADStateMachine",
    # Optimized presets
    "get_optimized_preset",
    "get_available_presets",
    "get_best_vad_for_language",
]
