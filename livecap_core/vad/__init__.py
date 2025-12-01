"""VAD (Voice Activity Detection) module for livecap_core.

音声活動検出のためのモジュール。
複数のバックエンド（Silero, TenVAD, WebRTC）をサポート。

Usage:
    from livecap_core.vad import VADProcessor, VADConfig

    # 言語に最適化された VAD を使用（推奨）
    # 日本語 → TenVAD, 英語 → WebRTC
    processor = VADProcessor.from_language("ja")
    for chunk in audio_source:
        segments = processor.process_chunk(chunk, sample_rate=16000)
        for segment in segments:
            if segment.is_final:
                transcribe(segment.audio)

    # デフォルト設定（Silero VAD）
    processor = VADProcessor()

    # カスタム設定
    config = VADConfig(
        threshold=0.6,
        min_speech_ms=300,
    )
    processor = VADProcessor(config=config)

    # 別のバックエンドを使用
    from livecap_core.vad.backends import WebRTCVAD
    processor = VADProcessor(backend=WebRTCVAD(mode=3))

Supported languages for from_language():
    - "ja" (Japanese): TenVAD - 7.2% CER
    - "en" (English): WebRTC - 3.3% WER
"""

from .config import VADConfig
from .presets import (
    VAD_OPTIMIZED_PRESETS,
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
    "VAD_OPTIMIZED_PRESETS",
    "get_optimized_preset",
    "get_available_presets",
    "get_best_vad_for_language",
]
