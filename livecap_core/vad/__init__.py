"""VAD (Voice Activity Detection) module for livecap_core.

音声活動検出のためのモジュール。
Silero VAD v5/v6 をデフォルトバックエンドとして使用。

Usage:
    from livecap_core.vad import VADProcessor, VADConfig

    # デフォルト設定で処理
    processor = VADProcessor()
    for chunk in audio_source:
        segments = processor.process_chunk(chunk, sample_rate=16000)
        for segment in segments:
            if segment.is_final:
                transcribe(segment.audio)

    # カスタム設定
    config = VADConfig(
        threshold=0.6,
        min_speech_ms=300,
    )
    processor = VADProcessor(config=config)
"""

from .config import VADConfig
from .processor import VADProcessor
from .state_machine import VADSegment, VADState, VADStateMachine

__all__ = [
    "VADConfig",
    "VADProcessor",
    "VADSegment",
    "VADState",
    "VADStateMachine",
]
