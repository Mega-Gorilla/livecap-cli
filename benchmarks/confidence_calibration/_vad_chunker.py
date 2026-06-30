"""Silero VAD probability stream → speech segment list (Issue #338 PR-β)。

`SileroVAD.process()` (`livecap_cli/vad/backends/silero.py`) は 512 samples
@ 16 kHz ごとに probability (0.0-1.0) を返す API なので、speech segment
切り出しは caller 側で boundary detection を実装する必要がある。本 module
は build_corpus.py が呼ぶ最小限の logic:

1. ``compute_vad_probabilities()`` — SileroVAD で audio 全体を 512-sample
   window で sliding、probability stream を生成
2. ``detect_speech_segments()`` — probability stream + parameters から
   speech segment list (frame index) を pure logic で生成 (test しやすい)
3. ``chunk_audio_by_vad()`` — 上記 2 つを合成、(start_sec, end_sec) 秒の
   segment list を返す

設計判断 (Plan D1):
- threshold = 0.5 (Silero default、speech/non-speech boundary 判定)
- min_speech_sec = 0.5 (calibration sample に有意な長さ)
- max_segment_sec = 3.0 (engine.transcribe() で 1-3 秒 chunk が扱いやすい)
- min_silence_sec = 0.3 (segment 境界を確定する silence の最小長)
- hysteresis: probability < threshold が ``min_silence_sec`` 連続で初めて
  segment 終了 (短い瞬間 dip で segment を切らない)
- max_segment_sec を超える場合は均等 split (1 segment 内で複数 chunk 化)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD frame size (samples @ 16 kHz)。`SileroVAD.frame_size` の値と一致。
FRAME_SIZE = 512
SAMPLE_RATE = 16000
FRAME_SEC = FRAME_SIZE / SAMPLE_RATE  # = 0.032 秒


def compute_vad_probabilities(
    audio: np.ndarray,
    *,
    vad: Optional[object] = None,
) -> list[float]:
    """16 kHz mono float32 audio から SileroVAD probability stream を生成。

    Args:
        audio: 16 kHz mono float32 numpy array。
        vad: 既存 SileroVAD instance (test 用、None なら新規生成)。

    Returns:
        各 frame (32 ms) の probability list (長さ = ``len(audio) // 512``)。
        末尾の半端 sample は drop (Silero VAD は 512 samples 厳密要求)。
    """
    if vad is None:
        from livecap_cli.vad.backends.silero import SileroVAD

        vad = SileroVAD(onnx=True)

    # 新規 stream のため reset
    if hasattr(vad, "reset"):
        vad.reset()

    n_frames = len(audio) // FRAME_SIZE
    probabilities: list[float] = []
    for i in range(n_frames):
        chunk = audio[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        prob = vad.process(chunk)
        probabilities.append(float(prob))
    return probabilities


def detect_speech_segments(
    probabilities: list[float],
    *,
    threshold: float = 0.5,
    min_speech_sec: float = 0.5,
    max_segment_sec: float = 3.0,
    min_silence_sec: float = 0.3,
    frame_sec: float = FRAME_SEC,
) -> list[tuple[int, int]]:
    """Probability stream から speech segment の frame index list を生成。

    Pure logic (実 VAD instance 不要)、test しやすい。

    Args:
        probabilities: 各 frame の VAD probability (0.0-1.0)。
        threshold: probability がこれ以上の frame を speech 候補。
        min_speech_sec: speech segment の最小長 (短すぎる segment は drop)。
        max_segment_sec: speech segment の最大長 (超える場合は均等 split)。
        min_silence_sec: segment 終了判定の連続 silence 最小長 (hysteresis)。
        frame_sec: 1 frame の長さ秒 (default 0.032 = 512/16000)。

    Returns:
        ``(start_frame, end_frame)`` tuple list、end は exclusive。
    """
    if not probabilities:
        return []
    if min_speech_sec <= 0 or max_segment_sec <= 0 or frame_sec <= 0:
        raise ValueError(
            f"durations must be positive: min_speech={min_speech_sec}, "
            f"max_segment={max_segment_sec}, frame={frame_sec}"
        )

    min_speech_frames = max(1, int(round(min_speech_sec / frame_sec)))
    max_segment_frames = max(1, int(round(max_segment_sec / frame_sec)))
    min_silence_frames = max(1, int(round(min_silence_sec / frame_sec)))

    raw_segments: list[tuple[int, int]] = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i, prob in enumerate(probabilities):
        if prob >= threshold:
            if not in_speech:
                in_speech = True
                speech_start = i
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                # 連続 silence が min_silence_sec を超えたら segment 終了
                if silence_count >= min_silence_frames:
                    speech_end = i - silence_count + 1
                    if speech_end - speech_start >= min_speech_frames:
                        raw_segments.append((speech_start, speech_end))
                    in_speech = False
                    silence_count = 0

    # Tail (in_speech のまま終了した場合)
    if in_speech:
        speech_end = len(probabilities) - silence_count
        if speech_end - speech_start >= min_speech_frames:
            raw_segments.append((speech_start, speech_end))

    # max_segment_frames を超える segment を均等 split
    result: list[tuple[int, int]] = []
    for start, end in raw_segments:
        duration_frames = end - start
        if duration_frames <= max_segment_frames:
            result.append((start, end))
        else:
            n_chunks = int(np.ceil(duration_frames / max_segment_frames))
            chunk_size = duration_frames // n_chunks
            for k in range(n_chunks):
                chunk_start = start + k * chunk_size
                chunk_end = start + (k + 1) * chunk_size if k < n_chunks - 1 else end
                if chunk_end - chunk_start >= min_speech_frames:
                    result.append((chunk_start, chunk_end))

    return result


def chunk_audio_by_vad(
    audio: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech_sec: float = 0.5,
    max_segment_sec: float = 3.0,
    min_silence_sec: float = 0.3,
    vad: Optional[object] = None,
) -> list[tuple[float, float]]:
    """16 kHz mono audio を speech segment に切り出し (sec 単位、秒の tuple list)。

    Args:
        audio: 16 kHz mono float32 numpy array。
        threshold/min_speech_sec/max_segment_sec/min_silence_sec: VAD parameter
            (default は Plan D1 の値)。
        vad: 既存 SileroVAD instance (test 用、None なら新規生成)。

    Returns:
        ``(start_sec, end_sec)`` tuple list、各 segment の audio 範囲。
        end は exclusive (= ``audio[int(start*16000):int(end*16000)]`` で切出)。
    """
    probabilities = compute_vad_probabilities(audio, vad=vad)
    segments = detect_speech_segments(
        probabilities,
        threshold=threshold,
        min_speech_sec=min_speech_sec,
        max_segment_sec=max_segment_sec,
        min_silence_sec=min_silence_sec,
        frame_sec=FRAME_SEC,
    )
    return [(start * FRAME_SEC, end * FRAME_SEC) for start, end in segments]
