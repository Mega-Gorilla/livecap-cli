"""VADステートマシン

4状態の遷移を管理し、音声セグメントを検出する。
livecap-guiの実装を簡素化。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from .config import VADConfig


class VADState(Enum):
    """VAD状態"""

    SILENCE = auto()  # 無音
    POTENTIAL_SPEECH = auto()  # 音声の可能性（検証中）
    SPEECH = auto()  # 確定した音声
    ENDING = auto()  # 音声終了処理中


@dataclass(slots=True)
class VADSegment:
    """検出された音声セグメント"""

    audio: np.ndarray
    start_time: float
    end_time: float
    is_final: bool


class VADStateMachine:
    """
    VADステートマシン

    4状態の遷移を管理し、音声セグメントを検出する。

    状態遷移:
        SILENCE → POTENTIAL_SPEECH → SPEECH → ENDING → SILENCE

    Args:
        config: VAD設定

    Usage:
        sm = VADStateMachine(VADConfig())

        for frame, probability, timestamp in frames:
            segment = sm.process_frame(frame, probability, timestamp)
            if segment is not None:
                process_segment(segment)

        # 処理終了時
        final_segment = sm.finalize(current_timestamp)
    """

    # Silero VAD v5+ のフレーム設定
    # 512 samples (32ms @ 16kHz) 固定
    FRAME_MS: int = 32
    FRAME_SAMPLES: int = 512  # 32ms @ 16kHz

    def __init__(self, config: VADConfig):
        self.config = config
        self._state = VADState.SILENCE
        self._state_entry_time = time.time()

        # フレームカウンタ（ミリ秒から変換）
        self._speech_frames = 0
        self._silence_frames = 0

        # しきい値（フレーム数に変換）
        self._min_speech_frames = max(1, config.min_speech_ms // self.FRAME_MS)
        self._min_silence_frames = max(1, config.min_silence_ms // self.FRAME_MS)
        # Silero VAD は speech_pad_ms を前後両方のパディングに使用
        # 最低1フレームを保証（100ms // 32ms = 3フレーム）
        self._padding_frames = max(1, config.speech_pad_ms // self.FRAME_MS)

        # バッファ
        self._pre_buffer: list[np.ndarray] = []
        self._speech_buffer: list[np.ndarray] = []
        self._segment_start_time: float = 0.0

        # 中間結果管理
        self._last_interim_time = 0.0
        self._interim_frame_count = 0

    @property
    def state(self) -> VADState:
        """現在の状態"""
        return self._state

    def process_frame(
        self,
        audio_frame: np.ndarray,
        probability: float,
        timestamp: float,
    ) -> Optional[VADSegment]:
        """
        1フレーム（32ms）を処理

        Args:
            audio_frame: 音声データ（float32, 512 samples @ 16kHz）
            probability: VAD確率（0.0-1.0）
            timestamp: 現在のタイムスタンプ

        Returns:
            完成したセグメント（なければNone）
        """
        is_speech = probability >= self.config.threshold

        # 状態別処理
        if self._state == VADState.SILENCE:
            return self._handle_silence(audio_frame, is_speech, timestamp)
        elif self._state == VADState.POTENTIAL_SPEECH:
            return self._handle_potential_speech(audio_frame, is_speech, timestamp)
        elif self._state == VADState.SPEECH:
            return self._handle_speech(audio_frame, is_speech, timestamp)
        elif self._state == VADState.ENDING:
            return self._handle_ending(audio_frame, is_speech, timestamp)

        return None

    def _handle_silence(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """SILENCE状態の処理"""
        # プリバッファに追加（パディング用）
        self._pre_buffer.append(frame)
        if len(self._pre_buffer) > self._padding_frames:
            self._pre_buffer.pop(0)

        if is_speech:
            self._transition_to(VADState.POTENTIAL_SPEECH)
            self._speech_frames = 1
            self._segment_start_time = timestamp - (
                len(self._pre_buffer) * self.FRAME_MS / 1000
            )
            # プリバッファをスピーチバッファにコピー
            self._speech_buffer = list(self._pre_buffer)
            self._speech_buffer.append(frame)

        return None

    def _handle_potential_speech(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """POTENTIAL_SPEECH状態の処理"""
        self._speech_buffer.append(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0

            if self._speech_frames >= self._min_speech_frames:
                self._transition_to(VADState.SPEECH)
        else:
            self._silence_frames += 1

            # タイムアウト: 音声として確定せず破棄
            if self._silence_frames >= self._min_silence_frames:
                self._transition_to(VADState.SILENCE)
                self._reset_buffers()

        return None

    def _handle_speech(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """SPEECH状態の処理"""
        self._speech_buffer.append(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

            if self._silence_frames >= self._min_silence_frames:
                self._transition_to(VADState.ENDING)

        # 最大発話時間チェック
        if self.config.max_speech_ms > 0:
            total_ms = len(self._speech_buffer) * self.FRAME_MS
            if total_ms >= self.config.max_speech_ms:
                return self._finalize_segment(timestamp, is_final=True)

        # 中間結果チェック
        return self._check_interim(timestamp)

    def _handle_ending(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """ENDING状態の処理（ポストパディング）"""
        self._speech_buffer.append(frame)

        if is_speech:
            # 音声再開 → SPEECH に戻る
            self._transition_to(VADState.SPEECH)
            self._silence_frames = 0
            return None

        self._silence_frames += 1

        if self._silence_frames >= self._padding_frames:
            return self._finalize_segment(timestamp, is_final=True)

        return None

    def _transition_to(self, new_state: VADState) -> None:
        """状態遷移"""
        self._state = new_state
        self._state_entry_time = time.time()
        if new_state == VADState.SILENCE:
            self._speech_frames = 0
            self._silence_frames = 0

    def _reset_buffers(self) -> None:
        """バッファリセット"""
        self._speech_buffer = []
        self._pre_buffer = []
        self._last_interim_time = 0.0
        self._interim_frame_count = 0

    def _check_interim(self, timestamp: float) -> Optional[VADSegment]:
        """中間結果の送信チェック"""
        current_time = time.time()
        frames_since_last = len(self._speech_buffer) - self._interim_frame_count
        duration_ms = frames_since_last * self.FRAME_MS

        if duration_ms >= self.config.interim_min_duration_ms and (
            current_time - self._last_interim_time
        ) * 1000 >= self.config.interim_interval_ms:
            self._last_interim_time = current_time
            self._interim_frame_count = len(self._speech_buffer)

            return VADSegment(
                audio=np.concatenate(self._speech_buffer),
                start_time=self._segment_start_time,
                end_time=timestamp,
                is_final=False,
            )

        return None

    def _finalize_segment(self, timestamp: float, is_final: bool) -> VADSegment:
        """セグメントを確定"""
        segment = VADSegment(
            audio=(
                np.concatenate(self._speech_buffer)
                if self._speech_buffer
                else np.array([], dtype=np.float32)
            ),
            start_time=self._segment_start_time,
            end_time=timestamp,
            is_final=is_final,
        )

        self._transition_to(VADState.SILENCE)
        self._reset_buffers()

        return segment

    def finalize(self, timestamp: float) -> Optional[VADSegment]:
        """処理を終了し、残っているセグメントを返す"""
        if self._state in (
            VADState.SPEECH,
            VADState.ENDING,
            VADState.POTENTIAL_SPEECH,
        ):
            if self._speech_buffer:
                return self._finalize_segment(timestamp, is_final=True)
        return None

    def reset(self) -> None:
        """状態を完全にリセット"""
        self._state = VADState.SILENCE
        self._state_entry_time = time.time()
        self._speech_frames = 0
        self._silence_frames = 0
        self._reset_buffers()
