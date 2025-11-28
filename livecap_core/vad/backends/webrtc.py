"""WebRTC VAD バックエンド

WebRTC VAD を使用した音声活動検出。
VADBackend Protocol を実装。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class WebRTCVAD:
    """
    WebRTC VAD バックエンド

    VADBackend Protocol を実装。
    軽量で高速な C 拡張ベースの VAD。

    Args:
        mode: 積極性レベル (0-3)
            - 0: 最も寛容（誤検出少、見逃し多）
            - 1: やや厳格
            - 2: 厳格
            - 3: 最も厳格（誤検出多、見逃し少）
        frame_duration_ms: フレーム長（10, 20, 30ms のいずれか）

    Raises:
        ImportError: webrtcvad がインストールされていない場合
        ValueError: 無効な mode または frame_duration_ms

    Usage:
        vad = WebRTCVAD(mode=3)

        # 320 samples @ 16kHz のチャンクを処理
        probability = vad.process(audio_chunk)

        # 新しいストリーム開始時
        vad.reset()
    """

    SAMPLE_RATE = 16000
    VALID_FRAME_DURATIONS = (10, 20, 30)

    def __init__(self, mode: int = 3, frame_duration_ms: int = 20):
        if mode not in range(4):
            raise ValueError(f"mode must be 0-3, got {mode}")
        if frame_duration_ms not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f"frame_duration_ms must be one of {self.VALID_FRAME_DURATIONS}, "
                f"got {frame_duration_ms}"
            )

        self._mode = mode
        self._frame_duration_ms = frame_duration_ms
        self._frame_size = (self.SAMPLE_RATE * frame_duration_ms) // 1000
        self._vad = None

        self._initialize()

    def _initialize(self) -> None:
        """VAD を初期化"""
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(self._mode)
            logger.info(
                f"WebRTC VAD loaded (mode={self._mode}, "
                f"frame_duration={self._frame_duration_ms}ms)"
            )
        except ImportError as e:
            raise ImportError(
                "webrtcvad is required. Install with: pip install livecap-core[vad-webrtc]"
            ) from e

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（frame_size samples @ 16kHz）

        Returns:
            probability (0.0 or 1.0) - WebRTC は binary 判定
        """
        # float32 [-1, 1] → int16 bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        is_speech = self._vad.is_speech(audio_bytes, self.SAMPLE_RATE)
        return 1.0 if is_speech else 0.0

    def reset(self) -> None:
        """内部状態をリセット（WebRTC VAD は状態を持たないが、Protocol 準拠のため実装）"""
        pass

    @property
    def frame_size(self) -> int:
        """16kHz での推奨フレームサイズ（samples）"""
        return self._frame_size

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        return f"webrtc_mode{self._mode}"

    @property
    def config(self) -> dict:
        """レポート用の設定パラメータを返す"""
        return {
            "mode": self._mode,
            "frame_duration_ms": self._frame_duration_ms,
        }
