"""TenVAD バックエンド

TEN Framework の VAD を使用した音声活動検出。
VADBackend Protocol を実装。

⚠️ TenVAD のライセンスは限定的です。使用前にライセンスをご確認ください。
https://github.com/TEN-framework/ten-vad
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class TenVAD:
    """
    TenVAD バックエンド

    VADBackend Protocol を実装。
    軽量で高速な VAD。16kHz 専用。

    ⚠️ TenVAD のライセンスは限定的です。使用前にライセンスをご確認ください。

    Args:
        hop_size: フレームサイズ（160 または 256 samples）
            - 160: 10ms
            - 256: 16ms（デフォルト）
        threshold: 音声判定閾値（0.0-1.0）

    Raises:
        ImportError: ten-vad がインストールされていない場合
        ValueError: 無効な hop_size

    Usage:
        vad = TenVAD(hop_size=256)

        # 256 samples @ 16kHz のチャンクを処理
        probability = vad.process(audio_chunk)

        # 新しいストリーム開始時
        vad.reset()
    """

    SAMPLE_RATE = 16000
    VALID_HOP_SIZES = (160, 256)

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        if hop_size not in self.VALID_HOP_SIZES:
            raise ValueError(
                f"hop_size must be one of {self.VALID_HOP_SIZES}, got {hop_size}"
            )

        self._hop_size = hop_size
        self._threshold = threshold
        self._vad = None

        # ライセンス警告
        warnings.warn(
            "TenVAD has limited license terms. "
            "Please review the license before commercial use: "
            "https://github.com/TEN-framework/ten-vad",
            UserWarning,
            stacklevel=2,
        )

        self._initialize()

    def _initialize(self) -> None:
        """VAD を初期化"""
        try:
            from ten_vad import TenVad

            self._vad = TenVad(
                hop_size=self._hop_size,
                threshold=self._threshold,
            )
            logger.info(
                f"TenVAD loaded (hop_size={self._hop_size}, threshold={self._threshold})"
            )
        except ImportError as e:
            raise ImportError(
                "ten-vad is required. Install with: pip install livecap-core[vad-tenvad]"
            ) from e

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（hop_size samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        # float32 [-1, 1] → int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # TenVAD の process は (probability, is_speech) を返す
        prob, _is_speech = self._vad.process(audio_int16)
        return float(prob)

    def reset(self) -> None:
        """内部状態をリセット"""
        if self._vad is not None:
            self._vad.reset()

    @property
    def frame_size(self) -> int:
        """16kHz での推奨フレームサイズ（samples）"""
        return self._hop_size

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        return "tenvad"

    @property
    def config(self) -> dict:
        """レポート用の設定パラメータを返す"""
        return {
            "hop_size": self._hop_size,
            "threshold": self._threshold,
        }
