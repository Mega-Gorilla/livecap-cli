"""Silero VAD バックエンド

Silero VAD v5/v6 を使用した音声活動検出。
VADBackend Protocol を実装。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD v5/v6 バックエンド

    VADBackend Protocol を実装。
    512 samples (32ms @ 16kHz) のチャンクを処理して確率を返す。

    Args:
        threshold: 音声判定閾値（VADProcessor で使用、ここでは参考値）
        onnx: ONNX ランタイムを使用するか（推奨: True）

    Raises:
        ImportError: silero-vad または torch がインストールされていない場合

    Usage:
        vad = SileroVAD(onnx=True)

        # 512 samples @ 16kHz のチャンクを処理
        probability = vad.process(audio_chunk)

        # 新しいストリーム開始時
        vad.reset()
    """

    def __init__(self, threshold: float = 0.5, onnx: bool = True):
        self._threshold = threshold
        self._onnx = onnx
        self._model: Any = None
        self._torch: Any = None

        self._initialize()

    def _initialize(self) -> None:
        """モデルを初期化"""
        try:
            import torch

            self._torch = torch
        except ImportError as e:
            raise ImportError(
                "torch is required for Silero VAD. "
                "Install with: pip install livecap-core[vad]"
            ) from e

        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=self._onnx)
            logger.info(f"Silero VAD loaded (onnx={self._onnx})")
        except ImportError as e:
            raise ImportError(
                "silero-vad is required. Install with: pip install livecap-core[vad]"
            ) from e

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（512 samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        # numpy → torch tensor
        if not isinstance(audio, self._torch.Tensor):
            audio_tensor = self._torch.from_numpy(audio.astype(np.float32))
        else:
            audio_tensor = audio

        # Silero VAD は (512,) の 1D tensor を期待
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        return self._model(audio_tensor, 16000).item()

    def reset(self) -> None:
        """内部状態をリセット（新しい音声ストリーム開始時に呼ぶ）"""
        if self._model is not None:
            self._model.reset_states()
