"""JaVAD バックエンド（ベンチマーク専用）

JaVAD (Japanese Voice Activity Detection) を使用した音声活動検出。
バッチ処理専用。ストリーミング処理には使用できません。

⚠️ このバックエンドは VADBackend Protocol に準拠していません。
   ベンチマーク専用のバッチ処理インターフェースを提供します。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class JaVADPipeline:
    """
    JaVAD バッチ処理（ベンチマーク専用）

    ⚠️ ストリーミング非対応
    リアルタイム用途には silero, webrtc, tenvad を使用してください。

    JaVAD は per-frame probability ではなく segments を返すため、
    VADBackend Protocol に準拠していません。

    Args:
        model: モデル種別
            - "tiny": 640ms window、即時検出向け
            - "balanced": 1920ms window、バランス型（デフォルト）
            - "precise": 3840ms window、最高精度

    Raises:
        ImportError: javad がインストールされていない場合
        ValueError: 無効な model

    Usage:
        pipeline = JaVADPipeline(model="balanced")

        # 音声全体を処理してセグメントを取得
        segments = pipeline.process_audio(audio, sample_rate=16000)
        # segments: [(start_time, end_time), ...]

        # ベンチマーク用: セグメントから音声を抽出
        for start, end in segments:
            segment_audio = audio[int(start * sr):int(end * sr)]
    """

    VALID_MODELS = ("tiny", "balanced", "precise")
    WINDOW_SIZES = {
        "tiny": 640,      # ms
        "balanced": 1920,  # ms
        "precise": 3840,   # ms
    }

    def __init__(self, model: str = "balanced"):
        if model not in self.VALID_MODELS:
            raise ValueError(
                f"model must be one of {self.VALID_MODELS}, got {model}"
            )

        self._model_name = model
        self._processor: Any = None

        self._initialize()

    def _initialize(self) -> None:
        """プロセッサを初期化"""
        try:
            from javad import Processor

            self._processor = Processor(model_name=self._model_name)
            logger.info(
                f"JaVAD loaded (model={self._model_name}, "
                f"window={self.WINDOW_SIZES[self._model_name]}ms)"
            )
        except ImportError as e:
            raise ImportError(
                "javad is required. Install with: pip install livecap-core[vad-javad]"
            ) from e

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[tuple[float, float]]:
        """
        音声全体を処理してセグメントを返す

        Args:
            audio: float32形式の音声データ
            sample_rate: サンプルレート（16000 推奨）

        Returns:
            セグメントのリスト [(start_time, end_time), ...]
            時間は秒単位
        """
        # JaVAD の intervals() は [(start, end), ...] を返す
        segments = self._processor.intervals(audio)
        return [(float(s), float(e)) for s, e in segments]

    def reset(self) -> None:
        """内部状態をリセット"""
        # JaVAD Processor はステートレスなので不要だが、互換性のため実装
        pass

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        return f"javad_{self._model_name}"

    @property
    def window_size_ms(self) -> int:
        """ウィンドウサイズ（ミリ秒）"""
        return self.WINDOW_SIZES[self._model_name]

    @property
    def config(self) -> dict:
        """レポート用の設定パラメータを返す"""
        return {
            "model": self._model_name,
            "window_ms": self.WINDOW_SIZES[self._model_name],
        }
