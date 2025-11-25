"""VAD バックエンド

プラグイン可能なVADバックエンドを提供。
VADBackend Protocol を実装することで独自のVADを追加可能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    pass


class VADBackend(Protocol):
    """
    VADバックエンドのプロトコル

    このプロトコルを実装することで、任意のVADバックエンドを使用可能。

    Usage:
        class MyVAD:
            def process(self, audio: np.ndarray) -> float:
                # VAD処理
                return probability

            def reset(self) -> None:
                # 状態リセット
                pass

        # VADProcessor に渡す
        processor = VADProcessor(backend=MyVAD())
    """

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（512 samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        ...

    def reset(self) -> None:
        """内部状態をリセット（新しい音声ストリーム開始時に呼ぶ）"""
        ...


# SileroVAD は遅延インポート（torch 依存を避けるため）
def __getattr__(name: str):
    """遅延インポート for SileroVAD."""
    if name == "SileroVAD":
        from .silero import SileroVAD

        return SileroVAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VADBackend", "SileroVAD"]
