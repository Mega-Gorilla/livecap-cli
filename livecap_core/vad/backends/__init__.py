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

            @property
            def frame_size(self) -> int:
                return 512  # samples @ 16kHz

            @property
            def name(self) -> str:
                return "my_vad"

        # VADProcessor に渡す
        processor = VADProcessor(backend=MyVAD())
    """

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（frame_size samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        ...

    def reset(self) -> None:
        """内部状態をリセット（新しい音声ストリーム開始時に呼ぶ）"""
        ...

    @property
    def frame_size(self) -> int:
        """
        16kHz での推奨フレームサイズ（samples）

        バックエンドごとに異なる:
        - Silero: 512 samples (32ms)
        - WebRTC: 320 samples (20ms)
        - TenVAD: 256 samples (16ms)
        """
        ...

    @property
    def name(self) -> str:
        """バックエンド識別子（例: "silero", "webrtc", "tenvad"）"""
        ...


# VAD バックエンドは遅延インポート（依存関係を避けるため）
def __getattr__(name: str):
    """遅延インポート for VAD backends."""
    if name == "SileroVAD":
        from .silero import SileroVAD

        return SileroVAD
    if name == "WebRTCVAD":
        from .webrtc import WebRTCVAD

        return WebRTCVAD
    if name == "TenVAD":
        from .tenvad import TenVAD

        return TenVAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VADBackend", "SileroVAD", "WebRTCVAD", "TenVAD"]
