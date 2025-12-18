"""音声ソース基底クラス

AudioSource ABC と DeviceInfo を定義。
同期/非同期両方のイテレータインターフェースを提供。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """
    オーディオデバイス情報

    Attributes:
        index: デバイスインデックス
        name: デバイス名
        channels: 入力チャンネル数
        sample_rate: デフォルトサンプリングレート
        is_default: デフォルトデバイスかどうか
    """

    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


class AudioSource(ABC):
    """
    音声ソースの抽象基底クラス

    同期/非同期両方のイテレータインターフェースを提供。
    コンテキストマネージャとしても使用可能。

    Usage:
        # コンテキストマネージャとして
        with MicrophoneSource() as mic:
            for chunk in mic:
                process(chunk)

        # 非同期イテレータとして
        async with MicrophoneSource() as mic:
            async for chunk in mic:
                await process(chunk)
    """

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 100):
        """
        Args:
            sample_rate: サンプリングレート (Hz)
            chunk_ms: チャンクサイズ (ミリ秒)
        """
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """キャプチャがアクティブかどうか"""
        return self._is_active

    @abstractmethod
    def start(self) -> None:
        """キャプチャを開始"""

    @abstractmethod
    def stop(self) -> None:
        """キャプチャを停止"""

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        音声チャンクを読み取り（ブロッキング）

        Args:
            timeout: タイムアウト（秒）。None の場合は無限に待機。

        Returns:
            音声データ（float32, mono）。データがない場合は None。
        """

    def __iter__(self) -> Iterator[np.ndarray]:
        """同期イテレータ"""
        if not self._is_active:
            self.start()
        try:
            while self._is_active:
                chunk = self.read(timeout=1.0)
                if chunk is not None:
                    yield chunk
        finally:
            self.stop()

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """非同期イテレータ"""
        import asyncio

        if not self._is_active:
            self.start()
        try:
            loop = asyncio.get_event_loop()
            while self._is_active:
                # ブロッキング読み取りを別スレッドで実行
                chunk = await loop.run_in_executor(
                    None, lambda: self.read(timeout=0.1)
                )
                if chunk is not None:
                    yield chunk
                else:
                    await asyncio.sleep(0.01)
        finally:
            self.stop()

    @classmethod
    @abstractmethod
    def list_devices(cls) -> list[DeviceInfo]:
        """
        利用可能なデバイス一覧を取得

        Returns:
            DeviceInfo のリスト
        """

    def __enter__(self) -> "AudioSource":
        """コンテキストマネージャ: 開始"""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """コンテキストマネージャ: 終了"""
        self.stop()

    async def __aenter__(self) -> "AudioSource":
        """非同期コンテキストマネージャ: 開始"""
        self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """非同期コンテキストマネージャ: 終了"""
        self.stop()
