"""ファイル音声ソース

テスト・デバッグ用のファイルベース音声ソース。
リアルタイム処理のシミュレーションに使用。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import AudioSource, DeviceInfo

logger = logging.getLogger(__name__)


class FileSource(AudioSource):
    """
    ファイルからの音声ストリーム

    テスト・デバッグ用。音声ファイルを読み込み、
    チャンク単位でストリーミングする。

    Args:
        file_path: 音声ファイルパス
        sample_rate: 出力サンプリングレート（リサンプリングされる）
        chunk_ms: チャンクサイズ（ミリ秒）
        realtime: True の場合、リアルタイムシミュレーション（チャンク間で待機）

    Usage:
        # 高速テスト（待機なし）
        with FileSource("test.wav", realtime=False) as src:
            for chunk in src:
                process(chunk)

        # リアルタイムシミュレーション
        with FileSource("test.wav", realtime=True) as src:
            for chunk in src:
                print(f"Received {len(chunk)} samples")
    """

    def __init__(
        self,
        file_path: Union[Path, str],
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        realtime: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, chunk_ms=chunk_ms)
        self.file_path = Path(file_path)
        self.realtime = realtime

        self._audio: Optional[np.ndarray] = None
        self._position: int = 0
        self._file_sample_rate: int = 0

    def start(self) -> None:
        """ファイルを読み込んでストリーミングを開始"""
        if self._is_active:
            return

        import soundfile as sf

        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        # ファイル読み込み
        audio, self._file_sample_rate = sf.read(self.file_path, dtype="float32")

        # モノラル化（ステレオの場合は平均）
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)

        # リサンプリング（必要な場合）
        if self._file_sample_rate != self.sample_rate:
            audio = self._resample(audio, self._file_sample_rate, self.sample_rate)

        self._audio = audio
        self._position = 0
        self._is_active = True

        duration = len(self._audio) / self.sample_rate
        logger.info(
            f"FileSource started: {self.file_path.name} "
            f"({duration:.2f}s, {self.sample_rate}Hz)"
        )

    def stop(self) -> None:
        """ストリーミングを停止"""
        if not self._is_active:
            return

        self._is_active = False
        self._audio = None
        self._position = 0

        logger.info("FileSource stopped")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        次のチャンクを読み取り

        Args:
            timeout: 未使用（ファイルソースでは即座に返す）

        Returns:
            音声チャンク（float32）。ファイル終端の場合は None。
        """
        if self._audio is None or self._position >= len(self._audio):
            self._is_active = False
            return None

        # チャンク取得
        end = min(self._position + self.chunk_size, len(self._audio))
        chunk = self._audio[self._position:end].copy()
        self._position = end

        # リアルタイムシミュレーション
        if self.realtime and len(chunk) == self.chunk_size:
            time.sleep(self.chunk_ms / 1000)

        # 最終チャンクが短い場合はパディング
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode="constant")

        return chunk

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]:
        """ファイルソースにはデバイスがないため空リストを返す"""
        return []

    @property
    def duration(self) -> float:
        """ファイルの総再生時間（秒）"""
        if self._audio is None:
            return 0.0
        return len(self._audio) / self.sample_rate

    @property
    def position_seconds(self) -> float:
        """現在の再生位置（秒）"""
        return self._position / self.sample_rate

    @property
    def remaining_seconds(self) -> float:
        """残り再生時間（秒）"""
        return self.duration - self.position_seconds

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        リサンプリング

        効率的な整数比リサンプリングを使用。
        """
        from math import gcd

        from scipy import signal

        # 最大公約数で整数比を計算
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        # ポリフェーズリサンプリング
        resampled = signal.resample_poly(audio, up, down)

        return resampled.astype(np.float32)

    def reset(self) -> None:
        """再生位置を先頭に戻す"""
        self._position = 0
        if self._audio is not None:
            self._is_active = True
