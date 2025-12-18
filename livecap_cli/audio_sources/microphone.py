"""マイク音声ソース

システムマイクからのリアルタイム音声キャプチャ。
sounddevice ライブラリを使用。
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from .base import AudioSource, DeviceInfo

logger = logging.getLogger(__name__)


class MicrophoneSource(AudioSource):
    """
    マイクからの音声ストリーム

    システムのデフォルトマイクまたは指定デバイスから
    音声をキャプチャする。

    Args:
        device: デバイスインデックスまたは名前。None でデフォルト。
        sample_rate: サンプリングレート (Hz)
        chunk_ms: チャンクサイズ（ミリ秒）

    Usage:
        # デフォルトマイクから
        with MicrophoneSource() as mic:
            for chunk in mic:
                process(chunk)

        # 特定デバイスから
        devices = MicrophoneSource.list_devices()
        with MicrophoneSource(device=devices[0].index) as mic:
            for chunk in mic:
                process(chunk)
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ):
        super().__init__(sample_rate=sample_rate, chunk_ms=chunk_ms)
        self.device = device
        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

    def start(self) -> None:
        """マイクキャプチャを開始"""
        if self._is_active:
            return

        def callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            """sounddevice コールバック"""
            if status:
                logger.warning(f"Audio callback status: {status}")
            # float32 モノラルに変換してキューに追加
            audio = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            self._queue.put(audio.flatten().astype(np.float32))

        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=callback,
            )
            self._stream.start()
            self._is_active = True

            device_name = self._get_device_name()
            logger.info(
                f"MicrophoneSource started: {device_name} "
                f"({self.sample_rate}Hz, {self.chunk_ms}ms chunks)"
            )
        except sd.PortAudioError as e:
            logger.error(f"Failed to start microphone: {e}")
            raise RuntimeError(f"Failed to start microphone: {e}") from e

    def stop(self) -> None:
        """マイクキャプチャを停止"""
        if not self._is_active:
            return

        self._is_active = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # キューをクリア
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.info("MicrophoneSource stopped")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        次のチャンクを読み取り

        Args:
            timeout: タイムアウト（秒）。None で無限待機。

        Returns:
            音声チャンク（float32）。タイムアウト時は None。
        """
        if not self._is_active:
            return None

        try:
            chunk = self._queue.get(timeout=timeout)
            return chunk
        except queue.Empty:
            return None

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]:
        """
        利用可能な入力デバイス一覧を取得

        Returns:
            入力チャンネルを持つデバイスのリスト
        """
        devices: list[DeviceInfo] = []
        try:
            default_device = sd.default.device[0]  # 入力デバイスのデフォルト
        except Exception:
            default_device = None

        try:
            for i, dev in enumerate(sd.query_devices()):
                # 入力チャンネルがあるデバイスのみ
                if dev["max_input_channels"] > 0:
                    devices.append(
                        DeviceInfo(
                            index=i,
                            name=dev["name"],
                            channels=dev["max_input_channels"],
                            sample_rate=int(dev["default_samplerate"]),
                            is_default=(i == default_device),
                        )
                    )
        except sd.PortAudioError as e:
            logger.warning(f"Failed to query devices: {e}")

        return devices

    def _get_device_name(self) -> str:
        """現在のデバイス名を取得"""
        try:
            if self.device is None:
                default_idx = sd.default.device[0]
                if default_idx is not None:
                    return sd.query_devices(default_idx)["name"]
                return "default"
            elif isinstance(self.device, int):
                return sd.query_devices(self.device)["name"]
            else:
                return str(self.device)
        except Exception:
            return "unknown"
