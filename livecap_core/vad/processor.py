"""VADプロセッサ

VADバックエンドとステートマシンを組み合わせて
音声セグメントを検出する。
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Any, Optional

import numpy as np

from .backends import VADBackend
from .config import VADConfig
from .state_machine import VADSegment, VADState, VADStateMachine

logger = logging.getLogger(__name__)


class VADProcessor:
    """
    VADプロセッサ

    VADバックエンドとステートマシンを組み合わせて
    音声セグメントを検出する。

    Args:
        config: VAD設定（None でデフォルト）
        backend: VADバックエンド（None で Silero VAD）

    Usage:
        # デフォルト設定
        processor = VADProcessor()

        # カスタム設定
        processor = VADProcessor(
            config=VADConfig(threshold=0.6),
        )

        # 別のバックエンドを使用
        from livecap_core.vad.backends import WebRTCVAD
        processor = VADProcessor(backend=WebRTCVAD(mode=3))

        # 音声チャンクを処理
        for chunk in audio_source:
            segments = processor.process_chunk(chunk, sample_rate=16000)
            for segment in segments:
                if segment.is_final:
                    transcribe(segment.audio)

        # 処理終了時
        final_segment = processor.finalize()
    """

    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        backend: Optional[VADBackend] = None,
    ):
        self.config = config or VADConfig()
        self._backend = backend
        self._state_machine = VADStateMachine(self.config)
        self._current_time = 0.0

        # 残余バッファ（フレームサイズに満たない音声を保持）
        self._residual: Optional[np.ndarray] = None

        # Silero VAD 初期化（バックエンドが指定されていない場合）
        if self._backend is None:
            self._backend = self._create_default_backend()

        # フレームサイズをバックエンドから取得
        self._frame_size = self._backend.frame_size
        logger.debug(
            f"VADProcessor initialized with {self._backend.name} "
            f"(frame_size={self._frame_size})"
        )

    @classmethod
    def from_language(cls, language: str) -> "VADProcessor":
        """言語に最適なVADを使用してVADProcessorを作成

        ベンチマーク結果に基づき、言語ごとに最適なVADバックエンドと
        パラメータを自動選択する。

        Args:
            language: 言語コード ("ja", "en")

        Returns:
            最適化されたVADProcessor

        Raises:
            ValueError: 未サポート言語の場合
            ImportError: 必要なVADバックエンドがインストールされていない場合

        Example:
            # 日本語に最適化（TenVAD使用）
            processor = VADProcessor.from_language("ja")

            # 英語に最適化（WebRTC使用）
            processor = VADProcessor.from_language("en")
        """
        from .presets import get_available_presets, get_best_vad_for_language

        # 最適なVADとプリセットを取得
        result = get_best_vad_for_language(language)

        if result is None:
            # プリセットがない言語 → エラー（サポート言語を動的に取得）
            available = get_available_presets()
            supported = sorted(set(lang for _, lang in available))
            raise ValueError(
                f"No optimized preset for language '{language}'. "
                f"Supported languages: {', '.join(supported)}. "
                f"Use VADProcessor() for default Silero VAD."
            )

        vad_type, preset = result
        vad_config = VADConfig.from_dict(preset["vad_config"])
        backend_params = preset.get("backend", {})

        # バックエンドを作成
        backend = cls._create_backend(vad_type, backend_params)

        logger.info(f"Created VADProcessor with {vad_type} optimized for '{language}'")
        return cls(config=vad_config, backend=backend)

    @classmethod
    def _create_backend(
        cls, vad_type: str, backend_params: dict[str, Any]
    ) -> VADBackend:
        """バックエンドを作成

        Args:
            vad_type: VADバックエンドの種類 ("silero", "tenvad", "webrtc")
            backend_params: プリセットから取得したバックエンド固有パラメータ

        Returns:
            VADBackend インスタンス

        Raises:
            ImportError: VADバックエンドがインストールされていない場合
            ValueError: 未知のVADタイプの場合

        Note:
            SileroVADのthresholdはVADConfigで管理されるため、
            バックエンドには渡さない（onnx=Trueのみ指定）
        """
        if vad_type == "silero":
            from .backends.silero import SileroVAD

            return SileroVAD(onnx=True, **backend_params)
        elif vad_type == "tenvad":
            from .backends.tenvad import TenVAD

            return TenVAD(**backend_params)
        elif vad_type == "webrtc":
            from .backends.webrtc import WebRTCVAD

            return WebRTCVAD(**backend_params)
        else:
            raise ValueError(f"Unknown VAD type: {vad_type}")

    def _create_default_backend(self) -> VADBackend:
        """デフォルトの Silero VAD バックエンドを作成"""
        try:
            from .backends.silero import SileroVAD

            return SileroVAD(threshold=self.config.threshold, onnx=True)
        except ImportError as e:
            raise ImportError(
                "Silero VAD is required for default VAD backend. "
                "Install with: pip install livecap-core[vad]"
            ) from e

    def process_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[VADSegment]:
        """
        音声チャンクを処理

        Args:
            audio: 音声データ（float32）
            sample_rate: サンプリングレート

        Returns:
            検出されたセグメントのリスト
        """
        # リサンプリング（必要な場合）
        if sample_rate != self.SAMPLE_RATE:
            audio = self._resample(audio, sample_rate)

        # 残余バッファと結合
        if self._residual is not None:
            audio = np.concatenate([self._residual, audio])
            self._residual = None

        segments: list[VADSegment] = []

        # フレーム単位で処理（バックエンドのフレームサイズを使用）
        i = 0
        while i + self._frame_size <= len(audio):
            frame = audio[i : i + self._frame_size]

            # VAD処理
            probability = self._backend.process(frame)

            # ステートマシン更新
            self._current_time += self._frame_size / self.SAMPLE_RATE
            segment = self._state_machine.process_frame(
                audio_frame=frame,
                probability=probability,
                timestamp=self._current_time,
            )

            if segment is not None:
                segments.append(segment)

            i += self._frame_size

        # 残余を保存
        if i < len(audio):
            self._residual = audio[i:]

        return segments

    def finalize(self) -> Optional[VADSegment]:
        """処理を終了し、残っているセグメントを返す"""
        return self._state_machine.finalize(self._current_time)

    def reset(self) -> None:
        """状態をリセット"""
        self._state_machine.reset()
        self._current_time = 0.0
        self._residual = None
        if self._backend is not None:
            self._backend.reset()

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """リサンプリング"""
        from scipy import signal

        # 効率的な整数比リサンプリング
        g = gcd(orig_sr, self.SAMPLE_RATE)
        up = self.SAMPLE_RATE // g
        down = orig_sr // g

        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    @property
    def state(self) -> VADState:
        """現在のVAD状態"""
        return self._state_machine.state

    @property
    def current_time(self) -> float:
        """現在の処理時間（秒）"""
        return self._current_time

    @property
    def frame_size(self) -> int:
        """フレームサイズ（samples @ 16kHz）"""
        return self._frame_size

    @property
    def backend_name(self) -> str:
        """使用中のバックエンド名"""
        return self._backend.name
