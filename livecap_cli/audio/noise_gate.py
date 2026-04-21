"""リアルタイムノイズゲート

サンプル単位のエンベロープフォロワーにより、環境ノイズを減衰させる。
VAD の前段処理として使用し、ノイズによる VAD 誤検出を防止する。

livecap-gui v2 の RealtimeNoiseGate から移植。numba で高速化。
"""

from __future__ import annotations

import logging

import numba
import numpy as np

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _process_loop(
    audio: np.ndarray,
    output: np.ndarray,
    envelope: float,
    gate_open: int,
    release_counter: int,
    threshold: float,
    attack_coeff: float,
    release_coeff: float,
    release_samples: int,
    noise_floor: float,
) -> tuple[float, int, int]:
    """numba JIT コンパイルされたノイズゲート処理ループ。

    サンプル単位の逐次処理でエンベロープフォロワーとゲート判定を行う。
    状態管理のためベクトル化ではなくループ処理を採用。
    """
    for i in range(len(audio)):
        abs_sample = abs(audio[i])

        # エンベロープフォロワー（音声の包絡線を追跡）
        if abs_sample > envelope:
            # アタック（エンベロープを上昇）
            envelope += attack_coeff * (abs_sample - envelope)
        else:
            # リリース（エンベロープを減衰）
            envelope *= 1.0 - release_coeff

        # ゲート判定（リリースカウンターでホールド時間を実装）
        if envelope > threshold:
            gate_open = 1
            release_counter = release_samples
        elif release_counter > 0:
            release_counter -= 1
        else:
            gate_open = 0

        # 出力（ゲートが閉じている時はノイズフロアレベルに減衰）
        if gate_open:
            output[i] = audio[i]
        else:
            output[i] = audio[i] * noise_floor

    return envelope, gate_open, release_counter


class NoiseGate:
    """リアルタイムノイズゲート

    サンプル単位のエンベロープフォロワーにより、
    環境ノイズを減衰させる。VAD の前段処理として使用。
    numba JIT で高速化（< 0.1ms/100ms chunk）。
    """

    def __init__(
        self,
        threshold_db: float = -35,
        attack_ms: float = 0.5,
        release_ms: float = 30,
        sample_rate: int = 16000,
    ) -> None:
        # パラメータ検証（v2 から移植）
        if not -80 <= threshold_db <= 0:
            logger.warning(
                "Invalid threshold %sdB, using -35dB", threshold_db
            )
            threshold_db = -35

        if not 0.1 <= attack_ms <= 100:
            logger.warning(
                "Invalid attack time %sms, using 0.5ms", attack_ms
            )
            attack_ms = 0.5

        if not 1 <= release_ms <= 1000:
            logger.warning(
                "Invalid release time %sms, using 30ms", release_ms
            )
            release_ms = 30

        self._threshold = 10 ** (threshold_db / 20)
        self._attack_samples = max(1, int(attack_ms * sample_rate / 1000))
        self._release_samples = max(1, int(release_ms * sample_rate / 1000))

        # エンベロープフォロワーの係数を事前計算（指数平滑化）
        self._attack_coeff = 1.0 - np.exp(-1.0 / self._attack_samples)
        self._release_coeff = 1.0 - np.exp(-1.0 / self._release_samples)

        # ノイズフロア（ゲートが閉じた時の減衰量: -60dB）
        self._noise_floor = 10 ** (-60 / 20)

        # ゲート状態
        self._envelope: float = 0.0
        self._gate_open: int = 0  # numba 互換のため int (0/1)
        self._release_counter: int = 0

        logger.info(
            "NoiseGate initialized: threshold=%sdB, attack=%sms, release=%sms",
            threshold_db,
            attack_ms,
            release_ms,
        )

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """音声チャンクにノイズゲートを適用。

        Args:
            audio_chunk: 入力音声データ（float32, 1次元）

        Returns:
            ゲート処理後の音声データ
        """
        if len(audio_chunk) == 0:
            return audio_chunk

        output = np.empty_like(audio_chunk)

        self._envelope, self._gate_open, self._release_counter = _process_loop(
            audio_chunk,
            output,
            self._envelope,
            self._gate_open,
            self._release_counter,
            self._threshold,
            self._attack_coeff,
            self._release_coeff,
            self._release_samples,
            self._noise_floor,
        )

        return output

    def reset(self) -> None:
        """内部状態をリセット。"""
        self._envelope = 0.0
        self._gate_open = 0
        self._release_counter = 0
