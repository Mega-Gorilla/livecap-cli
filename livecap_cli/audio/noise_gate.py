"""リアルタイムノイズゲート

サンプル単位のエンベロープフォロワーにより、環境ノイズを減衰させる。
VAD の前段処理として使用し、ノイズによる VAD 誤検出を防止する。

livecap-gui v2 の RealtimeNoiseGate から移植。numba で高速化。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

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


@dataclass(frozen=True)
class NoiseAnalysis:
    """ノイズサンプル分析結果。

    dB 単位のレベルサンプル列からノイズフロア/ピーク/推奨閾値を算出した結果。
    CLI ``levels`` コマンドおよび GUI キャリブレーション UI 双方から使用される。
    """

    noise_floor_db: float
    noise_peak_db: float
    suggested_threshold_db: float
    danger_zone: tuple[float, float]
    safe_zone_min_db: float
    sample_count: int
    duration_s: float


def analyze_noise_samples(
    samples_db: Sequence[float] | np.ndarray,
    sample_rate_hz: float = 10.0,
) -> NoiseAnalysis:
    """ノイズ測定サンプル列から推奨閾値と危険ゾーンを計算する。

    Args:
        samples_db: dB 単位のレベルサンプル (RMS 値を 20*log10 したもの)。
        sample_rate_hz: サンプル取得レート (duration_s 計算用)。

    Returns:
        NoiseAnalysis: 分析結果 dataclass。

    Raises:
        ValueError: samples_db が空、または sample_rate_hz が非正値の場合。

    Note:
        マージン値の根拠は livecap-gui PR #294 実測:
        - ±5 dB 領域は「死のゾーン」 (gate flicker 多発)
        - +10 dB 以上が安全マージン
    """
    samples = np.asarray(samples_db, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("samples_db must not be empty")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    noise_floor = float(np.percentile(samples, 25))
    noise_peak = float(np.percentile(samples, 95))

    return NoiseAnalysis(
        noise_floor_db=noise_floor,
        noise_peak_db=noise_peak,
        suggested_threshold_db=noise_peak + 10.0,
        danger_zone=(noise_floor - 5.0, noise_floor + 5.0),
        safe_zone_min_db=noise_peak + 5.0,
        sample_count=int(samples.size),
        duration_s=float(samples.size / sample_rate_hz),
    )
