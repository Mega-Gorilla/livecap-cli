"""ノイズキャリブレーション分析ユーティリティ。

録音したノイズサンプル列から推奨閾値・危険ゾーンを算出する純関数群。
CLI ``levels`` コマンドおよび GUI キャリブレーション UI 双方から再利用される。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NoiseAnalysis:
    """ノイズサンプル分析結果。

    dB 単位のレベルサンプル列からノイズフロア/ピーク/推奨閾値を算出した結果。
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
