"""ノイズキャリブレーション分析ユーティリティ。

録音したノイズサンプル列から推奨閾値・危険ゾーンを算出する純関数群。
CLI ``levels`` コマンドおよび GUI キャリブレーション UI 双方から再利用される。

NoiseGate (``livecap_cli/audio/noise_gate.py``) の判定は
**per-sample envelope follower** (実 peak 追跡) で行われるため、calibration
側も **per-chunk peak (|x|.max())** を入力にして単位を揃える。
``samples_db`` (chunk RMS) は noise floor / RMS p95 の diagnostic としてのみ
使用し、``suggested_threshold_db`` は ``peak_p95 + PEAK_SAFETY_MARGIN_DB``
で求める。

経緯 (livecap-gui#331 / livecap-cli#291): 旧実装は chunk RMS p95 + 10 dB を
推奨値としており、White noise の crest factor ≈ 11 dB が偶然 ``+10`` で
吸収されていただけだった。impulsive noise (キーボード/呼吸/breath bursts)
では crest factor が大きくなり threshold が peak の下に潜り、envelope
follower が瞬間超え → 無音時 hallucination を引き起こしていた。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

PEAK_SAFETY_MARGIN_DB = 6.0
"""``suggested_threshold_db = peak_p95_db + PEAK_SAFETY_MARGIN_DB``。

``+6 dB`` は NoiseGate 既定 (``attack_ms=0.5``, ``release_ms=100``,
``sample_rate=16000``) に対する実測ベースのマージン。``attack_ms`` を大幅に
短くすると envelope の peak 追従が鋭くなるため margin の見直しが必要。

将来 follow-up (#283 と組で別 issue 化予定): NoiseGate の envelope follower
filter を calibration 入力に対して simulate し envelope の 95%ile を取れば
margin を 1-2 dB に縮められる。
"""


@dataclass(frozen=True)
class NoiseAnalysis:
    """ノイズサンプル分析結果。

    Attributes:
        noise_floor_db: chunk RMS の 25%ile (RMS-unit, diagnostic)。
        noise_rms_p95_db: chunk RMS の 95%ile (RMS-unit, diagnostic)。
        peak_p95_db: per-chunk ``|x|.max()`` の 95%ile (peak-unit)。
            ``suggested_threshold_db`` の計算基準。NoiseGate envelope
            follower の単位と一致する。
        suggested_threshold_db: ``peak_p95_db + PEAK_SAFETY_MARGIN_DB``。
            NoiseGate の ``threshold_db`` にそのまま渡せる値。
        danger_zone: ``(noise_floor_db - 5, noise_floor_db + 5)``。
            **RMS-unit diagnostic**: 手動で閾値をこの RMS 範囲に設定すると
            floor の揺らぎで gate がフリッカーするため避けるべき領域。
            ``suggested_threshold_db`` は peak-unit のため直接比較不可。
        sample_count: 入力サンプル数。
        duration_s: 録音時間 (秒)。
    """

    noise_floor_db: float
    noise_rms_p95_db: float
    peak_p95_db: float
    suggested_threshold_db: float
    danger_zone: tuple[float, float]
    sample_count: int
    duration_s: float


def analyze_noise_samples(
    samples_db: Sequence[float] | np.ndarray,
    peak_samples_db: Sequence[float] | np.ndarray,
    sample_rate_hz: float = 10.0,
) -> NoiseAnalysis:
    """ノイズ測定サンプル列から推奨閾値を計算する。

    NoiseGate (``livecap_cli/audio/noise_gate.py``) の envelope follower は
    per-sample ``|x|`` を追跡するため、calibration も per-chunk
    ``|x|.max()`` を収集して unit を揃える。chunk RMS は noise_floor /
    noise_rms_p95 の diagnostic としてのみ使う (``suggested_threshold_db``
    には影響しない)。

    Args:
        samples_db: chunk RMS の dB 列 (``20*log10(rms(chunk))``)。
        peak_samples_db: chunk peak の dB 列
            (``20*log10(|chunk|.max())``)。``len(peak_samples_db) ==
            len(samples_db)`` でなければならない。
        sample_rate_hz: chunk 取得レート (``duration_s`` の計算用)。

    Returns:
        NoiseAnalysis: 分析結果 (``suggested_threshold_db`` は peak ベース)。

    Raises:
        ValueError: ``samples_db`` / ``peak_samples_db`` が空、長さ不一致、
            または ``sample_rate_hz`` が非正値の場合。
    """
    samples = np.asarray(samples_db, dtype=np.float64)
    peaks = np.asarray(peak_samples_db, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("samples_db must not be empty")
    if peaks.size == 0:
        raise ValueError("peak_samples_db must not be empty")
    if peaks.size != samples.size:
        raise ValueError(
            "samples_db and peak_samples_db length mismatch "
            f"({samples.size} vs {peaks.size})"
        )
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    noise_floor = float(np.percentile(samples, 25))
    noise_rms_p95 = float(np.percentile(samples, 95))
    peak_p95 = float(np.percentile(peaks, 95))

    return NoiseAnalysis(
        noise_floor_db=noise_floor,
        noise_rms_p95_db=noise_rms_p95,
        peak_p95_db=peak_p95,
        suggested_threshold_db=peak_p95 + PEAK_SAFETY_MARGIN_DB,
        danger_zone=(noise_floor - 5.0, noise_floor + 5.0),
        sample_count=int(samples.size),
        duration_s=float(samples.size / sample_rate_hz),
    )
