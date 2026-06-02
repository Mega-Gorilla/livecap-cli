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

ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0
"""``suggested_engine_min_rms_dbfs = noise_rms_p95_db + ENGINE_MIN_RMS_SAFETY_MARGIN_DB``。

EnergyGate (``StreamTranscriber._should_skip_low_energy``) は per-segment
frame RMS を判定するため、calibration 段階の chunk RMS p95 を基準に
safety margin を加えて推奨値を作る。``+6 dB`` は livecap-gui#331 の
empirical probe と livecap-cli#292 の実音源プローブ結果に基づく経験値。

Note: ``PEAK_SAFETY_MARGIN_DB`` (peak-unit, NoiseGate) と数値が同一でも
物理量が異なる (per-sample peak vs per-frame RMS)。閾値共有不可。
"""

ENERGY_METRICS = ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms")
"""EnergyGate がサポートする per-segment metric の名前一覧。

``_segment_energy_dbfs()`` の ``metric`` 引数および CLI
``--engine-energy-metric`` で受け付ける値。
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
        suggested_engine_min_rms_dbfs:
            ``noise_rms_p95_db + engine_min_rms_margin_db``。
            ``StreamTranscriber.engine_min_rms_dbfs`` に渡せる値
            (per-frame RMS unit; #292 EnergyGate)。``noise_rms_p95_db``
            と物理量が一致するため、NoiseGate threshold とは別の値。
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
    suggested_engine_min_rms_dbfs: float
    danger_zone: tuple[float, float]
    sample_count: int
    duration_s: float


def analyze_noise_samples(
    samples_db: Sequence[float] | np.ndarray,
    peak_samples_db: Sequence[float] | np.ndarray,
    sample_rate_hz: float = 10.0,
    *,
    engine_min_rms_margin_db: float = ENGINE_MIN_RMS_SAFETY_MARGIN_DB,
) -> NoiseAnalysis:
    """ノイズ測定サンプル列から推奨閾値を計算する。

    NoiseGate (``livecap_cli/audio/noise_gate.py``) の envelope follower は
    per-sample ``|x|`` を追跡するため、calibration も per-chunk
    ``|x|.max()`` を収集して unit を揃える。chunk RMS は noise_floor /
    noise_rms_p95 の diagnostic および ``suggested_engine_min_rms_dbfs``
    (#292 EnergyGate) の計算基準として使う。

    Args:
        samples_db: chunk RMS の dB 列 (``20*log10(rms(chunk))``)。
        peak_samples_db: chunk peak の dB 列
            (``20*log10(|chunk|.max())``)。``len(peak_samples_db) ==
            len(samples_db)`` でなければならない。
        sample_rate_hz: chunk 取得レート (``duration_s`` の計算用)。
        engine_min_rms_margin_db: ``suggested_engine_min_rms_dbfs`` 計算用
            の safety margin (dB)。default は :data:`ENGINE_MIN_RMS_SAFETY_MARGIN_DB`。
            user が任意に変更可能 (CLI ``--engine-min-rms-margin``)。

    Returns:
        NoiseAnalysis: 分析結果 (``suggested_threshold_db`` は peak ベース、
            ``suggested_engine_min_rms_dbfs`` は RMS ベース)。

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
        suggested_engine_min_rms_dbfs=noise_rms_p95 + engine_min_rms_margin_db,
        danger_zone=(noise_floor - 5.0, noise_floor + 5.0),
        sample_count=int(samples.size),
        duration_s=float(samples.size / sample_rate_hz),
    )


def _segment_energy_dbfs(
    audio: np.ndarray,
    sample_rate: int,
    metric: str = "max_frame_rms",
    frame_ms: float = 32.0,
) -> float:
    """Per-segment energy を選択 metric で測定し dBFS で返す (#292 EnergyGate)。

    本関数は ``StreamTranscriber._should_skip_low_energy`` から呼ばれ、
    low-energy segment を engine に渡す前に短絡するための判定値を返す。

    Args:
        audio: float32 audio samples (mono)。
        sample_rate: サンプリングレート (Hz)。
        metric: per-segment energy 指標 (member of :data:`ENERGY_METRICS`):

            - ``"max_frame_rms"`` (default): ``frame_ms`` 窓ごとの RMS の max。
              VAD padding 希釈に耐性 (短文発話/小声でも実 speech frame の
              energy を捕捉)。
            - ``"whole_rms"``: segment 全体の RMS。Aggressive、padding 希釈
              リスクあり。stress test で最も多くの silent segment を drop。
            - ``"p95_frame_rms"``: ``frame_ms`` 窓ごとの RMS の 95%ile。
              max と whole の中庸、複数 frame の合意必要。
            - ``"top3_frame_rms"``: ``frame_ms`` 窓ごとの RMS の top-3 の mean。
              単発 transient (1 frame だけ大きい) に対する false-pass resistance。
        frame_ms: frame-based metrics で使う窓長 (ms)。``frame_ms <= 0`` や
            audio が 1 frame に満たない場合は ``whole_rms`` に fallback。

    Returns:
        Energy 値 (dBFS)。

    Raises:
        ValueError: ``metric`` が :data:`ENERGY_METRICS` に含まれない場合。

    Note:
        本関数が返す値は per-segment / per-frame の RMS dBFS。
        NoiseGate の per-sample peak envelope (``noise_gate.py``) とは
        物理量が異なるため、閾値を共有してはいけない。

    Trade-off (#292 empirical, livecap-cli#292 テスト結果.mov):
        Stress test (silent audio @ peak -3 dBFS norm) で 1s window × 73、
        threshold -45 dBFS:

        - whole_rms:        55/73 (75%) drop, padding 希釈に弱い
        - max_frame_rms:    19/73 (26%) drop, padding 希釈に強い
        - p95_frame_rms:    29/73 (40%) drop, 中庸
        - top3_frame_rms:   22/73 (30%) drop, transient resistance

        Production 用途 (VAD default threshold で正常 segment) では
        ``max_frame_rms`` (default) が padding 希釈耐性で勝る。stress 用途
        (VAD false-positive 多数) では ``whole_rms`` も検討価値あり。
    """
    if metric not in ENERGY_METRICS:
        raise ValueError(
            f"unknown metric: {metric!r} (choose from {ENERGY_METRICS})"
        )
    if frame_ms <= 0 or metric == "whole_rms":
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return 20.0 * float(np.log10(max(rms, 1e-10)))
    frame_n = int(sample_rate * frame_ms / 1000.0)
    if frame_n <= 0 or len(audio) < frame_n:
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        return 20.0 * float(np.log10(max(rms, 1e-10)))
    n_frames = len(audio) // frame_n
    frames = (
        audio[: n_frames * frame_n].reshape(n_frames, frame_n).astype(np.float32)
    )
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    if metric == "max_frame_rms":
        value = float(np.max(frame_rms))
    elif metric == "p95_frame_rms":
        value = float(np.percentile(frame_rms, 95))
    else:  # top3_frame_rms
        k = min(3, len(frame_rms))
        value = float(np.mean(np.sort(frame_rms)[-k:]))
    return 20.0 * float(np.log10(max(value, 1e-10)))
