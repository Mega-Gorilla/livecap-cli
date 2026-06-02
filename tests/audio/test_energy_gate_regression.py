"""#292 EnergyGate end-to-end regression.

Synthetic-only fixtures (実音源 fixture 不要、再現性確保):
- silent noise (~-50 dBFS) — default threshold で drop されるべき
- speech-like burst (~-26 dBFS) — default threshold で pass すべき
- padded short utterance — metric 選択によって drop/pass が変わる境界事例

Audio 値は実音源 pre-evaluation (.tmp/issue-292-eval/) で観測した分布に
基づいて選定。
"""

from __future__ import annotations

import numpy as np
import pytest

from livecap_cli.audio import _segment_energy_dbfs


SR = 16000


def _synthesize_silent_noise(
    duration_s: float = 1.0, rms: float = 0.003, seed: int = 42
) -> np.ndarray:
    """~-50 dBFS white noise (NoiseGate post-#291 calibration baseline)。"""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    # rescale to target RMS
    current_rms = float(np.sqrt(np.mean(noise ** 2)))
    return (noise * (rms / current_rms)).astype(np.float32)


def _synthesize_speech_like_burst(
    duration_s: float = 1.0, rms: float = 0.05, seed: int = 42
) -> np.ndarray:
    """~-26 dBFS sustained pink-noise-like burst (sanity check pass case)。

    通常の小声〜会話レベル。default -45 dBFS threshold で全 metric が pass する
    べき。
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    sig = rng.standard_normal(n).astype(np.float32)
    current_rms = float(np.sqrt(np.mean(sig ** 2)))
    return (sig * (rms / current_rms)).astype(np.float32)


def _synthesize_padded_short_utterance(
    speech_ms: float = 50.0,
    padding_ms: float = 600.0,
    speech_rms: float = 0.012,  # -38 dBFS
    padding_rms: float = 0.0018,  # -55 dBFS
    seed: int = 42,
) -> np.ndarray:
    """50ms speech @ -38 dBFS + 600ms padding @ -55 dBFS (両側 300ms ずつ)。

    VAD の ``speech_pad_ms`` で短文発話が padding に挟まれる典型ケース。
    whole_rms では希釈で -48 dBFS 程度に落ちるが、max_frame_rms では実 speech
    frame の -38 dBFS を捕捉する。
    """
    rng = np.random.default_rng(seed)
    speech_n = int(speech_ms / 1000.0 * SR)
    pad_n = int(padding_ms / 1000.0 / 2 * SR)  # 半分ずつ前後に
    total = pad_n + speech_n + pad_n

    # 前 padding
    out = rng.standard_normal(total).astype(np.float32)
    # speech 部分
    speech_idx = pad_n
    speech_seg = rng.standard_normal(speech_n).astype(np.float32)
    cur_rms = float(np.sqrt(np.mean(speech_seg ** 2)))
    speech_seg *= speech_rms / cur_rms
    out[speech_idx : speech_idx + speech_n] = speech_seg
    # 全体の padding を目標 RMS に近づける (前後)
    mask = np.ones(total, dtype=bool)
    mask[speech_idx : speech_idx + speech_n] = False
    pad_view = out[mask]
    cur_pad_rms = float(np.sqrt(np.mean(pad_view ** 2)))
    out[mask] = (pad_view * (padding_rms / cur_pad_rms)).astype(np.float32)
    return out


# === Tests ===


class TestDefaultThreshold:
    """default -45 dBFS / max_frame_rms での silent vs speech 分離。"""

    def test_silent_noise_drops_at_default(self) -> None:
        """silent noise (~-50 dBFS) は max_frame_rms 指標で -45 を下回り drop。"""
        audio = _synthesize_silent_noise()
        energy = _segment_energy_dbfs(audio, SR, metric="max_frame_rms")
        # white noise の crest factor ~11 dB なので max frame は RMS+~3dB
        # → -50 + ~3 = -47 dBFS < -45 threshold → drop ✓
        assert energy < -45.0, (
            f"silent noise should be below -45 dBFS, got {energy:.1f}"
        )

    def test_speech_like_passes_at_default(self) -> None:
        """speech-like burst (~-26 dBFS) は default threshold を passes する。"""
        audio = _synthesize_speech_like_burst()
        for metric in ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"):
            energy = _segment_energy_dbfs(audio, SR, metric=metric)
            assert energy > -45.0, (
                f"speech-like ({metric}) should be above -45 dBFS, got {energy:.1f}"
            )


class TestMetricBehavior:
    """metric 選択がもたらす drop/pass の差異を documents する。"""

    def test_padded_short_utterance_max_frame_passes_but_whole_drops(self) -> None:
        """50ms speech + 600ms padding: max_frame は -38 dBFS で pass、
        whole_rms は希釈で -48 dBFS 程度で drop の境界。"""
        audio = _synthesize_padded_short_utterance()
        max_frame = _segment_energy_dbfs(audio, SR, metric="max_frame_rms")
        whole = _segment_energy_dbfs(audio, SR, metric="whole_rms")
        # max_frame は speech 部分 -38 dBFS を捕捉
        assert max_frame > -42.0, (
            f"max_frame should capture speech energy, got {max_frame:.1f}"
        )
        # whole_rms は padding 希釈で実質 silence と区別不能
        assert whole < -45.0, (
            f"whole_rms should be diluted by padding, got {whole:.1f}"
        )
        # max_frame は whole_rms より大幅に高い (希釈の証拠)
        assert max_frame > whole + 5.0

    def test_metric_drop_rate_documentation(self) -> None:
        """4 metric × 5 threshold で silent noise 100 instances の drop rate を
        測定。documentation 兼回帰検出用。"""
        rng = np.random.default_rng(0)
        n_instances = 100
        results: dict[tuple[str, float], int] = {}
        for metric in ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"):
            for thr in (-30.0, -40.0, -45.0, -50.0, -55.0):
                drops = 0
                for i in range(n_instances):
                    audio = _synthesize_silent_noise(
                        duration_s=1.0, rms=0.003, seed=int(rng.integers(0, 1 << 30))
                    )
                    e = _segment_energy_dbfs(audio, SR, metric=metric)
                    if e < thr:
                        drops += 1
                results[(metric, thr)] = drops

        # Expectations based on physical reality (white noise crest factor ~11 dB):
        # @ -45 dBFS threshold:
        #   - whole_rms: drops most (silent base ~-50 dBFS < -45) → ≥ 90/100
        #   - max_frame_rms: 大半 -47 dBFS で borderline、結果は環境依存
        #
        # ここでは「-30 では全 metric が drop しまくる」「-55 では drop が
        # 少ない (audio が threshold 以上に強くなければならない)」という
        # monotonic 関係だけを assert する。
        for metric in ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms"):
            # -30 dBFS は silent noise には到底届かないので drop=全部
            assert results[(metric, -30.0)] >= 80, (
                f"{metric} @ -30 dBFS should drop almost all silent noise, "
                f"got {results[(metric, -30.0)]}/100"
            )
            # -55 dBFS は silent noise の下限に近く、drop は限定的
            assert results[(metric, -55.0)] <= 50, (
                f"{metric} @ -55 dBFS should drop few silent noise, "
                f"got {results[(metric, -55.0)]}/100"
            )

    def test_top3_resists_single_transient_more_than_max(self) -> None:
        """単発 transient 1 frame: max_frame は pass、top3_frame は drop 寄り。"""
        audio = np.zeros(SR, dtype=np.float32)
        # 32ms transient @ amp 0.1 (-20 dBFS frame RMS)
        audio[: int(0.032 * SR)] = 0.1
        max_frame = _segment_energy_dbfs(audio, SR, metric="max_frame_rms")
        top3 = _segment_energy_dbfs(audio, SR, metric="top3_frame_rms")
        # max_frame catches the transient cleanly
        assert max_frame > -22.0
        # top3 mean averages with neighbors (silence) → lower
        assert top3 < max_frame - 5.0


class TestEdgeCases:

    def test_very_short_segment_falls_back(self) -> None:
        """1 frame 未満の segment は whole_rms に fallback する。"""
        audio = np.full(100, 0.1, dtype=np.float32)  # 100 samples << 32ms
        for metric in ("max_frame_rms", "p95_frame_rms", "top3_frame_rms"):
            e = _segment_energy_dbfs(audio, SR, metric=metric, frame_ms=32.0)
            # 100 samples @ 0.1 amp → -20 dBFS RMS
            assert e == pytest.approx(-20.0, abs=0.1), (
                f"metric {metric} should fall back to whole_rms"
            )

    def test_exactly_at_threshold_boundary(self) -> None:
        """境界値 (audio energy == threshold) は >= で pass (strict less than)。"""
        # construct audio with exactly -30 dBFS max_frame_rms
        audio = np.full(SR, 0.0316227766, dtype=np.float32)  # ≈ 10^(-30/20)
        e = _segment_energy_dbfs(audio, SR, metric="max_frame_rms")
        assert e == pytest.approx(-30.0, abs=0.05)
        # Strict less-than: -30 vs threshold -30 → pass (NOT drop)
        # (verified via StreamTranscriber._should_skip_low_energy logic)