"""Unit tests for ``livecap_cli.audio.transient_detector`` (Issue #295 PR-B).

The detector is exercised against deterministic synthetic stimuli so the six
DSP features and the AND-decision can be asserted directly. Synthesisers
that already live in ``benchmarks/non_speech_filter/corpus.py`` are reused
to keep one source of truth for waveform shaping.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.non_speech_filter.corpus import (
    _single_clap,
    _synthesize_applause_burst,
    _synthesize_silence_amplified,
    _synthesize_speech_proxy,
)
from livecap_cli.audio.transient_detector import (
    VALID_MODES,
    TransientDetector,
    TransientDetectorConfig,
    TransientFeatures,
)


SAMPLE_RATE = 16000


# ---------- Helpers ------------------------------------------------------


def _config(**overrides) -> TransientDetectorConfig:
    """Return a fresh config with any overrides applied."""
    base = dict(
        mode="observe",
        flatness_min=0.30,
        centroid_min_hz=2500.0,
        zcr_min=0.12,
        onset_ratio=3.0,
        voiced_max=0.25,
        rms_min_db=-35.0,
        frame_ms=32.0,
        hop_ms=16.0,
    )
    base.update(overrides)
    return TransientDetectorConfig(**base)


def _process_full(audio: np.ndarray, config=None) -> tuple[TransientDetector, list[TransientFeatures]]:
    detector = TransientDetector(config or _config(), sample_rate=SAMPLE_RATE)
    _output, events = detector.process(audio.astype(np.float32))
    return detector, events


# ---------- Config validation -------------------------------------------


class TestConfigValidation:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            TransientDetectorConfig(mode="loud")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frame_ms": 0.0},
            {"frame_ms": float("nan")},
            {"hop_ms": 0.0},
            {"hop_ms": float("inf")},
            {"frame_ms": 16.0, "hop_ms": 32.0},  # hop > frame
            {"pitch_min_hz": 500.0, "pitch_max_hz": 400.0},
            {"onset_baseline_window_frames": 0},
            {"onset_baseline_warmup_frames": -1},
            {"flatness_min": float("nan")},
        ],
    )
    def test_invalid_numeric_kwargs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TransientDetectorConfig(**kwargs)

    def test_valid_default(self) -> None:
        cfg = TransientDetectorConfig()
        assert cfg.mode in VALID_MODES


# ---------- Feature computation -----------------------------------------


class TestFeatureComputation:
    """Per-feature behaviour against canonical stimuli."""

    def test_white_noise_has_high_flatness_and_centroid(self) -> None:
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1
        detector, _ = _process_full(audio)
        # Telemetry should record many frames where flatness/centroid passed.
        tel = detector.telemetry
        assert tel.frames_processed > 30
        assert tel.pass_flatness > tel.frames_processed * 0.5
        assert tel.pass_centroid > tel.frames_processed * 0.5

    def test_low_frequency_tone_has_low_flatness_and_low_centroid(self) -> None:
        t = np.arange(SAMPLE_RATE).astype(np.float32) / SAMPLE_RATE
        audio = 0.3 * np.sin(2 * np.pi * 200.0 * t)
        detector, _ = _process_full(audio.astype(np.float32))
        tel = detector.telemetry
        # A 200 Hz pure tone must not satisfy flatness>0.30 nor centroid>2500.
        # Some frames during the onset may transiently pass but the vast
        # majority should reject.
        assert tel.pass_flatness < tel.frames_processed * 0.10
        assert tel.pass_centroid < tel.frames_processed * 0.10

    def test_zcr_high_for_noise_low_for_tone(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1
        t = np.arange(SAMPLE_RATE).astype(np.float32) / SAMPLE_RATE
        tone = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

        det_noise, _ = _process_full(noise)
        det_tone, _ = _process_full(tone)

        # White noise saturates the zcr threshold; a 150 Hz tone never gets
        # close.
        assert det_noise.telemetry.pass_zcr > det_noise.telemetry.frames_processed * 0.5
        assert det_tone.telemetry.pass_zcr < det_tone.telemetry.frames_processed * 0.10

    def test_voiced_speech_proxy_has_low_unvoiced_count(self) -> None:
        # Speech proxy is highly voiced; voiced_ratio should rarely fall
        # below voiced_max (0.25), so the "applause-side" voiced pass count
        # is small.
        audio = _synthesize_speech_proxy(duration_ms=1500.0, sample_rate=SAMPLE_RATE)
        detector, _ = _process_full(audio)
        tel = detector.telemetry
        # Generously allow a few frames near the syllabic AM zero-crossings.
        assert tel.pass_voiced < tel.frames_processed * 0.30

    def test_silence_does_not_trigger_rms_pass(self) -> None:
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        detector, events = _process_full(audio)
        assert events == []
        # rms_db is -inf for pure silence which never passes rms_min.
        assert detector.telemetry.pass_rms == 0


# ---------- Applause-like AND decision -----------------------------------


class TestApplauseDecision:
    """The end-to-end AND decision on the canonical fixtures."""

    def test_burst_applause_triggers_some_frames(self) -> None:
        # Rapid burst — the synthetic case that current pipelines fail on.
        audio = _synthesize_applause_burst(
            n_claps=7, inter_clap_ms=120.0, rms_db=-15.0, sample_rate=SAMPLE_RATE
        )
        detector, events = _process_full(audio.astype(np.float32))
        # At least one frame inside the burst must be flagged.
        assert detector.telemetry.applause_frames >= 1
        assert len(events) >= 1
        # And every event must point to a frame with the AND result.
        for evt in events:
            assert evt.is_applause_like is True

    def test_voiced_speech_does_not_trigger(self) -> None:
        audio = _synthesize_speech_proxy(duration_ms=1500.0, sample_rate=SAMPLE_RATE)
        _detector, events = _process_full(audio.astype(np.float32))
        assert events == []

    def test_silence_amplified_does_not_trigger(self) -> None:
        # Mirrors the #292 silence-amplified fixture (low RMS noise).
        audio = _synthesize_silence_amplified(
            duration_ms=1500.0, rms_db=-50.0, sample_rate=SAMPLE_RATE
        )
        _detector, events = _process_full(audio.astype(np.float32))
        # rms_min_db (-35) gates this case out unconditionally.
        assert events == []

    def test_loosening_voiced_max_does_not_suddenly_trigger_speech(self) -> None:
        # Even a generous voiced_max ceiling must not turn voiced speech
        # into a transient — the other AND terms must still gate it.
        audio = _synthesize_speech_proxy(duration_ms=1500.0, sample_rate=SAMPLE_RATE)
        cfg = _config(voiced_max=0.95)
        _detector, events = _process_full(audio.astype(np.float32), config=cfg)
        assert events == []

    def test_single_loud_clap_passes_rms_gate(self) -> None:
        clap = _single_clap(duration_ms=80.0, rms_db=-18.0, sample_rate=SAMPLE_RATE, seed=11)
        # Pad with silence so warmup completes before the clap.
        pad = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s
        audio = np.concatenate([pad, clap.astype(np.float32), pad])
        detector, _events = _process_full(audio)
        # Should detect at least one applause-like frame post-warmup.
        assert detector.telemetry.applause_frames >= 1


# ---------- Stateful streaming ------------------------------------------


class TestStreamingEquivalence:
    """Chunked feeds should match the single-chunk result."""

    def test_chunked_matches_single_call(self) -> None:
        audio = _synthesize_applause_burst(
            n_claps=5, inter_clap_ms=150.0, sample_rate=SAMPLE_RATE
        ).astype(np.float32)

        full = TransientDetector(_config(), sample_rate=SAMPLE_RATE)
        full.process(audio)

        chunked = TransientDetector(_config(), sample_rate=SAMPLE_RATE)
        # Feed 100 ms chunks; the residual buffer should glue frames.
        chunk_size = SAMPLE_RATE // 10
        for i in range(0, audio.size, chunk_size):
            chunked.process(audio[i : i + chunk_size])

        # Telemetry must match exactly (deterministic stimulus).
        assert chunked.telemetry.frames_processed == full.telemetry.frames_processed
        assert chunked.telemetry.applause_frames == full.telemetry.applause_frames

    def test_reset_clears_streaming_state(self) -> None:
        audio = _synthesize_applause_burst(
            n_claps=3, sample_rate=SAMPLE_RATE
        ).astype(np.float32)
        detector = TransientDetector(_config(), sample_rate=SAMPLE_RATE)
        detector.process(audio)
        assert detector.telemetry.frames_processed > 0
        prior_applause = detector.telemetry.applause_frames

        detector.reset()
        # Counters survive reset; streaming state does not.
        assert detector.telemetry.applause_frames == prior_applause
        # Re-processing the same audio after reset adds the same number of
        # frames again (state cleared).
        detector.process(audio)
        assert detector.telemetry.applause_frames == 2 * prior_applause


# ---------- Mode semantics ----------------------------------------------


class TestModes:
    def test_observe_mode_does_not_modify_audio(self) -> None:
        audio = _synthesize_applause_burst(sample_rate=SAMPLE_RATE).astype(np.float32)
        detector = TransientDetector(_config(mode="observe"), sample_rate=SAMPLE_RATE)
        output, _events = detector.process(audio)
        assert output is audio  # observe returns the input unchanged

    def test_on_mode_zeros_applause_frames(self) -> None:
        audio = _synthesize_applause_burst(
            n_claps=7, inter_clap_ms=120.0, sample_rate=SAMPLE_RATE
        ).astype(np.float32)
        detector = TransientDetector(_config(mode="on"), sample_rate=SAMPLE_RATE)
        output, events = detector.process(audio)
        assert len(events) >= 1
        # The masked output has strictly less energy than the input
        # whenever at least one frame was flagged.
        assert float(np.sum(output ** 2)) < float(np.sum(audio ** 2))
        # The output array is a fresh allocation, never the input.
        assert output is not audio

    def test_on_mode_does_not_zero_clean_speech(self) -> None:
        audio = _synthesize_speech_proxy(
            duration_ms=1500.0, sample_rate=SAMPLE_RATE
        ).astype(np.float32)
        detector = TransientDetector(_config(mode="on"), sample_rate=SAMPLE_RATE)
        output, events = detector.process(audio)
        assert events == []
        # No frame masked → output equals input bit-for-bit.
        np.testing.assert_array_equal(output, audio)
