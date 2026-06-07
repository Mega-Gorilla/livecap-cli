"""DSP transient / applause detector — Issue #295 Phase 1 Layer 1.

Detects short broadband impulses (claps, desk taps, etc.) ahead of the VAD so
the upstream gate stack can either *observe* them (default) or *zero them
out* (``mode='on'``). Six per-frame DSP features are AND-combined for high
precision; the design rationale is documented in Issue #295.

Design summary:

- Frame-based processing (32 ms frame, 16 ms hop @ 16 kHz).
- Stateful: residual buffer + rolling onset baseline + previous spectrum.
- Three modes:

  - ``off``     — detector is not constructed (caller passes ``None``)
  - ``observe`` — features + telemetry only, audio passes through unchanged
  - ``on``      — applause-flagged frames are zeroed-out in the output
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


VALID_MODES: tuple[str, ...] = ("off", "observe", "on")
"""Allowed values for :attr:`TransientDetectorConfig.mode`."""


@dataclass(frozen=True, slots=True)
class TransientDetectorConfig:
    """Configuration for :class:`TransientDetector`.

    All thresholds default to the values recommended in Issue #295 v3. The
    AND combination semantics treat each threshold as a *condition* that
    must be satisfied for the frame to be classified as applause-like.
    """

    mode: Literal["off", "observe", "on"] = "observe"

    # Threshold conditions (AND-combined).
    flatness_min: float = 0.30
    centroid_min_hz: float = 2500.0
    zcr_min: float = 0.12
    onset_ratio: float = 3.0
    voiced_max: float = 0.25
    rms_min_db: float = -35.0

    # Frame settings.
    frame_ms: float = 32.0
    hop_ms: float = 16.0

    # Voiced-confidence (autocorrelation peak search range).
    pitch_min_hz: float = 60.0
    pitch_max_hz: float = 400.0

    # Onset baseline tracking.
    onset_baseline_window_frames: int = 31
    onset_baseline_warmup_frames: int = 8

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"TransientDetectorConfig.mode must be one of {VALID_MODES}, "
                f"got {self.mode!r}"
            )
        if not math.isfinite(self.frame_ms) or self.frame_ms <= 0:
            raise ValueError("frame_ms must be a finite positive number")
        if not math.isfinite(self.hop_ms) or self.hop_ms <= 0:
            raise ValueError("hop_ms must be a finite positive number")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must be <= frame_ms (overlap, not skip)")
        if self.pitch_max_hz <= self.pitch_min_hz:
            raise ValueError("pitch_max_hz must be > pitch_min_hz")
        if self.onset_baseline_window_frames <= 0:
            raise ValueError("onset_baseline_window_frames must be > 0")
        if self.onset_baseline_warmup_frames < 0:
            raise ValueError("onset_baseline_warmup_frames must be >= 0")
        for name in (
            "flatness_min",
            "centroid_min_hz",
            "zcr_min",
            "onset_ratio",
            "voiced_max",
            "rms_min_db",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class TransientFeatures:
    """Per-frame DSP feature snapshot.

    ``is_applause_like`` is ``True`` iff every per-feature condition in the
    parent :class:`TransientDetectorConfig` passed for this frame. Frames
    inside the warmup window always have ``is_applause_like=False``.
    """

    start_time_s: float
    end_time_s: float
    spectral_flatness: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    onset_strength: float
    onset_strength_baseline: float
    voiced_ratio: float
    rms_db: float
    is_applause_like: bool


@dataclass
class TransientDetectorTelemetry:
    """Cumulative counters for sensitivity analysis and runtime logging.

    Per-feature ``pass_*`` counters increment whenever the corresponding
    condition was satisfied for a frame (independently of the AND result).
    They make it easy to spot which feature is the bottleneck during
    threshold sweeps.
    """

    frames_processed: int = 0
    applause_frames: int = 0
    audio_chunks_processed: int = 0
    pass_flatness: int = 0
    pass_centroid: int = 0
    pass_zcr: int = 0
    pass_onset: int = 0
    pass_voiced: int = 0
    pass_rms: int = 0

    def snapshot(self) -> "TransientDetectorTelemetry":
        """Return an immutable copy of the current counters."""
        return TransientDetectorTelemetry(
            frames_processed=self.frames_processed,
            applause_frames=self.applause_frames,
            audio_chunks_processed=self.audio_chunks_processed,
            pass_flatness=self.pass_flatness,
            pass_centroid=self.pass_centroid,
            pass_zcr=self.pass_zcr,
            pass_onset=self.pass_onset,
            pass_voiced=self.pass_voiced,
            pass_rms=self.pass_rms,
        )


class TransientDetector:
    """Stateful frame-based DSP transient detector.

    Use ``mode='observe'`` (default) to feed the metric layer without
    altering the audio. Use ``mode='on'`` to additionally zero-out frames
    that the AND combination classifies as applause-like.

    The detector is *not* thread-safe by itself; each
    :class:`StreamTranscriber` owns one instance.
    """

    def __init__(
        self,
        config: TransientDetectorConfig,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

        self.config = config
        self._sample_rate = sample_rate

        self._frame_samples = max(1, int(round(sample_rate * config.frame_ms / 1000.0)))
        self._hop_samples = max(1, int(round(sample_rate * config.hop_ms / 1000.0)))
        if self._hop_samples > self._frame_samples:
            self._hop_samples = self._frame_samples

        # Pre-computed window + freq axis.
        self._window = np.hanning(self._frame_samples).astype(np.float32)
        # Compensate for window energy loss so RMS-style metrics stay
        # comparable across frame_ms choices.
        self._window_rms = float(np.sqrt(np.mean(self._window.astype(np.float64) ** 2)))
        self._freqs = np.fft.rfftfreq(self._frame_samples, 1.0 / sample_rate).astype(np.float32)

        # Lag range for voiced confidence (autocorrelation peak search).
        self._lag_min = max(2, int(math.floor(sample_rate / config.pitch_max_hz)))
        self._lag_max = max(
            self._lag_min + 1,
            min(self._frame_samples - 1, int(math.ceil(sample_rate / config.pitch_min_hz))),
        )

        # State.
        self._residual: np.ndarray = np.zeros(0, dtype=np.float32)
        self._absolute_time_s: float = 0.0
        self._prev_spectrum: np.ndarray | None = None
        self._onset_history: list[float] = []
        self._frame_count: int = 0

        # Telemetry (accumulates across ``reset()`` per docstring; recreate
        # the detector to fully zero counters).
        self._telemetry = TransientDetectorTelemetry()

        logger.info(
            "TransientDetector initialised: mode=%s, frame=%.1f ms (%d samples), "
            "hop=%.1f ms (%d samples), thresholds {flatness>%.2f, centroid>%.0f Hz, "
            "zcr>%.2f, onset_ratio>%.1f, voiced<%.2f, rms>%.0f dBFS}",
            config.mode,
            config.frame_ms,
            self._frame_samples,
            config.hop_ms,
            self._hop_samples,
            config.flatness_min,
            config.centroid_min_hz,
            config.zcr_min,
            config.onset_ratio,
            config.voiced_max,
            config.rms_min_db,
        )

    # ---- Public API -----------------------------------------------------

    def process(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, list[TransientFeatures]]:
        """Process one audio chunk.

        Returns:
            ``(output_audio, applause_frames)``. ``output_audio`` is the
            input unchanged for ``observe`` mode and the input with
            applause-flagged frames zeroed for ``on`` mode.
            ``applause_frames`` is the (possibly empty) list of
            :class:`TransientFeatures` whose ``is_applause_like`` is
            ``True`` — features for non-applause frames are *not* returned
            to keep the streaming hot path cheap.

        Streaming semantics — **causal, no lookahead**:

        - Feature computation is **continuous** across calls: a residual
          buffer keeps the tail samples that did not form a full frame so
          the next call resumes seamlessly. ``telemetry.frames_processed``
          and ``telemetry.applause_frames`` therefore match between
          ``process(full_audio)`` and a chunked feed of the same audio.

        - In ``"on"`` mode masking is **only applied to the samples that
          belong to the current chunk**. Frames that start in the residual
          (i.e. inside audio already returned to the caller in a previous
          call) can no longer be muted retroactively. The output of a
          chunked feed therefore leaks slightly more energy than the
          equivalent single-chunk feed; the chunked output is a
          best-effort upper bound, not a bit-exact reconstruction.

        Adding a 1-frame lookahead delay would close the gap at the cost
        of +32 ms latency; that enhancement is tracked separately and
        intentionally out of scope for the PR-B initial deliverable.
        """
        if self.config.mode == "off":  # pragma: no cover - caller should pass None
            return audio, []

        audio_f32 = np.asarray(audio, dtype=np.float32)
        self._telemetry.audio_chunks_processed += 1

        # Combine with residual so features stay continuous across calls.
        if self._residual.size > 0:
            combined = np.concatenate([self._residual, audio_f32])
        else:
            combined = audio_f32

        # ``audio_f32`` starts at this index inside ``combined`` — used to
        # translate frame positions into output-chunk coordinates when
        # masking for ``on`` mode.
        chunk_offset = combined.size - audio_f32.size

        output: np.ndarray
        if self.config.mode == "on":
            output = audio_f32.copy()
        else:
            output = audio_f32

        applause_frames: list[TransientFeatures] = []
        pos = 0
        sr = self._sample_rate

        while pos + self._frame_samples <= combined.size:
            frame = combined[pos : pos + self._frame_samples]
            absolute_t = (
                self._absolute_time_s + (pos - chunk_offset) / sr
            )

            features = self._compute_features(frame, absolute_t)
            self._telemetry.frames_processed += 1
            self._update_pass_counters(features)

            if features.is_applause_like:
                self._telemetry.applause_frames += 1
                applause_frames.append(features)
                if self.config.mode == "on":
                    mask_start = max(0, pos - chunk_offset)
                    mask_end = min(
                        audio_f32.size, pos + self._frame_samples - chunk_offset
                    )
                    if mask_end > mask_start:
                        output[mask_start:mask_end] = 0.0

            pos += self._hop_samples

        # Keep the tail that did not fit a full frame so the next call can
        # resume seamlessly.
        self._residual = (
            combined[pos:].copy() if pos < combined.size else np.zeros(0, dtype=np.float32)
        )
        self._absolute_time_s += audio_f32.size / sr

        return output, applause_frames

    def reset(self) -> None:
        """Drop streaming state. Telemetry counters are preserved.

        Call this at the start of every fresh audio source so frame
        positions and the onset baseline restart from zero. Counters keep
        accumulating because they are used for end-of-run telemetry and
        sensitivity analysis; reconstruct the detector instead if you
        want a fully clean slate.
        """
        self._residual = np.zeros(0, dtype=np.float32)
        self._absolute_time_s = 0.0
        self._prev_spectrum = None
        self._onset_history.clear()
        self._frame_count = 0

    @property
    def telemetry(self) -> TransientDetectorTelemetry:
        """Cumulative counter snapshot (safe to read at any time)."""
        return self._telemetry.snapshot()

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def hop_samples(self) -> int:
        return self._hop_samples

    # ---- Feature computation -------------------------------------------

    def _compute_features(
        self, frame: np.ndarray, start_time_s: float
    ) -> TransientFeatures:
        """Compute the six per-frame DSP features and the AND decision."""
        sr = self._sample_rate
        frame_f32 = frame.astype(np.float32, copy=False)

        # Time-domain features.
        rms = float(np.sqrt(np.mean(frame_f32.astype(np.float64) ** 2)))
        rms_db = 20.0 * math.log10(rms) if rms > 0.0 else -math.inf

        # Sign-change density. ``np.diff(np.sign(x))`` returns ``±2`` per
        # crossing; halving and normalising by frame length gives the
        # per-sample crossing rate.
        zcr = float(
            np.count_nonzero(np.diff(np.signbit(frame_f32))) / max(1, frame_f32.size - 1)
        )

        # Frequency-domain features (windowed FFT).
        windowed = frame_f32 * self._window
        spectrum = np.abs(np.fft.rfft(windowed)).astype(np.float32)
        power = spectrum * spectrum

        spectral_flatness = self._spectral_flatness(power)
        spectral_centroid_hz = self._spectral_centroid_hz(spectrum)

        # Onset strength via positive spectral flux.
        onset_strength = self._onset_strength(spectrum)
        onset_strength_baseline = self._onset_baseline()

        # Voiced confidence via normalised autocorrelation peak.
        voiced_ratio = self._voiced_confidence(frame_f32)

        # Track onset history and prev spectrum AFTER deriving features so
        # the baseline used for this frame's decision is the rolling
        # window of *previous* frames.
        self._onset_history.append(onset_strength)
        if len(self._onset_history) > self.config.onset_baseline_window_frames:
            self._onset_history.pop(0)
        self._prev_spectrum = spectrum
        self._frame_count += 1

        # Decision (warmup frames always reject).
        if self._frame_count <= self.config.onset_baseline_warmup_frames:
            is_applause = False
        else:
            is_applause = (
                rms_db > self.config.rms_min_db
                and spectral_flatness > self.config.flatness_min
                and spectral_centroid_hz > self.config.centroid_min_hz
                and zcr > self.config.zcr_min
                and onset_strength
                > onset_strength_baseline * self.config.onset_ratio
                and voiced_ratio < self.config.voiced_max
            )

        return TransientFeatures(
            start_time_s=start_time_s,
            end_time_s=start_time_s + self._frame_samples / sr,
            spectral_flatness=spectral_flatness,
            spectral_centroid_hz=spectral_centroid_hz,
            zero_crossing_rate=zcr,
            onset_strength=onset_strength,
            onset_strength_baseline=onset_strength_baseline,
            voiced_ratio=voiced_ratio,
            rms_db=rms_db,
            is_applause_like=is_applause,
        )

    @staticmethod
    def _spectral_flatness(power_spectrum: np.ndarray) -> float:
        """Wiener entropy on the *power* spectrum; range ``[0, 1]``."""
        if power_spectrum.size == 0:
            return 0.0
        eps = 1e-12
        # Use log-space mean to avoid underflow for very low power.
        log_power = np.log(power_spectrum + eps)
        geom_mean = float(np.exp(np.mean(log_power)))
        arith_mean = float(np.mean(power_spectrum) + eps)
        return geom_mean / arith_mean

    def _spectral_centroid_hz(self, magnitude_spectrum: np.ndarray) -> float:
        """Magnitude-weighted mean frequency."""
        total = float(np.sum(magnitude_spectrum))
        if total <= 0.0:
            return 0.0
        return float(np.sum(self._freqs * magnitude_spectrum) / total)

    def _onset_strength(self, spectrum: np.ndarray) -> float:
        """Positive spectral flux against the previous frame's spectrum."""
        if self._prev_spectrum is None:
            return 0.0
        diff = spectrum - self._prev_spectrum
        return float(np.sum(np.maximum(diff, 0.0)))

    def _onset_baseline(self) -> float:
        """Median of the rolling onset-strength history (excluding now)."""
        if not self._onset_history:
            return 0.0
        return float(np.median(np.asarray(self._onset_history, dtype=np.float64)))

    def _voiced_confidence(self, frame: np.ndarray) -> float:
        """Normalised autocorrelation peak inside the pitch band.

        Returns the ratio ``r[lag_peak] / r[0]`` where ``lag_peak`` is the
        argmax of the autocorrelation inside the configured pitch lag
        band. Falls back to 0 if the frame is silent or the band is empty.
        """
        if frame.size < 4:
            return 0.0
        # Mean-centre to suppress DC.
        x = frame.astype(np.float64) - float(np.mean(frame))
        r0 = float(np.dot(x, x))
        if r0 <= 0.0:
            return 0.0
        lag_lo = self._lag_min
        lag_hi = min(self._lag_max, x.size - 1)
        if lag_hi <= lag_lo:
            return 0.0
        # Direct dot-product autocorrelation across the lag window. Sample
        # counts here are ~256-512 so the O(N^2) cost is < 1 ms; keeping
        # the implementation numpy-only avoids a scipy.signal dep.
        best = 0.0
        for lag in range(lag_lo, lag_hi + 1):
            r_lag = float(np.dot(x[:-lag], x[lag:]))
            if r_lag > best:
                best = r_lag
        return best / r0

    def _update_pass_counters(self, features: TransientFeatures) -> None:
        """Increment per-feature pass counters for sensitivity analysis."""
        c = self.config
        if features.rms_db > c.rms_min_db:
            self._telemetry.pass_rms += 1
        if features.spectral_flatness > c.flatness_min:
            self._telemetry.pass_flatness += 1
        if features.spectral_centroid_hz > c.centroid_min_hz:
            self._telemetry.pass_centroid += 1
        if features.zero_crossing_rate > c.zcr_min:
            self._telemetry.pass_zcr += 1
        if features.onset_strength > features.onset_strength_baseline * c.onset_ratio:
            self._telemetry.pass_onset += 1
        if features.voiced_ratio < c.voiced_max:
            self._telemetry.pass_voiced += 1
