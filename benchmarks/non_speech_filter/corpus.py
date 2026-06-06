"""Synthetic audio corpus for non-speech filter evaluation (Issue #295 PR-0).

Provides deterministic numpy-only audio synthesis for:
- negative set: applause, keyboard, door, cough, music, silence — should be
  rejected by the multi-layered defense.
- positive set: speech proxies (normal, short utterance, post-applause speech)
  — must be preserved.

All synthesizers return float32 mono audio at 16 kHz, seeded for
reproducibility. Acoustic fidelity is intentionally limited; real audio
fixtures can be supplied via ``LIVECAP_NON_SPEECH_CORPUS_DIR`` (see conftest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

SAMPLE_RATE: int = 16000


@dataclass(frozen=True)
class CorpusItem:
    """One labeled audio clip used by the evaluation harness.

    Attributes:
        label: Identifier used in metric breakdowns.
        kind: ``"negative"`` (should NOT trigger ASR) or
              ``"positive"`` (must trigger ASR).
        is_short_utterance: True if this is a short response (はい / OK
              etc.); contributes to the dedicated short-utterance recall.
        audio: float32 mono PCM at 16 kHz, range ~[-1, 1].
    """

    label: str
    kind: Literal["negative", "positive"]
    is_short_utterance: bool
    audio: np.ndarray


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -math.inf
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms)


def _scale_to_rms_db(audio: np.ndarray, target_db: float) -> np.ndarray:
    """Rescale ``audio`` so its RMS matches ``target_db`` (dBFS)."""
    current_db = _rms_db(audio)
    if not math.isfinite(current_db):
        return audio
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def _single_clap(
    duration_ms: float = 80.0,
    rms_db: float = -18.0,
    decay_tau: float = 5.0,
    bandpass_hz: tuple[float, float] = (200.0, 7800.0),
    sample_rate: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """One applause-like transient: bandpassed noise with exponential decay."""
    n = max(8, int(sample_rate * duration_ms / 1000.0))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    envelope = np.exp(-np.linspace(0.0, decay_tau, n)).astype(np.float32)
    clap = noise * envelope
    lo, hi = bandpass_hz
    nyq = sample_rate * 0.5
    hi = min(hi, nyq - 50.0)
    if lo < hi:
        sos = scipy_signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        clap = scipy_signal.sosfilt(sos, clap).astype(np.float32)
    return _scale_to_rms_db(clap, rms_db)


def _synthesize_silence_amplified(
    duration_ms: float = 1000.0,
    rms_db: float = -50.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 7,
) -> np.ndarray:
    """Low-amplitude white noise — mirrors the #292 silence-amplified fixture."""
    n = int(sample_rate * duration_ms / 1000.0)
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(n).astype(np.float32)
    return _scale_to_rms_db(audio, rms_db)


def _synthesize_applause_single(
    duration_ms: float = 800.0,
    rms_db: float = -16.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 101,
) -> np.ndarray:
    """One sharp clap centered in a short silence window."""
    n = int(sample_rate * duration_ms / 1000.0)
    audio = np.zeros(n, dtype=np.float32)
    clap = _single_clap(duration_ms=80.0, rms_db=rms_db, sample_rate=sample_rate, seed=seed)
    start = max(0, n // 4)
    end = min(n, start + len(clap))
    audio[start:end] = clap[: end - start]
    return audio


def _synthesize_applause_burst(
    n_claps: int = 7,
    inter_clap_ms: float = 150.0,
    rms_db: float = -14.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 202,
) -> np.ndarray:
    """A short rush of claps over ~1 second."""
    gap = int(sample_rate * inter_clap_ms / 1000.0)
    parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for i in range(n_claps):
        clap = _single_clap(
            duration_ms=80.0 + float(rng.uniform(-10, 10)),
            rms_db=rms_db + float(rng.uniform(-3, 3)),
            sample_rate=sample_rate,
            seed=seed + i,
        )
        parts.append(clap)
        parts.append(np.zeros(gap, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _synthesize_applause_distant(
    duration_ms: float = 1000.0,
    rms_db: float = -28.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 303,
) -> np.ndarray:
    """Distant / reverberant applause: dense overlapping low-amp claps."""
    n = int(sample_rate * duration_ms / 1000.0)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for i in range(40):
        clap = _single_clap(
            duration_ms=120.0,
            rms_db=rms_db,
            bandpass_hz=(400.0, 4500.0),  # high-freq attenuation
            sample_rate=sample_rate,
            seed=seed + i,
        )
        start = int(rng.uniform(0, max(1, n - len(clap))))
        end = start + len(clap)
        audio[start:end] += clap[: end - start]
    audio = np.tanh(audio).astype(np.float32)
    return _scale_to_rms_db(audio, rms_db)


def _synthesize_keyboard_tap_train(
    n_taps: int = 8,
    inter_tap_ms: float = 100.0,
    rms_db: float = -18.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 404,
) -> np.ndarray:
    """Series of sharp clicks with short duration — typing-like."""
    gap = int(sample_rate * inter_tap_ms / 1000.0)
    parts: list[np.ndarray] = []
    for i in range(n_taps):
        tap = _single_clap(
            duration_ms=20.0,
            rms_db=rms_db,
            decay_tau=8.0,
            bandpass_hz=(800.0, 6000.0),
            sample_rate=sample_rate,
            seed=seed + i,
        )
        parts.append(tap)
        parts.append(np.zeros(gap, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _synthesize_door_close(
    duration_ms: float = 600.0,
    rms_db: float = -16.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 505,
) -> np.ndarray:
    """Single low-frequency thump."""
    n = int(sample_rate * duration_ms / 1000.0)
    audio = np.zeros(n, dtype=np.float32)
    thump_n = int(sample_rate * 0.18)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(thump_n).astype(np.float32)
    envelope = np.exp(-np.linspace(0.0, 4.0, thump_n)).astype(np.float32)
    thump = noise * envelope
    nyq = sample_rate * 0.5
    sos = scipy_signal.butter(4, [50.0 / nyq, 400.0 / nyq], btype="band", output="sos")
    thump = scipy_signal.sosfilt(sos, thump).astype(np.float32)
    audio[: len(thump)] = thump
    return _scale_to_rms_db(audio, rms_db)


def _synthesize_cough(
    duration_ms: float = 500.0,
    rms_db: float = -18.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 606,
) -> np.ndarray:
    """Narrow band-limited noise burst (~ cough-like)."""
    n = int(sample_rate * duration_ms / 1000.0)
    audio = np.zeros(n, dtype=np.float32)
    burst_n = int(sample_rate * 0.18)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(burst_n).astype(np.float32)
    envelope = np.hanning(burst_n).astype(np.float32)
    burst = noise * envelope
    nyq = sample_rate * 0.5
    sos = scipy_signal.butter(4, [350.0 / nyq, 3500.0 / nyq], btype="band", output="sos")
    burst = scipy_signal.sosfilt(sos, burst).astype(np.float32)
    audio[: len(burst)] = burst
    return _scale_to_rms_db(audio, rms_db)


def _synthesize_music_chord(
    duration_ms: float = 1500.0,
    rms_db: float = -20.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Sustained harmonic mixture (CMajor chord) — music-like."""
    n = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(n).astype(np.float32) / sample_rate
    fundamentals = (261.63, 329.63, 392.00)  # C major
    audio = np.zeros(n, dtype=np.float32)
    for f0 in fundamentals:
        for k in range(1, 5):
            audio += (1.0 / k) * np.sin(2.0 * math.pi * f0 * k * t).astype(np.float32)
    envelope = np.minimum(t / 0.05, np.ones_like(t)).astype(np.float32)  # 50ms fade-in
    audio = audio * envelope
    return _scale_to_rms_db(audio, rms_db)


def _synthesize_speech_proxy(
    duration_ms: float = 1200.0,
    rms_db: float = -20.0,
    f0_hz: float = 150.0,
    formants_hz: tuple[float, ...] = (700.0, 1100.0, 2400.0),
    formant_bw_hz: float = 250.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 909,
) -> np.ndarray:
    """Speech-like signal via source-filter model with realistic modulation.

    The signal carries pitch jitter, syllable-rate amplitude modulation,
    formant drift, and aspiration noise so that data-driven VADs (Silero
    especially) recognize it as speech rather than a static tone. Per-frame
    autocorrelation still peaks near ``f0_hz``, so the Layer-1 voiced_ratio
    heuristic also classifies the signal as voiced.
    """
    n = int(sample_rate * duration_ms / 1000.0)
    rng = np.random.default_rng(seed)
    t = np.arange(n).astype(np.float32) / sample_rate

    # Pitch jitter: 3 % vibrato at ~5 Hz + small flicker.
    vibrato = (
        1.0
        + 0.03 * np.sin(2.0 * math.pi * 5.0 * t).astype(np.float32)
        + 0.005 * rng.standard_normal(n).astype(np.float32)
    )
    instantaneous_f0 = (f0_hz * vibrato).astype(np.float32)

    # Phase = cumulative integral of f0 -> proper FM (not amplitude vibrato).
    phase = 2.0 * math.pi * np.cumsum(instantaneous_f0) / sample_rate

    # Harmonic source (band-limited pulse approximation).
    source = np.zeros(n, dtype=np.float32)
    n_harmonics = min(24, int((sample_rate * 0.5 - 100.0) / max(1.0, f0_hz)))
    for k in range(1, n_harmonics + 1):
        source += (1.0 / k) * np.sin(k * phase).astype(np.float32)

    # Aspiration noise (5 % of source energy).
    noise = 0.05 * rng.standard_normal(n).astype(np.float32)
    excitation = source + noise

    # Formant chain via bandpass + drifting center freq.
    nyq = sample_rate * 0.5
    audio = np.zeros(n, dtype=np.float32)
    for fmt in formants_hz:
        drift_hz = fmt * (1.0 + 0.03 * np.sin(2.0 * math.pi * 2.5 * t).astype(np.float32))
        center = float(np.clip(drift_hz.mean(), 100.0, nyq - 100.0))
        lo = max(60.0, center - formant_bw_hz)
        hi = min(nyq - 50.0, center + formant_bw_hz)
        if lo >= hi:
            continue
        sos = scipy_signal.butter(2, [lo / nyq, hi / nyq], btype="band", output="sos")
        audio += scipy_signal.sosfilt(sos, excitation).astype(np.float32)

    # Syllable-rate amplitude envelope (~4 Hz).
    am = 0.5 + 0.5 * np.sin(2.0 * math.pi * 4.0 * t).astype(np.float32) ** 2
    audio = audio * am

    # Soft attack/release.
    fade = int(sample_rate * 0.03)
    if fade > 0 and len(audio) > 2 * fade:
        audio[:fade] *= np.linspace(0.0, 1.0, fade).astype(np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade).astype(np.float32)

    return _scale_to_rms_db(audio, rms_db)


def _synthesize_short_utterance(
    duration_ms: float = 220.0,
    rms_db: float = -20.0,
    f0_hz: float = 160.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Short voiced response (~ 200 ms) — analog for はい / OK / うん.

    Long enough to be classified as speech but short enough to be at risk of
    being dropped by aggressive ``min_speech_ms`` settings.
    """
    return _synthesize_speech_proxy(
        duration_ms=duration_ms,
        rms_db=rms_db,
        f0_hz=f0_hz,
        formants_hz=(750.0, 1200.0),
        sample_rate=sample_rate,
    )


def _synthesize_post_applause_speech(
    applause_ms: float = 300.0,
    gap_ms: float = 200.0,
    speech_ms: float = 900.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 707,
) -> np.ndarray:
    """Applause burst followed by speech — must preserve the speech portion."""
    applause = _synthesize_applause_burst(
        n_claps=3,
        inter_clap_ms=100.0,
        rms_db=-15.0,
        sample_rate=sample_rate,
        seed=seed,
    )
    target_applause = int(sample_rate * applause_ms / 1000.0)
    if len(applause) > target_applause:
        applause = applause[:target_applause]
    gap = np.zeros(int(sample_rate * gap_ms / 1000.0), dtype=np.float32)
    speech = _synthesize_speech_proxy(duration_ms=speech_ms, sample_rate=sample_rate)
    return np.concatenate([applause, gap, speech]).astype(np.float32)


def _synthesize_overlapping_applause_speech(
    duration_ms: float = 1200.0,
    speech_rms_db: float = -22.0,
    applause_rms_db: float = -22.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 808,
) -> np.ndarray:
    """Applause and speech mixed at similar levels — recall is hard."""
    speech = _synthesize_speech_proxy(
        duration_ms=duration_ms,
        rms_db=speech_rms_db,
        sample_rate=sample_rate,
    )
    n = len(speech)
    applause = _synthesize_applause_burst(
        n_claps=4,
        inter_clap_ms=200.0,
        rms_db=applause_rms_db,
        sample_rate=sample_rate,
        seed=seed,
    )
    if len(applause) < n:
        applause = np.concatenate(
            [applause, np.zeros(n - len(applause), dtype=np.float32)]
        )
    else:
        applause = applause[:n]
    return (speech + applause).astype(np.float32)


def build_synthetic_corpus(sample_rate: int = SAMPLE_RATE) -> list[CorpusItem]:
    """Construct the canonical synthetic evaluation corpus.

    The list is the source of truth for baseline metric labels. Callers
    typically split into negative / positive subsets via ``item.kind``.
    """
    items: list[CorpusItem] = []

    # Negative set (8 items)
    items.append(
        CorpusItem(
            label="applause_single",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_applause_single(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="applause_burst",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_applause_burst(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="applause_distant",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_applause_distant(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="keyboard_taps",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_keyboard_tap_train(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="door_close",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_door_close(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="cough",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_cough(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="music_chord",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_music_chord(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="silence_amplified",
            kind="negative",
            is_short_utterance=False,
            audio=_synthesize_silence_amplified(sample_rate=sample_rate),
        )
    )

    # Positive set (5 items, including 2 short utterances)
    items.append(
        CorpusItem(
            label="normal_speech",
            kind="positive",
            is_short_utterance=False,
            audio=_synthesize_speech_proxy(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="short_utterance_hai",
            kind="positive",
            is_short_utterance=True,
            audio=_synthesize_short_utterance(
                duration_ms=220.0, f0_hz=170.0, sample_rate=sample_rate
            ),
        )
    )
    items.append(
        CorpusItem(
            label="short_utterance_ok",
            kind="positive",
            is_short_utterance=True,
            audio=_synthesize_short_utterance(
                duration_ms=260.0, f0_hz=140.0, sample_rate=sample_rate
            ),
        )
    )
    items.append(
        CorpusItem(
            label="post_applause_speech",
            kind="positive",
            is_short_utterance=False,
            audio=_synthesize_post_applause_speech(sample_rate=sample_rate),
        )
    )
    items.append(
        CorpusItem(
            label="overlapping_applause_speech",
            kind="positive",
            is_short_utterance=False,
            audio=_synthesize_overlapping_applause_speech(sample_rate=sample_rate),
        )
    )

    return items
