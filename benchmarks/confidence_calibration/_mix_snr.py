"""SNR-based audio mixing helper for Layer 3 noisy_speech corpus (Issue #338).

Given clean speech and noise arrays, mixes them at a target SNR (dB) with
RMS-based scaling. Handles length matching (tile / truncate) and clipping.

Design (Plan D4):

* Numpy-only — no lhotse / torchaudio dependency. RMS power computation is
  ``np.mean(x**2)``, closed-form scale factor from target dB.
* Length matching: shorter of the two is tiled or truncated to match the
  speech length (speech is the reference; we don't extend it).
* Clip detection: if ``|mix|.max() > 1.0``, log warn and renormalize by
  scaling the whole mix down to peak 0.95 (SNR ratio preserved because
  we scale speech and mixed noise by the same factor).
* Deterministic: no random seed involved.

Used by ``gen_mixed_noisy_speech.py`` to produce Layer 3 corpus entries
(``label=noisy_speech``) for the ``noisy_speech_frr by SNR`` Pareto gate
in Issue #334 PR-4.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Peak target after clip renormalization (leaves headroom below 1.0)
_CLIP_RENORM_PEAK = 0.95


def _rms_power(x: np.ndarray) -> float:
    """Return RMS power = mean(x^2)。 x が空なら 0.0 を返す。"""
    if x.size == 0:
        return 0.0
    return float(np.mean(x.astype(np.float64) ** 2))


def match_length(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """``source`` の長さを ``reference`` に合わせる (tile or truncate)。

    - source > reference: 先頭から truncate
    - source < reference: tile 後 truncate
    - source == reference: そのまま返却 (copy)
    - source が空: reference と同 shape のゼロを返す
    """
    ref_len = len(reference)
    src_len = len(source)
    if src_len == 0:
        return np.zeros(ref_len, dtype=np.float32)
    if src_len == ref_len:
        return source.astype(np.float32, copy=True)
    if src_len > ref_len:
        return source[:ref_len].astype(np.float32, copy=True)
    # src_len < ref_len: tile
    n_tiles = math.ceil(ref_len / src_len)
    tiled = np.tile(source, n_tiles)[:ref_len]
    return tiled.astype(np.float32, copy=False)


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """``speech`` + scaled ``noise`` の mix を target SNR (dB) で返す。

    Args:
        speech: 1D float array、 speech signal (reference length)。
        noise: 1D float array、 noise signal (length adjusted internally)。
        snr_db: target Signal-to-Noise Ratio in dB。 正の値ほど speech dominant。

    Returns:
        ``float32`` 1D array、 length = len(speech)。 clip されている可能性あり
        (caller で ``check_and_renorm`` を呼んで renormalize することを推奨)。

    Special cases (Plan D4):
        * len(speech) == 0: 空配列を返す
        * noise の RMS power == 0.0 (無音): scaling 不能、 speech を copy して返す
        * speech の RMS power == 0.0: speech をそのまま返す (noise を混ぜても
          SNR = -inf の意味しかないため)
    """
    if len(speech) == 0:
        return np.zeros(0, dtype=np.float32)

    speech = speech.astype(np.float32, copy=False)
    noise_matched = match_length(speech, noise)

    p_speech = _rms_power(speech)
    p_noise = _rms_power(noise_matched)

    if p_noise == 0.0:
        logger.debug("mix_at_snr: noise power is zero, returning speech unchanged")
        return speech.astype(np.float32, copy=True)
    if p_speech == 0.0:
        logger.debug("mix_at_snr: speech power is zero, returning speech (all-zero) unchanged")
        return speech.astype(np.float32, copy=True)

    # 10 * log10(P_speech / (P_noise * scale^2)) = snr_db
    # => scale = sqrt(P_speech / (P_noise * 10^(snr_db/10)))
    scale = math.sqrt(p_speech / (p_noise * (10.0 ** (snr_db / 10.0))))
    mixed = speech + noise_matched.astype(np.float32) * np.float32(scale)
    return mixed.astype(np.float32, copy=False)


def check_and_renorm(
    mixed: np.ndarray,
    peak_target: float = _CLIP_RENORM_PEAK,
) -> tuple[np.ndarray, bool]:
    """``|mixed|.max() > 1.0`` なら peak_target まで scale down。

    Args:
        mixed: mixed audio (float32)。
        peak_target: renormalize 後の目標 peak (0 < value <= 1.0)。

    Returns:
        ``(renormed_audio, was_clipped)`` tuple。 clip 検出時は
        renormed_audio = mixed * (peak_target / peak)、 was_clipped=True。
        SNR ratio は全 sample を同 scale で縮めるため保持される。
    """
    if mixed.size == 0:
        return mixed.astype(np.float32, copy=False), False
    peak = float(np.abs(mixed).max())
    if peak <= 1.0:
        return mixed.astype(np.float32, copy=False), False
    scale = peak_target / peak
    renormed = (mixed * scale).astype(np.float32, copy=False)
    logger.warning(
        "mix clipped (|max|=%.3f), renormalized peak to %.3f (SNR ratio preserved)",
        peak,
        peak_target,
    )
    return renormed, True


def compute_snr_db(speech: np.ndarray, noise_component: np.ndarray) -> Optional[float]:
    """Debug/test 用: 実測 SNR (dB) を back-compute。

    Args:
        speech: 元 speech signal
        noise_component: mixed から speech を引いた成分 = noise_matched * scale

    Returns:
        SNR (dB)、 noise_component の power が 0 なら None (infinite SNR)。
    """
    p_speech = _rms_power(speech)
    p_noise = _rms_power(noise_component)
    if p_noise == 0.0:
        return None
    return 10.0 * math.log10(p_speech / p_noise)
