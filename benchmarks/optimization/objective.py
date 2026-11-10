"""Objective function for VAD parameter optimization.

Defines the objective function that Optuna optimizes to minimize CER/WER.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

import numpy as np
import optuna

from benchmarks.common.datasets import AudioFile
from benchmarks.common.metrics import calculate_cer, calculate_wer
from benchmarks.vad.factory import create_vad
from livecap_cli.audio import (
    DEFAULT_ENGINE_ENERGY_FRAME_MS,
    DEFAULT_ENGINE_ENERGY_METRIC,
    DEFAULT_ENGINE_MIN_RMS_DBFS,
    should_drop_low_energy,
)
from livecap_cli.transcription.confidence_filter import FilterConfig, should_reject

from .param_spaces import suggest_params

if TYPE_CHECKING:
    from livecap_cli import TranscriptionEngine

logger = logging.getLogger(__name__)

#: 目的関数が評価する ASR 出力 (#470)。
#:
#: - ``"production"`` (既定): CLI / ``StreamTranscriber`` と同じ後段 — EnergyGate
#:   (ASR 前、CLI 既定値) → transcribe → confidence filter (``--confidence-filter`` 既定の
#:   ``on``)。**本番で消える segment を罰しない** (例: Qwen3-ASR が非音声区間に出す system
#:   prompt 漏れは filter が落とす。tenvad/en/qwen3asr の preset で raw WER 0.115 → 0.040)
#: - ``"raw"``: 後段なしの生の ASR 出力 (2026-11 以前の preset はこの条件で作られている)
OBJECTIVES = ("production", "raw")


def validate_objective(objective: str) -> str:
    """``objective`` が ``OBJECTIVES`` のどれかでなければ ``ValueError``。"""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective!r} (expected one of {OBJECTIVES})")
    return objective


class VADObjective:
    """Objective function for VAD parameter optimization.

    This class is called by Optuna to evaluate a set of VAD parameters.
    It creates a VAD with the suggested parameters, processes the dataset,
    and returns the average CER (for Japanese) or WER (for English).

    Args:
        vad_type: VAD type to optimize ("silero", "tenvad", "webrtc")
        language: Language code ("ja" or "en")
        engine: ASR engine for transcription
        dataset: List of AudioFile objects to evaluate
        objective: ``"production"`` (CLI と同じ EnergyGate + confidence filter を通す、既定) or
            ``"raw"`` (生の ASR 出力)。``OBJECTIVES`` 参照

    Example:
        objective = VADObjective(
            vad_type="silero",
            language="ja",
            engine=engine,
            dataset=audio_files,
        )
        study.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        vad_type: str,
        language: str,
        engine: "TranscriptionEngine",
        dataset: list[AudioFile],
        objective: str = "production",
    ) -> None:
        self.vad_type = vad_type
        self.language = language
        self.engine = engine
        self.dataset = dataset
        self.objective = validate_objective(objective)

        # Determine metric based on language
        self._is_japanese = language == "ja"

        # 本番と同じ後段 (objective="production")。filter の engine 別 threshold は
        # engine 表示名で引くので、CLI と同じく get_engine_name() を渡す
        self._filter_config = FilterConfig(mode="on")
        self._engine_name = engine.get_engine_name() if objective == "production" else ""

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a set of VAD parameters.

        Args:
            trial: Optuna trial with suggested parameters

        Returns:
            Average CER (Japanese) or WER (English) across the dataset.
            Returns 1.0 for failed trials (as penalty).
        """
        try:
            # 1. Suggest parameters
            backend_params, vad_config = suggest_params(trial, self.vad_type)

            logger.debug(
                f"Trial {trial.number}: backend_params={backend_params}, "
                f"vad_config={vad_config}"
            )

            # 2. Create VAD with suggested parameters
            # backend_params includes mode for WebRTC, threshold for others
            vad = create_vad(self.vad_type, backend_params, vad_config)

            # 3. Evaluate on dataset
            scores: list[float] = []

            for audio_file in self.dataset:
                score = self._evaluate_file(vad, audio_file)
                scores.append(score)

            # 4. Return average score
            avg_score = statistics.mean(scores)

            logger.debug(
                f"Trial {trial.number}: avg_score={avg_score:.4f} "
                f"(n={len(scores)})"
            )

            return avg_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 1.0  # Maximum penalty for failed trials

    def _evaluate_file(self, vad, audio_file: AudioFile) -> float:
        """Evaluate a single audio file.

        Args:
            vad: VAD backend instance
            audio_file: Audio file to evaluate

        Returns:
            CER or WER score (0.0-1.0+, lower is better)
        """
        # Process audio with VAD
        segments = vad.process_audio(audio_file.audio, audio_file.sample_rate)

        # No segments detected = complete failure
        if not segments:
            logger.debug(f"No segments detected for: {audio_file.stem}")
            return 1.0

        # Transcribe segments and concatenate
        transcript = self._transcribe_segments(segments, audio_file)

        # Calculate metric
        if self._is_japanese:
            return calculate_cer(
                audio_file.transcript,
                transcript,
                lang="ja",
                normalize=True,
            )
        else:
            return calculate_wer(
                audio_file.transcript,
                transcript,
                lang="en",
                normalize=True,
            )

    def _transcribe_segments(
        self,
        segments: list[tuple[float, float]],
        audio_file: AudioFile,
    ) -> str:
        """Transcribe detected segments using ASR engine.

        Args:
            segments: List of (start_time, end_time) tuples in seconds
            audio_file: Audio file being processed

        Returns:
            Concatenated transcript from all segments
        """
        transcripts: list[str] = []
        sample_rate = audio_file.sample_rate
        audio = audio_file.audio

        for start_time, end_time in segments:
            # Convert time to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Clamp to audio boundaries
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            # Extract segment audio
            segment_audio = audio[start_sample:end_sample]

            # Resample if needed
            engine_sr = self.engine.get_required_sample_rate()
            if sample_rate != engine_sr:
                segment_audio = self._resample(segment_audio, sample_rate, engine_sr)
                current_sr = engine_sr
            else:
                current_sr = sample_rate

            text = self._transcribe_segment(segment_audio, current_sr)
            if text:
                transcripts.append(text)

        return "".join(transcripts) if self._is_japanese else " ".join(transcripts)

    def _transcribe_segment(self, segment_audio: np.ndarray, sample_rate: int) -> str:
        """1 segment を文字起こしする。``objective="production"`` では CLI と同じ後段を通す。

        処理順は CLI file mode / realtime と同じ: EnergyGate (ASR 前) → transcribe →
        confidence filter (ASR 後、``on`` の reject 判定)。drop / reject された segment は
        空文字 (本番でも字幕に出ない)。
        """
        if self.objective == "production":
            drop, _energy = should_drop_low_energy(
                segment_audio,
                sample_rate,
                threshold_dbfs=DEFAULT_ENGINE_MIN_RMS_DBFS,
                metric=DEFAULT_ENGINE_ENERGY_METRIC,
                frame_ms=DEFAULT_ENGINE_ENERGY_FRAME_MS,
            )
            if drop:
                return ""
        try:
            result = self.engine.transcribe(segment_audio, sample_rate)
        except Exception as e:
            logger.warning(f"Transcription error: {e}")
            return ""
        if self.objective == "production":
            rejected, _reason = should_reject(result, self._filter_config, engine_name=self._engine_name)
            if rejected:
                return ""
        return result.text or ""

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio using scipy's polyphase filter."""
        from math import gcd
        from scipy import signal

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        resampled = signal.resample_poly(audio, up, down)
        return resampled.astype(np.float32)


__all__ = ["VADObjective"]
