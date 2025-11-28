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

from .param_spaces import suggest_params

if TYPE_CHECKING:
    from engines.base_engine import TranscriptionEngine

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.vad_type = vad_type
        self.language = language
        self.engine = engine
        self.dataset = dataset

        # Determine metric based on language
        self._is_japanese = language == "ja"

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

            # 2. Resolve VAD ID for factory
            # WebRTC optimization uses "webrtc" as vad_type, but factory
            # expects "webrtc_mode{N}". Map using the suggested mode.
            vad_id = self._resolve_vad_id(backend_params)

            # 3. Create VAD with suggested parameters
            vad = create_vad(vad_id, backend_params, vad_config)

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

    def _resolve_vad_id(self, backend_params: dict) -> str:
        """Resolve optimization vad_type to factory vad_id.

        Maps generic optimization types to specific factory IDs:
        - "webrtc" -> "webrtc_mode{N}" based on suggested mode parameter

        Args:
            backend_params: Suggested backend parameters

        Returns:
            VAD ID suitable for create_vad()
        """
        if self.vad_type == "webrtc":
            # WebRTC optimization suggests mode as a parameter
            # Map to specific factory ID: webrtc_mode0, webrtc_mode1, etc.
            mode = backend_params.get("mode", 3)
            return f"webrtc_mode{mode}"

        # Other VAD types use vad_type directly
        return self.vad_type

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

            # Transcribe
            try:
                text, _ = self.engine.transcribe(segment_audio, current_sr)
                if text:
                    transcripts.append(text)
            except Exception as e:
                logger.warning(f"Transcription error: {e}")
                continue

        return "".join(transcripts) if self._is_japanese else " ".join(transcripts)

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
