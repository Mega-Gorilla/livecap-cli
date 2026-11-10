"""VAD 最適化の目的関数が本番 CLI と同じ後段を評価すること (#470)。

``objective="production"`` (既定) は CLI / ``StreamTranscriber`` と同じ順で
EnergyGate (ASR 前) → transcribe → confidence filter (``on``) を通す。本番で字幕に出ない
segment (低エネルギー / filter reject) を罰しないため。例: Qwen3-ASR が非音声区間に出す
system prompt 漏れは filter が落とすが、後段なしの目的関数は WER 0.115 と評価していた
(本番経路では 0.040)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.common.datasets import AudioFile
from benchmarks.optimization.objective import OBJECTIVES, VADObjective
from benchmarks.optimization.vad_optimizer import OptimizationResult, VADOptimizer
from livecap_cli.engines.base_engine import EngineConfidence, TranscriptionResult

SR = 16000


class _ScriptedEngine:
    """segment の振幅で返す結果を変える fake engine (Qwen3-ASR の avg_logprob を模す)。"""

    def __init__(self) -> None:
        self.calls = 0

    def get_engine_name(self) -> str:
        return "Qwen3-ASR 1.7B"  # filter の engine 別 threshold (qwen3-asr: -0.42) を引く名前

    def get_required_sample_rate(self) -> int:
        return SR

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        self.calls += 1
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.2:  # 発話
            return TranscriptionResult("hello world", 1.0, EngineConfidence(avg_logprob=-0.1))
        # 非音声区間の system prompt 漏れ (低 confidence)
        return TranscriptionResult("you are speech recognition", 0.1, EngineConfidence(avg_logprob=-1.5))


def _audio_file(audio: np.ndarray) -> AudioFile:
    # 遅延ロードの値を先に埋めておく (実ファイルは読まない)
    return AudioFile(
        path=Path("utt.wav"),
        transcript_path=Path("utt.txt"),
        language="en",
        _audio=audio,
        _sample_rate=SR,
        _transcript="hello world",
    )


def _objective(objective: str, engine) -> VADObjective:
    return VADObjective(vad_type="silero", language="en", engine=engine, dataset=[], objective=objective)


def _segment(amplitude: float, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_production_is_the_default():
    assert VADObjective(vad_type="silero", language="en", engine=_ScriptedEngine(), dataset=[]).objective == "production"
    assert OBJECTIVES == ("production", "raw")


def test_production_drops_what_the_confidence_filter_rejects():
    engine = _ScriptedEngine()
    speech = np.concatenate([_segment(0.5), _segment(0.05)])  # 発話 + 低 confidence の漏れ (EnergyGate は通る音量)
    file = _audio_file(speech)
    segments = [(0.0, 0.5), (0.5, 1.0)]

    assert _objective("production", engine)._transcribe_segments(segments, file) == "hello world"
    assert _objective("raw", engine)._transcribe_segments(segments, file) == "hello world you are speech recognition"


def test_production_skips_asr_for_low_energy_segments():
    """EnergyGate (-45 dBFS) は ASR 前に落とす — 本番同様、engine を呼ばない。"""
    quiet = _segment(1e-4)  # ≈ -80 dBFS
    file = _audio_file(quiet)

    production_engine = _ScriptedEngine()
    assert _objective("production", production_engine)._transcribe_segments([(0.0, 0.5)], file) == ""
    assert production_engine.calls == 0

    raw_engine = _ScriptedEngine()
    _objective("raw", raw_engine)._transcribe_segments([(0.0, 0.5)], file)
    assert raw_engine.calls == 1


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="Unknown objective"):
        _objective("nope", _ScriptedEngine())
    with pytest.raises(ValueError, match="Unknown objective"):
        VADOptimizer(vad_type="silero", language="en", engine_id="qwen3asr_large", device="cpu", objective="nope")


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_exported_preset_records_the_objective(objective):
    result = OptimizationResult(
        vad_type="silero",
        language="en",
        best_params={
            "vad_config_threshold": 0.5,
            "vad_config_neg_threshold": 0.35,
            "vad_config_min_speech_ms": 250,
            "vad_config_min_silence_ms": 100,
            "vad_config_speech_pad_ms": 100,
        },
        best_score=0.04,
        n_trials=2,
        engine_id="qwen3asr_large",
        objective=objective,
    )

    assert result.build_preset_dict()["metadata"]["objective"] == objective
    assert result.to_dict()["objective"] == objective
