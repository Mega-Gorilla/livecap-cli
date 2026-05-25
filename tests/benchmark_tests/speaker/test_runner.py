"""Tests for SpeakerBenchmarkRunner using synthetic audio + mock backend.

These tests avoid heavy models (NeMo/SpeechBrain/pyannote), real VAD detection
reliability, and the gitignored conversation data by injecting segments and/or
monkeypatching the audio loading.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.speaker import runner as runner_mod
from benchmarks.speaker.cli import parse_args
from benchmarks.speaker.runner import SpeakerBenchmarkConfig, SpeakerBenchmarkRunner

SR = 16000


def _tone(freq: float, dur_s: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur_s)) / SR
    sig = np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.shape)
    return sig.astype(np.float32)


def _two_speaker_segments(n_each: int = 8) -> list[np.ndarray]:
    """Alternating low/high tone clips simulating two speakers."""
    segs: list[np.ndarray] = []
    for i in range(n_each):
        segs.append(_tone(200.0, seed=i))
        segs.append(_tone(3000.0, seed=100 + i))
    return segs


class TestBenchmarkBackendMock:
    def test_mock_backend_produces_ok_result(self) -> None:
        config = SpeakerBenchmarkConfig(backends=["mock"], device="cpu")
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments()
        run._audio_duration = sum(len(s) for s in run._segments) / SR

        result = run._benchmark_backend("mock")

        assert result.status == "ok"
        assert result.num_segments == len(run._segments)
        assert result.embedding_dim == 32
        assert result.embed_latency_ms_p50 is not None
        assert result.rtf is not None and result.rtf >= 0
        assert result.ram_peak_mb is not None

    def test_mock_separates_two_speakers(self) -> None:
        config = SpeakerBenchmarkConfig(backends=["mock"], device="cpu")
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments()
        run._audio_duration = sum(len(s) for s in run._segments) / SR

        result = run._benchmark_backend("mock")
        # Two clearly distinct tone groups -> strong separability.
        assert result.silhouette is not None
        assert result.silhouette > 0.5
        assert sorted(result.cluster_sizes) == [8, 8]


class TestSegmentExport:
    def test_segment_report_written(self, tmp_path) -> None:
        segs = _two_speaker_segments(n_each=4)
        config = SpeakerBenchmarkConfig(backends=["mock"], device="cpu", asr_engine=None)
        run = SpeakerBenchmarkRunner(config)
        run._segments = segs
        run._spans = [(i * 0.5, i * 0.5 + 0.5) for i in range(len(segs))]
        run._audio_duration = sum(len(s) for s in segs) / SR

        result = run._benchmark_backend("mock")
        assert result.status == "ok"
        assert len(run._detail) == len(segs)

        transcripts = [f"発話{i}" for i in range(len(segs))]
        run._write_segment_report(tmp_path, "mock", run._detail, transcripts)

        md = tmp_path / "segments_mock.md"
        js = tmp_path / "segments_mock.json"
        assert md.exists() and js.exists()
        data = json.loads(js.read_text(encoding="utf-8"))
        assert len(data["segments"]) == len(segs)
        assert data["segments"][0]["text"] == "発話0"
        assert "cluster" in data["segments"][0]


class TestTo16kMono:
    def test_downsamples_48k_to_16k(self) -> None:
        sr = 48000
        audio = np.zeros(sr, dtype=np.float32)  # 1 second
        out = SpeakerBenchmarkRunner._to_16k_mono(audio, sr)
        assert abs(len(out) - SR) <= 1  # ~16000 samples

    def test_converts_stereo_to_mono(self) -> None:
        stereo = np.zeros((16000, 2), dtype=np.float32)
        out = SpeakerBenchmarkRunner._to_16k_mono(stereo, SR)
        assert out.ndim == 1
        assert len(out) == 16000

    def test_passthrough_16k_mono(self) -> None:
        audio = np.ones(16000, dtype=np.float32)
        out = SpeakerBenchmarkRunner._to_16k_mono(audio, SR)
        assert len(out) == 16000


class TestDetailReset:
    def test_detail_reset_on_failed_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(_bid):
            raise ValueError("create failed")

        monkeypatch.setattr(runner_mod, "create_embedding_backend", _boom)
        config = SpeakerBenchmarkConfig(backends=["mock"], device="cpu", asr_engine=None)
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments(n_each=2)
        run._detail = [{"idx": 99, "cluster": 0}]  # stale detail from a prior backend

        result = run._benchmark_backend("mock")
        assert result.status == "failed"
        assert run._detail == []  # reset, not the stale list


class TestCalibrationGuards:
    def test_gold_without_labels_file_errors_at_cli(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--backend", "mock", "--calibrate", "--label-source", "gold"])

    def test_silver_without_labels_file_errors_at_cli(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--backend", "mock", "--calibrate", "--label-source", "silver"])

    def test_gold_with_labels_file_ok_at_cli(self, tmp_path) -> None:
        f = tmp_path / "labels.csv"
        f.write_text("idx,speaker\n0,A\n", encoding="utf-8")
        ns = parse_args(
            ["--backend", "mock", "--calibrate", "--label-source", "gold",
             "--labels-file", str(f)]
        )
        assert ns.label_source == "gold" and ns.labels_file is not None

    def test_calibrate_skips_not_fallback_when_gold_labels_missing(self) -> None:
        # No labels_file with gold: calibration is skipped (NOT silently self).
        config = SpeakerBenchmarkConfig(
            backends=["mock"], device="cpu", asr_engine=None,
            calibrate=True, label_source="gold", labels_file=None,
        )
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments()
        run._spans = [(i * 0.5, i * 0.5 + 0.5) for i in range(len(run._segments))]
        run._audio_duration = sum(len(s) for s in run._segments) / SR

        result = run._benchmark_backend("mock")
        assert result.status == "ok"  # measurement still succeeds
        assert result.eer is None  # no fabricated self-consistency EER
        assert "skipped" in (result.cal_label_source or "")

    def test_calibrate_self_still_computes(self) -> None:
        config = SpeakerBenchmarkConfig(
            backends=["mock"], device="cpu", asr_engine=None,
            calibrate=True, label_source="self",
        )
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments()
        run._spans = [(i * 0.5, i * 0.5 + 0.5) for i in range(len(run._segments))]
        run._audio_duration = sum(len(s) for s in run._segments) / SR

        result = run._benchmark_backend("mock")
        assert result.cal_label_source == "self"
        assert result.eer is not None


class TestGracefulSkip:
    def test_unavailable_backend_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Stub:
            def load(self, device: str) -> None:
                raise ImportError("backend deps not installed")

            def extract_embedding(self, audio, sample_rate=16000):  # pragma: no cover
                raise AssertionError("should not be called")

            @property
            def name(self) -> str:
                return "stub"

            @property
            def embedding_dim(self) -> int:
                return 1

        monkeypatch.setattr(runner_mod, "create_embedding_backend", lambda bid: _Stub())

        config = SpeakerBenchmarkConfig(backends=["titanet"], device="cpu")
        run = SpeakerBenchmarkRunner(config)
        run._segments = _two_speaker_segments(n_each=2)
        run._audio_duration = 1.0

        result = run._benchmark_backend("titanet")
        assert result.status == "skipped"
        assert "not installed" in result.detail


class TestRunEndToEnd:
    def test_run_writes_reports(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = _two_speaker_segments()
        audio = np.concatenate(segs)

        monkeypatch.setattr(SpeakerBenchmarkRunner, "_load_audio", lambda self: audio)
        monkeypatch.setattr(
            SpeakerBenchmarkRunner, "_segment_audio", lambda self, a: segs
        )

        config = SpeakerBenchmarkConfig(
            backends=["mock"],
            device="cpu",
            output_dir=tmp_path,
            isolate=False,
            asr_engine=None,  # no heavy ASR model in tests
        )
        run = SpeakerBenchmarkRunner(config)
        result_dir = run.run()

        assert (result_dir / "results.json").exists()
        assert (result_dir / "summary.md").exists()

        payload = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
        assert payload["benchmark_type"] == "speaker"
        assert payload["results"][0]["backend"] == "mock"
        assert payload["results"][0]["status"] == "ok"
