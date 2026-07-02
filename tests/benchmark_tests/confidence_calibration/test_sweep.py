"""Tests for ``benchmarks.confidence_calibration.sweep`` (Issue #338 PR-β)。

MockEngine 経由で end-to-end test、実 model 不要。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import argparse

from benchmarks.confidence_calibration._core import LabeledSample
from benchmarks.confidence_calibration.sweep import (
    _parse_engine_kwargs,
    breakdown_list,
    main,
    measure_signals,
)


# ----------------- _parse_engine_kwargs ---------------------------------


class TestParseEngineKwargs:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["use_int8=true"], {"use_int8": True}),
            (["use_int8=false"], {"use_int8": False}),
            (["batch_size=24"], {"batch_size": 24}),
            (["threshold=-0.5"], {"threshold": -0.5}),
            (["model_size=base"], {"model_size": "base"}),
            ([], {}),
            (
                ["use_int8=true", "model_size=base", "batch=4"],
                {"use_int8": True, "model_size": "base", "batch": 4},
            ),
        ],
    )
    def test_parse(self, raw, expected):
        assert _parse_engine_kwargs(raw) == expected

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            _parse_engine_kwargs(["malformed"])


# ----------------- measure_signals (MockEngine 経由) --------------------


class _MockEngineConfidence:
    """Mock の engine_confidence、avg_logprob 等を持つ。"""

    def __init__(
        self,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
        token_confidence_mean: float | None = None,
    ):
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.token_confidence_mean = token_confidence_mean
        self.compression_ratio = None

    @property
    def is_available(self) -> bool:
        return any(
            v is not None
            for v in (
                self.avg_logprob,
                self.no_speech_prob,
                self.token_confidence_mean,
            )
        )


class _MockTranscriptionResult:
    def __init__(self, text: str, engine_confidence: _MockEngineConfidence):
        self.text = text
        self.confidence = 1.0
        self.engine_confidence = engine_confidence


class _MockEngineForSweep:
    """sweep.measure_signals テスト用 mock engine。"""

    def __init__(self, signal_values_by_path: dict[str, float | None]):
        self.signal_values_by_path = signal_values_by_path
        self.transcribe_call_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> _MockTranscriptionResult:
        self.transcribe_call_count += 1
        # path は外側で割り当てる、ここでは call_count から代理
        # transcribe は audio から signal を引けないので、measure_signals 側で
        # path → signal の対応を取る test 構成にする (test 専用 hack)
        return _MockTranscriptionResult(
            text="dummy text",
            engine_confidence=_MockEngineConfidence(avg_logprob=-0.2),
        )


class _MockCorpusItem:
    """pipeline.CalibrationCorpusItem の minimal mock。"""

    def __init__(self, path: str, label: str, signal_value: float | None, language: str = "ja"):
        self.path = path
        self.label = label
        self.audio = np.zeros(16000, dtype=np.float32)
        self.sample_rate = 16000
        self.metadata = {"language": language}
        self._signal_value = signal_value  # test 用 hack、engine が見て返す


class _DeterministicMockEngine:
    """transcribe で各 item の事前定義 signal を返す mock。"""

    def __init__(self):
        self.transcribe_call_count = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> _MockTranscriptionResult:
        # 各 call 番号を見て signal を変えるのは複雑なので、static avg_logprob を返す
        # measure_signals test では item.audio で signal を制御 (audio[0] に埋め込み)
        self.transcribe_call_count += 1
        signal = float(audio[0]) if len(audio) > 0 and audio[0] != 0 else -0.2
        return _MockTranscriptionResult(
            text=f"text_{self.transcribe_call_count}",
            engine_confidence=_MockEngineConfidence(avg_logprob=signal),
        )


class _ItemWithSignalAudio:
    """audio[0] に signal_value を埋め込んだ test corpus item。"""

    def __init__(self, path: str, label: str, signal_value: float, language: str = "ja"):
        self.path = path
        self.label = label
        # audio[0] に signal を埋め込み、engine が見て返す
        self.audio = np.full(16000, signal_value, dtype=np.float32)
        self.sample_rate = 16000
        self.metadata = {"language": language}


class TestMeasureSignals:
    def test_measure_extracts_signal_values(self):
        items = [
            _ItemWithSignalAudio("speech_001.wav", "speech", -0.05),
            _ItemWithSignalAudio("non_speech_001.wav", "non_speech", -0.50),
        ]
        engine = _DeterministicMockEngine()
        samples = measure_signals(items, engine, "avg_logprob")
        assert len(samples) == 2
        assert engine.transcribe_call_count == 2
        # signal_value は audio[0] が transcribe 内で avg_logprob に
        assert samples[0].signal_value == pytest.approx(-0.05, abs=1e-3)
        assert samples[0].label == "speech"
        assert samples[1].signal_value == pytest.approx(-0.50, abs=1e-3)
        assert samples[1].label == "non_speech"

    def test_transcribe_failure_records_none(self):
        items = [_ItemWithSignalAudio("a.wav", "speech", -0.1)]
        engine = MagicMock()
        engine.transcribe.side_effect = RuntimeError("model crashed")
        samples = measure_signals(items, engine, "avg_logprob")
        assert len(samples) == 1
        assert samples[0].signal_value is None
        assert "model crashed" in samples[0].metadata["transcribe_error"]

    def test_full_metadata_pass_through(self):
        """Phase 6a: manifest 由来の任意 metadata (snr_db / subtype / etc) が
        ``LabeledSample.metadata`` に full pass-through される。"""
        item = _ItemWithSignalAudio("a.wav", "noisy_speech", -0.15)
        # manifest 相当の追加 metadata
        item.metadata = {
            "language": "ja",
            "snr_db": 10.0,
            "subtype": "clapping",
            "noise_source_dataset": "esc50",
            "reference_text_matched": "テキスト",
        }
        engine = _DeterministicMockEngine()
        samples = measure_signals([item], engine, "avg_logprob")
        assert len(samples) == 1
        m = samples[0].metadata
        # manifest 由来の全 key が到達
        assert m["snr_db"] == 10.0
        assert m["subtype"] == "clapping"
        assert m["noise_source_dataset"] == "esc50"
        assert m["reference_text_matched"] == "テキスト"
        # engine/result 由来の 3 key も存在
        assert "text" in m
        assert m["language"] == "ja"
        assert "is_available" in m

    def test_error_path_preserves_manifest_metadata(self):
        """Phase 6a: transcribe error 時も manifest metadata を pass-through
        (SNR 別集計時に error sample も分類可能に)。"""
        item = _ItemWithSignalAudio("a.wav", "speech", -0.1)
        item.metadata = {"language": "ja", "snr_db": 5.0, "subtype": "engine"}
        engine = MagicMock()
        engine.transcribe.side_effect = RuntimeError("crash")
        samples = measure_signals([item], engine, "avg_logprob")
        assert samples[0].signal_value is None
        assert samples[0].metadata["snr_db"] == 5.0
        assert samples[0].metadata["subtype"] == "engine"
        assert "crash" in samples[0].metadata["transcribe_error"]


# ----------------- Phase 6a: breakdown_list argparse type -------------------


class TestBreakdownList:
    def test_single_key(self):
        assert breakdown_list("snr_db") == ["snr_db"]

    def test_multiple_keys(self):
        assert breakdown_list("snr_db,subtype,noise_source_dataset") == [
            "snr_db",
            "subtype",
            "noise_source_dataset",
        ]

    def test_strips_whitespace(self):
        assert breakdown_list("snr_db, subtype , noise_source_dataset") == [
            "snr_db",
            "subtype",
            "noise_source_dataset",
        ]

    def test_rejects_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="must not be empty"):
            breakdown_list("")

    def test_rejects_whitespace_only(self):
        with pytest.raises(argparse.ArgumentTypeError, match="must not be empty"):
            breakdown_list("   ")

    def test_rejects_empty_item(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty item"):
            breakdown_list("snr_db,,subtype")

    def test_rejects_duplicate(self):
        with pytest.raises(argparse.ArgumentTypeError, match="duplicate key"):
            breakdown_list("snr_db,snr_db")

    def test_rejects_duplicate_across_positions(self):
        with pytest.raises(argparse.ArgumentTypeError, match="duplicate key"):
            breakdown_list("snr_db,subtype,snr_db")


# ----------------- main() end-to-end (mock engine_factory) -------------


def _write_silence_wav(path: Path, duration_sec: float = 0.1) -> None:
    n = int(duration_sec * 16000)
    audio = np.zeros(n, dtype=np.float32)
    sf.write(str(path), audio, 16000)


class TestMainE2E:
    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_main_writes_report(self, mock_create: MagicMock, tmp_path: Path):
        # Mock engine
        engine = MagicMock()
        engine.load_model = MagicMock()
        engine.get_engine_name.return_value = "MockEngine"
        # 各 transcribe で異なる avg_logprob (audio 中身による)、ここでは固定 mock pattern
        # → 各 transcribe call の avg_logprob を audio[0] で制御
        def fake_transcribe(audio, sr):
            return _MockTranscriptionResult(
                text="t",
                engine_confidence=_MockEngineConfidence(avg_logprob=float(audio[0])),
            )

        engine.transcribe.side_effect = fake_transcribe
        engine.cleanup = MagicMock()
        mock_create.return_value = engine

        # Build corpus (4 sample: 2 speech + 2 non_speech)
        corpus_dir = tmp_path / "corpus"
        clean_dir = corpus_dir / "ja_clean"
        non_dir = corpus_dir / "ja_non_speech"
        clean_dir.mkdir(parents=True)
        non_dir.mkdir(parents=True)

        # 各 wav の audio[0] に signal を埋め込み (test hack)
        for path, signal in [
            (clean_dir / "a.wav", -0.05),
            (clean_dir / "b.wav", -0.10),
            (non_dir / "c.wav", -0.45),
            (non_dir / "d.wav", -0.50),
        ]:
            n = 16000
            audio = np.full(n, signal, dtype=np.float32)
            sf.write(str(path), audio, 16000)

        manifest = corpus_dir / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "path": "ja_clean/a.wav",
                            "label": "speech",
                            "language": "ja",
                        }
                    ),
                    json.dumps(
                        {
                            "path": "ja_clean/b.wav",
                            "label": "speech",
                            "language": "ja",
                        }
                    ),
                    json.dumps(
                        {
                            "path": "ja_non_speech/c.wav",
                            "label": "non_speech",
                            "language": "ja",
                        }
                    ),
                    json.dumps(
                        {
                            "path": "ja_non_speech/d.wav",
                            "label": "non_speech",
                            "language": "ja",
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        rc = main(
            [
                "--engine", "mock",
                "--signal", "avg_logprob",
                "--corpus-dir", str(corpus_dir),
                "--threshold-min", "-0.6",
                "--threshold-max", "0.0",
                "--step", "0.05",
                "--output", str(output),
                "--quantization", "float32",
                "--filter-by-language", "ja",
            ]
        )
        assert rc == 0
        assert output.exists()
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["engine"] == "mock"
        assert report["signal_field"] == "avg_logprob"
        assert report["direction"] == "reject_if_less"
        assert report["sample_count"]["speech"] == 2
        assert report["sample_count"]["non_speech"] == 2
        assert report["metadata"]["quantization"] == "float32"
        assert report["metadata"]["language"] == "ja"
        # 完全分離 (speech ≈ -0.05/-0.10、non_speech ≈ -0.45/-0.50) で F1=1.0
        assert report["recommended_metrics"]["f1"] == 1.0

    def test_main_returns_error_when_no_corpus_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("LIVECAP_CALIBRATION_CORPUS_DIR", raising=False)
        rc = main(
            [
                "--engine", "mock",
                "--signal", "avg_logprob",
                "--output", str(tmp_path / "report.json"),
            ]
        )
        assert rc == 1

    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_main_returns_error_when_no_items_after_filter(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        engine = MagicMock()
        engine.load_model = MagicMock()
        mock_create.return_value = engine

        corpus_dir = tmp_path / "corpus"
        clean_dir = corpus_dir / "ja_clean"
        clean_dir.mkdir(parents=True)
        _write_silence_wav(clean_dir / "a.wav")

        manifest = corpus_dir / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {"path": "ja_clean/a.wav", "label": "speech", "language": "ja"}
            ),
            encoding="utf-8",
        )

        # filter by en, corpus は ja のみ → 0 件で error
        rc = main(
            [
                "--engine", "mock",
                "--signal", "avg_logprob",
                "--corpus-dir", str(corpus_dir),
                "--filter-by-language", "en",
                "--output", str(tmp_path / "report.json"),
            ]
        )
        assert rc == 1

    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_breakdown_by_flag_produces_per_key_sections(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        """Phase 6a: ``--breakdown-by snr_db,subtype`` 指定時、 report JSON に
        ``report["breakdown"][key]`` が populate される。"""
        engine = MagicMock()
        engine.load_model = MagicMock()
        engine.get_engine_name.return_value = "MockEngine"

        def fake_transcribe(audio, sr):
            return _MockTranscriptionResult(
                text="t",
                engine_confidence=_MockEngineConfidence(avg_logprob=float(audio[0])),
            )

        engine.transcribe.side_effect = fake_transcribe
        engine.cleanup = MagicMock()
        mock_create.return_value = engine

        corpus_dir = tmp_path / "corpus"
        clean_dir = corpus_dir / "ja_clean"
        noisy_dir = corpus_dir / "ja_noisy_speech"
        clean_dir.mkdir(parents=True)
        noisy_dir.mkdir(parents=True)

        for path, signal in [
            (clean_dir / "a.wav", -0.05),   # clean speech, no snr_db
            (noisy_dir / "b.wav", -0.15),   # noisy_speech, SNR 10
            (noisy_dir / "c.wav", -0.35),   # noisy_speech, SNR 0
        ]:
            n = 16000
            audio = np.full(n, signal, dtype=np.float32)
            sf.write(str(path), audio, 16000)

        manifest = corpus_dir / "manifest.jsonl"
        manifest.write_text(
            "\n".join([
                json.dumps({
                    "path": "ja_clean/a.wav",
                    "label": "speech",
                    "language": "ja",
                }),
                json.dumps({
                    "path": "ja_noisy_speech/b.wav",
                    "label": "noisy_speech",
                    "language": "ja",
                    "snr_db": 10.0,
                    "subtype": "clapping",
                }),
                json.dumps({
                    "path": "ja_noisy_speech/c.wav",
                    "label": "noisy_speech",
                    "language": "ja",
                    "snr_db": 0.0,
                    "subtype": "clapping",
                }),
            ]),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        rc = main([
            "--engine", "mock",
            "--signal", "avg_logprob",
            "--corpus-dir", str(corpus_dir),
            "--threshold-min", "-0.5",
            "--threshold-max", "0.0",
            "--step", "0.1",
            "--output", str(output),
            "--filter-by-language", "ja",
            "--breakdown-by", "snr_db,subtype",
        ])
        assert rc == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        # 全体 sweep は backward compat
        assert report["sample_count"]["speech"] == 1
        assert report["sample_count"]["noisy_speech"] == 2
        # Phase 6a: breakdown が populate されている
        assert "snr_db" in report["breakdown"]
        assert "subtype" in report["breakdown"]
        # snr_db bucket: clean (__none__) 1 + SNR 10 (1) + SNR 0 (1)
        snr_counts = report["breakdown"]["snr_db"]["value_counts"]
        assert snr_counts == {"__none__": 1, "10.0": 1, "0.0": 1}
        # subtype bucket: clean (__none__) 1 + clapping 2
        subtype_counts = report["breakdown"]["subtype"]["value_counts"]
        assert subtype_counts == {"__none__": 1, "clapping": 2}
        # metadata.breakdown_by で実施した key を記録
        assert report["metadata"]["breakdown_by"] == ["snr_db", "subtype"]

    @patch("livecap_cli.engines.engine_factory.EngineFactory.create_engine")
    def test_backward_compat_no_breakdown_flag(
        self, mock_create: MagicMock, tmp_path: Path
    ):
        """Phase 1 report との backward compat: ``--breakdown-by`` 未指定時、
        ``report["breakdown"] == {}``。"""
        engine = MagicMock()
        engine.load_model = MagicMock()
        engine.get_engine_name.return_value = "MockEngine"

        def fake_transcribe(audio, sr):
            return _MockTranscriptionResult(
                text="t",
                engine_confidence=_MockEngineConfidence(avg_logprob=float(audio[0])),
            )

        engine.transcribe.side_effect = fake_transcribe
        engine.cleanup = MagicMock()
        mock_create.return_value = engine

        corpus_dir = tmp_path / "corpus"
        clean_dir = corpus_dir / "ja_clean"
        clean_dir.mkdir(parents=True)
        for path, signal in [
            (clean_dir / "a.wav", -0.05),
            (clean_dir / "b.wav", -0.50),
        ]:
            audio = np.full(16000, signal, dtype=np.float32)
            sf.write(str(path), audio, 16000)

        manifest = corpus_dir / "manifest.jsonl"
        manifest.write_text(
            "\n".join([
                json.dumps({"path": "ja_clean/a.wav", "label": "speech", "language": "ja"}),
                json.dumps({"path": "ja_clean/b.wav", "label": "non_speech", "language": "ja"}),
            ]),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        rc = main([
            "--engine", "mock",
            "--signal", "avg_logprob",
            "--corpus-dir", str(corpus_dir),
            "--threshold-min", "-0.6",
            "--threshold-max", "0.0",
            "--step", "0.1",
            "--output", str(output),
            "--filter-by-language", "ja",
        ])
        assert rc == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        # 追加 field は空 dict で存在 (Phase 1 report との互換)
        assert report["breakdown"] == {}
        # metadata.breakdown_by は unused (未 populate)
        assert "breakdown_by" not in report["metadata"]
