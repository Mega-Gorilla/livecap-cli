"""Tests for ``benchmarks.confidence_calibration.parse_observe`` (Issue #338 PR-α)。

Log file + labels.jsonl の parse + join 動作を pin。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.confidence_calibration.parse_observe import (
    LabelEntry,
    LogEntry,
    load_labels,
    main,
    normalize_engine_id,
    parse_log_line,
    parse_observe_log,
)


# ----------------- parse_log_line --------------------------------------


class TestParseLogLine:
    def test_valid_observe_line(self):
        """``confidence_filter[observe]: <JSON>`` を正しく parse。"""
        line = (
            "2026-06-29 10:00:00 INFO livecap_cli.transcription.confidence_filter "
            'confidence_filter[observe]: {"source_id": "mic_001", "engine": "reazonspeech", '
            '"text": "hello", "decision": "pass", "reason": null, '
            '"engine_confidence": {"no_speech_prob": null, "avg_logprob": -0.15, '
            '"compression_ratio": null, "token_confidence_mean": null, "is_available": true}}'
        )
        entry = parse_log_line(line, signal_field="avg_logprob")
        assert entry is not None
        assert entry.source_id == "mic_001"
        assert entry.engine == "reazonspeech"
        assert entry.signal_value == -0.15
        assert entry.decision == "pass"

    def test_line_without_prefix_returns_none(self):
        assert parse_log_line("random log line\n", "avg_logprob") is None
        assert parse_log_line("", "avg_logprob") is None

    def test_malformed_json_returns_none(self):
        line = "confidence_filter[observe]: {malformed"
        assert parse_log_line(line, "avg_logprob") is None

    def test_missing_signal_field_returns_none_value(self):
        """signal_field が engine_confidence に無い時は signal_value=None。"""
        line = (
            'confidence_filter[observe]: {"source_id": "x", "engine": "reazonspeech", '
            '"text": "", "decision": "pass", "reason": null, '
            '"engine_confidence": {"is_available": true}}'
        )
        entry = parse_log_line(line, signal_field="avg_logprob")
        assert entry is not None
        assert entry.signal_value is None


# ----------------- load_labels ------------------------------------------


class TestLoadLabels:
    """``load_labels()`` returns (by_composite, by_text, by_source) tuple (PR #339 fix)."""

    def test_load_valid_labels(self, tmp_path: Path):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            '{"source_id": "mic_001", "label": "speech", "text": "hello"}\n'
            '{"source_id": "mic_002", "label": "non_speech", "subtype": "applause"}\n',
            encoding="utf-8",
        )
        by_composite, by_text, by_source = load_labels(labels)
        # source_id 単独 index
        assert "mic_001" in by_source
        assert "mic_002" in by_source
        assert by_source["mic_001"].label == "speech"
        assert by_source["mic_002"].metadata["subtype"] == "applause"
        # (source_id, text) index
        assert ("mic_001", "hello") in by_text
        # occurrence_index は entry に無いので by_composite は空
        assert len(by_composite) == 0

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_labels(tmp_path / "nonexistent.jsonl")

    def test_malformed_line_skipped(self, tmp_path: Path, caplog):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            '{"source_id": "mic_001", "label": "speech"}\n'
            "{malformed json line\n"
            '{"source_id": "mic_002", "label": "non_speech"}\n',
            encoding="utf-8",
        )
        _, _, by_source = load_labels(labels)
        assert "mic_001" in by_source
        assert "mic_002" in by_source
        assert len(by_source) == 2

    def test_missing_source_id_skipped(self, tmp_path: Path):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            '{"label": "speech"}\n'  # source_id 欠落
            '{"source_id": "mic_001", "label": "speech"}\n',
            encoding="utf-8",
        )
        _, _, by_source = load_labels(labels)
        assert "mic_001" in by_source
        assert len(by_source) == 1

    def test_occurrence_index_indexed(self, tmp_path: Path):
        """``occurrence_index`` を含む label は ``by_composite`` に index される (PR #339 fix)。"""
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            '{"source_id": "default", "occurrence_index": 0, "label": "speech"}\n'
            '{"source_id": "default", "occurrence_index": 1, "label": "non_speech"}\n',
            encoding="utf-8",
        )
        by_composite, _, _ = load_labels(labels)
        assert ("default", 0) in by_composite
        assert ("default", 1) in by_composite
        assert by_composite[("default", 0)].label == "speech"
        assert by_composite[("default", 1)].label == "non_speech"

    def test_duplicate_source_id_warns(self, tmp_path: Path, caplog):
        """source_id 重複は last-wins、warn を出す (multi-utterance log の hint)。"""
        import logging

        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            '{"source_id": "default", "label": "speech"}\n'
            '{"source_id": "default", "label": "non_speech"}\n',
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            _, _, by_source = load_labels(labels)
        assert by_source["default"].label == "non_speech"  # last wins
        assert any("duplicated" in r.getMessage() for r in caplog.records)


# ----------------- normalize_engine_id --------------------------------


class TestNormalizeEngineId:
    """PR #339 codex-review fix: engine display name の正規化。"""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("ReazonSpeech K2 (CPU, Int8)", "reazonspeech"),
            ("ReazonSpeech K2 (CPU, Float32)", "reazonspeech"),
            ("WhisperS2T base", "whispers2t"),
            ("Qwen3-ASR 0.6B", "qwen3-asr"),
            ("voxtral", "voxtral"),
            ("Canary 1B Flash", "canary"),
            ("reazonspeech", "reazonspeech"),  # 既に ID
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalize(self, input_str: str, expected: str):
        assert normalize_engine_id(input_str) == expected


# ----------------- parse_observe_log (E2E join) -------------------------


def _make_log_line(source_id: str, engine: str, signal_value: float | None) -> str:
    ec = {
        "no_speech_prob": None,
        "avg_logprob": signal_value,
        "compression_ratio": None,
        "token_confidence_mean": None,
        "is_available": signal_value is not None,
    }
    payload = {
        "source_id": source_id,
        "engine": engine,
        "text": f"text for {source_id}",
        "decision": "pass",
        "reason": None,
        "engine_confidence": ec,
    }
    return f"confidence_filter[observe]: {json.dumps(payload, ensure_ascii=False)}"


class TestParseObserveLog:
    def test_full_join(self, tmp_path: Path):
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("mic_001", "reazonspeech", -0.10),
                    _make_log_line("mic_002", "reazonspeech", -0.45),
                    _make_log_line("mic_003", "reazonspeech", -0.50),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "mic_001", "label": "speech"}),
                    json.dumps({"source_id": "mic_002", "label": "non_speech"}),
                    json.dumps({"source_id": "mic_003", "label": "non_speech"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",
            signal_field="avg_logprob",
        )
        assert len(samples) == 3
        speech = [s for s in samples if s.label == "speech"]
        non_speech = [s for s in samples if s.label == "non_speech"]
        assert len(speech) == 1
        assert len(non_speech) == 2

    def test_skip_other_engine(self, tmp_path: Path):
        """target engine 以外の log entry は skip。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("mic_001", "reazonspeech", -0.10),
                    _make_log_line("mic_002", "whispers2t", -0.45),  # skip
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            json.dumps({"source_id": "mic_001", "label": "speech"}) + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",
            signal_field="avg_logprob",
        )
        assert len(samples) == 1
        # path は "source_id#occurrence" format (PR #339 fix)
        assert samples[0].path == "mic_001#0"

    def test_display_engine_string_matched_via_normalize(self, tmp_path: Path):
        """observe log の engine field が display string でも CLI ID と match (PR #339 fix)。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("default", "ReazonSpeech K2 (CPU, Int8)", -0.10),
                    _make_log_line("default", "ReazonSpeech K2 (CPU, Float32)", -0.45),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "default", "occurrence_index": 0, "label": "speech"}),
                    json.dumps({"source_id": "default", "occurrence_index": 1, "label": "non_speech"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",  # ← CLI ID
            signal_field="avg_logprob",
        )
        # 両 display string が "reazonspeech" に normalize されて match
        assert len(samples) == 2
        assert samples[0].label == "speech"
        assert samples[1].label == "non_speech"

    def test_composite_key_match_for_shared_source_id(self, tmp_path: Path):
        """source_id=default の複数 utterance を occurrence_index で個別 join (PR #339 fix)。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("default", "reazonspeech", -0.05),  # occurrence 0
                    _make_log_line("default", "reazonspeech", -0.45),  # occurrence 1
                    _make_log_line("default", "reazonspeech", -0.50),  # occurrence 2
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "default", "occurrence_index": 0, "label": "speech"}),
                    json.dumps({"source_id": "default", "occurrence_index": 1, "label": "non_speech"}),
                    json.dumps({"source_id": "default", "occurrence_index": 2, "label": "non_speech"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",
            signal_field="avg_logprob",
        )
        assert len(samples) == 3
        assert samples[0].label == "speech"
        assert samples[1].label == "non_speech"
        assert samples[2].label == "non_speech"
        # path は source_id#occurrence で識別可能
        assert samples[0].path == "default#0"
        assert samples[1].path == "default#1"
        assert samples[2].path == "default#2"

    def test_text_based_match_as_fallback(self, tmp_path: Path):
        """occurrence_index 無しでも text で match できる (PR #339 fix)。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        # _make_log_line() は text="text for {source_id}" を出す
        log.write_text(
            _make_log_line("default", "reazonspeech", -0.10) + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            json.dumps(
                {
                    "source_id": "default",
                    "text": "text for default",  # _make_log_line と一致
                    "label": "speech",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",
            signal_field="avg_logprob",
        )
        assert len(samples) == 1
        assert samples[0].label == "speech"

    def test_unmatched_log_skipped(self, tmp_path: Path):
        """labels に無い source_id の log entry は skip。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("mic_001", "reazonspeech", -0.10),
                    _make_log_line("mic_unlabeled", "reazonspeech", -0.45),  # skip
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            json.dumps({"source_id": "mic_001", "label": "speech"}) + "\n",
            encoding="utf-8",
        )
        samples = parse_observe_log(
            log_path=log,
            labels_path=labels,
            engine="reazonspeech",
            signal_field="avg_logprob",
        )
        assert len(samples) == 1

    def test_log_file_missing_raises(self, tmp_path: Path):
        labels = tmp_path / "labels.jsonl"
        labels.write_text("", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            parse_observe_log(
                log_path=tmp_path / "nonexistent.jsonl",
                labels_path=labels,
                engine="reazonspeech",
                signal_field="avg_logprob",
            )


# ----------------- CLI main() end-to-end --------------------------------


class TestMain:
    def test_main_writes_report(self, tmp_path: Path, capsys):
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        output = tmp_path / "report.json"
        log.write_text(
            "\n".join(
                [
                    _make_log_line("mic_001", "reazonspeech", -0.05),
                    _make_log_line("mic_002", "reazonspeech", -0.10),
                    _make_log_line("mic_003", "reazonspeech", -0.45),
                    _make_log_line("mic_004", "reazonspeech", -0.50),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            "\n".join(
                [
                    json.dumps({"source_id": "mic_001", "label": "speech"}),
                    json.dumps({"source_id": "mic_002", "label": "speech"}),
                    json.dumps({"source_id": "mic_003", "label": "non_speech"}),
                    json.dumps({"source_id": "mic_004", "label": "non_speech"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rc = main(
            [
                "--log", str(log),
                "--labels", str(labels),
                "--engine", "reazonspeech",
                "--signal", "avg_logprob",
                "--threshold-min", "-0.6",
                "--threshold-max", "0.0",
                "--step", "0.05",
                "--output", str(output),
                "--quantization", "float32",
                "--language", "ja",
            ]
        )
        assert rc == 0
        assert output.exists()
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["engine"] == "reazonspeech"
        assert report["signal_field"] == "avg_logprob"
        assert report["direction"] == "reject_if_less"
        assert report["sample_count"]["speech"] == 2
        assert report["sample_count"]["non_speech"] == 2
        assert report["metadata"]["quantization"] == "float32"
        assert report["metadata"]["language"] == "ja"
        # 完全分離なので F1=1.0
        assert report["recommended_metrics"]["f1"] == 1.0

    def test_main_no_match_returns_error(self, tmp_path: Path):
        """matched sample が無い場合 rc=1。"""
        log = tmp_path / "observe.jsonl"
        labels = tmp_path / "labels.jsonl"
        log.write_text(
            _make_log_line("mic_unmatched", "reazonspeech", -0.10) + "\n",
            encoding="utf-8",
        )
        labels.write_text(
            json.dumps({"source_id": "mic_999", "label": "speech"}) + "\n",
            encoding="utf-8",
        )
        rc = main(
            [
                "--log", str(log),
                "--labels", str(labels),
                "--engine", "reazonspeech",
                "--signal", "avg_logprob",
                "--output", str(tmp_path / "report.json"),
            ]
        )
        assert rc == 1
