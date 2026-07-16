"""FileTranscriptionPipeline の全滅/部分失敗の区別テスト (Issue #362)。

gui#392 (v3.1.0 空 SRT 障害) の増幅要因だった「全 segment の transcriber 例外を
success=True で完走し 0 byte SRT を出力する」silent failure の回帰テスト。
engine / torch / FFmpeg 不要 (plain WAV + mock transcriber / custom segmenter)。
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from livecap_cli.transcription.file_pipeline import (
    FileProcessingResult,
    FileTranscriptionCancelled,
    FileTranscriptionPipeline,
    FileTranscriptionProgress,
)


def _write_wav(path: Path, seconds: float = 3.0, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.zeros(int(sample_rate * seconds), dtype=np.int16).tobytes())


def _three_segmenter(audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
    """3 秒の音声を 1 秒ずつ 3 segment に切る (per-segment 挙動を制御するため)。"""
    return [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.wav"
    _write_wav(path)
    return path


@pytest.fixture
def pipeline() -> FileTranscriptionPipeline:
    p = FileTranscriptionPipeline(segmenter=_three_segmenter)
    yield p
    p.close()


class TestTotalFailure:
    """全 ASR 呼び出しが例外 → success=False (Issue #362 の核心)"""

    def test_all_segments_raise_promotes_to_failure(self, pipeline, audio_path):
        """gui#392 相当: 全 segment で TypeError → success=False + 原因入り error"""

        def broken_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise TypeError("cannot unpack non-iterable TranscriptionResult object")

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=broken_transcriber,
            write_subtitles=True,
        )

        assert result.success is False
        assert result.output_path is None
        assert result.subtitles == []
        assert "All 3 ASR segment calls failed" in result.error
        assert "TypeError" in result.error
        assert "cannot unpack" in result.error

    def test_total_failure_does_not_create_srt(self, pipeline, audio_path):
        """全滅時は write_subtitles=True でも SRT を新規作成しない"""

        def broken_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise RuntimeError("model broken")

        pipeline.process_file(
            audio_path,
            segment_transcriber=broken_transcriber,
            write_subtitles=True,
        )

        assert not audio_path.with_suffix(".srt").exists()

    def test_total_failure_preserves_existing_srt(self, pipeline, audio_path):
        """全滅時に既存 SRT を空ファイルで上書きしない"""
        existing_srt = audio_path.with_suffix(".srt")
        original_content = "1\n00:00:00,000 --> 00:00:01,000\n既存の字幕\n"
        existing_srt.write_text(original_content, encoding="utf-8")

        def broken_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise RuntimeError("model broken")

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=broken_transcriber,
            write_subtitles=True,
        )

        assert result.success is False
        assert existing_srt.read_text(encoding="utf-8") == original_content

    def test_total_failure_metadata_counts(self, pipeline, audio_path):
        """全滅時も metadata から asr_calls == asr_errors を確認できる"""

        def broken_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise ValueError("bad contract")

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=broken_transcriber,
            write_subtitles=False,
        )

        assert result.metadata["asr_calls"] == 3
        assert result.metadata["asr_errors"] == 3
        assert result.metadata["empty_results"] == 0
        assert result.metadata["segment_count"] == 0
        assert result.metadata["sample_rate"] == 16000
        assert result.metadata["duration_seconds"] == pytest.approx(3.0)


class TestPartialFailure:
    """部分失敗は従来どおり success=True (segment 単位 fail-soft 維持)"""

    def test_partial_failure_stays_success(self, pipeline, audio_path):
        calls = [0]

        def flaky_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("transient")
            return f"text {calls[0]}"

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=flaky_transcriber,
            write_subtitles=True,
        )

        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.exists()
        assert len(result.subtitles) == 2
        assert result.metadata["asr_calls"] == 3
        assert result.metadata["asr_errors"] == 1
        assert result.metadata["empty_results"] == 0

    def test_mixed_outcomes_counted_separately(self, pipeline, audio_path):
        """1 成功・1 正常空・1 例外 → 3 種の件数を正しく区別"""
        calls = [0]

        def mixed_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            calls[0] += 1
            if calls[0] == 1:
                return "こんにちは"
            if calls[0] == 2:
                return ""  # 正常な空認識
            raise RuntimeError("boom")

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=mixed_transcriber,
            write_subtitles=False,
        )

        assert result.success is True
        assert len(result.subtitles) == 1
        assert result.metadata["asr_calls"] == 3
        assert result.metadata["asr_errors"] == 1
        assert result.metadata["empty_results"] == 1


class TestLegitimatelyEmpty:
    """正常な空 (無音等) は success=True のまま (gui#392 レビュー合意)"""

    def test_all_empty_recognition_stays_success(self, pipeline, audio_path):
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: "",
            write_subtitles=True,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 3
        assert result.metadata["asr_errors"] == 0
        assert result.metadata["empty_results"] == 3
        # 正常な空は SRT (空) を従来どおり出力する
        assert result.output_path is not None
        assert result.output_path.exists()

    def test_zero_asr_calls_stays_success(self, tmp_path):
        """切り出し区間が全て空 → ASR 呼び出し 0 件 → success=True"""
        audio_path = tmp_path / "test.wav"
        _write_wav(audio_path, seconds=1.0)

        def out_of_range_segmenter(audio, sample_rate):
            return [(10.0, 11.0)]  # 1 秒音声の範囲外 → slice が空

        pipeline = FileTranscriptionPipeline(segmenter=out_of_range_segmenter)
        try:
            result = pipeline.process_file(
                audio_path,
                segment_transcriber=lambda a, s: "should not be called",
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert result.success is True
        assert result.metadata["asr_calls"] == 0
        assert result.metadata["asr_errors"] == 0

    def test_success_metadata_always_has_counts(self, pipeline, audio_path):
        """成功経路でも件数内訳が常時格納される"""
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: "text",
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 3
        assert result.metadata["asr_errors"] == 0
        assert result.metadata["empty_results"] == 0
        assert result.metadata["segment_count"] == 3


class TestSegmentationEmpty:
    """注入 segmenter の [] は「セグメントなし」= ASR 呼び出しゼロ (#366 Phase 1)。

    旧挙動 (全音声 1 segment への fallback) は VAD が正しく無音判定した
    場合に全音声が ASR へ流れる逆転があった。
    """

    def test_injected_empty_means_no_segments(self, tmp_path):
        audio_path = tmp_path / "test.wav"
        _write_wav(audio_path, seconds=1.0)
        calls = [0]

        def transcriber(audio, sample_rate):
            calls[0] += 1
            return "should not be called"

        pipeline = FileTranscriptionPipeline(segmenter=lambda a, s: [])
        try:
            result = pipeline.process_file(
                audio_path,
                segment_transcriber=transcriber,
                write_subtitles=True,
            )
        finally:
            pipeline.close()

        assert result.success is True
        assert result.subtitles == []
        assert calls[0] == 0  # transcriber 未呼出 (全音声 fallback しない)
        assert result.metadata["asr_calls"] == 0
        assert result.metadata["segmentation_empty"] is True
        assert result.metadata["detected_segment_count"] == 0
        # write_subtitles=True では空 SRT を生成 (仕様)
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.output_path.read_text(encoding="utf-8") == ""

    def test_none_segmenter_keeps_whole_audio_fallback(self, tmp_path):
        """segmenter=None は従来どおり全音声 1 segment (fallback 温存の pin)"""
        audio_path = tmp_path / "test.wav"
        _write_wav(audio_path, seconds=1.0)

        pipeline = FileTranscriptionPipeline()
        try:
            result = pipeline.process_file(
                audio_path,
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
            )
        finally:
            pipeline.close()

        assert result.success is True
        assert result.metadata["asr_calls"] == 1
        assert result.metadata["detected_segment_count"] == 1
        assert result.metadata["segmentation_empty"] is False
        assert len(result.subtitles) == 1
        assert result.subtitles[0].start == 0.0
        assert result.subtitles[0].end == pytest.approx(1.0)

    def test_detected_segment_count_reflects_segmenter(self, pipeline, audio_path):
        """detected_segment_count は検出数 (segment_count = 字幕数とは別)"""
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: "",
            write_subtitles=False,
        )

        assert result.metadata["detected_segment_count"] == 3
        assert result.metadata["segment_count"] == 0  # 全件空認識 → 字幕 0
        assert result.metadata["segmentation_empty"] is False


class TestCancellation:
    """FileTranscriptionCancelled は asr_errors に数えず再送出"""

    def test_cancelled_reraised_not_counted(self, pipeline, audio_path):
        def cancelling_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise FileTranscriptionCancelled()

        with pytest.raises(FileTranscriptionCancelled):
            pipeline.process_file(
                audio_path,
                segment_transcriber=cancelling_transcriber,
                write_subtitles=False,
            )


class TestProcessFilesCallbacks:
    """process_files: success=False 返却時の callback 契約 (Issue #362)"""

    def test_error_callback_fires_on_returned_failure(self, tmp_path):
        audio_path = tmp_path / "test.wav"
        _write_wav(audio_path)

        def broken_transcriber(audio: np.ndarray, sample_rate: int) -> str:
            raise TypeError("contract mismatch")

        errors: list[tuple[str, Exception | None]] = []
        results: list[FileProcessingResult] = []
        progresses: list[FileTranscriptionProgress] = []

        pipeline = FileTranscriptionPipeline(segmenter=_three_segmenter)
        try:
            returned = pipeline.process_files(
                [audio_path],
                segment_transcriber=broken_transcriber,
                write_subtitles=False,
                error_callback=lambda msg, exc: errors.append((msg, exc)),
                result_callback=results.append,
                progress_callback=progresses.append,
            )
        finally:
            pipeline.close()

        # error_callback: 返却された failure でも 1 回発火 (exc=None)
        assert len(errors) == 1
        assert "All 3 ASR segment calls failed" in errors[0][0]
        assert errors[0][1] is None

        # result_callback: failed result を受領
        assert len(results) == 1
        assert results[0].success is False

        assert returned[0].success is False

        # progress: 最終通知が "failed"
        final = [p for p in progresses if p.status in ("processed", "failed")]
        assert final and final[-1].status == "failed"

    def test_error_callback_not_fired_on_success(self, tmp_path):
        audio_path = tmp_path / "test.wav"
        _write_wav(audio_path)

        errors: list[tuple[str, Exception | None]] = []

        pipeline = FileTranscriptionPipeline(segmenter=_three_segmenter)
        try:
            returned = pipeline.process_files(
                [audio_path],
                segment_transcriber=lambda a, s: "text",
                write_subtitles=False,
                error_callback=lambda msg, exc: errors.append((msg, exc)),
            )
        finally:
            pipeline.close()

        assert errors == []
        assert returned[0].success is True


class TestSegmentOutcomeContract:
    """SegmentOutcome の不変条件 (#366 Phase 2)"""

    def test_drop_reason_requires_empty_text(self):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_FILTER_REJECT

        with pytest.raises(ValueError, match="text=''"):
            SegmentOutcome(text="認識結果", drop_reason=REASON_FILTER_REJECT)

    def test_energy_gate_requires_asr_called_false(self):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_ENERGY_GATE

        with pytest.raises(ValueError, match="asr_called must be False"):
            SegmentOutcome(text="", drop_reason=REASON_ENERGY_GATE, asr_called=True)

    def test_post_asr_reasons_require_asr_called_true(self):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import (
            REASON_ENGINE_EMPTY,
            REASON_FILTER_REJECT,
        )

        for reason in (REASON_FILTER_REJECT, REASON_ENGINE_EMPTY):
            with pytest.raises(ValueError, match="asr_called must be True"):
                SegmentOutcome(text="", drop_reason=reason, asr_called=False)

    def test_classmethods(self):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_ENERGY_GATE

        ok = SegmentOutcome.success("text")
        assert ok.text == "text" and ok.drop_reason is None and ok.asr_called

        drop = SegmentOutcome.dropped(REASON_ENERGY_GATE, asr_called=False)
        assert drop.text == "" and drop.drop_reason == REASON_ENERGY_GATE
        assert drop.asr_called is False

    def test_unknown_reason_allowed_with_empty_text(self):
        """未知 reason は前方互換で許容 (drop_counts へ)"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome

        out = SegmentOutcome.dropped("future:new_filter")
        assert out.drop_reason == "future:new_filter"


class TestStructuredCounting:
    """SegmentOutcome の正規化集計 (#366 Phase 2 規則表)"""

    @staticmethod
    def _outcomes_transcriber(outcomes):
        """segment ごとに指定 outcome を順に返す transcriber"""
        it = iter(outcomes)

        def transcriber(audio, sample_rate):
            value = next(it)
            if isinstance(value, Exception):
                raise value
            return value

        return transcriber

    def test_mixed_drop_reject_success(self, pipeline, audio_path):
        """issue #366 の混在ケース期待値そのまま"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import (
            REASON_ENERGY_GATE,
            REASON_FILTER_REJECT,
        )

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=self._outcomes_transcriber([
                SegmentOutcome.dropped(REASON_ENERGY_GATE, asr_called=False),
                SegmentOutcome.dropped(REASON_FILTER_REJECT),
                SegmentOutcome.success("text"),
            ]),
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["detected_segment_count"] == 3
        assert result.metadata["asr_calls"] == 2       # gate drop は数えない
        assert result.metadata["asr_errors"] == 0
        assert result.metadata["drop_counts"] == {
            "energy_gate:low_rms": 1,
            "confidence_filter:reject": 1,
        }
        assert result.metadata["empty_results"] == 0
        assert result.metadata["segment_count"] == 1

    def test_all_energy_gate_drops_stay_success(self, pipeline, audio_path):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_ENERGY_GATE

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: SegmentOutcome.dropped(
                REASON_ENERGY_GATE, asr_called=False
            ),
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 0
        assert result.metadata["drop_counts"] == {"energy_gate:low_rms": 3}

    def test_all_filter_rejects_stay_success(self, pipeline, audio_path):
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_FILTER_REJECT

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: SegmentOutcome.dropped(
                REASON_FILTER_REJECT
            ),
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 3   # ASR は呼ばれている
        assert result.metadata["drop_counts"] == {"confidence_filter:reject": 3}

    def test_gate_drop_plus_all_errors_promotes_failure(self, pipeline, audio_path):
        """gate drop 1 + 残り全例外 → asr_calls == asr_errors → success=False
        (#362 全滅判定と干渉しない)"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_ENERGY_GATE

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=self._outcomes_transcriber([
                SegmentOutcome.dropped(REASON_ENERGY_GATE, asr_called=False),
                RuntimeError("boom 1"),
                RuntimeError("boom 2"),
            ]),
            write_subtitles=False,
        )

        assert result.success is False
        assert "All 2 ASR segment calls failed" in result.error
        assert result.metadata["drop_counts"] == {"energy_gate:low_rms": 1}

    def test_legacy_empty_vs_structured_engine_empty(self, pipeline, audio_path):
        """legacy 空文字は empty_results / structured REASON_ENGINE_EMPTY は
        drop_counts (別勘定)"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome
        from livecap_cli.transcription.utterance import REASON_ENGINE_EMPTY

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=self._outcomes_transcriber([
                "",                                           # legacy 空
                SegmentOutcome.dropped(REASON_ENGINE_EMPTY),  # structured 空
                "text",                                       # legacy 正常
            ]),
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 3
        assert result.metadata["empty_results"] == 1
        assert result.metadata["drop_counts"] == {"engine:empty_text": 1}
        assert result.metadata["segment_count"] == 1

    def test_str_and_outcome_interchangeable(self, pipeline, audio_path):
        """str と SegmentOutcome の混在受理 (legacy 契約の維持)"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=self._outcomes_transcriber([
                "legacy text",
                SegmentOutcome.success("structured text"),
                "  ",                                         # 空白のみ → empty
            ]),
            write_subtitles=False,
        )

        assert result.success is True
        assert result.metadata["asr_calls"] == 3
        assert [s.text for s in result.subtitles] == ["legacy text", "structured text"]
        assert result.metadata["empty_results"] == 1

    def test_drop_counts_always_in_metadata(self, pipeline, audio_path):
        """drop なしでも drop_counts (空 dict) を常時格納"""
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: "text",
            write_subtitles=False,
        )

        assert result.metadata["drop_counts"] == {}


class TestContractViolations:
    """PR #371 レビュー反映: 契約違反の per-segment failure 化と不変条件の強化"""

    def test_non_str_return_counted_as_asr_error_not_raised(self, pipeline, audio_path):
        """transcriber が非 str/SegmentOutcome を返す → pipeline-level 例外に
        せず #362 の per-segment failure として集計 (全件なら success=False)"""
        result = pipeline.process_file(
            audio_path,
            segment_transcriber=lambda a, s: 123,  # 契約違反
            write_subtitles=False,
        )

        assert result.success is False
        assert result.metadata["asr_calls"] == result.metadata["asr_errors"] == 3
        assert "TypeError" in result.error
        assert "SegmentTranscriber must return" in result.error

    def test_asr_called_false_without_drop_reason_rejected(self):
        """drop_reason なし (成功/空候補) は asr_called=True 必須 —
        字幕を生成しながら asr_calls を増やさない矛盾 outcome を封鎖"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome

        with pytest.raises(ValueError, match="asr_called must be True"):
            SegmentOutcome(text="subtitle", asr_called=False)

    def test_empty_string_drop_reason_rejected(self):
        """drop_reason='' は drop_counts[''] を生むため拒否"""
        from livecap_cli.transcription.file_pipeline import SegmentOutcome

        with pytest.raises(ValueError, match="non-empty"):
            SegmentOutcome(text="", drop_reason="", asr_called=False)
