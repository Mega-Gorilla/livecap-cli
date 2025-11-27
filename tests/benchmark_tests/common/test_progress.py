"""Unit tests for ProgressReporter."""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest

from benchmarks.common.progress import EngineProgress, ProgressReporter


class TestProgressReporter:
    """ProgressReporter テスト"""

    def test_init_local_environment(self):
        """ローカル環境での初期化"""
        with mock.patch.dict(os.environ, {}, clear=True):
            reporter = ProgressReporter(
                benchmark_type="asr",
                mode="quick",
                languages=["ja", "en"],
                total_engines=4,
            )
            assert reporter._is_github_actions is False
            assert reporter._step_summary_path is None

    def test_init_github_actions_environment(self):
        """GitHub Actions 環境での初期化"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            summary_path = f.name

        try:
            with mock.patch.dict(
                os.environ,
                {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": summary_path},
            ):
                reporter = ProgressReporter(
                    benchmark_type="asr",
                    mode="full",
                    languages=["ja"],
                    total_engines=10,
                )
                assert reporter._is_github_actions is True
                assert reporter._step_summary_path == summary_path

                # Check that header was written
                with open(summary_path) as f:
                    content = f.read()
                assert "ASR Benchmark Progress" in content
                assert "full" in content
                assert "10" in content
        finally:
            os.unlink(summary_path)

    def test_format_time_seconds(self):
        """時間フォーマット（秒）"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 1)
        assert reporter._format_time(30) == "30s"
        assert reporter._format_time(59) == "59s"

    def test_format_time_minutes(self):
        """時間フォーマット（分）"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 1)
        assert reporter._format_time(60) == "1m0s"
        assert reporter._format_time(90) == "1m30s"
        assert reporter._format_time(3599) == "59m59s"

    def test_format_time_hours(self):
        """時間フォーマット（時間）"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 1)
        assert reporter._format_time(3600) == "1h0m"
        assert reporter._format_time(7200) == "2h0m"
        assert reporter._format_time(5400) == "1h30m"

    def test_engine_started(self, capsys):
        """エンジン開始報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 2)
        reporter.engine_started("parakeet_ja", "ja", 100)

        assert reporter._current_engine is not None
        assert reporter._current_engine.engine_id == "parakeet_ja"
        assert reporter._current_engine.language == "ja"
        assert reporter._current_engine.files_total == 100
        assert reporter._current_engine.status == "running"

    def test_engine_completed(self, capsys):
        """エンジン完了報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 2)
        reporter.engine_started("parakeet_ja", "ja", 100)
        reporter.engine_completed("parakeet_ja", wer=0.15, cer=0.08, rtf=0.05)

        assert reporter._progress.engines_completed == 1
        assert len(reporter._progress.engine_progress) == 1

        progress = reporter._progress.engine_progress[0]
        assert progress.wer == 0.15
        assert progress.cer == 0.08
        assert progress.rtf == 0.05
        assert progress.status == "completed"

    def test_engine_skipped(self, capsys):
        """エンジンスキップ報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 2)
        reporter.engine_skipped("voxtral", "Does not support ja")

        assert reporter._progress.engines_completed == 1

    def test_engine_failed(self, capsys):
        """エンジン失敗報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 2)
        reporter.engine_started("parakeet_ja", "ja", 100)
        reporter.engine_failed("parakeet_ja", "CUDA out of memory")

        assert reporter._progress.engines_completed == 1
        assert reporter._current_engine is None

    def test_file_completed(self):
        """ファイル完了報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 1)
        reporter.engine_started("parakeet_ja", "ja", 100)

        assert reporter._current_engine.files_completed == 0
        reporter.file_completed()
        assert reporter._current_engine.files_completed == 1
        reporter.file_completed()
        assert reporter._current_engine.files_completed == 2

    def test_benchmark_completed(self, capsys):
        """ベンチマーク完了報告"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 2)
        reporter.benchmark_started()
        reporter.engine_started("engine1", "ja", 10)
        reporter.engine_completed("engine1", wer=0.1, cer=0.05, rtf=0.1)
        reporter.engine_started("engine2", "ja", 10)
        reporter.engine_completed("engine2", wer=0.2, cer=0.1, rtf=0.2)
        reporter.benchmark_completed()

        assert reporter._progress.engines_completed == 2

    def test_github_actions_step_summary(self):
        """GitHub Actions Step Summary への書き込み"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            summary_path = f.name

        try:
            with mock.patch.dict(
                os.environ,
                {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": summary_path},
            ):
                reporter = ProgressReporter("asr", "full", ["ja"], 2)
                reporter.benchmark_started()
                reporter.engine_started("parakeet_ja", "ja", 100)
                for _ in range(100):
                    reporter.file_completed()
                reporter.engine_completed("parakeet_ja", wer=0.15, cer=0.08, rtf=0.05)

                # Read and verify summary content
                with open(summary_path) as f:
                    content = f.read()

                # Header should be present
                assert "ASR Benchmark Progress" in content
                # Engine row should be present
                assert "parakeet_ja" in content
                assert "15.0%" in content  # WER
                assert "0.05x" in content  # RTF
                assert "100/100" in content  # files
                assert "✅" in content  # status
        finally:
            os.unlink(summary_path)

    def test_estimate_remaining(self):
        """残り時間推定"""
        reporter = ProgressReporter("asr", "quick", ["ja"], 4)
        reporter.benchmark_started()

        # No completed engines yet
        assert reporter._estimate_remaining() is None

        # After completing 2 engines
        reporter._progress.engines_completed = 2
        remaining = reporter._estimate_remaining()
        assert remaining is not None


class TestEngineProgress:
    """EngineProgress テスト"""

    def test_default_values(self):
        """デフォルト値"""
        progress = EngineProgress(
            engine_id="test_engine",
            language="ja",
            files_total=100,
        )
        assert progress.files_completed == 0
        assert progress.wer is None
        assert progress.cer is None
        assert progress.rtf is None
        assert progress.elapsed_s == 0.0
        assert progress.status == "pending"

    def test_custom_values(self):
        """カスタム値"""
        progress = EngineProgress(
            engine_id="test_engine",
            language="en",
            files_total=50,
            files_completed=25,
            wer=0.15,
            cer=0.08,
            rtf=0.05,
            elapsed_s=120.5,
            status="completed",
        )
        assert progress.files_completed == 25
        assert progress.wer == 0.15
        assert progress.status == "completed"
