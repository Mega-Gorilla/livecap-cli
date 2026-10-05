"""VADOptimizer が ASR ベンチマークと同じ engine options で engine を作ること。

Regression tests for #265 (language / whispers2t options) and #470 (Qwen3-ASR に言語が
渡らず自動言語検出のまま最適化されていた)。options の中身は
``tests/benchmark_tests/common/test_engines.py`` (``build_engine_options``) で固定する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("optuna", reason="optuna not installed")

from benchmarks.common.engines import build_engine_options
from benchmarks.optimization.vad_optimizer import VADOptimizer


@pytest.mark.parametrize(
    "engine_id, language",
    [
        ("qwen3asr_large", "ja"),
        ("qwen3asr", "en"),
        ("whispers2t", "en"),
        ("canary", "en"),
        ("parakeet_ja", "ja"),
    ],
)
def test_engine_is_created_with_the_shared_benchmark_options(engine_id, language):
    opt = VADOptimizer(vad_type="silero", language=language, engine_id=engine_id, device="cpu")
    engine = MagicMock()

    with patch("benchmarks.optimization.vad_optimizer.EngineFactory.create_engine", return_value=engine) as create:
        assert opt.engine is engine

    create.assert_called_once_with(engine_id, device="cpu", **build_engine_options(engine_id, language))
    engine.load_model.assert_called_once()


def test_qwen3asr_large_is_optimized_with_the_recognition_language():
    """#470: Qwen3-ASR は言語指定 (CLI と同じ scores 経路) で最適化する。"""
    opt = VADOptimizer(vad_type="silero", language="ja", engine_id="qwen3asr_large", device="cpu")

    with patch("benchmarks.optimization.vad_optimizer.EngineFactory.create_engine", return_value=MagicMock()) as create:
        _ = opt.engine

    assert create.call_args.kwargs["language"] == "ja"
