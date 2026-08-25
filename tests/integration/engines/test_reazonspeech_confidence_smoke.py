"""ReazonSpeech の `avg_logprob` を実モデルで pin する (Issue #377)。

なぜ転写テキストの一致では足りないのか
------------------------------------
``avg_logprob`` の供給元 ``OfflineRecognitionResult.ys_log_probs`` は
**sherpa-onnx 1.12.39 で expose されるようになった Python 側の result schema**
であり (``reazonspeech_engine.py`` の docstring 参照)、**依存更新で変わり得る**。
schema が消えても転写テキストは正常に出るため、テキスト比較では検出できない。

その場合 confidence filter は ReazonSpeech に対して pass-through へ degrade し、
非音声区間が字幕として出るようになる — #295 の元 motivation そのものに戻る。

なぜ厳密値を固定しないのか
------------------------
量子化 (int8 / float32) とハードウェアで値は動く。ここで守りたいのは
**(1) schema が生きていること**と **(2) clean sample が filter を通ること**の 2 点で、
どちらも閾値との相対関係で表現できる。厳密値を pin すると本質と無関係な理由で
落ちる。

`test_smoke_engines.py::test_token_confidence_populated` の ReazonSpeech 版に
あたる (あちらは NeMo 系の ``token_confidence_mean`` を見ており、
ReazonSpeech の ``avg_logprob`` はどの実モデルテストにも pin されていなかった)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livecap_cli.engines import EngineFactory

# `tests/integration/engines/` は package ではない (``__init__.py`` なし) ため、
# pytest の prepend import mode がこのディレクトリを sys.path へ入れる。相対 import は
# 使えない。
from test_smoke_engines import (  # type: ignore[import-not-found]
    CASES,
    EngineSmokeCase,
    _build_engine_options,
    _cleanup_gpu_memory,
    _guard_gpu,
    _prepare_audio,
    _skip_or_fail,
)

#: `FilterConfig.avg_logprob_thresholds["reazonspeech"]`。
#: #334 PR-4 で Phase 2 report §2.1 Pareto relaxed_B に合わせて -0.2 から緩和した値。
REJECT_THRESHOLD = -0.40

_REAZONSPEECH_CASES = [
    pytest.param(case, marks=pytest.mark.gpu) if case.requires_gpu else pytest.param(case)
    for case in CASES
    if case.engine == "reazonspeech"
]


@pytest.mark.engine_smoke
@pytest.mark.parametrize("case", _REAZONSPEECH_CASES, ids=lambda c: c.id)
def test_avg_logprob_populated(case: EngineSmokeCase, tmp_path: Path) -> None:
    """``engine_confidence.avg_logprob`` が生きており、clean sample が閾値を通ること。"""
    _guard_gpu(case)

    audio_path = _prepare_audio(case, tmp_path)

    try:
        engine = EngineFactory.create_engine(
            engine_type=case.engine,
            device=case.device,
            **_build_engine_options(case),
        )
    except ImportError as exc:
        _skip_or_fail(f"{case.engine} dependencies are missing: {exc}")
    except Exception as exc:
        _skip_or_fail(f"Failed to initialise engine {case.engine}: {exc}")

    try:
        try:
            engine.load_model()
        except Exception as exc:
            _skip_or_fail(f"Model for {case.engine} is unavailable: {exc}")

        # FileTranscriptionPipeline を介さず直接呼ぶ (VAD segment 単位の生 signal)。
        import soundfile as sf

        audio, sample_rate = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = engine.transcribe(audio.astype(np.float32), sample_rate)

        assert result.engine_confidence is not None, (
            "engine_confidence is None — TranscriptionResult の schema が壊れた可能性"
        )

        avg_logprob = result.engine_confidence.avg_logprob
        assert isinstance(avg_logprob, float), (
            f"avg_logprob={avg_logprob!r} (型 {type(avg_logprob).__name__})。"
            "sherpa-onnx の OfflineRecognitionResult.ys_log_probs が取れなくなった"
            "可能性がある — confidence filter が pass-through へ degrade する。"
        )

        assert avg_logprob > REJECT_THRESHOLD, (
            f"{case.id}: avg_logprob={avg_logprob:.4f} <= 閾値 {REJECT_THRESHOLD}。"
            "clean な speech sample が reject 側に来ている — calibration が"
            "依存更新で壊れた可能性がある (#334 PR-4 の Pareto relaxed_B 前提)。"
        )
    finally:
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        _cleanup_gpu_memory()
