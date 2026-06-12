from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_ROOT = ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import gc

from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.transcription import FileTranscriptionPipeline
from utils.text_normalization import normalize_text

pytestmark = pytest.mark.engine_smoke

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "audio"


def _cleanup_gpu_memory() -> None:
    """Force GPU memory cleanup to prevent VRAM accumulation between tests."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


GPU_ENABLED = os.getenv("LIVECAP_ENABLE_GPU_SMOKE") == "1"
STRICT = os.getenv("LIVECAP_REQUIRE_ENGINE_SMOKE") == "1"

KEYWORD_HINTS: dict[str, dict[str, list[str]]] = {
    "en/librispeech_1089-134686-0001": {
        "en": ["stuff", "belly"],
    },
    "ja/jsut_basic5000_0001": {
        "ja": ["水をマレーシアから買わなくてはならない"],
    },
}


@dataclass(frozen=True)
class EngineSmokeCase:
    id: str
    engine: str
    language: str
    audio_stem: str
    device: str | None
    requires_gpu: bool = False
    min_vram_gb: float | None = None  # Minimum VRAM required in GB
    model_size: str | None = None  # Model size for WhisperS2T


@dataclass(frozen=True)
class ModelCacheStatus:
    path: Path
    cached: bool


CASES: list[EngineSmokeCase] = [
    # ==========================================================================
    # CPU Tests (GitHub-hosted runners)
    # ==========================================================================
    # ReazonSpeech on CPU is disabled due to sherpa-onnx/onnxruntime ABI issues on hosted runners.
    # See PR #34 for details. It is tested on GPU self-hosted runners instead.
    EngineSmokeCase(
        id="whispers2t_cpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cpu",
        model_size="base",
    ),
    # ==========================================================================
    # GPU Tests - Japanese Engines (self-hosted runners)
    # ==========================================================================
    EngineSmokeCase(
        id="reazonspeech_gpu_ja",
        engine="reazonspeech",
        language="ja",
        audio_stem="ja/jsut_basic5000_0001",
        device="cuda",
        requires_gpu=True,
    ),
    EngineSmokeCase(
        id="parakeet_ja_gpu_ja",
        engine="parakeet_ja",
        language="ja",
        audio_stem="ja/jsut_basic5000_0001",
        device="cuda",
        requires_gpu=True,
    ),
    # Note: whispers2t_base_gpu_ja removed - WhisperS2T Base has low accuracy for Japanese
    # compared to dedicated Japanese engines (ReazonSpeech, Parakeet JA)
    # ==========================================================================
    # GPU Tests - English Engines (self-hosted runners)
    # ==========================================================================
    EngineSmokeCase(
        id="whispers2t_base_gpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        model_size="base",
    ),
    EngineSmokeCase(
        id="parakeet_gpu_en",
        engine="parakeet",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
    ),
    EngineSmokeCase(
        id="canary_gpu_en",
        engine="canary",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
    ),
    EngineSmokeCase(
        id="voxtral_gpu_en",
        engine="voxtral",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        min_vram_gb=16,  # Voxtral needs ~9.5GB, but with other cached models ~16GB required
    ),
    # ==========================================================================
    # GPU Tests - WhisperS2T Variants (self-hosted runners)
    # ==========================================================================
    EngineSmokeCase(
        id="whispers2t_tiny_gpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        model_size="tiny",
    ),
    EngineSmokeCase(
        id="whispers2t_small_gpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        model_size="small",
    ),
    EngineSmokeCase(
        id="whispers2t_medium_gpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        model_size="medium",
    ),
    EngineSmokeCase(
        id="whispers2t_large_v3_gpu_en",
        engine="whispers2t",
        language="en",
        audio_stem="en/librispeech_1089-134686-0001",
        device="cuda",
        requires_gpu=True,
        model_size="large-v3",
        # min_vram_gb removed - testing if GPU memory cleanup resolves OOM
    ),
]

PARAM_CASES = [
    pytest.param(case, marks=pytest.mark.gpu) if case.requires_gpu else pytest.param(case)
    for case in CASES
]


def _skip_or_fail(reason: str) -> None:
    if STRICT:
        pytest.fail(reason)
    pytest.skip(reason)


def _prepare_audio(case: EngineSmokeCase, tmp_path: Path) -> Path:
    source = ASSETS_ROOT / f"{case.audio_stem}.wav"
    if not source.exists():
        pytest.fail(f"Audio fixture missing: {source}")
    destination = tmp_path / source.name
    shutil.copy2(source, destination)
    return destination


def _load_expected(case: EngineSmokeCase) -> str:
    expected_path = ASSETS_ROOT / f"{case.audio_stem}.txt"
    if not expected_path.exists():
        pytest.fail(f"Expected transcript missing: {expected_path}")
    return expected_path.read_text(encoding="utf-8")


def _guard_gpu(case: EngineSmokeCase) -> None:
    if not case.requires_gpu:
        return
    if not GPU_ENABLED:
        _skip_or_fail("GPU smoke tests disabled (set LIVECAP_ENABLE_GPU_SMOKE=1 to enable).")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        _skip_or_fail(f"torch is required for GPU smoke tests: {exc}")
        return
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        if case.engine == "reazonspeech":
            # Allow CPU fallback for ReazonSpeech on GPU runners without CUDA (e.g. Windows CI)
            return
        _skip_or_fail("CUDA is not available on this runner.")
    # Check VRAM requirement
    if case.min_vram_gb is not None:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < case.min_vram_gb:
            _skip_or_fail(
                f"Insufficient VRAM: {total_vram_gb:.1f}GB available, "
                f"{case.min_vram_gb}GB required for {case.engine}"
            )


def _build_engine_options(case: EngineSmokeCase) -> dict:
    """Build engine options for the test case."""
    options = {}
    # Multi-language engines need language parameter
    if case.engine in ("whispers2t", "canary", "voxtral"):
        options["language"] = case.language
    # WhisperS2T model_size parameter
    if case.model_size is not None:
        options["model_size"] = case.model_size
    return options


def _model_cache_status(engine) -> ModelCacheStatus | None:
    manager = getattr(engine, "model_manager", None)
    get_path = getattr(engine, "_get_local_model_path", None)
    verifier = getattr(engine, "_verify_model_integrity", None)
    if not manager or not get_path or not verifier:
        return None

    try:
        models_dir = manager.get_models_dir(engine.engine_name)
        local_path = get_path(models_dir)
        cached = bool(verifier(local_path))
        return ModelCacheStatus(path=local_path, cached=cached)
    except Exception:
        return None


def _build_transcriber(engine):
    def _transcribe(audio: np.ndarray, sample_rate: int) -> str:
        result = engine.transcribe(audio, sample_rate)
        return result.text

    return _transcribe


def _assert_transcript_matches(observed: str, expected: str, lang: str, case: EngineSmokeCase) -> None:
    observed_norm = normalize_text(observed, lang=lang)
    expected_norm = normalize_text(expected, lang=lang)
    keyword_hints = KEYWORD_HINTS.get(case.audio_stem, {}).get(lang)

    if keyword_hints:
        missing_keywords = [
            kw for kw in keyword_hints if normalize_text(kw, lang=lang) not in observed_norm
        ]
        assert not missing_keywords, f"Missing keyword(s) {missing_keywords} in '{observed_norm}'"
        return

    if lang == "en":
        missing = [token for token in expected_norm.split() if token not in observed_norm]
        assert not missing, f"Missing tokens {missing} in observed transcript: '{observed_norm}'"
    else:
        assert expected_norm in observed_norm, f"Expected '{expected_norm}' to appear in '{observed_norm}'"


@pytest.mark.parametrize("case", PARAM_CASES, ids=lambda c: c.id)
def test_engine_smoke_with_real_audio(case: EngineSmokeCase, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    _guard_gpu(case)

    audio_path = _prepare_audio(case, tmp_path)
    expected_text = _load_expected(case)
    engine_options = _build_engine_options(case)

    # Determine actual device (fallback to cpu if cuda requested but unavailable for reazonspeech)
    device = case.device
    import torch
    if case.engine == "reazonspeech" and device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    try:
        engine = EngineFactory.create_engine(
            engine_type=case.engine,
            device=device,
            **engine_options,
        )
    except ImportError as exc:
        _skip_or_fail(f"{case.engine} dependencies are missing: {exc}")
    except Exception as exc:
        _skip_or_fail(f"Failed to initialise engine {case.engine}: {exc}")

    # Note: Cache check removed - some engines (canary, voxtral) use HuggingFace cache
    # instead of models_dir, so _model_cache_status returns false negatives.
    # Let load_model() fail naturally if the model is truly unavailable.

    pipeline = FileTranscriptionPipeline()

    try:
        try:
            engine.load_model()
        except Exception as exc:
            _skip_or_fail(f"Model for {case.engine} is unavailable or failed to load: {exc}")

        result = pipeline.process_file(
            audio_path,
            segment_transcriber=_build_transcriber(engine),
            write_subtitles=False,
        )
    finally:
        pipeline.close()
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        # Force GPU memory cleanup to prevent VRAM accumulation between tests
        _cleanup_gpu_memory()

    assert result.success, f"Engine {case.engine} failed: {result.error}"
    transcript = " ".join(segment.text for segment in result.subtitles)
    assert transcript, "Engine returned an empty transcript"

    _assert_transcript_matches(transcript, expected_text, case.language, case)


# Issue #321 PR #2 merge gate: NeMo fallback chain 削除後、Canary / Parakeet
# (ja/en) で token_confidence_mean が populate されることを実機 verify。
# Silent degradation 検出 (Path 2/3 fallback を削除したため、Path 1/1.5 が
# 失敗するなら hard fail させて token_confidence なしの状態を見落とさない)。
_CONFIDENCE_POPULATE_CASES = [
    case for case in CASES if case.engine in ("canary", "parakeet", "parakeet_ja")
]


@pytest.mark.parametrize(
    "case", _CONFIDENCE_POPULATE_CASES, ids=lambda c: c.id
)
def test_token_confidence_populated(case: EngineSmokeCase, tmp_path: Path) -> None:
    """NeMo `change_decoding_strategy` (Path 1 / Path 1.5) が成功し、
    `engine_confidence.token_confidence_mean` が populate されることを pin。

    本 test は Issue #321 PR #2 で Canary / Parakeet の fallback chain
    (Path 2/3 + `return_hypotheses` TypeError fallback) を削除した際の
    merge gate。populate が落ちると confidence filter が pass-through に
    degrade するため、実機 GPU で必ず検証する。
    """
    _guard_gpu(case)

    audio_path = _prepare_audio(case, tmp_path)
    engine_options = _build_engine_options(case)

    try:
        engine = EngineFactory.create_engine(
            engine_type=case.engine,
            device=case.device,
            **engine_options,
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

        # FileTranscriptionPipeline を介さず engine.transcribe() を直接呼ぶ
        # (engine_confidence は VAD segment 単位の生 signal を見たいため)。
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = engine.transcribe(audio.astype(np.float32), sr)

        assert result.engine_confidence is not None, (
            f"{case.engine}: engine_confidence is None — TranscriptionResult "
            "の schema が壊れた可能性"
        )
        token_conf = result.engine_confidence.token_confidence_mean
        assert token_conf is not None, (
            f"{case.engine}: token_confidence_mean not populated. "
            "Issue #321 PR #2 fallback removal may have caused silent "
            "degradation — confidence filter は pass-through に degrade する。"
        )
        assert token_conf > 0.0, (
            f"{case.engine}: token_confidence_mean={token_conf} expected > 0.0 "
            "for real speech audio"
        )
    finally:
        cleanup = getattr(engine, "cleanup", None)
        if callable(cleanup):
            cleanup()
        _cleanup_gpu_memory()
