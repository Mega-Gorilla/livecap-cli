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

from engines.engine_factory import EngineFactory
from livecap_core.config.defaults import get_default_config
from livecap_core.transcription import FileTranscriptionPipeline
from utils.text_normalization import normalize_text

pytestmark = pytest.mark.engine_smoke

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "audio"
GPU_ENABLED = os.getenv("LIVECAP_ENABLE_GPU_SMOKE") == "1"
STRICT = os.getenv("LIVECAP_REQUIRE_ENGINE_SMOKE") == "1"

KEYWORD_HINTS: dict[str, dict[str, list[str]]] = {
    "librispeech_test-clean_1089-134686-0001_en": {
        "en": ["stuff", "belly"],
    },
    "jsut_basic5000_0001_ja": {
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


@dataclass(frozen=True)
class ModelCacheStatus:
    path: Path
    cached: bool


CASES: list[EngineSmokeCase] = [
    EngineSmokeCase(
        id="reazonspeech_cpu_ja",
        engine="reazonspeech",
        language="ja",
        audio_stem="jsut_basic5000_0001_ja",
        device="cpu",
    ),
    EngineSmokeCase(
        id="whispers2t_cpu_en",
        engine="whispers2t_base",
        language="en",
        audio_stem="librispeech_test-clean_1089-134686-0001_en",
        device="cpu",
    ),
    EngineSmokeCase(
        id="whispers2t_gpu_en",
        engine="whispers2t_base",
        language="en",
        audio_stem="librispeech_test-clean_1089-134686-0001_en",
        device="cuda",
        requires_gpu=True,
    ),
    EngineSmokeCase(
        id="parakeet_gpu_en",
        engine="parakeet",
        language="en",
        audio_stem="librispeech_test-clean_1089-134686-0001_en",
        device="cuda",
        requires_gpu=True,
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
        _skip_or_fail("CUDA is not available on this runner.")


def _build_config(case: EngineSmokeCase) -> dict:
    config = get_default_config()
    transcription = config.get("transcription", {})
    transcription["engine"] = case.engine
    transcription["input_language"] = case.language
    config["transcription"] = transcription
    return config


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
        text, _confidence = engine.transcribe(audio, sample_rate)
        return text

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
    config = _build_config(case)

    try:
        engine = EngineFactory.create_engine(
            engine_type=case.engine,
            device=case.device,
            config=config,
        )
    except ImportError as exc:
        _skip_or_fail(f"{case.engine} dependencies are missing: {exc}")
    except Exception as exc:
        _skip_or_fail(f"Failed to initialise engine {case.engine}: {exc}")

    cache_status = _model_cache_status(engine)
    if cache_status and not cache_status.cached:
        _skip_or_fail(f"Model cache missing for {case.engine}: {cache_status.path}")

    pipeline = FileTranscriptionPipeline(config=config)

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

    assert result.success, f"Engine {case.engine} failed: {result.error}"
    transcript = " ".join(segment.text for segment in result.subtitles)
    assert transcript, "Engine returned an empty transcript"

    _assert_transcript_matches(transcript, expected_text, case.language, case)
