"""ASR engine management for benchmarks.

Provides:
- BenchmarkEngineManager: Manages ASR engine creation and caching
- build_engine_options: engine options shared by ASR benchmarks and VAD optimization
"""

from __future__ import annotations

import logging
from typing import Any

from livecap_cli import TranscriptionEngine

__all__ = ["BenchmarkEngineManager", "TranscriptionEngine", "build_engine_options"]

logger = logging.getLogger(__name__)


def build_engine_options(engine_id: str, language: str) -> dict[str, Any]:
    """Engine options for benchmarks / VAD optimization (one definition for both).

    - **language**: passed to every engine whose constructor takes one
      (``EngineInfo.accepts_language`` = multilingual engines). Measuring with the same
      recognition language as the CLI matters: without it Qwen3-ASR runs in auto-detect
      mode (wrapper fallback path) instead of the production path (#470).
    - **whispers2t**: ``large-v3`` (best accuracy) with its built-in VAD disabled, so the
      benchmark measures the ASR itself and VAD optimization does not run two VADs.

    Args:
        engine_id: Engine identifier
        language: Target language code

    Returns:
        Engine options for ``EngineFactory.create_engine``

    Raises:
        ValueError: If ``engine_id`` is not registered
    """
    from livecap_cli.engines.metadata import EngineMetadata

    info = EngineMetadata.get(engine_id)
    if info is None:
        raise ValueError(f"Unknown engine: {engine_id}")

    options: dict[str, Any] = {}
    if info.accepts_language:
        options["language"] = language
    if engine_id == "whispers2t":
        options["model_size"] = "large-v3"
        options["use_vad"] = False
    return options


class BenchmarkEngineManager:
    """Manages ASR engine creation and caching for benchmarks.

    Engines are cached by (engine_id, device, language) to avoid
    repeated model loading during benchmark runs.

    Usage:
        manager = BenchmarkEngineManager()

        # Get or create engine (cached)
        engine = manager.get_engine("reazonspeech", "cuda", "ja")

        # Run transcription
        result = engine.transcribe(audio, sample_rate)
        transcript = result.text

        # Clean up all engines when done
        manager.clear_cache()
    """

    def __init__(self) -> None:
        """Initialize engine manager."""
        self._cache: dict[str, TranscriptionEngine] = {}
        self._model_memory: dict[str, float] = {}  # GPU memory after model load

    def get_engine(
        self,
        engine_id: str,
        device: str = "cuda",
        language: str = "ja",
    ) -> TranscriptionEngine:
        """Get or create an ASR engine.

        Args:
            engine_id: Engine identifier (e.g., 'reazonspeech', 'whispers2t')
            device: Device to use ('cuda' or 'cpu')
            language: Target language code

        Returns:
            TranscriptionEngine instance

        Raises:
            ImportError: If engine dependencies are not available
            ValueError: If engine_id is unknown
        """
        # For unified whispers2t, include model_size in cache key
        cache_key = f"{engine_id}_{device}_{language}"

        if cache_key not in self._cache:
            logger.info(f"Creating engine: {engine_id} (device={device}, language={language})")
            engine = self._create_engine(engine_id, device, language)
            self._cache[cache_key] = engine

        return self._cache[cache_key]

    def get_model_memory(self, engine_id: str, device: str, language: str) -> float | None:
        """Get GPU memory used by model after loading.

        Args:
            engine_id: Engine identifier
            device: Device used
            language: Language code

        Returns:
            GPU memory in MB, or None if not measured
        """
        cache_key = f"{engine_id}_{device}_{language}"
        return self._model_memory.get(cache_key)

    def _create_engine(
        self,
        engine_id: str,
        device: str,
        language: str,
    ) -> TranscriptionEngine:
        """Create a new ASR engine instance.

        Args:
            engine_id: Engine identifier
            device: Device to use
            language: Target language

        Returns:
            Engine instance
        """
        # Import here to avoid circular imports
        from livecap_cli.engines.engine_factory import EngineFactory

        # Build engine options (also verifies the engine exists)
        engine_options = build_engine_options(engine_id, language)

        # Clear existing engines to ensure accurate VRAM measurement
        # This prevents VRAM accumulation when measuring model memory
        if self._cache:
            logger.debug(f"Clearing {len(self._cache)} cached engines before loading new engine")
            self._clear_cache_keep_memory()

        # Measure GPU memory before/after load
        gpu_tracker = None
        try:
            from .metrics import GPUMemoryTracker
            gpu_tracker = GPUMemoryTracker()
            if gpu_tracker.available:
                gpu_tracker.reset_peak()
        except ImportError:
            pass

        # Create and load engine
        engine = EngineFactory.create_engine(
            engine_type=engine_id,
            device=device,
            **engine_options,
        )
        engine.load_model()

        # Record model memory
        if gpu_tracker and gpu_tracker.available:
            cache_key = f"{engine_id}_{device}_{language}"
            self._model_memory[cache_key] = gpu_tracker.get_allocated() or 0.0

        logger.info(f"Engine loaded: {engine.get_engine_name()}")
        return engine

    def unload_engine(
        self,
        engine_id: str,
        device: str = "cuda",
        language: str = "ja",
    ) -> bool:
        """Unload a specific engine and release its GPU memory.

        Args:
            engine_id: Engine identifier
            device: Device used
            language: Language code

        Returns:
            True if engine was unloaded, False if not found
        """
        cache_key = f"{engine_id}_{device}_{language}"

        if cache_key not in self._cache:
            return False

        engine = self._cache[cache_key]
        try:
            cleanup = getattr(engine, "cleanup", None)
            if callable(cleanup):
                cleanup()
            logger.debug(f"Unloaded engine: {cache_key}")
        except Exception as e:
            logger.warning(f"Error unloading {cache_key}: {e}")

        del self._cache[cache_key]
        # Keep model_memory for reporting purposes
        return True

    def _clear_cache_keep_memory(self) -> None:
        """Clear engine cache but keep memory measurements for reporting."""
        for cache_key, engine in self._cache.items():
            try:
                cleanup = getattr(engine, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                logger.debug(f"Cleaned up: {cache_key}")
            except Exception as e:
                logger.warning(f"Error cleaning up {cache_key}: {e}")

        self._cache.clear()
        # Note: _model_memory is intentionally kept for reporting

    def clear_cache(self) -> None:
        """Clean up and release all cached engines."""
        logger.info(f"Clearing {len(self._cache)} cached engines")

        for cache_key, engine in self._cache.items():
            try:
                cleanup = getattr(engine, "cleanup", None)
                if callable(cleanup):
                    cleanup()
                logger.debug(f"Cleaned up: {cache_key}")
            except Exception as e:
                logger.warning(f"Error cleaning up {cache_key}: {e}")

        self._cache.clear()
        self._model_memory.clear()

    def get_cached_engines(self) -> list[str]:
        """Get list of currently cached engine keys."""
        return list(self._cache.keys())

    @staticmethod
    def get_engines_for_language(language: str) -> list[str]:
        """Get list of engine IDs that support a language.

        Args:
            language: Language code

        Returns:
            List of engine IDs
        """
        from livecap_cli.engines.metadata import EngineMetadata
        return EngineMetadata.get_engines_for_language(language)

    @staticmethod
    def get_all_engines() -> dict[str, Any]:
        """Get information about all available engines.

        Returns:
            Dictionary mapping engine ID to EngineInfo
        """
        from livecap_cli.engines.metadata import EngineMetadata
        return EngineMetadata.get_all()
