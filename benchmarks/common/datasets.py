"""Dataset management for benchmarks.

Provides:
- AudioFile: Represents a single audio file with transcript
- Dataset: Collection of audio files for a language
- DatasetManager: Manages dataset loading based on mode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# Optional imports
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


__all__ = ["AudioFile", "Dataset", "DatasetManager", "DatasetError"]

logger = logging.getLogger(__name__)


# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"
PREPARED_DIR = ASSETS_DIR / "prepared"


class DatasetError(Exception):
    """Error related to dataset operations."""
    pass


@dataclass
class AudioFile:
    """Represents a single audio file for benchmarking.

    Audio data is loaded lazily on first access to save memory.
    """

    path: Path
    transcript_path: Path
    language: str
    _audio: np.ndarray | None = field(default=None, repr=False)
    _sample_rate: int | None = field(default=None, repr=False)
    _transcript: str | None = field(default=None, repr=False)

    @property
    def stem(self) -> str:
        """Get the file stem (filename without extension)."""
        return self.path.stem

    @property
    def audio(self) -> np.ndarray:
        """Get audio data (lazy loaded)."""
        if self._audio is None:
            self._load_audio()
        return self._audio  # type: ignore

    @property
    def sample_rate(self) -> int:
        """Get sample rate (lazy loaded with audio)."""
        if self._sample_rate is None:
            self._load_audio()
        return self._sample_rate  # type: ignore

    @property
    def transcript(self) -> str:
        """Get transcript text (lazy loaded)."""
        if self._transcript is None:
            self._load_transcript()
        return self._transcript  # type: ignore

    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        return len(self.audio) / self.sample_rate

    def _load_audio(self) -> None:
        """Load audio data from file."""
        if not SOUNDFILE_AVAILABLE:
            raise ImportError("soundfile is required. Install with: pip install soundfile")

        if not self.path.exists():
            raise DatasetError(f"Audio file not found: {self.path}")

        self._audio, self._sample_rate = sf.read(self.path)

        # Convert to mono if stereo
        if self._audio.ndim > 1:
            self._audio = self._audio.mean(axis=1)

    def _load_transcript(self) -> None:
        """Load transcript from file."""
        if not self.transcript_path.exists():
            raise DatasetError(f"Transcript file not found: {self.transcript_path}")

        self._transcript = self.transcript_path.read_text(encoding="utf-8").strip()

    def unload(self) -> None:
        """Unload audio data to free memory."""
        self._audio = None
        self._sample_rate = None


@dataclass
class Dataset:
    """Collection of audio files for benchmarking."""

    language: str
    files: list[AudioFile] = field(default_factory=list)
    source_dir: Path | None = None

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[AudioFile]:
        return iter(self.files)

    def get_files_for_engine(self, engine_id: str) -> Iterator[AudioFile]:
        """Get files compatible with a specific engine.

        Checks if the engine supports this dataset's language. If not,
        logs a debug message and yields nothing.

        Args:
            engine_id: Engine identifier

        Yields:
            AudioFile objects (if engine supports the language)
        """
        from engines.metadata import EngineMetadata

        info = EngineMetadata.get(engine_id)
        if info is None:
            logger.warning(f"Unknown engine: {engine_id}, returning all files")
            yield from self.files
            return

        supported_languages = info.supported_languages
        if supported_languages and self.language not in supported_languages:
            logger.debug(
                f"Skipping dataset {self.language}: "
                f"{engine_id} only supports {supported_languages}"
            )
            return

        yield from self.files

    def is_compatible_with_engine(self, engine_id: str) -> bool:
        """Check if this dataset is compatible with an engine.

        Args:
            engine_id: Engine identifier

        Returns:
            True if engine supports this dataset's language
        """
        from engines.metadata import EngineMetadata

        info = EngineMetadata.get(engine_id)
        if info is None:
            return True  # Unknown engine, assume compatible

        supported_languages = info.supported_languages
        if not supported_languages:
            return True  # No language restriction

        return self.language in supported_languages


class DatasetManager:
    """Manages dataset loading based on execution mode.

    Modes:
    - "debug": Use audio/ directory (git-tracked, 1 file/language) - for CI smoke tests
    - "quick": Use prepared/ directory (30 files/language) - for fast benchmarks
    - "standard": Use prepared/ directory (100 files/language) - for standard evaluation
    - "full": Use prepared/ directory (all files) - for complete evaluation
    - "auto": Use prepared/ if exists, otherwise audio/
    """

    def __init__(
        self,
        audio_dir: Path | None = None,
        prepared_dir: Path | None = None,
    ) -> None:
        """Initialize dataset manager.

        Args:
            audio_dir: Path to audio/ directory (default: tests/assets/audio)
            prepared_dir: Path to prepared/ directory (default: tests/assets/prepared)
        """
        self.audio_dir = audio_dir or AUDIO_DIR
        self.prepared_dir = prepared_dir or PREPARED_DIR

    def get_dataset(
        self,
        language: str,
        mode: str = "auto",
        limit: int | None = None,
    ) -> Dataset:
        """Get dataset for a specific language.

        Args:
            language: Language code ('ja', 'en', etc.)
            mode: Execution mode ('debug', 'quick', 'standard', 'full', 'auto')
            limit: Maximum number of files (overrides mode default)

        Returns:
            Dataset object

        Raises:
            DatasetError: If required directory doesn't exist
        """
        if mode == "debug":
            # Debug mode: use audio/ directory (1 file, for CI smoke tests)
            return self._load_from_audio(language, limit)
        elif mode == "quick":
            # Quick mode: use prepared/ with 30 files (for fast benchmarks/optimization)
            return self._load_from_prepared(language, limit or 30)
        elif mode == "standard":
            # Standard mode: use prepared/ with 100 files
            return self._load_from_prepared(language, limit or 100)
        elif mode == "full":
            # Full mode: use prepared/ with all files
            return self._load_from_prepared(language, limit)
        elif mode == "auto":
            if self._prepared_exists(language):
                return self._load_from_prepared(language, limit)
            else:
                return self._load_from_audio(language, limit)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_datasets(
        self,
        languages: list[str] | None = None,
        mode: str = "auto",
        limit: int | None = None,
    ) -> dict[str, Dataset]:
        """Get datasets for multiple languages.

        Args:
            languages: List of language codes (default: ['ja', 'en'])
            mode: Execution mode
            limit: Maximum number of files per language

        Returns:
            Dictionary mapping language code to Dataset
        """
        if languages is None:
            languages = ["ja", "en"]

        return {lang: self.get_dataset(lang, mode, limit) for lang in languages}

    def _prepared_exists(self, language: str) -> bool:
        """Check if prepared/ directory exists for language."""
        lang_dir = self.prepared_dir / language
        return lang_dir.exists() and any(lang_dir.glob("*.wav"))

    def _load_from_audio(self, language: str, limit: int | None) -> Dataset:
        """Load dataset from audio/ directory."""
        lang_dir = self.audio_dir / language
        if not lang_dir.exists():
            raise DatasetError(
                f"Audio directory not found: {lang_dir}\n"
                f"Expected structure: tests/assets/audio/{language}/"
            )

        files = self._scan_directory(lang_dir, language, limit)
        return Dataset(language=language, files=files, source_dir=lang_dir)

    def _load_from_prepared(self, language: str, limit: int | None) -> Dataset:
        """Load dataset from prepared/ directory."""
        lang_dir = self.prepared_dir / language
        if not lang_dir.exists():
            raise DatasetError(
                f"Prepared directory not found: {lang_dir}\n"
                f"Run: python scripts/prepare_benchmark_data.py --mode standard\n"
                f"to generate benchmark data."
            )

        files = self._scan_directory(lang_dir, language, limit)
        if not files:
            raise DatasetError(
                f"No audio files found in: {lang_dir}\n"
                f"Run: python scripts/prepare_benchmark_data.py --mode standard"
            )

        return Dataset(language=language, files=files, source_dir=lang_dir)

    def _scan_directory(
        self,
        directory: Path,
        language: str,
        limit: int | None,
    ) -> list[AudioFile]:
        """Scan directory for audio files with transcripts.

        Args:
            directory: Directory to scan
            language: Language code
            limit: Maximum number of files

        Returns:
            List of AudioFile objects
        """
        files: list[AudioFile] = []

        # Find all WAV files and sort naturally
        wav_files = sorted(directory.glob("*.wav"), key=lambda p: p.stem)

        for wav_path in wav_files:
            if limit is not None and len(files) >= limit:
                break

            # Look for matching transcript
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                logger.warning(f"No transcript for: {wav_path.name}")
                continue

            files.append(AudioFile(
                path=wav_path,
                transcript_path=txt_path,
                language=language,
            ))

        return files
