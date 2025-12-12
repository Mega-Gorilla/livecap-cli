"""Core file transcription pipeline (Phase 0.7 + Phase 6a translation)."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from livecap_core.translation.base import BaseTranslator

try:  # pragma: no cover - optional dependency
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:  # pragma: no cover - exercised when dependency missing
    sf = None  # type: ignore
    _HAS_SOUNDFILE = False

try:  # pragma: no cover - optional dependency
    import ffmpeg  # type: ignore
    _HAS_FFMPEG = True
except ImportError:  # pragma: no cover
    ffmpeg = None  # type: ignore
    _HAS_FFMPEG = False

from livecap_core.resources import FFmpegManager, FFmpegNotFoundError, get_ffmpeg_manager

logger = logging.getLogger(__name__)

# Phase 6a: Translation constants (shared with StreamTranscriber)
MAX_CONTEXT_BUFFER = 100  # Maximum sentences to keep for context


# === Data models & callback types ================================================================


@dataclass(slots=True)
class FileSubtitleSegment:
    """Recognised subtitle content for a time-span within a file."""

    index: int
    start: float
    end: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 6a: Translation fields (optional, backward compatible)
    translated_text: Optional[str] = None
    target_language: Optional[str] = None


@dataclass(slots=True)
class FileTranscriptionProgress:
    """Progress payload emitted while processing a batch of files."""

    current: int
    total: int
    status: str = ""
    context: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class FileProcessingResult:
    """Result produced for each processed file."""

    source_path: Path
    success: bool
    output_path: Optional[Path]
    error: Optional[str] = None
    subtitles: list[FileSubtitleSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[FileTranscriptionProgress], None]
StatusCallback = Callable[[str], None]
FileResultCallback = Callable[[FileProcessingResult], None]
ErrorCallback = Callable[[str, Optional[Exception]], None]
SegmentTranscriber = Callable[[np.ndarray, int], str]
Segmenter = Callable[[np.ndarray, int], List[Tuple[float, float]]]


# === Pipeline implementation ====================================================================


class FileTranscriptionCancelled(Exception):
    """Raised when cancellation is requested during pipeline execution."""


class FileTranscriptionPipeline:
    """
    Core pipeline responsible for orchestrating file-mode transcription.

    Responsibilities handled here:
        * media extraction via FFmpeg (when必要)
        * audio loading & resampling
        * segmentation (via injectable callable)
        * SRT generation
        * progress / status callback wiring

    Responsibilities deliberately excluded (caller supplied):
        * ASR engine lifecycle & inference (`segment_transcriber`)
        * translated status/i18n message generation
        * GUI/Qt integration
    """

    SUPPORTED_AUDIO_EXTENSIONS = {
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".wma",
        ".opus",
    }

    def __init__(
        self,
        *,
        ffmpeg_manager: Optional[FFmpegManager] = None,
        segmenter: Optional[Segmenter] = None,
    ) -> None:
        self._ffmpeg_manager = ffmpeg_manager or get_ffmpeg_manager()
        self._segmenter = segmenter
        self._temp_root = Path(tempfile.mkdtemp(prefix="livecap-file-pipeline-"))
        self._ffmpeg_path: Optional[str] = None
        self._ffprobe_path: Optional[str] = None
        self._initialise_ffmpeg_environment()

    # --------------------------------------------------------------------- public API ------------
    def process_files(
        self,
        file_paths: Sequence[str | Path],
        *,
        segment_transcriber: SegmentTranscriber,
        # Phase 6a: Translation parameters
        translator: Optional[BaseTranslator] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        translation_timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
        status_callback: Optional[StatusCallback] = None,
        result_callback: Optional[FileResultCallback] = None,
        error_callback: Optional[ErrorCallback] = None,
        write_subtitles: bool = True,
        write_translated_subtitles: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[FileProcessingResult]:
        """
        Process multiple files sequentially.

        Args:
            file_paths: files to process.
            segment_transcriber: callable invoked for each speech segment; expected to
                return recognised text (empty string permitted).
            translator: Optional translator for real-time translation.
            source_lang: Source language code (required if translator is set).
            target_lang: Target language code (required if translator is set).
            translation_timeout: Optional timeout for translation (seconds).
            progress_callback: optional progress sink.
            status_callback: textual status updates (caller can translate/relay).
            result_callback: called after each file is processed.
            error_callback: invoked when pipeline level errors occur.
            write_subtitles: when True, write `.srt` alongside source file.
            write_translated_subtitles: when True, write translated `.srt` file.

        Returns:
            List of FileProcessingResult in the same order as `file_paths`.
        """
        results: list[FileProcessingResult] = []
        total = len(file_paths)

        for index, path in enumerate(file_paths):
            file_path = Path(path)
            self._check_cancel(should_cancel)
            if progress_callback:
                progress_callback(
                    FileTranscriptionProgress(
                        current=index,
                        total=total,
                        status="processing",
                        context={"file": str(file_path)},
                    )
                )
            if status_callback:
                status_callback(f"processing:{file_path.name}")

            try:
                result = self.process_file(
                    file_path,
                    segment_transcriber=segment_transcriber,
                    translator=translator,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    translation_timeout=translation_timeout,
                    write_subtitles=write_subtitles,
                    write_translated_subtitles=write_translated_subtitles,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                )
            except FileTranscriptionCancelled:
                raise
            except Exception as exc:  # pragma: no cover - integration path
                logger.error("File transcription failed for %s", file_path, exc_info=True)
                result = FileProcessingResult(
                    source_path=file_path,
                    success=False,
                    output_path=None,
                    error=str(exc),
                )
                if error_callback:
                    error_callback(str(exc), exc)

            results.append(result)
            if result_callback:
                result_callback(result)

            if progress_callback:
                progress_callback(
                    FileTranscriptionProgress(
                        current=index + 1,
                        total=total,
                        status="processed" if result.success else "failed",
                        context={
                            "file": str(file_path),
                            "success": result.success,
                            "error": result.error,
                        },
                    )
                )

        return results

    def process_file(
        self,
        file_path: str | Path,
        *,
        segment_transcriber: SegmentTranscriber,
        # Phase 6a: Translation parameters
        translator: Optional[BaseTranslator] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        translation_timeout: Optional[float] = None,
        write_subtitles: bool = True,
        write_translated_subtitles: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> FileProcessingResult:
        """
        Process a single file and optionally write subtitles.

        Args:
            file_path: Path to the audio/video file.
            segment_transcriber: Callable for ASR transcription.
            translator: Optional translator for real-time translation.
            source_lang: Source language code (required if translator is set).
            target_lang: Target language code (required if translator is set).
            translation_timeout: Optional timeout for translation (seconds).
            write_subtitles: Write original language SRT file.
            write_translated_subtitles: Write translated SRT file.
            progress_callback: Progress update callback.
            should_cancel: Cancellation check callback.

        Returns:
            FileProcessingResult detailing success flag, subtitles, and output path.
        """
        # Validate translator parameters
        self._validate_translator_params(translator, source_lang, target_lang)

        source = Path(file_path)
        self._check_cancel(should_cancel)
        working_audio = self._extract_audio(source)
        try:
            audio_data, sample_rate = self._load_audio(working_audio)
            self._check_cancel(should_cancel)
            segments = self._segment(audio_data, sample_rate)
            subtitles = self._transcribe_segments(
                segments,
                audio_data,
                sample_rate,
                segment_transcriber,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                translation_timeout=translation_timeout,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            )
            output_path = None
            translated_output_path = None
            if write_subtitles:
                output_path = self._write_srt(source, subtitles)
            if write_translated_subtitles and translator:
                translated_output_path = self._write_translated_srt(
                    source, subtitles, target_lang
                )

            metadata = {
                "segment_count": len(subtitles),
                "duration_seconds": float(len(audio_data) / sample_rate),
                "sample_rate": sample_rate,
            }
            if translator:
                metadata["translation_enabled"] = True
                metadata["target_language"] = target_lang
            if translated_output_path:
                metadata["translated_srt_path"] = str(translated_output_path)

            return FileProcessingResult(
                source_path=source,
                success=True,
                output_path=output_path,
                subtitles=subtitles,
                metadata=metadata,
            )
        finally:
            if working_audio != source and working_audio.exists():
                working_audio.unlink(missing_ok=True)

    def close(self) -> None:
        """Cleanup temporary resources."""
        if self._temp_root.exists():
            shutil.rmtree(self._temp_root, ignore_errors=True)

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self.close()

    # ---------------------------------------------------------------- utilities ------------------
    def _initialise_ffmpeg_environment(self) -> None:
        """Resolve ffmpeg/ffprobe paths if available."""
        try:
            executable = self._ffmpeg_manager.configure_environment()
            self._ffmpeg_path = str(executable)
        except FFmpegNotFoundError:
            fallback = shutil.which("ffmpeg")
            if fallback:
                self._ffmpeg_path = fallback
            else:
                logger.info("FFmpeg not found; extraction will be unavailable.")

        try:
            probe = self._ffmpeg_manager.resolve_probe()
            if probe:
                self._ffprobe_path = str(probe)
        except FFmpegNotFoundError:
            self._ffprobe_path = shutil.which("ffprobe")

        if self._ffmpeg_path:
            os.environ.setdefault("FFMPEG_BINARY", self._ffmpeg_path)
        if self._ffprobe_path:
            os.environ.setdefault("FFPROBE_BINARY", self._ffprobe_path)

    def _extract_audio(self, source: Path) -> Path:
        """Extract audio stream when the source is not already an audio file."""
        if source.suffix.lower() in self.SUPPORTED_AUDIO_EXTENSIONS:
            return source

        if not self._ffmpeg_path:
            raise RuntimeError(
                f"FFmpeg executable is required to extract audio from {source}"
            )
        if not _HAS_FFMPEG:
            raise RuntimeError(
                "ffmpeg-python is not installed; install via `pip install ffmpeg-python`."
            )

        destination = self._temp_root / f"{source.stem}_audio.wav"
        stream = ffmpeg.input(str(source))
        stream = ffmpeg.output(
            stream,
            str(destination),
            ac=1,
            ar=16000,
            acodec="pcm_s16le",
        )
        stream = ffmpeg.overwrite_output(stream)

        try:
            ffmpeg.run(
                stream,
                cmd=self._ffmpeg_path,
                capture_stdout=True,
                capture_stderr=True,
            )
        except ffmpeg.Error as exc:  # pragma: no cover - exercised via integration
            stderr = exc.stderr.decode() if exc.stderr else str(exc)
            raise RuntimeError(f"Failed to extract audio: {stderr}") from exc

        return destination

    def _load_audio(self, audio_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        """Load audio file using soundfile (preferred) or librosa fallback."""
        if _HAS_SOUNDFILE:
            try:
                audio, sample_rate = self._load_with_soundfile(audio_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("soundfile failed, falling back to librosa: %s", exc)
                audio, sample_rate = self._load_with_librosa(audio_path, target_sr)
        else:  # pragma: no cover - executed when dependency missing
            audio, sample_rate = self._load_with_librosa(audio_path, target_sr)

        if sample_rate != target_sr:
            audio = self._resample(audio, sample_rate, target_sr)
            sample_rate = target_sr

        return audio.astype(np.float32), sample_rate

    def _load_with_soundfile(self, audio_path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sample_rate

    def _load_with_librosa(
        self, audio_path: Path, target_sr: int
    ) -> tuple[np.ndarray, int]:
        try:
            import librosa  # pragma: no cover - heavy dep, used only when required
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "librosa is required for audio loading fallback. Install via `pip install librosa`."
            ) from exc
        audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        if sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        return audio, sample_rate

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            from scipy import signal  # pragma: no cover - optional dependency
        except ImportError:  # pragma: no cover
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples)

    def _segment(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        if self._segmenter:
            segments = self._segmenter(audio, sample_rate)
            if segments:
                return segments
        duration = len(audio) / sample_rate
        return [(0.0, duration)]

    def _transcribe_segments(
        self,
        segments: Iterable[Tuple[float, float]],
        audio: np.ndarray,
        sample_rate: int,
        transcriber: SegmentTranscriber,
        *,
        translator: Optional[BaseTranslator] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        translation_timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> list[FileSubtitleSegment]:
        segment_list = list(segments)
        subtitles: list[FileSubtitleSegment] = []
        total_segments = len(segment_list) if segment_list else 0

        # Phase 6a: Context buffer for translation (file-scoped)
        context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        for index, (start, end) in enumerate(segment_list, start=1):
            self._check_cancel(should_cancel)
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            segment_audio = audio[start_idx:end_idx]
            if segment_audio.size == 0:
                continue
            try:
                text = transcriber(segment_audio, sample_rate).strip()
            except FileTranscriptionCancelled:
                raise
            except Exception as exc:  # pragma: no cover - delegated to error callback
                logger.error("Segment transcription failed (%s-%s): %s", start, end, exc)
                text = ""
            if not text:
                continue

            # Phase 6a: Translation processing
            translated_text = None
            result_target_lang = None
            if translator and source_lang and target_lang:
                translated_text, result_target_lang = self._translate_text(
                    text=text,
                    translator=translator,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context_buffer=context_buffer,
                    timeout=translation_timeout,
                )
                # Add to context buffer regardless of translation success
                context_buffer.append(text)

            subtitles.append(
                FileSubtitleSegment(
                    index=index,
                    start=start,
                    end=end,
                    text=text,
                    metadata={"duration": end - start},
                    translated_text=translated_text,
                    target_language=result_target_lang,
                )
            )
            if progress_callback:
                progress_callback(
                    FileTranscriptionProgress(
                        current=index,
                        total=total_segments,
                        status="segment",
                        context={"start": start, "end": end},
                    )
                )
        return subtitles

    def _write_srt(
        self,
        source: Path,
        subtitles: list[FileSubtitleSegment],
    ) -> Path:
        output_path = source.with_suffix(".srt")
        content = self._build_srt(subtitles)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return output_path

    def _build_srt(self, subtitles: list[FileSubtitleSegment]) -> str:
        lines: list[str] = []
        for segment in subtitles:
            lines.append(str(segment.index))
            lines.append(f"{self._format_timestamp(segment.start)} --> {self._format_timestamp(segment.end)}")
            lines.append(segment.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(position: float) -> str:
        td = timedelta(seconds=position)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    @staticmethod
    def _check_cancel(should_cancel: Optional[Callable[[], bool]]) -> None:
        if should_cancel and should_cancel():
            raise FileTranscriptionCancelled()

    # ---------------------------------------------------------------- Phase 6a: Translation --------
    @staticmethod
    def _validate_translator_params(
        translator: Optional[BaseTranslator],
        source_lang: Optional[str],
        target_lang: Optional[str],
    ) -> None:
        """Validate translator parameters."""
        if translator is None:
            return

        # Check if translator is initialized
        if not translator.is_initialized():
            raise ValueError(
                "Translator is not initialized. Call load_model() first."
            )

        # Require non-empty language parameters when translator is set
        if not source_lang or not target_lang:
            raise ValueError(
                "source_lang and target_lang are required when translator is set."
            )
        # Also check for whitespace-only strings
        if not source_lang.strip() or not target_lang.strip():
            raise ValueError(
                "source_lang and target_lang cannot be empty or whitespace-only."
            )

        # Warn if language pair may not be supported
        supported_pairs = translator.get_supported_pairs()
        if supported_pairs and (source_lang, target_lang) not in supported_pairs:
            logger.warning(
                "Language pair (%s -> %s) may not be supported by %s",
                source_lang,
                target_lang,
                translator.get_translator_name(),
            )

    def _translate_text(
        self,
        text: str,
        translator: BaseTranslator,
        source_lang: str,
        target_lang: str,
        context_buffer: deque[str],
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Translate text with optional timeout.

        Args:
            text: Text to translate.
            translator: Translator instance.
            source_lang: Source language code.
            target_lang: Target language code.
            context_buffer: Context buffer for translation.
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (translated_text, target_language) or (None, None) on failure.
        """
        try:
            # Get context from buffer
            context_len = translator.default_context_sentences
            context = list(context_buffer)[-context_len:] if context_buffer else None

            # Treat timeout <= 0 as no timeout (invalid value)
            effective_timeout = timeout if timeout is not None and timeout > 0 else None

            if effective_timeout is not None:
                # Use ThreadPoolExecutor for timeout support
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        translator.translate,
                        text,
                        source_lang,
                        target_lang,
                        context,
                    )
                    try:
                        result = future.result(timeout=effective_timeout)
                        return result.text, target_lang
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "Translation timed out after %.1fs for text: %s...",
                            effective_timeout,
                            text[:50],
                        )
                        return None, None
            else:
                # No timeout - direct call
                result = translator.translate(text, source_lang, target_lang, context)
                return result.text, target_lang

        except Exception as exc:
            logger.warning("Translation failed: %s", exc)
            return None, None

    def _write_translated_srt(
        self,
        source: Path,
        subtitles: list[FileSubtitleSegment],
        target_lang: Optional[str],
    ) -> Optional[Path]:
        """Write translated SRT file."""
        # Filter segments with translations
        translated_segments = [s for s in subtitles if s.translated_text]
        if not translated_segments:
            logger.warning("No translated segments to write")
            return None

        # Create output path with language suffix
        suffix = f"_{target_lang}" if target_lang else "_translated"
        output_path = source.with_stem(f"{source.stem}{suffix}").with_suffix(".srt")

        content = self._build_translated_srt(translated_segments)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return output_path

    def _build_translated_srt(self, subtitles: list[FileSubtitleSegment]) -> str:
        """Build SRT content from translated segments."""
        lines: list[str] = []
        for segment in subtitles:
            if segment.translated_text:
                lines.append(str(segment.index))
                lines.append(
                    f"{self._format_timestamp(segment.start)} --> "
                    f"{self._format_timestamp(segment.end)}"
                )
                lines.append(segment.translated_text)
                lines.append("")
        return "\n".join(lines)


__all__ = [
    "FileTranscriptionPipeline",
    "FileTranscriptionProgress",
    "FileProcessingResult",
    "FileSubtitleSegment",
    "FileTranscriptionCancelled",
    "ProgressCallback",
    "StatusCallback",
    "FileResultCallback",
    "ErrorCallback",
    "SegmentTranscriber",
    "Segmenter",
]
