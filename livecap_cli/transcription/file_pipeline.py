"""Core file transcription pipeline (Phase 0.7 + Phase 6a translation)."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from livecap_cli.translation.base import BaseTranslator

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

from livecap_cli.resources import FFmpegManager, FFmpegNotFoundError, get_ffmpeg_manager
from livecap_cli.transcription.stream import drain_translation
from livecap_cli.translation.retry import FILE_RETRY_POLICY
from livecap_cli.translation.retry import for_translator as retry_for_translator
from livecap_cli.transcription.srt import write_srt
from livecap_cli.transcription.utterance import (
    REASON_ENERGY_GATE,
    REASON_ENGINE_EMPTY,
    REASON_FILTER_REJECT,
)

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


@dataclass(slots=True, frozen=True)
class SegmentOutcome:
    """`SegmentTranscriber` の structured 返り値 (Issue #366 Phase 2)。

    caller (CLI / GUI) 側の filter 判定結果を pipeline へ運ぶ。legacy の
    `str` 返却も引き続き受理される (正規化規則は `_transcribe_segments`
    参照)。

    不変条件 (`__post_init__` で fail-fast):

    - ``drop_reason`` 付きは ``text == ""`` 必須
    - ``REASON_ENERGY_GATE`` は ASR 前に弾く gate のため ``asr_called=False``
    - ``REASON_FILTER_REJECT`` / ``REASON_ENGINE_EMPTY`` は ASR 後のため
      ``asr_called=True``
    - 未知の drop_reason は許容 (metadata の ``drop_counts`` へ、前方互換)
    """

    text: str
    drop_reason: Optional[str] = None   # utterance.py の REASON_* 語彙
    asr_called: bool = True             # engine.transcribe() を実際に呼んだか

    def __post_init__(self) -> None:
        # 型不変条件 (PR #371 レビュー): drop_reason の非 str は
        # drop_counts.get() の unhashable エラーとして pipeline-level に
        # 漏れる / 非 str key が dict[str, int] 契約を壊すため構築時に拒否。
        # transcriber 内での構築失敗は per-segment try が #362 どおり集計する。
        if not isinstance(self.text, str):
            raise TypeError(
                f"text must be str, got {type(self.text).__name__}"
            )
        if self.drop_reason is not None and not isinstance(self.drop_reason, str):
            raise TypeError(
                f"drop_reason must be str or None, got {type(self.drop_reason).__name__}"
            )
        if not isinstance(self.asr_called, bool):
            raise TypeError(
                f"asr_called must be bool, got {type(self.asr_called).__name__}"
            )
        if self.drop_reason == "":
            raise ValueError(
                "drop_reason must be a non-empty reason string or None "
                "(empty string would pollute drop_counts)"
            )
        if self.drop_reason is None and not self.asr_called:
            raise ValueError(
                "SegmentOutcome without drop_reason is a subtitle/empty "
                "candidate - asr_called must be True (pre-ASR drops must "
                "carry a non-empty drop_reason)"
            )
        if self.drop_reason is not None and self.text != "":
            raise ValueError(
                "SegmentOutcome with drop_reason must have text='' "
                f"(got drop_reason={self.drop_reason!r}, text={self.text!r})"
            )
        if self.drop_reason == REASON_ENERGY_GATE and self.asr_called:
            raise ValueError(
                "REASON_ENERGY_GATE drops happen before ASR - "
                "asr_called must be False"
            )
        if (
            self.drop_reason in (REASON_FILTER_REJECT, REASON_ENGINE_EMPTY)
            and not self.asr_called
        ):
            raise ValueError(
                f"{self.drop_reason} drops happen after ASR - "
                "asr_called must be True"
            )

    @classmethod
    def success(cls, text: str) -> "SegmentOutcome":
        return cls(text=text)

    @classmethod
    def dropped(cls, reason: str, *, asr_called: bool = True) -> "SegmentOutcome":
        return cls(text="", drop_reason=reason, asr_called=asr_called)


@dataclass(slots=True)
class _SegmentTranscriptionOutcome:
    """Aggregated result of `_transcribe_segments` (Issue #362 / #366).

    Tracks per-segment ASR outcomes so `process_file` can distinguish
    total failure (every ASR call raised) from partial failure and from
    legitimately-empty recognition.
    """

    subtitles: list[FileSubtitleSegment]
    asr_calls: int = 0        # engine を実際に呼んだ数 (SegmentOutcome.asr_called 基準)
    asr_errors: int = 0       # transcriber raised (Cancelled excluded, re-raised)
    empty_results: int = 0    # 明示 drop_reason なしで text=="" (legacy 契約を維持)
    first_error: Optional[Exception] = None
    drop_counts: Dict[str, int] = field(default_factory=dict)
    # drop_reason 別の件数 (#366 Phase 2 の統計正本)。reason 追加時に
    # field を増やさず dict key で拡張する。


ProgressCallback = Callable[[FileTranscriptionProgress], None]
StatusCallback = Callable[[str], None]
FileResultCallback = Callable[[FileProcessingResult], None]
ErrorCallback = Callable[[str, Optional[Exception]], None]
SegmentTranscriber = Callable[[np.ndarray, int], "str | SegmentOutcome"]
Segmenter = Callable[[np.ndarray, int], List[Tuple[float, float]]]
AudioPreprocessor = Callable[[np.ndarray, int], np.ndarray]
# (audio, sample_rate) -> processed (#366 Phase 3)。ロード直後・segmenter 前に
# 1 回だけ適用され、以降の VAD 判定・ASR slice・EnergyGate は処理済み配列を見る。
# 戻り値契約: np.ndarray / 1 次元 / shape・dtype とも入力と同一 (float32)。
# sample rate は意味論的契約 — preprocessor は変更せず、返却音声は引数で
# 渡された同一 sample rate として解釈される。例外は file-level failure。


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
        audio_preprocessor: Optional[AudioPreprocessor] = None,
    ) -> None:
        self._ffmpeg_manager = ffmpeg_manager or get_ffmpeg_manager()
        self._segmenter = segmenter
        self._audio_preprocessor = audio_preprocessor
        self._temp_root = Path(tempfile.mkdtemp(prefix="livecap-file-pipeline-"))
        self._ffmpeg_path: Optional[str] = None
        self._ffprobe_path: Optional[str] = None
        # One worker for the whole pipeline, created on first use. Per-call
        # executors let a timed-out translation keep running while the next
        # segment started another one, so the same translator - and the same
        # requests.Session - was used concurrently (Issue #402).
        self._translation_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # close() が「translator をもう使っていない」ことを保証するために、
        # 実行中の翻訳を覚えておく (Issue #402)。
        self._translation_inflight: Optional[concurrent.futures.Future] = None
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
            error_callback: invoked when pipeline level errors occur — either a
                raised exception or a file completing with ``success=False``
                (e.g. all ASR segment calls failed; Issue #362).
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
            else:
                # Issue #362: process_file can now *return* success=False (e.g. every
                # ASR segment call raised). Keep the error_callback contract
                # consistent for both raised and returned failures.
                if not result.success and error_callback:
                    error_callback(result.error or "", None)

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
            # ロード直後の cancel は preprocessor より前に確認する — 任意の
            # 公開 callable (副作用あり得る) を cancel 後に走らせない
            self._check_cancel(should_cancel)
            # #366 Phase 3: 前処理はここで 1 回だけ。以降の VAD 判定・ASR slice・
            # EnergyGate はすべて処理済み配列を見る (realtime と同じ意味論)
            audio_data = self._apply_audio_preprocessor(audio_data, sample_rate)
            # 前処理は長尺 file で時間がかかるため、segmentation 前にも再確認
            self._check_cancel(should_cancel)
            segments = self._segment(audio_data, sample_rate)
            # Issue #366: 検出セグメント数 (字幕数 segment_count とは別) と
            # 「注入 segmenter がセグメントなしと判定した」flag。
            # (VAD の false negative もあり得るため no_speech ではなくこの名前)
            detected_segment_count = len(segments)
            segmentation_empty = self._segmenter is not None and not segments
            outcome = self._transcribe_segments(
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
            subtitles = outcome.subtitles
            counts = {
                "asr_calls": outcome.asr_calls,
                "asr_errors": outcome.asr_errors,
                "empty_results": outcome.empty_results,
            }

            # Issue #362: every ASR call raised -> promote to file-level failure.
            # Do NOT write (or overwrite) any SRT in this state.
            if outcome.asr_calls > 0 and outcome.asr_errors == outcome.asr_calls:
                err = outcome.first_error
                error_msg = (
                    f"All {outcome.asr_calls} ASR segment calls failed; "
                    f"first error: {type(err).__name__}: {err}"
                )
                logger.error("File transcription produced no output for %s: %s", source, error_msg)
                return FileProcessingResult(
                    source_path=source,
                    success=False,
                    output_path=None,
                    error=error_msg,
                    subtitles=[],
                    metadata={
                        **counts,
                        "segment_count": 0,
                        "detected_segment_count": detected_segment_count,
                        "segmentation_empty": segmentation_empty,
                        "drop_counts": dict(outcome.drop_counts),
                        "duration_seconds": float(len(audio_data) / sample_rate),
                        "sample_rate": sample_rate,
                    },
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
                **counts,
                "segment_count": len(subtitles),
                "detected_segment_count": detected_segment_count,
                "segmentation_empty": segmentation_empty,
                "drop_counts": dict(outcome.drop_counts),
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
        """Cleanup temporary resources.

        `getattr` guard: `__init__` が TypeError 等で本体未実行のまま
        インスタンスが GC された場合でも `__del__` → `close()` が
        二次 AttributeError を出さないようにする (Issue #363)。
        """
        # translator は呼び出し側が所有しており、close() の後に cleanup() される。
        # 待たずに返すと、借りている requests.Session を使っている最中に閉じられる
        # ことになる (Issue #402)。cancel_futures=True は実行中の future を止めない。
        inflight = getattr(self, "_translation_inflight", None)
        if inflight is not None and not inflight.done():
            # **打ち切らない。** 上限を設けて諦めると、まさに待つ理由だったケースで
            # 借用中の translator を owner に cleanup させることになる。
            # 詳細は drain_translation() の docstring。
            drain_translation(inflight)
        self._translation_inflight = None

        executor = getattr(self, "_translation_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._translation_executor = None

        temp_root = getattr(self, "_temp_root", None)
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

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

    def _apply_audio_preprocessor(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """注入された audio_preprocessor を適用し、戻り値契約を検証する (#366)。

        契約違反は fail-fast (`TypeError`/`ValueError`)、preprocessor の例外と
        あわせて **file-level failure** としてそのまま送出する (`process_files`
        が failed result へ変換する既存経路 #362)。
        """
        if self._audio_preprocessor is None:
            return audio
        processed = self._audio_preprocessor(audio, sample_rate)
        if not isinstance(processed, np.ndarray):
            raise TypeError(
                "audio_preprocessor must return np.ndarray, "
                f"got {type(processed).__name__}"
            )
        if processed.ndim != 1 or processed.shape != audio.shape:
            raise ValueError(
                f"audio_preprocessor must preserve the 1-D shape {audio.shape}, "
                f"got shape {processed.shape}"
            )
        if processed.dtype != audio.dtype:
            raise ValueError(
                f"audio_preprocessor must preserve dtype {audio.dtype}, "
                f"got {processed.dtype}"
            )
        return processed

    def _segment(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        if self._segmenter is not None:
            # 注入 segmenter の [] は「音声セグメントなし」= ASR 呼び出しゼロ (#366)。
            # 全音声 fallback すると VAD が正しく無音判定した場合に全音声が
            # ASR へ流れて hallucination を招く。全音声 1 segment 処理には
            # segmenter=None を使う。
            return list(self._segmenter(audio, sample_rate))
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
    ) -> _SegmentTranscriptionOutcome:
        segment_list = list(segments)
        outcome = _SegmentTranscriptionOutcome(subtitles=[])
        subtitles = outcome.subtitles
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
                raw = transcriber(segment_audio, sample_rate)
                # 型検証と正規化も同じ per-segment try 内で行う — 契約違反
                # (非 str/SegmentOutcome 返却等) は #362 どおり asr_errors に
                # 集計し、全件なら success=False へ昇格させる (pipeline-level
                # 例外にしない)
                if isinstance(raw, SegmentOutcome):
                    structured: Optional[SegmentOutcome] = raw
                    text = raw.text.strip()
                elif isinstance(raw, str):
                    structured = None
                    text = raw.strip()
                else:
                    raise TypeError(
                        "SegmentTranscriber must return str or SegmentOutcome, "
                        f"got {type(raw).__name__}"
                    )
            except FileTranscriptionCancelled:
                raise
            except Exception as exc:
                logger.error("Segment transcription failed (%s-%s): %s", start, end, exc)
                # 例外は従来どおり「ASR 試行」として数える (#362 の全滅判定)
                outcome.asr_calls += 1
                outcome.asr_errors += 1
                if outcome.first_error is None:
                    outcome.first_error = exc
                continue

            # #366 Phase 2 正規化規則: legacy str と SegmentOutcome を受理。
            # asr_calls は「engine を実際に呼んだ数」— gate drop
            # (asr_called=False) は数えない。
            if structured is not None:
                if structured.asr_called:
                    outcome.asr_calls += 1
                if structured.drop_reason is not None:
                    # drop は empty_results / asr_errors に混ぜない (統計正本は
                    # drop_counts、reason は REASON_* 語彙)
                    outcome.drop_counts[structured.drop_reason] = (
                        outcome.drop_counts.get(structured.drop_reason, 0) + 1
                    )
                    continue
            else:
                outcome.asr_calls += 1  # legacy str — 意味不変

            if not text:
                # 明示 drop_reason なしの空 (legacy "" / success("")) — 従来契約
                outcome.empty_results += 1
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
        return outcome

    def _write_srt(
        self,
        source: Path,
        subtitles: list[FileSubtitleSegment],
    ) -> Path:
        return write_srt(source.with_suffix(".srt"), subtitles)

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
            # context_len=0 の場合は文脈を使わない（[-0:] は [:] と同義で全履歴が渡るため）
            context_len = translator.default_context_sentences
            context = (
                list(context_buffer)[-context_len:]
                if context_buffer and context_len > 0
                else None
            )

            # Treat timeout <= 0 as no timeout (invalid value)
            effective_timeout = timeout if timeout is not None and timeout > 0 else None

            # Retry lives here, not in the adapter: only the caller knows whether
            # this is a file job (worth retrying) or a live subtitle (Issue #402
            # D10). Only TranslationNetworkError is retried.
            # The per-attempt budget comes from the translator, not from a
            # constant: this policy is applied to any BaseTranslator, and a local
            # model cannot promise a bound at all (Issue #402).
            policy = retry_for_translator(FILE_RETRY_POLICY, translator)

            def attempt():
                return policy.call(
                    lambda: translator.translate(text, source_lang, target_lang, context)
                )

            if effective_timeout is not None:
                # A single pipeline-owned worker, so a timed-out translation can
                # never run alongside the next one. Two earlier shapes were both
                # wrong (Issue #402):
                #   `with ThreadPoolExecutor(...)` -> shutdown(wait=True) on exit,
                #     so the timeout bounded when we stopped *waiting*, not when
                #     we returned.
                #   a per-call executor with shutdown(wait=False) -> returned
                #     promptly but left the worker running, and the next segment
                #     called the same translator concurrently, sharing one
                #     requests.Session.

                # Do not submit while one is still running. Queuing would have
                # been harmless for concurrency (one worker), but it *overwrites*
                # the in-flight reference: a queued call that times out gets
                # cancelled and looks done, so close() would skip draining the
                # one that is actually running and hand a busy translator back to
                # its owner (Issue #402).
                #
                # Nothing is lost by skipping: the queued call would wait behind
                # the running one and time out without ever starting.
                running = self._translation_inflight
                if running is not None and not running.done():
                    logger.warning(
                        "Translation timed out after %.1fs", effective_timeout
                    )
                    return None, None

                future = self._translation_worker().submit(attempt)
                self._translation_inflight = future
                try:
                    result = future.result(timeout=effective_timeout)
                    return result.text, target_lang
                except concurrent.futures.TimeoutError:
                    # Drop it only if it never started. A cancelled future reports
                    # done(), so keeping a *running* one here is what lets close()
                    # find it and drain before the owner cleans the translator up.
                    if future.cancel():
                        self._translation_inflight = None
                    # Never log the text itself: it is the user's speech, and
                    # this line used to write 50 characters of it to disk on
                    # every timeout (Issue #402 D8).
                    logger.warning(
                        "Translation timed out after %.1fs", effective_timeout
                    )
                    return None, None
            else:
                # No timeout - direct call
                result = attempt()
                return result.text, target_lang

        except Exception as exc:
            logger.warning("Translation failed: %s", exc)
            return None, None

    def _translation_worker(self) -> concurrent.futures.ThreadPoolExecutor:
        """The pipeline's single translation worker (created on first use)."""
        if self._translation_executor is None:
            self._translation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="livecap-translate"
            )
        return self._translation_executor

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

        return write_srt(output_path, translated_segments, translated=True)


__all__ = [
    "FileTranscriptionPipeline",
    "FileTranscriptionProgress",
    "FileProcessingResult",
    "FileSubtitleSegment",
    "FileTranscriptionCancelled",
    "SegmentOutcome",
    "ProgressCallback",
    "StatusCallback",
    "FileResultCallback",
    "ErrorCallback",
    "SegmentTranscriber",
    "Segmenter",
    "AudioPreprocessor",
]
