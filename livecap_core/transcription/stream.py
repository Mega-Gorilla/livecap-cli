"""ストリーミング文字起こし

VADプロセッサとASRエンジンを組み合わせて
リアルタイム文字起こしを行う。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np

from ..vad import VADConfig, VADProcessor, VADSegment
from .result import InterimResult, TranscriptionResult

if TYPE_CHECKING:
    from ..audio_sources import AudioSource

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """文字起こしエラーの基底クラス"""

    pass


class EngineError(TranscriptionError):
    """エンジン関連のエラー"""

    pass


class TranscriptionEngine(Protocol):
    """文字起こしエンジンのプロトコル

    既存の BaseEngine と互換性のあるインターフェース。
    """

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """音声データを文字起こしする

        Args:
            audio: 音声データ（numpy配列, float32）
            sample_rate: サンプリングレート

        Returns:
            (text, confidence) のタプル
        """
        ...

    def get_required_sample_rate(self) -> int:
        """エンジンが要求するサンプリングレートを取得"""
        ...


class StreamTranscriber:
    """
    ストリーミング文字起こし

    VADプロセッサとASRエンジンを組み合わせて
    リアルタイム文字起こしを行う。

    Args:
        engine: 文字起こしエンジン（BaseEngine互換）
        vad_config: VAD設定（vad_processor未指定時に使用）
        vad_processor: VADプロセッサ（テスト用に注入可能）
        source_id: 音声ソース識別子
        max_workers: 文字起こし用スレッド数（デフォルト: 1）

    Usage:
        # 基本的な使い方
        transcriber = StreamTranscriber(engine=engine)

        with MicrophoneSource() as mic:
            for result in transcriber.transcribe_sync(mic):
                print(f"[{result.start_time:.2f}s] {result.text}")

        # 非同期使用
        async with MicrophoneSource() as mic:
            async for result in transcriber.transcribe_async(mic):
                print(result.text)

        # コールバック方式
        transcriber.set_callbacks(
            on_result=lambda r: print(f"[確定] {r.text}"),
            on_interim=lambda r: print(f"[途中] {r.text}"),
        )
        for chunk in mic:
            transcriber.feed_audio(chunk, mic.sample_rate)
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ):
        self.engine = engine
        self.source_id = source_id
        self._sample_rate = engine.get_required_sample_rate()

        # VADプロセッサ（注入または新規作成）
        if vad_processor is not None:
            self._vad = vad_processor
        else:
            self._vad = VADProcessor(config=vad_config)

        # 文字起こし用スレッドプール
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # 結果キュー
        self._result_queue: queue.Queue[
            Union[TranscriptionResult, InterimResult]
        ] = queue.Queue()

        # コールバック
        self._on_result: Optional[Callable[[TranscriptionResult], None]] = None
        self._on_interim: Optional[Callable[[InterimResult], None]] = None

    def set_callbacks(
        self,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_interim: Optional[Callable[[InterimResult], None]] = None,
    ) -> None:
        """コールバックを設定

        Args:
            on_result: 確定結果のコールバック
            on_interim: 中間結果のコールバック
        """
        self._on_result = on_result
        self._on_interim = on_interim

    def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """
        音声チャンクを入力

        VAD でセグメントが検出された場合、文字起こしを実行するため
        ブロッキングが発生する。非同期処理が必要な場合は
        transcribe_async() を使用すること。

        結果は get_result() / get_interim() で取得するか、
        コールバックで受け取る。

        Args:
            audio: 音声データ（float32）
            sample_rate: サンプリングレート

        Note:
            セグメント検出時は engine.transcribe() が呼ばれるため
            処理時間はエンジンに依存する（数十ms〜数百ms）。
        """
        # VAD処理
        segments = self._vad.process_chunk(audio, sample_rate)

        for segment in segments:
            if segment.is_final:
                try:
                    result = self._transcribe_segment(segment)
                    if result:
                        self._result_queue.put(result)
                        if self._on_result:
                            self._on_result(result)
                except EngineError as e:
                    logger.warning(f"Transcription failed, skipping segment: {e}")
            else:
                # 中間結果
                interim = self._transcribe_interim(segment)
                if interim:
                    self._result_queue.put(interim)
                    if self._on_interim:
                        self._on_interim(interim)

    def get_result(
        self, timeout: Optional[float] = None
    ) -> Optional[TranscriptionResult]:
        """確定結果を取得（ブロッキング）

        Args:
            timeout: タイムアウト（秒）、Noneで即時リターン

        Returns:
            TranscriptionResult またはNone
        """
        try:
            result = self._result_queue.get(timeout=timeout)
            if isinstance(result, TranscriptionResult):
                return result
            # InterimResultは無視して次を待つ
            return self.get_result(timeout=0.001) if timeout else None
        except queue.Empty:
            return None

    def get_interim(self) -> Optional[InterimResult]:
        """中間結果を取得（ノンブロッキング）

        Returns:
            InterimResult またはNone
        """
        try:
            result = self._result_queue.get_nowait()
            if isinstance(result, InterimResult):
                return result
            # TranscriptionResultは戻す
            self._result_queue.put(result)
            return None
        except queue.Empty:
            return None

    def finalize(self) -> Optional[TranscriptionResult]:
        """処理を終了し、残っているセグメントを文字起こし

        Returns:
            残りのセグメントのTranscriptionResult、またはNone
        """
        segment = self._vad.finalize()
        if segment and segment.is_final:
            try:
                return self._transcribe_segment(segment)
            except EngineError as e:
                logger.warning(f"Final transcription failed: {e}")
                return None
        return None

    def reset(self) -> None:
        """状態をリセット"""
        self._vad.reset()
        # キューをクリア
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def _transcribe_segment(
        self, segment: VADSegment
    ) -> Optional[TranscriptionResult]:
        """セグメントを文字起こし（同期）

        Args:
            segment: VADセグメント

        Returns:
            TranscriptionResult またはNone

        Raises:
            EngineError: 文字起こしに失敗した場合
        """
        if len(segment.audio) == 0:
            return None

        try:
            text, confidence = self.engine.transcribe(segment.audio, self._sample_rate)

            if not text or not text.strip():
                return None

            return TranscriptionResult(
                text=text.strip(),
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    async def _transcribe_segment_async(
        self, segment: VADSegment
    ) -> Optional[TranscriptionResult]:
        """セグメントを文字起こし（非同期、executor使用）

        Args:
            segment: VADセグメント

        Returns:
            TranscriptionResult またはNone

        Raises:
            EngineError: 文字起こしに失敗した場合
        """
        if len(segment.audio) == 0:
            return None

        loop = asyncio.get_running_loop()
        try:
            text, confidence = await loop.run_in_executor(
                self._executor,
                self.engine.transcribe,
                segment.audio,
                self._sample_rate,
            )

            if not text or not text.strip():
                return None

            return TranscriptionResult(
                text=text.strip(),
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Async transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    def _transcribe_interim(self, segment: VADSegment) -> Optional[InterimResult]:
        """中間結果の文字起こし

        Args:
            segment: VADセグメント

        Returns:
            InterimResult またはNone
        """
        if len(segment.audio) == 0:
            return None

        try:
            text, _ = self.engine.transcribe(segment.audio, self._sample_rate)

            if not text or not text.strip():
                return None

            return InterimResult(
                text=text.strip(),
                accumulated_time=segment.end_time - segment.start_time,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Interim transcription error: {e}", exc_info=True)
            return None

    def close(self) -> None:
        """リソースを解放"""
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        """デストラクタ: リソースを確実に解放"""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass  # GC 時のエラーは無視

    def __enter__(self) -> "StreamTranscriber":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # === 高レベルAPI ===

    def transcribe_sync(
        self,
        audio_source: "AudioSource",
    ) -> Iterator[TranscriptionResult]:
        """
        同期ストリーム処理

        Args:
            audio_source: AudioSourceインスタンス

        Yields:
            TranscriptionResult
        """
        for chunk in audio_source:
            self.feed_audio(chunk, audio_source.sample_rate)

            while True:
                result = self.get_result(timeout=0)
                if result:
                    yield result
                else:
                    break

        # 最終セグメント
        final = self.finalize()
        if final:
            yield final

    async def transcribe_async(
        self,
        audio_source: "AudioSource",
    ) -> AsyncIterator[TranscriptionResult]:
        """
        非同期ストリーム処理

        VAD処理はメインスレッドで実行し、
        文字起こしは ThreadPoolExecutor で実行する。

        Args:
            audio_source: AudioSourceインスタンス

        Yields:
            TranscriptionResult
        """
        async for chunk in audio_source:
            # VAD処理は軽いのでメインスレッドで実行
            segments = self._vad.process_chunk(chunk, audio_source.sample_rate)

            for segment in segments:
                if segment.is_final:
                    # 文字起こしは executor で実行
                    try:
                        result = await self._transcribe_segment_async(segment)
                        if result:
                            yield result
                    except EngineError as e:
                        logger.warning(f"Async transcription failed: {e}")
                # 中間結果は同期で処理（高速なため）
                elif self._on_interim:
                    interim = self._transcribe_interim(segment)
                    if interim:
                        self._on_interim(interim)

            # 他のタスクに制御を譲る
            await asyncio.sleep(0)

        # 最終セグメント
        final_segment = self._vad.finalize()
        if final_segment and final_segment.is_final:
            try:
                result = await self._transcribe_segment_async(final_segment)
                if result:
                    yield result
            except EngineError as e:
                logger.warning(f"Final async transcription failed: {e}")

    @property
    def vad_state(self):
        """現在のVAD状態"""
        return self._vad.state

    @property
    def sample_rate(self) -> int:
        """エンジンが要求するサンプリングレート"""
        return self._sample_rate
