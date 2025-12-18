"""ストリーミング文字起こし

VADプロセッサとASRエンジンを組み合わせて
リアルタイム文字起こしを行う。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import queue
from collections import deque
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    Iterator,
    List,
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
    from ..translation.base import BaseTranslator

logger = logging.getLogger(__name__)

# 翻訳用の文脈バッファの最大サイズ
MAX_CONTEXT_BUFFER = 100

# 翻訳タイムアウト（秒）: Riva-4B など重いモデルでの ASR ブロック防止
# 環境変数 LIVECAP_TRANSLATION_TIMEOUT で上書き可能
_DEFAULT_TRANSLATION_TIMEOUT = 10.0


def _get_translation_timeout() -> float:
    """環境変数から翻訳タイムアウトを取得（安全なパース）"""
    env_value = os.environ.get("LIVECAP_TRANSLATION_TIMEOUT")
    if env_value is None:
        return _DEFAULT_TRANSLATION_TIMEOUT

    try:
        timeout = float(env_value)
    except ValueError:
        logger.warning(
            "Invalid LIVECAP_TRANSLATION_TIMEOUT value '%s', using default %.1fs",
            env_value,
            _DEFAULT_TRANSLATION_TIMEOUT,
        )
        return _DEFAULT_TRANSLATION_TIMEOUT

    if timeout <= 0:
        logger.warning(
            "LIVECAP_TRANSLATION_TIMEOUT must be positive (got %.1f), using default %.1fs",
            timeout,
            _DEFAULT_TRANSLATION_TIMEOUT,
        )
        return _DEFAULT_TRANSLATION_TIMEOUT

    return timeout


TRANSLATION_TIMEOUT = _get_translation_timeout()


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

    def get_engine_name(self) -> str:
        """エンジン名を取得"""
        ...

    def cleanup(self) -> None:
        """リソースのクリーンアップ"""
        ...


class StreamTranscriber:
    """
    ストリーミング文字起こし

    VADプロセッサとASRエンジンを組み合わせて
    リアルタイム文字起こしを行う。
    オプションで翻訳エンジンを統合し、ASR + 翻訳のパイプラインを提供。

    Args:
        engine: 文字起こしエンジン（BaseEngine互換）
        translator: 翻訳エンジン（BaseTranslator）。指定時は source_lang/target_lang 必須
        source_lang: 翻訳元言語コード（translator 指定時は必須）
        target_lang: 翻訳先言語コード（translator 指定時は必須）
        vad_config: VAD設定（vad_processor未指定時に使用）
        vad_processor: VADプロセッサ（テスト用に注入可能）
        source_id: 音声ソース識別子
        max_workers: 文字起こし用スレッド数（デフォルト: 1）

    Usage:
        # 基本的な使い方（翻訳なし）
        transcriber = StreamTranscriber(engine=engine)

        with MicrophoneSource() as mic:
            for result in transcriber.transcribe_sync(mic):
                print(f"[{result.start_time:.2f}s] {result.text}")

        # 翻訳付き
        translator = TranslatorFactory.create_translator("google")
        transcriber = StreamTranscriber(
            engine=engine,
            translator=translator,
            source_lang="ja",
            target_lang="en",
        )
        for result in transcriber.transcribe_sync(mic):
            print(f"[JA] {result.text}")
            if result.translated_text:
                print(f"[EN] {result.translated_text}")

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
        translator: Optional["BaseTranslator"] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ):
        self.engine = engine
        self.source_id = source_id
        self._sample_rate = engine.get_required_sample_rate()

        # 翻訳設定
        self._translator = translator
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._context_buffer: deque[str] = deque(maxlen=MAX_CONTEXT_BUFFER)

        # translator 設定時のバリデーション
        if translator is not None:
            if not translator.is_initialized():
                raise ValueError(
                    "Translator not initialized. Call load_model() first."
                )
            if source_lang is None or target_lang is None:
                raise ValueError(
                    "source_lang and target_lang are required when translator is set."
                )
            # 言語ペアの事前警告
            pairs = translator.get_supported_pairs()
            if pairs and (source_lang, target_lang) not in pairs:
                logger.warning(
                    "Language pair (%s -> %s) may not be supported by %s",
                    source_lang,
                    target_lang,
                    translator.get_translator_name(),
                )

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
        # 翻訳用文脈バッファをクリア
        self._context_buffer.clear()
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

            text = text.strip()

            # 翻訳処理（translator が設定されている場合）
            translated_text, target_language = self._translate_text(text)

            return TranscriptionResult(
                text=text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
                translated_text=translated_text,
                target_language=target_language,
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

            text = text.strip()

            # 翻訳処理（translator が設定されている場合）
            # 翻訳も executor で実行（ブロッキング回避）、タイムアウト付き
            # Note: _do_translate_direct を使用（_translate_text ではない）
            # _translate_text は内部で executor.submit() するためデッドロックの原因になる
            translated_text: Optional[str] = None
            target_language: Optional[str] = None

            if self._translator:
                try:
                    translated_text, target_language = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            self._do_translate_direct,
                            text,
                        ),
                        timeout=TRANSLATION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Async translation timed out after {TRANSLATION_TIMEOUT}s"
                    )
                    # Note: _do_translate_direct は executor 上で継続実行され、
                    # 完了時に文脈バッファへ追加するため、ここでは追加しない
                except asyncio.CancelledError:
                    # キャンセル時は再送出（asyncio の規約）
                    raise
                except Exception as e:
                    # _do_translate_direct は内部例外を握るため、ここには基本到達しない
                    # （到達するのは CancelledError 以外の asyncio 系例外のみ）
                    logger.warning(f"Async translation failed: {e}")

            return TranscriptionResult(
                text=text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
                translated_text=translated_text,
                target_language=target_language,
            )
        except Exception as e:
            logger.error(f"Async transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    def _do_translate_direct(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        テキストを翻訳（executor 提出なし、直接実行）

        Args:
            text: 翻訳対象テキスト

        Returns:
            (translated_text, target_language) のタプル
            翻訳に失敗した場合は (None, None)

        Note:
            このメソッドは同期的に翻訳を実行し、タイムアウト制御は呼び出し側が担当。
            _transcribe_segment_async から executor 経由で呼ばれる想定。
            デッドロック回避のため、executor への二重提出を避ける。
        """
        if not self._translator or not text:
            return None, None

        # 公開プロパティから context_sentences を取得
        # context_len=0 の場合は文脈を使わない（[-0:] は [:] と同義で全履歴が渡るため）
        context_len = self._translator.default_context_sentences
        context: Optional[List[str]] = (
            list(self._context_buffer)[-context_len:] if context_len > 0 else None
        )

        try:
            trans_result = self._translator.translate(
                text,
                self._source_lang,  # type: ignore[arg-type]
                self._target_lang,  # type: ignore[arg-type]
                context=context,
            )

            # 文脈バッファに追加
            self._context_buffer.append(text)

            return trans_result.text, self._target_lang

        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            # 翻訳失敗しても文脈バッファには追加（次の翻訳の文脈として使用）
            self._context_buffer.append(text)
            return None, None

    def _translate_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        テキストを翻訳（タイムアウト付き、同期パス用）

        Args:
            text: 翻訳対象テキスト

        Returns:
            (translated_text, target_language) のタプル
            translator が設定されていないか、翻訳に失敗/タイムアウトした場合は (None, None)

        Note:
            TRANSLATION_TIMEOUT（デフォルト10秒）を超過した場合、
            翻訳をスキップして (None, None) を返し、ASR パイプラインを継続。
            これは Riva-4B など重いモデルでのブロック防止策。

            同期パス（feed_audio, transcribe_sync）から呼ばれる想定。
            非同期パス（transcribe_async）では _do_translate_direct を使用。
        """
        if not self._translator or not text:
            return None, None

        # 公開プロパティから context_sentences を取得
        # context_len=0 の場合は文脈を使わない（[-0:] は [:] と同義で全履歴が渡るため）
        context_len = self._translator.default_context_sentences
        context: Optional[List[str]] = (
            list(self._context_buffer)[-context_len:] if context_len > 0 else None
        )

        def do_translate() -> str:
            """翻訳実行（executor 内で呼ばれる）"""
            trans_result = self._translator.translate(  # type: ignore[union-attr]
                text,
                self._source_lang,  # type: ignore[arg-type]
                self._target_lang,  # type: ignore[arg-type]
                context=context,
            )
            return trans_result.text

        try:
            # タイムアウト付きで翻訳を実行
            future = self._executor.submit(do_translate)
            translated = future.result(timeout=TRANSLATION_TIMEOUT)

            # 文脈バッファに追加
            self._context_buffer.append(text)

            return translated, self._target_lang

        except concurrent.futures.TimeoutError:
            logger.warning(
                f"Translation timed out after {TRANSLATION_TIMEOUT}s, skipping translation"
            )
            # タイムアウトしても文脈バッファには追加
            self._context_buffer.append(text)
            return None, None

        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            # 翻訳失敗しても文脈バッファには追加（次の翻訳の文脈として使用）
            self._context_buffer.append(text)
            return None, None

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
