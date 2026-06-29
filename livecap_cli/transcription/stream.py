"""ストリーミング文字起こし

VADプロセッサとASRエンジンを組み合わせて
リアルタイム文字起こしを行う。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import queue
from collections import deque
from dataclasses import dataclass, replace
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

from ..audio import ENERGY_METRICS, _segment_energy_dbfs
# Runtime import (codex-review on #309): TYPE_CHECKING のみだと
# typing.get_type_hints() で NameError になるため通常 import に格上げ。
# `livecap_cli.engines.base_engine` は `livecap_cli.transcription` を import
# していないので循環依存はない (本ファイルで grep 確認済)。
from ..engines.base_engine import (
    TranscriptionResult as EngineTranscriptionResult,
)
from ..vad import VADConfig, VADProcessor, VADSegment
from .confidence_filter import FilterConfig, apply_filter
from .result import InterimResult, TranscriptionResult
from .result_coalescer import ResultCoalescer
from .utterance import (
    REASON_EMPTY_AUDIO,
    REASON_ENERGY_GATE,
    REASON_ENGINE_EMPTY,
    REASON_FILTER_REJECT,
    UtteranceSettledEvent,
)

if TYPE_CHECKING:
    from ..audio import NoiseGate, TransientDetector
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


@dataclass(frozen=True)
class _SegmentTranscriptionOutcome:
    """Internal: ``_transcribe_segment*`` の return type (Issue #332)。

    ``Optional[TranscriptionResult]`` だけでは 4 drop branch (empty audio /
    energy_gate / filter reject / engine empty) の reason を caller が
    区別できないため、本 wrapper で drop_reason を保持する。``engine_error``
    だけは raise → caller catch で settled 発火するため本 outcome には含めない。

    Public re-export せず、consumer は ``UtteranceSettledEvent`` 経由で
    reason を受け取る (Issue #332 rev6 design)。
    """

    result: Optional[TranscriptionResult]
    drop_reason: Optional[str]

    @classmethod
    def success(cls, result: TranscriptionResult) -> "_SegmentTranscriptionOutcome":
        return cls(result=result, drop_reason=None)

    @classmethod
    def dropped(cls, reason: str) -> "_SegmentTranscriptionOutcome":
        return cls(result=None, drop_reason=reason)


class TranscriptionEngine(Protocol):
    """文字起こしエンジンのプロトコル

    既存の BaseEngine と互換性のあるインターフェース。

    **API contract (Issue #321 PR #3 で厳格化)**:

    実装者は ``transcribe()`` から **必ず**
    ``livecap_cli.engines.base_engine.TranscriptionResult`` を返すこと。
    pre-1.0 cleanup で legacy adapter fallback (tuple / dict / str / None)
    を全て削除済 (Issue #321 PR #3):

    - ``confidence_filter.py::apply_filter`` は ``result.engine_confidence``
      に bare attribute access、契約違反 (tuple / dict / str / None) を
      渡された場合は ``AttributeError`` が **caller (StreamTranscriber) に
      propagate して fail-fast** する (PR #320 / PR #322 / PR #323 の
      framework-trust precedent と整合)

    Note:
        戻り値 ``EngineTranscriptionResult`` は engines パッケージの
        ``livecap_cli.engines.base_engine.TranscriptionResult`` の runtime import
        による alias で、本 module 内の ``TranscriptionResult``
        (= ``livecap_cli.transcription.result.TranscriptionResult``、coalescer
        出力用) とは別の dataclass です。codex-review on #309 で指摘された
        ``typing.get_type_hints()`` での NameError を避けるため、
        ``TYPE_CHECKING`` ではなく runtime block で import しています。
    """

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> "EngineTranscriptionResult":
        """音声データを文字起こしする

        Args:
            audio: 音声データ（numpy配列, float32）
            sample_rate: サンプリングレート

        Returns:
            EngineTranscriptionResult: ``TranscriptionResult`` dataclass
            (``livecap_cli.engines.base_engine.TranscriptionResult``)。
            ``text`` / ``confidence`` / ``engine_confidence`` を持ち、
            attribute access (``result.text`` 等) で値取得する。

            **必ず TranscriptionResult を返すこと**。tuple / dict / str /
            None は契約違反 (Issue #321 PR #3)。``apply_filter``
            (``StreamTranscriber`` 経路) で ``AttributeError`` が caller
            に propagate して **fail-fast** する。詳細は本 Protocol class
            docstring の "API contract" section を参照。
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

        # Issue #332: utterance lifecycle observation hook
        from livecap_cli import UtteranceSettledEvent, REASON_FILTER_REJECT

        def on_settled(event: UtteranceSettledEvent) -> None:
            if not event.emitted and event.reason == REASON_FILTER_REJECT:
                gui.clear_interim()  # consumer 側 state を即時 clear

        transcriber.set_callbacks(
            on_result=on_result,
            on_interim=on_interim,
            on_utterance_settled=on_settled,
        )
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
        result_coalescer: Optional[ResultCoalescer] = None,
        noise_gate: Optional["NoiseGate"] = None,
        transient_detector: Optional["TransientDetector"] = None,
        engine_min_rms_dbfs: float = -45.0,
        engine_energy_metric: str = "max_frame_rms",
        engine_energy_frame_ms: float = 32.0,
        filter_config: Optional[FilterConfig] = None,
    ):
        self.engine = engine
        self.source_id = source_id
        self._sample_rate = engine.get_required_sample_rate()

        # === Confidence filter (PR-A.1 / Issue #308) ===
        # PR-A.0 で expose した engine_confidence を見て「非音声」判定 output を
        # 字幕に出る前に弾く。default は `mode="on"` (Issue #308 v3.1)。
        # `filter_config=None` は内部で `FilterConfig()` (= mode="on") を構築
        # するため、CLI / 直接 API どちらも default ON で動作する。
        # post-ASR filter/reject を無効化するには `--confidence-filter off`
        # または `LIVECAP_CONFIDENCE_FILTER=off` (CLI) もしくは
        # `filter_config=FilterConfig(mode="off")` (直接 API) を指定する
        # (各 engine の generation parameter — Canary greedy / Voxtral greedy /
        # qwen3asr repetition_penalty 等 — は filter mode と独立で固定)。
        self._filter_config = filter_config or FilterConfig()
        # get_engine_name() は Protocol だが MockEngine 等 test 用 mock では
        # 実装されない可能性があるため、safe getattr で fallback。
        try:
            self._engine_name = engine.get_engine_name()
        except AttributeError:
            self._engine_name = type(engine).__name__
        self._log_filter_banner()

        # === EnergyGate 設定 (#292) ===
        # per-segment energy ガード: low-RMS segment を engine.transcribe() に
        # 渡さないことで low-energy hallucination ("うん"/"ピッ"/"え?") を抑制。
        # NoiseGate (per-sample peak envelope, pre-VAD) と物理量が異なる相補的
        # 防御層。`-inf` 渡しで完全 opt-out。
        if engine_energy_metric not in ENERGY_METRICS:
            raise ValueError(
                f"engine_energy_metric must be one of {ENERGY_METRICS}, "
                f"got {engine_energy_metric!r}"
            )
        # threshold: finite or -inf only. Reject nan / +inf because:
        # - nan: `energy_dbfs < nan` is always False → gate silently disabled
        # - +inf: every segment dropped → no transcription
        threshold = float(engine_min_rms_dbfs)
        if math.isnan(threshold):
            raise ValueError(
                f"engine_min_rms_dbfs cannot be NaN "
                f"(got {engine_min_rms_dbfs!r}). "
                "Use a finite number or float('-inf') to opt out."
            )
        if threshold == float("inf"):
            raise ValueError(
                f"engine_min_rms_dbfs cannot be +inf "
                f"(got {engine_min_rms_dbfs!r}). "
                "Use a finite number or float('-inf') to opt out."
            )
        # frame_ms: must be finite positive. Reject nan / inf (would crash
        # later in int(sample_rate * frame_ms / 1000.0) or bypass <=0 check).
        frame_ms = float(engine_energy_frame_ms)
        if not math.isfinite(frame_ms) or frame_ms <= 0:
            raise ValueError(
                "engine_energy_frame_ms must be a finite positive number, "
                f"got {engine_energy_frame_ms!r}"
            )
        self._engine_min_rms_dbfs = threshold
        self._engine_energy_metric = engine_energy_metric
        self._engine_energy_frame_ms = frame_ms
        # callsite-separated drop counters (final_sync / final_async / interim)
        self._dropped_low_energy_final_sync = 0
        self._dropped_low_energy_final_async = 0
        self._dropped_low_energy_interim = 0

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
        # Issue #332: Utterance lifecycle observation hook (opt-in callback)
        self._on_utterance_settled: Optional[
            Callable[[UtteranceSettledEvent], None]
        ] = None

        # 短文結合（常時有効）
        self._coalescer = (
            result_coalescer
            if result_coalescer is not None
            else ResultCoalescer()
        )

        # ノイズゲート（opt-in）
        self._noise_gate = noise_gate
        # Layer 1: DSP transient detector (#295 PR-B, opt-in). None means
        # the layer is bypassed entirely (no overhead).
        self._transient_detector = transient_detector

        # Issue #334 Finding 6: Qwen3-ASR auto-detect + filter on の組合せで
        # silent fail-open する UX gap を 1 回 warn で notify
        # (programmatic API 利用者向け、CLI default は --language ja で保護済)
        self._maybe_warn_qwen3_auto_detect_fail_open()

    def _maybe_warn_qwen3_auto_detect_fail_open(self) -> None:
        """Qwen3-ASR auto-detect + filter on の組合せ silent fail-open を 1 回 warn (Issue #334 Finding 6)。

        Qwen3ASREngine は ``language=None`` (auto-detect) で wrapper fallback path
        に入り、``engine_confidence`` が全 None となる。filter は fail-open 規約で
        pass-through するため、user 視点では「filter on にしたのに reject が一切ない」
        挙動になる。``Qwen3ASREngine.__init__`` は ``FilterConfig`` を受けないため、
        両方を知る ``StreamTranscriber.__init__`` で警告するのが architectural に正しい。

        Duck typing で engine 検出 (``isinstance`` は循環 import / Mock false negative
        を回避)。``engine.engine_name == "qwen3asr"`` (internal ID、line 244 of
        ``qwen3asr_engine.py``) と ``engine._asr_language is None`` (line 248) の 2
        attribute を check する。
        """
        if self._filter_config.mode == "off":
            return
        if getattr(self.engine, "engine_name", "") != "qwen3asr":
            return
        if getattr(self.engine, "_asr_language", "sentinel") is not None:
            return
        logger.warning(
            "Qwen3-ASR auto-detect mode (language=None): confidence filter is "
            "effectively disabled (engine_confidence unavailable in this path). "
            "Specify language explicitly to enable filtering (e.g., "
            "language='Japanese'). See Issue #334 Finding 6."
        )

    def set_callbacks(
        self,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_interim: Optional[Callable[[InterimResult], None]] = None,
        on_utterance_settled: Optional[
            Callable[[UtteranceSettledEvent], None]
        ] = None,
    ) -> None:
        """コールバックを設定

        Args:
            on_result: 確定結果のコールバック
            on_interim: 中間結果のコールバック
            on_utterance_settled: 論理 utterance が settle した時点で発火
                する観測 hook (Issue #332)。``emitted=True`` なら final
                result が delivery boundary に渡された (callback / queue /
                generator yield いずれか) 直後、``emitted=False`` なら
                silent drop された時点。Drop reason は ``REASON_*`` 定数
                (``REASON_FILTER_REJECT`` 等) または ``engine_error:<type>``
                の動的文字列。Consumer が interim state を確実に clear
                するための lifecycle event。

                Delivery ordering:
                - ``feed_audio`` (callback path): ``on_result`` 完了 **後**
                  に ``on_utterance_settled`` を発火 (同期実行、stack frame
                  内で順序保証)。
                - ``transcribe_async`` (async generator): ``yield`` の
                  **直前** に発火 (yield 後の code は caller が次の
                  ``__anext__()`` を呼ぶまで実行されないため、break で永久
                  未発火になるのを回避)。
                - ``finalize`` (list return): result を list append する
                  **直前** に発火 (generator path と整合)。

                ``**kwargs`` は受け取らない: 未知 kwarg は ``TypeError`` で
                即時 fail (signature の純粋性、policy「不要な後方互換は
                廃する」、Issue #332 rev2)。
        """
        self._on_result = on_result
        self._on_interim = on_interim
        self._on_utterance_settled = on_utterance_settled

    def _emit_result(self, result: TranscriptionResult) -> None:
        """確定結果をキュー投入 + コールバック呼び出し。"""
        self._result_queue.put(result)
        if self._on_result:
            self._on_result(result)

    def _emit_utterance_settled(
        self,
        *,
        emitted: bool,
        reason: Optional[str],
        start_time: float,
        end_time: float,
    ) -> None:
        """``UtteranceSettledEvent`` を構築し callback を呼ぶ (Issue #332)。

        Caller side で 7 Tier 1 hook point (empty audio / energy_gate / filter
        reject / engine empty / engine error / coalescer push emission /
        coalescer flush emission) すべてから呼ばれる single funnel。
        ``on_utterance_settled`` が未登録なら no-op。
        """
        if self._on_utterance_settled is None:
            return
        event = UtteranceSettledEvent(
            emitted=emitted,
            reason=reason,
            source_id=self.source_id,
            utterance_start_time=start_time,
            utterance_end_time=end_time,
        )
        self._on_utterance_settled(event)

    def _engine_error_reason(self, err: EngineError) -> str:
        """``engine_error:<ExceptionType>`` reason を構築する (Issue #332)。

        ``raise EngineError(...) from e`` で chain された場合は ``__cause__``
        (inner exception) の type 名を、chain なし (``__cause__ is None``) の
        場合は ``EngineError`` 自身の type 名を使用する。``"NoneType"`` が
        reason 文字列に混入するのを防ぐ。
        """
        cause = err.__cause__ or err
        return f"engine_error:{type(cause).__name__}"

    def _handle_final_segment_callback(self, segment: VADSegment) -> None:
        """feed_audio path: outcome → settled + ``_emit_result``。

        Delivery ordering (Issue #332 rev6):
        - Drop path: 即時に ``settled(False, reason)`` を発火 (delivery なし)
        - Success path: ``_emit_result(merged)`` (queue + on_result callback)
          完了直後に ``settled(True, None)`` を発火 (consumer は「result 受信
          → settle 通知」の順で観測)
        """
        try:
            outcome = self._transcribe_segment(segment)
        except EngineError as e:
            self._emit_utterance_settled(
                emitted=False,
                reason=self._engine_error_reason(e),
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            logger.warning(f"Transcription failed, skipping segment: {e}")
            return

        if outcome.drop_reason is not None:
            self._emit_utterance_settled(
                emitted=False,
                reason=outcome.drop_reason,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            return

        # outcome.result is non-None on success path
        assert outcome.result is not None
        for merged in self._coalescer.push(outcome.result, segment.end_time):
            merged = self._apply_translation_sync(merged)
            self._emit_result(merged)
            self._emit_utterance_settled(
                emitted=True,
                reason=None,
                start_time=merged.start_time,
                end_time=merged.end_time,
            )

    def _flush_coalescer_callback(
        self, now: float, *, force: bool = False
    ) -> None:
        """coalescer.flush() → ``_emit_result`` + settled (feed_audio 用)。

        ``flush()`` が None を返した場合は Tier 2 no-event (logical utterance
        が存在しない、Issue #332 rev6 で defer)。
        """
        flushed = self._coalescer.flush(now, force=force)
        if flushed is None:
            return
        flushed = self._apply_translation_sync(flushed)
        self._emit_result(flushed)
        self._emit_utterance_settled(
            emitted=True,
            reason=None,
            start_time=flushed.start_time,
            end_time=flushed.end_time,
        )

    def _handle_final_segment_for_list(
        self,
        segment: VADSegment,
        outputs: List[TranscriptionResult],
    ) -> None:
        """finalize path: outcome → settled + outputs.append。

        Delivery boundary は list append。settled は append **前** に発火
        (generator path との ordering 整合、Issue #332 rev6)。
        """
        try:
            outcome = self._transcribe_segment(segment)
        except EngineError as e:
            self._emit_utterance_settled(
                emitted=False,
                reason=self._engine_error_reason(e),
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            logger.warning(f"Final transcription failed: {e}")
            return

        if outcome.drop_reason is not None:
            self._emit_utterance_settled(
                emitted=False,
                reason=outcome.drop_reason,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            return

        assert outcome.result is not None
        for merged in self._coalescer.push(outcome.result, segment.end_time):
            merged = self._apply_translation_sync(merged)
            self._emit_utterance_settled(
                emitted=True,
                reason=None,
                start_time=merged.start_time,
                end_time=merged.end_time,
            )
            outputs.append(merged)

    def _flush_coalescer_for_list(
        self,
        outputs: List[TranscriptionResult],
        now: float,
        *,
        force: bool = False,
    ) -> None:
        """coalescer.flush() → settled + outputs.append (finalize 用)。"""
        flushed = self._coalescer.flush(now, force=force)
        if flushed is None:
            return
        flushed = self._apply_translation_sync(flushed)
        self._emit_utterance_settled(
            emitted=True,
            reason=None,
            start_time=flushed.start_time,
            end_time=flushed.end_time,
        )
        outputs.append(flushed)

    async def _handle_final_segment_async(
        self, segment: VADSegment
    ) -> AsyncIterator[TranscriptionResult]:
        """transcribe_async path: outcome → settled + yield (async generator)。

        Delivery ordering (Issue #332 rev6):
        - Drop path: 即時に ``settled(False, reason)`` を発火 (yield なし)
        - Success path: ``yield merged`` の **直前** に ``settled(True, None)``
          を発火。yield 後の code は caller が次の ``__anext__()`` を呼ぶまで
          実行されないため (caller break で永久未発火 bug)、必ず yield 前に
          settled を commit する。
        """
        try:
            outcome = await self._transcribe_segment_async(segment)
        except EngineError as e:
            self._emit_utterance_settled(
                emitted=False,
                reason=self._engine_error_reason(e),
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            logger.warning(f"Async transcription failed: {e}")
            return

        if outcome.drop_reason is not None:
            self._emit_utterance_settled(
                emitted=False,
                reason=outcome.drop_reason,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            return

        assert outcome.result is not None
        for merged in self._coalescer.push(outcome.result, segment.end_time):
            merged = await self._apply_translation_async(merged)
            self._emit_utterance_settled(
                emitted=True,
                reason=None,
                start_time=merged.start_time,
                end_time=merged.end_time,
            )
            yield merged

    async def _flush_coalescer_async(
        self, now: float, *, force: bool = False
    ) -> AsyncIterator[TranscriptionResult]:
        """coalescer.flush() → settled + yield (transcribe_async 用)。"""
        flushed = self._coalescer.flush(now, force=force)
        if flushed is None:
            return
        flushed = await self._apply_translation_async(flushed)
        self._emit_utterance_settled(
            emitted=True,
            reason=None,
            start_time=flushed.start_time,
            end_time=flushed.end_time,
        )
        yield flushed

    def _apply_translation_sync(
        self, result: TranscriptionResult
    ) -> TranscriptionResult:
        """coalescer 出力に翻訳を適用する（同期パス用）。"""
        translated_text, target_language = self._translate_text(result.text)
        if translated_text is not None:
            return replace(
                result,
                translated_text=translated_text,
                target_language=target_language,
            )
        return result

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
        # Layer 0+1 pre-VAD processing: NoiseGate (#291) then transient
        # detector (#295 PR-B). Kept in one helper so feed_audio() and
        # transcribe_async() cannot drift out of sync (the original PR-B
        # missed the async branch and bypassed the detector entirely).
        audio = self._apply_pre_vad_processing(audio)

        # VAD処理
        segments = self._vad.process_chunk(audio, sample_rate)

        for segment in segments:
            if segment.is_final:
                # Issue #332: outcome + settled event を helper に集約
                self._handle_final_segment_callback(segment)
            else:
                # 中間結果は coalescer を経由しない
                interim = self._transcribe_interim(segment)
                if interim:
                    self._result_queue.put(interim)
                    if self._on_interim:
                        self._on_interim(interim)

        # タイムアウト flush（セグメント処理後に実行し、同一チャンク内の
        # マージ機会を先に消費してから残留 pending をタイムアウト判定する）
        self._flush_coalescer_callback(self._vad.current_time)

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

    def finalize(self) -> List[TranscriptionResult]:
        """処理を終了し、残っているセグメントを文字起こし

        Returns:
            最終結果のリスト（0〜2 件）
        """
        results: List[TranscriptionResult] = []

        # 最終 VAD セグメントを先に処理（pending とのマージ機会を保持）
        segment = self._vad.finalize()
        if segment and segment.is_final:
            # Issue #332: outcome + settled event を helper に集約
            self._handle_final_segment_for_list(segment, results)

        # coalescer に残った保留分を強制 flush
        self._flush_coalescer_for_list(results, 0.0, force=True)

        return results

    def _apply_pre_vad_processing(self, audio: np.ndarray) -> np.ndarray:
        """Run NoiseGate (#291) + Layer 1 transient detector (#295 PR-B).

        Shared by ``feed_audio`` (sync path) and ``transcribe_async`` so
        the pre-VAD stack stays a single source of truth. The transient
        detector returns ``(processed_audio, events)``; events are
        currently ignored because PR-B ships without the Layer 2 cooldown
        consumer (that lives in PR-C).
        """
        if self._noise_gate is not None:
            audio = self._noise_gate.process(audio)
        if self._transient_detector is not None:
            audio, _events = self._transient_detector.process(audio)
        return audio

    def reset(self) -> None:
        """状態をリセット"""
        self._vad.reset()
        self._coalescer.reset()
        if self._noise_gate is not None:
            self._noise_gate.reset()
        if self._transient_detector is not None:
            self._transient_detector.reset()
        # 翻訳用文脈バッファをクリア
        self._context_buffer.clear()
        # キューをクリア
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def _should_skip_low_energy(
        self, audio: np.ndarray, kind: str
    ) -> bool:
        """#292 EnergyGate: per-segment energy が threshold 未満なら True。

        Args:
            audio: VADSegment.audio (padding 込み)。
            kind: callsite 種別 ``'final_sync'`` / ``'final_async'`` /
                ``'interim'``。drop counter の分離計上に使用。

        Returns:
            True なら呼び出し側で ``return None`` して engine.transcribe() を
            skip すべき。

        Note:
            ``engine_min_rms_dbfs == -inf`` の場合は energy 計算自体を skip
            (完全 opt-out)。
        """
        if self._engine_min_rms_dbfs <= float("-inf"):
            return False
        energy_dbfs = _segment_energy_dbfs(
            audio,
            self._sample_rate,
            metric=self._engine_energy_metric,
            frame_ms=self._engine_energy_frame_ms,
        )
        if energy_dbfs < self._engine_min_rms_dbfs:
            if kind == "final_sync":
                self._dropped_low_energy_final_sync += 1
            elif kind == "final_async":
                self._dropped_low_energy_final_async += 1
            elif kind == "interim":
                self._dropped_low_energy_interim += 1
            logger.debug(
                "EnergyGate skip (%s, metric=%s, frame=%.1fms): "
                "%.1f dBFS < %.1f dBFS",
                kind,
                self._engine_energy_metric,
                self._engine_energy_frame_ms,
                energy_dbfs,
                self._engine_min_rms_dbfs,
            )
            return True
        return False

    def _transcribe_segment(
        self, segment: VADSegment
    ) -> _SegmentTranscriptionOutcome:
        """セグメントを文字起こし（同期）

        Args:
            segment: VADセグメント

        Returns:
            ``_SegmentTranscriptionOutcome``: success path では
            ``result`` に ``TranscriptionResult``、4 drop path では
            ``drop_reason`` に ``REASON_*`` 定数 (Issue #332)。

        Raises:
            EngineError: engine.transcribe() が raise した場合 (caller catch
                で settled event を発火させる、本 method では catch しない)。
        """
        if len(segment.audio) == 0:
            return _SegmentTranscriptionOutcome.dropped(REASON_EMPTY_AUDIO)
        if self._should_skip_low_energy(segment.audio, "final_sync"):
            return _SegmentTranscriptionOutcome.dropped(REASON_ENERGY_GATE)

        try:
            engine_result = self.engine.transcribe(segment.audio, self._sample_rate)

            # PR-A.1: confidence filter (Issue #308 v3.1)
            # engine_result を unpack せず受け取り、apply_filter() 経由で
            # engine_confidence を見る。Issue #332: None drop は
            # REASON_FILTER_REJECT として outcome に反映。
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
            )
            if engine_result is None:
                return _SegmentTranscriptionOutcome.dropped(REASON_FILTER_REJECT)
            text = engine_result.text
            confidence = engine_result.confidence

            if not text or not text.strip():
                return _SegmentTranscriptionOutcome.dropped(REASON_ENGINE_EMPTY)

            text = text.strip()

            # 翻訳は coalescer 出力後に実行するため、ここではスキップ
            return _SegmentTranscriptionOutcome.success(
                TranscriptionResult(
                    text=text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    is_final=True,
                    confidence=confidence,
                    language=self._source_lang or "",
                    source_id=self.source_id,
                )
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    async def _apply_translation_async(
        self, result: TranscriptionResult
    ) -> TranscriptionResult:
        """coalescer 出力に翻訳を適用する（非同期パス用）。"""
        if not self._translator:
            return result
        loop = asyncio.get_running_loop()
        try:
            translated_text, target_language = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, self._do_translate_direct, result.text
                ),
                timeout=TRANSLATION_TIMEOUT,
            )
            if translated_text is not None:
                return replace(
                    result,
                    translated_text=translated_text,
                    target_language=target_language,
                )
        except asyncio.TimeoutError:
            logger.warning(
                f"Coalesced translation timed out after {TRANSLATION_TIMEOUT}s"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Coalesced translation failed: {e}")
        return result

    async def _transcribe_segment_async(
        self, segment: VADSegment
    ) -> _SegmentTranscriptionOutcome:
        """セグメントを文字起こし（非同期、executor使用）

        Args:
            segment: VADセグメント

        Returns:
            ``_SegmentTranscriptionOutcome``: sync 版と同じ形式 (Issue #332)。

        Raises:
            EngineError: engine.transcribe() が raise した場合。
        """
        if len(segment.audio) == 0:
            return _SegmentTranscriptionOutcome.dropped(REASON_EMPTY_AUDIO)
        if self._should_skip_low_energy(segment.audio, "final_async"):
            return _SegmentTranscriptionOutcome.dropped(REASON_ENERGY_GATE)

        loop = asyncio.get_running_loop()
        try:
            engine_result = await loop.run_in_executor(
                self._executor,
                self.engine.transcribe,
                segment.audio,
                self._sample_rate,
            )

            # PR-A.1: confidence filter (Issue #308 v3.1)
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
            )
            if engine_result is None:
                return _SegmentTranscriptionOutcome.dropped(REASON_FILTER_REJECT)
            text = engine_result.text
            confidence = engine_result.confidence

            if not text or not text.strip():
                return _SegmentTranscriptionOutcome.dropped(REASON_ENGINE_EMPTY)

            text = text.strip()

            # 翻訳は coalescer 出力後に実行するため、ここではスキップ
            return _SegmentTranscriptionOutcome.success(
                TranscriptionResult(
                    text=text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    is_final=True,
                    confidence=confidence,
                    language=self._source_lang or "",
                    source_id=self.source_id,
                )
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
        if self._should_skip_low_energy(segment.audio, "interim"):
            return None

        try:
            engine_result = self.engine.transcribe(segment.audio, self._sample_rate)

            # PR-A.1: confidence filter (Issue #308 v3.1)
            # interim 字幕でも hallucination を弾くため filter 適用 (reviewer Mod 1)。
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
            )
            if engine_result is None:
                return None
            text = engine_result.text  # interim では confidence 未使用

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

    def _log_filter_banner(self) -> None:
        """Confidence filter の起動 banner (PR-A.1 / Issue #308 v3.1)。

        engine 初期化完了時に 1 行 INFO log を出力。default `on` への user 認知を
        担保し、escape 方法 (CLI flag / env var) を case 別に案内する。
        """
        cfg = self._filter_config
        if cfg.mode == "on":
            # PR-A.4.1 (Issue #311): voxtral avg_logprob < -1.0 を追加。
            # PR-A.4.2 (Issue #311): canary も同 token_conf_threshold を共用
            # (Parakeet と同 path、`token_confidence_mean` を populate)。
            # PR-A.4.3 (Issue #311 [#316]): parakeet 英語 (TDT only) も同 path
            # を共用 (preserve_alignments + confidence_cfg で populate)。
            # PR-A.5.1 (Issue #317): reazonspeech も avg_logprob path だが
            # engine-specific threshold (avg_logprob_thresholds dict) で
            # Voxtral と別 calibration。
            # ``avg_logprob_threshold is None`` は user 明示 opt-out の case
            # (Voxtral 経路を完全 off) で、その場合は banner にも出さない。
            parts = [
                f"whispers2t no_speech_prob > {cfg.no_speech_threshold}",
                f"parakeet (ja/en) / canary token_conf < {cfg.token_conf_threshold}",
            ]
            if cfg.avg_logprob_threshold is not None:
                parts.append(
                    f"voxtral avg_logprob < {cfg.avg_logprob_threshold}"
                )
            # PR-A.5.1: engine-specific threshold の clause を for loop で構築
            for engine, thr in sorted(cfg.avg_logprob_thresholds.items()):
                parts.append(f"{engine} avg_logprob < {thr}")
            logger.info(
                "Confidence filter: ON (%s). "
                "Disable: --confidence-filter off or LIVECAP_CONFIDENCE_FILTER=off",
                ", ".join(parts),
            )
        elif cfg.mode == "observe":
            logger.info(
                "Confidence filter: OBSERVE (logging only, no reject)"
            )
        else:
            logger.info("Confidence filter: OFF")

    def close(self) -> None:
        """リソースを解放"""
        total_dropped = (
            self._dropped_low_energy_final_sync
            + self._dropped_low_energy_final_async
            + self._dropped_low_energy_interim
        )
        if total_dropped > 0 and self._engine_min_rms_dbfs > float("-inf"):
            logger.info(
                "EnergyGate dropped %d segments: "
                "%d final-sync, %d final-async, %d interim "
                "(metric=%s, threshold=%.1f dBFS)",
                total_dropped,
                self._dropped_low_energy_final_sync,
                self._dropped_low_energy_final_async,
                self._dropped_low_energy_interim,
                self._engine_energy_metric,
                self._engine_min_rms_dbfs,
            )
        # Layer 1 transient detector telemetry (#295 PR-B).
        if self._transient_detector is not None:
            tel = self._transient_detector.telemetry
            mode = self._transient_detector.config.mode
            if tel.frames_processed > 0:
                logger.info(
                    "TransientDetector (mode=%s) processed %d frames, "
                    "flagged %d as applause-like; per-feature passes: "
                    "rms=%d, flatness=%d, centroid=%d, zcr=%d, onset=%d, voiced=%d",
                    mode,
                    tel.frames_processed,
                    tel.applause_frames,
                    tel.pass_rms,
                    tel.pass_flatness,
                    tel.pass_centroid,
                    tel.pass_zcr,
                    tel.pass_onset,
                    tel.pass_voiced,
                )
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
        for final in self.finalize():
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
            # Pre-VAD layers (NoiseGate + transient detector).
            chunk = self._apply_pre_vad_processing(chunk)

            # VAD処理は軽いのでメインスレッドで実行
            segments = self._vad.process_chunk(chunk, audio_source.sample_rate)

            for segment in segments:
                if segment.is_final:
                    # Issue #332: outcome + settled event を helper に集約
                    async for merged in self._handle_final_segment_async(segment):
                        yield merged
                elif self._on_interim:
                    interim = self._transcribe_interim(segment)
                    if interim:
                        self._on_interim(interim)

            # タイムアウト flush（セグメント処理後）
            async for flushed in self._flush_coalescer_async(
                self._vad.current_time
            ):
                yield flushed

            # 他のタスクに制御を譲る
            await asyncio.sleep(0)

        # 最終セグメント + coalescer flush（finalize のインライン版）
        final_segment = self._vad.finalize()
        if final_segment and final_segment.is_final:
            async for merged in self._handle_final_segment_async(final_segment):
                yield merged

        async for flushed in self._flush_coalescer_async(0.0, force=True):
            yield flushed

    @property
    def vad_state(self):
        """現在のVAD状態"""
        return self._vad.state

    @property
    def sample_rate(self) -> int:
        """エンジンが要求するサンプリングレート"""
        return self._sample_rate
