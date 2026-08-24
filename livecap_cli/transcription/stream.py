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
    Union,
)

import numpy as np

from ..audio import ENERGY_METRICS, should_drop_low_energy
# Runtime import (codex-review on #309): TYPE_CHECKING のみだと
# typing.get_type_hints() で NameError になるため通常 import に格上げ。
# `livecap_cli.engines.base_engine` は `livecap_cli.transcription` を import
# していないので循環依存はない (本ファイルで grep 確認済)。
from ..engines.base_engine import (
    TranscriptionResult as EngineTranscriptionResult,
)
from ..vad import VADConfig, VADProcessor, VADSegment
from .confidence_filter import FilterConfig, apply_filter
from .result import InterimResult, TranscriptionResult, TranslationState
from .result_coalescer import ResultCoalescer
from .translation_status import TranslationErrorType, TranslationStatusEvent
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

# 翻訳の待ち時間 (秒)。環境変数 LIVECAP_TRANSLATION_TIMEOUT で上書き可能。
#
# Issue #402 D10: 既定を 10.0 -> 5.0 に変更した。これはリアルタイム字幕の経路で、
# **遅れて届いた翻訳は字幕として無価値**である (今話している内容と重なって出る)。
# 10 秒は明確に長すぎる一方、実測 (Session 再利用時の中央値 155-191ms、観測した
# 最悪 1331ms) に対して 5 秒は 4 倍近い余裕があり、回線の遅い環境や重いローカル
# モデルでも正常な翻訳を切らずに済む。
#
# なお本値は「待つのをやめる時刻」であって「翻訳処理が止まる時刻」ではない。
# 実行中の 1 試行を外から止める手段は無く、そちらは translator 自身の timeout の
# 役目である (:attr:`BaseTranslator.estimated_attempt_seconds` 参照)。
_DEFAULT_TRANSLATION_TIMEOUT = 5.0


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

#: ``close()`` が実行中の翻訳を待つとき、「まだ待っている」と知らせるまでの秒数。
#: **待ち切る上限ではない** — 打ち切ると借用中の translator を owner が cleanup
#: することになり、待つ理由そのものが失われる (下記 ``_drain_translation``)。
TRANSLATION_DRAIN_NOTICE_SECONDS = 5.0


def drain_translation(inflight: "concurrent.futures.Future") -> None:
    """実行中の翻訳が終わるまで待つ。**打ち切らない。**

    翻訳 worker は translator を**借りている**だけで、所有者 (CLI / GUI) は
    ``close()`` が返った直後に ``cleanup()`` する。上限を設けて諦めると、まさに
    待つ理由だったケース (時間のかかっている翻訳) で、使用中の
    ``requests.Session`` を閉じさせることになる。しかも参照を捨てるので、後から
    待ち直すこともできない。

    上限の候補だった ``estimated_attempt_seconds`` は PR 1 で **soft estimate で
    あって上限の保証ではない**と整理した値であり、resource safety の根拠には
    使えない。

    打ち切らなくても失うものは無い: ``ThreadPoolExecutor`` の worker は
    **non-daemon** で、CPython は interpreter 終了時にこれを join する
    (``concurrent.futures.thread._python_exit``)。待たずに返したところで、ハング
    した worker からプロセスが解放されるわけではない。1 試行を打ち切るのは
    translator 自身の timeout の役目である。

    ただし黙って止まって見えるのは困るので、長引いたら 1 度だけ知らせる。
    """
    notified = False
    while True:
        try:
            # exception() は結果の例外を送出せずに完了を待つ。
            inflight.exception(timeout=TRANSLATION_DRAIN_NOTICE_SECONDS)
            return
        except concurrent.futures.TimeoutError:
            if not notified:
                notified = True
                logger.warning(
                    "Waiting for an in-flight translation to finish before releasing "
                    "the translator. If this hangs, the translator has no timeout of "
                    "its own."
                )


@dataclass(frozen=True, slots=True)
class _TranslationOutcome:
    """1 回の翻訳試行の結果 (内部型)。

    以前は ``(translated_text, target_language)`` のタプルで、失敗は
    ``(None, None)`` に潰していた。**呼び出し側が理由を失う**ため、失敗を表に出す
    こと自体ができなかった (Issue #402 D1)。

    worker スレッドの中で作られ、caller 側で :meth:`StreamTranscriber._settle_translation`
    が解釈する。**worker 内で callback を呼ばない**ための受け渡し役でもある。
    """

    state: TranslationState
    translated_text: Optional[str] = None
    target_language: Optional[str] = None
    error_type: Optional[TranslationErrorType] = None
    message: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.state == "failed"


def _classify_translation_error(exc: BaseException) -> TranslationErrorType:
    """翻訳例外を通知用の粒度へ落とす。

    ``TranslationNetworkError`` は待てば直る可能性があり、それ以外は設定や
    レイアウト変更など待っても直らないもの。「待てば直るか」は
    :attr:`TranslationStatusEvent.recoverable` がこの種別から導出する。
    """
    from ..translation.exceptions import TranslationNetworkError

    if isinstance(exc, TranslationNetworkError):
        return "network"
    return "fatal"


def _sanitized_message(exc: BaseException) -> str:
    """通知に載せてよい文言だけを取り出す。

    翻訳対象テキストは Google への GET query に入るため、通信ライブラリの例外文字列
    には発話内容が URL ごと含まれ得る。adapter 側で ``from None`` と構造化フィールド
    により除去済みだが (Issue #402 D8)、ここは**イベントとして GUI まで届く**経路
    なので、adapter 以外の例外が紛れ込んでも発話が出ないよう型名に落とす。
    """
    from ..translation.exceptions import TranslationError

    if isinstance(exc, TranslationError):
        # adapter が sanitize 済み。provider/reason/status_code しか持たない。
        return str(exc)
    return type(exc).__name__


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

            # resolved 値をログする (CLAUDE.md の pre-1.0 方針)。
            # translator 自身の見積が待ち時間より大きいと、毎回 timeout してから
            # 次の segment が skip される — 設定の食い違いが見えるようにしておく
            # (adapter の timeout を配線するのは生成側: CLI は #403、GUI は
            # livecap-gui#407)。
            estimated = getattr(translator, "estimated_attempt_seconds", None)
            logger.info(
                "Translation: %s, waiting up to %.1fs per segment%s",
                translator.get_translator_name(),
                TRANSLATION_TIMEOUT,
                f" (translator estimates {estimated:.1f}s)" if estimated else "",
            )
            if estimated and estimated > TRANSLATION_TIMEOUT:
                logger.warning(
                    "Translator estimates %.1fs per attempt but we only wait %.1fs; "
                    "segments will time out and the next ones will be skipped. "
                    "Lower the translator's timeout or raise LIVECAP_TRANSLATION_TIMEOUT.",
                    estimated,
                    TRANSLATION_TIMEOUT,
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
        # Issue #402 D1: 翻訳エンジンの状態通知 (opt-in callback)
        self._on_translation_status: Optional[
            Callable[[TranslationStatusEvent], None]
        ] = None

        # 翻訳エンジンの健康状態。segment ごとに通知を連打しないための記憶
        # (Issue #402 D1)。失敗が続く間は黙り、直ったら 1 回だけ知らせる。
        self._translation_healthy = True
        self._translation_failures = 0
        self._translation_skips = 0
        # 翻訳は ASR とは別の worker で走らせる。共用していたため、居座った翻訳が
        # 文字起こし自体を止めていた (既定 max_workers=1)。遅延生成 (Issue #402 D2)。
        self._translation_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # in-flight は常に 1 件。前の翻訳が終わっていなければ今回は飛ばす
        # (Issue #402 D10)。順番を守って遅れて全部出すより、落とす方が字幕としては良い。
        self._translation_inflight: Optional[concurrent.futures.Future] = None
        # reset を跨いで走っている worker が、新セッションの文脈バッファへ古い発話を
        # 書き戻さないための世代番号 (Issue #402)。
        self._translation_generation = 0

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
        on_translation_status: Optional[
            Callable[[TranslationStatusEvent], None]
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
            on_translation_status: 翻訳エンジンが壊れた / 直ったときに発火する
                (Issue #402)。**segment ごとには呼ばれない** — 状態が変わったとき
                だけ 1 回。失敗が続く間は黙り、復旧したら ``recovered`` が出る。
                個々の字幕が原文のままである理由は
                ``TranscriptionResult.translation_state`` を見ること。

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
        self._on_translation_status = on_translation_status

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

    def _emit_translation_status(self, event: TranslationStatusEvent) -> None:
        """``TranslationStatusEvent`` の single funnel (Issue #402 D1)。

        **caller 側から呼ぶこと。** worker スレッドの中で callback を呼ぶと、
        consumer が UI スレッドを前提にしていた場合に壊れる。

        callback の例外はここで握る — 通知の失敗で文字起こしまで止まるのは本末転倒
        (この hook は「翻訳が壊れた」ことを伝えるためのものであり、それ自身が新しい
        壊れ方を持ち込んではいけない)。
        """
        if self._on_translation_status is None:
            return
        try:
            self._on_translation_status(event)
        except Exception:  # noqa: BLE001 - consumer の落ち度で転写を止めない
            logger.warning(
                "on_translation_status callback raised; continuing", exc_info=True
            )

    def _settle_translation(
        self, result: TranscriptionResult, outcome: _TranslationOutcome
    ) -> TranscriptionResult:
        """翻訳結果を result へ反映し、状態が変わっていれば通知する。

        sync / async 双方がここを通る唯一の経路。3 箇所に散っていた
        ``except Exception: logger.warning(...)`` を 1 つに集約したもの。

        状態遷移は 3 つだけ: healthy->failed で通知、failed->failed は沈黙、
        failed->healthy で復旧通知 (Issue #402 D1)。
        """
        translator_name = (
            self._translator.get_translator_name() if self._translator else "unknown"
        )

        if outcome.failed:
            self._translation_failures += 1
            if self._translation_healthy:
                self._translation_healthy = False
                logger.warning(
                    "Translation is failing (%s): %s", outcome.error_type, outcome.message
                )
                self._emit_translation_status(
                    TranslationStatusEvent.failed(
                        translator_name,
                        outcome.error_type or "fatal",
                        # message は必須。理由の分からない失敗通知は受け手が
                        # ユーザへ何も説明できない。
                        outcome.message or "translation failed",
                    )
                )
        elif outcome.state in ("translated", "empty"):
            # skip は「壊れた」ではないので健康状態を動かさない。輻輳時の方針であり、
            # それで復旧扱いにすると failed 通知が skip のたびに解除されてしまう。
            if not self._translation_healthy:
                self._translation_healthy = True
                logger.info("Translation recovered")
                self._emit_translation_status(
                    TranslationStatusEvent.recovered(translator_name)
                )
        elif outcome.state == "skipped_busy":
            self._translation_skips += 1

        if outcome.translated_text is not None:
            return replace(
                result,
                translated_text=outcome.translated_text,
                target_language=outcome.target_language,
                translation_state=outcome.state,
            )
        return replace(result, translation_state=outcome.state)

    def _translation_worker(self) -> concurrent.futures.ThreadPoolExecutor:
        """翻訳専用の worker (初回使用時に生成)。

        ASR と共用していた頃は、居座った翻訳が文字起こし自体をブロックしていた
        (Issue #402 D2)。``max_workers=1`` なので後続はキューへ回り、
        in-flight は構造的に 1 件になる。
        """
        if self._translation_executor is None:
            self._translation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="livecap-translate"
            )
        return self._translation_executor

    def _remember_context(self, text: str, generation: int) -> None:
        """文脈バッファへ追加する。ただし reset を跨いだ分は捨てる。

        翻訳は worker で走るので、``reset()`` の後に完了することがある。そのまま
        追加すると**前セッションの発話が新セッションの文脈に混ざる** (Issue #402)。
        """
        if generation != self._translation_generation:
            logger.debug("Dropping context from a previous session (reset in between)")
            return
        self._context_buffer.append(text)

    def _translation_busy(self) -> bool:
        """前の翻訳がまだ走っているか (Issue #402 D10)。

        timeout した future は誰も読まないまま残る。それを見て次を飛ばすことで、
        古い翻訳が積み上がって「数秒前の発話の字幕が今の音声に重なる」状態を防ぐ。
        """
        inflight = self._translation_inflight
        return inflight is not None and not inflight.done()

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
        return self._settle_translation(result, self._translate_text(result.text))

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

        # 翻訳の状態も新セッション扱いにする (Issue #402)。持ち越すと、前セッション
        # の failed のせいで次の障害が通知されず、逆に最初の成功が前セッションに
        #対する recovered として出てしまう。
        self._translation_healthy = True
        self._translation_failures = 0
        self._translation_skips = 0
        # **in-flight は捨てない。** 参照だけ消すと、走っている worker と新しい翻訳が
        # 同じ translator / requests.Session を並行利用する。単一 worker のまま
        # 残しておけば、終わるまで新しい segment は skipped_busy になる。
        self._translation_generation += 1
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
        # #366 Phase 2: 判定式は file mode と共有の単一判定点へ委譲
        should_drop, energy_dbfs = should_drop_low_energy(
            audio,
            self._sample_rate,
            threshold_dbfs=self._engine_min_rms_dbfs,
            metric=self._engine_energy_metric,
            frame_ms=self._engine_energy_frame_ms,
        )
        if should_drop:
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
            # Issue #351: is_interim=False を明示 (final sync path)
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
                is_interim=False,
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
        """coalescer 出力に翻訳を適用する（非同期パス用）。

        通知は :meth:`_settle_translation` に集約する — worker の中では callback を
        呼ばない (Issue #402 D1)。
        """
        if not self._translator:
            return self._settle_translation(
                result, _TranslationOutcome(state="not_requested")
            )

        # 前の翻訳が残っていれば飛ばす。同期パスと同じ single-flight 契約
        # (Issue #402 D10)。
        if self._translation_busy():
            logger.debug("Translation still busy; skipping this segment")
            return self._settle_translation(
                result, _TranslationOutcome(state="skipped_busy")
            )

        # **executor へ直接 submit する。** ``loop.run_in_executor()`` は
        # ``asyncio.Future`` を返すが、``_shutdown_translation_worker()`` は
        # ``concurrent.futures.Future`` として扱う (``exception(timeout=...)``)。
        # 型を揃えないと async で timeout した後の ``close()`` が TypeError になる。
        # 待機側だけ ``wrap_future`` で asyncio 側へ持ち上げる (Issue #402)。
        #
        # generation も **submit 前**に読む。worker の中で読むと、submit から
        # worker 本体が動き出すまでの間に reset された場合に新しい世代を拾う。
        generation = self._translation_generation
        future = self._translation_worker().submit(
            self._do_translate_direct, result.text, generation
        )
        self._translation_inflight = future

        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)), timeout=TRANSLATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            # shield しているので future 自体は生き続ける。結果は誰も読まないため
            # 古い翻訳が後から字幕に混ざることはなく、次の segment は
            # _translation_busy() を見て飛ばす。
            outcome = _TranslationOutcome(
                state="failed",
                error_type="timeout",
                message=f"Translation did not finish within {TRANSLATION_TIMEOUT:.1f}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - 分類して通知へ回す
            error_type = _classify_translation_error(e)
            outcome = _TranslationOutcome(
                state="failed",
                error_type=error_type,
                message=_sanitized_message(e),
            )

        return self._settle_translation(result, outcome)

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
            # Issue #351: is_interim=False を明示 (final async path)
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
                is_interim=False,
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

    def _do_translate_direct(self, text: str, generation: int) -> _TranslationOutcome:
        """テキストを翻訳する (executor 提出なし、直接実行)。

        Returns:
            :class:`_TranslationOutcome`。**失敗を潰さない** — 以前は
            ``(None, None)`` に落としており、呼び出し側が理由を失っていた
            (Issue #402 D1)。

        Note:
            タイムアウト制御は呼び出し側が担当。executor への二重提出を避けるため
            ここでは submit しない。

            **これは worker スレッドの中で動く。** callback を呼んではいけない —
            通知は caller 側の :meth:`_settle_translation` が行う。
        """
        if not self._translator or not text:
            return _TranslationOutcome(state="not_requested")

        # generation は **caller が submit 前に決めて渡す**。ここ (worker の中) で
        # 読むと、submit からこの行に到達するまでの間に reset された場合に
        # **新しい世代を読んでしまい**、旧セッションの発話が新セッションの文脈へ
        # 入る (Issue #402)。

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
            self._remember_context(text, generation)

            if not trans_result.text.strip():
                return _TranslationOutcome(state="empty")
            return _TranslationOutcome(
                state="translated",
                translated_text=trans_result.text,
                target_language=self._target_lang,
            )

        except Exception as e:  # noqa: BLE001 - 分類して caller へ渡す
            error_type = _classify_translation_error(e)
            # 翻訳失敗しても文脈バッファには追加（次の翻訳の文脈として使用）
            self._remember_context(text, generation)
            return _TranslationOutcome(
                state="failed",
                error_type=error_type,
                message=_sanitized_message(e),
            )

    def _translate_text(self, text: str) -> _TranslationOutcome:
        """テキストを翻訳する (タイムアウト付き、同期パス用)。

        Returns:
            :class:`_TranslationOutcome`。失敗・タイムアウト・輻輳スキップを
            区別して返す (Issue #402 D1 / D10)。

        Note:
            TRANSLATION_TIMEOUT (既定 5 秒、``LIVECAP_TRANSLATION_TIMEOUT``) を超過した場合、
            翻訳をスキップして ASR パイプラインを継続する。

            同期パス（feed_audio, transcribe_sync）から呼ばれる想定。
            非同期パス（transcribe_async）では _do_translate_direct を使用。
        """
        if not self._translator or not text:
            return _TranslationOutcome(state="not_requested")

        # 前の翻訳がまだ走っているなら今回は飛ばす (Issue #402 D10)。キューへ積むと
        # 数秒前の発話に対する字幕が今の音声に重なって出てしまう。
        if self._translation_busy():
            logger.debug("Translation still busy; skipping this segment")
            return _TranslationOutcome(state="skipped_busy")

        generation = self._translation_generation

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

        # 翻訳専用 worker。ASR と共用していた頃は居座った翻訳が文字起こしを
        # 止めていた (Issue #402 D2)。
        future = self._translation_worker().submit(do_translate)
        self._translation_inflight = future

        try:
            translated = future.result(timeout=TRANSLATION_TIMEOUT)

            # 文脈バッファに追加
            self._remember_context(text, generation)

            if not translated.strip():
                return _TranslationOutcome(state="empty")
            return _TranslationOutcome(
                state="translated",
                translated_text=translated,
                target_language=self._target_lang,
            )

        except concurrent.futures.TimeoutError:
            # future はそのまま残す。誰も結果を読まないので、あとから完了しても
            # 古い翻訳が字幕に混ざることはない (Issue #402 D10)。次の segment は
            # _translation_busy() を見て飛ばす。
            # タイムアウトしても文脈バッファには追加
            self._remember_context(text, generation)
            return _TranslationOutcome(
                state="failed",
                error_type="timeout",
                message=f"Translation did not finish within {TRANSLATION_TIMEOUT:.1f}s",
            )

        except Exception as e:  # noqa: BLE001 - 分類して caller へ渡す
            error_type = _classify_translation_error(e)
            # 翻訳失敗しても文脈バッファには追加（次の翻訳の文脈として使用）
            self._remember_context(text, generation)
            return _TranslationOutcome(
                state="failed",
                error_type=error_type,
                message=_sanitized_message(e),
            )

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
            # Issue #351: is_interim=True を明示 (interim path)。 calibration
            # harness (parse_observe.py) が default で除外する想定 (別 PR)。
            engine_result = apply_filter(
                engine_result,
                self._filter_config,
                source_id=self.source_id,
                engine_name=self._engine_name,
                is_interim=True,
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
        # 翻訳のテレメトリ (Issue #402)。skip は障害ではなく輻輳時の方針なので、
        # 失敗と分けて出す — 混ぜると「翻訳が壊れている」と読めてしまう。
        if self._translation_failures or self._translation_skips:
            logger.info(
                "Translation: %d failed, %d skipped (busy)",
                self._translation_failures,
                self._translation_skips,
            )

        self._executor.shutdown(wait=False)
        # 明示的な close は **翻訳が translator を使い終わるまで待つ**。
        self._shutdown_translation_worker(drain=True)

    def _shutdown_translation_worker(self, *, drain: bool) -> None:
        """翻訳 worker を解放する。ASR とは別に持っているので個別に畳む。

        Args:
            drain: 実行中の翻訳が終わるまで待つか。

        **明示的な ``close()`` では待つ必要がある** (Issue #402)。translator は
        呼び出し側 (CLI / GUI) が所有しており、``close()`` の直後に
        ``translator.cleanup()`` が呼ばれる。待たずに返すと、**借りている
        ``requests.Session`` を使っている最中に閉じられる**ことになり、所有権の
        契約が成立しない。``cancel_futures=True`` は実行中の future を止めない。

        デストラクタからは待たない — GC 中にブロックするのは危険であり、
        そこでの厳密さより安全に抜けることを優先する。
        """
        inflight = self._translation_inflight
        if drain and inflight is not None and not inflight.done():
            drain_translation(inflight)

        executor = getattr(self, "_translation_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._translation_executor = None
        self._translation_inflight = None

    def __del__(self) -> None:
        """デストラクタ: リソースを確実に解放"""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass  # GC 時のエラーは無視
        try:
            # GC 中にブロックしない。厳密さより安全に抜けることを優先する。
            self._shutdown_translation_worker(drain=False)
        except Exception:
            pass

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
