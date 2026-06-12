"""ASR engine 内部の信頼度信号を見て、字幕に出る前に「非音声」判定 output を弾く filter。

PR-A.0 ([#309]) で expose した `TranscriptionResult.engine_confidence` を読み、
WhisperS2T の ``no_speech_prob`` / Parakeet_ja の ``token_confidence_mean`` で
非音声と判定された output を ``None`` drop する (Phase 1 Layer 3 / Issue #308 v3.1)。

設計判断 (Issue #308 v3.1 + codex-review #310 Item 4):

- ``apply_filter()`` は ``Optional[TranscriptionResult]`` を返す。``None`` 戻りは
  caller (`stream.py` 3 経路) で ``return None`` の silent drop を意味する。
  旧設計 (``text=""`` で後段に渡す) は coalescer 側で空 text propagation が
  起きるため不採用。
- ``mode`` が ``"off"`` の時は filter を完全 skip、log 出力もしない (旧挙動)。
- ``mode`` が ``"observe"`` の時は **pass / reject 両方を JSON 構造化 log** で
  出力、reject はしない (PR-A.3 calibration 用の data 収集)。reject だけ
  だと閾値マージン / speech recall 側の安全域を解析できないため、必ず
  pass 側も記録する (codex-review #310 Item 4 対応)。
- ``mode`` が ``"on"`` (default) の時は reject 時に JSON log + ``None`` 返却。
  pass 側は log しない (production で spam 防止)。
- ``engine_confidence.is_available is False`` の結果は pass-through
  (fail-open)。qwen3asr の auto-detect mode (``--language=auto``、wrapper
  fallback path) や、各 engine が score 抽出に失敗した case が該当する。
  fail-open は意図的な設計で、信号不在の場合に silently 全 reject される
  ことを防ぐ。Voxtral は PR-A.4.1 ([#311]) から
  ``avg_logprob`` を populate するため filter 対象 (下記 strict gate)。Canary
  は PR-A.4.2 ([#311]) から ``token_confidence_mean`` を populate するため
  filter 対象 (Parakeet_ja と同 ``token_conf_threshold`` を共用)。Parakeet
  英語は PR-A.4.3 ([#316]) から同 ``token_confidence_mean`` を populate する
  (NeMo TDT + ``preserve_alignments`` 経由、Parakeet_ja と同 helper を流用)。
  **ReazonSpeech は PR-A.5.1 ([#317]) から ``avg_logprob`` (sherpa-onnx の
  ``ys_log_probs`` mean、負の log probability) を populate するため filter
  対象 (Voxtral と同 path、ただし engine-specific threshold を使用)。
  qwen3asr は PR-A.5.2 ([#318]) から wrapper bypass + ``output_scores=True
  + repetition_penalty=1.1 + no_repeat_ngram_size=3`` 経由で ``avg_logprob``
  を populate (両言語 en/ja で Phase 1 probe 確認、threshold -0.3)。**
- threshold は実機 verify 済の値を default (WhisperS2T: 0.5、Parakeet (ja/en)
  / Canary: 0.005、Voxtral: -1.0 [PR-A.4.1 smoke verify 2026-06-11]、
  ReazonSpeech: -0.2 [PR-A.5.1 smoke verify 2026-06-11]、qwen3asr: -0.3
  [PR-A.5.2 smoke verify 2026-06-12])。``FilterConfig``
  の constructor kwargs で override 可能 (sweep harness が PR-A.3 で
  programmatic に threshold を sweep するため、``avg_logprob_threshold=None``
  で **global fallback only** opt-out 可能。engine-specific dict entries は
  独立に active のまま残る — 完全 off にしたい場合は dict も空にする
  必要あり、後述)。
- PR-A.5.1 (Issue #317) で engine-specific threshold dict
  (``avg_logprob_thresholds: Dict[str, float]``) を追加。Voxtral (margin
  -1.0) と ReazonSpeech (margin -0.2) は同 ``avg_logprob`` field を共用する
  が分布が桁違いのため engine-specific threshold が必要。``voxtral`` は
  margin +1.0 が大きく global ``avg_logprob_threshold = -1.0`` で十分なため
  engine-specific dict には load せず global fallback path を使う (design
  choice、技術負債ではない)。**Opt-out 規約 (codex-review Point 4)**:
  ``avg_logprob_threshold=None`` のみで global は off になるが
  ``avg_logprob_thresholds["reazonspeech"]`` 等は active のまま。完全 off
  には ``FilterConfig(avg_logprob_threshold=None, avg_logprob_thresholds={})``
  と両方を空に。

Log format: ``confidence_filter[<mode>]: <JSON>``。JSON は PR-A.3 parser が
壊れにくい安定 format。schema は ``_decision_to_dict()`` を参照。

PR-A.1 では CLI flag (``--confidence-filter``) と env var
(``LIVECAP_CONFIDENCE_FILTER``) で mode のみ override 可能。per-threshold CLI flag
は PR-A.3 calibration 結果次第で追加判断 (本 PR では追加しない)。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from ..engines.base_engine import EngineConfidence, TranscriptionResult

logger = logging.getLogger(__name__)


FilterMode = Literal["off", "observe", "on"]


@dataclass(frozen=True)
class FilterConfig:
    """Confidence filter の設定。

    Attributes:
        mode: ``"off"`` (filter なし) / ``"observe"`` (判定 + log のみ) /
            ``"on"`` (default、reject 適用)。
        no_speech_threshold: WhisperS2T の ``no_speech_prob`` が **これより上**
            なら reject (PR-A.0 実機 verify: speech 0.036 vs non-speech 0.66、
            0.5 で 100% 分類可能)。
        token_conf_threshold: Parakeet_ja の ``token_confidence_mean`` が
            **これより下** なら reject (PR-A.0 実機 verify: speech 0.05 vs
            non-speech 0.0000029、0.005 で 100% 分類可能)。
        avg_logprob_threshold: Voxtral 等の ``avg_logprob`` 用 strict-gated
            **global default** threshold (PR-A.4.1 Issue #311)。
            ``no_speech_prob`` / ``token_confidence_mean`` の両方が None の
            時のみ評価される (WhisperS2T 退行回避)。PR-A.4.1 実機 smoke
            verify (2026-06-11) で Voxtral speech mean -0.42 vs non-speech
            -1.53 → margin +1.0 → midpoint -1.02 → ``-1.0`` で 100% 分類可能。
            ``avg_logprob_thresholds`` dict に entry がない engine で fallback。
        avg_logprob_thresholds: **engine-specific** ``avg_logprob`` threshold の
            dict (PR-A.5.1 Issue #317)。``engine_name`` で lookup、entry なし
            時は ``avg_logprob_threshold`` (global) に fallback。Voxtral は
            margin +1.0 が大きく global ``-1.0`` で十分なため dict に load
            しない (design choice)。ReazonSpeech は ``-0.2`` を default load
            (PR-A.5.1 smoke verify 2026-06-11 で speech mean -0.11 vs
            non-speech -0.45 → margin +0.34 → threshold -0.2 で分類)。

            **Opt-out 規約 (PR-A.5.1 codex-review Point 4 で明示)**:

            - ``avg_logprob_threshold=None``: **global fallback のみ off**。
              ``avg_logprob_thresholds`` dict に entry がある engine
              (ReazonSpeech default) は引き続き active のまま。
            - 全 avg_logprob 判定を完全 off にしたい場合は **両方を空に**:
              ``FilterConfig(avg_logprob_threshold=None,
              avg_logprob_thresholds={})`` を渡す。
            - ReazonSpeech のみ off にしたい場合は dict から削除 or
              ``FilterConfig(avg_logprob_thresholds={})`` (Voxtral は
              ``avg_logprob_threshold`` 経由で fallback)。
        compression_ratio_threshold: 未使用予約 field (将来拡張)。
    """

    mode: FilterMode = "on"
    no_speech_threshold: float = 0.5
    token_conf_threshold: float = 0.005
    avg_logprob_threshold: Optional[float] = -1.0
    avg_logprob_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "reazonspeech": -0.2,  # PR-A.5.1 ([#317]) smoke verify 2026-06-11
        # PR-A.5.2 ([#318]) qwen3asr: -0.3 を default load。両言語 (en/ja) で
        # `repetition_penalty=1.1 + no_repeat_ngram_size=3` 適用後の Phase 1
        # probe 値 — EN: speech -0.05、applause -1.08、margin +0.21 / JA:
        # speech -0.20、applause -0.46、desk_tap -0.50、margin +0.27 →
        # threshold -0.3 で両言語 safe (JA speech margin +0.10)。Phase 4 unit
        # test で display string "Qwen3-ASR 0.6B" → _engine_id_from_name() で
        # "qwen3-asr" に normalize されることを pin (PR-A.5.1 codex Point 1
        # learning を pre-empt)。
        "qwen3-asr": -0.3,  # PR-A.5.2 ([#318]) smoke verify 2026-06-12
    })
    compression_ratio_threshold: Optional[float] = None


@dataclass(frozen=True)
class FilterDecision:
    """Filter 判定の structured log entry。

    PR-A.3 calibration が parse 可能な形式で、reject 理由を engine_confidence
    込みで永続化できる。

    Attributes:
        source_id: StreamTranscriber の source 識別子 (mic 名 / file path 等)。
        engine: ``engine.get_engine_name()`` の値。
        text: ASR engine が生成した text (reject されても保持)。
        decision: ``"pass"`` か ``"reject"``。observe モードでも reject 判定は記録。
        reason: reject 理由 (例: ``"no_speech_prob 0.66 > 0.5"``)。pass の場合 None。
        engine_confidence: 判定に使った engine_confidence (signal 全体)。
    """

    source_id: str
    engine: str
    text: str
    decision: Literal["pass", "reject"]
    reason: Optional[str]
    engine_confidence: EngineConfidence


def _engine_id_from_name(engine_name: Optional[str]) -> Optional[str]:
    """Display engine name から threshold lookup 用 ID を抽出 (PR-A.5.1 codex-review Point 1)。

    ``StreamTranscriber`` は ``engine.get_engine_name()`` の戻り値を
    ``apply_filter(engine_name=...)`` に渡す。これは display 用 string
    (例: ``"ReazonSpeech K2 (CPU, Int8)"``) で、各 engine が
    ``self.engine_name`` attribute に持つ ID (例: ``"reazonspeech"``) と
    異なる。本 helper は前者から後者を抽出して
    ``FilterConfig.avg_logprob_thresholds`` の dict key と照合可能にする。

    抽出ロジック (strip + lowercase + first whitespace-separated word):

    - ``"ReazonSpeech K2 (CPU, Int8)"`` → ``"reazonspeech"``
    - ``"ReazonSpeech K2 (CPU, Float32)"`` → ``"reazonspeech"``
    - ``"WhisperS2T base"`` → ``"whispers2t"``
    - ``"voxtral"`` → ``"voxtral"``
    - ``"canary"`` → ``"canary"``
    - ``"MockEngine"`` → ``"mockengine"``
    - ``None`` / ``""`` / 空白のみ → ``None``

    既存 engine が ``engine_name`` attribute (ID) として小文字単一 word を
    持つ慣習 (``reazonspeech_engine.py:41`` ``self.engine_name = 'reazonspeech'``、
    ``voxtral_engine.py:148``、``canary_engine.py:104``、
    ``whispers2t_engine.py:161``) と整合する。
    """
    if not engine_name:
        return None
    stripped = engine_name.strip()
    if not stripped:
        return None
    return stripped.lower().split()[0]


def should_reject(
    result: TranscriptionResult,
    config: FilterConfig,
    engine_name: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Filter 判定。``(rejected, reason)`` を返す。

    判定順位:

    1. ``engine_confidence.is_available is False`` → fail-open (pass、reason=None)
    2. ``no_speech_prob > config.no_speech_threshold`` → reject (WhisperS2T 主)
    3. ``token_confidence_mean < config.token_conf_threshold`` → reject
       (Parakeet (ja/en) / Canary 主)
    4. **strict-gated** ``avg_logprob < threshold`` → reject
       (Voxtral / ReazonSpeech 主、PR-A.4.1 / PR-A.5.1 で追加)。threshold は
       ``config.avg_logprob_thresholds[engine_name]`` (engine-specific) →
       ``config.avg_logprob_threshold`` (global fallback) の順で lookup。
    5. それ以外 → pass

    各 signal は None を許容 (engine ごとに populate される field が異なるため)。

    PR-A.4.1 (Issue #311 v2.1) の strict gating ルール:

    - WhisperS2T は ``no_speech_prob`` **と** ``avg_logprob`` を両方 populate
      する (``whispers2t_engine.py:18-77``)。avg_logprob の global 判定を
      入れると、no_speech_prob が pass でも avg_logprob で false reject される
      退行が起きる。
    - そのため avg_logprob 判定は ``no_speech_prob is None`` AND
      ``token_confidence_mean is None`` の時のみ評価する (strict gate)。
    - Voxtral / ReazonSpeech のように avg_logprob だけ populate する engine
      では active になり、WhisperS2T / Parakeet (ja/en) / Canary は早期
      return で avg_logprob 経路に到達しない (退行ゼロ)。

    PR-A.5.1 (Issue #317) の engine-specific threshold:

    - Voxtral と ReazonSpeech は ``avg_logprob`` field を共用するが分布が
      桁違い (Voxtral speech -0.42 vs ReazonSpeech speech -0.11、Voxtral
      non-speech -1.53 vs ReazonSpeech non-speech -0.45)。global ``-1.0``
      threshold は ReazonSpeech には機能しない (全 pass)。
    - ``config.avg_logprob_thresholds: Dict[str, float]`` で engine-specific
      threshold を lookup。dict に entry がない engine (e.g. ``voxtral``) は
      ``config.avg_logprob_threshold`` (global default ``-1.0``) を使用
      (Voxtral は margin +1.0 が大きく global で十分、design choice)。
    - user が ``avg_logprob_threshold=None`` を渡すと **global fallback の
      みが off** になる。engine-specific dict (``avg_logprob_thresholds``)
      に entry がある engine (ReazonSpeech 等) は引き続き active。avg_logprob
      判定経路を完全 opt-out するには ``FilterConfig(avg_logprob_threshold
      =None, avg_logprob_thresholds={})`` のように両方を空にする
      (PR-A.5.1 codex-review Point 4 で仕様明示)。
    """
    ec = result.engine_confidence
    if not ec.is_available:
        return False, None

    if (
        ec.no_speech_prob is not None
        and ec.no_speech_prob > config.no_speech_threshold
    ):
        return True, (
            f"no_speech_prob {ec.no_speech_prob:.3f} > {config.no_speech_threshold}"
        )

    if (
        ec.token_confidence_mean is not None
        and ec.token_confidence_mean < config.token_conf_threshold
    ):
        return True, (
            f"token_confidence_mean {ec.token_confidence_mean:.5f} "
            f"< {config.token_conf_threshold}"
        )

    # PR-A.4.1 / PR-A.5.1: avg_logprob は他 signal 不在時のみ判定。
    # WhisperS2T も avg_logprob を populate するが、no_speech_prob path で
    # 既に処理済のためここでは見ない (false positive 回避)。
    # engine-specific threshold (PR-A.5.1) → global fallback の順で lookup。
    # **PR-A.5.1 codex-review Point 1 (HIGH)**: engine_name は display 用
    # string ("ReazonSpeech K2 (CPU, Int8)" 等) で渡されるため、dict key
    # (engine ID "reazonspeech") と一致しない。``_engine_id_from_name()``
    # で normalize してから lookup する。
    if (
        ec.no_speech_prob is None
        and ec.token_confidence_mean is None
        and ec.avg_logprob is not None
    ):
        threshold: Optional[float] = None
        engine_id = _engine_id_from_name(engine_name)
        if engine_id is not None:
            threshold = config.avg_logprob_thresholds.get(engine_id)
        if threshold is None:
            threshold = config.avg_logprob_threshold
        if threshold is not None and ec.avg_logprob < threshold:
            # debug 用に display name + 抽出 ID 両方を log (Point 1 検証可能に)
            if engine_name and engine_id and engine_id != engine_name.lower():
                engine_tag = f" (engine={engine_name}, id={engine_id})"
            elif engine_name:
                engine_tag = f" (engine={engine_name})"
            else:
                engine_tag = ""
            return True, (
                f"avg_logprob {ec.avg_logprob:.3f} < {threshold}{engine_tag}"
            )

    return False, None


def _decision_to_dict(decision: "FilterDecision") -> Dict[str, Any]:
    """``FilterDecision`` を PR-A.3 parser が読みやすい dict に変換 (Item 4 対応)。

    ``engine_confidence`` を inline 展開し、schema を:

    .. code-block:: json

        {
            "source_id": "default",
            "engine": "whispers2t",
            "text": "...",
            "decision": "reject",
            "reason": "no_speech_prob 0.800 > 0.5",
            "engine_confidence": {
                "no_speech_prob": 0.8,
                "avg_logprob": null,
                "compression_ratio": null,
                "token_confidence_mean": null,
                "is_available": true
            }
        }

    に固定する。`pass` 判定では ``reason`` が ``null`` になる。calibration parser
    は jsonl で受けて pandas DataFrame 等に load 可能。
    """
    ec = decision.engine_confidence
    return {
        "source_id": decision.source_id,
        "engine": decision.engine,
        "text": decision.text,
        "decision": decision.decision,
        "reason": decision.reason,
        "engine_confidence": {
            "no_speech_prob": ec.no_speech_prob,
            "avg_logprob": ec.avg_logprob,
            "compression_ratio": ec.compression_ratio,
            "token_confidence_mean": ec.token_confidence_mean,
            "is_available": ec.is_available,
        },
    }


def apply_filter(
    result,
    config: FilterConfig,
    *,
    source_id: str,
    engine_name: str,
) -> Optional[TranscriptionResult]:
    """Filter を適用し、reject 時は ``None`` を返す。

    Modes:

    - ``"off"``: filter を skip。log 出力もしない (旧 PR-A.0 挙動)。
    - ``"observe"``: 判定 + **pass / reject 両方** JSON log。result は素通り
      (PR-A.3 calibration 用の data 収集、codex-review #310 Item 4 対応)。
    - ``"on"``: 判定 + **reject 時のみ** JSON log + ``None`` 返却 (silent drop)。
      pass は production で spam 防止のため log しない。

    Log format: ``"confidence_filter[<mode>]: <JSON>"``。JSON schema は
    ``_decision_to_dict()`` を参照。

    fail-open:

    - ``engine_confidence.is_available is False`` の結果は pass-through。
      qwen3asr auto-detect mode (``--language=auto``、wrapper fallback path)
      や score 抽出失敗時などが該当する。Voxtral は PR-A.4.1 ([#311]) から
      avg_logprob を populate するため filter 対象 (strict-gated、
      ``should_reject`` docstring 参照)。Canary は PR-A.4.2 ([#311]) から
      token_confidence_mean を populate するため filter 対象 (Parakeet_ja
      と同 path 共用)。ReazonSpeech は PR-A.5.1 ([#317]) から、qwen3asr
      (language 明示時) は PR-A.5.2 ([#318]) から avg_logprob を populate
      するため filter 対象 (それぞれ engine-specific threshold)。
    Args:
        result: ASR engine の戻り値 (必ず ``TranscriptionResult``)。
            Issue #321 PR #3 で旧 tuple/dict adapter fallback を削除済、
            契約違反は ``AttributeError`` を raise して fail-fast
            (``TranscriptionEngine`` Protocol docstring 参照)。
        config: filter mode + thresholds。
        source_id: log 用の source 識別子。
        engine_name: log 用の engine 名 (``engine.get_engine_name()`` の値)。

    Returns:
        ``"on"`` モードで reject 時のみ ``None``。それ以外は result そのまま。
    """
    if config.mode == "off":
        return result

    # PR-A.5.1 (Issue #317): engine_name を pass-through で
    # engine-specific threshold lookup を有効化
    rejected, reason = should_reject(result, config, engine_name=engine_name)

    decision = FilterDecision(
        source_id=source_id,
        engine=engine_name,
        text=result.text,
        decision="reject" if rejected else "pass",
        reason=reason,
        engine_confidence=result.engine_confidence,
    )

    # Log 出力規約 (codex-review #310 Item 4):
    # - observe: pass/reject 両方を log (calibration 用 data 収集)
    # - on: reject のみ log (production spam 防止)
    should_log = (config.mode == "observe") or rejected
    if should_log:
        log_payload = json.dumps(_decision_to_dict(decision), ensure_ascii=False)
        logger.info("confidence_filter[%s]: %s", config.mode, log_payload)

    if config.mode == "on" and rejected:
        return None

    return result
