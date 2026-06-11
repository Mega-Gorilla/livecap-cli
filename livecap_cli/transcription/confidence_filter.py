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
- ``engine_confidence.is_available is False`` の engine (ReazonSpeech /
  qwen3asr / Canary) は無条件 pass-through (fail-open)。fail-open は意図的な
  設計で、未対応 engine が silently 全 reject されることを防ぐ。Voxtral は
  PR-A.4.1 ([#311]) から ``avg_logprob`` を populate するため filter 対象
  (下記 strict gate)。
- threshold は実機 verify 済の値を default (WhisperS2T: 0.5、Parakeet_ja:
  0.005、Voxtral: -1.0 [PR-A.4.1 smoke verify 2026-06-11])。``FilterConfig``
  の constructor kwargs で override 可能 (sweep harness が PR-A.3 で
  programmatic に threshold を sweep するため、``avg_logprob_threshold=None``
  で Voxtral 経路を完全 opt-out 可能)。

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
        avg_logprob_threshold: Voxtral の ``avg_logprob`` 用 strict-gated
            threshold (PR-A.4.1 Issue #311)。``no_speech_prob`` /
            ``token_confidence_mean`` の両方が None の時のみ評価される
            (WhisperS2T 退行回避)。PR-A.4.1 実機 smoke verify (2026-06-11)
            で speech mean -0.42 vs non-speech -1.53 → margin +1.0 → midpoint
            -1.02 → ``-1.0`` で 100% 分類可能を確認。
        compression_ratio_threshold: 未使用予約 field (将来拡張)。
    """

    mode: FilterMode = "on"
    no_speech_threshold: float = 0.5
    token_conf_threshold: float = 0.005
    avg_logprob_threshold: Optional[float] = -1.0
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


def should_reject(
    result: TranscriptionResult,
    config: FilterConfig,
) -> Tuple[bool, Optional[str]]:
    """Filter 判定。``(rejected, reason)`` を返す。

    判定順位:

    1. ``engine_confidence.is_available is False`` → fail-open (pass、reason=None)
    2. ``no_speech_prob > config.no_speech_threshold`` → reject (WhisperS2T 主)
    3. ``token_confidence_mean < config.token_conf_threshold`` → reject
       (Parakeet_ja / Canary 主)
    4. **strict-gated** ``avg_logprob < config.avg_logprob_threshold`` → reject
       (Voxtral 主、PR-A.4.1 で追加)
    5. それ以外 → pass

    各 signal は None を許容 (engine ごとに populate される field が異なるため)。

    PR-A.4.1 (Issue #311 v2.1) の strict gating ルール:

    - WhisperS2T は ``no_speech_prob`` **と** ``avg_logprob`` を両方 populate
      する (``whispers2t_engine.py:18-77``)。avg_logprob の global 判定を
      入れると、no_speech_prob が pass でも avg_logprob で false reject される
      退行が起きる。
    - そのため avg_logprob 判定は ``no_speech_prob is None`` AND
      ``token_confidence_mean is None`` の時のみ評価する (strict gate)。
    - Voxtral のように avg_logprob だけ populate する engine では active に
      なり、WhisperS2T / Parakeet_ja は早期 return で avg_logprob 経路に
      到達しない (退行ゼロ)。
    - ``config.avg_logprob_threshold = -1.0`` が default (PR-A.4.1 smoke
      verify 2026-06-11 で Voxtral speech worst -0.523 vs non-speech
      -1.525 の midpoint -1.02 を rounded した値、Whisper 慣習値とも一致)。
      user が明示的に ``avg_logprob_threshold=None`` を渡せば avg_logprob
      判定経路を完全 opt-out 可能 (Voxtral の debug 時等)。
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

    # PR-A.4.1: avg_logprob は他 signal 不在時のみ判定 (Voxtral 主)。
    # WhisperS2T も avg_logprob を populate するが、no_speech_prob path で
    # 既に処理済のためここでは見ない (false positive 回避)。
    if (
        ec.no_speech_prob is None
        and ec.token_confidence_mean is None
        and ec.avg_logprob is not None
        and config.avg_logprob_threshold is not None
        and ec.avg_logprob < config.avg_logprob_threshold
    ):
        return True, (
            f"avg_logprob {ec.avg_logprob:.3f} < {config.avg_logprob_threshold}"
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

    - ``engine_confidence.is_available is False`` の engine
      (ReazonSpeech / qwen3asr / Canary) は無条件 pass-through。Voxtral は
      PR-A.4.1 ([#311]) から avg_logprob を populate するため filter 対象
      (strict-gated、``should_reject`` docstring 参照)。
    - ``result`` が ``engine_confidence`` 属性を持たない (例: 一部の test
      mock) は pass-through。``shared_engine_manager.py`` の defensive
      パターンと整合する safety net。

    Args:
        result: ASR engine の戻り値 (``TranscriptionResult``、または
            ``engine_confidence`` を持たない test mock)。
        config: filter mode + thresholds。
        source_id: log 用の source 識別子。
        engine_name: log 用の engine 名 (``engine.get_engine_name()`` の値)。

    Returns:
        ``"on"`` モードで reject 時のみ ``None``。それ以外は result そのまま。
    """
    if config.mode == "off":
        return result

    # Legacy fallback: TranscriptionResult 以外 (tuple 等) は filter 対象外。
    # PR-A.0 で adapter は全て TranscriptionResult を返すよう更新済だが、test
    # の MockEngine 等が tuple を返すケースに対応。
    if not hasattr(result, "engine_confidence"):
        return result

    rejected, reason = should_reject(result, config)

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
