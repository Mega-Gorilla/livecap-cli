"""Unit tests for ``livecap_cli.transcription.confidence_filter`` (PR-A.1 / Issue #308).

実 ASR モデルを load せず、``TranscriptionResult`` / ``EngineConfidence`` mock で
filter logic の全 path を pin する。
"""
from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError

import pytest

from livecap_cli.engines.base_engine import EngineConfidence, TranscriptionResult
from livecap_cli.transcription.confidence_filter import (
    FilterConfig,
    FilterDecision,
    apply_filter,
    should_reject,
)


def _build_result(
    *,
    text: str = "テスト",
    confidence: float = 1.0,
    no_speech_prob: float | None = None,
    avg_logprob: float | None = None,
    compression_ratio: float | None = None,
    token_confidence_mean: float | None = None,
) -> TranscriptionResult:
    ec = EngineConfidence(
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        token_confidence_mean=token_confidence_mean,
    )
    return TranscriptionResult(text=text, confidence=confidence, engine_confidence=ec)


class TestFilterConfigDefaults:
    def test_default_mode_is_on(self):
        """v3.1: production default は ``on`` (filter 適用)。"""
        cfg = FilterConfig()
        assert cfg.mode == "on"

    def test_default_thresholds_from_pr_a0_verify(self):
        """PR-A.0 実機 verify 値 + PR-A.4.1 Voxtral smoke verify 値を default に固定。"""
        cfg = FilterConfig()
        assert cfg.no_speech_threshold == 0.5
        assert cfg.token_conf_threshold == 0.005
        # PR-A.4.1 (Issue #311 v2.1) で Voxtral smoke verify (2026-06-11) に基づき
        # default を None → -1.0 に変更。speech 4 clip mean=-0.42 vs non-speech
        # mean=-1.53、margin +1.0、midpoint -1.02 → -1.0 で 100% 分類。
        assert cfg.avg_logprob_threshold == -1.0

    def test_future_thresholds_default_none(self):
        """compression_ratio_threshold は未使用予約 (将来拡張)。"""
        cfg = FilterConfig()
        assert cfg.compression_ratio_threshold is None

    def test_frozen_prevents_mutation(self):
        cfg = FilterConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.mode = "off"  # type: ignore[misc]


class TestShouldRejectWhisperS2T:
    """WhisperS2T (``no_speech_prob``) の判定挙動。"""

    def test_low_no_speech_prob_passes(self):
        """PR-A.0 verify 値 (speech 0.036) は threshold 0.5 を踏まない。"""
        result = _build_result(no_speech_prob=0.036)
        rejected, reason = should_reject(result, FilterConfig())
        assert rejected is False
        assert reason is None

    def test_high_no_speech_prob_rejects(self):
        """PR-A.0 verify 値 (non-speech 0.66) は threshold 0.5 を踏む。"""
        result = _build_result(no_speech_prob=0.66)
        rejected, reason = should_reject(result, FilterConfig())
        assert rejected is True
        assert reason is not None
        assert "no_speech_prob" in reason
        assert "0.5" in reason

    def test_exactly_threshold_does_not_reject(self):
        """境界値: ``> threshold`` であり ``>=`` ではないため、ぴったりは通す。"""
        result = _build_result(no_speech_prob=0.5)
        rejected, _ = should_reject(result, FilterConfig())
        assert rejected is False

    def test_custom_threshold_override(self):
        result = _build_result(no_speech_prob=0.4)
        rejected, _ = should_reject(result, FilterConfig(no_speech_threshold=0.3))
        assert rejected is True


class TestShouldRejectParakeet:
    """Parakeet_ja (``token_confidence_mean``) の判定挙動。"""

    def test_high_token_confidence_passes(self):
        """PR-A.0 verify 値 (speech 0.05) は threshold 0.005 を上回る。"""
        result = _build_result(token_confidence_mean=0.0504)
        rejected, _ = should_reject(result, FilterConfig())
        assert rejected is False

    def test_low_token_confidence_rejects(self):
        """PR-A.0 verify 値 (non-speech 0.0000029) は threshold を下回る。"""
        result = _build_result(token_confidence_mean=0.0000029)
        rejected, reason = should_reject(result, FilterConfig())
        assert rejected is True
        assert reason is not None
        assert "token_confidence_mean" in reason

    def test_exactly_threshold_does_not_reject(self):
        """境界値: ``< threshold`` であり ``<=`` ではない。"""
        result = _build_result(token_confidence_mean=0.005)
        rejected, _ = should_reject(result, FilterConfig())
        assert rejected is False

    def test_short_utterances_corpus_value_passes(self):
        """PR-B corpus ``short_utterances_mixed.wav`` (0.0104) は通す。"""
        result = _build_result(token_confidence_mean=0.0104)
        rejected, _ = should_reject(result, FilterConfig())
        assert rejected is False


class TestShouldRejectFailOpen:
    """ReazonSpeech / qwen3asr は ``is_available=False`` → 常に pass。

    Voxtral は PR-A.4.1 ([#311]) から ``avg_logprob`` を populate するため
    filter 対象 (strict-gated、``TestAvgLogprobStrictGate`` 参照)。
    Canary は PR-A.4.2 ([#311]) から ``token_confidence_mean`` を populate
    するため filter 対象 (Parakeet_ja と同 ``token_conf_threshold`` を共用)。"""

    def test_all_none_engine_confidence_passes(self):
        result = _build_result()  # 全 field None
        rejected, reason = should_reject(result, FilterConfig())
        assert rejected is False
        assert reason is None

    def test_only_avg_logprob_set_with_explicit_none_threshold_passes(self):
        """``avg_logprob_threshold=None`` を明示すれば avg_logprob は無視される。

        PR-A.4.1 で default が None → -1.0 に変更されたが、user が threshold を
        明示的に None に設定する場合 (= avg_logprob path を opt-out) は filter
        判定を行わず pass する。
        """
        result = _build_result(avg_logprob=-5.0)
        config = FilterConfig(avg_logprob_threshold=None)
        rejected, _ = should_reject(result, config)
        assert rejected is False

    def test_only_avg_logprob_set_below_default_threshold_rejects(self):
        """PR-A.4.1 default で avg_logprob のみ populate + low value → reject。

        Voxtral fail-open は engine_confidence 全 None ケース。avg_logprob だけ
        populate されるのは Voxtral path で、default threshold -1.0 を下回れば
        reject される。
        """
        result = _build_result(avg_logprob=-5.0)
        rejected, reason = should_reject(result, FilterConfig())
        assert rejected is True
        assert reason is not None
        assert "avg_logprob" in reason


class TestApplyFilterModes:
    """3 mode (``off`` / ``observe`` / ``on``) の差分挙動。"""

    @pytest.fixture
    def reject_target(self) -> TranscriptionResult:
        # 確実に reject される result (non-speech 高 prob)
        return _build_result(text="ノイズ", no_speech_prob=0.8)

    @pytest.fixture
    def pass_target(self) -> TranscriptionResult:
        # 確実に pass する result (speech 低 prob)
        return _build_result(text="こんにちは", no_speech_prob=0.04)

    def test_off_mode_passes_through_reject_target(self, reject_target):
        """``off`` モード: reject 判定対象でも素通り。"""
        out = apply_filter(
            reject_target,
            FilterConfig(mode="off"),
            source_id="test",
            engine_name="whispers2t",
        )
        assert out is reject_target

    def test_off_mode_emits_no_log(self, reject_target, caplog):
        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                reject_target,
                FilterConfig(mode="off"),
                source_id="test",
                engine_name="whispers2t",
            )
        assert not caplog.records, "off モードでは log を出さない"

    def test_observe_mode_passes_through_reject_target(self, reject_target):
        """``observe`` モード: reject 判定でも result を返す (log のみ)。"""
        out = apply_filter(
            reject_target,
            FilterConfig(mode="observe"),
            source_id="test",
            engine_name="whispers2t",
        )
        assert out is reject_target

    def test_observe_mode_emits_json_log_on_reject(self, reject_target, caplog):
        """codex-review #310 Item 4: observe mode log は安定 JSON 形式 + reject。"""
        import json

        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                reject_target,
                FilterConfig(mode="observe"),
                source_id="my_mic",
                engine_name="whispers2t",
            )
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert msg.startswith("confidence_filter[observe]: ")
        payload_str = msg.split("confidence_filter[observe]: ", 1)[1]
        payload = json.loads(payload_str)
        # Schema 固定: PR-A.3 parser 用
        assert payload["source_id"] == "my_mic"
        assert payload["engine"] == "whispers2t"
        assert payload["decision"] == "reject"
        assert payload["reason"] is not None
        assert "no_speech_prob" in payload["reason"]
        # engine_confidence は inline 展開
        ec = payload["engine_confidence"]
        assert ec["no_speech_prob"] == pytest.approx(0.8)
        assert ec["is_available"] is True

    def test_observe_mode_emits_json_log_on_pass(self, pass_target, caplog):
        """codex-review #310 Item 4: observe mode は pass 側も JSON log 出力。

        PR-A.3 calibration が閾値マージン / speech recall 安全域を解析する
        ためには reject 側だけでなく pass 側の engine_confidence も必要。
        """
        import json

        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                pass_target,
                FilterConfig(mode="observe"),
                source_id="my_mic",
                engine_name="whispers2t",
            )
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        payload_str = msg.split("confidence_filter[observe]: ", 1)[1]
        payload = json.loads(payload_str)
        assert payload["decision"] == "pass"
        assert payload["reason"] is None  # pass では reason なし
        # engine_confidence は inline 展開
        ec = payload["engine_confidence"]
        assert ec["no_speech_prob"] == pytest.approx(0.04)
        assert ec["is_available"] is True

    def test_on_mode_returns_none_on_reject(self, reject_target):
        """``on`` モード: reject 時は ``None`` 返却 (silent drop)。"""
        out = apply_filter(
            reject_target,
            FilterConfig(mode="on"),
            source_id="test",
            engine_name="whispers2t",
        )
        assert out is None

    def test_on_mode_emits_json_log_on_reject(self, reject_target, caplog):
        """on mode は reject のみ JSON log (production spam 防止)。"""
        import json

        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                reject_target,
                FilterConfig(mode="on"),
                source_id="test",
                engine_name="whispers2t",
            )
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert msg.startswith("confidence_filter[on]: ")
        payload = json.loads(msg.split("confidence_filter[on]: ", 1)[1])
        assert payload["decision"] == "reject"

    def test_on_mode_emits_no_log_on_pass(self, pass_target, caplog):
        """on mode の pass 側は log なし (production spam 防止)。"""
        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                pass_target,
                FilterConfig(mode="on"),
                source_id="test",
                engine_name="whispers2t",
            )
        assert not caplog.records

    def test_on_mode_passes_through_pass_target(self, pass_target):
        """``on`` モードでも pass 判定なら result はそのまま。"""
        out = apply_filter(
            pass_target,
            FilterConfig(mode="on"),
            source_id="test",
            engine_name="whispers2t",
        )
        assert out is pass_target


class TestApplyFilterFailOpen:
    """``is_available=False`` の engine は全 mode で pass-through。"""

    def test_on_mode_passes_through_unavailable_engine(self):
        """ReazonSpeech 想定 (engine_confidence 全 None)。"""
        result = _build_result(text="ピッ")
        out = apply_filter(
            result,
            FilterConfig(mode="on"),
            source_id="test",
            engine_name="reazonspeech",
        )
        assert out is result, "fail-open: is_available=False の engine は pass-through"

    def test_on_mode_no_log_for_unavailable_engine(self, caplog):
        """fail-open は log も出さない (engine が常に対象外 = log spam 防止)。"""
        result = _build_result(text="ピッ")
        with caplog.at_level(
            logging.INFO, logger="livecap_cli.transcription.confidence_filter"
        ):
            apply_filter(
                result,
                FilterConfig(mode="on"),
                source_id="test",
                engine_name="reazonspeech",
            )
        assert not caplog.records


class TestFilterDecisionDataclass:
    """``FilterDecision`` の dataclass 挙動を pin。"""

    def test_filter_decision_fields(self):
        ec = EngineConfidence(no_speech_prob=0.8)
        decision = FilterDecision(
            source_id="mic_0",
            engine="whispers2t",
            text="ノイズ",
            decision="reject",
            reason="no_speech_prob 0.800 > 0.5",
            engine_confidence=ec,
        )
        assert decision.source_id == "mic_0"
        assert decision.engine == "whispers2t"
        assert decision.text == "ノイズ"
        assert decision.decision == "reject"
        assert decision.reason is not None
        assert decision.engine_confidence is ec

    def test_filter_decision_is_frozen(self):
        ec = EngineConfidence()
        decision = FilterDecision(
            source_id="x",
            engine="x",
            text="x",
            decision="pass",
            reason=None,
            engine_confidence=ec,
        )
        with pytest.raises(FrozenInstanceError):
            decision.text = "changed"  # type: ignore[misc]


class TestRegressionPrA0Values:
    """PR-A.0 実機 verify 値で filter 挙動が期待通りであることを pin。

    本 test は PR-A.0 の signal 分離度が PR-A.1 threshold で正しく分類できる
    ことを永続化する (将来の threshold 変更で regression を検出可能)。
    """

    @pytest.mark.parametrize(
        "engine_name,no_speech_prob,token_conf,expected_reject",
        [
            # WhisperS2T (no_speech_prob)
            ("whispers2t", 0.036, None, False),  # normal_speech_neko.wav
            ("whispers2t", 0.635, None, True),   # desk_tap.wav
            ("whispers2t", 0.662, None, True),   # applause_5_claps.wav
            # Parakeet_ja (token_confidence_mean)
            ("parakeet_ja", None, 0.1023, False),  # applause_then_speech.wav
            ("parakeet_ja", None, 0.0504, False),  # normal_speech_neko.wav
            ("parakeet_ja", None, 0.0383, False),  # overlapping_applause_speech.wav
            ("parakeet_ja", None, 0.0104, False),  # short_utterances_mixed.wav
            ("parakeet_ja", None, 0.0003, True),   # desk_tap.wav
            ("parakeet_ja", None, 0.0000029, True),  # applause_5_claps.wav
        ],
    )
    def test_pr_b_corpus_classification(
        self, engine_name, no_speech_prob, token_conf, expected_reject
    ):
        result = _build_result(
            no_speech_prob=no_speech_prob,
            token_confidence_mean=token_conf,
        )
        rejected, _ = should_reject(result, FilterConfig())
        assert rejected is expected_reject, (
            f"{engine_name}: no_speech_prob={no_speech_prob} "
            f"token_conf={token_conf} expected_reject={expected_reject} got={rejected}"
        )


class TestAvgLogprobStrictGate:
    """PR-A.4.1 (Issue #311 v2.1): avg_logprob 判定の **strict gating** を pin。

    判定規約:
    - WhisperS2T (no_speech_prob populate) は avg_logprob 経路に到達しない
    - Parakeet_ja (token_confidence_mean populate) も到達しない
    - Voxtral-like (両方 None + avg_logprob のみ) で初めて評価
    - PR-A.4.1 で default は ``-1.0`` (smoke verify margin +1.002 由来)。
      ``config.avg_logprob_threshold=None`` を明示すれば完全 opt-out
    """

    def test_whispers2t_pass_when_avg_logprob_low_but_no_speech_ok(self):
        """WhisperS2T 退行ゼロ pin: no_speech_prob pass、avg_logprob 低くても reject しない。"""
        result = _build_result(
            no_speech_prob=0.1,    # speech 範囲
            avg_logprob=-5.0,      # 低いが gated で見ない
        )
        config = FilterConfig(avg_logprob_threshold=-1.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False
        assert reason is None

    def test_parakeet_pass_when_avg_logprob_low_but_token_conf_ok(self):
        """Parakeet 退行ゼロ pin: token_confidence_mean pass、avg_logprob 低くても reject しない。"""
        result = _build_result(
            token_confidence_mean=0.05,  # speech 範囲
            avg_logprob=-5.0,            # 低いが gated で見ない
        )
        config = FilterConfig(avg_logprob_threshold=-1.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False
        assert reason is None

    def test_voxtral_like_reject_when_avg_logprob_below_threshold(self):
        """Voxtral active 化 pin: 他 signal 不在 + avg_logprob 低 + threshold 設定。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-3.5,
        )
        config = FilterConfig(avg_logprob_threshold=-2.0)
        rejected, reason = should_reject(result, config)
        assert rejected is True
        assert reason is not None
        assert "avg_logprob" in reason
        assert "-3.500" in reason
        assert "-2.0" in reason

    def test_voxtral_like_pass_when_avg_logprob_above_threshold(self):
        """Voxtral pass pin: avg_logprob が threshold より上なら pass。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig(avg_logprob_threshold=-2.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False
        assert reason is None

    def test_voxtral_like_pass_when_threshold_explicitly_none(self):
        """``avg_logprob_threshold=None`` 明示で active 化されない pin。

        PR-A.4.1 で default は -1.0 に変更されたが、user が CLI/API で
        ``avg_logprob_threshold=None`` を明示すれば avg_logprob 判定経路は完全 off。
        """
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-10.0,  # 極端に低くても threshold None なら pass
        )
        config = FilterConfig(avg_logprob_threshold=None)
        assert config.avg_logprob_threshold is None
        rejected, reason = should_reject(result, config)
        assert rejected is False
        assert reason is None

    def test_voxtral_like_pass_when_avg_logprob_none(self):
        """全 None (= EngineConfidence().is_available is False) → fail-open。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=None,
        )
        config = FilterConfig(avg_logprob_threshold=-2.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False  # is_available=False で fail-open
        assert reason is None

    def test_gate_input_no_speech_prob_blocks_avg_logprob(self):
        """no_speech_prob populate (pass 値) + avg_logprob 低: avg_logprob 経路スキップ。"""
        result = _build_result(
            no_speech_prob=0.3,   # < 0.5 で no_speech reject にならない
            avg_logprob=-5.0,     # 低いが gated
        )
        config = FilterConfig(avg_logprob_threshold=-1.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False

    def test_gate_input_token_confidence_blocks_avg_logprob(self):
        """token_confidence_mean populate (pass 値) + avg_logprob 低: avg_logprob 経路スキップ。"""
        result = _build_result(
            token_confidence_mean=0.5,  # > 0.005 で reject にならない
            avg_logprob=-5.0,           # 低いが gated
        )
        config = FilterConfig(avg_logprob_threshold=-1.0)
        rejected, reason = should_reject(result, config)
        assert rejected is False


class TestEngineSpecificAvgLogprobThreshold:
    """PR-A.5.1 (Issue #317): engine-specific avg_logprob threshold dict を pin。

    Voxtral と ReazonSpeech は同 ``avg_logprob`` field を共用するが、分布が
    桁違い (Voxtral speech -0.42、non-speech -1.53、threshold -1.0 / ReazonSpeech
    speech -0.11、non-speech -0.45、threshold -0.2)。global threshold -1.0
    は ReazonSpeech に機能しないため、``avg_logprob_thresholds`` dict で
    engine-specific calibration を実現する。

    判定規約:
    - ``engine_name=`` を ``should_reject(...)`` に pass
    - dict に entry あり (e.g. "reazonspeech") → engine-specific threshold 適用
    - dict に entry なし (e.g. "voxtral") → ``avg_logprob_threshold`` (global) fallback
    - ``engine_name=None`` → global fallback
    """

    def test_reazonspeech_active_with_engine_specific_threshold(self):
        """``engine_name='reazonspeech'`` で dict default ``-0.2`` が適用、speech mean 範囲は pass。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.1,  # speech 範囲 (> -0.2)
        )
        config = FilterConfig()  # default: reazonspeech=-0.2、avg_logprob_threshold=-1.0
        rejected, reason = should_reject(result, config, engine_name="reazonspeech")
        assert rejected is False
        assert reason is None

    def test_reazonspeech_reject_when_below_engine_specific_threshold(self):
        """``engine_name='reazonspeech'`` で non-speech 範囲 (-0.5) → reject。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig()  # reazonspeech=-0.2
        rejected, reason = should_reject(result, config, engine_name="reazonspeech")
        assert rejected is True
        assert reason is not None
        assert "avg_logprob" in reason
        assert "-0.500" in reason
        assert "-0.2" in reason
        assert "engine=reazonspeech" in reason

    def test_voxtral_uses_global_fallback_when_not_in_dict(self):
        """``engine_name='voxtral'`` は dict にない → global ``avg_logprob_threshold = -1.0`` fallback。

        Voxtral 退行ゼロ pin: PR-A.4.1 と同じ -1.0 threshold が適用される。
        """
        # speech (-0.5) は -1.0 threshold で pass
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig()  # avg_logprob_threshold=-1.0、voxtral は dict にない
        rejected, reason = should_reject(result, config, engine_name="voxtral")
        assert rejected is False

        # non-speech (-1.5) は -1.0 threshold で reject
        result_reject = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-1.5,
        )
        rejected, reason = should_reject(result_reject, config, engine_name="voxtral")
        assert rejected is True
        assert "-1.0" in reason
        assert "engine=voxtral" in reason

    def test_no_engine_name_uses_global_fallback(self):
        """``engine_name=None`` で global fallback (旧 PR-A.4.1 挙動と整合)。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-1.5,  # reject
        )
        config = FilterConfig()  # avg_logprob_threshold=-1.0
        rejected, reason = should_reject(result, config, engine_name=None)
        assert rejected is True
        # engine 名 tag なし (engine_name=None)
        assert "engine=" not in reason

    def test_explicit_engine_threshold_override_via_constructor(self):
        """user が ``FilterConfig(avg_logprob_thresholds={...})`` で override 可能。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.3,
        )
        # ReazonSpeech default -0.2 を -0.5 に緩める override
        config = FilterConfig(avg_logprob_thresholds={"reazonspeech": -0.5})
        rejected, reason = should_reject(result, config, engine_name="reazonspeech")
        assert rejected is False  # -0.3 > -0.5 で pass

    def test_engine_in_dict_with_none_global_still_applies(self):
        """``avg_logprob_threshold=None`` + dict entry あり → engine-specific は active。

        global opt-out しても engine-specific threshold は独立に機能する。
        """
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig(
            avg_logprob_threshold=None,  # global opt-out
            avg_logprob_thresholds={"reazonspeech": -0.2},  # ReazonSpeech は active
        )
        rejected, reason = should_reject(result, config, engine_name="reazonspeech")
        assert rejected is True  # ReazonSpeech specific threshold が active

    def test_engine_not_in_dict_with_none_global_pass_through(self):
        """``avg_logprob_threshold=None`` + dict にない engine → 完全 pass。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-10.0,  # 極端に低い
        )
        config = FilterConfig(
            avg_logprob_threshold=None,
            avg_logprob_thresholds={"reazonspeech": -0.2},
        )
        # voxtral は dict にない + global None → 完全 pass
        rejected, reason = should_reject(result, config, engine_name="voxtral")
        assert rejected is False


class TestEngineIdNormalization:
    """PR-A.5.1 codex-review Point 1 (HIGH、blocking) — production display
    string で engine ID lookup が機能すること を pin。

    背景:
    - ``StreamTranscriber`` は ``engine.get_engine_name()`` を
      ``apply_filter(engine_name=...)`` に渡す。
    - ReazonSpeech の ``get_engine_name()`` は ``"ReazonSpeech K2 (CPU,
      Int8)"`` / ``"ReazonSpeech K2 (CPU, Float32)"`` を返す
      (``reazonspeech_engine.py:549``)。
    - 旧実装は ``config.avg_logprob_thresholds.get(engine_name)`` で直接
      lookup していたため、上記 display string で hit せず global fallback
      ``-1.0`` が適用 → PR の主目的が production で完全に効かない bug。
    - 本 PR で ``_engine_id_from_name()`` helper を追加、display string
      → first whitespace-separated word の lowercase ID 変換を導入。

    本 test class は **実 ``get_engine_name()`` 相当 string** で
    threshold lookup が正しく機能することを pin する。
    """

    def test_reazonspeech_int8_display_string_matches_dict_key(self):
        """``"ReazonSpeech K2 (CPU, Int8)"`` で reazonspeech threshold (-0.2) 適用。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,  # -0.2 より低い → reject される
        )
        config = FilterConfig()  # default reazonspeech=-0.2
        rejected, reason = should_reject(
            result, config, engine_name="ReazonSpeech K2 (CPU, Int8)"
        )
        assert rejected is True
        assert reason is not None
        assert "-0.2" in reason
        # debug 用: engine_name と id 両方表示
        assert "ReazonSpeech K2 (CPU, Int8)" in reason
        assert "id=reazonspeech" in reason

    def test_reazonspeech_float32_display_string_matches_dict_key(self):
        """``"ReazonSpeech K2 (CPU, Float32)"`` で reazonspeech threshold (-0.2) 適用。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig()
        rejected, reason = should_reject(
            result, config, engine_name="ReazonSpeech K2 (CPU, Float32)"
        )
        assert rejected is True
        assert "-0.2" in reason
        assert "id=reazonspeech" in reason

    def test_reazonspeech_display_string_speech_avg_passes(self):
        """``"ReazonSpeech K2 (CPU, Int8)"`` + speech 範囲 avg_logprob → pass。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.1,  # speech 範囲、-0.2 より上 → pass
        )
        config = FilterConfig()
        rejected, reason = should_reject(
            result, config, engine_name="ReazonSpeech K2 (CPU, Int8)"
        )
        assert rejected is False

    def test_whispers2t_display_string_uses_global_fallback(self):
        """``"WhisperS2T base"`` (dict にない id "whispers2t") → global -1.0 fallback。"""
        result_pass = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,  # -1.0 より上、global pass
        )
        config = FilterConfig()
        rejected, _ = should_reject(
            result_pass, config, engine_name="WhisperS2T base"
        )
        assert rejected is False

        result_reject = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-1.5,  # -1.0 より下、global reject
        )
        rejected, reason = should_reject(
            result_reject, config, engine_name="WhisperS2T base"
        )
        assert rejected is True
        assert "-1.0" in reason

    def test_lowercase_id_engine_name_also_works(self):
        """Backward compat: 既に ID-form ("reazonspeech") で渡された場合も正常動作。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-0.5,
        )
        config = FilterConfig()
        rejected, reason = should_reject(
            result, config, engine_name="reazonspeech"
        )
        assert rejected is True
        assert "-0.2" in reason

    def test_empty_engine_name_uses_global_fallback(self):
        """空文字 / 空白のみ engine_name → global fallback (id 抽出不能)。"""
        result = _build_result(
            no_speech_prob=None,
            token_confidence_mean=None,
            avg_logprob=-1.5,
        )
        config = FilterConfig()
        for empty in ("", "   "):
            rejected, reason = should_reject(result, config, engine_name=empty)
            # global fallback -1.0 で -1.5 → reject
            assert rejected is True

    def test_helper_engine_id_from_name_exact_mappings(self):
        """``_engine_id_from_name()`` の input/output mapping を pin。"""
        from livecap_cli.transcription.confidence_filter import _engine_id_from_name

        # Production engine display strings (from each engine's get_engine_name())
        assert _engine_id_from_name("ReazonSpeech K2 (CPU, Int8)") == "reazonspeech"
        assert _engine_id_from_name("ReazonSpeech K2 (CPU, Float32)") == "reazonspeech"
        assert _engine_id_from_name("WhisperS2T base") == "whispers2t"
        assert _engine_id_from_name("voxtral") == "voxtral"
        assert _engine_id_from_name("canary") == "canary"
        assert _engine_id_from_name("MockEngine") == "mockengine"
        # ID-form (backward compat)
        assert _engine_id_from_name("reazonspeech") == "reazonspeech"
        # Edge cases
        assert _engine_id_from_name(None) is None
        assert _engine_id_from_name("") is None
        assert _engine_id_from_name("   ") is None
