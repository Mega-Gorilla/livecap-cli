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
    """ReazonSpeech / qwen3asr / voxtral / canary は ``is_available=False`` → 常に pass。"""

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
    - `config.avg_logprob_threshold is None` (default) では完全 off
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
