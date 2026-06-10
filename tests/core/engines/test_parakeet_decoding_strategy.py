"""Parakeet `_configure_decoding_with_confidence` の挙動 pin (PR #309)。

Hybrid TDT-CTC model (parakeet_ja) と pure RNNT model (parakeet 英語) で
decoding strategy 設定が分岐することを実 NeMo なしで verify する。

検証する不変条件:
1. Hybrid model (cur_decoder 属性あり) では CTC decoder switch が優先実行される
2. CTC switch が失敗した場合は legacy (RNNT path) に fallback
3. Non-hybrid model (cur_decoder なし) では CTC switch は試行せず legacy へ
4. すべての fallback で例外を raise しない
"""
import importlib
from unittest.mock import MagicMock

import pytest

parakeet_engine = importlib.import_module(
    "livecap_cli.engines.parakeet_engine"
)
ParakeetEngine = parakeet_engine.ParakeetEngine


@pytest.fixture
def fake_engine_with_model():
    """ParakeetEngine を __new__ で生成し、model 属性を MagicMock で差し込める fixture。"""
    engine = ParakeetEngine.__new__(ParakeetEngine)
    engine.engine_name = "parakeet_ja"
    engine.model_name = "nvidia/parakeet-tdt_ctc-0.6b-ja"
    # PR #309: metadata.default_params で parakeet_ja の default は greedy_batch
    # (CTC decoder + NeMo 推奨 strategy)
    engine.decoding_strategy = "greedy_batch"
    engine.device = "cpu"
    engine.torch_device = "cpu"
    engine._initialized = True
    engine.progress_callback = None
    engine.model_metadata = {}
    return engine


def _hybrid_model_mock() -> MagicMock:
    """Hybrid model (cur_decoder attribute を持つ) の mock。"""
    m = MagicMock()
    m.cur_decoder = 'rnnt'  # spec attribute、属性が存在することが重要
    return m


def _pure_rnnt_model_mock() -> MagicMock:
    """Pure RNNT model (cur_decoder なし) の mock。

    MagicMock の default は全 attribute を持ってしまうため、明示的に
    `spec` を制限して `cur_decoder` を持たせないようにする。
    """
    return MagicMock(spec=['change_decoding_strategy'])


class TestHybridModelCTCSwitch:
    """parakeet_ja (Hybrid TDT-CTC) で CTC switch が優先実行されることを pin。"""

    def test_hybrid_model_attempts_ctc_decoder(self, fake_engine_with_model):
        fake_model = _hybrid_model_mock()
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # 最初の call が CTC switch であること
        assert fake_model.change_decoding_strategy.call_count >= 1
        first_call = fake_model.change_decoding_strategy.call_args_list[0]
        assert first_call.kwargs.get('decoder_type') == 'ctc', (
            "Hybrid model では CTC decoder への切替が最初に試行されること"
        )

        # 渡された cfg が self.decoding_strategy + confidence cfg を含むこと
        # PR #309: hardcoded 'greedy_batch' ではなく metadata 由来の値を使う
        cfg = first_call.args[0]
        assert cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        assert cfg.get('greedy', {}).get('preserve_frame_confidence') is True
        assert cfg.get('confidence_cfg', {}).get('preserve_token_confidence') is True

    def test_hybrid_ctc_failure_falls_back_to_legacy(self, fake_engine_with_model):
        """CTC switch が TypeError 等で失敗したら strategy-only fallback に進む。"""
        fake_model = _hybrid_model_mock()

        call_count = {'n': 0}

        def side_effect(*args, **kwargs):
            call_count['n'] += 1
            if call_count['n'] == 1:
                # 第 1 call (CTC switch) は失敗
                assert kwargs.get('decoder_type') == 'ctc'
                raise TypeError("CTC switch not supported")
            # 第 2 call (strategy-only fallback) は成功 (None 返却)
            return None

        fake_model.change_decoding_strategy.side_effect = side_effect
        fake_engine_with_model.model = fake_model

        # 例外を raise しない
        fake_engine_with_model._configure_decoding_with_confidence()

        # 2 つの call が走った: CTC switch (失敗) + strategy-only (成功)
        assert call_count['n'] == 2

        # 第 2 call は decoder_type なし、minimal cfg (confidence_cfg なし)
        second_call = fake_model.change_decoding_strategy.call_args_list[1]
        assert 'decoder_type' not in second_call.kwargs
        legacy_cfg = second_call.args[0]
        assert legacy_cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        # confidence_cfg は意図的に含まない (NeMo 拒否を避けるため、PR #309)
        assert 'confidence_cfg' not in legacy_cfg


class TestNonHybridModel:
    """Pure RNNT (parakeet 英語) では CTC switch を試行しないことを pin。"""

    def test_non_hybrid_skips_ctc_switch(self, fake_engine_with_model):
        fake_model = _pure_rnnt_model_mock()
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # decoder_type='ctc' の呼び出しは存在しないこと
        for call in fake_model.change_decoding_strategy.call_args_list:
            assert call.kwargs.get('decoder_type') != 'ctc', (
                "Pure RNNT model では CTC switch を試行しない (hasattr ガード)"
            )

    def test_non_hybrid_uses_strategy_only_path(self, fake_engine_with_model):
        fake_model = _pure_rnnt_model_mock()
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # minimal strategy cfg のみ (NeMo 拒否を避けるため confidence_cfg は含まない)
        assert fake_model.change_decoding_strategy.call_count >= 1
        first_call = fake_model.change_decoding_strategy.call_args_list[0]
        cfg = first_call.args[0]
        assert cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        assert 'confidence_cfg' not in cfg


class TestExceptionResilience:
    """すべての fallback path で例外を raise しないことを pin。"""

    def test_all_change_decoding_strategy_calls_failing_does_not_raise(
        self, fake_engine_with_model
    ):
        fake_model = _hybrid_model_mock()

        # change_decoding_strategy の全 call を失敗させる
        fake_model.change_decoding_strategy.side_effect = TypeError("always fails")
        fake_engine_with_model.model = fake_model

        # 例外を投げずに完走すること (model 自体は動作させるため)
        fake_engine_with_model._configure_decoding_with_confidence()
