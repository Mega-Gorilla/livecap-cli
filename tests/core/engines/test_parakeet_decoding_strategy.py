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

    def test_hybrid_ctc_failure_falls_back_to_path_1_5(self, fake_engine_with_model):
        """CTC switch が TypeError 等で失敗したら Path 1.5 (RNNT confidence_cfg) に fallback。

        PR-A.4.3 (#316) で Path 1.5 が Path 1 (Hybrid CTC) と Path 2 (strategy-only)
        の間に挿入された。Hybrid model で CTC switch が失敗した場合も、Path 1.5 が
        RNNT/TDT path に preserve_alignments + confidence_cfg を試行する (hybrid
        model の RNNT branch も confidence を populate できる可能性があるため)。
        """
        fake_model = _hybrid_model_mock()

        call_count = {'n': 0}

        def side_effect(*args, **kwargs):
            call_count['n'] += 1
            if call_count['n'] == 1:
                # 第 1 call (Path 1: CTC switch) は失敗
                assert kwargs.get('decoder_type') == 'ctc'
                raise TypeError("CTC switch not supported")
            # 第 2 call (Path 1.5: RNNT confidence_cfg) は成功
            return None

        fake_model.change_decoding_strategy.side_effect = side_effect
        fake_engine_with_model.model = fake_model

        # 例外を raise しない
        fake_engine_with_model._configure_decoding_with_confidence()

        # 2 つの call が走った: Path 1 CTC switch (失敗) + Path 1.5 RNNT confidence_cfg (成功)
        assert call_count['n'] == 2

        # 第 2 call は decoder_type なし、preserve_alignments + confidence_cfg を含む
        second_call = fake_model.change_decoding_strategy.call_args_list[1]
        assert 'decoder_type' not in second_call.kwargs
        path_1_5_cfg = second_call.args[0]
        assert path_1_5_cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        # PR-A.4.3: Path 1.5 では preserve_alignments + confidence_cfg を含む
        assert path_1_5_cfg.get('preserve_alignments') is True
        assert 'confidence_cfg' in path_1_5_cfg
        assert path_1_5_cfg['confidence_cfg'].get('preserve_token_confidence') is True


class TestNonHybridModel:
    """Pure RNNT/TDT (parakeet 英語) では Path 1.5 で confidence_cfg を試行することを pin。

    PR-A.4.3 (#316) で Path 1.5 (TDT 用 preserve_alignments + confidence_cfg) が
    追加され、Pure RNNT model でも token_confidence_mean が populate される
    挙動に変更。実機 probe で LibriSpeech 英語 → 0.2452 (threshold 0.005 の 49x)
    を確認済。
    """

    def test_non_hybrid_skips_ctc_switch(self, fake_engine_with_model):
        fake_model = _pure_rnnt_model_mock()
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # decoder_type='ctc' の呼び出しは存在しないこと
        for call in fake_model.change_decoding_strategy.call_args_list:
            assert call.kwargs.get('decoder_type') != 'ctc', (
                "Pure RNNT model では CTC switch を試行しない (hasattr ガード)"
            )

    def test_non_hybrid_uses_path_1_5_with_confidence_cfg(self, fake_engine_with_model):
        """PR-A.4.3 (#316) で Pure RNNT/TDT は Path 1.5 で confidence_cfg を試行。

        NeMo の制約 (`rnnt_decoding.py:280-282`): preserve_frame_confidence は
        preserve_alignments と同時設定必須。Path 1.5 は両方を含む dedicated
        config で confidence_cfg.preserve_token_confidence を有効化する。
        """
        fake_model = _pure_rnnt_model_mock()
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # Path 1.5 で confidence cfg を含む call が試行されること
        assert fake_model.change_decoding_strategy.call_count >= 1
        first_call = fake_model.change_decoding_strategy.call_args_list[0]
        cfg = first_call.args[0]
        assert cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        # PR-A.4.3: preserve_alignments + confidence_cfg を含むこと
        assert cfg.get('preserve_alignments') is True, (
            "PR-A.4.3 Path 1.5: NeMo の preserve_frame_confidence 制約を満たすため"
            " preserve_alignments=True が必須"
        )
        assert 'confidence_cfg' in cfg, (
            "PR-A.4.3 Path 1.5: pure RNNT/TDT path で token_confidence_mean を"
            " populate するため confidence_cfg を含むこと"
        )
        assert cfg['confidence_cfg'].get('preserve_token_confidence') is True
        # greedy nested cfg も preserve_alignments + preserve_frame_confidence を持つこと
        assert cfg.get('greedy', {}).get('preserve_alignments') is True
        assert cfg.get('greedy', {}).get('preserve_frame_confidence') is True

    def test_non_hybrid_falls_back_to_strategy_only_when_confidence_cfg_rejected(
        self, fake_engine_with_model
    ):
        """Path 1.5 が NeMo に rejected された場合、Path 2 (strategy-only) に fail-open。"""
        call_count = {'n': 0}

        def side_effect(*args, **kwargs):
            call_count['n'] += 1
            # 1 回目 (Path 1.5: confidence_cfg) を拒否、2 回目 (Path 2: strategy-only) は成功
            if call_count['n'] == 1:
                raise TypeError("preserve_alignments not supported in this NeMo version")
            return None

        fake_model = _pure_rnnt_model_mock()
        fake_model.change_decoding_strategy.side_effect = side_effect
        fake_engine_with_model.model = fake_model

        fake_engine_with_model._configure_decoding_with_confidence()

        # Path 1.5 (失敗) + Path 2 (成功) の 2 call
        assert call_count['n'] == 2
        # 2 つ目の call は strategy のみで confidence_cfg を含まない (fail-open)
        second_call = fake_model.change_decoding_strategy.call_args_list[1]
        legacy_cfg = second_call.args[0]
        assert legacy_cfg.get('strategy') == fake_engine_with_model.decoding_strategy
        assert 'confidence_cfg' not in legacy_cfg


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
