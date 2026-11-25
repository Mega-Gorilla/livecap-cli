"""生成型の ASR (Qwen3-ASR・Voxtral) の、生成の上限と log-prob の共通部品 (#503、#511)。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from livecap_cli.engines.generation_limit import max_new_tokens_for, stopped_at_token_limit


class TestMaxNewTokensFor:
    """``max(floor, ceil(秒 × per_second))``。秒は 0 以上 max_seconds 以下に収める。"""

    def test_floor_for_short_audio_and_growth_with_length(self):
        kw = {"floor": 448, "per_second": 16, "max_seconds": 1800}
        assert max_new_tokens_for(0.0, **kw) == 448
        assert max_new_tokens_for(28.0, **kw) == 448  # 28 秒 × 16 = 448 までは下限のまま
        assert max_new_tokens_for(28.01, **kw) == 449
        assert max_new_tokens_for(244.1, **kw) == 3906
        assert max_new_tokens_for(1800.0, **kw) == 28800
        assert max_new_tokens_for(3600.0, **kw) == 28800  # 1 回に扱える最長で止める
        assert max_new_tokens_for(-1.0, **kw) == 448


class TestStoppedAtTokenLimit:
    """停止 token を出す前に上限で止まったときだけ、切れたとみなす。"""

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            pytest.param([5] * 4, True, id="limit-without-stop-token"),
            pytest.param([5, 5, 5, 2], False, id="stop-token-at-the-limit"),
            pytest.param([5, 5, 2], False, id="below-the-limit"),
            pytest.param([5] * 5, True, id="over-the-limit"),
        ],
    )
    def test_cases(self, tokens, expected):
        assert stopped_at_token_limit(tokens, 4, {2}) is expected

    def test_tensor_like(self):
        torch = pytest.importorskip("torch")
        assert stopped_at_token_limit(torch.tensor([5, 5, 5, 5]), 4, {2}) is True


class TestChosenTokenLogprobs:
    """選ばれた token の log-prob を 1 ステップずつためる processor (PR #509 F1)。"""

    @staticmethod
    def _run(recorder, steps: int, vocab: int, seed: int = 0):
        """transformers の greedy の 1 ステップと同じ順で回す: 既定の processor → recorder → argmax。"""
        import torch

        gen = torch.Generator().manual_seed(seed)
        ids = torch.zeros((1, 4), dtype=torch.long)
        kept = []
        for _ in range(steps):
            logits = torch.randn((1, vocab), generator=gen) * 4
            logits[0, ids[0, -1]] -= 2.0  # 既定の processor (repetition_penalty など) の代わり
            scores = recorder(ids, logits)
            kept.append(scores)
            ids = torch.cat([ids, scores.argmax(-1, keepdim=True)], dim=-1)
        return ids, kept

    def test_values_match_compute_transition_scores(self):
        """``output_scores`` + ``compute_transition_scores(normalize_logits=True)`` と同じ値を返す。"""
        import torch
        from transformers import GenerationMixin

        from livecap_cli.engines.generation_limit import ChosenTokenLogprobs

        vocab, steps = 50, 12
        recorder = ChosenTokenLogprobs()
        sequences, kept = self._run(recorder, steps, vocab)

        fake_model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=vocab)))
        expected = GenerationMixin.compute_transition_scores(fake_model, sequences, tuple(kept), normalize_logits=True)
        actual = recorder.transition_scores(sequences[:, 4:])

        assert actual.shape == (1, steps)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_keeps_one_vocab_row_not_every_step(self):
        """全語彙の行は直前のステップの 1 本だけで、ステップごとには選ばれた token の値 1 つだけを持つ。"""
        from livecap_cli.engines.generation_limit import ChosenTokenLogprobs

        recorder = ChosenTokenLogprobs()
        self._run(recorder, steps=30, vocab=1000)

        assert recorder._last.shape == (1, 1000)
        assert len(recorder._picked) == 29
        assert all(value.shape == (1, 1) for value in recorder._picked)

    def test_no_step_gives_none(self):
        from livecap_cli.engines.generation_limit import ChosenTokenLogprobs

        assert ChosenTokenLogprobs().transition_scores(None) is None

    def test_scores_pass_through_unchanged(self):
        """スコアは変えずに返す (選ばれる token を変えない)。"""
        import torch

        from livecap_cli.engines.generation_limit import ChosenTokenLogprobs

        scores = torch.randn((1, 10))
        assert ChosenTokenLogprobs()(torch.zeros((1, 3), dtype=torch.long), scores) is scores
