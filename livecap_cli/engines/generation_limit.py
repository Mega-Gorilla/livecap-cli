"""生成型の ASR (Qwen3-ASR・Voxtral) の、生成の上限と、選ばれた token の log-prob (#503、#511)。

生成の上限 (``max_new_tokens``) が固定だと、長い区間の書き起こしが黙って途中で切れる。上限は音声の長さから
決め (:func:`max_new_tokens_for`)、上限で止まったか (:func:`stopped_at_token_limit`) を見て WARNING を出す。

上限を長くすると、``generate(output_scores=True)`` が残す全語彙のスコアが生成の長さに比例して増える。
confidence の log-prob は、選ばれた token の値だけを 1 ステップずつためて取る (:class:`ChosenTokenLogprobs`)。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["ChosenTokenLogprobs", "max_new_tokens_for", "stop_token_ids", "stopped_at_token_limit"]


def max_new_tokens_for(duration_s: float, *, floor: int, per_second: float, max_seconds: float) -> int:
    """音声の長さに見合う生成の上限: ``max(floor, ceil(秒 × per_second))``。

    ``floor`` (以前の固定値) を下限にするので、短い音声の生成は以前と同じ。秒は 0 以上 ``max_seconds``
    (その engine が 1 回に扱える音声の最長) 以下に収める。
    """
    seconds = min(max(float(duration_s), 0.0), float(max_seconds))
    return max(floor, math.ceil(seconds * per_second))


def stop_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """生成を止める token (EOS)。model の ``generation_config.eos_token_id`` (1 つまたは複数) と、tokenizer の
    ``eos_token_id``。

    PAD・BOS は入れない。1 本だけの生成では、それらを出しても生成は止まらない (Voxtral は EOS 2・BOS 1・
    PAD 11)。confidence の平均から除く token (特殊 token 全部) とは別に使う (PR #516 F2)。
    """
    ids: set[int] = set()
    generation_config = getattr(model, "generation_config", None)
    for value in (getattr(generation_config, "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        values = value if isinstance(value, (list, tuple, set)) else [value]
        ids.update(v for v in values if isinstance(v, int) and not isinstance(v, bool))
    return ids


def stopped_at_token_limit(gen_tokens: Any, max_new_tokens: int, stop_ids: set) -> bool:
    """生成が停止 token ではなく上限で止まったか (= 書き起こしが途中で切れたか)。

    ``gen_tokens`` は 1 本ぶんの生成した token ID (tensor-like / list)。``stop_ids`` は生成を止める token
    (:func:`stop_token_ids`)。上限ちょうどで停止 token を出したときは、切れていない。
    """
    tokens = gen_tokens.tolist() if hasattr(gen_tokens, "tolist") else list(gen_tokens)
    return len(tokens) >= max_new_tokens and int(tokens[-1]) not in stop_ids


class ChosenTokenLogprobs:
    """生成した token の log-prob だけを、1 ステップずつためる logits processor (#503、PR #509 F1)。

    ``generate(output_scores=True)`` は、全ステップの全語彙のスコア (1 ステップ 語彙数 × 4 バイト。
    Qwen3-ASR は 151,936、Voxtral は 131,072) を最後まで GPU に残し、
    ``compute_transition_scores(normalize_logits=True)`` はさらにそれを積んで log-softmax を取る。
    生成の上限を音声の長さに合わせて長くすると、長い生成でメモリを使い切りうる。
    ここでは各ステップの log-softmax を 1 本だけ持ち、次のステップで選ばれた token
    (``input_ids[:, -1]``) の値だけを取り出す。最後の token の値は :meth:`transition_scores` で取る。

    **greedy (``do_sample=False``) でだけ使う。** transformers は渡された processor を、``generate()`` が作る
    processor (``repetition_penalty`` / ``no_repeat_ngram_size``) の**後ろ**に並べ、greedy ではその後ろに
    足すものが無いので、ここで見るスコアは ``output_scores`` が残すスコアと同じ (処理後・float32) で、値も
    ``compute_transition_scores(normalize_logits=True)`` と同じになる。sampling では、temperature / top-k /
    top-p の warper がこの processor の**後ろ**に足される (transformers 5.22 の ``_get_logits_processor``) ので、
    選ぶのに使った分布とは違う値になる (PR #516 F1)。
    """

    def __init__(self) -> None:
        self._last: Any = None  # 直前のステップの log-softmax (batch, vocab)
        self._picked: list = []  # 確定したステップの、選ばれた token の log-prob (batch, 1) ずつ

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if self._last is not None:
            # 直前のステップで選ばれた token は、いま input_ids の末尾にある
            self._picked.append(self._last.gather(-1, input_ids[:, -1:]))
        self._last = torch.log_softmax(scores, dim=-1)
        return scores

    def transition_scores(self, gen_token_ids: Any) -> Any:
        """``compute_transition_scores(normalize_logits=True)`` と同じ形 ``(batch, 生成数)`` で返す。

        生成が 1 ステップも無かったときは ``None``。
        """
        import torch

        if self._last is None:
            return None
        picked = self._picked + [self._last.gather(-1, gen_token_ids[:, -1:])]
        return torch.cat(picked, dim=-1)
