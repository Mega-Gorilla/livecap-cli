"""Voxtral の生成の上限と、上限で止まったときの WARNING (#511)。

モデルは読み込まない。``model.generate`` を、transformers と同じく渡された logits processor を
1 ステップごとに呼ぶ fake に差し替え、engine の書き起こしの経路 (``_transcribe_single_chunk``) を通す。
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from livecap_cli.engines.metadata import EngineMetadata  # noqa: E402
from livecap_cli.engines.voxtral_engine import VoxtralEngine  # noqa: E402

_LOGGER = "livecap_cli.engines.voxtral_engine"
_EOS, _PAD, _BOS = 2, 11, 1
_VOCAB = 64
_PROMPT_LEN = 5


class _FakeInputs(dict):
    """``apply_transcription_request(...)`` の戻り値の代わり (``.to()`` と ``input_ids.shape`` だけ使う)。"""

    def __init__(self) -> None:
        super().__init__()
        self.input_ids = torch.zeros((1, _PROMPT_LEN), dtype=torch.long)

    def to(self, _device):
        return self


def _step_scores(token: int, logprob: float):
    """選ばれる ``token`` の log-softmax がちょうど ``logprob`` になる、1 ステップぶんのスコア。"""
    scores = torch.full((1, _VOCAB), float("-inf"))
    scores[0, token] = 0.0
    p = math.exp(logprob)
    scores[0, 0 if token != 0 else 3] = math.log((1 - p) / p)
    return scores


def _engine(generated: list[int], *, max_new_tokens=None, logprob: float = -0.2):
    """model / processor を fake にした、ロード済みの engine。``generated`` を順に生成したことにする。"""
    engine = VoxtralEngine.__new__(VoxtralEngine)
    engine._initialized = True
    engine.torch_device = "cpu"
    engine.language = "en"
    engine._asr_language = "en"
    engine.model_name = "test-model"
    engine.do_sample = False
    engine.temperature = 0.0
    engine.max_new_tokens = max_new_tokens

    def generate(**kwargs):
        ids = kwargs["input_ids"] if "input_ids" in kwargs else torch.zeros((1, _PROMPT_LEN), dtype=torch.long)
        for token in generated[: kwargs["max_new_tokens"]]:
            scores = _step_scores(token, logprob)
            for proc in kwargs.get("logits_processor") or []:
                scores = proc(ids, scores)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=-1)
        return SimpleNamespace(sequences=ids)

    engine.model = MagicMock()
    engine.model.generate.side_effect = generate
    engine.processor = MagicMock()
    engine.processor.apply_transcription_request.return_value = _FakeInputs()
    engine.processor.batch_decode.return_value = ["hello world"]
    engine.processor.tokenizer = SimpleNamespace(eos_token_id=_EOS, pad_token_id=_PAD, bos_token_id=_BOS)
    return engine


def _transcribe(engine: VoxtralEngine, seconds: float, tmp_path, sample_rate: int = 16000):
    audio = np.broadcast_to(np.float32(0.01), (int(seconds * sample_rate),))
    with (
        patch("livecap_cli.engines.voxtral_engine.sf.write"),
        patch("livecap_cli.engines.voxtral_engine.get_temp_dir", return_value=tmp_path),
    ):
        return engine.transcribe(np.array(audio), sample_rate)


class TestVoxtralGenerationLimit:
    def test_limit_keeps_448_for_short_audio_and_grows_with_length(self, tmp_path):
        """渡していなければ音声の長さから決める: 28 秒までは以前の 448、それより長いと 1 秒あたり 16 token。"""
        for seconds, expected in ((0.5, 448), (28.0, 448), (60.0, 960), (244.0, 3904)):
            engine = _engine([7, _EOS])
            _transcribe(engine, seconds, tmp_path)
            assert engine.model.generate.call_args.kwargs["max_new_tokens"] == expected, seconds

    def test_explicit_limit_is_kept(self, tmp_path):
        """数を渡したときは、音声の長さに関わらずその値 (以前の既定の 448 に戻したい利用者向け)。"""
        engine = _engine([7, _EOS], max_new_tokens=448)
        _transcribe(engine, 60.0, tmp_path)
        assert engine.model.generate.call_args.kwargs["max_new_tokens"] == 448

    def test_confidence_comes_from_the_chosen_token_recorder(self, tmp_path):
        """全語彙のスコアは残させず (output_scores を渡さない)、選ばれた token の log-prob をためて平均を出す。"""
        engine = _engine([7, 8, 9, _EOS], logprob=-0.2)
        result = _transcribe(engine, 1.0, tmp_path)
        kwargs = engine.model.generate.call_args.kwargs
        assert "output_scores" not in kwargs and kwargs["return_dict_in_generate"] is True
        assert [type(p).__name__ for p in kwargs["logits_processor"]] == ["ChosenTokenLogprobs"]
        engine.model.compute_transition_scores.assert_not_called()
        # 停止 token (EOS) は平均に入れない。どの token も -0.2
        assert result.engine_confidence.avg_logprob == pytest.approx(-0.2, abs=1e-6)
        assert result.confidence == pytest.approx(math.exp(-0.2), abs=1e-6)

    def test_stopping_at_the_limit_is_logged(self, tmp_path, caplog):
        """停止 token を出さずに上限で止まったら、書き起こしの後ろが欠けたと WARNING を出す。"""
        engine = _engine([7] * 600)
        with caplog.at_level("WARNING", logger=_LOGGER):
            result = _transcribe(engine, 10.0, tmp_path)
        assert result.text == "hello world"  # 切れたぶんまでの書き起こしは返す
        warnings = [r.getMessage() for r in caplog.records if "max_new_tokens limit (448)" in r.getMessage()]
        assert len(warnings) == 1 and "10.0 s of audio" in warnings[0]

    @pytest.mark.parametrize(
        "generated",
        [
            pytest.param([7] * 447 + [_EOS], id="stop-token-at-the-limit"),
            pytest.param([7] * 100 + [_EOS], id="below-the-limit"),
        ],
    )
    def test_finishing_with_the_stop_token_is_not_logged(self, tmp_path, caplog, generated):
        engine = _engine(generated)
        with caplog.at_level("WARNING", logger=_LOGGER):
            _transcribe(engine, 10.0, tmp_path)
        assert not [r for r in caplog.records if "max_new_tokens" in r.getMessage()]

    def test_limit_is_measured_after_resampling(self, tmp_path):
        """48 kHz で渡しても、長さは 16 kHz に変換したあとの秒数で数える。"""
        pytest.importorskip("librosa")
        engine = _engine([7, _EOS])
        _transcribe(engine, 60.0, tmp_path, sample_rate=48000)
        assert engine.model.generate.call_args.kwargs["max_new_tokens"] == 960


def test_default_params_do_not_fix_the_limit():
    """既定の設定は上限を渡さない (以前は 448 の固定)。constructor の既定も「音声の長さから決める」。"""
    assert "max_new_tokens" not in EngineMetadata.get("voxtral").default_params
    with patch("livecap_cli.engines.voxtral_engine.LibraryPreloader"):
        assert VoxtralEngine(device="cpu").max_new_tokens is None
