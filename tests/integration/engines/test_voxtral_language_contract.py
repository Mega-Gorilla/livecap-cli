"""Voxtral の language が上流でどう解釈されるかを、実 processor で固定する (Issue #418)。

**mock では「auto が本当に auto か」を確かめられない。** `tests/core/engines/
test_voxtral_language.py` は `apply_transcription_request` へ渡る**形**
(`[None]` / `["en"]`) を固定するが、mock なので上流が

- その `None` を「言語指定なし」として扱うのか
- 黙って既定言語へ落とすのか

を区別できない。**後者だと、英語音声のテストは通ったまま非英語で静かに劣化する** —
#377 (ReazonSpeech の fail-silent) と同じ形である。ここでは**生成されるプロンプト
そのもの**を見て、`lang:` トークンの有無で判定する。

**推論はしない。** tokenizer / preprocessor だけを読むのでモデル重みは不要 (ただし
snapshot ディレクトリは要るので、無ければ skip する)。

**token 数は pin しない** — tokenizer の更新で動くため。守りたいのは
「auto には言語トークンが無く、明示指定にはある」という**意味**の方である。

`engine_smoke` + `slow` なので通常の smoke step では収集されない。**実モデルが
常駐する self-hosted runner で PASSED を要求する** (`engine-smoke-gpu` の
「Run Voxtral language contract check」)。ゲートを置かないと**どこでも走らない
テスト**になる (#409 で踏んだ形)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.engine_smoke, pytest.mark.slow]

#: 音声は 1 件だけ渡すので、language list の長さも 1 でなければならない。
_AUDIO = Path(__file__).resolve().parents[2] / "assets" / "audio" / "en" / "librispeech_1089-134686-0001.wav"


def _local_model_dir() -> Path | None:
    """production と同じ解決規則でローカル snapshot を探す。"""
    from livecap_cli.engines.voxtral_engine import VoxtralEngine
    from livecap_cli.utils import get_models_dir

    # **path 解決規則を複製しない** — production の override をそのまま使う。
    # (`_get_local_model_path` は report_progress を通るので最小限の属性を与える)
    engine = VoxtralEngine.__new__(VoxtralEngine)
    engine.engine_name = "voxtral"
    engine.progress_callback = None
    engine.model_name = "mistralai/Voxtral-Mini-3B-2507"
    # base_engine.py:256 と同じく **engine 名を渡さない** models root を使う
    path = engine._get_local_model_path(get_models_dir())
    return path if (path / "tekken.json").is_file() else None


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    model_dir = _local_model_dir()
    if model_dir is None:
        pytest.skip("Voxtral の実 snapshot が無い")
    return AutoProcessor.from_pretrained(str(model_dir))


def _prompt(processor, language) -> str:
    """`apply_transcription_request` が組み立てるプロンプトを復元する。"""
    batch = processor.apply_transcription_request(
        language=language,
        audio=str(_AUDIO),
        model_id="mistralai/Voxtral-Mini-3B-2507",
    )
    return processor.tokenizer.decode(batch["input_ids"][0].tolist())


def test_auto_prompt_has_no_language_token(processor) -> None:
    """`[None]` = **言語指定なし**。既定言語へ落ちていない。"""
    prompt = _prompt(processor, [None])

    assert "lang:" not in prompt, (
        "auto のはずのプロンプトに言語トークンが入っている — "
        f"上流が既定言語へ落としている可能性がある: {prompt[-80:]!r}"
    )


#: voxtral が対応する言語から 2 つ (`ja` は engine 側で拒否されるので使わない)
@pytest.mark.parametrize("code", ["en", "fr"])
def test_concrete_language_prompt_has_language_token(processor, code: str) -> None:
    """明示指定は `lang:<code>` として渡る。"""
    prompt = _prompt(processor, [code])

    assert f"lang:{code}" in prompt, (
        f"明示指定 {code!r} がプロンプトへ反映されていない: {prompt[-80:]!r}"
    )


def test_auto_and_concrete_prompts_differ(processor) -> None:
    """**両者が同一なら上のどちらの assert も無意味**になる。前提を固定する。"""
    assert _prompt(processor, [None]) != _prompt(processor, ["en"])


def test_bare_none_is_still_rejected_upstream(processor) -> None:
    """**bare `None` を弾く上流の挙動そのもの**を記録する (Issue #418 の原因)。

    上流が `None` を受け入れるようになったら**ここが落ちる**。そのときは
    `_processor_languages()` の必要性を再検討する trigger になる。
    """
    with pytest.raises(TypeError):
        _prompt(processor, None)
