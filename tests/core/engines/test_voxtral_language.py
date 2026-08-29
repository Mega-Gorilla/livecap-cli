"""Voxtral の language 上流契約テスト (Issue #365)。

mistral-common `TranscriptionRequest.language` は `LanguageAlpha2 | None`
(自動検出 = None、具体言語 = ISO 639-1)。従来は `__init__` の生値
(default "auto" を含む) をそのまま `apply_transcription_request` へ渡して
おり契約外だった。`_asr_language` (auto/None -> None、concrete -> ISO 639-1)
を経由する修正を、モデルロードなしの mock processor で固定する。

**Issue #418**: 値は正しかったが**渡し方の形**がずれていた。上流の validator は
``str`` か list しか想定せず **bare ``None`` を弾く**ため、既定 (auto) の呼び出しが
必ず ``TypeError`` になっていた。``_processor_languages()` で 1 要素 list にする。
**mock では「auto が本当に auto か」は分からない** — それは
``tests/integration/engines/test_voxtral_language_contract.py`` が実 processor の
プロンプトで見る。

構築パターンは test_voxtral_sample_rate.py (#265) を踏襲。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livecap_cli.engines.voxtral_engine import VoxtralEngine


class TestResolveLanguage:
    """`_asr_language` への変換規則 (staticmethod、構築不要)"""

    @pytest.mark.parametrize("raw", [None, "", "auto", "AUTO", "Auto"])
    def test_auto_and_unset_resolve_to_none(self, raw):
        assert VoxtralEngine._resolve_language(raw) is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("en", "en"), ("EN", "en"), ("ja-JP", "ja"), ("pt-BR", "pt")],
    )
    def test_concrete_normalized_to_iso639_1(self, raw, expected):
        assert VoxtralEngine._resolve_language(raw) == expected


class TestProcessorLanguages:
    """``apply_transcription_request`` へ渡す**形** (Issue #418、staticmethod)"""

    @pytest.mark.parametrize(
        ("resolved", "expected"),
        [(None, [None]), ("en", ["en"]), ("fr", ["fr"])],
    )
    def test_wraps_in_a_single_element_list(self, resolved, expected):
        assert VoxtralEngine._processor_languages(resolved) == expected

    def test_never_returns_a_bare_value(self):
        """**bare ``None`` / bare ``str`` を返さない。**

        上流は ``str`` を自分で list 化するので ``"en"`` でも動くが、``None`` は
        弾かれる。**分岐を作らず常に list にする**ことで、auto と concrete が
        同じ経路を通る (片方だけ壊れる状態を作らない)。
        """
        for resolved in (None, "en", "fr"):
            assert isinstance(VoxtralEngine._processor_languages(resolved), list)


def _make_engine(asr_language) -> VoxtralEngine:
    """モデルロードなしで `_transcribe_single_chunk` を通せる最小 engine。"""
    engine = VoxtralEngine.__new__(VoxtralEngine)
    engine._initialized = True
    engine.model = MagicMock()
    engine.processor = MagicMock()
    engine.torch_device = "cpu"
    engine.language = "auto" if asr_language is None else asr_language
    engine._asr_language = asr_language
    engine.model_name = "test-model"
    engine.do_sample = False
    engine.max_new_tokens = 448

    mock_predicted_ids = MagicMock()
    engine.model.generate.return_value = mock_predicted_ids
    mock_predicted_ids.__getitem__ = MagicMock(return_value=mock_predicted_ids)
    engine.processor.batch_decode.return_value = ["hello world"]
    return engine


def _run_transcribe(engine: VoxtralEngine) -> None:
    audio = np.random.randn(int(16000 * 0.5)).astype(np.float32)

    with (
        patch("livecap_cli.engines.voxtral_engine.sf.write"),
        patch("livecap_cli.engines.voxtral_engine.get_temp_dir") as mock_temp_dir,
        patch("torch.no_grad"),
    ):
        mock_temp_dir.return_value = MagicMock()
        temp_path_mock = MagicMock()
        temp_path_mock.exists.return_value = True
        mock_temp_dir.return_value.__truediv__ = MagicMock(return_value=temp_path_mock)

        engine._transcribe_single_chunk(audio, 16000)


class TestUpstreamLanguageContract:
    """`apply_transcription_request` へ渡る language を固定 (受け入れ基準 #365)"""

    def test_auto_passes_a_single_element_none_list(self):
        """未指定 / --language auto -> language=[None] (Issue #418)

        **bare ``None`` を渡してはならない** — 上流の validator が弾き、
        既定の呼び出しが必ず ``TypeError`` になる。
        """
        engine = _make_engine(asr_language=None)

        _run_transcribe(engine)

        call_kwargs = engine.processor.apply_transcription_request.call_args.kwargs
        assert call_kwargs["language"] == [None]

    def test_concrete_language_passes_iso_code_upstream(self):
        """--language en -> language="en" """
        engine = _make_engine(asr_language="en")

        _run_transcribe(engine)

        call_kwargs = engine.processor.apply_transcription_request.call_args.kwargs
        assert call_kwargs["language"] == ["en"]

    def test_constructor_wires_asr_language(self):
        """実 constructor 経由でも auto -> None / en -> en が設定される"""
        with patch("livecap_cli.engines.voxtral_engine.LibraryPreloader"):
            auto_engine = VoxtralEngine(device="cpu")  # default language="auto"
            en_engine = VoxtralEngine(device="cpu", language="en")

        assert auto_engine._asr_language is None
        assert auto_engine.language == "auto"  # 生値はログ用に保持
        assert en_engine._asr_language == "en"
