"""adapter の言語データが正本 (EngineMetadata / data module) から派生することの固定 (Issue #230)。

- canary / reazonspeech / parakeet(_ja) / voxtral: `get_supported_languages()`
  が metadata と同一データを返す (モデルロード不要 — `__new__` + `engine_name`
  設定のみで呼べる method 実装であることも contract の一部)
- whispers2t / qwen3asr: data-only module 正本からの派生 (静的比較)
- qwen3asr の意図的差 (constructor=auto / CLI=ja) の pin
- auto 対応 adapter がモデルロードなしで上流値へ変換できること
"""

from __future__ import annotations

import inspect

import pytest

from livecap_cli.engines.metadata import EngineMetadata
from livecap_cli.engines.canary_engine import CanaryEngine
from livecap_cli.engines.parakeet_engine import ParakeetEngine
from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine
from livecap_cli.engines.qwen3asr_languages import QWEN_ASR_LANGUAGE_NAMES
from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine
from livecap_cli.engines.voxtral_engine import VoxtralEngine
from livecap_cli.engines.whisper_languages import WHISPER_LANGUAGES
from livecap_cli.engines.whispers2t_engine import WhisperS2TEngine


def _adapter_languages(engine_cls, engine_name: str) -> list:
    """モデルロードなしで adapter の get_supported_languages() を呼ぶ。

    method は self.engine_name のみに依存する契約 (#230) — constructor や
    モデル状態への依存が入るとこの helper が壊れて検出される。
    """
    engine = engine_cls.__new__(engine_cls)
    engine.engine_name = engine_name
    return engine.get_supported_languages()


class TestAdapterDerivesFromMetadata:
    """metadata 正本 engine 群: adapter が同一データを返す"""

    @pytest.mark.parametrize(
        ("engine_cls", "engine_name"),
        [
            (CanaryEngine, "canary"),
            (ReazonSpeechEngine, "reazonspeech"),
            (VoxtralEngine, "voxtral"),
            (ParakeetEngine, "parakeet"),
            (ParakeetEngine, "parakeet_ja"),
        ],
    )
    def test_adapter_matches_metadata(self, engine_cls, engine_name):
        expected = list(EngineMetadata.get(engine_name).supported_languages)

        result = _adapter_languages(engine_cls, engine_name)

        assert result == expected
        # 返却は毎回独立した list (caller の変更が正本へ届かない)
        result.append("xx")
        assert "xx" not in EngineMetadata.get(engine_name).supported_languages


class TestDataModuleAuthorities:
    """data-only module 正本 engine 群: adapter と metadata が同源から派生"""

    def test_whispers2t_adapter_and_metadata_share_source(self):
        engine = WhisperS2TEngine.__new__(WhisperS2TEngine)
        assert engine.get_supported_languages() == list(WHISPER_LANGUAGES)
        assert EngineMetadata.get("whispers2t").supported_languages == tuple(
            WHISPER_LANGUAGES
        )

    def test_qwen3asr_adapter_and_metadata_share_source(self):
        # class 属性 alias は data module と同一 object (派生の同源性)
        assert Qwen3ASREngine.QWEN_ASR_LANGUAGE_NAMES is QWEN_ASR_LANGUAGE_NAMES
        assert Qwen3ASREngine.SUPPORTED_LANGUAGES == tuple(QWEN_ASR_LANGUAGE_NAMES)
        assert EngineMetadata.get("qwen3asr").supported_languages == tuple(
            QWEN_ASR_LANGUAGE_NAMES
        )

        engine = Qwen3ASREngine.__new__(Qwen3ASREngine)
        assert engine.get_supported_languages() == list(QWEN_ASR_LANGUAGE_NAMES)


class TestAuthorityImmutability:
    """正本への変更経路の封鎖 (#230 レビュー: field 再代入 / map item 代入)"""

    def test_engine_info_field_reassignment_blocked(self):
        """frozen dataclass: `info.supported_languages += (...)` の field
        再代入経路を封鎖 (tuple 化だけでは防げなかった hole)"""
        import dataclasses

        info = EngineMetadata.get("canary")
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.supported_languages = ("en", "de", "fr", "es", "ja")  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.supported_languages += ("ja",)  # type: ignore[misc]

        # 受理判定は不変
        with pytest.raises(ValueError, match="not supported by engine 'canary'"):
            EngineMetadata.resolve_language("canary", "ja")

    def test_qwen_language_map_item_assignment_blocked(self):
        """MappingProxyType: 正本 map への item 代入を封鎖 —
        adapter だけが新言語を受理する split-brain を防ぐ"""
        with pytest.raises(TypeError):
            Qwen3ASREngine.QWEN_ASR_LANGUAGE_NAMES["xx"] = "Injected"  # type: ignore[index]

        # adapter / metadata とも契約不変
        with pytest.raises(ValueError):
            Qwen3ASREngine._resolve_language("xx")
        with pytest.raises(ValueError):
            EngineMetadata.resolve_language("qwen3asr", "xx")

    def test_qwen_supported_languages_is_tuple(self):
        assert isinstance(Qwen3ASREngine.SUPPORTED_LANGUAGES, tuple)


class TestQwen3ASRIntentionalDivergence:
    """qwen3asr: constructor default (auto) と CLI 既定 (ja) の意図的差を固定。

    PR-A.5.2 の設計判断 — CLI は confidence filter の avg_logprob 経路を
    既定で有効にするため ja、直接 API 利用は従来どおり自動検出。
    `cli_default_language` は CLI policy であり constructor default との
    一致を要求しない (issue #230 レビュー合意)。
    """

    def test_constructor_default_is_auto_detect(self):
        sig = inspect.signature(Qwen3ASREngine.__init__)
        assert sig.parameters["language"].default is None

    def test_cli_default_is_ja(self):
        assert EngineMetadata.get("qwen3asr").cli_default_language == "ja"


class TestAutoAdaptersConvertWithoutModelLoad:
    """auto 対応 adapter はモデルロードなしで有効な上流値へ変換できる。

    具体的な上流値 (None) は engine 別テスト — voxtral は
    test_voxtral_language.py::TestUpstreamLanguageContract 参照。
    """

    def test_qwen3asr_auto_resolves_without_model(self):
        assert Qwen3ASREngine._resolve_language("auto") is None
        assert Qwen3ASREngine._resolve_language(None) is None
        assert Qwen3ASREngine._resolve_language("ja") == "Japanese"

    def test_voxtral_auto_resolves_without_model(self):
        assert VoxtralEngine._resolve_language("auto") is None
