"""Qwen3-ASR エンジンのユニットテスト"""
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np

import pytest

from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.engines.metadata import EngineInfo, EngineMetadata


class TestQwen3ASREngineMetadata:
    """Qwen3-ASR エンジンメタデータのテスト"""

    def test_qwen3asr_engine_registered(self):
        """qwen3asr エンジンが EngineMetadata に登録されていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert info.id == "qwen3asr"
        assert info.display_name == "Qwen3-ASR 0.6B"
        assert info.module == ".qwen3asr_engine"
        assert info.class_name == "Qwen3ASREngine"

    def test_qwen3asr_supported_languages(self):
        """qwen3asr エンジンが 30 言語をサポートしていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        # 30 languages supported
        assert len(info.supported_languages) == 30
        # Key languages are supported
        assert "ja" in info.supported_languages  # Japanese
        assert "en" in info.supported_languages  # English
        assert "zh" in info.supported_languages  # Chinese
        assert "ko" in info.supported_languages  # Korean
        assert "yue" in info.supported_languages  # Cantonese

    def test_qwen3asr_device_support(self):
        """qwen3asr エンジンが CPU と CUDA をサポートしていることを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert "cpu" in info.device_support
        assert "cuda" in info.device_support

    def test_qwen3asr_default_params(self):
        """qwen3asr エンジンのデフォルトパラメータを確認"""
        info = EngineMetadata.get("qwen3asr")
        assert info is not None
        assert info.default_params["model_name"] == "Qwen/Qwen3-ASR-0.6B-hf"
        assert info.default_params["engine_id"] == "qwen3asr"

    def test_qwen3asr_in_available_engines(self):
        """qwen3asr が利用可能なエンジンリストに含まれていることを確認"""
        engines = EngineFactory.get_available_engines()
        assert "qwen3asr" in engines

    def test_qwen3asr_in_japanese_engines(self):
        """qwen3asr が日本語対応エンジンリストに含まれていることを確認"""
        ja_engines = EngineFactory.get_engines_for_language("ja")
        assert "qwen3asr" in ja_engines

    def test_qwen3asr_in_chinese_engines(self):
        """qwen3asr が中国語対応エンジンリストに含まれていることを確認"""
        zh_engines = EngineFactory.get_engines_for_language("zh")
        assert "qwen3asr" in zh_engines

    def test_qwen3asr_in_cantonese_engines(self):
        """qwen3asr が広東語対応エンジンリストに含まれていることを確認"""
        yue_engines = EngineFactory.get_engines_for_language("yue")
        assert "qwen3asr" in yue_engines


class TestQwen3ASREngineCreation:
    """Qwen3-ASR エンジン作成のテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_creation_with_defaults(self, mock_availability):
        """デフォルトパラメータでエンジンを作成できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        assert engine.engine_name == "qwen3asr"
        assert engine.model_name == "Qwen/Qwen3-ASR-0.6B-hf"
        assert engine.language is None  # 自動検出
        assert engine.torch_device == "cpu"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_creation_with_custom_params(self, mock_availability):
        """カスタムパラメータでエンジンを作成できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(
            device="cpu",
            language="ja",
            model_name="Qwen/Qwen3-ASR-1.7B-hf",
            engine_id="qwen3asr_large",
        )

        assert engine.engine_name == "qwen3asr_large"
        assert engine.model_name == "Qwen/Qwen3-ASR-1.7B-hf"
        assert engine.language == "ja"
        assert engine._asr_language == "Japanese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_get_supported_languages(self, mock_availability):
        """サポート言語リストを取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        languages = engine.get_supported_languages()

        assert len(languages) == 30
        assert "ja" in languages
        assert "en" in languages
        assert "zh" in languages

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_get_required_sample_rate(self, mock_availability):
        """要求サンプルレートが 16kHz であることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        assert engine.get_required_sample_rate() == 16000

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_get_engine_name_0_6b(self, mock_availability):
        """0.6B モデルのエンジン名を取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", model_name="Qwen/Qwen3-ASR-0.6B-hf")
        assert engine.get_engine_name() == "Qwen3-ASR 0.6B"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_get_engine_name_1_7b(self, mock_availability):
        """1.7B モデルのエンジン名を取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", model_name="Qwen/Qwen3-ASR-1.7B-hf")
        assert engine.get_engine_name() == "Qwen3-ASR 1.7B"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_engine_get_model_metadata(self, mock_availability):
        """モデルメタデータを取得できることを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")
        metadata = engine.get_model_metadata()

        assert metadata["name"] == "Qwen/Qwen3-ASR-0.6B-hf"
        assert metadata["version"] == "0.6B"
        assert metadata["format"] == "transformers"


class TestQwen3ASRAvailabilityCheck:
    """Qwen3-ASR 可用性チェックのテスト"""

    def test_check_qwen3asr_availability_not_installed(self):
        """transformers の Qwen3-ASR を import できない場合 False を返すことを確認"""
        import livecap_cli.engines.qwen3asr_engine as module

        # Reset the global state
        original_value = module.QWEN3ASR_AVAILABLE
        module.QWEN3ASR_AVAILABLE = None

        try:
            # 非 frozen 環境でインポートが ImportError を発生させる場合をシミュレート
            def mock_import(name, *args, **kwargs):
                if name == "transformers":
                    raise ImportError("cannot import name 'Qwen3ASRForConditionalGeneration'")
                return original_import(name, *args, **kwargs)

            import builtins
            original_import = builtins.__import__

            with patch.object(builtins, "__import__", side_effect=mock_import):
                # Force re-check
                module.QWEN3ASR_AVAILABLE = None
                result = module.check_qwen3asr_availability()
                assert result is False
        finally:
            # Reset for other tests
            module.QWEN3ASR_AVAILABLE = original_value

    def test_check_qwen3asr_availability_caches_result(self):
        """可用性チェックの結果がキャッシュされることを確認"""
        import livecap_cli.engines.qwen3asr_engine as module

        # Save original value
        original_value = module.QWEN3ASR_AVAILABLE

        try:
            # Set cached value
            module.QWEN3ASR_AVAILABLE = True

            # Should return cached value without checking
            result = module.check_qwen3asr_availability()
            assert result is True
        finally:
            # Reset for other tests
            module.QWEN3ASR_AVAILABLE = original_value


class TestQwen3ASRDependencyCheck:
    """Qwen3-ASR 依存関係チェックのテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_check_dependencies_raises_when_not_available(self, mock_availability):
        """transformers の Qwen3-ASR が利用不可の場合 ImportError が発生することを確認"""
        mock_availability.return_value = False

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        with pytest.raises(ImportError, match="Qwen3-ASR requires transformers"):
            engine._check_dependencies()

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_check_dependencies_passes_when_available(self, mock_availability):
        """transformers の Qwen3-ASR が利用可能な場合、依存関係チェックが成功することを確認"""
        mock_availability.return_value = True

        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        # Should not raise
        engine._check_dependencies()


class TestQwen3ASRTranscribeValidation:
    """Qwen3-ASR transcribe メソッドのバリデーションテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_transcribe_raises_when_not_initialized(self, mock_availability):
        """初期化前に transcribe を呼び出すと RuntimeError が発生することを確認"""
        mock_availability.return_value = True

        import numpy as np
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu")

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)


class TestQwen3ASRLanguageConversion:
    """Qwen3-ASR 言語コード変換のテスト"""

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_iso639_1_conversion(self, mock_availability):
        """ISO 639-1 コードが言語名に変換されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="ja")
        assert engine.language == "ja"
        assert engine._asr_language == "Japanese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_iso639_3_conversion(self, mock_availability):
        """ISO 639-3 コード (yue) が Cantonese に変換されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="yue")
        assert engine._asr_language == "Cantonese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_bcp47_conversion(self, mock_availability):
        """BCP-47 コード (zh-CN) が Chinese に変換されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="zh-CN")
        assert engine._asr_language == "Chinese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_language_name_passthrough(self, mock_availability):
        """言語名がそのままパススルーされることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="Japanese")
        assert engine._asr_language == "Japanese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_language_name_case_insensitive(self, mock_availability):
        """言語名の大文字小文字を区別しないことを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="japanese")
        assert engine._asr_language == "Japanese"

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_auto_detect_none(self, mock_availability):
        """None が自動検出としてパススルーされることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language=None)
        assert engine._asr_language is None

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_auto_string_normalized_to_none(self, mock_availability):
        """'auto' が None（自動検出）に正規化されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="auto")
        assert engine._asr_language is None

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_empty_string_normalized_to_none(self, mock_availability):
        """空文字が None（自動検出）に正規化されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        engine = Qwen3ASREngine(device="cpu", language="")
        assert engine._asr_language is None

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_unsupported_language_raises_error(self, mock_availability):
        """未対応の言語コードで ValueError が発生することを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        with pytest.raises(ValueError, match="Unsupported language"):
            Qwen3ASREngine(device="cpu", language="zz")

    @patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability")
    def test_all_supported_languages_convert(self, mock_availability):
        """全30言語の ISO コードが正しく変換されることを確認"""
        mock_availability.return_value = True
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        for iso_code, expected_name in Qwen3ASREngine.QWEN_ASR_LANGUAGE_NAMES.items():
            engine = Qwen3ASREngine(device="cpu", language=iso_code)
            assert engine._asr_language == expected_name, (
                f"Language '{iso_code}' should convert to '{expected_name}', "
                f"got '{engine._asr_language}'"
            )

    def test_supported_languages_matches_language_names_keys(self):
        """SUPPORTED_LANGUAGES と QWEN_ASR_LANGUAGE_NAMES のキーが一致することを確認"""
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        assert set(Qwen3ASREngine.SUPPORTED_LANGUAGES) == set(Qwen3ASREngine.QWEN_ASR_LANGUAGE_NAMES.keys()), (
            f"SUPPORTED_LANGUAGES と QWEN_ASR_LANGUAGE_NAMES のキーが不一致: "
            f"差分={set(Qwen3ASREngine.SUPPORTED_LANGUAGES) ^ set(Qwen3ASREngine.QWEN_ASR_LANGUAGE_NAMES.keys())}"
        )


class _FakeBatch(dict):
    """``apply_transcription_request`` の戻り値 (``BatchFeature``) の代わり。``.to()`` を記録する。"""

    def __init__(self, prompt_len: int):
        super().__init__(input_ids=SimpleNamespace(shape=(1, prompt_len)))
        self.moved_to: list = []

    def to(self, target):
        self.moved_to.append(target)
        return self


class _FakeSequences:
    """``outputs.sequences`` の代わり。``[:, prompt_len:]`` で生成部分を返す。"""

    def __init__(self, prompt_len: int, generated: list):
        self.prompt_len = prompt_len
        self.generated = generated

    def __getitem__(self, index):
        rows, cols = index
        assert rows == slice(None) and cols == slice(self.prompt_len, None)
        return [self.generated]


def _loaded_engine(language: Optional[str], *, generated: list, decoded: str, scores=None):
    """model / processor を fake に差し替えた、ロード済みの engine。"""
    from livecap_cli.engines.qwen3asr_engine import Qwen3ASRBundle, Qwen3ASREngine

    with patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability", return_value=True):
        engine = Qwen3ASREngine(device="cpu", language=language)
    prompt_len = 5
    batch = _FakeBatch(prompt_len)
    outputs = SimpleNamespace(sequences=_FakeSequences(prompt_len, generated), scores=object())
    processor = MagicMock()
    processor.apply_transcription_request.return_value = batch
    processor.batch_decode.return_value = [decoded]
    processor.decode.return_value = decoded
    processor.tokenizer = SimpleNamespace(eos_token_id=151645, pad_token_id=151645, bos_token_id=None)
    model = MagicMock()
    model.device = "cpu"
    model.dtype = "bfloat16"
    model.generate.return_value = outputs
    model.generation_config = SimpleNamespace(eos_token_id=[151643, 151645], pad_token_id=151645, bos_token_id=None)
    model.compute_transition_scores.return_value = [scores if scores is not None else []]
    engine.model = Qwen3ASRBundle(model=model, processor=processor)
    engine._initialized = True
    return engine, model, processor, batch


class TestQwen3ASRTranscribe:
    """transformers 本体の Qwen3-ASR で転写する経路 (#471)。"""

    def test_forced_language_uses_the_pr_a52_prompt_and_generation_settings(self):
        """言語を指定したときは PR-A.5.2 と同じ prompt / penalty で生成し、本文を素の batch_decode で取る。"""
        engine, model, processor, batch = _loaded_engine("ja", generated=[11, 12, 151645], decoded="こんにちは", scores=[-0.1, -0.3, -0.01])

        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)

        request = processor.apply_transcription_request.call_args
        assert request.kwargs["language"] == "Japanese"
        assert request.kwargs["prompt"] == "You are a speech recognition model."
        assert batch.moved_to == ["cpu", "bfloat16"]
        generate = model.generate.call_args.kwargs
        assert generate["max_new_tokens"] == 256
        assert generate["repetition_penalty"] == 1.1
        assert generate["no_repeat_ngram_size"] == 3
        assert generate["do_sample"] is False
        assert generate["output_scores"] is True
        assert generate["return_dict_in_generate"] is True
        processor.batch_decode.assert_called_once_with(
            [[11, 12, 151645]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        assert result.text == "こんにちは"
        # 生成を止めた token (151645) は平均に入れない
        assert result.engine_confidence.avg_logprob == pytest.approx(-0.2)
        assert result.confidence == pytest.approx(np.exp(-0.2))

    def test_both_stop_tokens_are_excluded_from_avg_logprob(self):
        """``-hf`` の tokenizer の pad は 151645 なので、151643 (もう 1 つの停止 token) も除外する。"""
        engine, _model, _processor, _batch = _loaded_engine("en", generated=[21, 151643], decoded="hi", scores=[-0.5, -0.001])

        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)

        assert result.engine_confidence.avg_logprob == pytest.approx(-0.5)

    def test_auto_language_matches_the_previous_qwen_asr_transcribe(self):
        """自動検出は以前の qwen-asr の transcribe と同じ: system prompt 空、penalty なし、512 token、本文を解析。"""
        engine, model, processor, _batch = _loaded_engine(None, generated=[31, 32], decoded="hello")

        result = engine.transcribe(np.zeros(16000, dtype=np.float32), 16000)

        request = processor.apply_transcription_request.call_args
        assert request.kwargs["language"] is None
        assert request.kwargs["prompt"] is None
        generate = model.generate.call_args.kwargs
        assert generate["max_new_tokens"] == 512
        assert "repetition_penalty" not in generate and "no_repeat_ngram_size" not in generate
        assert generate["do_sample"] is False
        processor.decode.assert_called_once_with([31, 32], return_format="transcription_only")
        model.compute_transition_scores.assert_not_called()
        assert result.text == "hello"
        assert not result.engine_confidence.is_available  # fail-open (PR-A.5.2 から変えていない)

    def test_too_short_audio_skips_generation(self):
        engine, model, _processor, _batch = _loaded_engine("ja", generated=[1], decoded="x")

        result = engine.transcribe(np.zeros(100, dtype=np.float32), 16000)

        assert result.text == ""
        model.generate.assert_not_called()


class TestQwen3ASRLoad:
    """transformers 本体でのロード (#471)。"""

    def test_incomplete_weights_fail_loud(self):
        """欠けた / 余った重みは乱数で初期化したまま使わず、例外にする (別形式の dir を渡したとき)。"""
        from livecap_cli.engines.qwen3asr_engine import _require_complete_weights

        with pytest.raises(RuntimeError, match="missing_keys=2"):
            _require_complete_weights(
                {"missing_keys": ["model.audio_tower.conv2d1.weight", "lm_head.weight"], "unexpected_keys": []},
                Path("model-dir"),
            )
        with pytest.raises(RuntimeError, match="unexpected_keys=1"):
            _require_complete_weights({"unexpected_keys": ["thinker.model.embed_tokens.weight"]}, Path("model-dir"))

    def test_complete_weights_pass(self):
        from livecap_cli.engines.qwen3asr_engine import _require_complete_weights

        _require_complete_weights({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}, Path("model-dir"))

    def test_load_reads_the_local_dir_and_invalidates_it_on_incomplete_weights(self, tmp_path):
        """ローカル dir を processor / model に渡し、重みが合わなければ manifest を無効化する (self-heal)。"""
        from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine

        with patch("livecap_cli.engines.qwen3asr_engine.check_qwen3asr_availability", return_value=True):
            engine = Qwen3ASREngine(device="cpu", dtype="float32")
        fake_model = MagicMock()
        model_cls = MagicMock()
        model_cls.from_pretrained.return_value = (fake_model, {"missing_keys": ["lm_head.weight"]})
        processor_cls = MagicMock()
        fake_transformers = SimpleNamespace(AutoProcessor=processor_cls, Qwen3ASRForConditionalGeneration=model_cls)

        with patch.dict("sys.modules", {"transformers": fake_transformers}), \
                patch.object(engine, "_require_model_dir", return_value=tmp_path), \
                patch.object(engine, "_invalidate_model_dir") as invalidate, \
                patch("livecap_cli.engines.qwen3asr_engine.ModelMemoryCache.get", return_value=None):
            with pytest.raises(RuntimeError, match="missing_keys=1"):
                engine._load_model_from_path(tmp_path)

        processor_cls.from_pretrained.assert_called_once_with(str(tmp_path))
        args, kwargs = model_cls.from_pretrained.call_args
        assert args == (str(tmp_path),)
        assert kwargs == {"device_map": "cpu", "dtype": "float32", "output_loading_info": True}
        invalidate.assert_called_once()
