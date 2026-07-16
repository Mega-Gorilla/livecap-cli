"""CLI file 文字起こし経路 (`_transcribe_file`) の E2E regression テスト (Issue #363)。

Phase 6B 以来 `FileTranscriptionPipeline` の実在しない API を呼んで全滅していた
経路の復旧を固定する。方針 (issue #363 受け入れ基準):

- engine / torch / FFmpeg / network 不要 (plain WAV + fake engine/translator)
- **pipeline 自体は mock しない** — 実 `process_file()` を通し CLI との契約を固定
- fake engine は実 `TranscriptionResult` (engines.base_engine) を返す
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from livecap_cli import cli
from livecap_cli.engines.engine_factory import EngineFactory
from livecap_cli.engines.base_engine import TranscriptionResult as EngineResult
from livecap_cli.translation.base import BaseTranslator
from livecap_cli.translation.factory import TranslatorFactory
from livecap_cli.translation.result import TranslationResult


# ---------------------------------------------------------------- helpers ----


def _write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.zeros(int(sample_rate * seconds), dtype=np.int16).tobytes())


def _three_segmenter(audio, sample_rate):
    """3 秒音声を 1 秒 ×3 segment に分割 (per-segment 挙動の制御用)。"""
    return [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


class FakeEngine:
    """実 EngineResult を返す fake。呼び出し記録付き。"""

    def __init__(
        self,
        text: str = "こんにちは",
        raise_error: bool = False,
        raise_on_load: bool = False,
    ):
        self._text = text
        self._raise = raise_error
        self._raise_on_load = raise_on_load
        self.load_model_calls = 0
        self.transcribe_calls = 0
        self.cleanup_calls = 0

    def load_model(self) -> None:
        self.load_model_calls += 1
        if self._raise_on_load:
            raise RuntimeError("model download failed")

    def transcribe(self, audio, sample_rate) -> EngineResult:
        self.transcribe_calls += 1
        if self._raise:
            raise RuntimeError("model broken")
        return EngineResult(text=self._text, confidence=0.9)

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeTranslator(BaseTranslator):
    """BaseTranslator 契約準拠の fake。

    `_initialized=False` 起点 + `load_model()` で True — CLI が `load_model()`
    を呼ばないと pipeline の `_validate_translator_params` が ValueError を
    出すため、translator lifecycle の契約 (#363) をテストで検出できる。
    """

    def __init__(self, fail_on_calls: Optional[set[int]] = None, fail_all: bool = False):
        super().__init__(default_context_sentences=0)
        self._initialized = False
        self._fail_on_calls = fail_on_calls or set()
        self._fail_all = fail_all
        self.translate_calls: list[tuple[str, str, str]] = []
        self.load_model_calls = 0
        self.cleanup_calls = 0

    def load_model(self) -> None:
        self.load_model_calls += 1
        self._initialized = True

    def translate(self, text, source_lang, target_lang, context=None) -> TranslationResult:
        call_no = len(self.translate_calls) + 1
        self.translate_calls.append((text, source_lang, target_lang))
        if self._fail_all or call_no in self._fail_on_calls:
            raise RuntimeError("translation boom")
        return TranslationResult(
            text=f"EN:{text}",
            original_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    def get_supported_pairs(self):
        return []

    def get_translator_name(self) -> str:
        return "fake"

    def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.fixture(autouse=True)
def neutral_file_segmenter(monkeypatch):
    """既存 E2E を #366 Phase 1 以前の挙動 (全音声 1 segment) に固定する seam。

    `_build_file_segmenter` は既定 (--vad auto) で実 VADProcessor (Silero 等の
    実 backend) を構築するため、無音 WAV を使う本 suite では segment 0 件に
    なり content assert が成立しない。None (segmenter 未注入 = 全音声
    fallback) に差し替え、VAD 配線自体は TestVadFileMode で検証する。
    実関数を yield するので、必要なテストは再 patch で実挙動に戻せる。
    """
    real = cli._build_file_segmenter
    monkeypatch.setattr(cli, "_build_file_segmenter", lambda args: None)
    yield real


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.wav"
    _write_wav(path)
    return path


@pytest.fixture
def fake_engine(monkeypatch) -> FakeEngine:
    engine = FakeEngine()
    _patch_engine_factory(monkeypatch, engine)
    return engine


def _patch_engine_factory(monkeypatch, engine: FakeEngine) -> dict:
    calls: dict = {"count": 0, "kwargs": None}

    def fake_create_engine(engine_type, device=None, **engine_options):
        calls["count"] += 1
        calls["kwargs"] = {"engine_type": engine_type, "device": device, **engine_options}
        return engine

    monkeypatch.setattr(EngineFactory, "create_engine", fake_create_engine)
    return calls


def _patch_translator_factory(monkeypatch, translator: FakeTranslator) -> dict:
    calls: dict = {"count": 0, "kwargs": None}

    def fake_create_translator(translator_type, **translator_options):
        calls["count"] += 1
        calls["kwargs"] = {"translator_type": translator_type, **translator_options}
        return translator

    monkeypatch.setattr(TranslatorFactory, "create_translator", fake_create_translator)
    return calls


def _patch_three_segment_pipeline(monkeypatch) -> None:
    """実 pipeline のまま segmenter だけ注入 (複数 segment ケース用)。

    #366 Phase 1 以降、CLI は `_build_file_segmenter` の結果を pipeline へ
    明示的に渡すため、この seam を差し替える (autouse の neutral patch より
    後に適用され上書きされる)。
    """
    monkeypatch.setattr(cli, "_build_file_segmenter", lambda args: _three_segmenter)


# ---------------------------------------------------------------- tests ----


class TestFileTranscriptionSuccess:
    def test_output_file(self, wav_path, fake_engine, tmp_path, capsys):
        out = tmp_path / "result.srt"

        rc = cli.main(["transcribe", str(wav_path), "-o", str(out)])

        assert rc == 0
        content = out.read_text(encoding="utf-8")
        assert "こんにちは" in content
        assert "00:00:00,000 --> 00:00:01,000" in content
        # 入力横への sidecar を生成しない (write_subtitles=False)
        assert not wav_path.with_suffix(".srt").exists()
        # stdout に SRT を混入させない (進捗は stderr)
        captured = capsys.readouterr()
        assert "こんにちは" not in captured.out
        assert "Output written to:" in captured.err

    def test_stdout_when_no_output_option(self, wav_path, fake_engine, capsys):
        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        captured = capsys.readouterr()
        assert "こんにちは" in captured.out
        assert "00:00:00,000 --> 00:00:01,000" in captured.out

    def test_engine_cleanup_called(self, wav_path, fake_engine):
        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        assert fake_engine.load_model_calls == 1
        assert fake_engine.cleanup_calls == 1


class TestFileTranscriptionFailure:
    def test_total_asr_failure_exits_1_and_creates_no_file(
        self, wav_path, monkeypatch, tmp_path, capsys
    ):
        engine = FakeEngine(raise_error=True)
        _patch_engine_factory(monkeypatch, engine)
        out = tmp_path / "result.srt"

        rc = cli.main(["transcribe", str(wav_path), "-o", str(out)])

        assert rc == 1
        assert not out.exists()
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "ASR segment calls failed" in captured.err  # #362 の error 転送
        assert engine.cleanup_calls == 1  # 失敗経路でも cleanup

    def test_load_model_failure_cleans_up_engine(self, wav_path, monkeypatch, capsys):
        """load_model() 例外時も engine.cleanup() が呼ばれる (codex-review 指摘)。

        `_load_engine()` が caller へ engine を返す前に失敗するケースは
        caller の finally では拾えないため、helper 内で cleanup して re-raise
        する契約を固定 (realtime / file 両経路を共通 helper で保護)。
        """
        engine = FakeEngine(raise_on_load=True)
        _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 1
        assert "model download failed" in capsys.readouterr().err
        assert engine.load_model_calls == 1
        assert engine.cleanup_calls == 1

    def test_missing_input_validated_before_model_load(self, monkeypatch, capsys):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", "no_such_file.wav"])

        assert rc == 1
        assert "File not found" in capsys.readouterr().err
        assert factory_calls["count"] == 0  # モデルロード前に検証


class TestTranslation:
    def test_translate_success_writes_translated_srt(
        self, wav_path, fake_engine, monkeypatch, tmp_path, capsys
    ):
        translator = FakeTranslator()
        factory_calls = _patch_translator_factory(monkeypatch, translator)
        out = tmp_path / "result.srt"

        rc = cli.main(
            [
                "transcribe", str(wav_path), "-o", str(out),
                "--translate", "google", "--language", "ja", "--target-lang", "en",
            ]
        )

        assert rc == 0
        content = out.read_text(encoding="utf-8")
        assert "EN:こんにちは" in content
        assert "\nこんにちは\n" not in content  # 翻訳 SRT に原文行を出さない
        # 言語ペアを factory (constructor) へ routing (#363: OPUS-MT 対応)
        assert factory_calls["kwargs"]["source_lang"] == "ja"
        assert factory_calls["kwargs"]["target_lang"] == "en"
        # translator lifecycle: load_model / cleanup
        assert translator.load_model_calls == 1
        assert translator.cleanup_calls == 1
        assert translator.translate_calls == [("こんにちは", "ja", "en")]

    def test_translate_total_failure_exits_1_no_file_no_fallback(
        self, wav_path, fake_engine, monkeypatch, tmp_path, capsys
    ):
        translator = FakeTranslator(fail_all=True)
        _patch_translator_factory(monkeypatch, translator)
        out = tmp_path / "result.srt"

        rc = cli.main(
            ["transcribe", str(wav_path), "-o", str(out), "--translate", "google"]
        )

        assert rc == 1
        assert not out.exists()  # 原文 SRT への silent fallback をしない
        captured = capsys.readouterr()
        assert "translation failed for all 1 segments" in captured.err
        assert "こんにちは" not in captured.out
        assert translator.cleanup_calls == 1

    def test_translate_partial_failure_outputs_only_translated(
        self, tmp_path, monkeypatch, capsys
    ):
        wav = tmp_path / "input.wav"
        _write_wav(wav, seconds=3.0)
        engine = FakeEngine()
        _patch_engine_factory(monkeypatch, engine)
        translator = FakeTranslator(fail_on_calls={2})  # 3 segment 中 2 個目だけ失敗
        _patch_translator_factory(monkeypatch, translator)
        _patch_three_segment_pipeline(monkeypatch)
        out = tmp_path / "result.srt"

        rc = cli.main(
            ["transcribe", str(wav), "-o", str(out), "--translate", "google"]
        )

        assert rc == 0
        content = out.read_text(encoding="utf-8")
        blocks = [b for b in content.split("\n\n") if b.strip()]
        assert len(blocks) == 2  # 翻訳成功 segment のみ
        assert blocks[0].startswith("1\n")
        assert blocks[1].startswith("3\n")  # 元 index 維持 (renumber しない)
        captured = capsys.readouterr()
        assert "Warning: translation failed for 1/3 segments" in captured.err


class TestRealtimeOnlyWarnings:
    def test_changed_option_warns(self, wav_path, fake_engine, capsys):
        # --vad は #366 Phase 1 で file mode 対応になったため、realtime-only の
        # 代表として transient filter を使う
        rc = cli.main(["transcribe", str(wav_path), "--transient-filter", "observe"])

        assert rc == 0
        err = capsys.readouterr().err
        assert "realtime-only" in err
        assert "--transient-filter" in err

    def test_defaults_do_not_warn(self, wav_path, fake_engine, capsys):
        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        assert "realtime-only" not in capsys.readouterr().err

    def test_confidence_filter_env_warns(self, wav_path, fake_engine, monkeypatch, capsys):
        monkeypatch.setenv("LIVECAP_CONFIDENCE_FILTER", "observe")

        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        assert "LIVECAP_CONFIDENCE_FILTER" in capsys.readouterr().err

    def test_multiple_changed_options_aggregated(self, wav_path, fake_engine, capsys):
        rc = cli.main(
            [
                "transcribe", str(wav_path),
                "--engine-min-rms", "-50", "--noise-gate",
            ]
        )

        assert rc == 0
        err = capsys.readouterr().err
        assert err.count("realtime-only") == 1  # 1 回の warning に集約
        assert "--engine-min-rms" in err
        assert "--noise-gate" in err


class TestLanguageResolutionE2E:
    """--language の単一解決 (#365): routing / fail-fast / 翻訳併用 / 起動ログ"""

    def test_explicit_language_routed_to_engine(self, wav_path, monkeypatch, capsys):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path), "--language", "en"])

        assert rc == 0
        assert factory_calls["kwargs"]["language"] == "en"
        err = capsys.readouterr().err
        assert "requested=en" in err
        assert "resolved=en" in err

    def test_unspecified_resolves_to_engine_default(self, wav_path, monkeypatch, capsys):
        """未指定 -> whispers2t (default engine) は ja (現状維持)"""
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        assert factory_calls["kwargs"]["language"] == "ja"
        err = capsys.readouterr().err
        assert "requested=(engine default)" in err
        assert "resolved=ja" in err

    def test_bcp47_normalized_before_engine(self, wav_path, monkeypatch, capsys):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path), "--language", "ja-JP"])

        assert rc == 0
        assert factory_calls["kwargs"]["language"] == "ja"

    def test_unsupported_language_fails_before_model_load(
        self, wav_path, monkeypatch, capsys
    ):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path), "--language", "xx"])

        assert rc == 1
        assert factory_calls["count"] == 0  # モデルロード前に fail-fast
        assert "not supported" in capsys.readouterr().err

    def test_malformed_language_fails_friendly(self, wav_path, monkeypatch, capsys):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path), "--language", "notalang!!"])

        assert rc == 1
        assert factory_calls["count"] == 0
        err = capsys.readouterr().err
        assert "Invalid language code" in err
        assert "Traceback" not in err

    def test_auto_rejected_for_whispers2t(self, wav_path, monkeypatch, capsys):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", str(wav_path), "--language", "auto"])

        assert rc == 1
        assert factory_calls["count"] == 0
        assert "does not support automatic" in capsys.readouterr().err

    def test_single_language_engine_mismatch_rejected(
        self, wav_path, monkeypatch, capsys
    ):
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(
            ["transcribe", str(wav_path), "--engine", "reazonspeech", "--language", "en"]
        )

        assert rc == 1
        assert factory_calls["count"] == 0
        assert "not supported by engine 'reazonspeech'" in capsys.readouterr().err

    def test_translate_with_resolved_auto_rejected(self, wav_path, monkeypatch, capsys):
        """voxtral 言語未指定 (resolved=auto) + --translate はモデルロード前に拒否"""
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)
        translator = FakeTranslator()
        _patch_translator_factory(monkeypatch, translator)

        rc = cli.main(
            ["transcribe", str(wav_path), "--engine", "voxtral", "--translate", "google"]
        )

        assert rc == 1
        assert factory_calls["count"] == 0
        assert "concrete source language" in capsys.readouterr().err

    def test_translate_with_engine_default_concrete_allowed(
        self, wav_path, monkeypatch, capsys
    ):
        """default whispers2t (resolved=ja) + --translate は従来どおり許可、
        translator へも resolved 値が一貫して渡る"""
        engine = FakeEngine()
        _patch_engine_factory(monkeypatch, engine)
        translator = FakeTranslator()
        translator_calls = _patch_translator_factory(monkeypatch, translator)

        rc = cli.main(["transcribe", str(wav_path), "--translate", "google"])

        assert rc == 0
        assert translator_calls["kwargs"]["source_lang"] == "ja"  # resolved 値
        assert translator.translate_calls[0][1] == "ja"


class TestVadFileMode:
    """#366 Phase 1: --vad の file mode 接続 / --vad off / no-speech semantics"""

    def test_build_file_segmenter_off_returns_none(self, neutral_file_segmenter):
        """--vad off → None (segmenter 未注入 = 全音声 1 segment)。VAD 構築なし"""
        import argparse

        real_builder = neutral_file_segmenter
        args = argparse.Namespace(vad="off", language="ja", engine="whispers2t")

        assert real_builder(args) is None

    def test_build_file_segmenter_auto_wraps_vad_processor(
        self, neutral_file_segmenter, monkeypatch
    ):
        """--vad auto → _get_vad_processor の結果を VADFileSegmenter に包む"""
        import argparse

        from livecap_cli.vad import VADFileSegmenter

        real_builder = neutral_file_segmenter
        fake_processor = MagicMock()
        captured = {}

        def fake_get_vad_processor(language, vad_backend, engine=None):
            captured.update(language=language, vad=vad_backend, engine=engine)
            return fake_processor

        monkeypatch.setattr(cli, "_get_vad_processor", fake_get_vad_processor)
        args = argparse.Namespace(vad="auto", language="ja", engine="whispers2t")

        segmenter = real_builder(args)

        assert isinstance(segmenter, VADFileSegmenter)
        # resolved language / backend / engine が preset 選択へ渡る (#365 連携)
        assert captured == {"language": "ja", "vad": "auto", "engine": "whispers2t"}

    def test_vad_off_e2e_processes_whole_audio(
        self, wav_path, fake_engine, neutral_file_segmenter, monkeypatch, capsys
    ):
        """--vad off E2E: 実 _build_file_segmenter 経由で従来の 1 segment 出力"""
        monkeypatch.setattr(cli, "_build_file_segmenter", neutral_file_segmenter)

        rc = cli.main(["transcribe", str(wav_path), "--vad", "off"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "00:00:00,000 --> 00:00:01,000" in out  # 全音声 1 segment
        assert "こんにちは" in out

    def test_no_speech_e2e(
        self, wav_path, fake_engine, monkeypatch, tmp_path, capsys
    ):
        """VAD がセグメントなし判定 → exit 0 / 空 SRT / stderr 情報 / ASR 未呼出"""
        monkeypatch.setattr(
            cli, "_build_file_segmenter", lambda args: (lambda a, s: [])
        )
        out_path = tmp_path / "result.srt"

        rc = cli.main(["transcribe", str(wav_path), "-o", str(out_path)])

        assert rc == 0
        captured = capsys.readouterr()
        assert "No speech segments detected." in captured.err
        assert captured.out == ""
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == ""  # 空 SRT
        assert fake_engine.transcribe_calls == 0  # ASR 未呼出

    def test_no_speech_stdout_mode(self, wav_path, fake_engine, monkeypatch, capsys):
        monkeypatch.setattr(
            cli, "_build_file_segmenter", lambda args: (lambda a, s: [])
        )

        rc = cli.main(["transcribe", str(wav_path)])

        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout は空 (SRT 混入なし)
        assert "No speech segments detected." in captured.err

    def test_realtime_vad_off_rejected_before_model_load(self, monkeypatch, capsys):
        """--realtime --vad off はモデルロード前に明確なエラー"""
        engine = FakeEngine()
        factory_calls = _patch_engine_factory(monkeypatch, engine)

        rc = cli.main(["transcribe", "--realtime", "--mic", "0", "--vad", "off"])

        assert rc == 1
        assert "file mode only" in capsys.readouterr().err
        assert factory_calls["count"] == 0  # engine factory 未呼出

    def test_vad_not_in_realtime_only_table(self):
        """--vad は realtime-only ではなくなった (warning 対象外)"""
        assert not any(opt == "--vad" for opt, _, _ in cli._REALTIME_ONLY_OPTIONS)


class TestTranscribeHelp:
    """`transcribe --help` の表示契約 (#363 follow-up)。"""

    def test_help_is_cp932_safe_and_annotates_realtime_only(self):
        """cp932 console (日本語 Windows) で --help が UnicodeEncodeError で
        crash しない (em-dash 等の非 cp932 文字混入の regression 固定) こと、
        realtime-only オプションが help 上で明記されていることを検証。"""
        import io
        import sys

        buf = io.BytesIO()
        wrapper = io.TextIOWrapper(buf, encoding="cp932")
        original_stdout = sys.stdout
        sys.stdout = wrapper
        try:
            with pytest.raises(SystemExit):
                cli.main(["transcribe", "--help"])
            wrapper.flush()
        finally:
            sys.stdout = original_stdout

        help_text = buf.getvalue().decode("cp932")
        assert help_text.count("[realtime only]") == len(cli._REALTIME_ONLY_OPTIONS)
        assert "write SRT to" in help_text  # -o 省略時 stdout の明記


class TestRealtimeOnlyDefaultsSync:
    """`_REALTIME_ONLY_OPTIONS` の default が parser 定義と一致すること。

    parser は `main()` 内 inline のため、`cmd_transcribe` を monkeypatch して
    parse 済み Namespace を捕捉し、表の (attr, default) と突き合わせる。
    parser 側の default 変更時に表の drift を CI で検出する (#363 / #366)。
    """

    def test_table_matches_parser_defaults(self, monkeypatch):
        captured: dict = {}

        def fake_cmd(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(cli, "cmd_transcribe", fake_cmd)
        rc = cli.main(["transcribe", "dummy.wav"])

        assert rc == 0
        args = captured["args"]
        for option, attr, default in cli._REALTIME_ONLY_OPTIONS:
            assert hasattr(args, attr), f"{option}: attr {attr!r} not in parser"
            assert getattr(args, attr) == default, (
                f"{option}: table default {default!r} != parser default "
                f"{getattr(args, attr)!r}"
            )
