"""発話ごとの一時 wav の寿命を engine ごとに固定する (Issue #419)。

対象は一時 wav を書く 4 engine (Parakeet / Canary / WhisperS2T / Voxtral)。Qwen3-ASR は
#492 以降、配列をそのまま processor へ渡すので対象外。

**ネイティブ API (consumer) は stub し、実際に渡された path を consumer の中で見る。**
「path が正しい」だけでは寿命を保証できないので、consumer 呼び出しの**最中に**ファイルが
存在して中身が音声であること、**抜けた後に**消えていることを確かめる。
モデルのロード・GPU は使わない。

置き場所は engine ごとに違い、変えていない (#419 の非スコープ):

- Parakeet / Canary: ``dir=`` 指定なし → ``tempfile.gettempdir()``
- WhisperS2T: ``self._tmp_dir`` (``get_temp_dir("whispers2t")``)
- Voxtral: ``get_temp_dir()``

テストでは各置き場所を ``tmp_path`` へ向け、終了後に残った ``*.wav`` を数える。
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from livecap_cli.engines import (
    canary_engine,
    parakeet_engine,
    voxtral_engine,
    whispers2t_engine,
)
from livecap_cli.engines.temp_wav import discard_temp_wav

SAMPLE_RATE = 16000
# 1 秒の正弦波。短すぎチェック (0.1 秒) を越え、正規化 (max > 1.0) にも掛からない。
AUDIO = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)).astype(
    np.float32
)


# --- engine ごとの組み立て -------------------------------------------------------


def _build_parakeet(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # dir= 指定なし → tempfile.gettempdir() に落ちる
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    engine = parakeet_engine.ParakeetEngine.__new__(parakeet_engine.ParakeetEngine)
    engine.engine_name = "parakeet"
    engine.model_name = "nvidia/parakeet-tdt-0.6b-v3"
    engine.decoding_strategy = "greedy"
    engine.device = "cpu"
    engine.torch_device = "cpu"
    engine._initialized = True
    engine.model = MagicMock()
    return engine, engine.model.transcribe


def _build_canary(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    engine = canary_engine.CanaryEngine.__new__(canary_engine.CanaryEngine)
    engine.language = "en"
    engine._initialized = True
    engine.model = MagicMock()
    return engine, engine.model.transcribe


def _build_whispers2t(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    engine = whispers2t_engine.WhisperS2TEngine.__new__(whispers2t_engine.WhisperS2TEngine)
    engine._tmp_dir = temp_dir
    engine._enable_profiling = False
    engine._asr_language = "en"
    engine.use_vad = False
    engine.batch_size = 1
    engine._initialized = True
    engine.model = MagicMock()
    return engine, engine.model.transcribe


def _build_voxtral(temp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(voxtral_engine, "get_temp_dir", lambda *a, **k: temp_dir)
    engine = voxtral_engine.VoxtralEngine.__new__(voxtral_engine.VoxtralEngine)
    engine._initialized = True
    engine.model = MagicMock()
    engine.processor = MagicMock()
    engine.torch_device = "cpu"
    engine.language = "en"
    engine._asr_language = "en"
    engine.model_name = "test-model"
    engine.do_sample = False
    engine.max_new_tokens = 448
    predicted_ids = MagicMock()
    predicted_ids.__getitem__ = MagicMock(return_value=predicted_ids)
    engine.model.generate.return_value = predicted_ids
    engine.processor.batch_decode.return_value = ["ok"]
    # wav を読むのは apply_transcription_request (以降は tensor だけを使う)
    return engine, engine.processor.apply_transcription_request


@dataclass(frozen=True)
class EngineCase:
    name: str
    module: Any
    build: Callable[[Path, pytest.MonkeyPatch], tuple[Any, MagicMock]]
    #: consumer の呼び出し引数から、渡された wav の path を取り出す
    path_from: Callable[[tuple, dict], str]
    #: consumer が正常に返す値 (engine はこれを "ok" という転写にする)
    ok_output: Callable[[], Any]


CASES = [
    EngineCase(
        "parakeet",
        parakeet_engine,
        _build_parakeet,
        lambda args, kwargs: kwargs["audio"][0],
        lambda: ["ok"],
    ),
    EngineCase(
        "canary",
        canary_engine,
        _build_canary,
        lambda args, kwargs: kwargs["audio"][0],
        lambda: [SimpleNamespace(text="ok")],
    ),
    EngineCase(
        "whispers2t",
        whispers2t_engine,
        _build_whispers2t,
        lambda args, kwargs: args[0][0],
        lambda: [{"text": "ok"}],
    ),
    EngineCase(
        "voxtral",
        voxtral_engine,
        _build_voxtral,
        lambda args, kwargs: kwargs["audio"],
        MagicMock,
    ),
]


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def case(request) -> EngineCase:
    return request.param


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "utterance-wav"
    d.mkdir()
    return d


def _install_consumer(case: EngineCase, consumer: MagicMock, body: Callable[[str], None]):
    """consumer を stub する。``body(path)`` を consumer 呼び出しの**最中に**実行する。"""

    def side_effect(*args, **kwargs):
        body(str(case.path_from(args, kwargs)))
        return case.ok_output()

    consumer.side_effect = side_effect


def _leftover_wavs(temp_dir: Path) -> list[str]:
    return sorted(p.name for p in temp_dir.glob("*.wav"))


def _fail_unlink_for(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``temp_dir`` 内の wav の削除だけを ``PermissionError`` にする (Windows のロック相当)。"""
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        p = Path(path)
        if p.suffix == ".wav" and p.parent == temp_dir:
            raise PermissionError(13, "locked by another process", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)


# --- 寿命 -----------------------------------------------------------------------


class TestLifetime:
    def test_file_holds_the_audio_while_consumer_runs_and_is_removed_after(
        self, case: EngineCase, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """consumer 呼び出し中は存在して中身が音声、正常終了後は消えている。

        **早すぎる削除の回帰ガード**でもある。置き場所が変わっていないことも見る。
        """
        engine, consumer = case.build(temp_dir, monkeypatch)
        seen: dict[str, Any] = {}

        def body(path: str) -> None:
            seen["path"] = path
            seen["exists"] = os.path.exists(path)
            info = sf.info(path)
            seen["samplerate"] = info.samplerate
            seen["frames"] = info.frames

        _install_consumer(case, consumer, body)

        result = engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert result.text == "ok"
        assert consumer.call_count == 1
        assert seen["exists"] is True
        assert Path(seen["path"]).parent == temp_dir
        assert seen["samplerate"] == SAMPLE_RATE
        assert seen["frames"] == len(AUDIO)
        assert not os.path.exists(seen["path"])
        assert _leftover_wavs(temp_dir) == []

    def test_removed_when_write_fails(
        self, case: EngineCase, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """**書き込み例外でも残らない** (#419 の主症状)。

        書きかけの中身を残してから失敗させる (ディスク逼迫の想定)。``NamedTemporaryFile``
        は名前を確保した時点で空ファイルを作るので、書き込みが cleanup の ``try`` の外に
        あると、ここで 1 ファイル残っていた。
        """
        engine, consumer = case.build(temp_dir, monkeypatch)

        def failing_write(path, *args, **kwargs):
            Path(path).write_bytes(b"RIFF-partial")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(case.module.sf, "write", failing_write)

        with pytest.raises(OSError, match="No space left on device"):
            engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert consumer.call_count == 0
        assert _leftover_wavs(temp_dir) == []

    def test_removed_when_consumer_raises(
        self, case: EngineCase, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """consumer の例外でも残らず、**その例外がそのまま**呼び出し側へ届く。"""
        engine, consumer = case.build(temp_dir, monkeypatch)
        boom = RuntimeError("consumer failed")
        seen: dict[str, Any] = {}

        def body(path: str) -> None:
            seen["path"] = path
            seen["exists"] = os.path.exists(path)
            raise boom

        _install_consumer(case, consumer, body)

        with pytest.raises(RuntimeError) as excinfo:
            engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert excinfo.value is boom
        assert seen["exists"] is True
        assert not os.path.exists(seen["path"])
        assert _leftover_wavs(temp_dir) == []


class TestCleanupFailure:
    def test_does_not_mask_the_consumer_error(
        self,
        case: EngineCase,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """削除に失敗しても、**本来の例外 (consumer の失敗) を上書きしない**。

        以前は ``finally`` の ``unlink`` の ``PermissionError`` が consumer の例外に置き換わり、
        呼び出し側に本当の失敗原因が届かなかった。消せなかった事実は WARNING に残す。
        """
        engine, consumer = case.build(temp_dir, monkeypatch)
        boom = RuntimeError("consumer failed")
        seen: dict[str, str] = {}

        def body(path: str) -> None:
            seen["path"] = path
            raise boom

        _install_consumer(case, consumer, body)
        _fail_unlink_for(temp_dir, monkeypatch)

        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.temp_wav"):
            with pytest.raises(RuntimeError) as excinfo:
                engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert excinfo.value is boom
        assert any(
            r.name == "livecap_cli.engines.temp_wav" and seen["path"] in r.getMessage()
            for r in caplog.records
        )

    def test_does_not_discard_a_successful_result(
        self,
        case: EngineCase,
        temp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """正常終了で削除だけ失敗したら、転写結果を返して WARNING を残す。"""
        engine, consumer = case.build(temp_dir, monkeypatch)
        _install_consumer(case, consumer, lambda path: None)
        _fail_unlink_for(temp_dir, monkeypatch)

        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.temp_wav"):
            result = engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert result.text == "ok"
        assert any(r.name == "livecap_cli.engines.temp_wav" for r in caplog.records)


class TestConcurrentCalls:
    def test_overlapping_call_neither_shares_nor_removes_the_other_file(
        self, case: EngineCase, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """呼び出し A の consumer 実行中に、同じ置き場所で B が始まって終わる。

        B の寿命を A の consumer の窓の中に丸ごと入れることで、スレッドを使わずに
        並行呼び出しの競合を決定的に再現する。**B は A と別のファイルを使い、B の cleanup は
        A の入力を消さない** (Voxtral の pid 固定名では両方とも崩れていた)。
        """
        outer_engine, outer_consumer = case.build(temp_dir, monkeypatch)
        inner_engine, inner_consumer = case.build(temp_dir, monkeypatch)
        seen: dict[str, Any] = {}

        _install_consumer(case, inner_consumer, lambda path: seen.setdefault("inner", path))

        def outer_body(path: str) -> None:
            seen["outer"] = path
            inner_result = inner_engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)
            seen["inner_text"] = inner_result.text
            seen["outer_exists_after_inner"] = os.path.exists(path)
            seen["inner_exists_after_inner"] = os.path.exists(seen["inner"])

        _install_consumer(case, outer_consumer, outer_body)

        result = outer_engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

        assert result.text == "ok"
        assert seen["inner_text"] == "ok"
        assert seen["inner"] != seen["outer"]
        assert seen["outer_exists_after_inner"] is True
        assert seen["inner_exists_after_inner"] is False
        assert _leftover_wavs(temp_dir) == []


def test_voxtral_names_are_unique_per_call_in_one_process(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Voxtral: 同一プロセスの連続呼び出しが同じ名前を使わない (#419)。

    以前は ``voxtral_temp_<pid>.wav`` 固定だった。置き場所 (``get_temp_dir()``) と
    ``voxtral_temp_`` 接頭辞は変えていない (非 ASCII path の棚卸し表の callsite でもある)。
    """
    case = next(c for c in CASES if c.name == "voxtral")
    engine, consumer = case.build(temp_dir, monkeypatch)
    paths: list[str] = []
    _install_consumer(case, consumer, paths.append)

    engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)
    engine._transcribe_single_chunk(AUDIO, SAMPLE_RATE)

    assert len(paths) == 2
    assert paths[0] != paths[1]
    for p in map(Path, paths):
        assert p.parent == temp_dir
        assert p.name.startswith("voxtral_temp_")
        assert p.suffix == ".wav"
    assert _leftover_wavs(temp_dir) == []


# --- discard_temp_wav ------------------------------------------------------------


class TestDiscardTempWav:
    def test_removes_the_file(self, tmp_path: Path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"x")
        discard_temp_wav(p)
        assert not p.exists()

    def test_missing_file_is_silent(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.temp_wav"):
            discard_temp_wav(tmp_path / "missing.wav")
        assert [r for r in caplog.records if r.name == "livecap_cli.engines.temp_wav"] == []

    def test_os_error_is_logged_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        p = tmp_path / "locked.wav"
        p.write_bytes(b"x")
        _fail_unlink_for(tmp_path, monkeypatch)

        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.temp_wav"):
            discard_temp_wav(p)

        assert p.exists()
        records = [r for r in caplog.records if r.name == "livecap_cli.engines.temp_wav"]
        assert [r.levelno for r in records] == [logging.WARNING]
        assert str(p) in records[0].getMessage()
