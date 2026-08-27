"""NeMo restore 境界が ASCII 保証された `%TEMP%` の中で走ること (Issue #379)。

非 ASCII な `%TEMP%` だと NeMo が内部で `.nemo` を展開した先の SentencePiece model を
読めず、**元例外が抽象クラスの二次例外にすり替わる** (#378 / PR #384 の片側 A/B で確定)。
`.nemo` 自体は wide path で通るので、直すレバーは `%TEMP%` の一時的な ASCII 化だけである。

**probe だけでは足りない。** heavy probe (`tests/nonascii/probes/native_models.py`) は NeMo を
**直接**呼ぶので、**engine 本体が wrapper を呼び忘れても検出できない**。ここでは production の
配線 — `_load_model_from_path()` が temp context の内側で `restore_from()` を呼ぶこと — を
NeMo 抜きで固定する。

期待する `boundary` は**棚卸し registry から導出**する。engine ファイル内に同じ文字列が
2 回現れる (temp context と helper 呼び出し) ので、**両方が registry の `boundary_id` と
一致すること**まで見る。
"""

from __future__ import annotations

import ast
import io
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest

import livecap_cli
from tests.nonascii.registry import BOUNDARIES

_PACKAGE_ROOT = Path(livecap_cli.__file__).parent
_REPO_ROOT = _PACKAGE_ROOT.parent

#: engine ごとの (module 名, class 名, registry の boundary_id, quiet にする logger)
_ENGINES = {
    "parakeet": (
        "livecap_cli.engines.parakeet_engine",
        "ParakeetEngine",
        "engine.parakeet.nemo_restore_from",
    ),
    "canary": (
        "livecap_cli.engines.canary_engine",
        "CanaryEngine",
        "engine.canary.nemo_restore_from",
    ),
}

_ENV_KEYS = ("TEMP", "TMP", "TMPDIR")


def _registry_boundary(boundary_id: str):
    for spec in BOUNDARIES:
        if spec.boundary_id == boundary_id:
            return spec
    raise AssertionError(f"registry に {boundary_id} が無い")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """staging root と cache root を tmp_path 配下へ寄せ、状態を必ず戻す。"""
    from livecap_cli.resources import _reset_resources_for_tests
    from livecap_cli.resources.configuration import clear_staging_roots
    from livecap_cli.paths import roots

    for name in ("ProgramData", "SystemDrive", "PUBLIC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("LIVECAP_CORE_ASCII_STAGING_DIR", raising=False)

    saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
    saved_tempdir = tempfile.tempdir

    _reset_resources_for_tests()
    clear_staging_roots()
    roots.reset_staging_root_cache()
    try:
        yield
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = saved_tempdir
        _reset_resources_for_tests()
        clear_staging_roots()
        roots.reset_staging_root_cache()


class _Recorder:
    """fake ``restore_from`` — 呼ばれた時点の temp 状態を記録する。"""

    def __init__(self, raises: BaseException | None = None, emit: str | None = None):
        self.calls: list[dict] = []
        self.raises = raises
        self.emit = emit

    def __call__(self, *, restore_path: str, map_location: str) -> Any:
        self.calls.append(
            {
                "restore_path": restore_path,
                "map_location": map_location,
                "gettempdir": tempfile.gettempdir(),
                "env_temp": os.environ.get("TEMP"),
            }
        )
        if self.emit is not None:
            logging.getLogger("nemo_logger").error(self.emit)
        if self.raises is not None:
            raise self.raises
        return object()


def _install_fake_nemo(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    """``sys.modules`` へ最小の NeMo スタブを差し込む (実 NeMo は不要)。"""

    class _Model:
        restore_from = staticmethod(recorder)

    models = types.SimpleNamespace(ASRModel=_Model, EncDecMultiTaskModel=_Model)

    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    asr.models = models
    utils = types.ModuleType("nemo.utils")
    utils.logging = logging.getLogger("nemo_logger")
    nemo.collections = collections
    collections.asr = asr
    nemo.utils = utils

    # **本物の NeMo と同じく ``propagate=False`` にする。** これが本 issue の前提で、
    # だからこそ一次エラーが app log に届かず relay が要る。
    monkeypatch.setattr(logging.getLogger("nemo_logger"), "propagate", False)

    for name, module in (
        ("nemo", nemo),
        ("nemo.collections", collections),
        ("nemo.collections.asr", asr),
        ("nemo.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    from livecap_cli.engines import nemo_utils

    monkeypatch.setattr(nemo_utils, "prepare_nemo_environment", lambda: None)
    for module_name, _cls, _bid in _ENGINES.values():
        module = __import__(module_name, fromlist=["_"])
        if hasattr(module, "prepare_nemo_environment"):
            monkeypatch.setattr(module, "prepare_nemo_environment", lambda: None)


def _make_engine(monkeypatch: pytest.MonkeyPatch, engine_key: str):
    module_name, class_name, _bid = _ENGINES[engine_key]
    module = __import__(module_name, fromlist=[class_name])
    engine = getattr(module, class_name)(device="cpu")

    from livecap_cli.engines.model_memory_cache import ModelMemoryCache

    monkeypatch.setattr(ModelMemoryCache, "get", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ModelMemoryCache, "set", staticmethod(lambda *a, **k: None))
    return engine


def _spy_on_temp_env(monkeypatch: pytest.MonkeyPatch, engine_key: str) -> list[dict]:
    """engine module の ``ascii_safe_temp_environment`` を記録付きで包む。"""
    module_name, _cls, _bid = _ENGINES[engine_key]
    module = __import__(module_name, fromlist=["_"])
    seen: list[dict] = []
    original = module.ascii_safe_temp_environment

    def _wrapper(**kwargs):
        seen.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(module, "ascii_safe_temp_environment", _wrapper)
    return seen


def _snapshot() -> dict:
    snap = {key: os.environ.get(key) for key in _ENV_KEYS}
    snap["tempdir"] = tempfile.tempdir
    return snap


class TestRestoreRunsInsideAsciiTempEnvironment:
    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_restore_from_sees_an_ascii_tempdir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        """**本 issue の核心。** ``restore_from()` が ASCII な ``%TEMP%`` の中で走る。"""
        recorder = _Recorder()
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)

        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")
        outer = tempfile.gettempdir()

        engine._load_model_from_path(model_path)

        assert len(recorder.calls) == 1, "restore_from が 1 回だけ呼ばれていない"
        call = recorder.calls[0]
        assert call["restore_path"] == str(model_path), ".nemo は元パスから直接読む"
        assert str(call["gettempdir"]).isascii(), (
            "restore_from が非 ASCII な %TEMP% のまま呼ばれている (#379 の欠陥そのもの)"
        )
        assert call["gettempdir"] != outer, "TEMP が移設されていない"
        assert call["env_temp"] == call["gettempdir"], "env と tempfile.tempdir がずれている"

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_boundary_and_purpose_match_the_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        """engine 別の boundary ID が渡り、registry と一致する。"""
        recorder = _Recorder()
        _install_fake_nemo(monkeypatch, recorder)
        seen = _spy_on_temp_env(monkeypatch, engine_key)
        engine = _make_engine(monkeypatch, engine_key)

        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")
        engine._load_model_from_path(model_path)

        _module, _cls, boundary_id = _ENGINES[engine_key]
        spec = _registry_boundary(boundary_id)
        assert seen == [{"boundary": boundary_id, "purpose": spec.staging_purpose}]
        assert spec.staging_api == "ascii_safe_temp_environment"
        assert spec.staging_purpose, "registry に staging_purpose が無い"

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_environment_is_restored_on_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        recorder = _Recorder()
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        before = _snapshot()
        engine._load_model_from_path(model_path)
        assert _snapshot() == before

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_environment_is_restored_on_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        recorder = _Recorder(raises=RuntimeError("boom"))
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        before = _snapshot()
        with pytest.raises(RuntimeError, match="boom"):
            engine._load_model_from_path(model_path)
        assert _snapshot() == before


class TestDiagnostics:
    """**元例外を殺さず、NeMo の一次エラーを app log へ届ける。**

    NeMo は具象クラス生成中の例外を捕捉して基底クラスへ fallback するので、
    最終例外の ``__cause__`` を辿っても元例外に到達できない。`nemo_logger` は
    ``propagate=False`` + 独自 handler なので windowed build では app log にも届かない。
    """

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_original_exception_is_not_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        sentinel = ValueError("sentinel-original")
        recorder = _Recorder(raises=sentinel)
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        with pytest.raises(ValueError) as excinfo:
            engine._load_model_from_path(model_path)
        assert excinfo.value is sentinel, (
            "元例外が置換されている。置換すると『抽象クラス例外にすり替わる』という "
            "#379 の症状を別の形で作り直すことになる"
        )

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_nemo_error_reaches_the_app_log_with_boundary_and_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        engine_key: str,
    ):
        recorder = _Recorder(
            raises=RuntimeError("secondary abstract-class error"),
            emit="sentencepiece: cannot open tokenizer.model",
        )
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        _module, _cls, boundary_id = _ENGINES[engine_key]
        with caplog.at_level(logging.ERROR, logger="livecap_cli.engines.nemo_utils"):
            with pytest.raises(RuntimeError):
                engine._load_model_from_path(model_path)

        text = "\n".join(
            r.getMessage() for r in caplog.records
            if r.name == "livecap_cli.engines.nemo_utils"
        )
        assert "sentencepiece" in text, "NeMo の一次エラーが app log に届いていない"
        assert boundary_id in text, "boundary が app log に無い"
        assert "model.nemo" in text, "モデルパスが app log に無い"

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_no_relay_when_the_record_already_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        engine_key: str,
    ):
        """**二重出力しない。** ``propagate=True`` なら root が既に受け取っている。

        その状態で転送すると、同じ内容が root へ 2 回出る。relay は
        「そのままでは app log に届かない」ときだけ付ける。
        """
        recorder = _Recorder(raises=RuntimeError("boom"), emit="primary-error-text")
        _install_fake_nemo(monkeypatch, recorder)
        monkeypatch.setattr(logging.getLogger("nemo_logger"), "propagate", True)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError):
                engine._load_model_from_path(model_path)

        relayed = [
            r for r in caplog.records
            if r.name == "livecap_cli.engines.nemo_utils"
            and "primary-error-text" in r.getMessage()
        ]
        assert relayed == [], "propagate=True なのに転送していて二重出力になる"

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_nemo_logger_state_is_restored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        """handler / level / propagate を成功・例外時とも元へ戻す。"""
        recorder = _Recorder(raises=RuntimeError("boom"), emit="primary")
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)
        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")

        watched = ("nemo_logger", "lhotse", "nemo.collections")
        before = {
            name: (
                logging.getLogger(name).level,
                list(logging.getLogger(name).handlers),
                logging.getLogger(name).propagate,
            )
            for name in watched
        }

        with pytest.raises(RuntimeError):
            engine._load_model_from_path(model_path)

        after = {
            name: (
                logging.getLogger(name).level,
                list(logging.getLogger(name).handlers),
                logging.getLogger(name).propagate,
            )
            for name in watched
        }
        assert after == before

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_non_ascii_path_does_not_break_logging(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        """**cp932 strict なストリームでもログ自体が落ちない。**

        日本語 Windows では stderr がリダイレクト時に cp932 + strict になる。
        素のパスを出すと `UnicodeEncodeError` でログが落ちるので `ascii()` で包む。
        """
        model_dir = tmp_path / "ユーザー"
        model_dir.mkdir()
        model_path = model_dir / "model.nemo"
        model_path.write_bytes(b"fake")

        recorder = _Recorder(raises=RuntimeError("boom"), emit="primary")
        _install_fake_nemo(monkeypatch, recorder)
        engine = _make_engine(monkeypatch, engine_key)

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")
        handler = logging.StreamHandler(stream)
        app_logger = logging.getLogger("livecap_cli.engines.nemo_utils")
        app_logger.addHandler(handler)
        saved_level, app_logger.level = app_logger.level, logging.ERROR
        raised: list[BaseException] = []
        handler.handleError = lambda record: raised.append(RuntimeError("log write failed"))
        try:
            with pytest.raises(RuntimeError):
                engine._load_model_from_path(model_path)
        finally:
            app_logger.removeHandler(handler)
            app_logger.level = saved_level

        assert not raised, "非 ASCII パスでログ出力自体が壊れた"


class TestSharedHelperIsUsed:
    """共通 helper を通ること — TEMP / logger 管理を engine へ複製しない (#379 要件 7)。"""

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_engine_calls_the_shared_restore_helper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine_key: str
    ):
        from livecap_cli.engines import nemo_utils

        recorder = _Recorder()
        _install_fake_nemo(monkeypatch, recorder)

        module_name, _cls, _bid = _ENGINES[engine_key]
        module = __import__(module_name, fromlist=["_"])
        seen: list[dict] = []
        original = nemo_utils.restore_nemo_model

        def _wrapper(model_class, model_path, **kwargs):
            seen.append(dict(kwargs))
            return original(model_class, model_path, **kwargs)

        monkeypatch.setattr(module, "restore_nemo_model", _wrapper)
        engine = _make_engine(monkeypatch, engine_key)

        model_path = tmp_path / "model.nemo"
        model_path.write_bytes(b"fake")
        engine._load_model_from_path(model_path)

        assert len(seen) == 1, "共通 helper を通っていない"
        assert seen[0]["boundary"] == _ENGINES[engine_key][2]

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_both_boundary_literals_match_the_registry(self, engine_key: str):
        """engine ファイル内の boundary 文字列が **2 箇所とも** registry と一致する。

        temp context と helper 呼び出しで同じ文字列を書くので、片方だけ直すと
        ログと棚卸しが静かにずれる。SSOT は registry である。
        """
        module_name, _cls, boundary_id = _ENGINES[engine_key]
        spec = _registry_boundary(boundary_id)
        source = (_REPO_ROOT / spec.callsite_file).read_text(encoding="utf-8")

        literals: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in ("ascii_safe_temp_environment", "restore_nemo_model"):
                continue
            for keyword in node.keywords:
                if keyword.arg == "boundary" and isinstance(keyword.value, ast.Constant):
                    literals.append(keyword.value.value)

        assert boundary_id in literals, f"{spec.callsite_file} に {boundary_id} が無い"
        restore_literals = [x for x in literals if x == boundary_id]
        assert len(restore_literals) == 2, (
            "temp context と helper 呼び出しの両方に同じ boundary リテラルが要る "
            f"(見つかった: {literals})"
        )

    @pytest.mark.parametrize("engine_key", sorted(_ENGINES))
    def test_unused_nemo_logging_import_is_gone(self, engine_key: str):
        """未使用の ``from nemo.utils import logging as nemo_logging`` を残さない。"""
        module_name, _cls, boundary_id = _ENGINES[engine_key]
        spec = _registry_boundary(boundary_id)
        source = (_REPO_ROOT / spec.callsite_file).read_text(encoding="utf-8")
        assert "nemo_logging" not in source
