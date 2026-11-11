"""``whisper_s2t`` は import の前に ffmpeg を PATH へ反映してから読む (#474)。

``whisper_s2t`` は import 時に ``os.system('ffmpeg -version')`` で PATH 上の ffmpeg を確かめ、確認は
最初の import の 1 回だけ。ffmpeg を PATH 以外 (``LIVECAP_FFMPEG_BIN`` など) で用意するホストでは、
import する前に PATH へ反映しないと確認が必ず失敗する。
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import livecap_cli
from livecap_cli.engines import whisper_s2t_import as helper

PACKAGE_ROOT = Path(livecap_cli.__file__).resolve().parent


class TestImportWhisperS2T:
    def test_exposes_ffmpeg_before_the_first_import(self, monkeypatch):
        calls: list = []
        fake = ModuleType("whisper_s2t")
        monkeypatch.delitem(sys.modules, "whisper_s2t", raising=False)
        monkeypatch.setattr(helper, "expose_ffmpeg_for_whisper_s2t", lambda: calls.append("expose"))
        monkeypatch.setattr(helper.importlib, "import_module", lambda name: calls.append(("import", name)) or fake)

        assert helper.import_whisper_s2t() is fake
        assert calls == ["expose", ("import", "whisper_s2t")]

    def test_an_imported_module_is_returned_as_is(self, monkeypatch):
        """import 済みなら確認は終わっている。反映し直さない。"""
        fake = ModuleType("whisper_s2t")
        monkeypatch.setitem(sys.modules, "whisper_s2t", fake)
        monkeypatch.setattr(helper, "expose_ffmpeg_for_whisper_s2t", lambda: pytest.fail("反映し直してはいけない"))

        assert helper.import_whisper_s2t() is fake

    def test_import_error_propagates(self, monkeypatch):
        """未導入は呼び出し側 (``_check_dependencies``) が案内付きの ImportError にする。"""
        monkeypatch.delitem(sys.modules, "whisper_s2t", raising=False)
        monkeypatch.setattr(helper, "expose_ffmpeg_for_whisper_s2t", lambda: None)

        def missing(name):
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(helper.importlib, "import_module", missing)
        with pytest.raises(ImportError):
            helper.import_whisper_s2t()


class TestExposeFFmpeg:
    def _manager(self, monkeypatch, expose_on_path):
        import livecap_cli.resources as resources

        monkeypatch.setattr(resources, "get_ffmpeg_manager", lambda: SimpleNamespace(expose_on_path=expose_on_path))

    def test_found_ffmpeg_is_exposed_quietly(self, monkeypatch, caplog):
        calls = []
        self._manager(monkeypatch, lambda: calls.append("expose") or Path("ffmpeg.exe"))

        with caplog.at_level(logging.DEBUG, logger=helper.__name__):
            helper.expose_ffmpeg_for_whisper_s2t()

        assert calls == ["expose"]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_missing_ffmpeg_only_warns(self, monkeypatch, caplog):
        """realtime の WhisperS2T は ffmpeg 無しで動くので、ロードは止めない。"""
        self._manager(monkeypatch, lambda: None)

        with caplog.at_level(logging.WARNING, logger=helper.__name__):
            helper.expose_ffmpeg_for_whisper_s2t()

        assert "FFmpeg が見つからない" in caplog.text

    def test_failure_to_expose_only_warns(self, monkeypatch, caplog):
        def broken():
            raise RuntimeError("resource graph is broken")

        self._manager(monkeypatch, broken)

        with caplog.at_level(logging.WARNING, logger=helper.__name__):
            helper.expose_ffmpeg_for_whisper_s2t()

        assert "resource graph is broken" in caplog.text


class TestCallSites:
    def test_every_whisper_s2t_import_goes_through_the_helper(self):
        """確認は最初の 1 回だけなので、1 か所でも直に import すると先を越されうる。"""
        offenders = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if path.name == "whisper_s2t_import.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                if any(name == "whisper_s2t" or name.startswith("whisper_s2t.") for name in names):
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{node.lineno}")
        assert not offenders, f"whisper_s2t を直に import している: {offenders}"

    def test_check_dependencies_uses_the_helper(self, monkeypatch):
        from livecap_cli.engines import whispers2t_engine
        from livecap_cli.engines.library_preloader import LibraryPreloader

        calls = []
        monkeypatch.setattr(whispers2t_engine, "import_whisper_s2t", lambda: calls.append("helper"))
        monkeypatch.setattr(LibraryPreloader, "wait_for_preload", staticmethod(lambda timeout=None: True))
        engine = SimpleNamespace(report_progress=lambda *args, **kwargs: None, device="cpu")

        whispers2t_engine.WhisperS2TEngine._check_dependencies(engine)

        assert calls == ["helper"]

    def test_preloader_uses_the_helper(self, monkeypatch):
        """preloader の別スレッドが engine のロードより先に import しうる。"""
        from livecap_cli.engines.library_preloader import LibraryPreloader

        calls = []
        monkeypatch.setattr(helper, "import_whisper_s2t", lambda: calls.append("helper"))
        monkeypatch.setitem(LibraryPreloader._preloaded, "whisper_s2t", False)

        LibraryPreloader._preload_whispers2t()

        assert calls == ["helper"]
        assert LibraryPreloader._preloaded["whisper_s2t"] is True
