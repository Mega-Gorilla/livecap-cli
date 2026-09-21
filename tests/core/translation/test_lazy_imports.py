"""Google だけを使うときに torch / transformers / ctranslate2 を読み込まないこと (Issue #454)。

``translation/impl/__init__.py`` が OPUS-MT / Riva を eager import していたため、
``TranslatorFactory.create_translator("google")`` で torch の読み込みが始まり、別スレッドの
``scipy.signal.resample_poly`` が初期化途中の ``sys.modules["torch"]`` に当たって落ちていた
(livecap-gui の実機ログ)。

**判定は subprocess で行う** — 同一プロセスだと先行テストの import が残り、緑のまま何も測らない。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import patch

import pytest

from livecap_cli.translation.factory import TranslatorFactory

_HEAVY = ("torch", "transformers", "ctranslate2", "livecap_cli.translation.impl.opus_mt", "livecap_cli.translation.impl.riva_instruct")


def _loaded_after(code: str) -> dict[str, bool]:
    script = textwrap.dedent(code) + textwrap.dedent(
        f"""
        import json, sys
        print("RESULT " + json.dumps({{m: m in sys.modules for m in {list(_HEAVY)!r}}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    import json

    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


class TestGoogleOnlyDoesNotLoadTorch:
    def test_importing_the_impl_package_loads_nothing_heavy(self):
        loaded = _loaded_after("import livecap_cli.translation.impl")
        assert not any(loaded.values()), f"impl/__init__ が重い module を eager import している: {loaded}"

    def test_creating_the_google_translator_loads_nothing_heavy(self):
        loaded = _loaded_after(
            """
            from livecap_cli.translation import TranslatorFactory
            t = TranslatorFactory.create_translator("google", source_lang="ja", target_lang="en")
            assert t.get_translator_name() == "google"
            """
        )
        assert not any(loaded.values()), f"Google だけなのに読み込まれた: {loaded}"

    def test_resample_in_another_thread_survives_google_translator_creation(self):
        """#454 の再現手順: スレッド A が create_translator("google")、スレッド B が resample_poly。
        eager import が無ければ torch の import 自体が起きないので、どのタイミングでも落ちない。
        B は stop 後も最低 5 回は resample を完了させる (1 度も走らずに通る形にしない)。回帰の検出は
        「B が落ちない」と「生成後も torch が sys.modules に無い」の両方で行う (後者は決定的)。"""
        script = """
            import threading, sys
            import numpy as np
            from scipy.signal import resample_poly
            errors = []
            stop = threading.Event()
            count = [0]
            def audio():
                x = np.zeros(48000, dtype=np.float32)
                while True:
                    try:
                        resample_poly(x, 1, 3)
                    except Exception as e:
                        errors.append(f"{type(e).__name__}: {e}"); break
                    count[0] += 1
                    if stop.is_set() and count[0] >= 5:
                        break
            def translator():
                from livecap_cli.translation import TranslatorFactory
                TranslatorFactory.create_translator("google", source_lang="ja", target_lang="en")
            b = threading.Thread(target=audio); a = threading.Thread(target=translator)
            b.start(); a.start(); a.join(); stop.set(); b.join()
            assert errors == [], errors
            assert count[0] >= 5, count
            assert "torch" not in sys.modules
            """
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]


class TestMissingExtraIsReportedAsImportError:
    """実装 module はあるが依存 (extra) が無い → 「未実装」ではなく ImportError + extra 名。"""

    def _missing(self, missing_name: str):
        real = __import__("importlib").import_module

        def fake_import(name, package=None):
            if name in (".impl.opus_mt", ".impl.riva_instruct"):
                raise ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
            return real(name, package)

        return patch("livecap_cli.translation.factory.importlib.import_module", fake_import)

    def test_opus_mt_names_translation_local(self):
        with self._missing("ctranslate2"):
            with pytest.raises(ImportError, match=r"translation-local.*ctranslate2") as info:
                TranslatorFactory.create_translator("opus_mt")
        assert not isinstance(info.value, NotImplementedError)

    def test_riva_names_translation_riva(self):
        with self._missing("torch"):
            with pytest.raises(ImportError, match=r"translation-riva.*torch"):
                TranslatorFactory.create_translator("riva_instruct")

    def test_missing_implementation_module_is_still_not_implemented(self):
        with self._missing("livecap_cli.translation.impl.opus_mt"):
            with pytest.raises(NotImplementedError, match="not yet implemented") as info:
                TranslatorFactory.create_translator("opus_mt")
        assert "riva_instruct" in str(info.value), "案内の一覧は metadata から (他 module を import しない)"

    def test_missing_submodule_of_a_declared_dependency_is_not_disguised_as_a_missing_extra(self):
        """`transformers` 自体はあるが `transformers.some_internal` が無い = 実装 / 配布物の不整合。
        extra の再インストール案内で隠さず、生の ModuleNotFoundError を送出する。"""
        with self._missing("transformers.some_internal"):
            with pytest.raises(ModuleNotFoundError, match="transformers.some_internal") as info:
                TranslatorFactory.create_translator("riva_instruct")
        assert "translation-riva" not in str(info.value)

    def test_unexpected_missing_module_is_not_disguised_as_a_missing_extra(self):
        """宣言していない module の ModuleNotFoundError (実装内部の typo / 欠落) は原因を隠さず素通し。"""
        with self._missing("some_typo_dependency"):
            with pytest.raises(ModuleNotFoundError, match="some_typo_dependency") as info:
                TranslatorFactory.create_translator("opus_mt")
        assert "translation-local" not in str(info.value)

    def test_error_paths_do_not_import_other_translators(self):
        """未実装エラーの案内文を作るときに Riva / OPUS-MT を import しない (subprocess で判定)。"""
        loaded = _loaded_after(
            """
            from unittest.mock import patch
            import importlib
            from livecap_cli.translation import TranslatorFactory
            real = importlib.import_module
            def fake(name, package=None):
                if name == ".impl.opus_mt":
                    raise ModuleNotFoundError("No module named 'livecap_cli.translation.impl.opus_mt'", name="livecap_cli.translation.impl.opus_mt")
                return real(name, package)
            with patch("livecap_cli.translation.factory.importlib.import_module", fake):
                try:
                    TranslatorFactory.create_translator("opus_mt")
                except NotImplementedError:
                    pass
                else:
                    raise AssertionError("NotImplementedError expected")
            """
        )
        assert not loaded["livecap_cli.translation.impl.riva_instruct"] and not loaded["torch"], loaded

    def test_google_is_created_without_touching_the_fake(self):
        with self._missing("torch"):
            assert TranslatorFactory.create_translator("google").get_translator_name() == "google"
