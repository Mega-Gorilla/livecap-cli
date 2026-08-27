"""download 境界が ASCII 保証 API に載っていること (Issue #375 PR 3)。

``unicode_safe_download_directory()`` は ``%TEMP%`` を ``cache_root/downloads`` へ
移設するだけで、**その ``cache_root`` はユーザー名を含み得る**ため ASCII 保証が無い
(棚卸し §5.1 で実測)。``unicode_safe`` を名乗る名前自体が「これを使えば ASCII 安全」
という誤読を招くので、呼び出しを :func:`ascii_safe_temp_environment` へ移して削除した。

**source を読んで検査するのはなぜか。** 実際に境界を通すには NeMo / qwen_asr /
huggingface_hub と実モデルが要る (CI にも本 PC にも揃わない)。「helper が消えたか」
だけを見ると、呼び出しを**単に削除しても緑になる** — 移設先が無くなったことに
気付けない。ここでは *どの境界が* *どの API を* *どの ``boundary`` 名で* 使うかまで
固定する。本 repo は `tests/nonascii/test_registry.py::test_callsites_exist` や
resource graph の直接生成禁止検査でも、規約を文章ではなく検査で保っている。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import livecap_cli

_ENGINES = Path(livecap_cli.__file__).parent / "engines"

#: 置換した 5 箇所。engine ファイルごとに期待する ``boundary`` 名。
#: ``engine.qwen3asr.from_pretrained`` は棚卸し表の既存行 (`tests/nonascii/registry.py`)
#: と**同じ文字列**にしてある — 実行時ログと棚卸し行を突き合わせられるようにするため。
EXPECTED_BOUNDARIES: dict[str, set[str]] = {
    "parakeet_engine.py": {"engine.parakeet.from_pretrained"},
    "canary_engine.py": {"engine.canary.from_pretrained"},
    "qwen3asr_engine.py": {"engine.qwen3asr.from_pretrained"},
    "reazonspeech_engine.py": {
        "engine.reazonspeech.download_int8",
        "engine.reazonspeech.download_float32",
    },
}


def _boundaries_in(source: str) -> list[str]:
    """``ascii_safe_temp_environment(...)`` の ``boundary`` 引数を集める。

    正規表現ではなく AST を使うのは、``boundary`` が**キーワード引数として実在する**
    ことまで確かめるため (必須キーワードなので、位置引数で渡すコードは存在しない)。
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "ascii_safe_temp_environment":
            continue
        for keyword in node.keywords:
            if keyword.arg == "boundary" and isinstance(keyword.value, ast.Constant):
                found.append(keyword.value.value)
    return found


class TestOldHelperIsGone:
    def test_importing_it_raises(self):
        """名前ごと消えていること (shim を残さない — pre-1.0 方針)。"""
        with pytest.raises(ImportError):
            from livecap_cli.utils import unicode_safe_download_directory  # noqa: F401

    def test_it_is_not_exported(self):
        utils = importlib.import_module("livecap_cli.utils")
        assert "unicode_safe_download_directory" not in utils.__all__
        assert not hasattr(utils, "unicode_safe_download_directory")

    def test_no_unicode_safe_name_survives_in_the_package(self):
        """**`unicode_safe` を名乗る名前が 1 つも残らない** (#375 の AC)。

        見るのは**識別子**であって文字列ではない — docstring やコメントに
        「この helper は削除した」と履歴を書くのは問題ない (むしろ必要)。
        """
        package_root = Path(livecap_cli.__file__).parent
        offenders: list[str] = []
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Name):
                    names.append(node.id)
                elif isinstance(node, ast.Attribute):
                    names.append(node.attr)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names.extend(alias.asname or alias.name for alias in node.names)
            hits = sorted({name for name in names if "unicode_safe" in name})
            if hits:
                offenders.append(f"{path.relative_to(package_root).as_posix()}: {hits}")
        assert offenders == [], f"unicode_safe を名乗る識別子が残っている: {offenders}"


class TestDownloadBoundariesUseTheAsciiSafeApi:
    @pytest.mark.parametrize("filename", sorted(EXPECTED_BOUNDARIES))
    def test_engine_no_longer_references_the_old_helper(self, filename: str):
        source = (_ENGINES / filename).read_text(encoding="utf-8")
        assert "unicode_safe_download_directory" not in source

    @pytest.mark.parametrize("filename", sorted(EXPECTED_BOUNDARIES))
    def test_engine_names_its_boundary(self, filename: str):
        """**削除ではなく移設**であることを固定する。

        呼び出しを消しただけでも「旧 helper への参照が無い」は通ってしまうので、
        期待する ``boundary`` 名で新 API を使っていることまで見る。
        """
        boundaries = _boundaries_in((_ENGINES / filename).read_text(encoding="utf-8"))
        assert set(boundaries) == EXPECTED_BOUNDARIES[filename]
        # 同じ境界を二重に包んでいないこと
        assert len(boundaries) == len(set(boundaries))

    def test_the_purpose_is_the_documented_slug(self):
        """5 箇所とも ``purpose="download"`` であること。

        purpose は staging root の直下のディレクトリ名になるので、境界ごとに
        バラバラだと reaper と診断ログの読み手が追えなくなる。
        """
        for filename in EXPECTED_BOUNDARIES:
            source = (_ENGINES / filename).read_text(encoding="utf-8")
            purposes = [
                keyword.value.value
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "ascii_safe_temp_environment"
                for keyword in node.keywords
                if keyword.arg == "purpose" and isinstance(keyword.value, ast.Constant)
            ]
            assert purposes == ["download"] * len(EXPECTED_BOUNDARIES[filename]), filename
