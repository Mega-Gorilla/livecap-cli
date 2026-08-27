"""download 境界が ASCII 保証 API に載っていること (Issue #375 PR 3)。

``unicode_safe_download_directory()`` は ``%TEMP%`` を ``cache_root/downloads`` へ
移設するだけで、**その ``cache_root`` はユーザー名を含み得る**ため ASCII 保証が無い
(棚卸し §5.1 で実測)。``unicode_safe`` を名乗る名前自体が「これを使えば ASCII 安全」
という誤読を招くので、呼び出しを :func:`ascii_safe_temp_environment` へ移して削除した。

**期待値は棚卸し registry から導出する。** 境界一覧をこのファイルにも書くと SSOT が
2 箇所に分裂し、実行時ログの ``boundary=`` を棚卸し行へ突合できなくなる
(PR 3 のレビュー指摘 2)。``BoundarySpec.staging_api`` を持つ行が「production code が
ASCII staging で包んでいる境界」の唯一の一覧であり、``boundary_id`` がそのまま
実行時の ``boundary=`` 文字列になる。

**source を読んで検査するのはなぜか。** 実際に境界を通すには NeMo / qwen_asr と実
モデル・ネットワークが要る (CI にも本 PC にも揃わない)。「helper が消えたか」だけを
見ると、呼び出しを**単に削除しても緑になる** — 移設先が無くなったことに気付けない。
ここでは *どの境界が* *どの API を* *どの ``boundary`` 名で* 使うかまで固定する。
"""

from __future__ import annotations

import ast
import importlib
from collections import defaultdict
from pathlib import Path

import pytest

import livecap_cli
from tests.nonascii.registry import BOUNDARIES

_PACKAGE_ROOT = Path(livecap_cli.__file__).parent
_REPO_ROOT = _PACKAGE_ROOT.parent

_TEMP_ENV_API = "ascii_safe_temp_environment"

#: staging API で包んだ境界を **registry から導出**する。
#: {repo 相対の callsite_file: {boundary_id, ...}}
EXPECTED_BOUNDARIES: dict[str, set[str]] = defaultdict(set)
for _spec in BOUNDARIES:
    if _spec.staging_api == _TEMP_ENV_API:
        EXPECTED_BOUNDARIES[_spec.callsite_file].add(_spec.boundary_id)
EXPECTED_BOUNDARIES = dict(EXPECTED_BOUNDARIES)

#: 旧 helper が包んでいたが、**意図的に包み直さなかった** callsite。
#: 書き込み先をすべて明示するので ``%TEMP%`` を消費せず、棚卸しでも ②wide-path が
#: 実測で確定している。包むと ASCII root を確保できない環境で正常系を新たに失敗させる。
DELIBERATELY_UNWRAPPED = ("reazonspeech_engine.py",)


def _calls_to(source: str, func_name: str) -> list[ast.Call]:
    """``func_name`` への呼び出しノードを集める。"""
    found: list[ast.Call] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == func_name:
            found.append(node)
    return found


def _kwarg(call: ast.Call, name: str):
    """キーワード引数の定数値を返す (無ければ ``None``)。

    正規表現ではなく AST を使うのは、``boundary`` / ``purpose`` が**キーワード引数として
    実在する**ことまで確かめるため。
    """
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _source_of(callsite_file: str) -> str:
    return (_REPO_ROOT / callsite_file).read_text(encoding="utf-8")


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
        offenders: list[str] = []
        for path in _PACKAGE_ROOT.rglob("*.py"):
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
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT).as_posix()}: {hits}")
        assert offenders == [], f"unicode_safe を名乗る識別子が残っている: {offenders}"


class TestRegistryIsTheSingleList:
    def test_registry_lists_the_wrapped_boundaries(self):
        """registry が空でないこと (導出元が壊れたら以降が空振りする)。"""
        assert EXPECTED_BOUNDARIES, (
            "staging_api を持つ BoundarySpec が 1 行も無い。registry からの導出が"
            "壊れているので、以降のテストは何も検査していない。"
        )

    @pytest.mark.parametrize("callsite_file", sorted(EXPECTED_BOUNDARIES))
    def test_callsite_file_exists(self, callsite_file: str):
        assert (_REPO_ROOT / callsite_file).is_file(), (
            f"{callsite_file} が無い。registry の callsite_file を更新すること。"
        )


class TestDownloadBoundariesUseTheAsciiSafeApi:
    @pytest.mark.parametrize("callsite_file", sorted(EXPECTED_BOUNDARIES))
    def test_callsite_no_longer_references_the_old_helper(self, callsite_file: str):
        assert "unicode_safe_download_directory" not in _source_of(callsite_file)

    @pytest.mark.parametrize("callsite_file", sorted(EXPECTED_BOUNDARIES))
    def test_callsite_names_its_boundary(self, callsite_file: str):
        """**削除ではなく移設**であることを固定する。

        呼び出しを消しただけでも「旧 helper への参照が無い」は通ってしまうので、
        registry が言う ``boundary`` 名で新 API を使っていることまで見る。
        """
        calls = _calls_to(_source_of(callsite_file), _TEMP_ENV_API)
        boundaries = [_kwarg(call, "boundary") for call in calls]
        assert set(boundaries) == EXPECTED_BOUNDARIES[callsite_file]
        # 同じ境界を二重に包んでいないこと
        assert len(boundaries) == len(set(boundaries))

    @pytest.mark.parametrize("callsite_file", sorted(EXPECTED_BOUNDARIES))
    def test_the_purpose_is_the_documented_slug(self, callsite_file: str):
        """``purpose="download"`` であること。

        purpose は staging root の直下のディレクトリ名になるので、境界ごとに
        バラバラだと reaper と診断ログの読み手が追えなくなる。
        """
        calls = _calls_to(_source_of(callsite_file), _TEMP_ENV_API)
        purposes = [_kwarg(call, "purpose") for call in calls]
        assert purposes == ["download"] * len(EXPECTED_BOUNDARIES[callsite_file])


class TestDeliberatelyUnwrapped:
    """**②wide-path が実測で確定している経路を ③staging に格上げしない。**

    ReazonSpeech の 2 経路は書き込み先をすべて明示する — ``download_file()`` は
    ``cache_root/downloads`` へ直接、``temporary_directory()`` は ``dir=``、
    ``snapshot_download`` は ``cache_dir=``。したがって ``%TEMP%`` を消費しない。
    実測でも 713 MB のダウンロード中に移設先へ落ちたファイルは 0 件だった。

    包むと **ASCII staging root を確保できない環境で、本来動くダウンロードが
    ``AsciiStagingUnavailableError`` になる**。旧 helper がそこに居たことは、
    包み直す理由にならない (pre-1.0 方針)。
    """

    @pytest.mark.parametrize("filename", DELIBERATELY_UNWRAPPED)
    def test_neither_the_old_helper_nor_a_new_wrapper(self, filename: str):
        source = (_PACKAGE_ROOT / "engines" / filename).read_text(encoding="utf-8")
        assert "unicode_safe_download_directory" not in source
        assert _TEMP_ENV_API not in source, (
            f"{filename} を ASCII staging で包み直している。②wide-path が実測で"
            f"確定している経路を ③staging に格上げすると、正常系を新たに失敗させる。"
        )

    @pytest.mark.parametrize("filename", DELIBERATELY_UNWRAPPED)
    def test_registry_does_not_claim_it_is_wrapped(self, filename: str):
        """registry 側にも「包んでいる」と書かれていないこと (両者の食い違い防止)。"""
        claimed = [
            spec.boundary_id
            for spec in BOUNDARIES
            if spec.staging_api and spec.callsite_file.endswith(filename)
        ]
        assert claimed == [], f"registry が包んでいると主張している: {claimed}"
