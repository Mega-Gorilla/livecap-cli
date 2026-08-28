"""extra 間で依存の上限がずれないこと (Issue #379)。

**上限 (`<`) は「このバージョン以上は壊れる」と分かったから付けたもの**であり、
別 extra で黙って外れると欠陥が復活する。実際に起きた:

- `engines-nemo` は `lightning<2.6` / `nemo-toolkit<2.5.0` を持っていたが、
  `all` は `nemo-toolkit>=2.3.0` だけで lightning の制約すら無かった
- extra は継承しないので、`pip install livecap-cli[all]` は lightning 2.6+ を選べる。
  2.6.0 が `NeptuneLogger` を削除した一方 NeMo 2.3.0 は無条件に import するため、
  `import nemo.collections.asr` が落ちる
- **repo の `uv.lock` は全 extra をまとめて解決するのでこの漏れを隠す** — CI が
  lock を使う限り気づけない。公開 package metadata を直接見る検査が要る
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# ``tomllib`` は Python 3.11+。本 repo は 3.10 も支える (`requires-python >=3.10,<3.13`)
# ので、3.10 では `tomli` へ落ちる。どちらも無ければ skip する — この検査は
# 3.11 / 3.12 の CI job で走れば metadata の drift は捕まえられる。
try:  # pragma: no cover - インタプリタ差
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        _toml = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    _toml is None, reason="tomllib (3.11+) も tomli も無い"
)

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: 上限を揃えるべき extra の対。(狭い方, 広い方)
#: 広い方は「まとめて入る」meta extra なので、狭い方の上限を必ず含む必要がある。
_MUST_AGREE = [("engines-nemo", "all")]


def _optional_dependencies() -> dict[str, list[str]]:
    data = _toml.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _capped(requirements: list[str]) -> dict[str, set[str]]:
    """上限つき要件を ``{正規化名: {specifier 文字列, ...}}`` で返す。

    ``nemo-toolkit`` と ``nemo_toolkit[asr]`` のように同じ配布物が別表記で
    並ぶので、名前は :func:`canonicalize_name` で正規化する。
    """
    found: dict[str, set[str]] = {}
    for raw in requirements:
        try:
            requirement = Requirement(raw)
        except Exception:  # noqa: BLE001 - 直接 URL 指定など
            continue
        if "<" not in str(requirement.specifier):
            continue
        found.setdefault(canonicalize_name(requirement.name), set()).add(
            str(requirement.specifier)
        )
    return found


def _all_specifiers(requirements: list[str]) -> dict[str, set[str]]:
    """``{正規化名: {specifier 文字列, ...}}`` (上限の有無を問わない)。"""
    found: dict[str, set[str]] = {}
    for raw in requirements:
        try:
            requirement = Requirement(raw)
        except Exception:  # noqa: BLE001
            continue
        found.setdefault(canonicalize_name(requirement.name), set()).add(
            str(requirement.specifier)
        )
    return found


def missing_caps(
    narrow_caps: dict[str, set[str]], wide_specifiers: dict[str, set[str]]
) -> list[str]:
    """狭い側の上限のうち、**広い側に無いもの**を列挙する。

    差分の向きが要点である (レビュー指摘)。見たいのは「狭い側の上限が広い側から
    **落ちていない**か」なので ``narrow - wide`` を取る。逆向き (``wide - narrow``)
    だと、狭い側が同じ配布物に複数の上限を持ち広い側がその一部だけを持つケースを
    見逃す — 例: narrow ``{"<2", "<3"}`` / wide ``{"<2"}`` で ``wide - narrow`` は空になる。

    **完全一致 (``==``) にはしない。** ``wide_specifiers`` は上限の無い要件も含むので、
    広い側が同じ配布物を別表記で追加で並べているだけでも落ちてしまう。
    ここで守りたいのは「上限が消えていないこと」であって「記述が同一であること」ではない。
    """
    problems: list[str] = []
    for name, specifiers in sorted(narrow_caps.items()):
        if name not in wide_specifiers:
            problems.append(f"{name}: 広い extra に無い (期待: {sorted(specifiers)})")
            continue
        missing = specifiers - wide_specifiers[name]
        if missing:
            problems.append(
                f"{name}: 上限 {sorted(missing)} が広い extra から落ちている "
                f"(広い側: {sorted(wide_specifiers[name])})"
            )
    return problems


@pytest.mark.parametrize(("narrow", "wide"), _MUST_AGREE)
def test_upper_bounds_are_not_dropped_by_the_meta_extra(narrow: str, wide: str) -> None:
    extras = _optional_dependencies()
    narrow_caps = _capped(extras[narrow])
    assert narrow_caps, f"{narrow} に上限つき要件が 1 つも無い (検査が空振りしている)"

    problems = missing_caps(narrow_caps, _all_specifiers(extras[wide]))
    assert not problems, (
        f"**{narrow} の上限が {wide} で外れている。** extra は継承しないので、"
        f"広い方にも同じ上限を書くこと (uv.lock は全 extra をまとめて解決するため "
        f"lock では気づけない):\n  " + "\n  ".join(problems)
    )


class TestMissingCapsDirection:
    """差分の向きを固定する。**これ自体が空振りしやすい検査**なので単体で押さえる。"""

    def test_detects_a_cap_dropped_entirely(self):
        problems = missing_caps({"pkg": {">=1,<2"}}, {"pkg": {">=1"}})
        assert problems and "pkg" in problems[0]

    def test_detects_a_missing_entry(self):
        problems = missing_caps({"pkg": {">=1,<2"}}, {})
        assert problems and "広い extra に無い" in problems[0]

    def test_detects_a_partially_dropped_cap(self):
        """**逆向きの差分では見逃すケース。**

        狭い側が同じ配布物に 2 つの上限を持ち、広い側がその一方だけを持つ。
        ``wide - narrow`` は空集合になるので、向きを間違えると素通りする。
        """
        problems = missing_caps({"pkg": {"<2", "<3"}}, {"pkg": {"<2"}})
        assert problems, "narrow の <3 が落ちているのに検出できていない"
        assert "<3" in problems[0]

    def test_accepts_extra_unrelated_specifiers_on_the_wide_side(self):
        """広い側が同じ配布物を**別表記で追加**していても落とさない。

        守りたいのは「上限が消えていないこと」であって「記述が同一であること」ではない
        (完全一致にしない理由)。
        """
        assert missing_caps({"pkg": {">=1,<2"}}, {"pkg": {">=1,<2", ">=1"}}) == []
