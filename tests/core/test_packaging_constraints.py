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

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: 上限を揃えるべき extra の対。(狭い方, 広い方)
#: 広い方は「まとめて入る」meta extra なので、狭い方の上限を必ず含む必要がある。
_MUST_AGREE = [("engines-nemo", "all")]


def _optional_dependencies() -> dict[str, list[str]]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
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


@pytest.mark.parametrize(("narrow", "wide"), _MUST_AGREE)
def test_upper_bounds_are_not_dropped_by_the_meta_extra(narrow: str, wide: str) -> None:
    extras = _optional_dependencies()
    narrow_caps = _capped(extras[narrow])
    assert narrow_caps, f"{narrow} に上限つき要件が 1 つも無い (検査が空振りしている)"

    wide_specifiers: dict[str, set[str]] = {}
    for raw in extras[wide]:
        try:
            requirement = Requirement(raw)
        except Exception:  # noqa: BLE001
            continue
        wide_specifiers.setdefault(canonicalize_name(requirement.name), set()).add(
            str(requirement.specifier)
        )

    problems: list[str] = []
    for name, specifiers in narrow_caps.items():
        if name not in wide_specifiers:
            problems.append(f"{name}: {wide} に無い (期待: {sorted(specifiers)})")
            continue
        missing = wide_specifiers[name] - specifiers
        if missing:
            problems.append(
                f"{name}: {wide} 側の {sorted(missing)} が {narrow} の "
                f"{sorted(specifiers)} と一致しない"
            )

    assert not problems, (
        f"**{narrow} の上限が {wide} で外れている。** extra は継承しないので、"
        f"広い方にも同じ上限を書くこと (uv.lock は全 extra をまとめて解決するため "
        f"lock では気づけない):\n  " + "\n  ".join(problems)
    )
