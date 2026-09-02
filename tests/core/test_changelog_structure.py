"""``CHANGELOG.md`` の ``[Unreleased]`` が構造として壊れていないこと (Issue #436)。

**壊れていた実例**: ``### Changed`` が 3 つ、``### Added`` / ``### Removed`` /
``### Fixed`` が 2 つずつあり、``AGENTS.md`` が名指ししている「``### Changed`` へ書く」
が**一意に決まらない**状態だった。書いた人はその時に見つけたブロックへ追記するしかなく、
実際 #387 PR D は **``### Added`` の下に Added の bullet を 1 つも持たないエントリ**
として入った。

**この検査が守るのは構造だけである。** ``Removed`` 主体のエントリを唯一の ``### Added``
へ足しても、ここは pass する。**意味上の分類は ``AGENTS.md`` のカテゴリ定義とレビューが
担保する** — bullet の種類を数えて機械判定すると、``### Fixed`` の中の「直すために何を
Added したか」を数えて**正しく置かれた修正エントリを Fixed から追い出す**ため、その方式は
採らない (既存 5 件で実測)。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: 許可する H3 と**その順序**。
#:
#: **Keep a Changelog を基礎とし、ローカル拡張を明示的に足した形式である** —
#: ``Documentation`` 節、H4 による詳細ブロック、``[Unreleased]`` 冒頭のサマリの 3 つが
#: 標準には無い拡張にあたる。``Documentation`` を許可するのは、この CHANGELOG が利用者
#: 向けの変更履歴だけでなく調査結果と設計判断も保存しており、docs-only の変更を
#: ``Added`` / ``Changed`` へ押し込むとその区別が落ちるためである (#436)。
ALLOWED_ORDER: tuple[str, ...] = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Documentation",
    "Security",
)


def unreleased_headings(text: str) -> list[int]:
    """``## [Unreleased]`` 見出しの行番号 (0-origin) をすべて返す。

    ``unreleased_sections()`` は**最初の 1 つ**を見て次の H2 で止めるので、2 つ目が
    あっても素通りする。Keep a Changelog は「今後の変更を集める ``Unreleased`` 節を
    先頭に 1 つ置く」構造なので、そこは別に固定する。
    """
    return [i for i, line in enumerate(text.split("\n")) if line.startswith("## [Unreleased]")]


def first_h2(text: str) -> str | None:
    """最初の H2 見出しを返す (無ければ ``None``)。"""
    for line in text.split("\n"):
        if line.startswith("## "):
            return line
    return None


def unreleased_sections(text: str) -> list[tuple[str, list[str]]]:
    """``## [Unreleased]`` 配下の H3 を ``(名前, 本文行)`` で返す。

    **解析は次の H2 で止める。** 止めないと ``## Migration Guide`` /
    ``## Issue References`` 配下の H3 まで巻き込み、実在しない重複を報告する
    (#436 の初版が実際にこれを踏み、``[Unreleased]`` を 2977 行と数えた)。
    """
    lines = text.split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("## [Unreleased]"))
    except StopIteration:  # pragma: no cover - CHANGELOG が壊れていない限り来ない
        raise AssertionError("## [Unreleased] が見つからない")
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )

    sections: list[tuple[str, list[str]]] = []
    for line in lines[start + 1 : end]:
        if line.startswith("### "):
            sections.append((line[4:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def _text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def test_unreleased_is_unique_and_first():
    """``## [Unreleased]`` はちょうど 1 つで、**最初の H2** であること。

    2 つ目があると ``unreleased_sections()`` は最初の節しか見ないため、
    **後ろの節は以下の検査を丸ごと素通りする**。
    """
    text = _text()
    positions = unreleased_headings(text)
    assert len(positions) == 1, (
        f"## [Unreleased] が {len(positions)} 個ある (行 {[p + 1 for p in positions]})。"
        "今後の変更を集める節は 1 つだけにすること — 2 つ目は検査を素通りする"
    )
    assert first_h2(text) == "## [Unreleased]", (
        f"最初の H2 が ## [Unreleased] ではない: {first_h2(text)!r}"
    )


def test_headings_are_allowed():
    unknown = sorted({n for n, _ in unreleased_sections(_text())} - set(ALLOWED_ORDER))
    assert not unknown, (
        f"[Unreleased] に許可されていない H3 がある: {unknown}。"
        f"許可: {list(ALLOWED_ORDER)}。増やすなら ALLOWED_ORDER と AGENTS.md を同時に直すこと"
    )


def test_headings_are_unique():
    names = [n for n, _ in unreleased_sections(_text())]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, (
        f"[Unreleased] に同名の H3 が複数ある: {dupes}。"
        "**どこへ追記するかが一意に決まらなくなる** — 統合すること (#436)"
    )


def test_heading_order():
    names = [n for n, _ in unreleased_sections(_text())]
    expected = [n for n in ALLOWED_ORDER if n in names]
    assert names == expected, f"H3 の順序が違う: {names} != {expected}"


def test_no_truly_empty_section():
    """**空白以外の本文を持たない節**だけを空とみなす。

    ``#### `` が無いことを空の判定に使ってはならない — ``Deprecated`` / ``Removed`` /
    ``Fixed`` / ``Security`` は H4 を持たないが bullet を 12 件持つ。#436 の初版は
    これを「空節 4 つ」と誤り、**実装していれば消していた**。
    """
    empty = [n for n, body in unreleased_sections(_text()) if not any(l.strip() for l in body)]
    assert not empty, f"本文が空の H3 がある: {empty}"


# --- 検査自体が効くことの確認 (変異) -----------------------------------------

_BASE = """# Changelog

## [Unreleased]

### Added

#### 何か

- **Added**: あるもの

### Removed

- 直接 bullet を置いた節 (H4 は無い)

## Migration Guide

### Added
### Added
### なにか独自の見出し
"""


def _names(text: str) -> list[str]:
    return [n for n, _ in unreleased_sections(text)]


def test_mutation_next_h2_is_not_scanned():
    """(c) ``## Migration Guide`` 以降の H3 を拾わない。"""
    assert _names(_BASE) == ["Added", "Removed"]


def test_mutation_duplicate_heading_is_detected():
    """(a) 同名 H3 を 2 つにしたら重複として検出される。"""
    names = _names(_BASE.replace("### Removed", "### Added\n\n### Removed", 1))
    assert [n for n in names if names.count(n) > 1] == ["Added", "Added"]


def test_mutation_unknown_heading_is_detected():
    """(b) 未知の H3 を足したら弾かれる。"""
    mutated = _BASE.replace("### Removed", "### Notes\n\n### Removed", 1)
    assert sorted(set(_names(mutated)) - set(ALLOWED_ORDER)) == ["Notes"]


def test_mutation_section_with_bullets_only_is_not_empty():
    """(d) H4 を持たないが bullet を持つ節を「空」と誤判定しない。"""
    empty = [n for n, body in unreleased_sections(_BASE) if not any(l.strip() for l in body)]
    assert empty == []


def test_mutation_truly_empty_section_is_detected():
    """真に空の節はちゃんと検出する (**検査が何も通さないだけ**、を防ぐ)。"""
    mutated = _BASE.replace("### Removed\n\n- 直接 bullet を置いた節 (H4 は無い)", "### Removed\n")
    empty = [n for n, body in unreleased_sections(mutated) if not any(l.strip() for l in body)]
    assert empty == ["Removed"]


def test_mutation_second_unreleased_is_detected():
    """2 つ目の ``## [Unreleased]`` を足したら検出される。"""
    mutated = _BASE + "\n## [Unreleased]\n\n### Notes\n"
    assert len(unreleased_headings(mutated)) == 2
    # **素通りすることの確認** — だからこそ test_unreleased_is_unique_and_first が要る
    assert _names(mutated) == ["Added", "Removed"]


def test_mutation_unreleased_not_first_is_detected():
    """``## [Unreleased]`` が先頭の H2 でなければ検出される。"""
    mutated = _BASE.replace("## [Unreleased]", "## [9.9.9]\n\n### Added\n\n## [Unreleased]", 1)
    assert first_h2(mutated) == "## [9.9.9]"


def test_mutation_wrong_order_is_detected():
    """(e) 順序を入れ替えたら検出される。"""
    names = ["Removed", "Added"]
    assert names != [n for n in ALLOWED_ORDER if n in names]
