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

**検証は :func:`validate_changelog_structure` へ集約してある。** 実ファイルの検査も変異
テストも**同じ関数**を呼ぶ — 変異側が同じ判定式をその場で書き直すと、**本番側の assertion
を壊しても変異テストは緑のまま**になり、回帰ゲートとして機能しない (レビュー指摘)。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

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


class ScanResult(NamedTuple):
    """``scan()`` の結果。

    ``unclosed_at`` は**閉じていない fence の開始行** (1-origin。閉じていれば ``None``)。
    **これを返さないと、閉じ fence を消しただけで以降の検査が黙って無効になる** —
    残りの行がすべて「code block の中」になり、未知見出し・重複・順序違反が
    検査対象から消える (レビュー指摘。実測で `validate` が `[]` を返した)。
    """

    rows: list[tuple[str, bool]]
    unclosed_at: int | None


def scan(text: str) -> ScanResult:
    """``(行, fence の内側か)`` と、閉じていない fence の位置を返す。

    **fenced code block の中の ``##`` / ``###`` を見出しと解釈してはならない。**
    CHANGELOG は Markdown の例を code block で載せるので、``## Example`` が入った
    時点で「次の H2」と誤認して解析が途中で終わり、**本物の H3 が検査されなくなる**。
    ``### Notes`` だけなら逆に未知の見出しとして誤検出する (レビュー指摘)。

    Markdown parser は要らない。開いた fence と**同じ文字・同じ長さ以上**で閉じる、
    という局所的な走査で足りる。
    """
    rows: list[tuple[str, bool]] = []
    fence: tuple[str, int] | None = None
    opened_at: int | None = None
    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if fence is None:
            for char in ("`", "~"):
                if stripped.startswith(char * 3):
                    fence = (char, len(stripped) - len(stripped.lstrip(char)))
                    opened_at = lineno
                    break
            rows.append((line, True if fence else False))
            continue
        char, width = fence
        # 閉じ fence は「その文字だけ」で構成され、開いたときの長さ以上であること
        if stripped.startswith(char * width) and set(stripped) == {char}:
            fence = None
            opened_at = None
        rows.append((line, True))
    return ScanResult(rows, opened_at)


def unreleased_headings(text: str) -> list[int]:
    """``## [Unreleased]`` 見出しの行番号 (0-origin) をすべて返す。

    ``unreleased_sections()`` は**最初の 1 つ**を見て次の H2 で止めるので、2 つ目が
    あっても素通りする。Keep a Changelog は「今後の変更を集める ``Unreleased`` 節を
    先頭に 1 つ置く」構造なので、そこは別に固定する。
    """
    return [
        i
        for i, (line, fenced) in enumerate(scan(text).rows)
        if not fenced and line.startswith("## [Unreleased]")
    ]


def first_h2(text: str) -> str | None:
    """最初の H2 見出しを返す (無ければ ``None``)。"""
    for line, fenced in scan(text).rows:
        if not fenced and line.startswith("## "):
            return line
    return None


def unreleased_sections(text: str) -> list[tuple[str, list[str]]]:
    """``## [Unreleased]`` 配下の H3 を ``(名前, 本文行)`` で返す。

    **解析は次の H2 で止める。** 止めないと ``## Migration Guide`` /
    ``## Issue References`` 配下の H3 まで巻き込み、実在しない重複を報告する
    (#436 の初版が実際にこれを踏み、``[Unreleased]`` を 2977 行と数えた)。

    **fence の中の行は本文として残す** — 見出しとして解釈しないだけである。
    空にしてしまうと、code block だけの節が「空の節」に見える。
    """
    scanned = scan(text).rows
    try:
        start = next(
            i
            for i, (line, fenced) in enumerate(scanned)
            if not fenced and line.startswith("## [Unreleased]")
        )
    except StopIteration:  # pragma: no cover - CHANGELOG が壊れていない限り来ない
        raise AssertionError("## [Unreleased] が見つからない")
    end = next(
        (
            i
            for i, (line, fenced) in enumerate(scanned[start + 1 :], start + 1)
            if not fenced and line.startswith("## ")
        ),
        len(scanned),
    )

    sections: list[tuple[str, list[str]]] = []
    for line, fenced in scanned[start + 1 : end]:
        if not fenced and line.startswith("### "):
            sections.append((line[4:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def validate_changelog_structure(text: str) -> list[str]:
    """構造上の問題を ``"<code>: <説明>"`` のリストで返す (問題が無ければ空)。

    **実ファイルの検査も変異テストもこの関数を呼ぶ。** 変異側が判定式を書き直すと、
    本番側の assertion を壊しても変異テストが緑のままになる。
    """
    problems: list[str] = []

    unclosed_at = scan(text).unclosed_at
    if unclosed_at is not None:
        problems.append(
            f"unclosed-fence: {unclosed_at} 行目で開いた code fence が閉じていない。"
            "**以降の行がすべて code block 内とみなされ、未知見出し・重複・順序違反が"
            "検査対象から消える** — 先にこれを直すこと"
        )

    positions = unreleased_headings(text)
    if len(positions) != 1:
        problems.append(
            f"unreleased-count: ## [Unreleased] が {len(positions)} 個ある "
            f"(行 {[p + 1 for p in positions]})。今後の変更を集める節は 1 つだけにすること — "
            "2 つ目は以下の検査を素通りする"
        )
    if positions and first_h2(text) != "## [Unreleased]":
        problems.append(f"unreleased-not-first: 最初の H2 が {first_h2(text)!r}")

    if not positions:
        return problems

    names = [n for n, _ in unreleased_sections(text)]

    unknown = sorted(set(names) - set(ALLOWED_ORDER))
    if unknown:
        problems.append(
            f"unknown-heading: 許可されていない H3 がある {unknown}。"
            f"許可: {list(ALLOWED_ORDER)}。増やすなら ALLOWED_ORDER と AGENTS.md を同時に直すこと"
        )

    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(
            f"duplicate-heading: 同名の H3 が複数ある {dupes}。"
            "**どこへ追記するかが一意に決まらなくなる** — 統合すること"
        )

    expected = [n for n in ALLOWED_ORDER if n in names]
    if names != expected and not dupes and not unknown:
        problems.append(f"heading-order: H3 の順序が違う {names} != {expected}")

    empty = [n for n, body in unreleased_sections(text) if not any(l.strip() for l in body)]
    if empty:
        problems.append(f"empty-section: 本文が空の H3 がある {empty}")

    return problems


def _text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def _codes(text: str) -> list[str]:
    return [p.split(":", 1)[0] for p in validate_changelog_structure(text)]


def test_changelog_is_structurally_valid():
    """実ファイルに構造上の問題が無いこと。

    検査内容: ``[Unreleased]`` がちょうど 1 つで先頭の H2 / 許可された H3 のみ /
    同名の H3 が無い / 順序が固定どおり / 本文が空の節が無い。
    """
    assert validate_changelog_structure(_text()) == []


# --- 検査自体が効くことの確認 (変異) -----------------------------------------
#
# **すべて validate_changelog_structure() を通す。** 判定式をここで書き直すと、
# 本番側を壊しても緑のままになる。

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


def test_base_fixture_is_valid():
    """変異の土台が**そのままでは問題を出さない**こと。

    ここが最初から赤いと、以降の変異テストは「何を検出したのか」を保証しない。
    """
    assert validate_changelog_structure(_BASE) == []


def test_mutation_next_h2_is_not_scanned():
    """(c) ``## Migration Guide`` 以降の H3 を拾わない。

    拾うと ``### Added`` の重複と未知の見出しを誤検出する。
    """
    assert [n for n, _ in unreleased_sections(_BASE)] == ["Added", "Removed"]
    assert validate_changelog_structure(_BASE) == []


def test_mutation_duplicate_heading_is_detected():
    """(a) 同名 H3 を 2 つにしたら落ちる。"""
    mutated = _BASE.replace("### Removed", "### Added\n\n### Removed", 1)
    assert "duplicate-heading" in _codes(mutated)


def test_mutation_unknown_heading_is_detected():
    """(b) 未知の H3 を足したら落ちる。"""
    mutated = _BASE.replace("### Removed", "### Notes\n\n### Removed", 1)
    assert "unknown-heading" in _codes(mutated)


def test_mutation_wrong_order_is_detected():
    """(e) 順序を入れ替えたら落ちる。

    **fixture を実際に変異させる。** 固定リストを比較するだけでは、本番側の順序検査を
    削っても緑のままになる (レビュー指摘)。
    """
    mutated = """# Changelog

## [Unreleased]

### Removed

- あるもの

### Added

- 別のもの
"""
    assert "heading-order" in _codes(mutated)


def test_mutation_truly_empty_section_is_detected():
    """真に空の節は検出する (**検査が何も通さないだけ**、を防ぐ)。"""
    mutated = _BASE.replace("### Removed\n\n- 直接 bullet を置いた節 (H4 は無い)", "### Removed\n")
    assert "empty-section" in _codes(mutated)


def test_mutation_section_with_bullets_only_is_not_empty():
    """(d) H4 を持たないが bullet を持つ節を「空」と誤判定しない。

    ``Deprecated`` / ``Removed`` / ``Fixed`` / ``Security`` は H4 を持たないが
    bullet を 12 件持つ。#436 の初版はこれを「空節 4 つ」と誤り、**実装していれば
    消していた**。
    """
    assert "empty-section" not in _codes(_BASE)


def test_mutation_second_unreleased_is_detected():
    """2 つ目の ``## [Unreleased]`` を足したら落ちる。"""
    mutated = _BASE + "\n## [Unreleased]\n\n### Notes\n"
    assert "unreleased-count" in _codes(mutated)
    # **素通りすることの確認** — だからこそ unreleased-count の検査が要る
    assert [n for n, _ in unreleased_sections(mutated)] == ["Added", "Removed"]


def test_mutation_unreleased_not_first_is_detected():
    """``## [Unreleased]`` が先頭の H2 でなければ落ちる。"""
    mutated = _BASE.replace("## [Unreleased]", "## [9.9.9]\n\n### Added\n\n## [Unreleased]", 1)
    assert "unreleased-not-first" in _codes(mutated)


def test_mutation_headings_inside_code_fence_are_ignored():
    """**fenced code block の中の見出しを実見出しとして解釈しない。**

    解釈すると ``## Example`` を「次の H2」と誤認して解析が途中で終わり、後続の
    本物の H3 が検査されなくなる。``### Notes`` は未知の見出しとして誤検出される。
    """
    fenced = """# Changelog

## [Unreleased]

### Added

#### Markdown の例を載せるエントリ

```markdown
## Example
### Notes
```

- **Added**: あるもの

### Removed

- 別のもの
"""
    assert [n for n, _ in unreleased_sections(fenced)] == ["Added", "Removed"]
    assert validate_changelog_structure(fenced) == []


def test_mutation_tilde_fence_is_handled():
    """``~~~`` の fence も同様に無視する。"""
    fenced = """# Changelog

## [Unreleased]

### Added

~~~
### Notes
~~~

- **Added**: あるもの
"""
    assert validate_changelog_structure(fenced) == []


def test_mutation_unclosed_fence_is_detected():
    """**閉じていない fence は以降の検査を黙って無効にする** — それ自体を報告する。

    閉じ fence を消すと残りの行がすべて「code block の中」になり、後続の実 H3 が
    検査対象から消える。この入力は **未知見出し (``Notes``) と順序違反を同時に含む**
    のに、修正前は ``validate`` が ``[]`` を返していた (レビュー指摘。実測済み)。
    """
    unclosed = """# Changelog

## [Unreleased]

### Added

```text
example

### Notes

- 閉じていない fence に隠される

### Fixed

- これも隠される
"""
    codes = _codes(unclosed)
    assert "unclosed-fence" in codes, "閉じ fence が無いこと自体を報告できていない"
    # bypass の実体 — 報告が無ければ、以下は「問題なし」に見えてしまう
    assert [n for n, _ in unreleased_sections(unclosed)] == ["Added"]


def test_mutation_unclosed_tilde_fence_is_detected():
    """``~~~`` でも同じ (開閉ロジックは共通)。"""
    unclosed = """# Changelog

## [Unreleased]

### Added

~~~

### Notes

- 隠される
"""
    assert "unclosed-fence" in _codes(unclosed)


def test_mutation_fence_only_section_is_not_empty():
    """code block だけの節を「空」と誤判定しない (fence 内も本文として残すこと)。"""
    fenced = """# Changelog

## [Unreleased]

### Added

```
example
```
"""
    assert "empty-section" not in _codes(fenced)
