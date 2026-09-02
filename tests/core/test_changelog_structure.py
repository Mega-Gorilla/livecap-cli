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

import re
from pathlib import Path
from typing import NamedTuple

import pytest

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
            rows.append((line, bool(fence)))
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
    if not positions:
        # ここから先は [Unreleased] 配下を見る検査で、節が無ければ意味を持たない
        return problems

    head = first_h2(text)
    if head != "## [Unreleased]":
        problems.append(f"unreleased-not-first: 最初の H2 が {head!r}")

    sections = unreleased_sections(text)
    names = [n for n, _ in sections]

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

    # 重複・未知があると expected 側が縮んで順序も必ず食い違う。**原因は 1 つ**なので、
    # 同じ崩れを 2 つの code で報告しない (直す順序が読み取れなくなる)。
    expected = [n for n in ALLOWED_ORDER if n in names]
    if names != expected and not dupes and not unknown:
        problems.append(f"heading-order: H3 の順序が違う {names} != {expected}")

    empty = [n for n, body in sections if not any(l.strip() for l in body)]
    if empty:
        problems.append(f"empty-section: 本文が空の H3 がある {empty}")

    return problems


#: `[#123]: https://...` の形の**リンク定義**行。
_DEFINITION = re.compile(r"^\[(#\d+)\]:\s")
#: 本文中の `[#123]` という**参照**。
_REFERENCE = re.compile(r"\[(#\d+)\]")
#: インライン code span。**中の参照はそもそもリンクにならない**ので定義を要求しない。
_CODE_SPAN = re.compile(r"`+[^`]*`+")


def validate_reference_links(text: str) -> list[str]:
    """``[#123]`` 参照に定義があることを検査する。

    **対象はファイル全体である** — ``validate_changelog_structure()`` が
    ``[Unreleased]`` 配下しか見ないのに対し、定義は末尾の ``## Issue References``
    にあり、参照は ``## Migration Guide`` 配下にもある (``[#64]`` / ``[#69]``〜)。
    スコープが違うので関数を分けてある。

    **GitHub はリポジトリ内のファイルでは ``#123`` を自動リンクしない。** 定義の無い
    参照は literal のまま表示されるので、定義済みのものだけがリンクになる不均一な
    状態になる (#438)。

    **未使用の定義 (定義はあるが参照が無い) は検査しない。** エントリより先に定義を
    書く運用を塞ぐためである。
    """
    defined: set[str] = set()
    used: dict[str, int] = {}
    for lineno, (line, fenced) in enumerate(scan(text).rows, 1):
        # **fence の中は定義としても参照としても数えない。** code block に書いた
        # `[#123]` の例を「未定義の参照」と誤検出させない。
        if fenced:
            continue
        # **インライン code span も同じ理由で外す。** ``[#123]`` のように参照の
        # *書き方* を説明した箇所は Markdown 上もリンクにならないので、定義を
        # 要求するのは誤りである (本 issue のエントリを書いた時点で実際に踏んだ)。
        bare = _CODE_SPAN.sub("", line)
        match = _DEFINITION.match(bare)
        if match:
            defined.add(match.group(1))
            continue
        for ref in _REFERENCE.findall(bare):
            used.setdefault(ref, lineno)

    missing = sorted(set(used) - defined, key=lambda r: int(r[1:]))
    if not missing:
        return []
    detail = ", ".join(f"{r} (L{used[r]})" for r in missing)
    return [
        f"undefined-reference: リンク定義の無い参照が {len(missing)} 件ある: {detail}。"
        "## Issue References へ定義を足すこと"
    ]


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


def test_changelog_reference_links_are_defined():
    """`[#123]` 参照がすべて定義されていること。

    定義が無いと、GitHub 上では**リンクにならず literal のまま**表示される。
    """
    assert validate_reference_links(_text()) == []


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


def test_base_fixture_is_valid_and_next_h2_is_not_scanned():
    """(c) 変異の土台が clean であること — **かつ、それが解析範囲の証明でもある**。

    ``_BASE`` の ``## Migration Guide`` 配下には ``### Added`` の重複と未知の
    ``### なにか独自の見出し`` を**わざと**置いてある。解析が次の H2 で止まらなければ、
    ここが赤くなる。

    土台が最初から赤いと、以降の変異テストは「何を検出したのか」を保証しない。
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


#: fence の開閉ロジックは ``` と ~~~ で共通なので、fence 系はこれで parameterize する。
FENCES = ("```", "~~~")


@pytest.mark.parametrize("fence", FENCES)
def test_mutation_headings_inside_code_fence_are_ignored(fence: str):
    """**fenced code block の中の見出しを実見出しとして解釈しない。**

    解釈すると ``## Example`` を「次の H2」と誤認して解析が途中で終わり、後続の
    本物の H3 が検査されなくなる。``### Notes`` は未知の見出しとして誤検出される。
    """
    fenced = f"""# Changelog

## [Unreleased]

### Added

#### Markdown の例を載せるエントリ

{fence}
## Example
### Notes
{fence}

- **Added**: あるもの

### Removed

- 別のもの
"""
    assert [n for n, _ in unreleased_sections(fenced)] == ["Added", "Removed"]
    assert validate_changelog_structure(fenced) == []


@pytest.mark.parametrize("fence", FENCES)
def test_mutation_unclosed_fence_is_detected(fence: str):
    """**閉じていない fence は以降の検査を黙って無効にする** — それ自体を報告する。

    閉じ fence を消すと残りの行がすべて「code block の中」になり、**未知見出し
    (``Notes``) と後続の実 H3 (``Fixed``) が検査から消える**。修正前は ``validate``
    が ``[]`` を返していた (レビュー指摘。実測済み)。

    **隠れている内容が本当に問題なのか**は、同じ本文で fence を閉じたときに
    ``unknown-heading`` が報告されることで示す。これを assert しないと、
    「隠されていた」と言いながら中身が無害だった可能性が残る。
    """
    body = """
### Notes

- 閉じていない fence に隠される

### Fixed

- これも隠される
"""
    unclosed = f"""# Changelog

## [Unreleased]

### Added

{fence}
example
{body}"""
    assert "unclosed-fence" in _codes(unclosed), "閉じ fence が無いこと自体を報告できていない"
    # bypass の実体 — 報告が無ければ、以下は「問題なし」に見えてしまう
    assert [n for n, _ in unreleased_sections(unclosed)] == ["Added"]

    # 同じ本文で fence を閉じれば、隠れていた問題がちゃんと出る
    closed = f"""# Changelog

## [Unreleased]

### Added

{fence}
example
{fence}
{body}"""
    assert "unclosed-fence" not in _codes(closed)
    assert "unknown-heading" in _codes(closed)


_REFERENCE_BASE = """# Changelog

## [Unreleased]

### Added

- 参照 [#123] を使う

## Issue References

[#123]: https://example.invalid/123
"""


def test_reference_base_fixture_is_valid():
    """参照の変異の土台が**そのままでは問題を出さない**こと。"""
    assert validate_reference_links(_REFERENCE_BASE) == []


def test_mutation_undefined_reference_is_detected():
    """定義の無い参照を足したら落ちる。"""
    mutated = _REFERENCE_BASE.replace("- 参照 [#123] を使う", "- 参照 [#123] と [#999] を使う", 1)
    problems = validate_reference_links(mutated)
    assert [p.split(":", 1)[0] for p in problems] == ["undefined-reference"]
    assert "#999" in problems[0]


def test_mutation_definition_removed_is_detected():
    """定義を消したら落ちる (足す方向だけでなく、消す方向でも検出する)。"""
    mutated = _REFERENCE_BASE.replace("[#123]: https://example.invalid/123\n", "", 1)
    assert "#123" in validate_reference_links(mutated)[0]


@pytest.mark.parametrize("fence", FENCES)
def test_mutation_reference_inside_fence_is_ignored(fence: str):
    """**code block の中の `[#999]` を未定義参照として誤検出しない。**

    CHANGELOG は Markdown の例を code block で載せるので、参照の書き方を説明した
    だけで赤くなってはならない。
    """
    fenced = _REFERENCE_BASE.replace(
        "- 参照 [#123] を使う",
        f"- 参照 [#123] を使う\n\n{fence}markdown\n参照は [#999] のように書く\n{fence}",
        1,
    )
    assert validate_reference_links(fenced) == []


@pytest.mark.parametrize("fence", FENCES)
def test_mutation_definition_inside_fence_does_not_count(fence: str):
    """**code block の中の定義例を定義として数えない。**

    数えると「書き方の例を載せただけで定義したことになる」ので、実際には未定義の
    まま緑になる。
    """
    fenced = _REFERENCE_BASE.replace(
        "- 参照 [#123] を使う",
        f"- 参照 [#123] と [#999] を使う\n\n{fence}\n[#999]: https://example.invalid/999\n{fence}",
        1,
    )
    assert "#999" in validate_reference_links(fenced)[0]


def test_mutation_reference_inside_inline_code_is_ignored():
    """**インライン code span の中の参照を未定義として誤検出しない。**

    ``[#999]`` のように参照の*書き方*を説明した箇所は、Markdown 上もリンクに
    ならないので定義を要求するのは誤りである。**本 issue のエントリを書いた時点で
    実際に踏んだ** (`[#123]` と書いただけで赤くなった)。
    """
    mutated = _REFERENCE_BASE.replace(
        "- 参照 [#123] を使う", "- 参照 [#123] を使う。書き方は `[#999]` である", 1
    )
    assert validate_reference_links(mutated) == []


def test_mutation_definition_inside_inline_code_does_not_count():
    """インライン code span の中の定義例を定義として数えない。"""
    mutated = _REFERENCE_BASE.replace(
        "- 参照 [#123] を使う",
        "- 参照 [#123] と [#999] を使う。定義は `[#999]: https://example.invalid/999` と書く",
        1,
    )
    assert "#999" in validate_reference_links(mutated)[0]


def test_unused_definition_is_not_reported():
    """**未使用の定義は報告しない。** エントリより先に定義を書く運用を塞がない。"""
    mutated = _REFERENCE_BASE + "[#456]: https://example.invalid/456\n"
    assert validate_reference_links(mutated) == []


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
