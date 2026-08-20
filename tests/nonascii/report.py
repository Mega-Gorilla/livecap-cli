"""棚卸し表のレンダリング (Issue #378)。

``registry`` (静的な分類) と ``results.json`` (実測) を突き合わせて
markdown を stdout に出す。**docs の §3 は手書きしない** — このコマンドの
出力を貼る。

    uv run python -m tests.nonascii.report --json benchmark_results/nonascii/<date>/results.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from .record import read_results
from .registry import (
    BOUNDARIES,
    SECTION_ORDER,
    BoundarySpec,
    Method,
    callsite_label,
)

_VERDICT_LABEL = {
    "pass": "✅ pass",
    "fail_loud": "⚠️ fail_loud",
    "fail_silent": "🔴 **fail_silent**",
    "skipped": "⏭ skipped",
    "error_harness": "🧪 error_harness",
}


def _escape(text: str) -> str:
    """markdown の表セルに入れても崩れないようにする。"""
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _summarise(results: list[dict], spec: BoundarySpec) -> tuple[str, str]:
    """(実測結果セル, 失敗の可視性セル) を返す。"""
    rows = [r for r in results if r["boundary_id"] == spec.boundary_id]
    if not rows:
        reason = spec.unmeasured_reason or "未実測"
        return f"— 未実測 ({_escape(reason)})", _escape(spec.failure_visibility)

    by_verdict: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_verdict[r["verdict"]].append(r["variant"])

    parts = []
    for verdict in ("fail_silent", "fail_loud", "error_harness", "skipped", "pass"):
        variants = by_verdict.get(verdict)
        if variants:
            label = _VERDICT_LABEL.get(verdict, verdict)
            parts.append(f"{label}: {', '.join(sorted(variants))}")
    measured = " / ".join(parts)

    visibility = spec.failure_visibility
    criteria = sorted({c for r in rows for c in (r.get("silent_criteria_hit") or [])})
    if criteria:
        visibility = (visibility + " " if visibility else "") + f"(判定根拠: {', '.join(criteria)})"
    mentions = [r for r in rows if r.get("error_mentions_path")]
    if mentions and not criteria:
        visibility = (visibility + " " if visibility else "") + "(エラーが問題のパスを名指しする)"
    skipped_reasons = sorted({r.get("skipped_reason") or "" for r in rows if r.get("skipped_reason")})
    if skipped_reasons:
        visibility = (visibility + " " if visibility else "") + f"(skip 理由: {skipped_reasons[0]})"

    return _escape(measured), _escape(visibility)


def render_table(results: list[dict]) -> str:
    out: list[str] = []
    by_section: dict[str, list[BoundarySpec]] = defaultdict(list)
    for spec in BOUNDARIES:
        by_section[spec.section].append(spec)

    for section in SECTION_ORDER:
        specs = by_section.get(section)
        if not specs:
            continue
        out.append(f"### {section.value}")
        out.append("")
        out.append(
            "| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | "
            "非 ASCII 実測 | 失敗の可視性 | 決定 | **実測で確定** | 粒度 | 追跡 |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for spec in specs:
            measured, visibility = _summarise(results, spec)
            if spec.measurement_caveat:
                visibility = (visibility + " " if visibility else "") + (
                    "計測範囲: " + _escape(spec.measurement_caveat)
                )
            verified = (
                f"**{spec.verified_method.value}**" if spec.verified_method else "— 未確定"
            )
            out.append(
                "| `{callsite}` | {path} | {receiver} | {wide} | {measured} | "
                "{visibility} | {method} | {verified} | {gran} | {issue} |".format(
                    callsite=callsite_label(spec),
                    path=_escape(spec.path_desc),
                    receiver=_escape(spec.receiver),
                    wide=_escape(spec.wide_path_support),
                    measured=measured,
                    visibility=visibility or "—",
                    method=spec.candidate_method.value,
                    verified=verified,
                    gran=spec.granularity,
                    issue=spec.followup_issue or "—",
                )
            )
        out.append("")
    return "\n".join(out)


def render_metadata(run: dict) -> str:
    rows = [
        ("OS / arch", f"{run['os']} / {run['machine']}"),
        ("Python", run["python"]),
        ("実測ホスト", "開発機 (Windows / 日本語ロケール)"),
        ("ANSI code page (ACP)", str(run["active_code_page"])),
        ("OEM code page", str(run["oem_code_page"])),
        ("filesystem encoding", run["fs_encoding"]),
        ("locale preferred encoding", run["preferred_encoding"]),
        ("Python UTF-8 mode", str(run["python_utf8_mode"])),
        ("LongPathsEnabled", str(run["long_paths_enabled"])),
        ("8.3 生成の状態", _escape(str(run["eight_dot_three_state"]))),
        ("Windows ユーザー名は ASCII か", str(run["username_is_ascii"])),
        ("システム %TEMP% は ASCII か", str(run["system_temp_is_ascii"])),
        ("プローブ root のボリューム", run["root_volume"]),
        ("採用した root 候補", run.get("root_label") or "(未記録)"),
        ("共有される親 root", run.get("root_parent") or "(未記録)"),
        ("この run の session root", run.get("nonascii_root") or "(未記録)"),
        (
            "回収した stale session",
            ", ".join(run.get("reaped_stale_sessions") or []) or "なし",
        ),
        (
            "落ちた root 候補",
            ", ".join(f"{k}: {v}" for k, v in (run.get("rejected_roots") or {}).items())
            or "なし",
        ),
        ("実モデルの実体化方式", run["materialization"]),
        ("対応した variant", ", ".join(run["variants_supported"])),
        (
            "非対応の variant",
            ", ".join(f"{k} ({v})" for k, v in (run["variants_skipped"] or {}).items()) or "なし",
        ),
        ("NFD 正規化の保存", str(run["normalization_preserved"])),
        ("有効な tier", ", ".join(run["tiers_enabled"])),
        ("git commit", run["git_commit"]),
        ("run_id", run["run_id"]),
        ("最終検証日", run["measured_at"]),
    ]
    lines = ["| 項目 | 値 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]

    lines.append("")
    lines.append("パッケージ版数:")
    lines.append("")
    lines.append("| パッケージ | 版 |")
    lines.append("|---|---|")
    for name, version in sorted(run["packages"].items()):
        lines.append(f"| {name} | {version} |")
    return "\n".join(lines)


def render_summary(results: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r["verdict"]] += 1
    method_counts: dict[str, int] = defaultdict(int)
    for spec in BOUNDARIES:
        method_counts[spec.candidate_method.value] += 1

    verified_counts: dict[str, int] = defaultdict(int)
    for spec in BOUNDARIES:
        key = spec.verified_method.value if spec.verified_method else "未確定"
        verified_counts[key] += 1
    n_verified = sum(1 for b in BOUNDARIES if b.verified_method)

    lines = [
        f"- 棚卸し行数: **{len(BOUNDARIES)}**、未分類 (決定なし): "
        f"**{sum(1 for b in BOUNDARIES if b.candidate_method not in set(Method))}**",
        f"- 実測レコード数: **{len(results)}**",
        "- **決定** の内訳: "
        + " / ".join(f"{k} {v} 行" for k, v in sorted(method_counts.items())),
        f"- **実測で確定** している行: **{n_verified} / {len(BOUNDARIES)}** — "
        + " / ".join(f"{k} {v} 行" for k, v in sorted(verified_counts.items())),
        "- 判定の内訳: "
        + (" / ".join(f"{_VERDICT_LABEL.get(k, k)} {v}" for k, v in sorted(counts.items())) or "なし"),
        "",
        "> 「決定」は source-check を含む分類、「実測で確定」は runtime 実測がその分類を"
        "裏付けている行だけを数える。issue #378 の ② の採用条件は「実測で非 ASCII が通る」"
        "なので、この 2 つを分けないと「未分類ゼロ」が実態より強い保証に見えてしまう。",
    ]
    return "\n".join(lines)


def _inject(doc: Path, run: dict, results: list[dict]) -> int:
    """doc のマーカー間へ自動生成セクションを差し込む。

    §0 (測定メタデータ) と §3 (棚卸し表) を手書きさせないための仕組み。
    マーカーが 1 つでも欠けていればエラーにする (黙って何もしないと、
    doc が古いまま「再生成した」と誤認されるため)。
    """
    sections = {
        "METADATA": "## 0. 測定メタデータ\n\n" + render_metadata(run),
        "SUMMARY": "## 集計\n\n" + render_summary(results),
        "TABLE": "## 3. 棚卸し表\n\n" + render_table(results),
    }
    text = doc.read_text(encoding="utf-8")
    for name, body in sections.items():
        begin = f"<!-- BEGIN:{name} -->"
        end = f"<!-- END:{name} -->"
        if begin not in text or end not in text:
            sys.stderr.write(f"マーカーが見つからない: {begin} / {end}\n")
            return 1
        head, _, rest = text.partition(begin)
        _, _, tail = rest.partition(end)
        text = head + begin + "\n" + body + "\n" + end + tail
    doc.write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="results.json のパス",
    )
    parser.add_argument(
        "--section",
        choices=("all", "metadata", "table", "summary"),
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先ファイル (省略時は stdout)",
    )
    parser.add_argument(
        "--inject",
        type=Path,
        default=None,
        help=(
            "既存の doc の <!-- BEGIN:X --> / <!-- END:X --> マーカー間へ "
            "METADATA / SUMMARY / TABLE を差し込む (§0 と §3 を手書きさせないため)"
        ),
    )
    args = parser.parse_args(argv)

    run, results = read_results(args.json)

    if args.inject is not None:
        return _inject(args.inject, run, results)

    blocks = []
    if args.section in ("all", "metadata"):
        blocks.append("## 0. 測定メタデータ\n\n" + render_metadata(run))
    if args.section in ("all", "summary"):
        blocks.append("## 集計\n\n" + render_summary(results))
    if args.section in ("all", "table"):
        blocks.append("## 3. 棚卸し表\n\n" + render_table(results))

    text = "\n\n".join(blocks) + "\n"

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        return 0

    # **本ツール自身が、本ツールが文書化している欠陥を踏まないようにする。**
    # Windows の stdout はパイプ接続時に locale (cp932) エンコーダになり、
    # ACP に無い文字を書くと UnicodeEncodeError で落ちる。出力ストリームを
    # 明示的に UTF-8 にするのが正しい対処であり、これは cli.py の SRT stdout
    # 出力にも同じことが言える (棚卸し表 §3.5 の cli.stdout_srt_write 行)。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
