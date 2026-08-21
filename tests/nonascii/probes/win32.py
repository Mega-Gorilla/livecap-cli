"""Windows 固有の照会プローブ (Issue #378)。

8.3 短縮名が ASCII staging の代替にならないことを、散文ではなく
**機械記録**として残すためのプローブ。
"""

from __future__ import annotations

import sys

from ..paths import eight_dot_three_state, short_path_name
from ..record import ProbeContext, ProbeSkipped
from . import probe


@probe("win32.short_path_name")
def win32_short_path_name(ctx: ProbeContext) -> dict:
    """``GetShortPathNameW`` が非 ASCII ディレクトリに別名を返すか。

    却下理由 3 つのうち (1)(3) をここで実測する:

    1. ``ユーザー`` は 8.3 に収まるので**別名が生成されない** —
       ``GetShortPathNameW`` は入力をそのまま返す。
    3. 別名が無いとき **エラーも signal も出さず**長い名前を返す。
       つまり 8.3 を採用することは、epic が消そうとしている silent
       degradation の上に修正を建てることになる。

    (2) 「現代の Windows では 8.3 生成が既定無効」はボリューム設定の照会
    (``fsutil 8dot3name query``) として run メタデータに記録する。
    """
    if sys.platform != "win32":
        raise ProbeSkipped("Windows 以外では 8.3 短縮名の概念が無い")

    target = ctx.root / "deep"
    target.mkdir(parents=True, exist_ok=True)
    (target / "payload.bin").write_bytes(b"x" * 16)
    ctx.stage("prepare")

    short = short_path_name(target)
    ctx.stage("query_short_name")

    volume = str(target)[:3]  # 例: "C:\\"
    return {
        # パスそのものは出さず、**性質**だけを観測値にする
        "short_name_returned": short is not None,
        "short_name_is_ascii": bool(short) and short.isascii(),
        "short_name_differs_from_input": bool(short) and short != str(target),
        "input_is_ascii": str(target).isascii(),
        "eight_dot_three_state": eight_dot_three_state(volume),
    }
