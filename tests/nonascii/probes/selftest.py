"""仕込み欠陥による自己検証プローブ (Issue #378)。

期待 verdict が既知のプローブを**モックではなく実の runner** に通すことで、
「ハーネスが本当に fail_silent を検出できるか」を CI 時点で言わせる。
``selftest.silent_truncation`` を ``fail_silent`` と分類できないハーネスは
証拠として使えない。

**プローブ実装の規約**: 戻り値の観測 dict に**パスそのものを含めてはならない**。
control (ASCII) と variant (非 ASCII) の観測を等値比較するため、パスを含めると
常に差分が出て全行が fail_silent になる。「読めたバイト数」「テキスト内容」
「shape」など、パスに依存しない事実だけを返すこと。
"""

from __future__ import annotations

import os
import time

from ..record import ProbeContext
from . import probe

_PAYLOAD = "livecap-probe-payload"


def _artifact(ctx: ProbeContext):
    """ASCII 名のファイルを variant ディレクトリ配下に作って返す。"""
    path = ctx.root / "artifact.txt"
    path.write_text(_PAYLOAD, encoding="utf-8")
    return path


def _is_nonascii(ctx: ProbeContext) -> bool:
    return not str(ctx.root).isascii()


@probe("selftest.pass")
def selftest_pass(ctx: ProbeContext) -> dict:
    """どのパスでも正しく読む。→ 偽陽性が無いことの証明。"""
    path = _artifact(ctx)
    ctx.stage("write")
    text = path.read_text(encoding="utf-8")
    ctx.stage("read")
    return {"text": text, "length": len(text)}


@probe("selftest.loud")
def selftest_loud(ctx: ProbeContext) -> dict:
    """非 ASCII のとき、**パスを名指しして**失敗する。→ fail_loud。"""
    path = _artifact(ctx)
    ctx.stage("write")
    if _is_nonascii(ctx):
        raise OSError(f"cannot open {path}")
    return {"text": path.read_text(encoding="utf-8"), "length": len(_PAYLOAD)}


@probe("selftest.silent_truncation")
def selftest_silent_truncation(ctx: ProbeContext) -> dict:
    """非 ASCII のとき**自分でエラーを握り潰して空を返す**。→ fail_silent (条件 1)。

    ``base_engine._verify_model_integrity`` の ``except Exception: return False``
    パターンの写し。本ハーネスの中核的な検出能力はこれで証明される。
    """
    path = _artifact(ctx)
    ctx.stage("write")
    try:
        if _is_nonascii(ctx):
            raise OSError("simulated narrow-path failure")
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = ""  # ← 握り潰し。呼び出し側からは成功に見える
    ctx.stage("read")
    return {"text": text, "length": len(text)}


@probe("selftest.silent_deferred")
def selftest_silent_deferred(ctx: ProbeContext) -> dict:
    """非 ASCII でも「ロード成功」、後段で IndexError。→ fail_silent (条件 2)。

    実測済みの ReazonSpeech 挙動 (ロードは通り decode が全件 IndexError) の写し。
    """
    _artifact(ctx)
    ctx.stage("write")
    ctx.stage("load")  # 非 ASCII でもここは通る
    if _is_nonascii(ctx):
        raise IndexError("list index out of range")
    ctx.stage("decode")
    return {"tokens": ["a", "b"], "length": 2}


@probe("selftest.silent_mangled")
def selftest_silent_mangled(ctx: ProbeContext) -> dict:
    """真因に関係なく汎用メッセージへすり替える。→ fail_silent (条件 3)。"""
    _artifact(ctx)
    ctx.stage("write")
    if _is_nonascii(ctx):
        raise RuntimeError("NVIDIA NeMo is not installed. Please run: pip install nemo_toolkit[asr]")
    ctx.stage("load")
    return {"loaded": True}


@probe("selftest.crash")
def selftest_crash(ctx: ProbeContext) -> dict:
    """非 ASCII でネイティブ abort 相当の即死。→ 子プロセス隔離が生き残ることの証明。"""
    _artifact(ctx)
    ctx.stage("write")
    if _is_nonascii(ctx):
        os._exit(3)
    ctx.stage("load")
    return {"loaded": True}


@probe("selftest.timeout")
def selftest_timeout(ctx: ProbeContext) -> dict:
    """非 ASCII でハングする。→ timeout 封じ込めの証明。"""
    _artifact(ctx)
    ctx.stage("write")
    if _is_nonascii(ctx):
        time.sleep(3600)
    ctx.stage("load")
    return {"loaded": True}
