"""registry × variant のプローブ実行 (Issue #378)。

判定は runner の differential 方式に委ね、ここでは
「**ハーネス自身が壊れていないこと**」だけを assert する:

- ``error_harness`` が出ていないこと (= control が通っていること)
- ``expected_verdict`` を持つ行は、その通りに観測されること (回帰ゲート)

「非 ASCII で壊れている」こと自体は**テスト失敗にしない**。それは
棚卸し表に記録すべき**測定結果**であって、CI を赤くする対象ではない
(修正は #375 / #379 / #377 の仕事)。壊れている行が直ったとき、または
新たに壊れたときに気付けるよう、``expected_verdict`` で固定する。
"""

from __future__ import annotations

import sys

import pytest

from .conftest import real_models_enabled
from .record import Verdict
from .registry import BOUNDARIES, BoundarySpec
from .runner import run_probe

pytestmark = pytest.mark.nonascii_paths

_CHEAP = [b for b in BOUNDARIES if b.tier == "cheap" and b.probe_id]
_REAL_MODEL = [b for b in BOUNDARIES if b.tier == "real_model" and b.probe_id]
_HEAVY = [b for b in BOUNDARIES if b.tier == "heavy" and b.probe_id]

#: real_model tier の probe_id → models root からの相対パス
_REAL_MODEL_SOURCES = {
    # int8 モデル (154 MB encoder)。float32 の reazon-research--reazonspeech-k2-v2 は
    # 592 MB あり、測定内容は変わらないので軽い方を使う。
    "sherpa.from_transducer.real": "reazonspeech/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
    "voxtral.from_pretrained": "mistralai--Voxtral-Mini-3B-2507",
    "voxtral.autoprocessor": "mistralai--Voxtral-Mini-3B-2507",
}

#: heavy tier の probe_id → models root からの相対パス
_HEAVY_SOURCES = {
    "nemo.restore_from": "nvidia--parakeet-tdt-0.6b-v2.nemo",
}


def _ids(specs: list[BoundarySpec]) -> list[str]:
    return [s.boundary_id for s in specs]


def _execute(session, spec: BoundarySpec, variant_id: str, *, timeout_s: float, payload=None):
    result = run_probe(
        spec.probe_id,
        variant_id=variant_id,
        base_root=session["base_root"],
        boundary_id=spec.boundary_id,
        payload=payload,
        timeout_s=timeout_s,
        apply_to=spec.granularity,
    )
    session["results"].append(result)
    return result


def _assert_harness_healthy(result, spec: BoundarySpec) -> None:
    assert result.verdict != Verdict.ERROR_HARNESS.value, (
        f"{spec.boundary_id} / {result.variant}: control (ASCII パス) の実行が失敗した。"
        f"境界のバグではなくプローブのバグである。"
        f" type={result.exception_type} msg={result.exception_message} notes={result.notes}"
    )


def _assert_expected_verdict(result, spec: BoundarySpec) -> None:
    if not spec.expected_verdict or result.verdict == Verdict.SKIPPED.value:
        return
    if (
        spec.expected_verdict_variant
        and result.variant != spec.expected_verdict_variant
    ):
        # この期待値は特定 variant でのみ成立する (encoding 依存の行)
        return
    if spec.expected_verdict_platform and sys.platform != spec.expected_verdict_platform:
        # この期待値は特定プラットフォームでのみ成立する。
        # 例: stdout のエンコーディングは Windows (ACP=cp932) では落ちるが
        # Linux CI (stdout=UTF-8) では落ちない — CI runner と開発機で検出できる
        # 失敗の部分集合が異なる、という棚卸し表 §7 の caveat の具体例。
        return
    assert result.verdict == spec.expected_verdict, (
        f"{spec.boundary_id} の挙動が変わった: expected={spec.expected_verdict} "
        f"actual={result.verdict}。"
        f"{spec.followup_issue or '実装 PR'} が直したのなら registry の "
        f"expected_verdict を更新し、棚卸し表 §3 を再生成すること。"
    )


@pytest.mark.parametrize("spec", _CHEAP, ids=_ids(_CHEAP))
def test_cheap_boundary(nonascii_session, spec: BoundarySpec):
    """cheap tier: 合成アーティファクトのみ。モデルもネットワークも不要。"""
    variants = nonascii_session["variants"]
    if not variants:
        pytest.skip(f"この FS が非 ASCII variant を受理しない: {nonascii_session['skipped_variants']}")

    for variant_id in variants:
        result = _execute(nonascii_session, spec, variant_id, timeout_s=180)
        _assert_harness_healthy(result, spec)
        _assert_expected_verdict(result, spec)


@pytest.mark.slow
@pytest.mark.parametrize("spec", _REAL_MODEL, ids=_ids(_REAL_MODEL))
def test_real_model_boundary(nonascii_session, spec: BoundarySpec):
    """real_model tier: ローカルの実モデルを使う (**ネットワークは使わない**)。

    ``LIVECAP_NONASCII_REAL_MODELS=1`` で有効化する。
    """
    if not real_models_enabled():
        pytest.skip("LIVECAP_NONASCII_REAL_MODELS=1 が未設定")

    models_root = nonascii_session["models_root"]
    relative = _REAL_MODEL_SOURCES.get(spec.probe_id)
    if models_root is None or relative is None:
        pytest.skip(f"{spec.probe_id} の実モデル所在が未定義")
    source = models_root / relative
    if not source.exists():
        pytest.skip(f"実モデルが存在しない: {relative}")

    variants = nonascii_session["variants"]
    if not variants:
        pytest.skip("非 ASCII variant を受理しない FS")

    # 実モデルは重いので代表 variant のみ (cjk_kana = 実世界ケース)
    variant_id = "cjk_kana" if "cjk_kana" in variants else variants[0]
    result = _execute(
        nonascii_session,
        spec,
        variant_id,
        timeout_s=900,
        payload={"model_source": str(source)},
    )
    _assert_harness_healthy(result, spec)
    _assert_expected_verdict(result, spec)


@pytest.mark.slow
@pytest.mark.parametrize("spec", _HEAVY, ids=_ids(_HEAVY))
def test_heavy_boundary(nonascii_session, spec: BoundarySpec):
    """heavy tier: ``uv sync --extra engines-nemo`` が必要。

    未導入環境では probe 側が ``ProbeSkipped`` を投げ、``skipped`` として
    理由付きで記録される (「試していない」ことが表に残る)。
    """
    pytest.importorskip("nemo", reason="nemo-toolkit 未導入 (engines-nemo extra)")

    models_root = nonascii_session["models_root"]
    relative = _HEAVY_SOURCES.get(spec.probe_id)
    if models_root is None or relative is None:
        pytest.skip(f"{spec.probe_id} の実モデル所在が未定義")
    source = models_root / relative
    if not source.exists():
        pytest.skip(f".nemo が存在しない: {relative}")

    variants = nonascii_session["variants"]
    variant_id = "cjk_kana" if "cjk_kana" in variants else (variants or ["cjk_kana"])[0]
    result = _execute(
        nonascii_session,
        spec,
        variant_id,
        timeout_s=1800,
        payload={"model_source": str(source)},
    )
    _assert_harness_healthy(result, spec)
    _assert_expected_verdict(result, spec)


def test_download_directory_data_loss_is_recorded(nonascii_session):
    """``unicode_safe_download_directory`` の共有 rmtree によるデータ消失。

    これは**非 ASCII 依存ではない**ため differential 判定では ``pass`` になる。
    したがって観測値に対して直接 assert する。#375 がヘルパを書き換えたら
    ここが落ちるので、そのとき期待値を反転させること。
    """
    result = run_probe(
        "utils.download_dir_data_loss",
        variant_id="control",
        base_root=nonascii_session["base_root"],
        boundary_id="utils.unicode_safe_download_directory",
        timeout_s=120,
    )
    if result.verdict == Verdict.SKIPPED.value:
        pytest.skip(result.skipped_reason or "probe skipped")

    obs = result.control_observation or result.observation
    assert isinstance(obs, dict), f"観測が取れていない: {result}"
    assert obs["victim_was_redirected_into_downloads"] is True, (
        "download スコープ中の NamedTemporaryFile が downloads/ へリダイレクト"
        "されなくなった。ヘルパの挙動が変わったので棚卸し表 §5 を更新すること。"
    )
    assert obs["victim_survived_scope_exit"] is False, (
        "**データ消失が解消されている。** #375 が unicode_safe_download_directory の"
        "共有 rmtree を修理したなら、この assert を True に反転し、棚卸し表 §5 の"
        "記述を更新すること。"
    )
