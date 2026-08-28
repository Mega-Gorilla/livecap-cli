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
from pathlib import Path

import pytest

from .conftest import real_models_enabled
from .record import Verdict
from .registry import BOUNDARIES, BoundarySpec
from .runner import run_probe

pytestmark = pytest.mark.nonascii_paths

_CHEAP = [b for b in BOUNDARIES if b.tier == "cheap" and b.probe_id]
_REAL_MODEL = [b for b in BOUNDARIES if b.tier == "real_model" and b.probe_id]
_HEAVY = [b for b in BOUNDARIES if b.tier == "heavy" and b.probe_id]

#: real_model tier の probe_id → models root からの相対パス。
#:
#: **tuple を書くと「最初に存在したものを使う」**。CI ランナーにどの variant が
#: 温まっているかは workflow 側の都合で変わるため、1 つに固定すると probe が
#: 黙って skip し、**緑のままゲートだけが失効する** (#377 で実際に起きた)。
_REAL_MODEL_SOURCES = {
    # int8 (154 MB encoder) を優先する — float32 は 592 MB あり、測定内容は同じ。
    # 無ければ float32 で成立させる。
    "sherpa.from_transducer.real": (
        "reazonspeech/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
        "reazonspeech/reazon-research--reazonspeech-k2-v2",
    ),
    "voxtral.from_pretrained": "mistralai--Voxtral-Mini-3B-2507",
    "voxtral.autoprocessor": "mistralai--Voxtral-Mini-3B-2507",
    "transformers.autoconfig.local_dir": "mistralai--Voxtral-Mini-3B-2507",
}

#: heavy tier の boundary_id → models root からの相対パス。
#: probe_id ではなく boundary_id で引くのは、同じプローブを別の .nemo で
#: 走らせる行 (parakeet / canary) があるため。
_HEAVY_SOURCES = {
    "engine.parakeet.nemo_restore_from": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    "engine.canary.nemo_restore_from": "nvidia--canary-1b-flash.nemo",
    "engine.nemo.untar_temp": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    "engine.nemo.restore_path_only": "nvidia--parakeet-tdt-0.6b-v2.nemo",
}

#: ``.nemo`` パスだけを非 ASCII にしたい行では、``%TEMP%`` を ASCII 側へ固定する。
#: そうしないと 2 つの副境界が同時に非 ASCII になり、主因を切り分けられない。
_HEAVY_ASCII_TEMP_BOUNDARIES = {"engine.nemo.restore_path_only"}


def _real_model_is_usable(probe_id: str, path: Path) -> bool:
    """候補ディレクトリが**実際に使えるか**まで見る。

    存在するだけで採用すると、先頭候補が不完全 (ダウンロード途中など) のときに
    完全な第 2 候補へ進めない。判定は probe 側の定義を再利用する — ここで
    ファイル名を書くと二重管理になる。
    """
    if not path.is_dir():
        return False
    if probe_id == "sherpa.from_transducer.real":
        from .probes.native_models import reazon_model_files

        return reazon_model_files(path) is not None
    return True


def _ids(specs: list[BoundarySpec]) -> list[str]:
    return [s.boundary_id for s in specs]


def _execute(
    session,
    spec: BoundarySpec,
    variant_id: str,
    *,
    timeout_s: float,
    payload=None,
    env_extra=None,
):
    result = run_probe(
        spec.probe_id,
        variant_id=variant_id,
        base_root=session["base_root"],
        boundary_id=spec.boundary_id,
        payload=payload,
        timeout_s=timeout_s,
        env_extra=env_extra,
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

    candidates = (relative,) if isinstance(relative, str) else tuple(relative)
    source = next(
        (models_root / c for c in candidates if _real_model_is_usable(spec.probe_id, models_root / c)),
        None,
    )
    if source is None:
        pytest.skip(f"実モデルが存在しない: {' / '.join(candidates)}")

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
    relative = _HEAVY_SOURCES.get(spec.boundary_id)
    if models_root is None or relative is None:
        pytest.skip(f"{spec.boundary_id} の実モデル所在が未定義")
    source = models_root / relative
    if source.is_dir():
        # 実環境では ``<name>.nemo/<name>.nemo`` と入れ子になっていることがある
        # (engine の relocation 由来)。同名ファイルがあればそれを使う。
        nested = source / source.name
        if nested.is_file():
            source = nested
    if not source.is_file():
        pytest.skip(f".nemo が存在しない: {relative}")

    variants = nonascii_session["variants"]
    variant_id = "cjk_kana" if "cjk_kana" in variants else (variants or ["cjk_kana"])[0]

    env_extra = None
    if spec.boundary_id in _HEAVY_ASCII_TEMP_BOUNDARIES:
        # %TEMP% を ASCII 側へ固定し、非 ASCII なのは .nemo のパスだけにする
        ascii_temp = nonascii_session["base_root"] / "_ascii_temp" / spec.boundary_id.replace(".", "_")
        ascii_temp.mkdir(parents=True, exist_ok=True)
        env_extra = {"TEMP": str(ascii_temp), "TMP": str(ascii_temp), "TMPDIR": str(ascii_temp)}

    result = _execute(
        nonascii_session,
        spec,
        variant_id,
        timeout_s=1800,
        payload={"model_source": str(source)},
        env_extra=env_extra,
    )
    # **probe が skip したら test も skip する。** ``_assert_expected_verdict`` は
    # skipped を早期 return するので、そのままだと**probe が動かなくても PASSED**
    # になる。CI はこの PASSED をゲートに使っているため、それでは
    # 「ゲートは緑だが対象経路を通っていない」状態になる (#379 で実際に踏んだ —
    # runner の lightning が新しすぎて NeMo が import できず、heavy 行が全部
    # 素通りしていた)。
    if result.verdict == Verdict.SKIPPED.value:
        pytest.skip(f"{spec.boundary_id}: probe skipped - {result.skipped_reason}")
    _assert_harness_healthy(result, spec)
    _assert_expected_verdict(result, spec)
