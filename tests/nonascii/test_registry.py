"""受け入れ条件の機械化 (Issue #378)。

issue の受け入れ条件を「レビュアが表を読む」から
「**表が不完全 / 陳腐化したら CI が落ちる**」に変える。
プローブは走らせないので既定スイートでも一瞬で終わる。
"""

from __future__ import annotations

import pytest

from .probes import load_all
from .registry import BOUNDARIES, BoundarySpec, Method, resolve_callsite_line

pytestmark = pytest.mark.nonascii_paths

_VALID_METHODS = set(Method)
_VALID_EVIDENCE = {"runtime", "source_check", "not_applicable"}
_VALID_TIERS = {"cheap", "real_model", "heavy", "network", "none"}


def _ids() -> list[str]:
    return [b.boundary_id for b in BOUNDARIES]


def test_boundary_ids_are_unique():
    ids = [b.boundary_id for b in BOUNDARIES]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"boundary_id が重複している: {sorted(duplicates)}"


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_no_unclassified_rows(spec: BoundarySpec):
    """**未分類ゼロ** — issue #378 の完了条件。

    全行が ①buffer / ②wide-path / ③staging / ④fail-fast / 非該当 の
    いずれかに割り当たっていること。
    """
    assert spec.adopted_method in _VALID_METHODS, (
        f"{spec.boundary_id}: adopted_method が未分類 ({spec.adopted_method!r})"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_no_unassigned_silent_failure_rows(spec: BoundarySpec):
    """**silent-failure ゼロ** — 黙って壊れる行に「現状維持」を割り当てない。

    ``expected_verdict == "fail_silent"`` の行は ①/③/④ のいずれかで、
    かつ追跡 issue を持たなければならない。② (wide-path = 現状維持で OK) は禁止。
    """
    if spec.expected_verdict != "fail_silent":
        return
    assert spec.adopted_method in {Method.BUFFER, Method.STAGING, Method.FAIL_FAST}, (
        f"{spec.boundary_id}: 黙って壊れると実測されている行に "
        f"{spec.adopted_method.value} (現状維持) が割り当たっている"
    )
    assert spec.followup_issue, (
        f"{spec.boundary_id}: 黙って壊れる行に追跡 issue が無い"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_callsites_exist(spec: BoundarySpec):
    """表がコードとずれていないこと。

    #375 / #379 / #377 が実装でコードを動かしたとき、棚卸し表が黙って
    腐るのを防ぐ。行番号ではなく symbol で追跡しているので、行が動くだけでは
    落ちない (シンボルが消えたときだけ落ちる)。
    """
    line = resolve_callsite_line(spec)
    assert line is not None, (
        f"{spec.boundary_id}: {spec.callsite_file} に "
        f"{spec.callsite_symbol!r} が見つからない。"
        f"コードが動いたなら registry の callsite_symbol を更新すること。"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_every_row_has_evidence(spec: BoundarySpec):
    """全行が証拠の種別を持ち、runtime 行は実在の probe を指していること。"""
    assert spec.evidence_kind in _VALID_EVIDENCE, (
        f"{spec.boundary_id}: 不正な evidence_kind {spec.evidence_kind!r}"
    )
    assert spec.tier in _VALID_TIERS, f"{spec.boundary_id}: 不正な tier {spec.tier!r}"

    if spec.evidence_kind == "runtime":
        assert spec.probe_id, f"{spec.boundary_id}: runtime 行に probe_id が無い"
        impls = load_all()
        assert spec.probe_id in impls, (
            f"{spec.boundary_id}: probe_id {spec.probe_id!r} が未実装"
        )
    else:
        assert spec.rationale.strip(), (
            f"{spec.boundary_id}: {spec.evidence_kind} 行には文章化された根拠が必須"
        )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_unmeasured_rows_state_why(spec: BoundarySpec):
    """未実測の行は理由を明記していること。

    「試していない」と「試したら通った」が表の上で混同されないようにする。
    """
    if spec.evidence_kind == "runtime" and spec.tier == "cheap":
        return
    if spec.evidence_kind == "not_applicable":
        return
    if spec.tier in {"heavy", "none"} or spec.unmeasured_reason:
        assert spec.unmeasured_reason or spec.rationale, (
            f"{spec.boundary_id}: 未実測なのに理由が書かれていない"
        )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_staging_rows_have_granularity(spec: BoundarySpec):
    """③staging の行は粒度 (file / dir / %TEMP%) が決まっていること。

    #375 の ``ascii_safe_path()`` は粒度によって使う機構が変わる
    (``os.link`` はファイル専用、junction はディレクトリ専用)。
    """
    if spec.adopted_method is not Method.STAGING:
        return
    assert spec.granularity in {"file", "dir", "%TEMP%"}, (
        f"{spec.boundary_id}: staging 行の粒度が未決定 ({spec.granularity!r})"
    )


def test_probe_ids_are_all_referenced():
    """実装済みだが registry から参照されていない probe を検出する。

    selftest 系は registry に載らないので除外する。
    """
    impls = load_all()
    referenced = {b.probe_id for b in BOUNDARIES if b.probe_id}
    orphans = sorted(
        pid for pid in impls if pid not in referenced and not pid.startswith("selftest.")
    )
    assert not orphans, f"registry から参照されていない probe: {orphans}"
