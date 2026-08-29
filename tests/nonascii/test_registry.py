"""受け入れ条件の機械化 (Issue #378)。

issue の受け入れ条件を「レビュアが表を読む」から
「**表が不完全 / 陳腐化したら CI が落ちる**」に変える。
プローブは走らせないので既定スイートでも一瞬で終わる。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probes import load_all
from .registry import (
    BOUNDARIES,
    REPO_ROOT,
    STAGING_APIS,
    BoundarySpec,
    Method,
    evidence_rows_for,
    registered_staging_calls,
    resolve_callsite_line,
    scan_staging_calls,
)

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
    いずれかに「**決定**」を持つこと。決定が実測で裏付けられているかは
    別の軸であり、``test_verified_method_requires_runtime_evidence`` が見る。
    """
    assert spec.candidate_method in _VALID_METHODS, (
        f"{spec.boundary_id}: candidate_method が未分類 ({spec.candidate_method!r})"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_verified_method_requires_runtime_evidence(spec: BoundarySpec):
    """**実測で確定した方式は、実測の裏付けが無ければ名乗れない。**

    issue #378 の ② の採用条件は「実測で非 ASCII が通る」なので、
    未実測 / skip / プローブが境界を覆っていない行を ② として数えると
    「未分類ゼロ」が実態より強い保証に見えてしまう (レビュー指摘 2)。
    """
    if spec.verified_method is None:
        return
    assert spec.evidence_kind == "runtime", (
        f"{spec.boundary_id}: verified_method があるのに evidence_kind が "
        f"{spec.evidence_kind!r}"
    )
    assert spec.probe_id, f"{spec.boundary_id}: verified_method があるのに probe_id が無い"
    assert not spec.unmeasured_reason, (
        f"{spec.boundary_id}: 未実測理由があるのに verified_method が設定されている "
        f"({spec.unmeasured_reason})"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_candidate_and_verified_agree(spec: BoundarySpec):
    """実測が候補を否定したまま放置しないこと。

    証拠が候補と食い違ったら、**証拠に従って候補を書き換える**のが本表の規律。
    (実例: file_pipeline の temp_root は当初 ③ を見込んでいたが、実測で
    後段の消費者がすべて wide path と分かり ② へ変更した。)
    """
    if spec.verified_method is None:
        return
    assert spec.verified_method is spec.candidate_method, (
        f"{spec.boundary_id}: candidate={spec.candidate_method.value} だが "
        f"verified={spec.verified_method.value}。証拠に合わせて candidate を更新するか、"
        f"なぜ食い違うのかを rationale に書くこと。"
    )


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_no_unassigned_silent_failure_rows(spec: BoundarySpec):
    """**silent-failure ゼロ** — 黙って壊れる行に「現状維持」を割り当てない。

    ``expected_verdict == "fail_silent"`` の行は ①/③/④ のいずれかで、
    かつ追跡 issue を持たなければならない。② (wide-path = 現状維持で OK) は禁止。
    """
    if spec.expected_verdict != "fail_silent":
        return
    assert spec.candidate_method in {Method.BUFFER, Method.STAGING, Method.FAIL_FAST}, (
        f"{spec.boundary_id}: 黙って壊れると実測されている行に "
        f"{spec.candidate_method.value} (現状維持) が割り当たっている"
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
    if spec.candidate_method is not Method.STAGING:
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


def _latest_results() -> tuple[Path, list[dict]] | None:
    root = REPO_ROOT / "benchmark_results" / "nonascii"
    files = sorted(root.glob("*/results.json"))
    if not files:
        return None
    latest = files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return latest, payload["results"]


def test_verified_rows_match_committed_evidence():
    """**verified_method の主張を、commit 済みの証拠 JSON と突き合わせる。**

    「実測で確定」と書いてあるのに実測レコードが無い / skip されている、
    という食い違いを機械的に検出する。棚卸し表の信頼性はここに掛かっている。
    """
    found = _latest_results()
    if found is None:
        pytest.skip("commit 済みの results.json が無い")
    path, results = found

    problems: list[str] = []
    for spec in BOUNDARIES:
        if spec.verified_method is None:
            continue
        # **boundary_id だけで引かない。** 同じ境界を別 probe で測り直したとき、
        # 古い probe の pass が新しい主張の証拠として通ってしまう (#413 で実証)。
        rows = evidence_rows_for(spec, results)
        if not rows:
            problems.append(
                f"{spec.boundary_id}: verified なのに "
                f"probe_id={spec.probe_id!r} / tier={spec.tier!r} の実測レコードが無い"
            )
            continue
        # **要求 variant が揃っていること。** cjk_kana だけの記録で
        # 「outside_acp まで確かめた」ことにしない。
        missing_variants = [
            v for v in spec.required_variants
            if v not in {r.get("variant") for r in rows}
        ]
        if missing_variants:
            problems.append(
                f"{spec.boundary_id}: verified なのに variant {missing_variants} の"
                f"実測レコードが無い"
            )
            continue
        verdicts = {r["verdict"] for r in rows}
        if "skipped" in verdicts or "error_harness" in verdicts:
            problems.append(
                f"{spec.boundary_id}: verified なのに {sorted(verdicts)} を含む"
            )
            continue
        if spec.verified_method is Method.WIDE_PATH and verdicts != {"pass"}:
            problems.append(
                f"{spec.boundary_id}: ②wide-path と主張しているが実測は {sorted(verdicts)}"
            )
        if spec.verified_method in {Method.STAGING, Method.FAIL_FAST} and verdicts == {"pass"}:
            problems.append(
                f"{spec.boundary_id}: {spec.verified_method.value} と主張しているが実測は全て pass"
            )
    assert not problems, (
        f"registry の verified_method が {path.name} の実測と食い違う:\n  "
        + "\n  ".join(problems)
    )


def test_measurement_caveat_rows_are_not_verified():
    """プローブが境界を覆っていない行は verified を名乗らないこと。

    「実測した」と「境界を実測した」は別である (レビュー指摘 5)。
    """
    offenders = [
        b.boundary_id
        for b in BOUNDARIES
        if not b.covers_boundary and b.verified_method is not None
    ]
    assert not offenders, (
        "プローブが境界そのものを通していないのに verified_method が設定されている: "
        f"{offenders}"
    )

    missing_caveat = [
        b.boundary_id for b in BOUNDARIES if not b.covers_boundary and not b.measurement_caveat
    ]
    assert not missing_caveat, (
        "covers_boundary=False の行は、何をどこまで測ったのかを "
        f"measurement_caveat に書くこと: {missing_caveat}"
    )


def test_unverified_rows_have_a_tracking_home():
    """**未確定行に追跡先を持たせる。**

    本 issue (#378) を閉じると未確定行の追跡先が失われる。そうならないよう、
    「実測で確定」に至っていない applicable 行は、以下のどちらかを必ず持つ:

    - ``followup_issue`` — 追加実測や修正を引き継ぐ issue
    - ``unmeasured_reason`` — **原理上 runtime 実測の対象外**である理由

    非該当 (パス境界でない) 行は分母から外す — 原理上 runtime で確定しようがない。
    """
    applicable = [b for b in BOUNDARIES if b.candidate_method is not Method.NOT_APPLICABLE]
    orphans = [
        b.boundary_id
        for b in applicable
        if b.verified_method is None and not b.followup_issue and not b.unmeasured_reason
    ]
    assert not orphans, (
        "未確定なのに追跡先も「対象外である理由」も無い行がある。"
        "follow-up issue を起票して followup_issue に書くか、"
        f"unmeasured_reason に理由を書くこと:\n  " + f"\n  ".join(orphans)
    )


# --- ASCII staging の実使用と registry の突き合わせ (#375 PR 3) ------------------


@pytest.mark.parametrize("spec", BOUNDARIES, ids=_ids())
def test_staging_metadata_is_self_consistent(spec: BoundarySpec):
    """``staging_api`` / ``staging_purpose`` が対で揃っていること。

    片方だけ埋まっていると、下の双方向突き合わせが「purpose 不一致」として
    落ちるだけで**原因が読めない**。ここで先に形を固定する。
    """
    if spec.staging_api is None:
        assert spec.staging_purpose is None, (
            f"{spec.boundary_id}: staging_api が無いのに staging_purpose がある"
        )
        return
    assert spec.staging_api in STAGING_APIS, (
        f"{spec.boundary_id}: 未知の staging_api {spec.staging_api!r} "
        f"(既知: {STAGING_APIS})"
    )
    assert spec.staging_purpose, (
        f"{spec.boundary_id}: staging_api があるのに staging_purpose が無い。"
        f"**両 API とも purpose を取る**ので、実際に渡している値を書くこと。"
    )


def test_every_staging_call_is_registered():
    """**registry を境界一覧の唯一の SSOT にする** (#375 PR 3 の再レビュー指摘 1)。

    registry -> code の一方向検査だけでは、**registry に無いファイルへ新しい
    ``ascii_safe_*`` 呼び出しを足しても検査対象にならず緑のまま**になる。
    ここでは ``livecap_cli`` を AST で走査した**実使用**と、``staging_api`` を持つ
    registry 行を**双方向で完全一致**させる。

    これにより #379 / #413 が新しい境界を包むときは、**registry へ行を足さない限り
    CI が落ちる** — 棚卸し表と実行時ログの boundary= が構造的にずれなくなる。
    """
    actual = scan_staging_calls()

    dynamic = [
        f"{call.callsite_file}:{call.lineno} ({call.api})"
        for call in actual
        if call.boundary is None or call.purpose is None
    ]
    assert not dynamic, (
        "boundary / purpose が定数リテラルでない ascii_safe_* 呼び出しがある。"
        "registry と突き合わせられないので、定数で渡すこと: " + ", ".join(dynamic)
    )

    actual_keys = {call.key() for call in actual}
    registered_keys = {call.key() for call in registered_staging_calls()}

    unregistered = sorted(actual_keys - registered_keys)
    stale = sorted(registered_keys - actual_keys)

    assert not unregistered, (
        "**registry に無い ascii_safe_* 呼び出しがある。** BoundarySpec を追加し、"
        "staging_api / staging_purpose を設定すること "
        f"(boundary_id は boundary= と同一文字列にする): {unregistered}"
    )
    assert not stale, (
        "**registry が『包んでいる』と主張しているが、コードにその呼び出しが無い。** "
        f"wrapper を外したなら staging_api / staging_purpose も外すこと: {stale}"
    )


def test_stale_probe_evidence_cannot_authorize_verified():
    """**古い probe の結果で `verified_method` を名乗れないこと** (Issue #413)。

    `test_verified_rows_match_committed_evidence` は以前 ``boundary_id`` だけで
    証拠を引いていた。そのため ``engine.*.utterance_wav`` は、**producer 側しか
    測らない ``tempfile.named_temporary_wav`` の pass** を持っているだけで
    ``verified_method=WIDE_PATH`` を名乗れた — consumer を一度も通していないのに、
    である。**新しい実測を一切せずに検査が通る**状態だった。

    ここでは合成した「古い probe の証拠」を使い、照合が probe_id / tier / variant
    まで見ることを固定する。**照合規則を緩めたらここが落ちる。**
    """
    spec = next(s for s in BOUNDARIES if s.boundary_id == "engine.parakeet.utterance_wav")
    assert spec.probe_id and spec.probe_id != "tempfile.named_temporary_wav", (
        "前提: この行は consumer probe を指しているはず"
    )

    stale = [
        {
            "boundary_id": spec.boundary_id,
            "probe_id": "tempfile.named_temporary_wav",   # <- 古い producer probe
            "tier": "cheap",
            "variant": v,
            "verdict": "pass",
        }
        for v in spec.required_variants
    ]
    assert evidence_rows_for(spec, stale) == [], (
        "古い probe_id の証拠が新しい主張の証拠として数えられている"
    )

    # tier だけ違う場合も数えない (同じ probe を別 tier で回した結果の混入)
    wrong_tier = [{**row, "probe_id": spec.probe_id} for row in stale]
    assert evidence_rows_for(spec, wrong_tier) == [], (
        "tier が食い違う証拠が数えられている"
    )

    # 正しい probe_id / tier なら数える (検査が何も通さないだけ、を防ぐ)
    fresh = [{**row, "probe_id": spec.probe_id, "tier": spec.tier} for row in stale]
    assert len(evidence_rows_for(spec, fresh)) == len(spec.required_variants)


def test_required_variants_are_known_ids():
    """``required_variants`` が実在の variant id であること。

    綴り違いは「要求したつもりで何も要求していない」状態になる。
    """
    from .paths import VARIANTS

    known = {v.id for v in VARIANTS}
    for spec in BOUNDARIES:
        unknown = [v for v in spec.required_variants if v not in known]
        assert not unknown, f"{spec.boundary_id}: 未知の variant {unknown}"
