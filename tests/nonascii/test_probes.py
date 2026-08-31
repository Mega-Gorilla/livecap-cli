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
_GPU = [b for b in BOUNDARIES if b.tier == "gpu" and b.probe_id]

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
    # #413: utterance_wav の consumer。probe_id 単位で引くので engine ごとに分けた。
    "asr.utterance_wav.whispers2t": "whispers2t_base",
    "asr.utterance_wav.voxtral": "mistralai--Voxtral-Mini-3B-2507",
    # **marker であってディレクトリではない** (#413 PR C)。重みは models root ではなく
    # 管理 HF cache にあるので、使えるかどうかは _real_model_is_usable が probe 側の
    # qwen3asr_snapshot_dir() へ委譲して確かめる。
    "asr.utterance_wav.qwen3asr": "Qwen--Qwen3-ASR-0.6B.marker",
}

#: heavy tier の boundary_id → models root からの相対パス。
#: probe_id ではなく boundary_id で引くのは、同じプローブを別の .nemo で
#: 走らせる行 (parakeet / canary) があるため。
_HEAVY_SOURCES = {
    "engine.parakeet.nemo_restore_from": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    "engine.canary.nemo_restore_from": "nvidia--canary-1b-flash.nemo",
    "engine.nemo.untar_temp": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    "engine.nemo.restore_path_only": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    # #413: utterance_wav の consumer。**engine 自身に .nemo を読ませる**ので
    # boundary_id 単位で引く (probe は engine 別だが source は同じ形)。
    "engine.parakeet.utterance_wav": "nvidia--parakeet-tdt-0.6b-v2.nemo",
    "engine.canary.utterance_wav": "nvidia--canary-1b-flash.nemo",
}



def _real_model_is_usable(probe_id: str, path: Path) -> bool:
    """候補ディレクトリが**実際に使えるか**まで見る。

    存在するだけで採用すると、先頭候補が不完全 (ダウンロード途中など) のときに
    完全な第 2 候補へ進めない。判定は probe 側の定義を再利用する — ここで
    ファイル名を書くと二重管理になる。
    """
    if probe_id == "asr.utterance_wav.qwen3asr":
        # **source は marker (ファイル) で、重みは別の場所にある。** 他と違って
        # is_dir() では判定できない。marker の存在と、管理 HF cache に snapshot が
        # あることの**両方**を要求する — marker だけを見て「使える」と答えると
        # real_model tier の「ネットワークを使わない」契約を破る。
        from .probes.utterance_wav import qwen3asr_snapshot_dir

        return path.is_file() and qwen3asr_snapshot_dir(_cache_root()) is not None
    if not path.is_dir():
        return False
    if probe_id == "sherpa.from_transducer.real":
        from .probes.native_models import reazon_model_files

        return reazon_model_files(path) is not None
    return True


def _cache_root() -> Path | None:
    """cache root の**素の**既定値 (env 注入前)。``_models_root()`` と同じ規律。"""
    try:
        from livecap_cli.resources import get_resource_configuration

        return Path(get_resource_configuration().cache_root)
    except Exception:
        return None


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


#: env 変数名 -> variant root からの相対 path (runner._child_env と同じ割り当て)。
#: TEMP は TMP / TMPDIR と連動する。
_ROOT_ENV_LAYOUT = {
    "TEMP": "temp",
    "LIVECAP_CORE_CACHE_DIR": "cache",
    "LIVECAP_RESOURCE_ROOT": "resources",
    "HF_HOME": "hf",
}


def _isolation_env(session, spec: BoundarySpec) -> dict[str, str] | None:
    """``ascii_pinned_roots`` の root を ASCII 側へ逃がす env を返す。

    worker は models / cache / resources / %TEMP% / HF_HOME を**すべて** variant root
    へ向ける。そのままでは**複数の境界を同時に非 ASCII にする**ことになり、失敗した
    ときにどれが原因か分からない — #422 は実際にこれで誤帰属しかけた。この行が測りたい
    1 つ以外を ASCII へ固定する。

    **env は worker の起動前に決める必要がある。** ``tempfile.gettempdir()`` や
    huggingface_hub は初回参照で値をキャッシュするので、probe の中で書き換えても
    間に合わない。
    """
    if not spec.ascii_pinned_roots:
        return None
    env: dict[str, str] = {}
    for name in spec.ascii_pinned_roots:
        leaf = _ROOT_ENV_LAYOUT[name]
        target = session["base_root"] / "_ascii_pinned" / spec.boundary_id.replace(".", "_") / leaf
        target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
        if name == "TEMP":
            env["TMP"] = env["TMPDIR"] = str(target)
    return env


def _slow_variants(session, spec: BoundarySpec) -> list[str]:
    """slow tier で回す variant を決める。

    **既定は代表 1 件** (実モデルは重い)。ただし ``required_variants`` を持つ行は
    **その全部を回す** — ``cjk_kana`` の ``ユーザー`` は cp932 の内側なので、consumer が
    narrow path (ACP 変換) で実装されていても**日本語 Windows なら通ってしまう**。
    ACP の外側 (``outside_acp``) まで通して初めて narrow path を排除できる。

    **足りなければ skip ではなく fail する。** 黙って cjk_kana だけで緑になると、
    「両 variant を要求している」という registry の記述が嘘になる。
    """
    variants = session["variants"]
    if spec.required_variants:
        missing = [v for v in spec.required_variants if v not in variants]
        if missing:
            pytest.fail(
                f"{spec.boundary_id}: 必須 variant {missing} をこの FS が受理しない "
                f"(受理: {sorted(variants)} / 除外理由: {session['skipped_variants']})。"
                "**cjk_kana だけで緑にしない** — cp932 の内側なので narrow path を"
                "見逃す。"
            )
        return list(spec.required_variants)
    if not variants:
        return []
    return ["cjk_kana" if "cjk_kana" in variants else variants[0]]


def _finalize_slow_results(results: list, spec: BoundarySpec) -> None:
    """slow tier の全 result をまとめて判定する。**順序が契約である。**

    1. **skip** — probe が動かなかったなら何も言えない (#379: skipped を PASSED と
       数えると「ゲートは緑だが対象経路を通っていない」)
    2. **harness health** — control が失敗しているなら境界のバグではない
    3. **control の安定性** — control 観測が variant を跨いでずれたなら、それは
       **path と無関係な非決定性**である
    4. **expected verdict** — ここまで通って初めて境界の判定を評価する

    **3 を 4 より先に置くのが要点である。** 逆順だと、非決定性で trial != control に
    なったときに ``fail_silent`` の assertion がその場で止まり、安定性検査へ到達
    しない。しかも証拠には ``fail_silent`` が残り、「非決定性は error_harness と
    する」という契約と食い違う (レビュー指摘)。
    """
    for result in results:
        if result.verdict == Verdict.SKIPPED.value:
            pytest.skip(f"{spec.boundary_id}: probe skipped - {result.skipped_reason}")
    for result in results:
        _assert_harness_healthy(result, spec)

    observed = [
        (r.variant, r.control_observation)
        for r in results
        if r.control_observation is not None
    ]
    if len(observed) >= 2:
        first_variant, first = observed[0]
        drift = [(v, o) for v, o in observed[1:] if o != first]
        if drift:
            # **証拠にも error_harness を残す。** assertion だけ変えても、
            # results.json には fail_silent が記録されてしまう。
            for r in results:
                r.verdict = Verdict.ERROR_HARNESS.value
                r.notes = (
                    "ASCII control の観測が variant を跨いで一致しない - "
                    "path と無関係な非決定性"
                )
            pytest.fail(
                f"{spec.boundary_id}: ASCII control の観測が variant を跨いで"
                f"一致しない ({first_variant}={first!r} vs {drift!r})。**path と"
                "無関係な非決定性**であり、境界のバグではない。seed / decoding を"
                "固定するか、この engine では fingerprint 比較を使わない判定へ"
                "変えること。"
            )

    for result in results:
        _assert_expected_verdict(result, spec)


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

    variant_ids = _slow_variants(nonascii_session, spec)
    if not variant_ids:
        pytest.skip("非 ASCII variant を受理しない FS")

    results = []
    for variant_id in variant_ids:
        result = _execute(
            nonascii_session,
            spec,
            variant_id,
            timeout_s=900,
            payload={
                "model_source": str(source),
                "models_root": str(models_root),
                # qwen3asr の重みだけは models root ではなく管理 HF cache 側にある
                # (#413 PR C)。heavy tier (parakeet / canary) は models root から
                # .nemo を読むので不要。
                "cache_root": str(_cache_root() or ""),
            },
            env_extra=_isolation_env(nonascii_session, spec),
        )
        results.append(result)
    # **判定はまとめて行う** (順序が契約 — _finalize_slow_results 参照)。
    # heavy tier には #379 で skip 伝播を入れたが real_model には無かった。
    _finalize_slow_results(results, spec)


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

    variant_ids = _slow_variants(nonascii_session, spec)
    if not variant_ids:
        variant_ids = ["cjk_kana"]

    # どの root を ASCII へ逃がすかは spec が持つ (ascii_pinned_roots)。
    env_extra = _isolation_env(nonascii_session, spec)

    results = []
    for variant_id in variant_ids:
        result = _execute(
            nonascii_session,
            spec,
            variant_id,
            timeout_s=1800,
            payload={"model_source": str(source), "models_root": str(models_root)},
            env_extra=env_extra,
        )
        results.append(result)
    # **判定はまとめて行う** (順序が契約 — _finalize_slow_results 参照)。
    _finalize_slow_results(results, spec)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("spec", _GPU, ids=_ids(_GPU))
def test_gpu_boundary(nonascii_session, spec: BoundarySpec):
    """gpu tier: **CUDA は要るがモデルは要らない** (Issue #422)。

    real_model / heavy と分けているのは、それらが実モデルの所在 / NeMo を要求し、
    見つからなければ **黙って skip する**ためである。この境界はどちらも要らないので、
    混ぜると「CUDA があるのに測っていない」状態が緑で通る。

    CUDA が無い環境では probe 側が ``ProbeSkipped`` を投げ、理由付きで記録される。
    """
    variant_ids = _slow_variants(nonascii_session, spec)
    if not variant_ids:
        pytest.skip("非 ASCII variant を受理しない FS")

    results = []
    for variant_id in variant_ids:
        result = _execute(
            nonascii_session,
            spec,
            variant_id,
            timeout_s=300,
            env_extra=_isolation_env(nonascii_session, spec),
        )
        results.append(result)
    # **判定はまとめて行う** (順序が契約 — _finalize_slow_results 参照)。
    _finalize_slow_results(results, spec)


# =============================================================================
# slow tier の判定順序 (Issue #413 PR A のレビュー指摘 3)
#
# **実モデル不要。** 合成した ProbeResult で順序そのものを固定する。
# =============================================================================


def _synthetic(variant: str, verdict: str, control_obs, **kw):
    from .record import EvidenceKind, ProbeResult, Tier

    return ProbeResult(
        boundary_id="engine.parakeet.utterance_wav",
        probe_id="asr.utterance_wav.parakeet",
        variant=variant,
        apply_to="dir",
        tier=Tier.HEAVY.value,
        evidence_kind=EvidenceKind.RUNTIME.value,
        verdict=verdict,
        control_observation=control_obs,
        **kw,
    )


class TestSlowResultFinalizationOrder:
    """``_finalize_slow_results`` の順序契約 (skip -> harness -> control -> verdict)。"""

    @property
    def _spec(self) -> BoundarySpec:
        return next(
            s for s in BOUNDARIES if s.boundary_id == "engine.parakeet.utterance_wav"
        )

    def test_all_pass_with_stable_control(self) -> None:
        obs = {"text_sha256": "a", "text_is_nonempty": True, "text_char_count": 3}
        _finalize_slow_results(
            [_synthetic("cjk_kana", "pass", obs), _synthetic("outside_acp", "pass", obs)],
            self._spec,
        )

    def test_control_drift_is_error_harness_not_fail_silent(self) -> None:
        """**非決定性は境界のバグではない。**

        逆順 (expected verdict を先に評価) だと fail_silent の assertion がその場で
        止まり、ここへ到達しない。**証拠にも error_harness が残る**ことまで見る。
        """
        results = [
            _synthetic("cjk_kana", "pass", {"text_sha256": "a"}),
            # control がずれた -> trial との比較で fail_silent になっている
            _synthetic("outside_acp", "fail_silent", {"text_sha256": "b"}),
        ]

        # pytest.fail / skip は BaseException 派生なので Exception では捕まらない
        with pytest.raises(pytest.fail.Exception) as excinfo:
            _finalize_slow_results(results, self._spec)

        assert "非決定性" in str(excinfo.value)
        assert all(r.verdict == Verdict.ERROR_HARNESS.value for r in results), (
            "assertion だけ変えても results.json には fail_silent が残ってしまう"
        )

    def test_regression_is_reported_when_control_is_stable(self) -> None:
        """**陰性対照** — 境界が壊れたら、ちゃんと失敗すること (#413 PR B)。

        他の 3 件は「順序」と「優先度」を見ており、**回帰そのものを捕まえる経路には
        テストが無かった**。ここが緑のまま `_assert_expected_verdict` の条件が壊れると、
        依存更新で wide-path が失われても**証拠には fail_silent が残るのにテストは通る**
        という最悪の形になる。

        control が variant を跨いで安定している (= 非決定性ではない) 以上、
        trial の差は**境界のバグ**として報告されなければならない。
        """
        obs = {"text_sha256": "a", "text_is_nonempty": True, "text_char_count": 3}
        results = [
            _synthetic("cjk_kana", "pass", obs),
            # control は同じ = 揺れではない。trial だけが崩れた。
            _synthetic("outside_acp", "fail_silent", obs),
        ]

        with pytest.raises(AssertionError) as excinfo:
            _finalize_slow_results(results, self._spec)

        message = str(excinfo.value)
        assert "fail_silent" in message, "実際の verdict を出すこと"
        assert self._spec.boundary_id in message, "どの境界が壊れたのか名指しすること"
        assert all(r.verdict == "fail_silent" for r in results if r.variant == "outside_acp"), (
            "error_harness へ書き換えてはならない - これは harness ではなく境界の問題"
        )

    def test_skip_wins_over_everything(self) -> None:
        """probe が動かなかったなら、**expected verdict を評価してはならない**。"""
        results = [
            _synthetic("cjk_kana", Verdict.SKIPPED.value, None, skipped_reason="no model"),
            _synthetic("outside_acp", "fail_silent", {"text_sha256": "b"}),
        ]

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _finalize_slow_results(results, self._spec)

        assert "probe skipped" in str(excinfo.value)

    def test_harness_error_is_reported_before_verdict(self) -> None:
        """control が失敗しているなら境界のバグではない。"""
        results = [_synthetic("cjk_kana", Verdict.ERROR_HARNESS.value, None)]

        with pytest.raises(AssertionError) as excinfo:
            _finalize_slow_results(results, self._spec)

        assert "control" in str(excinfo.value)
