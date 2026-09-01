"""ハーネスの pytest 統合 (Issue #378)。

**親プロセスの ``os.environ`` / ``tempfile.tempdir`` は絶対に触らない。**
env 注入は ``runner`` が ``subprocess.run(env=...)`` で行う。親を汚すと、
まさに測定対象である ``livecap_cli/utils/__init__.py`` の欠陥
(ロック無し・refcount 無しの env 書き換え) をハーネス内で再現してしまう。
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .paths import DEFAULT_VARIANT_IDS, eight_dot_three_state, supported_variants
from .record import ProbeResult, RunMetadata, write_results
from .registry import REPO_ROOT
from .roots import (
    create_session_root,
    reap_stale_sessions,
    release_session_root,
    resolve_base_root,
)

ENV_ROOT = "LIVECAP_NONASCII_ROOT"
ENV_REAL_MODELS = "LIVECAP_NONASCII_REAL_MODELS"


def pytest_addoption(parser):
    group = parser.getgroup("nonascii")
    group.addoption(
        "--nonascii-report",
        action="store",
        default=None,
        metavar="PATH",
        help="非 ASCII パス境界プローブの結果 JSON をこのパスへ書き出す (Issue #378)",
    )
    group.addoption(
        "--nonascii-variants",
        action="store",
        default=None,
        metavar="IDS",
        help="カンマ区切りの variant id。既定は paths.DEFAULT_VARIANT_IDS",
    )


def real_models_enabled() -> bool:
    return os.environ.get(ENV_REAL_MODELS, "").strip().lower() in {"1", "true", "yes"}


def _models_root() -> Path | None:
    """実モデルの所在。env 注入前の**素の**既定値を見る必要がある。"""
    try:
        from livecap_cli.resources import get_resource_configuration

        # preview を使う (Issue #375)。freeze しないので、この照会がハーネス側の
        # env 注入を先取りして固めてしまうことがない。**directory も作らない** —
        # 以前の ``ModelManager()`` は構築するだけで root を実体化していた。
        return Path(get_resource_configuration().models_root)
    except Exception:
        return None


def _pick_base_root() -> tuple[Path, str, list[tuple[str, str]]]:
    """(ASCII 保証された base root, 採用候補ラベル, 落ちた候補と理由)。

    **real-model tier の有無に関わらず** ASCII かつ書き込み可能な候補を探索する。
    システム ``%TEMP%`` へ無条件に落とすと、**まさに検証したい環境**
    (Windows ユーザー名が非 ASCII) で base root が非 ASCII になり、session ごと
    skip されてしまうため (レビュー指摘 1)。候補列は ``roots.py`` を参照。
    """
    return resolve_base_root(
        override=os.environ.get(ENV_ROOT),
        models_root=_models_root(),
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="session")
def nonascii_session(request) -> dict:
    """base root の確保、variant の対応判定、run メタデータの構築。"""
    override = os.environ.get(ENV_ROOT)
    try:
        parent_root, root_label, rejected_roots = _pick_base_root()
    except RuntimeError as exc:
        # **skip にしない。** cheap tier は既定スイートに載せている以上、
        # 「green = 実際に測った」でなければ意味がない。root が確保できない状態を
        # skip で流すと、LIVECAP_NONASCII_ROOT の typo や権限不足が
        # CI green のまま未測定になる (レビュー指摘 2)。
        hint = (
            "LIVECAP_NONASCII_ROOT の指定を見直すこと。"
            if override
            else "LIVECAP_NONASCII_ROOT で ASCII かつ書き込み可能なディレクトリを指定すること。"
        )
        pytest.fail(f"非 ASCII プローブ用の base root を確保できない: {exc} {hint}", pytrace=False)

    # 異常終了した過去 run の残骸を掃除する (生存中の run には触れない)。
    reaped = reap_stale_sessions(parent_root)

    # **この run 専用の root。** 固定 root を共有すると、並行 run が同じ probe パスを
    # 読み書きし、片方の teardown がもう片方の実行中データを消してしまう。
    base_root = create_session_root(parent_root)

    raw = request.config.getoption("--nonascii-variants")
    variant_ids = (
        tuple(v.strip() for v in raw.split(",") if v.strip())
        if raw
        else DEFAULT_VARIANT_IDS
    )
    ok, skipped = supported_variants(base_root, variant_ids)
    if not [v for v in ok if v != "control"]:
        # 非 ASCII variant が 1 つも通らない = cheap tier が何も測っていない。
        # これを skip で流すと green が「測った」を意味しなくなる。
        pytest.fail(
            "この filesystem は非 ASCII variant を 1 つも受理しない: "
            + " / ".join(f"{k}: {v}" for k, v in skipped.items()),
            pytrace=False,
        )

    # 正規化保存の可否 (macOS APFS 等で NFD が保たれないケースの検出)
    normalization_preserved = "nfd" not in skipped if "nfd" in variant_ids else None

    run = RunMetadata(
        run_id=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        measured_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        nonascii_root=str(base_root),
        root_parent=str(parent_root),
        root_label=root_label,
        rejected_roots=dict(rejected_roots),
        reaped_stale_sessions=list(reaped),
        root_volume=str(Path(base_root).anchor),
        eight_dot_three_state=eight_dot_three_state(str(Path(base_root).anchor)),
        variants_supported=list(ok),
        variants_skipped=dict(skipped),
        normalization_preserved=normalization_preserved,
    )

    state = {
        "base_root": base_root,
        "variants": [v for v in ok if v != "control"],
        "skipped_variants": skipped,
        "run": run,
        "results": [],
        "models_root": _models_root(),
    }
    yield state

    results: list[ProbeResult] = state["results"]
    report_path = request.config.getoption("--nonascii-report")
    if report_path and results:
        run.leftover_paths = _leftovers(base_root)
        run.materialization = _materialization(results)
        run.tiers_enabled = _tiers_from_results(results)
        write_results(Path(report_path), run, results)

    # **消すのはこの run の session root だけ。** 共有される親 (parent_root) には
    # 他の run の session root が並んでいる可能性があるので絶対に触らない —
    # 共有ディレクトリを rmtree する utils/__init__.py の欠陥を、ハーネス自身が
    # 繰り返さないため。ONNX の mmap 等で削除できないものは leftover として
    # 記録済みであり、cleanup 失敗で run を落とさない。
    # 使用中ロックを先に手放す。Windows では握ったままだと自分自身の rmtree も失敗する。
    release_session_root(base_root)
    if os.environ.get("LIVECAP_NONASCII_KEEP") not in {"1", "true", "yes"}:
        shutil.rmtree(base_root, ignore_errors=True)


def _tiers_from_results(results: "list[ProbeResult]") -> list[str]:
    """証拠 JSON の ``tiers_enabled``。**宣言ではなく実績から書く。**

    宣言 (``LIVECAP_NONASCII_REAL_MODELS`` の有無) から導くと嘘になる —
    heavy は ``importorskip("nemo")``、gpu は CUDA の有無でしか gate されず、
    **この env とは無関係に走る**。証拠 JSON が「この tier は走っていない」と
    主張したまま記録だけ入る状態を作らない。

    契約:

    - **1 件でも記録がある tier を挙げる。** ``verdict`` は問わない — `skipped` の
      記録も「その tier の node を回そうとした」証拠であり、なぜ測れなかったかは
      レコード側の ``skipped_reason`` に残る。したがってこの欄の意味は
      「**実行を試みた tier**」である
    - **記録が 1 件も無い tier は挙げない。** ``pytest.skip`` は ``_execute`` の前に
      抜けるので、丸ごと skip された tier はそもそもレコードを持たない
    - 重複を除き、安定した順序 (辞書順) で返す
    """
    return sorted({r.tier for r in results})


def _leftovers(base_root: Path) -> list[str]:
    if not base_root.exists():
        return []
    return [
        str(p.relative_to(base_root))
        for p in list(base_root.rglob("*"))[:50]
        if p.is_file()
    ][:20]


def _materialization(results: list[ProbeResult]) -> str:
    for r in results:
        obs = r.observation
        if isinstance(obs, dict) and obs.get("materialization"):
            return str(obs["materialization"])
    return "n/a"
