"""ハーネスの pytest 統合 (Issue #378)。

**親プロセスの ``os.environ`` / ``tempfile.tempdir`` は絶対に触らない。**
env 注入は ``runner`` が ``subprocess.run(env=...)`` で行う。親を汚すと、
まさに測定対象である ``livecap_cli/utils/__init__.py`` の欠陥
(ロック無し・refcount 無しの env 書き換え) をハーネス内で再現してしまう。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from .paths import DEFAULT_VARIANT_IDS, eight_dot_three_state, supported_variants
from .record import ProbeResult, RunMetadata, write_results

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
        from livecap_cli.resources.model_manager import ModelManager

        return Path(ModelManager().models_root)
    except Exception:
        return None


def _pick_base_root() -> tuple[Path, bool]:
    """(ASCII 保証された base root, 一時ディレクトリか) を返す。

    実モデル tier では ``os.link`` を効かせるためモデルと同一ボリュームに置く。
    hardlink が使えれば 740 MB のモデルでも 0 バイト・ミリ秒で実体化できる。
    """
    override = os.environ.get(ENV_ROOT)
    if override:
        root = Path(override)
        root.mkdir(parents=True, exist_ok=True)
        return root, False

    if real_models_enabled():
        models = _models_root()
        if models is not None:
            volume = Path(models.anchor)
            candidate = volume / "livecap-nonascii-probe"
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write-probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                if str(candidate).isascii():
                    return candidate, False
            except OSError:
                pass  # 権限が無ければ temp へ降格 (COPY 経路になる)

    return Path(tempfile.mkdtemp(prefix="livecap-nonascii-")), True


@pytest.fixture(scope="session")
def nonascii_session(request) -> dict:
    """base root の確保、variant の対応判定、run メタデータの構築。"""
    base_root, is_temp = _pick_base_root()
    if not str(base_root).isascii():
        pytest.skip(f"base root が非 ASCII のため variant を分離できない: {base_root!r}")

    raw = request.config.getoption("--nonascii-variants")
    variant_ids = (
        tuple(v.strip() for v in raw.split(",") if v.strip())
        if raw
        else DEFAULT_VARIANT_IDS
    )
    ok, skipped = supported_variants(base_root, variant_ids)

    # 正規化保存の可否 (macOS APFS 等で NFD が保たれないケースの検出)
    normalization_preserved = "nfd" not in skipped if "nfd" in variant_ids else None

    run = RunMetadata(
        run_id=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        measured_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        nonascii_root=str(base_root),
        root_volume=str(Path(base_root).anchor),
        eight_dot_three_state=eight_dot_three_state(str(Path(base_root).anchor)),
        tiers_enabled=["cheap"] + (["real_model"] if real_models_enabled() else []),
        variants_supported=list(ok),
        variants_skipped=dict(skipped),
        normalization_preserved=normalization_preserved,
    )

    state = {
        "base_root": base_root,
        "is_temp": is_temp,
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
        write_results(Path(report_path), run, results)

    if is_temp:
        # ONNX の mmap 等で削除できないものは leftover として記録済み。
        # cleanup 失敗で run を落とさない。
        shutil.rmtree(base_root, ignore_errors=True)


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


@pytest.fixture
def record_probe_result(nonascii_session):
    """結果をセッションに集約する (``--nonascii-report`` で JSON 化される)。"""

    def _record(result: ProbeResult) -> ProbeResult:
        nonascii_session["results"].append(result)
        return result

    return _record
