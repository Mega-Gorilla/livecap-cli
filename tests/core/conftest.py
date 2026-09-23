"""tests/core 共通の fixture 登録。実体は ``tests/core/model_root_fixtures.py``。"""

import pytest

from livecap_cli.engines import legacy_model_layouts
from tests.core.model_root_fixtures import model_root_sentinels  # noqa: F401


@pytest.fixture(autouse=True)
def _pin_external_model_caches(tmp_path, monkeypatch):
    """root の外の旧 cache (既定 HF cache / whisper_s2t の自前 cache、#453) を**存在しない** tmp へ pin する。

    unit test が開発機 / CI runner の実 ``~/.cache/huggingface/hub`` を読むと、そこにある snapshot
    次第で cold load が「取り込み」になったり ``livecap-cli info`` の出力が変わったりする。
    外の cache を扱うテストは ``model_root_sentinels`` (``external_hub`` / ``external_whisper``) か
    自前の monkeypatch で上書きする。
    """
    unused = tmp_path / "no-external-model-caches"
    monkeypatch.setattr(
        legacy_model_layouts,
        "external_hub_roots",
        lambda: [legacy_model_layouts.ExternalCacheRoot("default HF cache", unused / "hf-hub")],
    )

@pytest.fixture(autouse=True)
def _no_background_library_preload(monkeypatch):
    """``LibraryPreloader`` の daemon thread を core の unit test では起こさない。

    engine の **constructor** が ``LibraryPreloader.start_preloading(...)`` を呼び、daemon thread が
    ``import nemo.collections.asr`` / ``import transformers`` を**実物で**実行する。一方 unit test は
    ``sys.modules`` に stub を挿すので、両者が同時に走ると import machinery が壊れた中間状態を見て

        ModuleNotFoundError: No module named 'nemo.collections.asr'; 'nemo.collections' is not a package

    で落ちる (本 session で full suite 実行中に 1 度だけ再現。thread の timing 次第なので、テストを
    足すだけで当たり方が変わる)。preload は**先読みだけ**の最適化で、engine は load 時に必要な module を
    自分で import するため、無効化しても unit test の網羅は変わらない。
    実 engine を動かす ``tests/integration`` はこの conftest の外なので従来どおり preload する。
    """
    from livecap_cli.engines.library_preloader import LibraryPreloader

    # production の off-switch (``LibraryPreloader.enable(False)`` が立てる flag) を使う。
    # `start_preloading` 自体を差し替えると、その早期 return の分岐がテストから消える
    monkeypatch.setattr(LibraryPreloader, "_enabled", False)
    yield
    thread = LibraryPreloader._preload_thread
    if thread is not None and thread.is_alive():  # pragma: no cover - 取りこぼしの保険
        thread.join(timeout=60)
