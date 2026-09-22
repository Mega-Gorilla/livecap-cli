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
