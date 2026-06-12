"""Unit tests for ``CanaryEngine.__init__`` signature.

Issue #321 PR #1 (MEDIUM-1): PR-A.4.2 で削除した ``beam_size`` parameter は、
旧版では ``**kwargs`` で warn-then-swallow されていたが、本 cleanup で
``**kwargs`` 自体を削除し explicit signature に変更。caller が legacy
``beam_size=N`` を渡したら fail-fast (TypeError) になることを pin する。
"""
from __future__ import annotations

import pytest


class TestCanaryInitSignature:
    def test_init_rejects_beam_size_kwarg(self) -> None:
        from livecap_cli.engines.canary_engine import CanaryEngine

        with pytest.raises(TypeError, match="beam_size"):
            CanaryEngine(device="cpu", beam_size=3)

    def test_init_rejects_arbitrary_unknown_kwarg(self) -> None:
        from livecap_cli.engines.canary_engine import CanaryEngine

        with pytest.raises(TypeError):
            CanaryEngine(device="cpu", nonexistent_param="x")

    def test_init_accepts_documented_params(self) -> None:
        from livecap_cli.engines.canary_engine import CanaryEngine

        engine = CanaryEngine(
            device="cpu",
            language="en",
            model_name="nvidia/canary-1b-flash",
        )
        assert engine.language == "en"
        assert engine.model_name == "nvidia/canary-1b-flash"
        assert engine.engine_name == "canary"
