"""Issue #96: VRAM 事前チェック (warn-default + strict opt-in) を pin。

実 GPU / model load は不要。 ``get_available_vram`` / ``detect_device`` を
monkeypatch して各分岐 (warn / strict / 多段 fail-open) を検証する。
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from livecap_cli.engines.model_vram import (
    InsufficientVRAMError,
    check_vram_before_load,
    get_vram_requirement_mb,
)


def _patch_vram(monkeypatch, *, available_mb, resolved_device="cuda"):
    """``check_vram_before_load`` が import する util を差し替える。"""
    monkeypatch.setattr(
        "livecap_cli.utils.get_available_vram", lambda: available_mb
    )
    monkeypatch.setattr(
        "livecap_cli.utils.detect_device", lambda device, engine_name: resolved_device
    )


class TestGetVramRequirement:
    def test_size_specific_key(self):
        assert get_vram_requirement_mb("whispers2t", "large-v3") == 10000
        assert get_vram_requirement_mb("whispers2t", "base") == 2000

    def test_engine_unit_key(self):
        assert get_vram_requirement_mb("voxtral") == 9500
        assert get_vram_requirement_mb("voxtral", None) == 9500

    def test_unknown_size_falls_back_to_engine(self):
        # parakeet は engine 単位 key を持つので size 未登録でも fallback
        assert get_vram_requirement_mb("parakeet", "nonexistent") == 3000

    def test_unknown_engine_returns_none(self):
        assert get_vram_requirement_mb("foobar") is None
        assert get_vram_requirement_mb("foobar", "x") is None


class TestCheckVramBeforeLoad:
    def test_insufficient_default_warns_no_raise(self, monkeypatch, caplog):
        """cuda + 不足 + default: warning のみ、 raise しない。"""
        _patch_vram(monkeypatch, available_mb=4000)  # voxtral 9500 に不足
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            check_vram_before_load("voxtral", None, "cuda", strict=False)
        assert any("VRAM" in r.getMessage() for r in caplog.records)

    def test_insufficient_strict_raises(self, monkeypatch):
        """cuda + 不足 + strict: InsufficientVRAMError (attr 検証)。"""
        _patch_vram(monkeypatch, available_mb=4000)
        with pytest.raises(InsufficientVRAMError) as exc_info:
            check_vram_before_load("voxtral", None, "cuda", strict=True)
        err = exc_info.value
        assert err.engine_name == "voxtral"
        assert err.required_gb == pytest.approx(9500 / 1024)
        assert err.available_gb == pytest.approx(4000 / 1024)

    def test_sufficient_no_warn_no_raise(self, monkeypatch, caplog):
        """cuda + 十分: warn/raise なし。"""
        _patch_vram(monkeypatch, available_mb=24000)  # voxtral 9500 に十分
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            check_vram_before_load("voxtral", None, "cuda", strict=True)  # strict でも
        assert not caplog.records

    def test_cpu_device_skips(self, monkeypatch, caplog):
        """CPU 解決 device は skip (VRAM 無関係)。"""
        _patch_vram(monkeypatch, available_mb=100, resolved_device="cpu")
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            check_vram_before_load("voxtral", None, "cpu", strict=True)
        assert not caplog.records  # strict でも raise せず skip

    def test_no_gpu_skips(self, monkeypatch, caplog):
        """get_available_vram None (GPU/torch なし) は skip。"""
        _patch_vram(monkeypatch, available_mb=None)
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            check_vram_before_load("voxtral", None, "cuda", strict=True)
        assert not caplog.records

    def test_unknown_engine_skips(self, monkeypatch, caplog):
        """未知 engine (requirement None) は skip (fail-open)。"""
        _patch_vram(monkeypatch, available_mb=100)
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            check_vram_before_load("foobar_engine", None, "cuda", strict=True)
        assert not caplog.records


class TestLoadModelStep0Integration:
    """base_engine.load_model() の Step 0 が check を呼ぶことを最小 engine で検証。"""

    def _make_engine(self):
        from livecap_cli.engines.base_engine import BaseEngine, TranscriptionResult

        class _VRAMTestEngine(BaseEngine):
            def transcribe(self, audio_data, sample_rate):  # pragma: no cover
                return TranscriptionResult(text="", confidence=1.0)

            def get_engine_name(self):
                return "voxtral"

            def get_supported_languages(self):
                return ["en"]

            def get_required_sample_rate(self):
                return 16000

        eng = _VRAMTestEngine(device="cuda")
        eng.engine_name = "voxtral"  # requirement 表に載る id
        return eng

    def test_load_model_raises_in_strict_mode(self, monkeypatch):
        """strict env + 低 VRAM で load_model が Step 0 で InsufficientVRAMError。"""
        _patch_vram(monkeypatch, available_mb=1000)
        monkeypatch.setenv("LIVECAP_STRICT_VRAM_CHECK", "1")
        eng = self._make_engine()
        with pytest.raises(InsufficientVRAMError):
            eng.load_model()

    def test_load_model_warns_by_default(self, monkeypatch, caplog):
        """env 未設定 (default) は warn のみ — Step 0 では raise せず、
        後続 step (未実装の download 等) まで進んで別 error になる。"""
        _patch_vram(monkeypatch, available_mb=1000)
        monkeypatch.delenv("LIVECAP_STRICT_VRAM_CHECK", raising=False)
        eng = self._make_engine()
        with caplog.at_level(logging.WARNING, logger="livecap_cli.engines.model_vram"):
            with pytest.raises(Exception) as exc_info:
                eng.load_model()
        # Step 0 の warning は出るが、 InsufficientVRAMError ではない
        # (後続 step で download 等の別 error になる)
        assert not isinstance(exc_info.value, InsufficientVRAMError)
        assert any("VRAM" in r.getMessage() for r in caplog.records)
