"""`should_drop_low_energy` 共通 helper のテスト (Issue #366 Phase 2)。

realtime (`StreamTranscriber._should_skip_low_energy`) と file mode closure の
単一判定点。判定式の契約 (strict `<` / equality pass / -inf skip / validation)
を固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from livecap_cli.audio import _segment_energy_dbfs, should_drop_low_energy

_SR = 16000


def _tone(amplitude: float, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(_SR * seconds), dtype=np.float64)
    return (amplitude * np.sin(2 * np.pi * 440.0 * t / _SR)).astype(np.float32)


class TestJudgement:
    def test_below_threshold_drops_and_returns_energy(self):
        audio = _tone(0.001)  # ≈ -63 dBFS

        should_drop, energy = should_drop_low_energy(
            audio, _SR, threshold_dbfs=-45.0
        )

        assert should_drop is True
        assert energy is not None and energy < -45.0

    def test_above_threshold_passes(self):
        audio = _tone(0.1)  # ≈ -23 dBFS

        should_drop, energy = should_drop_low_energy(
            audio, _SR, threshold_dbfs=-45.0
        )

        assert should_drop is False
        assert energy is not None and energy > -45.0

    def test_equality_passes(self):
        """energy == threshold は pass (strict < — realtime と同一の境界)"""
        audio = _tone(0.05)
        energy = _segment_energy_dbfs(audio, _SR)

        should_drop, returned = should_drop_low_energy(
            audio, _SR, threshold_dbfs=energy  # threshold を実測値そのものに
        )

        assert should_drop is False
        assert returned == pytest.approx(energy)

    def test_neg_inf_skips_energy_computation(self, monkeypatch):
        """-inf は energy 計算自体を skip し (False, None)"""
        import livecap_cli.audio.analysis as analysis

        def _boom(*args, **kwargs):
            raise AssertionError("energy computation must be skipped")

        monkeypatch.setattr(analysis, "_segment_energy_dbfs", _boom)

        should_drop, energy = analysis.should_drop_low_energy(
            _tone(0.001), _SR, threshold_dbfs=float("-inf")
        )

        assert should_drop is False
        assert energy is None


class TestValidation:
    def test_nan_threshold_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            should_drop_low_energy(_tone(0.1), _SR, threshold_dbfs=float("nan"))

    def test_pos_inf_threshold_raises(self):
        with pytest.raises(ValueError, match=r"\+inf"):
            should_drop_low_energy(_tone(0.1), _SR, threshold_dbfs=float("inf"))

    def test_invalid_metric_propagates(self):
        """metric 検証は _segment_energy_dbfs の ValueError を素通し"""
        with pytest.raises(ValueError):
            should_drop_low_energy(
                _tone(0.1), _SR, threshold_dbfs=-45.0, metric="no_such_metric"
            )
