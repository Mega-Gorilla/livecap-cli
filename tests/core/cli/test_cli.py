from pathlib import Path

import pytest

from livecap_core import cli


@pytest.mark.parametrize("ensure_ffmpeg", [False])
def test_cli_diagnose_reports_i18n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensure_ffmpeg: bool) -> None:
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(tmp_path / "cache"))

    report = cli.diagnose(ensure_ffmpeg=ensure_ffmpeg)

    assert report.models_root
    assert report.cache_root
    assert report.i18n.fallback_count >= 0
    assert report.i18n.translator.registered in (True, False)
    assert isinstance(report.available_engines, list)
    # Phase 2: New diagnostic fields
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.vad_backends, list)
    # cuda_device can be None or str
    assert report.cuda_device is None or isinstance(report.cuda_device, str)
