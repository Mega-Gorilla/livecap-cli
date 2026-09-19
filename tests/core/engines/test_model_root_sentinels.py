"""``model_root_sentinels`` fixture (tests/core/engines/conftest.py) の自己検査 (Issue #456)。

fixture は teardown で assert するので、ここでは判定関数を直接呼んで「何を違反と見るか」を固定する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from livecap_cli.engines import hf_cache
from tests.core.engines.conftest import _file_listing, _transient_leftovers


def test_transient_leftovers_detects_hf_metadata_locks_and_partials(tmp_path):
    root = tmp_path / "models"
    (root / "org--model").mkdir(parents=True)
    (root / "org--model" / "config.json").write_text("{}", encoding="utf-8")
    assert _transient_leftovers(root) == []

    (root / "org--model" / ".cache" / "huggingface").mkdir(parents=True)
    (root / "org--model" / "model.bin.incomplete").write_bytes(b"")
    (root / "org--model.lock").write_bytes(b"")
    (root / ".org--model.abc.part").mkdir()
    hits = _transient_leftovers(root)
    assert any(".cache" in h for h in hits)
    assert any(h.endswith(".incomplete") for h in hits)
    assert any(h.endswith(".lock") for h in hits)
    assert any(h.endswith(".part") for h in hits)


def test_file_listing_ignores_empty_dirs(tmp_path):
    """`import whisper_s2t` が %LOCALAPPDATA% に作る空 dir は違反にしない (#430)。"""
    root = tmp_path / "ext"
    (root / "whisper_s2t" / "Cache" / "models").mkdir(parents=True)
    assert _file_listing(root) == set()
    (root / "whisper_s2t" / "Cache" / "models" / "x.bin").write_bytes(b"1")
    assert _file_listing(root) == {str(Path("whisper_s2t") / "Cache" / "models" / "x.bin")}


def test_fixture_passes_for_a_clean_fetch(model_root_sentinels):
    """fetch_repo_dir は models_root に transient を残さず、既定 HF cache にも書かない。"""
    from tests.core.engines.conftest import FakeSnapshotDownloadLocalDir

    roots = model_root_sentinels
    fake = FakeSnapshotDownloadLocalDir(files={"config.json": b"{}", "model.bin": b"w" * 8})
    with patch("huggingface_hub.snapshot_download", fake):
        dest = hf_cache.fetch_repo_dir(
            "org/model",
            hub_root=roots.hub_root,
            staging_root=roots.staging_root,
            destination=roots.models_root / "org--model",
        )
    assert dest.is_dir() and (dest / "config.json").is_file()
    # teardown が default_hub 空 / 外部 root 不変 / transient 無し を assert する
