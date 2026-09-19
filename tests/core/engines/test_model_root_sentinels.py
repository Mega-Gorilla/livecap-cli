"""``model_root_sentinels`` fixture (tests/core/engines/conftest.py) の自己検査 (Issue #456)。

fixture は teardown で assert するので、ここでは判定関数を直接呼んで「何を違反と見るか」を固定する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines import hf_cache
from tests.core.engines.conftest import _external_model_dirs, _file_listing, _transient_leftovers


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
    from tests.core.engines.test_hf_cache import _FakeSnapshotDownloadLocalDir

    roots = model_root_sentinels
    with patch("huggingface_hub.snapshot_download", _FakeSnapshotDownloadLocalDir()):
        dest = hf_cache.fetch_repo_dir(
            "org/model",
            hub_root=roots.hub_root,
            staging_root=roots.staging_root,
            destination=roots.models_root / "org--model",
        )
    assert dest.is_dir() and (dest / "config.json").is_file()
    # teardown が default_hub 空 / 外部 root 不変 / transient 無し を assert する


def test_external_candidates_are_tracked_even_when_absent(tmp_path, monkeypatch):
    """開始時に無い外部 dir も候補に残し、before は空集合 → テスト中に新規作成されれば差分に出る
    (PR #457 レビュー HIGH: 存在するものだけを見ると新規作成を見逃す)。"""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    candidates = _external_model_dirs()
    whisper_cache = tmp_path / "LocalAppData" / "whisper_s2t" / "whisper_s2t" / "Cache" / "models"
    assert whisper_cache in candidates and not whisper_cache.exists()

    before = _file_listing(whisper_cache)
    assert before == set()
    whisper_cache.mkdir(parents=True)
    (whisper_cache / "models--x" / "snapshots").mkdir(parents=True)
    (whisper_cache / "models--x" / "snapshots" / "model.bin").write_bytes(b"1")
    assert _file_listing(whisper_cache) - before, "新規作成された外部 dir のファイルが差分に出る"


def test_fixture_pins_imported_huggingface_hub_constants(model_root_sentinels):
    """env だけでなく import 済みの ``huggingface_hub.constants`` も sentinel を指す。"""
    import huggingface_hub.constants as hf

    assert Path(hf.HF_HUB_CACHE) == model_root_sentinels.default_hub
    assert hf.HF_HUB_OFFLINE is True


def test_dropping_cache_dir_fails_loud_under_fixture(model_root_sentinels):
    """**変異**: production が ``cache_dir=`` を落として本物の ``snapshot_download`` を呼ぶと、
    既定 cache = 空 sentinel + offline なので silent fallback せず必ず落ちる。"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    with pytest.raises((LocalEntryNotFoundError, OSError)):
        snapshot_download("org/model", local_dir=str(model_root_sentinels.staging_root / "x"), max_workers=1)
    assert not any(model_root_sentinels.default_hub.iterdir())
