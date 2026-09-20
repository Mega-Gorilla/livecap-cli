"""Riva-Translate-4B-Instruct の正本が **models_root** に置かれ、そこからロードされること (Issue #456 PR 2、#455)。

以前は ``AutoTokenizer.from_pretrained(<repo id>)`` / ``AutoModelForCausalLM.from_pretrained(<repo id>)`` が
7.9 GB を**既定 HF cache (root の外)** へ落としていた。

固定する契約:

* ``fetch_repo_dir`` で ``<models_root>/nvidia--Riva-Translate-4B-Instruct/`` (flattened dir + manifest) へ取る
  (staging は ``<cache_root>/downloads``、``README.md`` / ``.gitattributes`` は取らない)
* ``from_pretrained`` ×2 は**その dir** を受ける。cache hit は manifest + required だけ
* load 失敗で manifest を無効化 (self-heal)。取得失敗で正本を作らない

``snapshot_download`` と ``transformers`` は差し替える。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from livecap_cli.engines import model_store as ms
from livecap_cli.translation.exceptions import TranslationModelError
from livecap_cli.translation.impl.riva_instruct import RivaInstructTranslator
from tests.core.model_root_fixtures import FakeSnapshotDownloadLocalDir, write_repo_dir

REPO_ID = "nvidia/Riva-Translate-4B-Instruct"
DEST_NAME = "nvidia--Riva-Translate-4B-Instruct"
REPO_FILES = {
    "config.json": b'{"model_type": "llama"}',
    "generation_config.json": b"{}",
    "model.safetensors.index.json": b"{}",
    "model-00001-of-00002.safetensors": b"w" * 64,
    "model-00002-of-00002.safetensors": b"w" * 32,
    "special_tokens_map.json": b"{}",
    "tokenizer.json": b"{}",
    "tokenizer_config.json": b"{}",
    "README.md": b"#",
    ".gitattributes": b"",
}
MODEL_FILES = {k: v for k, v in REPO_FILES.items() if k not in ("README.md", ".gitattributes")}


@pytest.fixture
def managed(model_root_sentinels):
    roots = model_root_sentinels
    with patch("livecap_cli.translation.impl.riva_instruct.transformers") as tf, patch(
        "livecap_cli.utils.get_available_vram", return_value=None
    ):
        tf.AutoTokenizer.from_pretrained.return_value = MagicMock(name="tokenizer")
        model = MagicMock(name="model")
        model.to.return_value = model
        tf.AutoModelForCausalLM.from_pretrained.return_value = model
        yield roots, tf


def _fake(**kw):
    return FakeSnapshotDownloadLocalDir(files=REPO_FILES, **kw)


def _load(fake, device="cpu"):
    with patch("huggingface_hub.snapshot_download", fake):
        t = RivaInstructTranslator(device=device)
        t.load_model()
    return t


class TestColdCache:
    def test_downloads_via_staging_into_models_root(self, managed):
        roots, tf = managed
        fake = _fake()

        _load(fake)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID
        assert Path(call["local_dir"]) == roots.staging_root / DEST_NAME / "download"
        assert Path(call["cache_dir"]) == roots.hub_root
        assert "README.md" in call["ignore_patterns"]
        dest = roots.models_root / DEST_NAME
        manifest = ms.validate_repo_dir(dest, repo_id=REPO_ID, required=RivaInstructTranslator.REQUIRED_FILES)
        assert manifest is not None and sorted(f.path for f in manifest.files) == sorted(MODEL_FILES)
        assert not (roots.staging_root / DEST_NAME).exists()
        assert not any(roots.default_hub.iterdir())

    def test_from_pretrained_receives_models_root_dir_not_repo_id(self, managed):
        roots, tf = managed

        _load(_fake())

        dest = roots.models_root / DEST_NAME
        (tok_target,), _ = tf.AutoTokenizer.from_pretrained.call_args
        (model_target,), kwargs = tf.AutoModelForCausalLM.from_pretrained.call_args
        assert Path(tok_target) == dest and Path(model_target) == dest
        assert tok_target != REPO_ID and "cache_dir" not in kwargs


class TestCacheHit:
    def test_valid_manifest_skips_download(self, managed):
        roots, tf = managed
        write_repo_dir(roots.models_root / DEST_NAME, MODEL_FILES, repo_id=REPO_ID)

        _load(_fake(fail=AssertionError("hit")))

        tf.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_complete_dir_without_manifest_is_adopted(self, managed):
        """手で置いた / root の外から取り込んだ完全な dir (manifest 無し) は再取得せず採用する。"""
        roots, tf = managed
        write_repo_dir(roots.models_root / DEST_NAME, MODEL_FILES, repo_id=REPO_ID, with_manifest=False)

        _load(_fake(fail=AssertionError("adopt できるので呼ばれない")))

        manifest = ms.validate_repo_dir(roots.models_root / DEST_NAME, repo_id=REPO_ID, required=RivaInstructTranslator.REQUIRED_FILES)
        assert manifest is not None and manifest.source == "adopted"

    def test_missing_shard_is_not_a_hit(self, managed):
        roots, _ = managed
        write_repo_dir(roots.models_root / DEST_NAME, MODEL_FILES, repo_id=REPO_ID)
        (roots.models_root / DEST_NAME / "model-00002-of-00002.safetensors").unlink()
        fake = _fake()

        _load(fake)

        assert len(fake.calls) == 1


class TestFailure:
    def test_fetch_failure_creates_no_destination(self, managed):
        roots, tf = managed

        with pytest.raises(TranslationModelError, match="Failed to prepare"):
            _load(_fake(fail=RuntimeError("network down")))

        assert not (roots.models_root / DEST_NAME).exists()
        tf.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_load_failure_invalidates_manifest(self, managed):
        roots, tf = managed
        write_repo_dir(roots.models_root / DEST_NAME, MODEL_FILES, repo_id=REPO_ID)
        tf.AutoModelForCausalLM.from_pretrained.side_effect = OSError("corrupt shard")

        with pytest.raises(TranslationModelError, match="Failed to load"):
            _load(_fake(fail=AssertionError("hit")))

        assert ms.read_manifest(roots.models_root / DEST_NAME).source == ms.INVALIDATED_SOURCE
