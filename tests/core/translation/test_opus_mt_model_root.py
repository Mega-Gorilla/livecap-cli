"""OPUS-MT の正本が **models_root** (CTranslate2 model + tokenizer + manifest) だけになること (Issue #456 PR 2、#455)。

以前は ``TransformersConverter(<repo id>)`` と ``AutoTokenizer.from_pretrained(<repo id>)`` が変換元
snapshot (582 MB) を**既定 HF cache (root の外)** へ落とし、``load_model()`` のたびにそこを読んでいた。

固定する契約:

* 変換元は ``fetch_repo_dir`` で ``<cache_root>/downloads/opus-mt-source/…`` (staging) へ取り、変換後に消す。
  重みは ``model.safetensors`` を優先し、無ければ ``pytorch_model.bin``
* 正本 ``<models_root>/opus-mt/<org>--<name>/`` に CT2 model + tokenizer + manifest。
  ``ctranslate2.Translator`` と ``AutoTokenizer.from_pretrained`` は**その dir** を受ける (repo id は渡さない)
* cache hit は manifest + required だけ。#456 以前の変換済み dir (tokenizer 無し) は tokenizer だけ足して adopt
* load 失敗で manifest を無効化 (self-heal)。取得 / 変換失敗で正本を作らない

``snapshot_download`` / ``TransformersConverter`` / ``AutoTokenizer`` / ``ctranslate2.Translator`` は差し替える。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ctranslate2")
pytest.importorskip("transformers")

from livecap_cli.engines import model_store as ms
from livecap_cli.translation.exceptions import TranslationModelError
from livecap_cli.translation.impl.opus_mt import OpusMTTranslator
from tests.core.model_root_fixtures import FakeSnapshotDownloadLocalDir, file_fingerprints, write_hub_snapshot

REPO_ID = "Helsinki-NLP/opus-mt-ja-en"
DEST_NAME = "Helsinki-NLP--opus-mt-ja-en"
SOURCE_FILES = {
    "config.json": b'{"model_type": "marian"}',
    "generation_config.json": b"{}",
    "tokenizer_config.json": b'{"tokenizer_class": "MarianTokenizer"}',
    "vocab.json": b"{}",
    "source.spm": b"S",
    "target.spm": b"T",
    "model.safetensors": b"w" * 128,
    "README.md": b"#",
}
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "source.spm", "target.spm")


class _FakeConverter:
    """``ctranslate2.converters.TransformersConverter`` の代役: 入力 dir を記録し、CT2 の形を書く。"""

    instances: list = []

    def __init__(self, model_name_or_path):
        self.source = Path(model_name_or_path)
        self.converted_to = None
        _FakeConverter.instances.append(self)

    def convert(self, output_dir, quantization=None, **_):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        assert (self.source / "config.json").is_file(), "変換元はローカル dir"
        (out / "model.bin").write_bytes(b"ct2:" + (self.source / "model.safetensors").read_bytes()[:8] if (self.source / "model.safetensors").exists() else b"ct2:bin")
        (out / "config.json").write_bytes(b'{"bos_token": "<s>"}')
        (out / "shared_vocabulary.json").write_bytes(b"[]")
        self.converted_to = out


class _FakeTokenizer:
    def __init__(self, source: Path):
        self.source = source

    def save_pretrained(self, out):
        out = Path(out)
        for name in TOKENIZER_FILES:
            (out / name).write_bytes((self.source / name).read_bytes() if (self.source / name).exists() else b"?")
        (out / "special_tokens_map.json").write_bytes(b"{}")


@pytest.fixture
def managed(model_root_sentinels, monkeypatch):
    roots = model_root_sentinels
    _FakeConverter.instances = []
    from_pretrained_calls: list = []

    def fake_from_pretrained(path, *a, **k):
        from_pretrained_calls.append(Path(path))
        return _FakeTokenizer(Path(path))

    translator_cls = MagicMock(name="ctranslate2.Translator")
    monkeypatch.setattr("ctranslate2.converters.TransformersConverter", _FakeConverter)
    monkeypatch.setattr("livecap_cli.translation.impl.opus_mt.transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("livecap_cli.translation.impl.opus_mt.ctranslate2.Translator", translator_cls)
    yield type("NS", (), {})(), roots, from_pretrained_calls, translator_cls


def _fake(files=None, **kw):
    return FakeSnapshotDownloadLocalDir(files=files or SOURCE_FILES, **kw)


def _load(fake):
    with patch("huggingface_hub.snapshot_download", fake):
        t = OpusMTTranslator(source_lang="ja", target_lang="en")
        t.load_model()
    return t


def _dest(roots) -> Path:
    return roots.models_root / "opus-mt" / DEST_NAME


class TestColdConversion:
    def test_source_is_fetched_into_staging_and_removed(self, managed):
        _, roots, _, _ = managed
        fake = _fake()

        _load(fake)

        (call,) = fake.calls
        assert call["repo_id"] == REPO_ID
        assert Path(call["local_dir"]).is_relative_to(roots.staging_root), "変換元は staging (cache_root) へ"
        assert Path(call["cache_dir"]) == roots.hub_root
        assert "model.safetensors" in call["allow_patterns"] and "pytorch_model.bin" not in call["allow_patterns"]
        assert not (roots.staging_root / "opus-mt-source").exists() or not any((roots.staging_root / "opus-mt-source").iterdir()), "変換後に変換元を消す"
        assert not any(roots.default_hub.iterdir())

    def test_converter_and_tokenizer_read_the_local_source_not_the_repo_id(self, managed):
        _, roots, from_pretrained_calls, _ = managed

        _load(_fake())

        (conv,) = _FakeConverter.instances
        assert conv.source.is_relative_to(roots.staging_root), "TransformersConverter(<ローカル dir>)"
        assert str(conv.source) != REPO_ID
        assert from_pretrained_calls[0].is_relative_to(roots.staging_root), "tokenizer も変換元 dir から"

    def test_destination_holds_ct2_model_tokenizer_and_manifest(self, managed):
        _, roots, _, _ = managed

        _load(_fake())

        dest = _dest(roots)
        manifest = ms.validate_repo_dir(dest, repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES)
        assert manifest is not None and manifest.revision == "ct2:int8" and manifest.commit_sha == "c" * 40
        assert (dest / "model.bin").is_file() and (dest / "source.spm").read_bytes() == b"S"
        assert not (dest / "model.safetensors").exists() and not (dest / "pytorch_model.bin").exists(), "変換元の重みは正本に持ち込まない"

    def test_translator_and_tokenizer_load_from_models_root_dir(self, managed):
        _, roots, from_pretrained_calls, translator_cls = managed

        _load(_fake())

        (args, kwargs) = translator_cls.call_args
        assert Path(args[0]) == _dest(roots)
        assert kwargs["compute_type"] == "int8"
        assert from_pretrained_calls[-1] == _dest(roots), "load 時の tokenizer は正本 dir から (repo id ではない)"

    def test_falls_back_to_pytorch_model_bin_when_no_safetensors(self, managed):
        _, roots, _, _ = managed
        files = {k: v for k, v in SOURCE_FILES.items() if k != "model.safetensors"}
        files["pytorch_model.bin"] = b"b" * 128
        fake = _fake(files)

        _load(fake)

        assert [c["allow_patterns"][-1] for c in fake.calls] == ["model.safetensors", "pytorch_model.bin"]
        assert ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None


class TestExternalSourceSnapshot:
    """0.2.0 までの ``TransformersConverter(<repo id>)`` が既定 HF cache (root の外) に落とした変換元 (#453)。"""

    def test_conversion_uses_the_external_snapshot_without_download_and_leaves_it_untouched(self, managed):
        _, roots, _, _ = managed
        snapshot = write_hub_snapshot(roots.external_hub, REPO_ID, SOURCE_FILES)
        before = file_fingerprints(roots.external_hub)
        fake = _fake(fail=AssertionError("root の外の変換元から変換できるので download しない"))

        _load(fake)

        assert fake.calls == []
        (conv,) = _FakeConverter.instances
        assert conv.source.is_relative_to(roots.staging_root), "変換元は staging へ copy してから変換する"
        manifest = ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES)
        assert manifest is not None and manifest.commit_sha == snapshot.name, "由来の commit sha を残す"
        assert file_fingerprints(roots.external_hub) == before and snapshot.is_dir(), "外は消さない・変えない"
        assert not any((roots.staging_root / "opus-mt-source").iterdir()), "変換後に staging の copy は消す"

    def test_external_snapshot_with_only_pytorch_model_bin_is_used_offline(self, managed):
        _, roots, _, _ = managed
        files = {k: v for k, v in SOURCE_FILES.items() if k != "model.safetensors"}
        files["pytorch_model.bin"] = b"b" * 128
        write_hub_snapshot(roots.external_hub, REPO_ID, files)
        fake = _fake(fail=OSError("offline"))

        _load(fake)

        assert fake.calls == []
        assert ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None

    def test_incomplete_external_snapshot_falls_back_to_download(self, managed):
        _, roots, _, _ = managed
        write_hub_snapshot(roots.external_hub, REPO_ID, {"config.json": b"{}"})
        fake = _fake()

        _load(fake)

        assert len(fake.calls) == 1


class TestCacheHitAndAdopt:
    def test_valid_destination_skips_fetch_and_conversion(self, managed):
        _, roots, from_pretrained_calls, translator_cls = managed
        _load(_fake())
        _FakeConverter.instances.clear()
        translator_cls.reset_mock()

        _load(_fake(fail=AssertionError("hit なので取得しない")))

        assert _FakeConverter.instances == []
        assert Path(translator_cls.call_args.args[0]) == _dest(roots)

    def test_pre_456_converted_dir_gets_tokenizer_and_is_adopted_without_reconversion(self, managed):
        """#456 以前の変換済み dir: CT2 model だけで tokenizer が無い → tokenizer だけ取って adopt。"""
        _, roots, _, _ = managed
        dest = _dest(roots)
        dest.mkdir(parents=True)
        (dest / "model.bin").write_bytes(b"old-ct2")
        (dest / "config.json").write_bytes(b"{}")
        (dest / "shared_vocabulary.json").write_bytes(b"[]")
        fake = _fake()

        _load(fake)

        (call,) = fake.calls
        assert sorted(call["allow_patterns"]) == sorted(OpusMTTranslator.TOKENIZER_SOURCE_FILES), "tokenizer (+ HF config) だけ取る (重み 300 MB は取らない)"
        assert not any("model" in p for p in call["allow_patterns"])
        assert _FakeConverter.instances == [], "再変換しない"
        manifest = ms.validate_repo_dir(dest, repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES)
        assert manifest is not None and manifest.source == "adopted"
        assert (dest / "model.bin").read_bytes() == b"old-ct2" and (dest / "source.spm").read_bytes() == b"S"

    def test_destination_missing_ct2_model_is_rebuilt(self, managed):
        """manifest はあるが model.bin が消えている → miss → 作り直し (旧 dir は隔離)。"""
        _, roots, _, _ = managed
        _load(_fake())
        (_dest(roots) / "model.bin").unlink()
        _FakeConverter.instances.clear()

        _load(_fake())

        assert len(_FakeConverter.instances) == 1
        assert ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None


class TestFailure:
    def test_fetch_failure_creates_no_destination(self, managed):
        _, roots, _, _ = managed

        with pytest.raises(TranslationModelError, match="Failed to prepare model"):
            _load(_fake(fail=RuntimeError("network down")))

        assert not _dest(roots).exists()

    def test_conversion_failure_creates_no_destination_and_drops_the_partial_payload(self, managed, monkeypatch):
        """変換途中の payload は invalid なので消す。変換元は残す (次回は再取得せず変換から)。"""
        _, roots, _, _ = managed

        class Broken(_FakeConverter):
            def convert(self, output_dir, **_):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "model.bin").write_bytes(b"partial")
                raise RuntimeError("conversion exploded")

        monkeypatch.setattr("ctranslate2.converters.TransformersConverter", Broken)
        with pytest.raises(TranslationModelError, match="Failed to convert model"):
            _load(_fake())

        assert not _dest(roots).exists()
        staging = roots.staging_root / "opus-mt-source"
        assert not (staging / f"{DEST_NAME}.payload").exists(), "invalid な payload は残さない"
        assert ms.validate_repo_dir(staging / f"{DEST_NAME}.source", repo_id=REPO_ID) is not None, "変換元は残す"

        # 次回: 変換元は再取得せず (downloader が呼ばれない)、変換からやり直して publish
        monkeypatch.setattr("ctranslate2.converters.TransformersConverter", _FakeConverter)
        _load(_fake(fail=AssertionError("変換元は staging にあるので再取得しない")))
        assert ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None
        assert not staging.exists() or not any(staging.iterdir()), "成功後は staging を消す"

    def test_publish_failure_keeps_completed_payload_and_resumes_without_refetch_or_reconversion(self, managed, monkeypatch):
        """publish (`os.replace`) が一時的に失敗 → 完成済み payload と変換元を残し、次回は download も
        変換もせず publish から再開する (PR #459 レビュー MEDIUM: `publish_dir` の契約と揃える)。"""
        _, roots, _, translator_cls = managed
        real_replace = ms.os.replace

        def failing_replace(src, dst, *a, **k):
            if Path(dst) == _dest(roots):
                raise OSError("publish blocked")
            return real_replace(src, dst, *a, **k)

        with patch.object(ms.os, "replace", failing_replace):
            with pytest.raises(TranslationModelError, match="Failed to publish model"):
                _load(_fake())

        assert not _dest(roots).exists()
        staging = roots.staging_root / "opus-mt-source"
        payload = staging / f"{DEST_NAME}.payload"
        assert ms.validate_repo_dir(payload, repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None, "完成済み payload を残す"
        assert (staging / f"{DEST_NAME}.source" / "config.json").is_file(), "変換元を残す"
        _FakeConverter.instances.clear()

        _load(_fake(fail=AssertionError("payload があるので再取得しない")))

        assert _FakeConverter.instances == [], "再変換しない"
        assert ms.validate_repo_dir(_dest(roots), repo_id=REPO_ID, required=OpusMTTranslator.REQUIRED_FILES) is not None
        assert Path(translator_cls.call_args.args[0]) == _dest(roots)
        assert not payload.exists() and not (staging / f"{DEST_NAME}.source").exists(), "成功後は staging を消す"

    def test_load_failure_invalidates_manifest(self, managed):
        _, roots, _, translator_cls = managed
        _load(_fake())
        translator_cls.side_effect = RuntimeError("corrupt model.bin")

        with pytest.raises(TranslationModelError, match="Failed to load model"):
            _load(_fake(fail=AssertionError("hit")))

        assert ms.read_manifest(_dest(roots)).source == ms.INVALIDATED_SOURCE
        assert ms.validate_repo_dir(_dest(roots)) is None
