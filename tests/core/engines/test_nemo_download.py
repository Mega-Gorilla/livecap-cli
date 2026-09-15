"""NeMo (canary / parakeet) の ``.nemo`` が**管理 staging 経由で models root へ 1 部だけ**
落ちること (Issue #447)。

以前は ``from_pretrained(model_name=<repo>)`` を呼んでいた。NeMo は内部で
``hf_hub_download()`` を ``cache_dir=`` 無しで呼ぶので ``.nemo`` が既定の
``~/.cache/huggingface/hub`` へ落ち、``restore_from`` で ``%TEMP%`` へ untar してモデルを
構築し、``save_to()`` で models root へ**もう 1 部**書いていた。

固定する契約:

* ``hf_hub_download(repo, filename="<name>.nemo", local_dir=<cache_root>/downloads/...)``
  (NeMo と同じファイル名規則) → models root の ``<org>--<name>.nemo`` へ move → staging 消去
* **NeMo を import しない / ``from_pretrained`` を呼ばない / untar しない**
* 既存の ``.nemo`` があれば何もしない (移設しない)
* 失敗時は models root に何も残さない

``hf_hub_download`` は差し替え、``nemo`` は「触ったら落ちる」偽物にする。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines.model_memory_cache import ModelMemoryCache
from livecap_cli.resources import _reset_resources_for_tests, get_model_manager


class _FakeHfHubDownload:
    def __init__(self, *, fail: Exception | None = None):
        self.calls: list[dict] = []
        self.fail = fail

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        local_dir = Path(kwargs["local_dir"])
        (local_dir / ".cache" / "huggingface" / "download").mkdir(parents=True, exist_ok=True)
        if self.fail is not None:
            raise self.fail
        target = local_dir / kwargs["filename"]
        target.write_bytes(b"NEMO")
        return str(target)


class _Trap:
    """属性を触った瞬間に落ちる: ``nemo_asr.models.X.from_pretrained`` が呼ばれたら fail。"""

    def __getattr__(self, name):
        raise AssertionError(f"download 中に NeMo が使われた: .{name}")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    default_hub = tmp_path / "default-hf-hub"
    default_hub.mkdir()
    monkeypatch.setenv("LIVECAP_CORE_MODELS_DIR", str(models_root))
    monkeypatch.setenv("LIVECAP_CORE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(default_hub))
    _reset_resources_for_tests()
    ModelMemoryCache.clear()

    # NeMo は「触ったら落ちる」偽物に差し替える。download で import されないことの証明。
    fake_nemo_asr = types.ModuleType("nemo.collections.asr")
    fake_nemo_asr.models = _Trap()
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_nemo_asr)

    yield types.SimpleNamespace(models_root=models_root, cache_root=cache_root, default_hub=default_hub)
    _reset_resources_for_tests()
    ModelMemoryCache.clear()


def _cases():
    from livecap_cli.engines.canary_engine import CanaryEngine
    from livecap_cli.engines.parakeet_engine import ParakeetEngine

    return [
        pytest.param(lambda: ParakeetEngine(device="cpu"), "nvidia/parakeet-tdt-0.6b-v2", id="parakeet"),
        pytest.param(lambda: ParakeetEngine(device="cpu", engine_name="parakeet_ja"), "nvidia/parakeet-tdt_ctc-0.6b-ja", id="parakeet_ja"),
        pytest.param(lambda: CanaryEngine(device="cpu", language="en"), "nvidia/canary-1b-flash", id="canary"),
    ]


@pytest.mark.parametrize("make_engine,repo_id", _cases())
class TestNemoDownload:
    def test_nemo_file_goes_through_managed_staging_into_models_root(self, roots, make_engine, repo_id):
        engine = make_engine()
        assert engine.model_name == repo_id
        model_path = engine._get_local_model_path(get_model_manager().get_models_dir())
        fake = _FakeHfHubDownload()

        with patch("huggingface_hub.hf_hub_download", fake):
            engine._download_model(model_path, None, engine.model_manager)

        (call,) = fake.calls
        assert call["repo_id"] == repo_id
        assert call["filename"] == repo_id.split("/")[-1] + ".nemo", "NeMo と同じファイル名規則"
        staging = Path(call["local_dir"])
        assert staging == roots.cache_root / "downloads" / repo_id.replace("/", "--"), "管理 staging へ取る"
        assert "cache_dir" not in call
        assert model_path == roots.models_root / (repo_id.replace("/", "--") + ".nemo")
        assert model_path.is_file() and model_path.read_bytes() == b"NEMO", "models root へ **ファイル** として配置"
        assert not staging.exists(), "staging は消す — 保持は 1 部だけ"
        assert not any(roots.default_hub.iterdir()), "既定 HF cache には落ちない"

    def test_existing_nemo_is_left_alone(self, roots, make_engine, repo_id):
        engine = make_engine()
        model_path = engine._get_local_model_path(get_model_manager().get_models_dir())
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"existing")
        fake = _FakeHfHubDownload(fail=AssertionError("既存なら呼ばれない"))

        with patch("huggingface_hub.hf_hub_download", fake):
            engine._download_model(model_path, None, engine.model_manager)

        assert fake.calls == [] and model_path.read_bytes() == b"existing"

    def test_failure_leaves_models_root_untouched(self, roots, make_engine, repo_id):
        engine = make_engine()
        model_path = engine._get_local_model_path(get_model_manager().get_models_dir())
        fake = _FakeHfHubDownload(fail=ConnectionError("network down"))

        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(ConnectionError):
                engine._download_model(model_path, None, engine.model_manager)

        assert not model_path.exists()


def test_download_path_has_no_temp_staging_wrapper():
    """untar が起きなくなったので ``ascii_safe_temp_environment(purpose="download")`` は
    残さない (#434)。``nemo-restore`` 用途 (load 経路) は残る。"""
    import ast

    for name in ("canary_engine", "parakeet_engine"):
        src = Path("livecap_cli/engines") / f"{name}.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        purposes = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ascii_safe_temp_environment"
            for kw in node.keywords
            if kw.arg == "purpose" and isinstance(kw.value, ast.Constant)
        ]
        assert purposes == ["nemo-restore"], f"{name}: {purposes}"
