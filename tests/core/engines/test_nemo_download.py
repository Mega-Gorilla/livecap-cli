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
* 既存の ``.nemo`` があれば何もしない
* 失敗時は models root に何も残さない
* (#456) 正本は ``<models_root>/<org>--<name>.nemo`` の**ファイル**。旧 workaround が作った
  ``<models_root>/<engine>/<name>.nemo`` (parakeet / parakeet_ja) と、canary の path 欠陥が作った
  ``<name>.nemo/<name>.nemo`` (dir の中に同名ファイル) は cold load で正本の位置へ戻す。
  ``load_model()`` / ``_prepare_model_directory()`` の override は無い

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
        target.write_bytes(b"./.NEMO")
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

    # root の外の旧 cache (#453) は tmp に pin する。default_hub = 0.1.0 の NeMo ``from_pretrained`` が
    # ``~/.cache/huggingface/hub/models--nvidia--…/snapshots/<sha>/<name>.nemo`` に落としていた場所
    from livecap_cli.engines import legacy_model_layouts

    monkeypatch.setattr(
        legacy_model_layouts, "external_hub_roots", lambda: [legacy_model_layouts.ExternalCacheRoot("default HF cache", default_hub)]
    )

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
            engine._download_model(model_path, None)

        (call,) = fake.calls
        assert call["repo_id"] == repo_id
        assert call["filename"] == repo_id.split("/")[-1] + ".nemo", "NeMo と同じファイル名規則"
        staging = Path(call["local_dir"])
        assert staging == roots.cache_root / "downloads" / repo_id.replace("/", "--"), "管理 staging へ取る"
        assert Path(call["cache_dir"]) == roots.cache_root / "huggingface" / "hub", (
            "cache_dir も管理 hub を明示する (省略すると既定 HF_HUB_CACHE を lookup する)"
        )
        assert model_path == roots.models_root / (repo_id.replace("/", "--") + ".nemo")
        assert model_path.is_file() and model_path.read_bytes() == b"./.NEMO", "models root へ **ファイル** として配置"
        assert not staging.exists(), "staging は消す — 保持は 1 部だけ"
        assert not any(roots.default_hub.iterdir()), "既定 HF cache には落ちない"

    def test_existing_nemo_is_left_alone(self, roots, make_engine, repo_id):
        engine = make_engine()
        model_path = engine._get_local_model_path(get_model_manager().get_models_dir())
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"existing")
        fake = _FakeHfHubDownload(fail=AssertionError("既存なら呼ばれない"))

        with patch("huggingface_hub.hf_hub_download", fake):
            engine._download_model(model_path, None)

        assert fake.calls == [] and model_path.read_bytes() == b"existing"

    def test_failure_leaves_models_root_untouched(self, roots, make_engine, repo_id):
        engine = make_engine()
        model_path = engine._get_local_model_path(get_model_manager().get_models_dir())
        fake = _FakeHfHubDownload(fail=ConnectionError("network down"))

        with patch("huggingface_hub.hf_hub_download", fake):
            with pytest.raises(ConnectionError):
                engine._download_model(model_path, None)

        assert not model_path.exists()


@pytest.mark.parametrize("make_engine,repo_id", _cases())
class TestLegacyNemoLayouts:
    def _dest(self, engine):
        return engine._get_local_model_path(get_model_manager().get_models_dir())

    def test_engine_subdir_duplicate_is_moved_to_root(self, roots, make_engine, repo_id):
        """旧 ``load_model()`` override が ``<models_root>/<engine>/<name>.nemo`` へ移していた形。"""
        engine = make_engine()
        dest = self._dest(engine)
        legacy = roots.models_root / engine.engine_name / dest.name
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"./.legacy")
        fake = _FakeHfHubDownload(fail=AssertionError("旧配置から戻せるので呼ばれない"))

        with patch("huggingface_hub.hf_hub_download", fake):
            engine._reconcile_legacy_layouts(dest)
            engine._download_model(dest, None)

        assert dest.is_file() and dest.read_bytes() == b"./.legacy"
        assert not legacy.exists() and not legacy.parent.exists(), "空になった engine subdir も消す"
        assert engine._is_model_cached(dest)

    def test_root_file_plus_subdir_duplicate_keeps_root_and_drops_duplicate(self, roots, make_engine, repo_id):
        """実測 (runner root): ``models/parakeet/…`` 4.7 GB + root 側 2.4 GB の二重保持。"""
        engine = make_engine()
        dest = self._dest(engine)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"./.root")
        legacy = roots.models_root / engine.engine_name / dest.name
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"./.dup")

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload(fail=AssertionError("hit"))):
            engine._reconcile_legacy_layouts(dest)
            engine._download_model(dest, None)

        assert dest.read_bytes() == b"./.root"
        assert not legacy.exists()

    def test_cache_hit_path_removes_duplicate_without_download(self, roots, make_engine, repo_id):
        """template の hit 経路 (``_get_or_download_model``) でも重複が消える — 旧配置の整理は
        download phase ではなく cache 判定の前で行う (実測: root 側 hit のまま 4.7 GB の重複が残っていた)。"""
        engine = make_engine()
        dest = self._dest(engine)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"./.root")
        legacy = roots.models_root / engine.engine_name / dest.name
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"./.dup")

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload(fail=AssertionError("hit なので呼ばれない"))):
            resolved = engine._get_or_download_model(roots.models_root)

        assert resolved == dest and dest.read_bytes() == b"./.root"
        assert not legacy.exists() and not legacy.parent.exists()

    def test_corrupt_root_file_is_quarantined_and_valid_duplicate_wins(self, roots, make_engine, repo_id):
        """root 側が truncated (validator NG) + subdir に valid な複製 → root を隔離し複製を正本に
        (PR #458 レビュー HIGH: 以前は `is_file()` だけで root を正本扱いし、valid な複製を消していた)。"""
        engine = make_engine()
        dest = self._dest(engine)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"trunc")
        legacy = roots.models_root / engine.engine_name / dest.name
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"./.valid-dup")

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload(fail=AssertionError("複製から戻せるので呼ばれない"))):
            resolved = engine._get_or_download_model(roots.models_root)

        assert resolved == dest and dest.read_bytes() == b"./.valid-dup"
        assert not legacy.exists()
        assert any(".invalid-" in p.name and p.read_bytes() == b"trunc" for p in roots.models_root.iterdir()), "壊れた root は隔離 (削除しない)"

    def test_inner_less_nemo_dir_is_quarantined_then_cold_download_publishes(self, roots, make_engine, repo_id):
        """`<name>.nemo/` に同名ファイルが無い → 隔離して、cold download が file を publish できる
        (以前は dir を残したため `os.replace(file, dir)` が毎回失敗していた)。"""
        engine = make_engine()
        dest = self._dest(engine)
        dest.mkdir(parents=True)
        (dest / "unrelated.bin").write_bytes(b"?")
        fake = _FakeHfHubDownload()

        with patch("huggingface_hub.hf_hub_download", fake):
            resolved = engine._get_or_download_model(roots.models_root)

        assert len(fake.calls) == 1
        assert resolved == dest and dest.is_file() and dest.read_bytes() == b"./.NEMO"
        assert any(".invalid-" in p.name and (p / "unrelated.bin").is_file() for p in roots.models_root.iterdir())

    def test_default_hf_cache_nemo_is_copied_and_never_deleted(self, roots, make_engine, repo_id):
        """0.1.0 の NeMo ``from_pretrained`` が既定 HF cache (root の外) に落とした ``.nemo`` (#453)。"""
        from tests.core.model_root_fixtures import file_fingerprints, write_hub_snapshot

        engine = make_engine()
        dest = self._dest(engine)
        write_hub_snapshot(roots.default_hub, repo_id, {dest.name.split("--", 1)[1]: b"./.external"})
        before = file_fingerprints(roots.default_hub)
        fake = _FakeHfHubDownload(fail=AssertionError("root の外の .nemo から取り込めるので呼ばれない"))

        with patch("huggingface_hub.hf_hub_download", fake):
            engine._reconcile_legacy_layouts(dest)
            engine._download_model(dest, None)

        assert dest.is_file() and dest.read_bytes() == b"./.external"
        assert file_fingerprints(roots.default_hub) == before, "外は 1 byte も変えない"
        assert engine._is_model_cached(dest)

    def test_nested_nemo_dir_is_unnested(self, roots, make_engine, repo_id):
        """canary の旧 ``_prepare_model_directory`` が ``.nemo`` path を dir として返していた形:
        ``<name>.nemo/<name>.nemo`` (+ 隣に ``<name>.bin`` 等)。"""
        engine = make_engine()
        dest = self._dest(engine)
        dest.mkdir(parents=True)
        (dest / dest.name).write_bytes(b"./.nested")
        (dest / dest.name.replace(".nemo", ".bin")).write_bytes(b"stale")
        assert not engine._is_model_cached(dest), "dir になっている .nemo path は hit ではない"

        with patch("huggingface_hub.hf_hub_download", _FakeHfHubDownload(fail=AssertionError("hit"))):
            engine._reconcile_legacy_layouts(dest)
            engine._download_model(dest, None)

        assert dest.is_file() and dest.read_bytes() == b"./.nested"
        assert not any(p.name.startswith(".") for p in roots.models_root.iterdir()), "退避 dir を残さない"
        assert engine._is_model_cached(dest)


def test_no_path_overrides_left():
    """parakeet / reazonspeech の ``load_model()`` override (root → engine subdir へ移してから
    template を呼ぶ = template 側で再ダウンロード) と canary の ``_prepare_model_directory``
    override (``.nemo`` path を dir として返す) は #456 で削除した。"""
    from livecap_cli.engines.base_engine import BaseEngine
    from livecap_cli.engines.canary_engine import CanaryEngine
    from livecap_cli.engines.parakeet_engine import ParakeetEngine

    assert ParakeetEngine.load_model is BaseEngine.load_model
    assert CanaryEngine.load_model is BaseEngine.load_model
    assert CanaryEngine._prepare_model_directory is BaseEngine._prepare_model_directory
    assert ParakeetEngine._prepare_model_directory is BaseEngine._prepare_model_directory


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
