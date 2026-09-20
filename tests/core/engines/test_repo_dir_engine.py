"""``RepoDirModelMixin`` — flattened dir + manifest が正本の engine が共有する Template Method 実装 (#456)。

engine ごとの挙動は ``test_*_managed_cache.py`` が production 経路で固定する。ここでは
「4 engine が同じ 1 つの実装を使っている」ことと、mixin 単体の契約 (既定 path / required 付き
validate / fail loud / self-heal) を固定する — engine 側に override が再び生えて挙動が分岐する
(= #456 以前の 4 重複製) のを防ぐ。
"""

from __future__ import annotations

import pytest

from livecap_cli.engines import model_store as ms
from livecap_cli.engines.base_engine import BaseEngine
from livecap_cli.engines.repo_dir_engine import RepoDirModelMixin, RepoDirSpec
from tests.core.engines.conftest import write_repo_dir

REPO = "org/model"
FILES = {"config.json": b"{}", "model.bin": b"w" * 32}


class _Fake(RepoDirModelMixin):
    """BaseEngine 抜きで mixin だけを試す (model_manager / report_progress は使わない経路)。"""

    def __init__(self, spec: RepoDirSpec):
        self.spec = spec

    def _repo_dir_spec(self) -> RepoDirSpec:
        return self.spec


class TestSpec:
    def test_dir_name_flattens_repo_id(self):
        assert RepoDirSpec("Qwen/Qwen3-ASR-0.6B", ("config.json",)).dir_name == "Qwen--Qwen3-ASR-0.6B"

    def test_default_local_path_is_models_root_slash_dir_name(self, tmp_path):
        fake = _Fake(RepoDirSpec(REPO, ("config.json",)))
        assert fake._get_local_model_path(tmp_path) == tmp_path / "org--model"


class TestValidation:
    def test_cache_hit_requires_repo_variant_and_required(self, tmp_path):
        d = write_repo_dir(tmp_path / "org--model", FILES, repo_id=REPO, variant="base")
        assert _Fake(RepoDirSpec(REPO, ("config.json", "model.bin"), variant="base"))._is_model_cached(d)
        assert not _Fake(RepoDirSpec(REPO, ("config.json",), variant="small"))._is_model_cached(d), "variant 不一致"
        assert not _Fake(RepoDirSpec("org/other", ("config.json",), variant="base"))._is_model_cached(d), "repo 不一致"
        assert not _Fake(RepoDirSpec(REPO, ("tokenizer.json",), variant="base"))._is_model_cached(d), "required 欠落"
        assert _Fake(RepoDirSpec(REPO, ("config.json",), variant="base"))._verify_model_integrity(d)

    def test_require_model_dir_fails_loud(self, tmp_path):
        fake = _Fake(RepoDirSpec(REPO, ("config.json",)))
        with pytest.raises(RuntimeError, match="正本 dir が揃っていない"):
            fake._require_model_dir(tmp_path / "missing")
        d = write_repo_dir(tmp_path / "org--model", FILES, repo_id=REPO)
        assert fake._require_model_dir(d) == d

    def test_invalidate_writes_tombstone(self, tmp_path):
        d = write_repo_dir(tmp_path / "org--model", FILES, repo_id=REPO)
        fake = _Fake(RepoDirSpec(REPO, ("config.json",)))
        fake._invalidate_model_dir(d, reason="load failed")
        assert not fake._is_model_cached(d)
        assert ms.read_manifest(d).source == ms.INVALIDATED_SOURCE


class TestEnginesShareTheMixin:
    """4 engine が同じ実装を使う — override が生えたらここで気付く。"""

    @pytest.fixture(params=["qwen3asr", "whispers2t", "voxtral", "reazonspeech"])
    def engine_cls(self, request):
        mod = pytest.importorskip(f"livecap_cli.engines.{request.param}_engine")
        return {
            "qwen3asr": getattr(mod, "Qwen3ASREngine", None),
            "whispers2t": getattr(mod, "WhisperS2TEngine", None),
            "voxtral": getattr(mod, "VoxtralEngine", None),
            "reazonspeech": getattr(mod, "ReazonSpeechEngine", None),
        }[request.param]

    def test_mixin_precedes_base_engine(self, engine_cls):
        mro = engine_cls.__mro__
        assert mro.index(RepoDirModelMixin) < mro.index(BaseEngine)

    @pytest.mark.parametrize("name", ["_is_model_cached", "_verify_model_integrity", "_validate_model_dir", "_require_model_dir", "_invalidate_model_dir"])
    def test_cache_judgement_is_not_overridden(self, engine_cls, name):
        assert getattr(engine_cls, name) is getattr(RepoDirModelMixin, name), (
            f"{engine_cls.__name__}.{name}: cache 判定は mixin の 1 実装に集約する (#456)"
        )

    def test_download_and_reconcile_go_through_the_mixin(self, engine_cls):
        """override する engine (ReazonSpeech の tarball 削除) も mixin の実装を super() で通す。"""
        import inspect

        for name in ("_download_model", "_reconcile_legacy_layouts"):
            impl = getattr(engine_cls, name)
            if impl is getattr(RepoDirModelMixin, name):
                continue
            assert "super()." + name in inspect.getsource(impl), f"{engine_cls.__name__}.{name} が mixin を迂回している"
