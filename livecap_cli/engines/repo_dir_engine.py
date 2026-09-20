"""正本が **flattened dir + manifest** の engine に共通する Template Method の実装 (Issue #456)。

Qwen3-ASR / WhisperS2T / Voxtral / ReazonSpeech は「HF repo の必要ファイルを
``<models_root>/<org>--<name>/`` に置き、manifest (:mod:`livecap_cli.engines.model_store`) で
cache hit を決め、旧配置を取り込み (:mod:`livecap_cli.engines.legacy_model_layouts`)、
``fetch_repo_dir`` で取得する」という同じ契約を持つ。engine ごとに違うのは **repo / variant /
取るファイル / 必須ファイル / 旧 engine subdir** だけなので、それを :class:`RepoDirSpec` に
まとめ、4 つの step (:meth:`_is_model_cached` / :meth:`_verify_model_integrity` /
:meth:`_reconcile_legacy_layouts` / :meth:`_download_model`) をここで 1 度だけ実装する。

使い方::

    class FooEngine(RepoDirModelMixin, BaseEngine):
        def _repo_dir_spec(self) -> RepoDirSpec:
            return RepoDirSpec(repo_id=self.model_name, required=("config.json", "model.safetensors"))

        def _load_model_from_path(self, model_path):
            model_dir = self._require_model_dir(model_path)   # validate してから渡す
            try:
                return Foo.from_pretrained(str(model_dir))
            except Exception:
                self._invalidate_model_dir(model_path, reason="Foo from_pretrained failed")  # self-heal
                raise

mixin は :class:`~livecap_cli.engines.base_engine.BaseEngine` の**前**に置く (MRO で template の
既定実装を上書きするため)。``self.model_manager`` / ``self.report_progress`` は BaseEngine のもの。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .hf_cache import fetch_repo_dir
from .legacy_model_layouts import migrate_dir
from .model_store import Manifest, invalidate_manifest, validate_repo_dir

logger = logging.getLogger(__name__)

__all__ = ["RepoDirModelMixin", "RepoDirSpec"]


@dataclass(frozen=True)
class RepoDirSpec:
    """engine が正本 dir をどう作り、何を cache hit の条件にするか。"""

    #: HuggingFace repo id (``org/name``)。manifest の ``repo_id`` と照合する
    repo_id: str
    #: cache hit に必ず要るファイル名 (manifest に記録され、通常ファイルとして実在すること)
    required: Tuple[str, ...]
    #: 同じ repo の別バリアント (WhisperS2T の size、ReazonSpeech の int8 / float32)。manifest と照合
    variant: Optional[str] = None
    #: repo から取るファイル (``snapshot_download(allow_patterns=)``)。``None`` = 全部
    allow_patterns: Optional[Tuple[str, ...]] = None
    #: repo から取らないファイル (``snapshot_download(ignore_patterns=)``)
    ignore_patterns: Optional[Tuple[str, ...]] = None
    #: 旧 workaround が作っていた engine subdir (``<models_root>/<subdir>/<dir_name>``)。取り込んで消す
    legacy_subdirs: Tuple[str, ...] = ()

    @property
    def dir_name(self) -> str:
        """既定の正本 dir 名 ``<org>--<name>``。"""
        return self.repo_id.replace("/", "--")


class RepoDirModelMixin:
    """flattened dir + manifest を正本とする engine の共通 step。:meth:`_repo_dir_spec` だけ実装する。"""

    def _repo_dir_spec(self) -> RepoDirSpec:
        raise NotImplementedError

    # --- path / cache hit -------------------------------------------------------------

    def _get_local_model_path(self, models_dir: Path) -> Path:
        """既定は ``<models_root>/<org>--<name>/``。dir 名を変える engine は override する。"""
        return Path(models_dir) / self._repo_dir_spec().dir_name

    def _validate_model_dir(self, model_path: Path) -> Optional[Manifest]:
        """spec の repo_id / variant / required で :func:`validate_repo_dir` を通す。"""
        spec = self._repo_dir_spec()
        return validate_repo_dir(
            model_path, repo_id=spec.repo_id, variant=spec.variant, required=spec.required
        )

    def _is_model_cached(self, model_path: Path) -> bool:
        """manifest の全ファイルがサイズ一致で実在し、required が揃うときだけ hit (#456)。"""
        return self._validate_model_dir(model_path) is not None

    def _verify_model_integrity(self, model_path: Path) -> bool:
        return self._validate_model_dir(model_path) is not None

    def _require_model_dir(self, model_path: Path) -> Path:
        """load 直前の確認: 正本が揃っていなければ fail loud (repo ID へ silent fallback しない)。"""
        if self._validate_model_dir(model_path) is None:
            raise RuntimeError(f"{self._repo_dir_spec().repo_id} の正本 dir が揃っていない: {model_path}")
        logger.info(f"{self._repo_dir_spec().repo_id} をローカル dir からロード: {model_path}")
        return model_path

    def _invalidate_model_dir(self, model_path: Path, *, reason: str) -> None:
        """self-heal: load に失敗した dir の manifest を無効化し、次回の取得で隔離 → 再取得させる。"""
        invalidate_manifest(model_path, reason=reason)

    # --- template steps ----------------------------------------------------------------

    def _reconcile_legacy_layouts(self, model_path: Path) -> None:
        """旧配置 (0.1.0 / 0.2.0 の hub snapshot + marker、engine subdir の重複、manifest 無しの
        既存 dir) を正本へ取り込み、正本が確定したら重複を消す。cache 判定の前に毎回呼ばれる。"""
        spec = self._repo_dir_spec()
        manager = self.model_manager
        migrate_dir(
            model_path,
            repo_id=spec.repo_id,
            models_root=manager.models_root,
            cache_root=manager.cache_root,
            staging_root=manager.get_temp_dir("downloads"),
            required=spec.required,
            variant=spec.variant,
            allow_patterns=spec.allow_patterns,
            ignore_patterns=spec.ignore_patterns,
            engine_subdirs=spec.legacy_subdirs,
        )

    def _download_model(self, target_path: Path, progress_callback, model_manager=None) -> None:
        """Step 3 (20-70%): repo の必要ファイルを staging 経由で正本 dir へ publish する。

        取得の規則 (staging / manifest / atomic publish / offline / ``max_workers=1`` / lock) は
        :func:`livecap_cli.engines.hf_cache.fetch_repo_dir` を参照。
        """
        spec = self._repo_dir_spec()
        manager = model_manager or self.model_manager
        self.report_progress(25, f"Downloading into managed model root: {spec.repo_id}")
        fetch_repo_dir(
            spec.repo_id,
            hub_root=manager.get_huggingface_cache_dir(),
            staging_root=manager.get_temp_dir("downloads"),
            destination=target_path,
            variant=spec.variant,
            allow_patterns=spec.allow_patterns,
            ignore_patterns=spec.ignore_patterns,
            required=spec.required,
        )
        self.report_progress(70, f"Model ready: {target_path}")
