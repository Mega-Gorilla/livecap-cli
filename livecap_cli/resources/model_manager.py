"""Model storage utilities."""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

__all__ = ["ModelManager"]


class ModelManager:
    """Handle model and cache directory access.

    root の**解決**は :mod:`livecap_cli.resources.configuration` が行い、ここは
    解決済みの値を受け取って使うだけである (Issue #375)。

    Phase 1 公開仕様で保証するプロパティ / メソッド:
    - `models_root` / `cache_root`
    - `get_models_dir`
    - `get_temp_dir`
    - `temporary_directory`
    - `get_huggingface_cache_dir` (#428 — 旧 `huggingface_cache()` は `HF_HOME` を
      実行時に書き換えるだけで効いていなかったため削除)

    `download_file()` / `download_file_async()` (`urlretrieve` で `<cache_root>/downloads/` へ直接書く)
    は #456 で削除した — ModelRoot 契約 (staging → manifest → 原子的 publish) の外で、利用者も無かった。
    モデルの取得は `livecap_cli.engines.hf_cache` を使う。
    """

    def __init__(self, *, models_root: Path, cache_root: Path) -> None:
        """解決済みの root を受け取る。

        **env は読まない。** 優先順位の解決は
        :mod:`livecap_cli.resources.configuration` の責務で、ここが env を読むと
        freeze した configuration の外側で root が決まってしまう (Issue #375)。

        構築は :func:`livecap_cli.resources.graph.build_resource_graph` のみが
        行う。root の作成もここで起きる — ``configure_resources()`` は明示指定
        root しか検証せず、preview は filesystem を触らないため。
        """
        self._models_root = models_root
        self._cache_root = cache_root

        self._models_root.mkdir(parents=True, exist_ok=True)
        self._cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def models_root(self) -> Path:
        """Return the root directory where models are stored."""
        return self._models_root

    @property
    def cache_root(self) -> Path:
        """Return the cache directory used for temporary data."""
        return self._cache_root

    def get_models_dir(self) -> Path:
        """``models_root`` を返す (作成込み)。

        engine ごとの subdir (``<models_root>/<engine_name>/``) は #456 で廃止した。旧 workaround が
        そこへ正本を移してから template を呼び、template が root 側で miss して**再ダウンロード**
        していた (二重保持の原因)。正本は常に root 直下の ``<org>--<name>[.nemo]``。
        """
        self._models_root.mkdir(parents=True, exist_ok=True)
        return self._models_root

    def get_temp_dir(self, purpose: str = "runtime") -> Path:
        """Return a temp directory path for the given purpose."""
        path = self._cache_root / purpose
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_huggingface_cache_dir(self) -> Path:
        """``huggingface_hub`` の ``cache_dir=`` に渡す **transient** な管理 cache (Issue #428 / #456)。

        ``<cache_root>/huggingface/hub`` を返す。**完成済みモデルの正本はここではなく
        ``models_root``** (#456: engine は ``hf_cache.fetch_repo_dir()`` で
        ``<models_root>/<org>--<name>/`` へ flattened dir + manifest として publish する)。
        ここは ``local_dir`` モードでも ``huggingface_hub`` が lookup / lock に使うので
        明示的に渡し続ける。staging は ``get_temp_dir("downloads")``。

        **環境変数は触らない。** 以前の ``huggingface_cache()`` は実行時に ``HF_HOME`` を
        書き換えていたが、``huggingface_hub`` は **import 時に cache path を確定する**ので
        効かず、Qwen3-ASR の 1.8 GB は既定の ``~/.cache/huggingface`` へ落ちていた。
        呼び出し側は ``snapshot_download(repo_id, cache_dir=str(<この値>))`` のように
        **明示的に**渡すこと。既定 cache への silent fallback はしない。
        """
        path = self._cache_root / "huggingface" / "hub"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def temporary_directory(self, purpose: str = "downloads") -> Iterator[Path]:
        """
        Provide a temporary directory within the cache tree.
        """
        base = self.get_temp_dir(purpose)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            yield Path(tmp)
