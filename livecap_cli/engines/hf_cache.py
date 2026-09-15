"""HuggingFace Hub からの取得を **管理 cache に閉じる**共通 helper (Issue #428 / #430 / #447)。

設計は 1 つだけ: **先にローカルへ解決してから、ローカル path を engine へ渡す。**

* ``resolve_snapshot`` — repo 全体 (または ``allow_patterns`` の一部) を
  ``snapshot_download(cache_dir=<管理 hub>)`` で解決し、**成功後にのみ** marker を書く。
  Qwen3-ASR (#428) と WhisperS2T (#430) が使う
* ``download_file`` — 単一ファイル (NeMo の ``.nemo``、#447) を
  ``hf_hub_download(local_dir=<管理 staging>)`` で取り、最終位置へ move する。
  既定 HF cache には落とさず、**1 部しか保持しない**

共通の約束:

* **環境変数は触らない。** ``huggingface_hub`` は import 時に cache path を確定するので
  ``HF_HOME`` の実行時変更は効かない。管理 cache は ``cache_dir=`` / ``local_dir=`` で
  **明示的に**渡す (``ModelManager.get_huggingface_cache_dir()`` / ``get_temp_dir()``)
* **既定 cache への silent fallback はしない。** ``HF_HUB_OFFLINE=1`` で管理 cache に
  無ければ ``LocalEntryNotFoundError`` で fail loud
* marker は「どの snapshot を使うか」の記録であって、存在だけで cache hit にはしない
  (:func:`read_marker` の規則)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "download_file",
    "invalidate_marker",
    "read_marker",
    "resolve_snapshot",
    "write_marker",
]

#: snapshot の完全性確認で必ず要求するファイル。HF の transformers 系 / CTranslate2 系
#: とも ``config.json`` を持つ。
REQUIRED_FILE = "config.json"


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------


def write_marker(marker: Path, hub_root: Path, snapshot: Path) -> None:
    """marker を書く。**hub root からの相対 path** と、snapshot 内の全ファイルの一覧。

    絶対 path を書かないのは、cache root を変えた (``configure_resources(cache_dir=B)``)
    後に旧 root A の snapshot を cache hit として使い続けないため — marker は
    **現在の** hub root からしか解決しない (PR #446 レビュー指摘)。ファイル一覧は
    cache hit の完全性確認に使う (``config.json`` だけでは重み欠損を見逃す)。
    """
    hub_root = hub_root.resolve()
    snapshot = snapshot.resolve()
    relative = snapshot.relative_to(hub_root)  # 配下でなければ ValueError (呼び出し側で検査済み)
    files = sorted(
        p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"snapshot": relative.as_posix(), "files": files}, ensure_ascii=False),
        encoding="utf-8",
    )


def read_marker(marker: Path, hub_root: Path) -> Optional[Path]:
    """marker が指す snapshot を**現在の** ``hub_root`` 配下で解決する。

    次のいずれかなら ``None`` (= cache miss、再解決へ):

    * marker が無い / JSON でない (#428 以前の ``model=...`` 形式もここ)
    * 相対 path が ``hub_root`` の外へ出る (``..`` など)
    * snapshot に ``config.json`` が無い
    * marker に記録したファイルのどれかが無い (削除 / 壊れた symlink)

    既定 cache からは**移設しない** — 旧 marker は miss になり管理 cache へ再解決される。
    """
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("snapshot"), str):
        return None
    files = payload.get("files")
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        return None
    hub_root = hub_root.resolve()
    snapshot = (hub_root / payload["snapshot"]).resolve()
    if not snapshot.is_relative_to(hub_root):
        return None
    if not (snapshot / REQUIRED_FILE).is_file():
        return None
    if not all((snapshot / f).is_file() for f in files):
        return None
    return snapshot


def invalidate_marker(marker: Path, *, reason: str) -> None:
    """marker を消して次回 ``load_model()`` で再解決させる (self-heal)。

    manifest に無い形で snapshot が壊れて ``from_pretrained`` / ``load_model`` が落ちた
    とき、marker を残すと以後ダウンロード phase を永久に skip して落ち続ける。
    """
    marker.unlink(missing_ok=True)
    logger.warning(f"marker を無効化した (次回再解決): {marker} - {reason}")


# ---------------------------------------------------------------------------
# snapshot (repo 全体)
# ---------------------------------------------------------------------------


def resolve_snapshot(
    repo_id: str,
    *,
    hub_root: Path,
    marker: Path,
    allow_patterns: Optional[Iterable[str]] = None,
) -> Path:
    """repo の snapshot を管理 hub へ解決し、成功したら marker を書いて snapshot を返す。

    * ``cache_dir=hub_root`` を**明示**する。``HF_HUB_OFFLINE=1`` なら管理 cache だけから
      解決し、無ければ ``LocalEntryNotFoundError`` (既定 cache は見ない)
    * **marker は成功後にのみ書く。** 失敗時に marker を残すと次回 cache hit になる
    * ``max_workers=1``: huggingface_hub 0.36.0 / 1.31.0 は fresh な cache dir へ複数
      worker で落とすと symlink 可否の判定 (``are_symlinks_supported``) が thread 間で
      競合し、Windows (Developer Mode 無し) では ``WinError 1314`` で落ちる
      (実測、上流報告: huggingface/huggingface_hub#4915)。1 worker なら degraded
      (実ファイル) モードで正常に書ける。速度への影響は未計測で、安定性との
      trade-off として採用している
    """
    from huggingface_hub import snapshot_download

    hub_root = Path(hub_root)
    kwargs = {"cache_dir": str(hub_root), "max_workers": 1}
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    logger.info(f"snapshot を管理 cache へ解決: repo={repo_id} cache_dir={hub_root}")

    snapshot = Path(snapshot_download(repo_id, **kwargs)).resolve()
    if not snapshot.is_relative_to(hub_root.resolve()):
        raise RuntimeError(
            f"snapshot が管理 cache の外にある: {snapshot} (repo={repo_id}, cache_dir={hub_root})"
        )
    if not (snapshot / REQUIRED_FILE).is_file():
        raise RuntimeError(
            f"snapshot に {REQUIRED_FILE} が無い: {snapshot} (repo={repo_id}, cache_dir={hub_root})"
        )
    write_marker(marker, hub_root, snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# 単一ファイル (.nemo など)
# ---------------------------------------------------------------------------


def download_file(
    repo_id: str,
    filename: str,
    *,
    staging_dir: Path,
    destination: Path,
) -> Path:
    """repo の 1 ファイルを管理 staging へ取り、``destination`` へ move する (#447)。

    NeMo の ``from_pretrained()`` は ``hf_hub_download()`` を ``cache_dir=`` 無しで呼ぶので
    既定 HF cache へ落ち、その後 ``save_to()`` で models root へ**もう 1 部**書いていた。
    ここでは ``local_dir=<staging>`` (``<cache_root>/downloads/...``) へ取り、最終位置へ
    move して staging を消す — **保持するのは 1 部だけ**で、既定 cache には触れない。

    * ``HF_HUB_OFFLINE=1`` で staging に完了済みファイルが無ければ
      ``LocalEntryNotFoundError`` (既定 cache は見ない)
    * 失敗時は ``destination`` を作らない。staging の ``.incomplete`` は resume 用に残す
    * ``.cache/huggingface/`` (metadata) は staging ごと消す
    """
    from huggingface_hub import hf_hub_download

    staging_dir = Path(staging_dir)
    destination = Path(destination)
    staging_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"ファイルを管理 staging へ取得: repo={repo_id} file={filename} local_dir={staging_dir}"
    )

    fetched = Path(hf_hub_download(repo_id, filename=filename, local_dir=str(staging_dir)))
    if not fetched.is_file():
        raise RuntimeError(f"取得したファイルが無い: {fetched} (repo={repo_id}, file={filename})")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(fetched), str(destination))
    shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info(f"ファイルを配置: {destination}")
    return destination
