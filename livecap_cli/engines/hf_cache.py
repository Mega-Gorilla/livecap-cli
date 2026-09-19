"""HuggingFace Hub からの取得を **管理 cache に閉じる**共通 helper (Issue #428 / #430 / #447)。

設計は 1 つだけ: **先にローカルへ解決してから、ローカル path を engine へ渡す。**

* ``fetch_repo_dir`` — repo の必要ファイルを ``snapshot_download(local_dir=<管理 staging>)``
  で取り、**flattened dir + manifest** として ``models_root`` へ原子的に publish する
  (#456。:mod:`livecap_cli.engines.model_store` の契約)。ModelRoot 契約の対象 engine は
  すべてこれを使う
* ``resolve_snapshot`` — (#456 で置き換え中。PR 1 で削除) repo を hub 階層へ解決し marker を書く。
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
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Optional

from .model_store import (
    MANIFEST_NAME,
    build_manifest_from_dir,
    publish_dir,
    validate_repo_dir,
)

logger = logging.getLogger(__name__)

__all__ = [
    "download_file",
    "fetch_repo_dir",
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


def _publish_atomically(source: Path, destination: Path) -> None:
    """``source`` を ``destination`` へ**原子的に**配置し、失敗しても ``source`` を失わない。

    ``models_root`` と ``cache_root`` は別 volume になり得る (``configure_resources()`` で
    独立指定できる)。その場合 ``shutil.move`` は rename ではなく copy → 削除になり、途中で
    落ちると ``destination`` に**途中までの .nemo が残る**。``BaseEngine`` の完全性確認は
    先頭数 byte しか見ないので、truncated file が cache hit として固定されてしまう
    (PR #448 レビュー HIGH)。

    手順:

    1. ``destination`` と同じディレクトリ (= 同じ volume) の一意な temp を作る。
       同一 volume なら ``os.rename(source, temp)`` (瞬時、source は temp へ移る)。
       cross-volume で rename できなければ ``shutil.copy2(source, temp)`` (**source は残す**)
    2. ``os.replace(temp, destination)`` で publish (同一 volume 内の rename なので原子的)
    3. copy した場合だけ、publish 成功後に ``source`` を消す

    どの段階で失敗しても ``destination`` は作られず、**完了済みの download は
    ``source`` (staging) に残る**: rename 後に ``os.replace`` が失敗したら temp を source へ
    戻し、copy の場合は temp を消すだけ (PR #448 再レビュー)。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
    moved = False
    try:
        try:
            os.rename(source, temp)
            moved = True
        except OSError:
            # 別 volume (EXDEV / WinError 17) 等。source を残したまま copy する
            shutil.copy2(source, temp)
        os.replace(temp, destination)
    except BaseException:
        if moved:
            # 完了済み download は temp にある。staging へ戻す (resume 用)。戻せなくても
            # **消さない** — 数 GB の取得結果を失うより、temp の所在をログに残す方がよい
            try:
                os.replace(temp, source)
            except OSError as restore_exc:
                logger.error(
                    f"publish に失敗し、完了済み download を staging へ戻せなかった: {temp} "
                    f"({restore_exc})"
                )
        else:
            temp.unlink(missing_ok=True)
        raise
    if not moved:
        source.unlink(missing_ok=True)


def download_file(
    repo_id: str,
    filename: str,
    *,
    hub_root: Path,
    staging_dir: Path,
    destination: Path,
) -> Path:
    """repo の 1 ファイルを管理 staging へ取り、``destination`` へ原子的に配置する (#447)。

    NeMo の ``from_pretrained()`` は ``hf_hub_download()`` を ``cache_dir=`` 無しで呼ぶので
    既定 HF cache へ落ち、その後 ``save_to()`` で models root へ**もう 1 部**書いていた。
    ここでは ``local_dir=<staging>`` (``<cache_root>/downloads/...``) へ取り、最終位置へ
    publish して staging を消す — **保持するのは 1 部だけ**で、既定 cache には触れない。

    * ``cache_dir=hub_root`` も**明示する**。``local_dir=`` モードでも ``huggingface_hub`` は
      remote へ行く前に ``try_to_load_from_cache(cache_dir=...)`` で cache を探し、省略すると
      既定 ``HF_HUB_CACHE`` を見る — 既定 cache に同じ revision があれば黙ってそこから copy
      する (silent fallback、PR #448 レビュー)。管理 hub を渡せば探すのは管理 hub だけ
      (``local_dir`` モードでは cache 階層へは書かないので、管理 hub に永続 copy は増えない)
    * ``HF_HUB_OFFLINE=1`` で staging に完了済みファイルが無ければ
      ``LocalEntryNotFoundError`` (既定 cache は見ない)
    * **repo 単位の inter-process lock** (``<staging>.lock``、``filelock`` は直接依存として
      ``pyproject.toml`` に宣言) で download → publish → cleanup を直列化する。
      同じ repo を 2 process / 2 engine が同時に cold load しても、後続は lock 取得後に
      ``destination`` の実在を見て取得を skip する (staging を共有したまま ``move`` /
      ``rmtree`` が競合しない)
    * publish は :func:`_publish_atomically` (同一 volume の temp → ``os.replace``)。
      失敗時は ``destination`` を作らない。staging の ``.incomplete`` は resume 用に残す
    * ``.cache/huggingface/`` (metadata) は staging ごと消す
    """
    from filelock import FileLock
    from huggingface_hub import hf_hub_download

    hub_root = Path(hub_root)
    staging_dir = Path(staging_dir)
    destination = Path(destination)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staging_dir.with_name(staging_dir.name + ".lock")

    with FileLock(str(lock_path)):
        if destination.is_file():
            logger.info(f"別の取得が先に配置済み: {destination}")
            return destination

        staging_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ファイルを管理 staging へ取得: repo={repo_id} file={filename} "
            f"local_dir={staging_dir} cache_dir={hub_root}"
        )
        fetched = Path(
            hf_hub_download(
                repo_id, filename=filename, local_dir=str(staging_dir), cache_dir=str(hub_root)
            )
        )
        if not fetched.is_file():
            raise RuntimeError(
                f"取得したファイルが無い: {fetched} (repo={repo_id}, file={filename})"
            )

        _publish_atomically(fetched, destination)
        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info(f"ファイルを配置: {destination}")
    return destination


# ---------------------------------------------------------------------------
# flattened dir (ModelRoot 契約、#456)
# ---------------------------------------------------------------------------


def _staging_metadata(download_dir: Path) -> tuple:
    """``local_dir`` モードが ``download/.cache/huggingface/download/<file>.metadata`` に残す
    ``commit_hash`` / ``etag`` を読む (3 行のテキスト: commit_hash / etag / timestamp)。
    無ければ ``(None, {})``。manifest の ``commit_sha`` / ``files[].etag`` に使うだけで、
    無くても取得は成功する。"""
    meta_root = download_dir / ".cache" / "huggingface" / "download"
    commit: Optional[str] = None
    etags: dict = {}
    if not meta_root.is_dir():
        return commit, etags
    for meta in meta_root.rglob("*.metadata"):
        try:
            lines = meta.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if len(lines) < 2:
            continue
        rel = meta.relative_to(meta_root).as_posix()[: -len(".metadata")]
        commit = commit or (lines[0].strip() or None)
        if lines[1].strip():
            etags[rel] = lines[1].strip()
    return commit, etags


def fetch_repo_dir(
    repo_id: str,
    *,
    hub_root: Path,
    staging_root: Path,
    destination: Path,
    variant: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
    revision: Optional[str] = None,
    required: Optional[Iterable[str]] = None,
) -> Path:
    """repo の必要ファイルを **flattened dir + manifest** として ``destination`` へ配置する (#456)。

    ::

        <staging_root>/<org>--<name>/
          download/   snapshot_download(local_dir=ここ, cache_dir=<hub_root>, allow/ignore_patterns)
                      — huggingface_hub は local_dir 直下に .cache/huggingface/download/*.metadata /
                        .lock / *.incomplete を作る (実測)。ここに閉じ込める
          payload/    download/ から必要ファイルだけを rename で集め、manifest を書く
        <destination>/   payload/ だけを publish_dir() で原子的に配置

    * ``cache_dir=hub_root`` を**明示する** — ``local_dir`` モードでも ``huggingface_hub`` は
      remote へ行く前に ``try_to_load_from_cache(cache_dir=)`` を引き、省略すると既定
      ``HF_HUB_CACHE`` から silent fallback する (PR #448 レビュー)
    * ``HF_HUB_OFFLINE=1`` で staging に完了済みファイルが無ければ ``LocalEntryNotFoundError``
      (既定 cache は見ない)
    * ``max_workers=1`` (hf_hub 0.36.0 / 1.31.0 の symlink 判定 race、huggingface_hub#4915)
    * repo 単位の ``FileLock`` で download → publish → cleanup を直列化。後続は lock 取得後に
      ``destination`` が valid なら取得を skip
    * 成功したら staging を消す。**失敗時は残す** (``download/`` の ``.incomplete`` +
      metadata を次回 resume に使う。publish で失敗したときは完成済み ``payload/`` を次回そのまま
      publish し、再取得しない)。destination はどの段階で失敗しても作られない
    * ``required``: publish 前に payload に必ず要るファイル名 (無ければ fail loud。
      ``allow_patterns`` が何もマッチしなかった、repo の構成が変わった、等)
    """
    from filelock import FileLock
    from huggingface_hub import snapshot_download

    hub_root = Path(hub_root)
    staging_root = Path(staging_root)
    destination = Path(destination)
    staging = staging_root / repo_id.replace("/", "--")
    download_dir = staging / "download"
    payload_dir = staging / "payload"
    lock_path = staging.with_name(staging.name + ".lock")
    staging_root.mkdir(parents=True, exist_ok=True)

    def _valid(path: Path) -> bool:
        # 現在の required も含めて判定する: 完了済み payload の再利用 / destination の hit で、
        # 以前の呼び出しには無かった必須ファイルを欠いた dir を正本にしない (PR #457 再レビュー)
        return validate_repo_dir(path, repo_id=repo_id, variant=variant, required=required) is not None

    with FileLock(str(lock_path)):
        if _valid(destination):
            logger.info(f"別の取得が先に配置済み: {destination}")
            return destination

        # 前回 download は完了したが publish で失敗した場合、payload/ (manifest 込み) が
        # staging に残っている。**再取得せず**そこから publish をやり直す (PR #457 レビュー)
        if _valid(payload_dir):
            logger.info(f"完了済みの payload から publish を再試行: {payload_dir} -> {destination}")
            publish_dir(payload_dir, destination, validate=_valid)
            shutil.rmtree(staging, ignore_errors=True)
            return destination

        download_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"local_dir": str(download_dir), "cache_dir": str(hub_root), "max_workers": 1}
        if allow_patterns is not None:
            kwargs["allow_patterns"] = list(allow_patterns)
        if ignore_patterns is not None:
            kwargs["ignore_patterns"] = list(ignore_patterns)
        if revision is not None:
            kwargs["revision"] = revision
        logger.info(
            f"repo を管理 staging へ取得: repo={repo_id} local_dir={download_dir} cache_dir={hub_root}"
        )
        snapshot_download(repo_id, **kwargs)

        # download/ → payload/ (必要ファイルだけ。.cache/ と transient は持ち込まない)
        if payload_dir.exists():
            shutil.rmtree(payload_dir, ignore_errors=True)
        payload_dir.mkdir(parents=True, exist_ok=True)
        moved_any = False
        for path in sorted(download_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(download_dir)
            if rel.parts[0] == ".cache" or rel.name.endswith((".incomplete", ".lock", ".metadata")):
                continue
            target = payload_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(path, target)
            moved_any = True
        if not moved_any:
            raise RuntimeError(f"取得したファイルが無い (patterns が何にも一致しない?): repo={repo_id}")
        for name in required or ():
            if not (payload_dir / name).is_file():
                raise RuntimeError(f"必要ファイルが無い: {name} (repo={repo_id}, payload={payload_dir})")

        commit_sha, etags = _staging_metadata(download_dir)
        manifest = build_manifest_from_dir(
            payload_dir,
            repo_id=repo_id,
            variant=variant,
            revision=revision,
            commit_sha=commit_sha,
            source="download",
            etags=etags,
        )
        (payload_dir / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")

        publish_dir(payload_dir, destination, validate=_valid)
        shutil.rmtree(staging, ignore_errors=True)
        logger.info(f"repo を配置: {destination} ({len(manifest.files)} files, commit={commit_sha})")
    return destination
