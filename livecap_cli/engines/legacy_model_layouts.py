"""旧配置 (0.1.0 / 0.2.0 / engine subdir) から ModelRoot 契約の正本へ取り込む (Issue #456)。

**なぜ要るか。** 0.1.0 → 0.2.0 で HF cache の階層を 1 度変えており (#453)、#456 でさらに
``cache_root`` → ``models_root`` へ動かす。移設なしで出すと既存ユーザーは同じ 1.8〜5 GB を
**3 度目**に落とすことになる。同じ root の中にある旧配置は cold load 時に自動で取り込む。

対象 (すべて ``configure_resources()`` の root の**中**。root の外 — ``~/.cache/huggingface/hub`` /
``%LOCALAPPDATA%\\whisper_s2t`` — は #453 の範囲で、ここでは触らない):

| 旧配置 | 版 | 例 |
|---|---|---|
| ``<cache_root>/huggingface/hub/models--<org>--<name>/`` (+ ``<models_root>/<org>--<name>.marker``) | 0.2.0 | Qwen3-ASR / WhisperS2T |
| ``<cache_root>/huggingface/hub/transformers/models--…/`` | 0.2.0 | Voxtral |
| ``<cache_root>/huggingface/transformers/models--…/``、``<cache_root>/huggingface/models--…/`` | 0.1.0 | Voxtral / ReazonSpeech |
| ``<models_root>/<engine>/<name>/`` | 旧 workaround が作った engine subdir の重複 | ReazonSpeech |
| ``<models_root>/<engine>/<name>.nemo`` | 同上 | Parakeet / Parakeet JA |
| ``<models_root>/<name>.nemo/<name>.nemo`` (dir の中に同名ファイル) | canary の path 欠陥 | Canary |

規則:

* HF hub 階層の snapshot は **symlink を dereference して**実体化する (``snapshots/<sha>/<file>`` は
  ``../../blobs/<hash>`` への相対 symlink であり得る。dir ごと動かすと旧 cache を指したまま壊れる)
* 実体化 → manifest (``source="migrated"``) → ``publish_dir`` (validate 込み) の**後にだけ**旧側を消す
* 旧側の削除に失敗しても取り込みは成功扱い (ログに残す。次回の scan で見える)
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .model_store import (
    MANIFEST_NAME,
    Manifest,
    adopt_dir,
    build_manifest_from_dir,
    materialize_files,
    publish_dir,
    validate_repo_dir,
)

logger = logging.getLogger(__name__)

__all__ = ["LegacyCandidate", "find_legacy_dirs", "migrate_dir", "migrate_nemo_file", "scan_legacy_layouts"]


@dataclass(frozen=True)
class LegacyCandidate:
    kind: str  # "hub_snapshot" | "flattened_dir"
    source: Path  # 実ファイルを読む dir (snapshot dir または flattened dir)
    cleanup: tuple  # 取り込み成功後に消す path (root の中だけ)
    note: str = ""


# ---------------------------------------------------------------------------
# 探索
# ---------------------------------------------------------------------------


def _hub_roots(cache_root: Path) -> list:
    """旧配置の HF hub 階層が置かれ得る root (新しい版から順に)。"""
    hf = cache_root / "huggingface"
    return [hf / "hub", hf / "hub" / "transformers", hf / "transformers", hf]


def _snapshot_from_hub_repo(repo_dir: Path, marker: Optional[Path], hub_root: Path) -> Optional[Path]:
    """``models--org--name/`` の中で使う snapshot を決める: marker → refs/main → 唯一の snapshot。"""
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    if marker is not None and marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            candidate = (hub_root / payload["snapshot"]).resolve()
            if candidate.is_dir() and candidate.is_relative_to(hub_root.resolve()):
                return candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        try:
            sha = ref.read_text(encoding="utf-8").strip()
            if sha and (snapshots / sha).is_dir():
                return snapshots / sha
        except OSError:
            pass
    dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    return dirs[0] if len(dirs) == 1 else None


def find_legacy_dirs(
    *,
    repo_id: str,
    models_root: Path,
    cache_root: Path,
    destination_name: str,
    engine_subdirs: Sequence[str] = (),
) -> list:
    """flattened dir の正本 (``<models_root>/<destination_name>/``) へ取り込める旧配置を列挙する。"""
    models_root = Path(models_root)
    cache_root = Path(cache_root)
    repo_dirname = "models--" + repo_id.replace("/", "--")
    marker = models_root / f"{repo_id.replace('/', '--')}.marker"
    found: list = []

    for hub_root in _hub_roots(cache_root):
        repo_dir = hub_root / repo_dirname
        if not repo_dir.is_dir():
            continue
        snapshot = _snapshot_from_hub_repo(repo_dir, marker if hub_root == cache_root / "huggingface" / "hub" else None, hub_root)
        if snapshot is None:
            logger.warning(f"旧 HF cache に snapshot を特定できない (触らない): {repo_dir}")
            continue
        cleanup = [repo_dir] + ([marker] if marker.is_file() else [])
        found.append(LegacyCandidate("hub_snapshot", snapshot, tuple(cleanup), note=str(hub_root)))

    for subdir in engine_subdirs:
        dup = models_root / subdir / destination_name
        if dup.is_dir():
            found.append(LegacyCandidate("flattened_dir", dup, (dup,), note=f"engine subdir {subdir}"))
    return found


def _select(names: Iterable[str], allow_patterns, ignore_patterns) -> list:
    out = []
    for name in names:
        if allow_patterns is not None and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
            continue
        if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
            continue
        out.append(name)
    return out


def _remove_legacy(paths: Iterable[Path], *, roots: Sequence[Path]) -> None:
    """取り込み済みの旧側を消す。**root の中にあるものだけ**。失敗はログのみ。"""
    resolved_roots = [Path(r).resolve() for r in roots]
    for path in paths:
        path = Path(path)
        try:
            target = path.resolve()
            if not any(target.is_relative_to(r) for r in resolved_roots):
                logger.warning(f"root の外なので消さない: {path}")
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            logger.info(f"旧配置を削除した (正本へ取り込み済み): {path}")
            _prune_empty_parent(path, stop_at=resolved_roots)
        except OSError as exc:
            logger.warning(f"旧配置を削除できなかった (次回の scan に残る): {path} ({exc})")


def _prune_empty_parent(path: Path, *, stop_at: Sequence[Path]) -> None:
    """engine subdir (``<models_root>/reazonspeech/``) のように、旧配置を消して空になった
    親 dir を 1 段だけ消す。root 自身は消さない。"""
    parent = path.parent
    try:
        if parent.resolve() in stop_at:
            return
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            logger.info(f"空になった旧 dir を削除した: {parent}")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# flattened dir
# ---------------------------------------------------------------------------


def migrate_dir(
    destination: Path,
    *,
    repo_id: str,
    models_root: Path,
    cache_root: Path,
    staging_root: Path,
    required: Sequence[str],
    variant: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    engine_subdirs: Sequence[str] = (),
) -> Optional[Manifest]:
    """``destination`` を正本にできる既存資産があれば取り込み、manifest を返す。無ければ ``None``。

    順に試す: (1) destination 自身が manifest 無しで ``required`` を満たす → その場で採用
    (``adopt_dir``)、(2) 旧配置 (:func:`find_legacy_dirs`) を新しい版から順に実体化 → publish。
    正本が確定したら、同じ repo の旧配置は**すべて**消す (二重保持の解消)。どの候補からも
    取り込めなければ何も消さず ``None`` (呼び出し側が download する)。
    """
    destination = Path(destination)
    candidates = find_legacy_dirs(
        repo_id=repo_id,
        models_root=models_root,
        cache_root=cache_root,
        destination_name=destination.name,
        engine_subdirs=engine_subdirs,
    )
    all_cleanup = [path for c in candidates for path in c.cleanup]

    manifest = adopt_dir(destination, repo_id=repo_id, required=required, variant=variant)
    if manifest is not None:
        # 正本が既にある。同じ repo の旧配置 (二重保持) はもう要らない
        _remove_legacy(all_cleanup, roots=(models_root, cache_root))
        return manifest

    for candidate in candidates:
        names = sorted(
            p.relative_to(candidate.source).as_posix()
            for p in candidate.source.rglob("*")
            if p.is_file() and p.name != MANIFEST_NAME
        )
        selected = _select(names, allow_patterns, ignore_patterns)
        if any(name not in selected for name in required):
            logger.info(f"旧配置は必要ファイルを満たさない (skip): {candidate.source}")
            continue
        payload = Path(staging_root) / f"{destination.name}.migrate-{uuid.uuid4().hex[:8]}"
        try:
            mechanisms = materialize_files(candidate.source, payload, selected)
            manifest = build_manifest_from_dir(payload, repo_id=repo_id, variant=variant, source="migrated")
            (payload / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
            publish_dir(
                payload,
                destination,
                validate=lambda d: validate_repo_dir(d, repo_id=repo_id, variant=variant) is not None,
            )
        except Exception as exc:  # noqa: BLE001 - 次の候補 / download へ進む
            logger.warning(f"旧配置からの取り込みに失敗 (次へ): {candidate.source} ({exc})")
            shutil.rmtree(payload, ignore_errors=True)
            continue
        shutil.rmtree(payload, ignore_errors=True)
        logger.info(
            f"旧配置から正本へ取り込んだ: {candidate.source} -> {destination} "
            f"({candidate.kind}, {len(selected)} files, {set(mechanisms.values())})"
        )
        # 取り込んだ候補だけでなく、同じ repo の他の旧配置 (二重保持) も消す
        _remove_legacy(all_cleanup, roots=(models_root, cache_root))
        return validate_repo_dir(destination, repo_id=repo_id, variant=variant)
    return None


# ---------------------------------------------------------------------------
# single file (.nemo)
# ---------------------------------------------------------------------------


def migrate_nemo_file(
    destination: Path,
    *,
    models_root: Path,
    cache_root: Optional[Path] = None,
    repo_id: Optional[str] = None,
    engine_subdirs: Sequence[str] = (),
) -> bool:
    """``<models_root>/<name>.nemo`` を単一ファイルの正本にする。取り込んだら ``True``。

    * nested (``<name>.nemo/<name>.nemo``、canary の旧 path 欠陥): dir を退避し、中の同名ファイルを
      正本の位置へ ``os.replace``。dir に残った他のファイル (例: ``<name>.bin``) は dir ごと消す
    * engine subdir の重複 (``<models_root>/<engine>/<name>.nemo``): 正本が無ければ移す、
      あれば重複を消す
    * 0.1.0 の HF hub 階層 (``<cache_root>/huggingface/hub/models--<org>--<name>/snapshots/*/<name>.nemo``、
      NeMo の ``from_pretrained`` が落としていた形): 正本が無ければ実体化して publish、あれば repo dir を消す
    """
    destination = Path(destination)
    models_root = Path(models_root)
    name = destination.name
    migrated = False

    if destination.is_dir():
        inner = destination / name
        if inner.is_file():
            parked = destination.with_name(f".{name}.nested-{uuid.uuid4().hex[:8]}")
            os.rename(destination, parked)
            os.replace(parked / name, destination)
            shutil.rmtree(parked, ignore_errors=True)
            logger.info(f"nested な .nemo を正本の位置へ戻した: {destination}")
            migrated = True
        else:
            logger.warning(f".nemo の位置が dir で、中に同名ファイルが無い (触らない): {destination}")

    for subdir in engine_subdirs:
        dup = models_root / subdir / name
        if not dup.is_file():
            continue
        if destination.is_file():
            _remove_legacy([dup], roots=(models_root,))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(dup, destination)
            logger.info(f"engine subdir の .nemo を正本の位置へ移した: {dup} -> {destination}")
            _prune_empty_parent(dup, stop_at=(models_root.resolve(),))
            migrated = True

    if cache_root is not None and repo_id is not None:
        cache_root = Path(cache_root)
        repo_dirname = "models--" + repo_id.replace("/", "--")
        nemo_name = repo_id.split("/")[-1] + ".nemo"
        for hub_root in _hub_roots(cache_root):
            repo_dir = hub_root / repo_dirname
            if not repo_dir.is_dir():
                continue
            if not destination.is_file():
                snapshot = _snapshot_from_hub_repo(repo_dir, None, hub_root)
                source = snapshot / nemo_name if snapshot is not None else None
                if source is None or not source.is_file():
                    logger.warning(f"旧 HF cache に {nemo_name} を特定できない (触らない): {repo_dir}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 同じ dir (= 同じ volume) の temp dir へ実体化 (hardlink → copy) してから原子的に置く
                temp_dir = destination.with_name(f".{name}.{uuid.uuid4().hex[:8]}.part")
                try:
                    materialize_files(source.parent, temp_dir, [source.name])
                    os.replace(temp_dir / source.name, destination)
                except OSError as exc:
                    logger.warning(f"旧 HF cache の .nemo を取り込めなかった (次へ): {source} ({exc})")
                    continue
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"旧 HF cache の .nemo を正本の位置へ取り込んだ: {source} -> {destination}")
                migrated = True
            _remove_legacy([repo_dir], roots=(cache_root,))
    return migrated


# ---------------------------------------------------------------------------
# 可視化 (livecap-cli info)
# ---------------------------------------------------------------------------


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
    except OSError:
        pass
    return total


#: 旧 workaround / warm step が作っていた engine subdir。ここに実体があれば旧配置の重複。
LEGACY_ENGINE_SUBDIRS = ("parakeet", "parakeet_ja", "canary", "reazonspeech", "voxtral", "qwen3asr", "whispers2t")


def scan_legacy_layouts(models_root: Path, cache_root: Path) -> list:
    """root の中に残っている旧配置を (path, bytes) で列挙する (削除はしない)。"""
    models_root = Path(models_root)
    cache_root = Path(cache_root)
    hits: list = []
    hf = cache_root / "huggingface"
    for base in (hf / "hub", hf / "hub" / "transformers", hf / "transformers", hf):
        if base.is_dir():
            for repo_dir in base.glob("models--*"):
                hits.append((repo_dir, _dir_size(repo_dir)))
    downloads = cache_root / "downloads"
    if downloads.is_dir():
        for archive in downloads.glob("*.tar.bz2"):
            hits.append((archive, archive.stat().st_size))
    if models_root.is_dir():
        for marker in models_root.glob("*.marker"):
            hits.append((marker, marker.stat().st_size if marker.is_file() else 0))
        for child in models_root.iterdir():
            if child.is_dir() and child.suffix == ".nemo":
                hits.append((child, _dir_size(child)))
            elif child.is_dir() and ".invalid-" in child.name:
                # publish_dir が隔離した壊れた旧正本 (self-heal / 取得途中)。消すのは利用者の判断
                hits.append((child, _dir_size(child)))
            elif child.is_dir() and (child.name in LEGACY_ENGINE_SUBDIRS or child.name.startswith("whispers2t_")):
                size = _dir_size(child)
                if size:
                    hits.append((child, size))
    return hits
