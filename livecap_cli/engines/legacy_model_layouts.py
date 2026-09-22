"""旧配置 (0.1.0 / 0.2.0 / engine subdir) から ModelRoot 契約の正本へ取り込む (Issue #456)。

**なぜ要るか。** 0.1.0 → 0.2.0 で HF cache の階層を 1 度変えており (#453)、#456 でさらに
``cache_root`` → ``models_root`` へ動かす。移設なしで出すと既存ユーザーは同じ 1.8〜5 GB を
**3 度目**に落とすことになる。同じ root の中にある旧配置は cold load 時に自動で取り込む。

対象 (``configure_resources()`` の root の**中**。root の**外**にある旧 cache は次の表の後):

| 旧配置 | 版 | 例 |
|---|---|---|
| ``<cache_root>/huggingface/hub/models--<org>--<name>/`` (+ ``<models_root>/<org>--<name>.marker``) | 0.2.0 | Qwen3-ASR / WhisperS2T |
| ``<cache_root>/huggingface/hub/transformers/models--…/`` | 0.2.0 | Voxtral |
| ``<cache_root>/huggingface/transformers/models--…/``、``<cache_root>/huggingface/models--…/`` | 0.1.0 | Voxtral / ReazonSpeech |
| ``<models_root>/<engine>/<name>/`` | 旧 workaround が作った engine subdir の重複 | ReazonSpeech |
| ``<models_root>/<engine>/<name>.nemo`` | 同上 | Parakeet / Parakeet JA |
| ``<models_root>/<name>.nemo/<name>.nemo`` (dir の中に同名ファイル) | canary の path 欠陥 | Canary |
| ``<cache_root>/huggingface/hub/models--<org>--<name>/snapshots/*/<name>.nemo`` | 0.1.0 (NeMo の ``from_pretrained``) | Parakeet / Canary |
| ``<cache_root>/downloads/*.tar.bz2`` | 旧 int8 経路の download archive | ReazonSpeech |

root の**外** (#453、:func:`external_hub_roots`) — 0.1.0 以前の cli / 0.2.0 までの翻訳が落としていた場所:

| 旧 cache | 版 | 例 |
|---|---|---|
| 既定 HF cache ``huggingface_hub.constants.HF_HUB_CACHE`` (``~/.cache/huggingface/hub``) の ``models--…/`` | 0.1.0 (Qwen3-ASR / ReazonSpeech / NeMo ``.nemo``)、0.2.0 (Riva) | ``models--Qwen--Qwen3-ASR-0.6B`` |
| whisper_s2t の自前 cache ``platformdirs.user_cache_dir("whisper_s2t")/models`` | 0.1.0 (WhisperS2T) | ``models--Systran--faster-whisper-base`` |

root の外は**他アプリと共用**なので、候補としては root の中のものの後に並べ、取り込みは
hardlink / copy だけで、**削除は絶対にしない** (``LegacyCandidate.cleanup`` が空)。可視化は
:func:`scan_external_caches` (``livecap-cli info`` の ``External model caches`` 行)。

規則:

* HF hub 階層の snapshot は **symlink を dereference して**実体化する (``snapshots/<sha>/<file>`` は
  ``../../blobs/<hash>`` への相対 symlink であり得る。dir ごと動かすと旧 cache を指したまま壊れる)
* 実体化 → manifest (``source="migrated"``) → ``publish_dir`` (validate 込み) の**後にだけ**旧側を消す。
  単一ファイル (``.nemo``) も同じ: engine の validator を通る候補だけを正本にし、配置後にもう一度
  validate してから旧側を消す。validator を通らない候補は**触らない** (残骸は ``scan_legacy_layouts``
  = ``livecap-cli info`` に出る)
* validator を通らない正本 (truncated / nested の中身が壊れている / 同名ファイルの無い ``.nemo/`` dir)
  は :func:`model_store.quarantine` で ``<name>.invalid-<ts>`` へ隔離する (削除しない)
* 取り込みと取得は destination 単位の同じ lock (:func:`model_store.model_lock`) を取る
* ``snapshots/`` / ``blobs/`` にファイルの無い ``models--*`` は、新方式の
  ``snapshot_download(local_dir=, cache_dir=)`` が ``cache_dir`` 側に残す refs だけの metadata
  (許可された transient) なので、旧配置として列挙も取り込みもしない
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
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

from .model_store import (
    MANIFEST_NAME,
    Manifest,
    adopt_dir,
    build_manifest_from_dir,
    materialize_files,
    model_lock,
    publish_dir,
    publish_file,
    quarantine,
    validate_repo_dir,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExternalCacheHit",
    "ExternalCacheRoot",
    "KNOWN_MODEL_REPOS",
    "LegacyCandidate",
    "external_hub_roots",
    "find_legacy_dirs",
    "migrate_dir",
    "migrate_nemo_file",
    "remove_legacy_archives",
    "scan_external_caches",
    "scan_legacy_layouts",
]


@dataclass(frozen=True)
class LegacyCandidate:
    """flattened dir の正本へ取り込める旧配置 (:func:`find_legacy_dirs`)。"""

    kind: str  # "hub_snapshot" | "flattened_dir"
    source: Path  # 実ファイルを読む dir (snapshot dir または flattened dir)
    cleanup: Tuple[Path, ...]  # 取り込み成功後に消す path (root の中だけ)。root の外の候補は空
    note: str = ""


@dataclass(frozen=True)
class _NemoCandidate:
    """単一 ``.nemo`` の正本へ取り込める旧配置 (:func:`migrate_nemo_file`)。"""

    source: Path  # 旧 .nemo ファイル
    cleanup: Optional[Path]  # 取り込み後に消す path (ファイル自身、または hub の repo dir)。root の外の候補は None
    root: Path  # cleanup が必ずこの root の中にあること (外なら消さない)


@dataclass(frozen=True)
class ExternalCacheRoot:
    """root の外にある hub 階層の cache (:func:`external_hub_roots`)。読むだけ — 作らない、消さない。"""

    label: str  # "default HF cache" | "whisper_s2t cache"
    path: Path


# ---------------------------------------------------------------------------
# 探索
# ---------------------------------------------------------------------------


def _hub_roots(cache_root: Path) -> List[Path]:
    """旧配置の HF hub 階層が置かれ得る root (新しい版から順に)。"""
    hf = cache_root / "huggingface"
    return [hf / "hub", hf / "hub" / "transformers", hf / "transformers", hf]


def external_hub_roots() -> List[ExternalCacheRoot]:
    """root の**外**で、旧版の cli が hub 階層 (``models--<org>--<name>/snapshots/…``) を落としていた場所 (#453)。

    * 既定 HF cache: ``huggingface_hub.constants.HF_HUB_CACHE``。``HF_HOME`` / ``HF_HUB_CACHE`` env の
      解決は huggingface_hub 自身のもの (import 時に確定) を使う
    * whisper_s2t の自前 cache: ``platformdirs.user_cache_dir("whisper_s2t")/models``
      (``%LOCALAPPDATA%\\whisper_s2t\\whisper_s2t\\Cache\\models``)。platformdirs は whisper_s2t の依存で、
      無ければ whisper_s2t も無いので対象外。``import whisper_s2t`` はしない (upstream が import 時に
      ``os.makedirs`` する)

    存在しない path も返す (呼び出し側が ``is_dir()`` で見る)。dir は作らない。
    """
    from huggingface_hub import constants as hf_constants

    roots = [ExternalCacheRoot("default HF cache", Path(hf_constants.HF_HUB_CACHE))]
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        return roots
    roots.append(ExternalCacheRoot("whisper_s2t cache", Path(user_cache_dir("whisper_s2t")) / "models"))
    return roots


def _external_roots_outside(models_root: Path, cache_root: Path) -> List[ExternalCacheRoot]:
    """:func:`external_hub_roots` のうち、設定した root の**外**にあるものだけ。

    ``HF_HOME`` を ``cache_root`` の中に向けている環境では既定 HF cache が
    ``<cache_root>/huggingface/hub`` (root の中の旧配置) と同じ path になる。二重に列挙すると
    「root の中 (消す)」と「外 (消さない)」の扱いが衝突するので、中に解決するものは外として扱わない。
    """
    inside = [Path(models_root).resolve(), Path(cache_root).resolve()]
    out: List[ExternalCacheRoot] = []
    for root in external_hub_roots():
        try:
            resolved = root.path.resolve()
        except OSError:
            continue
        if any(resolved == base or resolved.is_relative_to(base) for base in inside):
            continue
        out.append(root)
    return out


def _hub_repo_has_payload(repo_dir: Path) -> bool:
    """``models--org--name/`` が旧配置 (実体を持つ) か。

    ``huggingface_hub`` の ``snapshot_download(local_dir=..., cache_dir=...)`` は新方式の fresh download
    でも ``cache_dir`` 側に ``models--<repo>/refs/main`` だけの metadata を残す (実測 7 KB)。
    ``snapshots/`` / ``blobs/`` にファイルが無い repo dir は**許可された transient** であり、
    旧配置として列挙も取り込みもしない (PR #458 レビュー MEDIUM)。
    """
    for sub in ("snapshots", "blobs"):
        base = repo_dir / sub
        if base.is_dir() and any(p.is_file() for p in base.rglob("*")):
            return True
    return False


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
    legacy_names: Sequence[str] = (),
) -> List[LegacyCandidate]:
    """flattened dir の正本 (``<models_root>/<destination_name>/``) へ取り込める旧配置を列挙する。

    ``legacy_names`` は #456 以前の正本 dir 名 (``<models_root>/<legacy_name>``、例: ReazonSpeech int8 の
    tarball 由来の名前)。engine subdir の重複もその名前で探す。

    root の**外** (:func:`external_hub_roots`) の hub snapshot は最後に並べ、``cleanup`` は空
    (取り込むだけで消さない、#453)。
    """
    models_root = Path(models_root)
    cache_root = Path(cache_root)
    repo_dirname = "models--" + repo_id.replace("/", "--")
    marker = models_root / f"{repo_id.replace('/', '--')}.marker"
    found: List[LegacyCandidate] = []

    for hub_root in _hub_roots(cache_root):
        repo_dir = hub_root / repo_dirname
        if not repo_dir.is_dir() or not _hub_repo_has_payload(repo_dir):
            continue  # 無い、または refs/ だけの transient metadata (新方式の lookup 跡)
        snapshot = _snapshot_from_hub_repo(repo_dir, marker if hub_root == cache_root / "huggingface" / "hub" else None, hub_root)
        if snapshot is None:
            logger.warning(f"旧 HF cache に snapshot を特定できない (触らない): {repo_dir}")
            continue
        cleanup = [repo_dir] + ([marker] if marker.is_file() else [])
        found.append(LegacyCandidate("hub_snapshot", snapshot, tuple(cleanup), note=str(hub_root)))

    for legacy_name in legacy_names:
        old = models_root / legacy_name
        if old.is_dir():
            found.append(LegacyCandidate("flattened_dir", old, (old,), note=f"legacy name {legacy_name}"))
    for subdir in engine_subdirs:
        for name in (destination_name, *legacy_names):
            dup = models_root / subdir / name
            if dup.is_dir():
                found.append(LegacyCandidate("flattened_dir", dup, (dup,), note=f"engine subdir {subdir}"))

    for external in _external_roots_outside(models_root, cache_root):
        repo_dir = external.path / repo_dirname
        if not repo_dir.is_dir() or not _hub_repo_has_payload(repo_dir):
            continue
        snapshot = _snapshot_from_hub_repo(repo_dir, None, external.path)
        if snapshot is None:
            logger.warning(f"root の外の旧 cache に snapshot を特定できない (触らない): {repo_dir}")
            continue
        found.append(LegacyCandidate("hub_snapshot", snapshot, (), note=f"external {external.label} (copy only, not deleted)"))
    return found


def _snapshot_commit_sha(candidate: LegacyCandidate) -> Optional[str]:
    """hub 階層の ``snapshots/<commit sha>/`` なら dir 名が commit sha (由来を manifest に残す)。"""
    name = candidate.source.name
    if candidate.kind == "hub_snapshot" and len(name) == 40 and all(c in "0123456789abcdef" for c in name):
        return name
    return None


def _select(
    names: Iterable[str], allow_patterns: Optional[Sequence[str]], ignore_patterns: Optional[Sequence[str]]
) -> List[str]:
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
    legacy_names: Sequence[str] = (),
) -> Optional[Manifest]:
    """``destination`` を正本にできる既存資産があれば取り込み、manifest を返す。無ければ ``None``。

    順に試す: (1) destination 自身が manifest 無しで ``required`` を満たす → その場で採用
    (``adopt_dir``)、(2) 旧配置 (:func:`find_legacy_dirs`) を新しい版から順に実体化 → publish。
    正本が確定したら、同じ repo の旧配置は**すべて**消す (二重保持の解消)。どの候補からも
    取り込めなければ何も消さず ``None`` (呼び出し側が download する)。
    """
    destination = Path(destination)
    required = tuple(required)  # generator でも 2 度目以降の走査が空にならないよう具体化
    # download (fetch_repo_dir) と同じ destination 単位の lock。2 process が同時に cold load しても
    # 旧配置の実体化 / 削除が競合しない (PR #458 レビュー)
    with model_lock(staging_root, destination):
        return _migrate_dir_locked(
            destination,
            repo_id=repo_id,
            models_root=Path(models_root),
            cache_root=Path(cache_root),
            staging_root=Path(staging_root),
            required=required,
            variant=variant,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            engine_subdirs=engine_subdirs,
            legacy_names=legacy_names,
        )


def _migrate_dir_locked(
    destination: Path,
    *,
    repo_id: str,
    models_root: Path,
    cache_root: Path,
    staging_root: Path,
    required: Sequence[str],
    variant: Optional[str],
    allow_patterns: Optional[Sequence[str]],
    ignore_patterns: Optional[Sequence[str]],
    engine_subdirs: Sequence[str],
    legacy_names: Sequence[str],
) -> Optional[Manifest]:
    candidates = find_legacy_dirs(
        repo_id=repo_id,
        models_root=models_root,
        cache_root=cache_root,
        destination_name=destination.name,
        engine_subdirs=engine_subdirs,
        legacy_names=legacy_names,
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
            manifest = build_manifest_from_dir(
                payload, repo_id=repo_id, variant=variant, source="migrated", commit_sha=_snapshot_commit_sha(candidate)
            )
            (payload / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
            publish_dir(
                payload,
                destination,
                validate=lambda d: validate_repo_dir(d, repo_id=repo_id, variant=variant, required=required) is not None,
            )
        except Exception as exc:  # noqa: BLE001 - 次の候補 / download へ進む
            logger.warning(f"旧配置からの取り込みに失敗 (次へ): {candidate.source} ({exc})")
            shutil.rmtree(payload, ignore_errors=True)
            continue
        shutil.rmtree(payload, ignore_errors=True)
        logger.info(
            f"旧配置から正本へ取り込んだ: {candidate.source} -> {destination} "
            f"({candidate.kind}{' / ' + candidate.note if candidate.note else ''}, {len(selected)} files, {set(mechanisms.values())})"
        )
        # 取り込んだ候補だけでなく、同じ repo の他の旧配置 (二重保持) も消す
        _remove_legacy(all_cleanup, roots=(models_root, cache_root))
        return validate_repo_dir(destination, repo_id=repo_id, variant=variant, required=required)
    return None


# ---------------------------------------------------------------------------
# single file (.nemo)
# ---------------------------------------------------------------------------


def migrate_nemo_file(
    destination: Path,
    *,
    models_root: Path,
    staging_root: Path,
    validate: Callable[[Path], bool],
    cache_root: Optional[Path] = None,
    repo_id: Optional[str] = None,
    engine_subdirs: Sequence[str] = (),
) -> bool:
    """``<models_root>/<name>.nemo`` を**validator を通る**単一ファイルの正本にする。取り込んだら ``True``。

    :func:`publish_dir` と同じ契約 (valid / quarantine / publish / 旧側の削除は検証後だけ):

    * destination が dir (canary の旧 path 欠陥 ``<name>.nemo/<name>.nemo``): 中の同名ファイルが
      validator を通れば正本の位置へ戻し、残り (例: ``<name>.bin``) は dir ごと消す。通らない /
      同名ファイルが無い dir は ``<name>.nemo.invalid-<ts>`` へ**隔離**して download が publish できる形にする
    * destination が corrupt / truncated なファイル: 隔離し、valid な旧配置があればそれを正本にする
    * 旧配置 (engine subdir の ``<engine>/<name>.nemo``、0.1.0 の hub ``models--…/snapshots/*/<name>.nemo``):
      正本が無ければ validator を通る候補を実体化 → 配置 → **配置後にもう一度 validate** して
      から採用。**旧側の削除は正本が valid になった後だけ** (通らない候補は触らずに残す →
      ``livecap-cli info`` の ``Legacy model layouts`` に出る)
    * 全体を destination 単位の lock (``download_file`` と共有) で囲む
    """
    destination = Path(destination)
    with model_lock(staging_root, destination):
        return _migrate_nemo_file_locked(
            destination,
            models_root=Path(models_root),
            validate=validate,
            cache_root=Path(cache_root) if cache_root is not None else None,
            repo_id=repo_id,
            engine_subdirs=engine_subdirs,
        )


def _migrate_nemo_file_locked(
    destination: Path,
    *,
    models_root: Path,
    validate: Callable[[Path], bool],
    cache_root: Optional[Path],
    repo_id: Optional[str],
    engine_subdirs: Sequence[str],
) -> bool:
    name = destination.name
    migrated = False

    def _valid_file(path: Path) -> bool:
        try:
            return path.is_file() and bool(validate(path))
        except Exception as exc:  # noqa: BLE001 - validator の例外は「invalid」と同じ扱い
            logger.warning(f".nemo の検証で例外 (invalid 扱い): {path} ({exc})")
            return False

    # 1. destination が dir (nested) → 中身が valid なら un-nest、そうでなければ隔離
    if destination.is_dir():
        inner = destination / name
        if _valid_file(inner):
            parked = destination.with_name(f".{name}.nested-{uuid.uuid4().hex[:8]}")
            os.rename(destination, parked)
            try:
                os.replace(parked / name, destination)
            except OSError as exc:
                # **rollback**: 退避した dir を元の名前へ戻す。戻せないと正規の path が消え、元データが
                # scan 対象外の hidden dir に取り残される (PR #458 再レビュー HIGH)
                try:
                    if destination.exists() and not destination.is_dir():
                        destination.unlink()
                    os.rename(parked, destination)
                except OSError as restore_exc:
                    logger.error(f"nested .nemo の退避 dir を戻せなかった: {parked} ({restore_exc})")
                raise RuntimeError(f"nested .nemo を正本の位置へ戻せなかった (元の配置へ復元した): {destination} ({exc})") from exc
            shutil.rmtree(parked, ignore_errors=True)
            logger.info(f"nested な .nemo を正本の位置へ戻した: {destination}")
            migrated = True
        else:
            quarantine(destination, reason="nested .nemo が無い / validator を通らない dir")

    # 2. destination が corrupt なファイル → 隔離 (旧配置 or download で作り直す)
    if destination.is_file() and not _valid_file(destination):
        quarantine(destination, reason=".nemo が validator を通らない (truncated / corrupt)")

    # 3. 旧配置の候補 — 近い場所 (engine subdir) から順に
    candidates: List[_NemoCandidate] = []
    for subdir in engine_subdirs:
        dup = models_root / subdir / name
        if dup.is_file():
            candidates.append(_NemoCandidate(dup, dup, models_root))
    if cache_root is not None and repo_id is not None:
        repo_dirname = "models--" + repo_id.replace("/", "--")
        nemo_name = repo_id.split("/")[-1] + ".nemo"
        hub_roots: List[Tuple[Path, Optional[Path]]] = [(hub_root, cache_root) for hub_root in _hub_roots(cache_root)]
        # root の外 (既定 HF cache) は最後。取り込むだけで消さない (#453)
        hub_roots += [(external.path, None) for external in _external_roots_outside(models_root, cache_root)]
        for hub_root, owner in hub_roots:
            repo_dir = hub_root / repo_dirname
            if not repo_dir.is_dir() or not _hub_repo_has_payload(repo_dir):
                continue
            snapshot = _snapshot_from_hub_repo(repo_dir, None, hub_root)
            source = snapshot / nemo_name if snapshot is not None else None
            if source is None or not source.is_file():
                logger.warning(f"旧 HF cache に {nemo_name} を特定できない (触らない): {repo_dir}")
                continue
            if owner is None:
                candidates.append(_NemoCandidate(source, None, hub_root))
            else:
                candidates.append(_NemoCandidate(source, repo_dir, owner))

    # 4. 正本が無ければ、validator を通る候補から作る (配置後にもう一度 validate)。
    #    validator を通らなかった候補は記録して、後の cleanup でも**触らない**
    invalid_sources: Set[Path] = set()
    if not destination.is_file():
        for candidate in candidates:
            source = candidate.source
            if not _valid_file(source):
                logger.info(f"旧配置の .nemo は validator を通らない (触らない): {source}")
                invalid_sources.add(source)
                continue
            # 旧側は残したまま (hardlink → copy) 原子的に置く。旧側の削除は検証後 (5.)
            try:
                publish_file(source, destination, keep_source=True)
            except OSError as exc:
                logger.warning(f"旧配置の .nemo を取り込めなかった (次へ): {source} ({exc})")
                continue
            if not _valid_file(destination):
                quarantine(destination, reason="取り込んだ .nemo が配置後の validate を通らない")
                continue
            where = " (root の外の cache、元は消さない)" if candidate.cleanup is None else ""
            logger.info(f"旧配置の .nemo を正本の位置へ取り込んだ{where}: {source} -> {destination}")
            migrated = True
            break

    # 5. 正本が valid になった後にだけ、同じモデルの旧配置 (重複) を消す。
    #    validator を通らなかった候補 (invalid_sources) と、まだ検証していない候補のうち invalid な
    #    ものは残す — 「検証していないものは消さない」(残骸は `livecap-cli info` に出る)
    if _valid_file(destination):
        for candidate in candidates:
            if candidate.cleanup is None or candidate.source in invalid_sources or not candidate.cleanup.exists():
                continue
            if candidate.source.is_file() and not _valid_file(candidate.source):
                logger.info(f"旧配置の .nemo は validator を通らないので残す: {candidate.source}")
                continue
            _remove_legacy([candidate.cleanup], roots=(candidate.root,))
    return migrated


def remove_legacy_archives(cache_root: Path, names: Iterable[str]) -> List[Path]:
    """``<cache_root>/downloads/<name>`` に残った旧 download archive (ReazonSpeech int8 の tarball) を消す。

    呼び出し側は**対応する正本が validator を通った後**に呼ぶ (#456 の PR 1 手順「完成 archive は
    初回起動時に削除」)。archive の展開先だった ``reazonspeech-extract/`` が空なら合わせて消す。
    消した path を返す。
    """
    cache_root = Path(cache_root)
    removed: List[Path] = []
    downloads = cache_root / "downloads"
    for name in names:
        archive = downloads / name
        if archive.is_file():
            _remove_legacy([archive], roots=(cache_root,))
            if not archive.exists():
                removed.append(archive)
    extract_dir = cache_root / "reazonspeech-extract"
    try:
        if extract_dir.is_dir() and not any(extract_dir.iterdir()):
            extract_dir.rmdir()
    except OSError:
        pass
    return removed


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


@dataclass(frozen=True)
class KnownRepo:
    """cli が使う HF repo と、その repo から作られる**正本の全 path** (``models_root`` 相対)。

    ``destinations`` の ``{flat}`` は ``<org>--<name>``。**1 repo から複数の正本ができる**場合
    (ReazonSpeech は float32 と int8 を同じ repo の別ファイルから作る) は全部並べる —
    片方しか無い状態を「取り込み済み」と表示すると、もう片方へ切り替えたときに再取得になる
    (PR #463 レビュー MEDIUM)。
    """

    pattern: str  # repo id (``model_name`` が設定できる family は fnmatch pattern)
    destinations: Tuple[str, ...]

    def resolve(self, repo_id: str) -> Tuple[str, ...]:
        flat = repo_id.replace("/", "--")
        return tuple(d.format(flat=flat) for d in self.destinations)


#: root の外の cache を列挙するときの対象 (他アプリのモデルは出さない) と、その採用判定に使う正本。
#: engine / translator の実 repo id / 正本 path と一致することを
#: ``tests/core/engines/test_model_store_contract.py`` で固定する (scan は engine を import しない)
KNOWN_MODEL_REPOS = (
    KnownRepo("Qwen/Qwen3-ASR-0.6B", ("{flat}",)),
    KnownRepo("Systran/faster-whisper-*", ("{flat}",)),
    KnownRepo("Systran/faster-distil-whisper-*", ("{flat}",)),
    KnownRepo("deepdml/faster-whisper-*", ("{flat}",)),
    KnownRepo("mistralai/Voxtral-Mini-3B-2507", ("{flat}",)),
    # 同じ repo から float32 (encoder/decoder/joiner .onnx) と int8 (.int8.onnx) の 2 つの正本を作る
    KnownRepo("reazon-research/reazonspeech-k2-v2", ("{flat}", "{flat}-int8")),
    KnownRepo("nvidia/parakeet-*", ("{flat}.nemo",)),
    KnownRepo("nvidia/canary-*", ("{flat}.nemo",)),
    KnownRepo("nvidia/Riva-Translate-4B-Instruct", ("{flat}",)),
    KnownRepo("Helsinki-NLP/opus-mt-*", ("opus-mt/{flat}",)),
)

#: 旧 workaround / warm step が作っていた engine subdir。ここに実体があれば旧配置の重複。
LEGACY_ENGINE_SUBDIRS = ("parakeet", "parakeet_ja", "canary", "reazonspeech", "voxtral", "qwen3asr", "whispers2t")
#: #456 以前の正本 dir 名 (root 直下)。engine の ``RepoDirSpec.legacy_names`` と同じ値 (scan は engine を import しない)
LEGACY_ROOT_DIR_NAMES = ("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",)


@dataclass(frozen=True)
class ExternalCacheHit:
    """root の外に残っている、cli が使う repo の旧 cache (:func:`scan_external_caches`)。"""

    path: Path  # ``models--<org>--<name>/`` の repo dir
    bytes: int
    repo_id: str
    #: この repo から作られる正本 (:class:`KnownRepo` の ``destinations`` 全部) が ``models_root`` に
    #: 揃っている。**「LiveCap にとって外の copy は要らない」までしか意味しない** — 外の cache は
    #: 他アプリと共用で、同一 volume では hardlink なので、削除の可否と解放量は利用者の判断
    adopted: bool
    label: str  # ExternalCacheRoot.label
    #: まだ ``models_root`` に無い正本 (``models_root`` 相対)。``adopted`` が False の理由
    missing: Tuple[str, ...] = ()


def _repo_id_from_hub_dirname(name: str) -> Optional[str]:
    if not name.startswith("models--"):
        return None
    parts = name[len("models--"):].split("--", 1)
    return f"{parts[0]}/{parts[1]}" if len(parts) == 2 and all(parts) else None


def _known_repo(repo_id: str) -> Optional[KnownRepo]:
    for known in KNOWN_MODEL_REPOS:
        if fnmatch.fnmatchcase(repo_id, known.pattern):
            return known
    return None


def _missing_destinations(models_root: Path, repo_id: str, known: KnownRepo) -> Tuple[str, ...]:
    """``known.destinations`` のうち ``models_root`` に**正本として無い**もの。

    dir は manifest が :func:`validate_repo_dir` を通ること (tombstone / 隔離された
    ``*.invalid-*`` は正本ではない)、``.nemo`` は通常ファイルであることを見る。
    """
    missing = []
    for relative in known.resolve(repo_id):
        target = models_root / relative
        if target.suffix == ".nemo":
            if not target.is_file():
                missing.append(relative)
            continue
        if not target.is_dir() or validate_repo_dir(target, repo_id=repo_id) is None:
            missing.append(relative)
    return tuple(missing)


def scan_external_caches(models_root: Path, cache_root: Path) -> List[ExternalCacheHit]:
    """root の**外** (:func:`external_hub_roots`) に残っている、cli が使う repo (:data:`KNOWN_MODEL_REPOS`)
    の旧 cache を列挙する (削除はしない、#453)。他アプリのモデルは出さない。

    ``adopted`` は「この repo から作られる正本が**全部** ``models_root`` にある」= LiveCap は外の
    copy を要らない、まで。外の cache は他アプリと共用なので、削除の判断は利用者に残る。
    """
    models_root = Path(models_root)
    hits: List[ExternalCacheHit] = []
    for external in _external_roots_outside(models_root, Path(cache_root)):
        if not external.path.is_dir():
            continue
        for repo_dir in sorted(external.path.glob("models--*")):
            repo_id = _repo_id_from_hub_dirname(repo_dir.name)
            known = _known_repo(repo_id) if repo_id else None
            if repo_id is None or known is None:
                continue
            if not repo_dir.is_dir() or not _hub_repo_has_payload(repo_dir):
                continue  # refs/ だけの metadata (実体なし) は列挙しない
            missing = _missing_destinations(models_root, repo_id, known)
            hits.append(
                ExternalCacheHit(repo_dir, _dir_size(repo_dir), repo_id, not missing, external.label, missing)
            )
    return hits


def scan_legacy_layouts(models_root: Path, cache_root: Path) -> List[Tuple[Path, int]]:
    """root の中に残っている旧配置を (path, bytes) で列挙する (削除はしない)。root の外は
    :func:`scan_external_caches`。"""
    models_root = Path(models_root)
    cache_root = Path(cache_root)
    hits: List[Tuple[Path, int]] = []
    hf = cache_root / "huggingface"
    for base in (hf / "hub", hf / "hub" / "transformers", hf / "transformers", hf):
        if base.is_dir():
            for repo_dir in base.glob("models--*"):
                if _hub_repo_has_payload(repo_dir):  # refs/ だけの transient metadata は除外
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
            elif ".invalid-" in child.name:
                # publish_dir / migrate_nemo_file が隔離した壊れた旧正本 (dir も file も)。
                # 消すのは利用者の判断 — 数 GB の .nemo が不可視にならないよう file も列挙する
                hits.append((child, _dir_size(child) if child.is_dir() else child.stat().st_size))
            elif child.is_dir() and (
                child.name in LEGACY_ENGINE_SUBDIRS
                or child.name in LEGACY_ROOT_DIR_NAMES
                or child.name.startswith("whispers2t_")
            ):
                size = _dir_size(child)
                if size:
                    hits.append((child, size))
    return hits
