"""ModelRoot 契約の共通実装 — flattened dir + manifest + 原子的 publish (Issue #456)。

**契約**

* ``models_root`` は「推論を再起動するために必要な永続資産」の**唯一の正本**。
  engine ごとに ``<models_root>/<org>--<name>/`` (flattened dir) か
  ``<models_root>/<org>--<name>.nemo`` (single file) を置く
* ``cache_root`` は staging (``<cache_root>/downloads/<repo>/{download,payload}``)、lock、
  ``.incomplete``、HTTP metadata だけ。**完成済みモデルの正本を残さない**
* dir の cache hit は「非空である」ことではなく :func:`validate_repo_dir` — manifest
  (:data:`MANIFEST_NAME`) に記録した全ファイルがサイズ一致で実在し、symlink が dir の外を
  指していないこと — **だけ**で決まる (旧 ``BaseEngine._is_model_cached`` は非空 dir を hit
  にしていた)
* publish は :func:`publish_dir` — destination と同じ volume の sibling temp へ集めてから
  ``os.replace`` する。失敗時に destination を作らず、完了済み payload を失わない
* wheel 同梱の VAD 資産 (:data:`MODEL_STORE_EXEMPT_ASSETS`) は install asset として
  **契約の対象外** (runtime に何も書かない。複製すると正本が 2 つになるだけ)

manifest の中身 (``schema_version`` 1)::

    {"schema_version": 1, "repo_id": "Qwen/Qwen3-ASR-0.6B", "revision": null,
     "commit_sha": "5eb1…", "variant": null, "source": "download",
     "files": [{"path": "config.json", "size": 1381, "etag": "\\"…\\""}, …]}

hash 照合は起動コストが大きいので既定では size だけを見る。``etag`` は取れたときだけ
記録する (HF の ``.metadata`` から。LFS は SHA-256)。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "INVALIDATED_SOURCE",
    "MANIFEST_NAME",
    "MODEL_STORE_EXEMPT_ASSETS",
    "SCHEMA_VERSION",
    "SINGLE_FILE_SUFFIXES",
    "Manifest",
    "ManifestFile",
    "adopt_dir",
    "build_manifest_from_dir",
    "invalidate_manifest",
    "is_safe_relative_path",
    "materialize_files",
    "model_lock",
    "model_lock_path",
    "publish_dir",
    "publish_file",
    "quarantine",
    "read_manifest",
    "validate_model_file",
    "validate_repo_dir",
    "write_manifest",
]

#: flattened dir の中に置く manifest のファイル名。
MANIFEST_NAME = "livecap-manifest.json"
SCHEMA_VERSION = 1

#: ModelRoot 契約の**明示例外** — wheel 同梱で runtime に書かれない install asset。
#: docs (``docs/architecture/model-store-contract.md``) の表と一致することを
#: ``tests/core/engines/test_model_store_contract.py`` が固定する。
MODEL_STORE_EXEMPT_ASSETS: Mapping[str, str] = {
    "silero_vad": "silero_vad/data/*.onnx, *.jit (importlib.resources で読む。runtime の書き込み無し)",
    "ten_vad": "ten_vad_library/ten_vad.dll (同梱 native library。runtime の書き込み無し)",
}

#: staging / metadata の名前。ModelRoot 内に**存在してはならない**もの。
TRANSIENT_MARKERS = (".cache", ".locks")
TRANSIENT_SUFFIXES = (".lock", ".incomplete", ".metadata", ".part")


def is_safe_relative_path(path: str) -> bool:
    """manifest の ``files[].path`` として許す形か: 空でない正規化済みの相対 POSIX path。

    ``.`` / ``..`` / 空要素、絶対 path、drive (``C:``) / UNC、backslash は拒否する。
    manifest が改竄 / 破損していても ``validate_repo_dir`` が ModelRoot の外のファイルを
    「正本の一部」として数えないようにするため (PR #457 レビュー HIGH)。
    """
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False
    if path.startswith("/") or ":" in path:
        return False
    parts = path.split("/")
    return all(part not in ("", ".", "..") for part in parts)


@dataclass(frozen=True)
class ManifestFile:
    path: str  # dir からの相対 path (posix、is_safe_relative_path を満たす)
    size: int
    etag: Optional[str] = None


@dataclass(frozen=True)
class Manifest:
    repo_id: str
    files: tuple = ()
    revision: Optional[str] = None
    commit_sha: Optional[str] = None
    variant: Optional[str] = None
    source: str = "download"  # "download" | "adopted" | "migrated"
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        payload = asdict(self)
        payload["files"] = [asdict(f) for f in self.files]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Optional["Manifest"]:
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            return None
        if not isinstance(payload.get("repo_id"), str) or not isinstance(payload.get("files"), list):
            return None
        files = []
        for entry in payload["files"]:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not is_safe_relative_path(entry["path"])
                or not isinstance(entry.get("size"), int)
                or isinstance(entry.get("size"), bool)
                or entry["size"] < 0
            ):
                return None
            etag = entry.get("etag")
            files.append(ManifestFile(entry["path"], entry["size"], etag if isinstance(etag, str) else None))
        return cls(
            repo_id=payload["repo_id"],
            files=tuple(files),
            revision=payload.get("revision") if isinstance(payload.get("revision"), str) else None,
            commit_sha=payload.get("commit_sha") if isinstance(payload.get("commit_sha"), str) else None,
            variant=payload.get("variant") if isinstance(payload.get("variant"), str) else None,
            source=payload.get("source") if isinstance(payload.get("source"), str) else "download",
        )


# ---------------------------------------------------------------------------
# manifest I/O
# ---------------------------------------------------------------------------


def write_manifest(directory: Path, manifest: Manifest) -> Path:
    path = Path(directory) / MANIFEST_NAME
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def read_manifest(directory: Path) -> Optional[Manifest]:
    """manifest を読む。無い / JSON でない / schema が違う → ``None``。"""
    path = Path(directory) / MANIFEST_NAME
    try:
        return Manifest.from_json(path.read_text(encoding="utf-8"))
    except OSError:
        return None


#: :func:`invalidate_manifest` が書く manifest の ``source``。``files`` は空なので
#: :func:`validate_repo_dir` は必ず miss、:func:`adopt_dir` も採用しない。
INVALIDATED_SOURCE = "invalidated"


def invalidate_manifest(directory: Path, *, reason: str) -> None:
    """manifest を「無効」に書き換えて次回 cache miss にする (self-heal)。

    manifest には無い形で dir が壊れて ``from_pretrained`` / ``load_model`` が落ちたとき、
    manifest を残すと以後ダウンロード phase を永久に skip して落ち続ける。**消すのではなく**
    ``files: []`` + ``source: "invalidated"`` の manifest に置き換える — 消すと
    :func:`adopt_dir` が「manifest の無い完全な dir」として同じ壊れた内容を再採用してしまう。
    次回の取得で :func:`publish_dir` がこの dir を ``<name>.invalid-<ts>`` へ隔離する。
    dir が無ければ何もしない。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return
    existing = read_manifest(directory)
    tombstone = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": existing.repo_id if existing is not None else "",
        "variant": existing.variant if existing is not None else None,
        "files": [],
        "source": INVALIDATED_SOURCE,
        "reason": reason,
    }
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(tombstone, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.warning(f"manifest を無効化した (次回再取得): {path} - {reason}")


def _is_transient_name(name: str) -> bool:
    return name in TRANSIENT_MARKERS or name.endswith(TRANSIENT_SUFFIXES)


def build_manifest_from_dir(
    directory: Path,
    *,
    repo_id: str,
    variant: Optional[str] = None,
    revision: Optional[str] = None,
    commit_sha: Optional[str] = None,
    source: str = "download",
    etags: Optional[Mapping[str, str]] = None,
) -> Manifest:
    """dir の実ファイルを走査して manifest を組み立てる (書き込みはしない)。

    manifest 自身と transient (``.cache`` / ``*.lock`` / ``*.incomplete`` / ``*.metadata`` /
    ``*.part``) は含めない。symlink は ``stat()`` (follow) の size を記録する。
    """
    directory = Path(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory)
        if rel.name == MANIFEST_NAME or any(_is_transient_name(part) for part in rel.parts):
            continue
        rel_posix = rel.as_posix()
        files.append(ManifestFile(rel_posix, path.stat().st_size, (etags or {}).get(rel_posix)))
    return Manifest(
        repo_id=repo_id,
        files=tuple(files),
        revision=revision,
        commit_sha=commit_sha,
        variant=variant,
        source=source,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


#: 単一ファイルの正本の先頭 4 byte。``.nemo`` は tar (``./.``) か zip (``PK\x03\x04``)、
#: ``.onnx`` は protobuf (``\x08\x01``)。それ以外の拡張子は形式が多様なので存在だけを見る
_FILE_MAGIC = {".nemo": (b"PK\x03\x04", b"./."), ".onnx": (b"\x08\x01",)}

#: 正本が dir ではなく**単一ファイル**になる拡張子。:func:`validate_model_file` が形式を見る側で、
#: 「この path は dir ではなくファイルの正本か」の判定もここを唯一の出所にする (#453)
SINGLE_FILE_SUFFIXES = tuple(_FILE_MAGIC)


def validate_model_file(path: Path) -> bool:
    """単一ファイルの正本 (``.nemo`` / ``.onnx``) が**形式として**正しいか (Issue #456 / #453)。

    ``BaseEngine._verify_model_integrity`` (engine の cache 判定と
    :func:`legacy_model_layouts.migrate_nemo_file` の validator) と
    :func:`legacy_model_layouts.scan_external_caches` の ``adopted`` 判定が**同じ実装**を使う
    ための SSOT。片方だけ弱いと「採用済み」の表示が engine の判定と食い違う (PR #463 レビュー)。
    """
    path = Path(path)
    if not path.is_file():
        return False
    magic = _FILE_MAGIC.get(path.suffix)
    if magic is None:
        return True  # .bin / .pt / .pth 等は多様なので存在だけ
    try:
        with open(path, "rb") as f:
            header = f.read(4)
    except OSError as exc:
        logger.error(f"単一ファイルの形式チェックに失敗: {path} ({exc})")
        return False
    return any(header.startswith(prefix) for prefix in magic)


def validate_repo_dir(
    directory: Path,
    *,
    repo_id: Optional[str] = None,
    variant: Optional[str] = None,
    required: Optional[Iterable[str]] = None,
) -> Optional[Manifest]:
    """dir が **正本として使える**ときだけ manifest を返す。それ以外は ``None`` (= miss)。

    * manifest が無い / 読めない / ``repo_id`` / ``variant`` が期待と違う
    * ``required`` (呼び出し側が**今**要求する必須ファイル名) のどれかが manifest に無い、または
      通常ファイルとして実在しない — manifest の自己整合性だけでは、required が後から増えた /
      publish 失敗時に残った古い payload を正本として通してしまう (PR #457 再レビュー)
    * ``files[]`` のどれかが無い、または size が違う (削除 / truncated copy / 壊れた symlink)
    * ``files[].path`` の実体 (``resolve()``) が dir の**外** — 最終要素の symlink だけでなく、
      親 dir の symlink、``..`` / 絶対 path を含む manifest も全 entry で拒否する
      (旧 HF cache の ``blobs/`` を指したまま移した形、改竄された manifest)
    * ``files[]`` が空

    **「非空 dir」では hit にしない** — それが旧 ``BaseEngine._is_model_cached`` の穴だった。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    manifest = read_manifest(directory)
    if manifest is None or not manifest.files:
        return None
    if repo_id is not None and manifest.repo_id != repo_id:
        return None
    if variant is not None and manifest.variant != variant:
        return None
    if required is not None:
        recorded = {f.path for f in manifest.files}
        for name in tuple(required):
            if name not in recorded or not (directory / name).is_file():
                return None
    root = directory.resolve()
    for entry in manifest.files:
        if not is_safe_relative_path(entry.path):
            return None
        path = directory / entry.path
        try:
            if not path.is_file():
                return None
            # **全 entry** で実体の containment を見る (symlink の有無に関係なく)
            if not path.resolve().is_relative_to(root):
                return None
            if path.stat().st_size != entry.size:
                return None
        except OSError:
            return None
    return manifest


def adopt_dir(
    directory: Path,
    *,
    repo_id: str,
    required: Iterable[str],
    variant: Optional[str] = None,
) -> Optional[Manifest]:
    """manifest の無い既存 dir を、``required`` が揃っていればその場で正本として採用する。

    migration 用 (Voxtral の ``save_pretrained`` 出力、ReazonSpeech の root 側 dir など、
    #456 以前に ``models_root`` へ置かれた flattened dir)。``required`` のどれかが無ければ
    ``None`` を返し、何も書かない。既に valid な manifest (``required`` も満たす) があればそれを返す。
    """
    directory = Path(directory)
    required = tuple(required)
    # 既存 manifest も**現在の** required で判定する — 旧 required で作られた manifest が valid でも、
    # 今欠けているファイルがあれば採用しない (採用すると完全な旧配置を消してしまう、PR #458 再レビュー)
    existing = validate_repo_dir(directory, repo_id=repo_id, variant=variant, required=required)
    if existing is not None:
        return existing
    if not directory.is_dir():
        return None
    if (directory / MANIFEST_NAME).exists():
        # manifest があるのに valid でない = 取得済みだが壊れた / 無効化された dir。
        # 「manifest の無い旧配置」ではないので採用しない (publish_dir が隔離する)
        logger.info(f"manifest が invalid な dir は採用しない (次の取得で隔離): {directory}")
        return None
    for name in required:
        if not (directory / name).is_file():
            return None
    manifest = build_manifest_from_dir(directory, repo_id=repo_id, variant=variant, source="adopted")
    if not manifest.files:
        return None
    write_manifest(directory, manifest)
    logger.info(f"既存の dir を正本として採用した (manifest を書いた): {directory}")
    return validate_repo_dir(directory, repo_id=repo_id, variant=variant, required=required)


# ---------------------------------------------------------------------------
# materialize / publish
# ---------------------------------------------------------------------------


def materialize_files(src_dir: Path, dst_dir: Path, names: Iterable[str]) -> dict:
    """``names`` を ``src_dir`` から ``dst_dir`` へ **symlink を dereference して**実体化する。

    HF hub cache の ``snapshots/<sha>/<file>`` は ``../../blobs/<hash>`` への相対 symlink で
    あり得る (symlink が使える環境)。dir ごと動かすと旧 cache を指したまま壊れるので、
    ファイル単位に解決先を読む。同一 volume なら hardlink (瞬時・容量ゼロ)、だめなら copy。
    戻り値は name → ``"hardlink"`` / ``"copy"``。
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    mechanisms: dict = {}
    for name in names:
        source = (src_dir / name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"materialize: {src_dir / name} が無い (resolve -> {source})")
        target = dst_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
            mechanisms[name] = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            mechanisms[name] = "copy"
    return mechanisms


def _quarantine_name(destination: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return destination.with_name(f"{destination.name}.invalid-{stamp}-{uuid.uuid4().hex[:6]}")


def quarantine(path: Path, *, reason: str) -> Path:
    """invalid な正本 (file / dir) を同じ dir 内の ``<name>.invalid-<ts>`` へ rename して**隔離**する。

    削除はしない (旧 marker 方式の dir や手動配置、壊れた ``.nemo`` を利用者が確認できるように)。
    隔離した path は ``livecap-cli info`` の ``Legacy model layouts`` に出る。
    """
    path = Path(path)
    target = _quarantine_name(path)
    os.rename(path, target)
    logger.warning(f"invalid な正本を隔離した: {path} -> {target.name} ({reason})")
    return target


def publish_file(source: Path, destination: Path, *, keep_source: bool = False) -> Path:
    """単一ファイルを ``destination`` へ**原子的に**配置し、失敗しても ``source`` を失わない。

    ``models_root`` と ``cache_root`` は別 volume になり得る (``configure_resources()`` で独立指定
    できる)。``shutil.move`` は cross-volume では copy → 削除になり、途中で落ちると destination に
    **途中までのファイルが残る**。単一ファイルの完全性確認は先頭数 byte しか見ないので、truncated
    file が cache hit として固定されてしまう (PR #448 レビュー)。

    手順:

    1. ``destination`` と同じ dir (= 同じ volume) の一意な temp ``.<name>.<uuid>.part`` へ:
       ``keep_source=False`` なら ``os.rename`` (瞬時)、cross-volume なら ``shutil.copy2`` (source は残す)。
       ``keep_source=True`` (旧配置からの取り込み) なら ``os.link`` → だめなら ``copy2`` (source は常に残す)
    2. ``os.replace(temp, destination)`` (同一 volume 内の rename なので原子的)
    3. ``keep_source=False`` で copy した場合だけ、publish 成功後に ``source`` を消す

    どの段階で失敗しても ``destination`` は作られない。rename 後に ``os.replace`` が失敗したら
    temp を ``source`` へ戻し (完了済み download を失わない)、それ以外は temp を消す。
    """
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
    moved = False
    try:
        if keep_source:
            try:
                os.link(source, temp)
            except OSError:
                shutil.copy2(source, temp)
        else:
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
                logger.error(f"publish に失敗し、完了済み download を戻せなかった: {temp} ({restore_exc})")
        else:
            temp.unlink(missing_ok=True)
        raise
    if not moved and not keep_source:
        source.unlink(missing_ok=True)
    return destination


def model_lock_path(staging_root: Path, destination: Path) -> Path:
    """destination 単位の inter-process lock のファイル。

    download (``fetch_repo_dir`` / ``download_file``) と migration (``migrate_dir`` /
    ``migrate_nemo_file``) が**同じ lock を共有**する — 2 process が同時に cold load しても、
    旧配置の rename / delete と取得 / publish が競合しない (PR #458 レビュー)。
    """
    return Path(staging_root) / f"{destination.name}.lock"


def model_lock(staging_root: Path, destination: Path):
    """:func:`model_lock_path` の ``FileLock`` (context manager)。"""
    from filelock import FileLock

    path = model_lock_path(staging_root, destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path))


def publish_dir(
    payload_dir: Path,
    destination: Path,
    *,
    validate: Callable[[Path], bool],
) -> Path:
    """``payload_dir`` を ``destination`` へ**原子的に**配置する。

    1. ``destination`` が ``validate()`` を通る → **skip** (payload は触らない。呼び出し側が消す)
    2. ``destination`` が存在するが invalid → 同じ dir 内の ``<name>.invalid-<ts>`` へ rename して
       **隔離** (削除しない — 旧 marker 方式の dir や手動配置を壊さない)
    3. ``payload_dir`` を ``destination`` と同じ volume の sibling temp ``.<name>.<uuid>.part`` へ:
       同一 volume なら ``os.rename`` (瞬時)、別 volume なら ``shutil.copytree`` (payload は残す)
    4. temp 上で ``validate()`` → ``os.replace(temp, destination)`` — destination はこの時点で
       無いので Windows でも原子的 (非空 dir の置換にはならない)
    5. 失敗: temp を消す (rename 済みなら temp を payload へ戻す)、隔離した旧 destination を
       元の名前へ戻す。例外は再送出。**どの段階で失敗しても destination は作られず、
       完了済み payload は失われない**
    """
    payload_dir = Path(payload_dir)
    destination = Path(destination)
    if not payload_dir.is_dir():
        raise FileNotFoundError(f"publish: payload が無い: {payload_dir}")

    if destination.exists():
        if destination.is_dir() and validate(destination):
            logger.info(f"publish: destination は既に valid、skip: {destination}")
            return destination
        quarantined = quarantine(destination, reason="publish 先が invalid")
    else:
        quarantined = None
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
    moved = False
    try:
        try:
            os.rename(payload_dir, temp)
            moved = True
        except OSError:
            # 別 volume (EXDEV / WinError 17) 等。payload を残したまま copy する
            shutil.copytree(payload_dir, temp, copy_function=shutil.copy2)
        if not validate(temp):
            raise RuntimeError(f"publish: payload が validate を通らない: {payload_dir}")
        os.replace(temp, destination)
    except BaseException:
        if moved:
            try:
                os.replace(temp, payload_dir)  # 完了済み payload を staging へ戻す
            except OSError as restore_exc:
                logger.error(f"publish に失敗し、payload を staging へ戻せなかった: {temp} ({restore_exc})")
        else:
            shutil.rmtree(temp, ignore_errors=True)
        if quarantined is not None and not destination.exists():
            try:
                os.rename(quarantined, destination)
            except OSError as restore_exc:
                logger.error(f"隔離した旧 destination を戻せなかった: {quarantined} ({restore_exc})")
        raise
    if not moved:
        shutil.rmtree(payload_dir, ignore_errors=True)
    logger.info(f"publish: {destination}")
    return destination
