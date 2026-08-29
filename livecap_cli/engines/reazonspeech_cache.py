"""ReazonSpeech の recognizer を一意に識別する cache identity (Issue #409)。

**なぜ要るか。** 旧 key は ``f"reazonspeech_{use_int8}_{model_path.name}"`` で、
``use_int8`` とディレクトリの **basename しか含んでいなかった**。そのため

1. **異なる models root の同名ディレクトリが衝突する**
2. **モデルファイルを差し替えても古い recognizer が返る**
3. ``num_threads`` / ``decoding_method`` を変えても同じキーになる
   (どちらも ``from_transducer()`` に**実際に渡している**)

**``ModelMemoryCache`` 本体は変更しない。** cache の実装は他 engine も共有しており、
本 module が直すのは「何を key にするか」であって「どう保持するか」ではない。

**「壊れた recognizer を保存しない」は本 module の責務ではない** — post-load health check と
保存ゲートは #392 が持つ。ここは identity だけを扱う。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: ReazonSpeech のリリースが使う epoch 番号。ファイル名に埋まっている。
MODEL_EPOCHS = 99

#: cache key の名前空間。**v1 (``reazonspeech_...``) とは決して衝突しない。**
CACHE_KEY_PREFIX = "reazonspeech:v2"

#: identity に含める native パッケージ。**core も含める** — native 処理には
#: ``sherpa-onnx-core`` も関係し、`pyproject.toml` は両者を同一版へ固定している
#: (「sherpa-onnx + sherpa-onnx-core must stay at the SAME version」)。
#: **版の一致を検証するのは本 module の責務ではない** (依存関係診断の別責務)。
NATIVE_PACKAGES = ("sherpa-onnx", "sherpa-onnx-core")

#: identity と constructor の両方が読むファイル。**role の順序が key に効く**ので固定する。
_ONNX_ROLES = ("encoder", "decoder", "joiner")


class ModelIdentityChangedError(RuntimeError):
    """recognizer の構築中にモデルファイルが変わった。

    そのまま保存すると**古い identity のキーへ新しい内容の recognizer が入る**。
    黙って保存するくらいなら失敗させる。
    """


def required_files(*, use_int8: bool) -> dict[str, str]:
    """tokens / encoder / decoder / joiner の**ファイル名の唯一の出所**。

    以前は engine 内の 4 箇所に同じリストが複製されていた。**identity が hash する
    ファイルと constructor が読むファイルがずれると cache が嘘をつく**ので、
    出所を 1 つにする。
    """
    quant = ".int8" if use_int8 else ""
    return {
        "tokens": "tokens.txt",
        "encoder": f"encoder-epoch-{MODEL_EPOCHS}-avg-1{quant}.onnx",
        "decoder": f"decoder-epoch-{MODEL_EPOCHS}-avg-1.onnx",
        "joiner": f"joiner-epoch-{MODEL_EPOCHS}-avg-1{quant}.onnx",
    }


def _assert_readable(path: Path) -> None:
    """**内容を 1 byte だけ読んで、実際に開けることを確かめる。**

    ``is_file()`` も ``stat()`` も metadata しか触らないので、**内容の読み取りだけが
    拒否されている状態を見逃す** (Windows の ACL、権限を落としたネットワーク共有など)。
    ONNX は identity へ stat しか入れないため、そのままだと **cold path は
    ``from_transducer()`` で落ちるのに warm path は cache hit を返す** —
    「同じ環境なのにプロセス内の順番で結果が変わる」という、本 issue が消そうとしている
    非決定性そのものになる。全内容の hash は要らない (数 GB を毎回読むことになる)。
    """
    with open(path, "rb") as handle:
        handle.read(1)


def resolve_model_files(model_path: Path, *, use_int8: bool) -> dict[str, Path]:
    """必要ファイルを**絶対 path** へ解決する。**足りない / 読めないなら fail loud。**

    **``resolve()`` してから組み立てる**のが契約である。相対 path のまま返すと、
    identity が記録する path (``_normalized_root`` / ``_file_stat`` は解決済み) と
    constructor が実際に開く path がずれ、cwd がプロセス内で変われば
    **同じ key が別のファイルを指し得る**。

    Raises:
        FileNotFoundError: 1 つでも欠けている場合。**ここで落とすのが要点** —
            identity を作れないまま cache を引くと、「identity 取得に失敗したら
            旧 key へ fallback する」経路を将来作り込む余地が生まれる。
        OSError: 1 つでも読み取れない場合 (``PermissionError`` など)。
    """
    root = Path(model_path).resolve()
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    unreadable: list[tuple[str, OSError]] = []
    for role, name in required_files(use_int8=use_int8).items():
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            _assert_readable(path)
        except OSError as exc:
            unreadable.append((name, exc))
            continue
        resolved[role] = path
    if missing:
        raise FileNotFoundError(
            f"ReazonSpeech model is incomplete at {ascii(str(root))}: "
            f"missing {sorted(missing)}. "
            "The cache identity cannot be computed, so the model is not loaded "
            "and no cached recognizer is returned."
        )
    if unreadable:
        name, exc = unreadable[0]
        raise OSError(
            f"ReazonSpeech model file is not readable at {ascii(str(root))}: "
            f"{sorted(item[0] for item in unreadable)}. "
            "No cached recognizer is returned, because a file the recognizer needs "
            "cannot be read."
        ) from exc
    return resolved


def _sha256_file(path: Path) -> str:
    """chunked SHA-256。

    ``resources/ffmpeg_manager.py`` に同方式の helper があるが、**private かつ
    FFmpeg module のものなので直接 import しない** (依存方向が不自然になる)。
    5 行なので engine 側に置く。汎用化が必要になったら公開 utility へ昇格する。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    """``importlib.metadata`` から版を取る。

    ``sherpa_onnx.__version__`` は **wrapper 側の版しか示さず**、
    ``sherpa-onnx-core`` の版が分からない。
    """
    return importlib.metadata.version(name)


def _normalized_root(model_path: Path) -> str:
    """**``str(Path)`` では不十分。**

    Windows は大文字小文字を区別せず、相対 path や symlink も同じモデルを指し得る。
    ``resolve()`` で実体へ、``normcase()`` で表記の揺れを吸収する。
    """
    return os.path.normcase(str(Path(model_path).resolve()))


def _file_stat(path: Path) -> tuple[str, int, int]:
    """``(resolved path, st_size, st_mtime_ns)``。

    **ONNX 本体の SHA-256 は取らない** — lookup のたびに数 GB を読むことになり、
    メモリ cache の利点を失う。
    """
    stat = path.stat()
    return (os.path.normcase(str(path.resolve())), stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """recognizer を一意に決める入力の集合。"""

    model_root: str
    tokens_sha256: str
    onnx_files: tuple[tuple[str, str, int, int], ...]   # (role, path, size, mtime_ns)
    use_int8: bool
    num_threads: int
    decoding_method: str
    native_versions: tuple[tuple[str, str], ...]        # (package, version)

    def _payload(self) -> Mapping[str, object]:
        """canonical serialization にかける素の dict。"""
        return {
            "model_root": self.model_root,
            "tokens_sha256": self.tokens_sha256,
            "onnx_files": [list(item) for item in self.onnx_files],
            "use_int8": self.use_int8,
            "num_threads": self.num_threads,
            "decoding_method": self.decoding_method,
            "native_versions": [list(item) for item in self.native_versions],
        }

    def cache_key(self) -> str:
        """``reazonspeech:v2:<sha256(identity)>``。

        長い path や非 ASCII path をそのまま key やログへ出さない。
        **``repr(dict)`` や素の ``str()`` に依存しない** — 決定的な serialization を
        使い、`test_reazonspeech_cache_key.py` が安定性を固定する。
        """
        blob = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return f"{CACHE_KEY_PREFIX}:{hashlib.sha256(blob).hexdigest()}"


def build_identity(
    model_path: Path,
    *,
    use_int8: bool,
    num_threads: int,
    decoding_method: str,
) -> ModelIdentity:
    """identity を組み立てる。**ファイルが無い / 読めないなら raise。**"""
    files = resolve_model_files(model_path, use_int8=use_int8)
    return ModelIdentity(
        model_root=_normalized_root(model_path),
        tokens_sha256=_sha256_file(files["tokens"]),
        onnx_files=tuple(
            (role,) + _file_stat(files[role]) for role in _ONNX_ROLES
        ),
        use_int8=use_int8,
        num_threads=num_threads,
        decoding_method=decoding_method,
        native_versions=tuple(
            (name, _package_version(name)) for name in NATIVE_PACKAGES
        ),
    )
