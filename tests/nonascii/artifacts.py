"""合成アーティファクトの生成と、実モデルの materialize (Issue #378)。

**合成優先の方針**: ほとんどの境界は GB 級の実モデルではなく、合成した
最小アーティファクトで測れる。実モデルが要るのは「ネイティブローダの
narrow path 挙動そのもの」を見る行だけである。

実モデルは ``os.link`` (NTFS ハードリンク、管理者権限不要・追加バイトゼロ) で
materialize する。ソースと同一ボリュームでなければ ``shutil.copy2`` に降格し、
どちらを使ったかを run メタデータに記録する。
"""

from __future__ import annotations

import base64
import os
import shutil
import wave
from pathlib import Path

from .record import ProbeSkipped

#: 最小の ONNX モデル (input → Add(0.0) → output、float32[1,4])。
#:
#: ``onnx`` パッケージは本プロジェクトの依存に含まれず、``torch`` も
#: ``engines-torch`` extra 側なので、**実行時に生成することはできない**。
#: そこで一度だけ生成して定数として commit する。
#:
#: 再生成コマンド (ephemeral overlay を使い、プロジェクト venv を汚さない)::
#:
#:     uv run --with onnx python -c "
#:     import base64, io, torch, torch.nn as nn
#:     class M(nn.Module):
#:         def forward(self, x): return x + 0.0
#:     b = io.BytesIO()
#:     torch.onnx.export(M(), (torch.zeros(1,4),), b, input_names=['input'],
#:                       output_names=['output'], opset_version=13, dynamo=False)
#:     print(base64.b64encode(b.getvalue()).decode())"
TINY_ONNX_B64 = (
    "CAcSB3B5dG9yY2gaBTIuOS4xOrABCj8SEi9Db25zdGFudF9vdXRwdXRfMBoJL0NvbnN0YW50Ig"
    "hDb25zdGFudCoUCgV2YWx1ZSoIEAFKBAAAAACgAQQKLgoFaW5wdXQKEi9Db25zdGFudF9vdXRw"
    "dXRfMBIGb3V0cHV0GgQvQWRkIgNBZGQSCm1haW5fZ3JhcGhaFwoFaW5wdXQSDgoMCAESCAoCCA"
    "EKAggEYhgKBm91dHB1dBIOCgwIARIICgIIAQoCCARCAhAN"
)


def tiny_onnx_bytes() -> bytes:
    return base64.b64decode(TINY_ONNX_B64)


def write_tiny_onnx(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tiny_onnx_bytes())
    return path


def write_invalid_onnx(path: Path) -> Path:
    """**意図的に不正な** ONNX。

    sherpa-onnx の差分プローブで使う: ASCII パスなら「protobuf が壊れている」、
    非 ASCII パスなら「ファイルを開けない」という**エラー署名の差**を見ることで、
    740 MB の実モデル無しに narrow path 挙動を証明できる。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"NOT-AN-ONNX-MODEL" * 8)
    return path


def write_tokens_txt(path: Path) -> Path:
    """sherpa-onnx が読む ``tokens.txt`` の最小形。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<blk> 0\n<unk> 1\nあ 2\nい 3\n", encoding="utf-8"
    )
    return path


def same_volume(a: Path, b: Path) -> bool:
    """``os.link`` が使えるか (同一ボリュームか) を事前判定する。

    Windows でも CPython は ``st_dev`` を埋めるのでボリューム判別に使える
    (実測: ``C:\\Windows`` と ``D:\\Codes`` で別値)。
    """
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _is_hardlink_of(src: Path, dst: Path) -> bool:
    """``dst`` が ``src`` と同一実体か (= hardlink か)。

    Windows でも ``os.stat`` は file index を ``st_ino`` に載せるので、
    ``(st_dev, st_ino)`` の一致で判定できる (実測で確認済み)。``st_ino`` が 0 を
    返す FS では**すべて同一に見えてしまう**ので、その場合は copy 扱いにする。

    **この述語は実体化の直後にも通す。** ``os.link`` の成功だけで ``"hardlink"`` と
    答えると、inode を使えない FS で 1 回目と 2 回目の答えが割れる。
    """
    try:
        a, b = os.stat(src), os.stat(dst)
    except OSError:
        return False
    return a.st_ino != 0 and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def materialize_file(src: Path, dst: Path) -> str:
    """``src`` を ``dst`` に実体化し、使った機構名を返す。

    hardlink → copy の順。**symlink は使わない** — 一部のネイティブライブラリは
    ``GetFinalPathNameByHandle`` 等で元パスを復元してしまい、測定対象の条件を
    壊す可能性があるため (実装側 ``ascii_safe_path()`` の realpath 危険と同じ話)。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # **"existing" を返してはならない。** ASCII control の root は variant を跨いで
        # 共有されるので、2 回目以降だけ観測が変わると差分判定が「path と無関係な
        # 非決定性」を検出し **error_harness** に落ちる (#387 で実際に踏んだ。
        # required_variants を 1 件から 2 件へ増やした瞬間に表面化した)。
        #
        # 返すべきは「今回コピーしたか」ではなく「**どう実体化されているか**」である。
        # 1 回目と同じ答えになるので観測が決定的になる。
        return "hardlink" if _is_hardlink_of(src, dst) else "copy"
    if same_volume(src, dst.parent):
        try:
            os.link(src, dst)
            # **`os.link` の成功だけで "hardlink" と答えてはならない。** 判定を
            # 既存ファイル側と揃えないと、`st_ino` を返さない FS で
            #     1 回目 (dst 無し) -> os.link 成功         -> "hardlink"
            #     2 回目 (dst あり) -> _is_hardlink_of=False -> "copy"
            # となり、**この関数が直したはずのドリフトが再発する**。同じ述語を
            # 通すことで「inode が使えないなら常に copy と答える」契約が閉じる。
            return "hardlink" if _is_hardlink_of(src, dst) else "copy"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def materialize_tree(src: Path, dst: Path, *, include: list[str] | None = None) -> dict:
    """ディレクトリを実体化する (子ごとに ``materialize_file``)。

    ``include`` を指定すると、その名前の子だけを実体化する。ReazonSpeech の
    モデルディレクトリは int8 と float32 の両方を持つため、必要な 4 ファイル
    だけを実体化して時間とディスクを節約する。
    """
    dst.mkdir(parents=True, exist_ok=True)
    mechanisms: dict[str, str] = {}
    for child in sorted(src.iterdir()):
        if not child.is_file():
            continue
        if include is not None and child.name not in include:
            continue
        mechanisms[child.name] = materialize_file(child, dst / child.name)
    return mechanisms


def dominant_mechanism(mechanisms: dict[str, str]) -> str:
    if not mechanisms:
        return "n/a"
    if all(m == "hardlink" for m in mechanisms.values()):
        return "hardlink"
    if any(m == "copy" for m in mechanisms.values()):
        return "copy"
    return "mixed"


__all__ = [
    "TINY_ONNX_B64",
    "dominant_mechanism",
    "materialize_file",
    "materialize_tree",
    "same_volume",
    "tiny_onnx_bytes",
    "write_invalid_onnx",
    "write_tiny_onnx",
    "write_tokens_txt",
]


#: probe 用の実発話が置いてある場所 (repo に commit 済み)。
_AUDIO_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "audio"


def load_probe_speech(stem: str) -> tuple[int, "object"]:
    """probe 用の実発話 (16 kHz mono float32) を ``(sample_rate, audio)`` で返す。

    **合成信号は使わない。** 220 Hz の正弦波などでは token が 1 つも出ないことがあり、
    その場合 token id -> symbol の lookup を**通らずに** pass できてしまう — probe が
    守っている経路を素通りする。

    Args:
        stem: ``tests/assets/audio`` からの相対 stem (例: ``"ja/jsut_basic5000_0001"``)。
            **モデルの言語に合ったものを選ぶこと** — 合わない言語だと転写が空になり、
            「control が非空」の前提を満たせない。
    """
    import numpy as np

    wav = _AUDIO_ASSETS / f"{stem}.wav"
    if not wav.is_file():  # pragma: no cover - 資産は repo に commit されている
        raise ProbeSkipped(f"probe 用音声が見つからない: {wav}")
    with wave.open(str(wav), "rb") as fh:
        sample_rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, audio
