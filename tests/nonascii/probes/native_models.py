"""ネイティブモデルローダの境界プローブ (Issue #378)。

ここが epic #380 の中核 — sherpa-onnx と NeMo/sentencepiece は narrow path で
**黙って**壊れる一方、onnxruntime は非 ASCII でも通ることが分かっている。
「同一プロセス内でライブラリごとに対応状況がバラバラ」という主張を、
実測として固定する。
"""

from __future__ import annotations

import os
from pathlib import Path

from ..artifacts import (
    dominant_mechanism,
    load_probe_speech,
    materialize_tree,
    tiny_onnx_bytes,
    write_tiny_onnx,
)
from ..record import ProbeContext, ProbeSkipped
from . import probe

#: ReazonSpeech engine が要求する**整合したファイルセット** (``reazonspeech_engine.py``
#: の ``required_files`` と同じ構成)。int8 の decoder は量子化されないので
#: ``.int8`` が付かない — **ファイルごとに独立して glob すると、壊れた int8 dir と
#: 完全な float32 ファイルが同居したときに混在セットを返してしまう**。セット単位で
#: 「全部そろっているか」を見る。
_REAZON_FILE_SETS: tuple[tuple[str, tuple[str, str, str, str]], ...] = (
    # int8 を優先する — 154 MB で float32 (592 MB) より軽く、測定内容は同じ。
    (
        "int8",
        (
            "tokens.txt",
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
        ),
    ),
    (
        "float32",
        (
            "tokens.txt",
            "encoder-epoch-99-avg-1.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
        ),
    ),
)


def reazon_model_files(src: Path) -> tuple[str, tuple[str, str, str, str]] | None:
    """``(variant, files)`` を返す。**完全にそろったセットだけ**を採用する。

    どちらの variant でも成立させるのは、CI ランナーにどれが温まっているかが
    workflow 側の都合で変わるためである。1 つに固定していた頃は、float32 しか
    無いランナーで probe が黙って skip し、**緑のままゲートだけが失効した**
    (#377)。
    """
    for variant, files in _REAZON_FILE_SETS:
        if all((src / name).is_file() for name in files):
            return variant, files
    return None


@probe("onnxruntime.InferenceSession.str_path")
def onnxruntime_session_path(ctx: ProbeContext) -> dict:
    """``ort.InferenceSession(<path>)`` — sherpa / whisper の下層。

    既知の実測では**通る**。方式②(現状維持) の根拠を固定する。
    なお ``InferenceSession`` は ``bytes`` も受けるので、仮に NG でも
    方式①へ退避できる (その事実も観測に残す)。
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ProbeSkipped(f"onnxruntime/numpy 未導入: {exc}") from exc

    model = write_tiny_onnx(ctx.root / "model.onnx")
    ctx.stage("write_model")

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    ctx.stage("load_from_path")
    out_path = session.run(
        None, {"input": np.arange(4, dtype=np.float32).reshape(1, 4)}
    )[0]

    # 方式①(bytes) が実際に使えることも同時に確認しておく
    session_buf = ort.InferenceSession(
        tiny_onnx_bytes(), providers=["CPUExecutionProvider"]
    )
    ctx.stage("load_from_bytes")
    out_buf = session_buf.run(
        None, {"input": np.arange(4, dtype=np.float32).reshape(1, 4)}
    )[0]

    return {
        "path_output": out_path.reshape(-1).tolist(),
        "bytes_output": out_buf.reshape(-1).tolist(),
        "bytes_api_available": True,
    }


@probe("sherpa.from_transducer.real")
def sherpa_from_transducer_real(ctx: ProbeContext) -> dict:
    """**wide-path regression** — 実 ReazonSpeech モデルで tokens.txt の読み取りを通す。

    sherpa-onnx **1.12.39 まで**は、非 ASCII パスでもロードは成功し decode が全件
    ``IndexError`` になった (``SymbolTable`` が narrow path の ``std::ifstream`` で
    tokens.txt を開けず、空のまま構築されるため)。ハーネスがこれを ``fail_silent``
    と分類できることが検出能力の証明だった。

    **1.13.6 で上流が修正済み** (PR #3255 — ``OpenInputFile()`` -> ``ToWideString()``)
    なので、現在の役割は **regression ゲート**である: 依存更新でこの経路が
    再び narrow path へ戻ったら落ちる (#377)。

    実モデルはローカルの models root から hardlink で実体化するので、
    ネットワークもディスクもほぼ消費しない。
    """
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as exc:
        raise ProbeSkipped(f"sherpa-onnx/numpy 未導入: {exc}") from exc

    source = ctx.payload.get("model_source")
    if not source:
        raise ProbeSkipped("model_source が指定されていない (real_model tier 未有効)")
    src = Path(source)
    if not src.is_dir():
        raise ProbeSkipped(f"実モデルが見つからない: {src.name}")

    found = reazon_model_files(src)
    if found is None:
        raise ProbeSkipped(f"ReazonSpeech モデルのファイル構成を認識できない: {src.name}")
    variant, files = found

    basedir = ctx.root / "model"
    mechanisms = materialize_tree(src, basedir, include=list(files))
    missing = [n for n in files if not (basedir / n).is_file()]
    if missing:
        raise ProbeSkipped(f"必要ファイルが揃っていない: {missing}")
    ctx.stage("materialize")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(str(basedir), "tokens.txt"),
        encoder=os.path.join(str(basedir), files[1]),
        decoder=os.path.join(str(basedir), files[2]),
        joiner=os.path.join(str(basedir), files[3]),
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    ctx.stage("load")   # ← 1.12.39 でも非 ASCII でここは通った (それが「黙る」ということ)

    # **合成信号は使わない。** 220 Hz の正弦波では token が 1 つも出ないことがあり、
    # その場合 token id -> SymbolTable の lookup を**通らずに** pass できてしまう —
    # まさに本 probe が守っている経路を素通りする。実発話を使い、token が出たことを
    # 下で必須にする。
    sample_rate, audio = load_probe_speech("ja/jsut_basic5000_0001")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    ctx.stage("decode")  # ← 1.12.39 ではここで IndexError

    # SymbolTable lookup を実際に通ったことの証明。空だと「何も引かずに通った」
    # だけで、regression ゲートとして無意味になる。
    tokens = list(getattr(stream.result, "tokens", None) or [])
    if not tokens:
        raise ProbeSkipped(
            "decode が token を 1 つも返さなかった - SymbolTable lookup を通って"
            "おらず、本 probe は修正対象経路を検証できていない"
        )

    return {
        "materialization": dominant_mechanism(mechanisms),
        "decoded_type": type(stream.result.text).__name__,
        "decoded_is_str": isinstance(stream.result.text, str),
        "model_variant": variant,
        "token_count": len(tokens),
    }


def _restore_nemo(ctx: ProbeContext, *, model_dir: Path) -> dict:
    """``.nemo`` を ``model_dir`` へ実体化して ``restore_from`` を通す共通処理。

    ``model_dir`` と ``%TEMP%`` のどちらを非 ASCII にするかを呼び出し側で
    変えることで、**どちらが主因か**を切り分けられる。
    """
    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:  # noqa: BLE001 - NeMo は ImportError 以外も投げる
        raise ProbeSkipped(
            f"nemo-toolkit 未導入 (`uv sync --extra engines-nemo` が必要): {exc}"
        ) from exc

    source = ctx.payload.get("model_source")
    if not source:
        raise ProbeSkipped("model_source が指定されていない (heavy tier 未有効)")
    src = Path(source)
    if not src.is_file():
        raise ProbeSkipped(f".nemo が見つからない: {src.name}")

    from ..artifacts import materialize_file

    staged = model_dir / src.name
    mechanism = materialize_file(src, staged)
    ctx.stage("materialize")

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(staged), map_location="cpu"
    )
    ctx.stage("restore_from")

    return {
        "materialization": mechanism,
        "model_class": type(model).__name__,
        "has_tokenizer": hasattr(model, "tokenizer"),
    }


def _ascii_side(ctx: ProbeContext, leaf: str) -> Path:
    """ASCII 保証されたスクラッチ領域 (control / trial で分ける)。"""
    side = Path(ctx.payload["ascii_scratch"]) / ("control" if ctx.is_control else "trial") / leaf
    side.mkdir(parents=True, exist_ok=True)
    return side


@probe("nemo.restore_from.nonascii_model_ascii_temp")
def nemo_restore_from_nonascii_model(ctx: ProbeContext) -> dict:
    """**.nemo のパスだけ**を非 ASCII にする (``%TEMP%`` は ASCII に固定)。

    これが落ちるなら、``restore_path`` そのものが narrow path 境界であり、
    ``%TEMP%`` の移設だけでは直らない — #379 は ``.nemo`` の staging も要る。
    ``%TEMP%`` の ASCII 固定は呼び出し側が ``env_extra`` で行う。
    """
    return _restore_nemo(ctx, model_dir=ctx.root / "model")


@probe("nemo.restore_from.ascii_model_nonascii_temp")
def nemo_restore_from_nonascii_temp(ctx: ProbeContext) -> dict:
    """**``%TEMP%`` だけ**を非 ASCII にする (``.nemo`` は ASCII 側に置く)。

    これが落ちるなら、NeMo が内部で選ぶ untar 先が narrow path 境界であり、
    ``.nemo`` を staging しても ``%TEMP%`` を移設しないと直らない。
    """
    return _restore_nemo(ctx, model_dir=_ascii_side(ctx, "nemo-model"))


@probe("nemo.restore_from")
def nemo_restore_from(ctx: ProbeContext) -> dict:
    """``nemo_asr.models.ASRModel.restore_from(restore_path=...)``。

    **実運用条件** — ``.nemo`` のパスも ``%TEMP%`` も非 ASCII になる。
    どちらが主因かは ``nemo.restore_from.nonascii_model_ascii_temp`` /
    ``nemo.restore_from.ascii_model_nonascii_temp`` の 2 行で分離している。

    heavy tier。``sentencepiece`` / ``nemo-toolkit`` は ``engines-nemo`` extra
    側にあり、既定の開発環境では未導入なので通常は skip される。
    有効化するには ``uv sync --extra engines-nemo``。
    """
    return _restore_nemo(ctx, model_dir=ctx.root / "model")
