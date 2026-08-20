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
    materialize_tree,
    tiny_onnx_bytes,
    write_invalid_onnx,
    write_tiny_onnx,
    write_tokens_txt,
)
from ..record import ProbeContext, ProbeSkipped
from . import probe

#: ReazonSpeech int8 モデルが必要とする 4 ファイル
#: (``reazonspeech_engine.py`` の ``required_files`` と同じ構成)
_REAZON_INT8_FILES = (
    "tokens.txt",
    "encoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
)


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


@probe("sherpa.from_transducer.diff")
def sherpa_from_transducer_diff(ctx: ProbeContext) -> dict:
    """**差分プローブ** — 実モデル無しで sherpa-onnx の narrow path を判定する。

    意図的に不正な ONNX と ``tokens.txt`` を置き、ASCII / 非 ASCII で
    **エラー署名がどう変わるか**を見る。

    - ASCII で「protobuf の解析に失敗」→ ファイルは開けている
    - 非 ASCII で「ファイルが開けない / 見つからない」→ **narrow path 確定**

    ``from_transducer`` は不正モデルに対して native 側で ``abort()`` し得るので、
    worker が子プロセスであることが必須 (親で走らせると run 全体が死ぬ)。
    """
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ProbeSkipped(f"sherpa-onnx 未導入: {exc}") from exc

    basedir = ctx.root / "model"
    basedir.mkdir(parents=True, exist_ok=True)
    write_tokens_txt(basedir / "tokens.txt")
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx"):
        write_invalid_onnx(basedir / name)
    ctx.stage("prepare_model_dir")

    error_class = None
    error_text = ""
    try:
        sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=os.path.join(str(basedir), "tokens.txt"),
            encoder=os.path.join(str(basedir), "encoder.onnx"),
            decoder=os.path.join(str(basedir), "decoder.onnx"),
            joiner=os.path.join(str(basedir), "joiner.onnx"),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
        )
        ctx.stage("constructed")
    except BaseException as exc:  # noqa: BLE001 - 署名の比較が目的
        error_class = type(exc).__name__
        error_text = str(exc)
    ctx.stage("attempted")

    lowered = error_text.lower()
    # **エラーの分類だけ**を観測にする (メッセージ本文にはパスが含まれるため、
    # そのまま返すと control と必ず差分が出てしまう)。
    #
    # 注意: 不正な ONNX は tokens.txt より**先に**検証されるため、この差分
    # プローブが到達できるのは ONNX 層までである。既知 NG の本体
    # (tokens.txt の SymbolTable 誤読) は real_model tier でしか測れない。
    return {
        "error_class": error_class,
        "mentions_parse_failure": any(
            k in lowered for k in ("parse", "protobuf", "proto", "invalid model", "load model")
        ),
        "mentions_open_failure": any(
            k in lowered for k in ("no such file", "not exist", "cannot open", "failed to open", "open file")
        ),
        "constructed_without_error": error_class is None,
    }


@probe("sherpa.from_transducer.real")
def sherpa_from_transducer_real(ctx: ProbeContext) -> dict:
    """**positive control** — 実 ReazonSpeech モデルで既知 NG を再現する。

    既知の挙動: 非 ASCII パスでもロードは**成功**し、decode が全件
    ``IndexError`` になる。ハーネスがこれを ``fail_silent`` と分類できることが、
    実ネイティブコードに対する検出能力の証明になる。

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

    basedir = ctx.root / "model"
    mechanisms = materialize_tree(src, basedir, include=list(_REAZON_INT8_FILES))
    missing = [n for n in _REAZON_INT8_FILES if not (basedir / n).is_file()]
    if missing:
        raise ProbeSkipped(f"必要ファイルが揃っていない: {missing}")
    ctx.stage("materialize")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(str(basedir), "tokens.txt"),
        encoder=os.path.join(str(basedir), _REAZON_INT8_FILES[1]),
        decoder=os.path.join(str(basedir), _REAZON_INT8_FILES[2]),
        joiner=os.path.join(str(basedir), _REAZON_INT8_FILES[3]),
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    ctx.stage("load")   # ← 非 ASCII でもここは通る (それが「黙る」ということ)

    audio = (
        0.1
        * np.sin(2 * np.pi * 220.0 * np.arange(16000, dtype=np.float64) / 16000.0)
    ).astype(np.float32)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)
    recognizer.decode_stream(stream)
    ctx.stage("decode")  # ← 既知 NG ではここで IndexError

    return {
        "materialization": dominant_mechanism(mechanisms),
        "decoded_type": type(stream.result.text).__name__,
        "decoded_is_str": isinstance(stream.result.text, str),
    }


@probe("nemo.restore_from")
def nemo_restore_from(ctx: ProbeContext) -> dict:
    """``nemo_asr.models.ASRModel.restore_from(restore_path=...)``。

    heavy tier。``sentencepiece`` / ``nemo-toolkit`` は ``engines-nemo`` extra
    側にあり、既定の開発環境では未導入なので通常は skip される。
    有効化するには ``uv sync --extra engines-nemo``。
    """
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401
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

    staged = ctx.root / "model" / src.name
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
