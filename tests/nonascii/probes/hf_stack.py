"""HuggingFace / torch / tokenizer 系の境界プローブ (Issue #378)。

cheap tier は合成アーティファクトとオフライン (``local_files_only``) で完結する。
real_model tier だけがローカルの実モデルを使う (**ネットワークは使わない**)。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from ..record import ProbeContext, ProbeSkipped
from . import probe


@probe("urllib.urlretrieve.file_url")
def urllib_urlretrieve_file_url(ctx: ProbeContext) -> dict:
    """``model_manager.download_file`` の ``urlretrieve(url, destination)``。

    ``file://`` を source にすることで、**ネットワーク無し**で実コード経路
    (保存先パスが非 ASCII) を通せる。
    """
    payload = b"livecap model payload" * 16
    source = ctx.root / "source.bin"
    source.write_bytes(payload)
    ctx.stage("prepare_source")

    url = source.resolve().as_uri()
    destination = ctx.root / "downloads" / "model.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(url, str(destination))
    ctx.stage("urlretrieve")

    return {
        "size": destination.stat().st_size,
        "content_matches": destination.read_bytes() == payload,
    }


@probe("huggingface_hub.local_files_only")
def huggingface_hub_local_files_only(ctx: ProbeContext) -> dict:
    """非 ASCII の ``HF_HOME`` に置いた snapshot をオフラインで解決する。

    ``model_manager.huggingface_cache()`` が ``HF_HOME`` 経由で渡す経路と同一。
    合成した ``models--org--name/{blobs,refs,snapshots}`` ツリーを使うので
    ネットワークもモデルも不要。
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ProbeSkipped(f"huggingface_hub 未導入: {exc}") from exc

    repo_id = "livecap-probe/tiny"
    hub = ctx.root / "hf" / "hub"
    repo_dir = hub / "models--livecap-probe--tiny"
    commit = "0" * 40

    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "probe"}), encoding="utf-8"
    )
    (snapshot / "README.md").write_text("probe", encoding="utf-8")
    ctx.stage("prepare_snapshot")

    resolved = snapshot_download(
        repo_id=repo_id, cache_dir=str(hub), local_files_only=True
    )
    ctx.stage("snapshot_download")

    config = Path(resolved) / "config.json"
    return {
        "resolved_under_probe_root": str(Path(resolved)).startswith(str(ctx.root)),
        "config_readable": config.is_file(),
        "model_type": json.loads(config.read_text(encoding="utf-8"))["model_type"]
        if config.is_file()
        else None,
    }


@probe("torch.load.path")
def torch_load_path(ctx: ProbeContext) -> dict:
    """``torch.load(<path>)`` — Voxtral の重み読み込み層。

    ``torch.load`` は ``IO[bytes]`` も受けるので、仮に NG でも方式①へ退避できる。
    その事実も観測に残す。
    """
    try:
        import torch
    except ImportError as exc:
        raise ProbeSkipped(f"torch 未導入 (engines-torch extra): {exc}") from exc

    tensor = torch.arange(8, dtype=torch.float32)
    path = ctx.root / "weights.pt"
    torch.save({"w": tensor}, str(path))
    ctx.stage("save")

    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    ctx.stage("load_from_path")

    with open(path, "rb") as fh:
        loaded_buf = torch.load(fh, map_location="cpu", weights_only=True)
    ctx.stage("load_from_fileobj")

    return {
        "path_values": loaded["w"].tolist(),
        "buffer_values": loaded_buf["w"].tolist(),
        "buffer_api_available": True,
    }


@probe("safetensors.load_file.path")
def safetensors_load_file_path(ctx: ProbeContext) -> dict:
    """``safetensors.torch.load_file(<path>)`` — Voxtral の ``use_safetensors=True`` 経路。

    ``safetensors.torch.load(data: bytes)`` があるので方式①も可能。
    """
    try:
        import torch
        from safetensors.torch import load, load_file, save_file
    except ImportError as exc:
        raise ProbeSkipped(f"safetensors/torch 未導入: {exc}") from exc

    path = ctx.root / "model.safetensors"
    save_file({"w": torch.arange(8, dtype=torch.float32)}, str(path))
    ctx.stage("save")

    from_path = load_file(str(path))
    ctx.stage("load_from_path")

    from_bytes = load(path.read_bytes())
    ctx.stage("load_from_bytes")

    return {
        "path_values": from_path["w"].tolist(),
        "bytes_values": from_bytes["w"].tolist(),
        "bytes_api_available": True,
    }


@probe("tokenizers.from_file")
def tokenizers_from_file(ctx: ProbeContext) -> dict:
    """``tokenizers.Tokenizer.from_file(<path>)`` — Rust native の読み込み経路。

    whispers2t / transformers の tokenizer ロードが共有する層。
    """
    try:
        from tokenizers import Tokenizer, models
    except ImportError as exc:
        raise ProbeSkipped(f"tokenizers 未導入: {exc}") from exc

    vocab = {"[UNK]": 0, "ab": 1, "cd": 2}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    path = ctx.root / "tokenizer.json"
    tokenizer.save(str(path))
    ctx.stage("save")

    loaded = Tokenizer.from_file(str(path))
    ctx.stage("from_file")

    encoded = loaded.encode("ab cd")
    return {"ids": list(encoded.ids), "vocab_size": loaded.get_vocab_size()}


# --- real_model tier ----------------------------------------------------------


def _require_model_source(ctx: ProbeContext) -> Path:
    source = ctx.payload.get("model_source")
    if not source:
        raise ProbeSkipped("model_source が指定されていない (real_model tier 未有効)")
    path = Path(source)
    if not path.exists():
        raise ProbeSkipped(f"実モデルが見つからない: {path.name}")
    return path


_CONFIG_SUFFIXES = {".json", ".txt", ".model"}


@probe("transformers.autoconfig.local_dir")
def transformers_autoconfig_local_dir(ctx: ProbeContext) -> dict:
    """``AutoConfig.from_pretrained(<local dir>)`` — config / index の解決層。

    **重みは読まない。** モデルローダ境界そのものではなく、その手前の
    「ローカルディレクトリから config と safetensors index を解決する」層を測る。
    実際のモデルロードは ``voxtral.from_pretrained`` が別途測る。
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ProbeSkipped(f"transformers 未導入: {exc}") from exc

    from ..artifacts import dominant_mechanism, materialize_tree

    src = _require_model_source(ctx)
    dst = ctx.root / "config-only"
    include = [p.name for p in src.iterdir() if p.is_file() and p.suffix in _CONFIG_SUFFIXES]
    mechanisms = materialize_tree(src, dst, include=include)
    ctx.stage("materialize")

    config = AutoConfig.from_pretrained(str(dst), local_files_only=True)
    ctx.stage("load_config")

    return {
        "materialization": dominant_mechanism(mechanisms),
        "model_type": getattr(config, "model_type", None),
        "n_config_files": len(mechanisms),
    }


@probe("voxtral.from_pretrained")
def voxtral_from_pretrained(ctx: ProbeContext) -> dict:
    """``VoxtralForConditionalGeneration.from_pretrained(<local dir>)`` — **実ロード**。

    レビュー指摘 5 への対応: 以前は ``AutoConfig`` しか呼んでおらず、モデルローダ
    境界の pass と主張するには弱かった。**重み (safetensors 2 shard / 8.8 GB) を
    含めて実体化し、実際にモデルを構築する**。

    hardlink が効けば実体化は 0 バイト・数ミリ秒で済むため、コストの大半は
    CPU 上のモデル構築 (実測 ~12 秒) である。
    """
    try:
        import torch  # noqa: F401
        from transformers import VoxtralForConditionalGeneration
    except ImportError as exc:
        raise ProbeSkipped(f"transformers/torch 未導入: {exc}") from exc

    from ..artifacts import dominant_mechanism, materialize_tree

    src = _require_model_source(ctx)
    dst = ctx.root / "model"
    # include=None = 重みを含む全ファイル
    mechanisms = materialize_tree(src, dst)
    ctx.stage("materialize")

    model = VoxtralForConditionalGeneration.from_pretrained(
        str(dst),
        dtype="bfloat16",
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    ctx.stage("load_model")

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "materialization": dominant_mechanism(mechanisms),
        "model_class": type(model).__name__,
        "n_files_materialized": len(mechanisms),
        # 完全一致で比較できるよう百万単位に丸める (パスに依存しない観測)
        "n_params_e6": n_params // 1_000_000,
    }


@probe("voxtral.autoprocessor")
def voxtral_autoprocessor(ctx: ProbeContext) -> dict:
    """``AutoProcessor.from_pretrained(<local dir>)`` — tokenizer / processor 層。"""
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ProbeSkipped(f"transformers 未導入: {exc}") from exc

    from ..artifacts import dominant_mechanism, materialize_tree

    src = _require_model_source(ctx)
    dst = ctx.root / "processor"
    include = [p.name for p in src.iterdir() if p.is_file() and p.suffix in _CONFIG_SUFFIXES]
    mechanisms = materialize_tree(src, dst, include=include)
    ctx.stage("materialize")

    try:
        processor = AutoProcessor.from_pretrained(str(dst), local_files_only=True)
    except ImportError as exc:
        # transformers は optional 依存が欠けているとき ImportError を投げる
        # (Voxtral の processor は mistral-common を要求する)。
        # 依存不足は「測定不能」であって境界のバグではない。
        raise ProbeSkipped(
            f"processor の optional 依存が未導入 "
            f"(`uv sync --extra engines-voxtral` が必要): {exc}"
        ) from exc
    ctx.stage("load_processor")

    tokenizer = getattr(processor, "tokenizer", None)
    return {
        "materialization": dominant_mechanism(mechanisms),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer else None,
    }


@probe("whispers2t.load_model")
def whispers2t_load_model(ctx: ProbeContext) -> dict:
    """``whisper_s2t.load_model(...)`` — HF hub + CTranslate2 (native)。

    この engine は ``manager.huggingface_cache()`` で包まれていないため、
    既定の HF cache が実世界の経路になる。worker が ``HF_HOME`` を非 ASCII
    root へ向けているので、その条件を再現できる。
    """
    try:
        import whisper_s2t  # noqa: F401
    except ImportError as exc:
        raise ProbeSkipped(f"whisper-s2t 未導入: {exc}") from exc

    raise ProbeSkipped(
        "既定 HF cache 配下のモデルを非 ASCII HF_HOME へ再配置する実装が未了。"
        "real_model tier の別 PR で対応する。"
    )


@probe("qwen3asr.from_pretrained")
def qwen3asr_from_pretrained(ctx: ProbeContext) -> dict:
    """``Qwen3ASR.from_pretrained(...)``。

    ``qwen_asr`` は ``engines-qwen3asr`` extra 側にあり、既定の開発環境では
    未導入なので skip される。HF snapshot 自体はローカルに存在する。
    """
    try:
        from qwen_asr import Qwen3ASR  # noqa: F401
    except ImportError as exc:
        raise ProbeSkipped(
            f"qwen_asr 未導入 (`uv sync --extra engines-qwen3asr` が必要): {exc}"
        ) from exc

    raise ProbeSkipped("qwen_asr 導入環境での実装は real_model tier の別 PR で対応する。")
