"""HuggingFace / torch / tokenizer 系の境界プローブ (Issue #378)。

cheap tier は合成アーティファクトとオフライン (``local_files_only``) で完結する。
real_model tier だけがローカルの実モデルを使う (**ネットワークは使わない**)。
"""

from __future__ import annotations

import json
import tempfile
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


def faster_whisper_snapshot_dir(model_size: str = "base") -> "Path | None":
    """WhisperS2T が実際に使う cache 内の faster-whisper snapshot。無ければ ``None``。

    既定を ``"base"`` にしてあるのは **CI の warm step が base を温めている**ため
    (``warm('whispers2t', 'cuda', 'en', 'base')``)。ここを変えるなら warm も揃えないと、
    CI で **snapshot が無くて skip** になりゲートが落ちる。

    **``HF_HOME`` は経路ではない。** WhisperS2T の CTranslate2 backend は自前の cache を
    ``snapshot_download(cache_dir=...)`` へ明示的に渡す::

        CACHE_DIR = user_cache_dir("whisper_s2t")            # whisper_s2t/__init__.py
        kwargs["cache_dir"] = f"{CACHE_DIR}/models"          # backends/ctranslate2/hf_utils.py

    ここでは ``platformdirs`` を**同じ引数で**呼んで所在を求める (自前で組み立てない)。
    親プロセス側の precondition からも使えるよう、``whisper_s2t`` の import は要求しない —
    あちらを import すると重い依存が芋づるで入る。**probe 側では import 後に
    ``hf_utils.CACHE_DIR`` と突き合わせて、ずれていたら fail loud させる。**

    この cache は ``%LOCALAPPDATA%`` 配下でユーザー名を含み、**我々からは設定できない**
    (``LOCALAPPDATA`` を差し替えても platformdirs が ``SHGetKnownFolderPath`` で解決する。
    実測)。その供給側の問題は **#430** が持つ。
    """
    try:
        from platformdirs import user_cache_dir
    except ImportError:
        return None

    repo = Path(user_cache_dir("whisper_s2t")) / "models" / f"models--Systran--faster-whisper-{model_size}"
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None
    return next((d for d in sorted(snapshots.iterdir()) if d.is_dir()), None)


@probe("whispers2t.load_model")
def whispers2t_load_model(ctx: ProbeContext) -> dict:
    """``whisper_s2t.load_model(<ローカル snapshot dir>)`` — CTranslate2 + tokenizers。

    **測るのは受け側のネイティブが非 ASCII path を扱えるかである。**
    ``WhisperModelCT2.__init__`` は同じ path を 2 つのネイティブへ渡す::

        ctranslate2.models.Whisper(self.model_path, ...)          # C++
        tokenizers.Tokenizer.from_file(model_path/"tokenizer.json")  # Rust

    **cache 経路は測っていない。** production はサイズ文字列 (``"base"``) を渡すので
    ``download_model()`` 側へ入る。本 probe は dir を渡して ``os.path.isdir`` 側へ入る。
    cache の所在と書き込みは **#430** が持つ。

    ``%TEMP%`` は ASCII へ固定してある (``ascii_pinned_roots``) — モデル path 以外の
    変数を混ぜないため。効いていなければ **fail loud** させる。
    """
    try:
        import whisper_s2t
        from whisper_s2t.backends.ctranslate2 import hf_utils
    except ImportError as exc:
        raise ProbeSkipped(f"whisper-s2t 未導入: {exc}") from exc

    from ..artifacts import dominant_mechanism, materialize_tree

    snapshot = faster_whisper_snapshot_dir()
    if snapshot is None:
        raise ProbeSkipped(
            "faster-whisper の snapshot が見つからない "
            "(`livecap-cli` で whispers2t base を 1 度ロードして温めること)"
        )
    # **所在の求め方が上流とずれていないことを確かめる。** platformdirs を同じ引数で
    # 呼んでいるだけなので通常は一致するが、上流が cache の決め方を変えたら気付きたい。
    if not str(snapshot).startswith(str(Path(hf_utils.CACHE_DIR) / "models")):
        raise RuntimeError(
            f"snapshot の所在が whisper_s2t の CACHE_DIR とずれている: "
            f"{ascii(str(snapshot))} / CACHE_DIR={ascii(str(hf_utils.CACHE_DIR))}"
        )

    # **%TEMP% の ASCII 固定が効いていること。** 効いていないとモデル path 以外の
    # 変数が混入し、失敗したときどちらが原因か切り分けられない。
    #
    # **「ASCII か」で判定してはならない。** control の root は常に ASCII なので
    # control では発火せず、trial だけが落ちて **fail_loud (= 境界が壊れた)** に
    # 見えてしまう。実際はハーネスの設定ミスである。**「variant root の外に
    # 逃がされているか」**で見ると control でも同じく落ち、error_harness になる。
    tmpdir = Path(tempfile.gettempdir()).resolve()
    if tmpdir.is_relative_to(ctx.root.resolve()):
        raise RuntimeError(
            f"%TEMP% が variant root 配下にある: {ascii(str(tmpdir))} - "
            "ascii_pinned_roots の TEMP 固定が効いていない (モデル path 以外の"
            "変数が混入する)"
        )
    if not str(tmpdir).isascii():
        raise RuntimeError(
            f"%TEMP% の固定先が非 ASCII: {ascii(str(tmpdir))} - "
            "ASCII 側へ逃がせていない"
        )

    dst = ctx.root / "ct2model"
    mechanisms = materialize_tree(snapshot, dst)
    ctx.stage("materialize")

    model = whisper_s2t.load_model(
        model_identifier=str(dst),
        backend="CTranslate2",
        device="cpu",
        compute_type="float32",
    )
    ctx.stage("load_model")

    resolved = Path(model.model_path).resolve()
    # **報告ではなく assert する。** observation に入れても control と trial の
    # **両方**が同じ値になるので差分判定では捕まらない — `os.path.isdir` 分岐に
    # 入り損ねて download_model() が共有 cache を返しても、両側とも False で
    # 一致して **pass になってしまう** (変異で確認済み)。
    if not resolved.is_relative_to(ctx.root.resolve()):
        raise RuntimeError(
            f"モデルが probe root 配下から読まれていない: {ascii(str(resolved))} "
            f"(root={ascii(str(ctx.root))}) - os.path.isdir 分岐に入らず "
            "download_model() へ落ちている。境界を通っていない"
        )
    return {
        "materialization": dominant_mechanism(mechanisms),
        "model_class": type(model).__name__,
        "tokenizer_class": type(model.tokenizer).__name__,
        "is_multilingual": bool(model.model.is_multilingual),
    }


@probe("qwen3asr.from_pretrained")
def qwen3asr_from_pretrained(ctx: ProbeContext) -> dict:
    """``Qwen3ASRModel.from_pretrained(<ローカル snapshot dir>)`` — **未緩和の %TEMP% で**。

    **本行はローカル snapshot からの load 境界である** (#387 で再定義した)。以前は
    「初回ダウンロード境界」と説明していたが、download / cache への書き込みは
    **#428** が持つ — ``ascii_safe_temp_environment()`` が変更するのは ``TEMP`` だけで
    HF cache には触れないので、両者は独立している。

    **``%TEMP%`` をあえて緩和しない。** production は::

        with ascii_safe_temp_environment(boundary=..., purpose="download"):
            model = Qwen3ASR.from_pretrained(self.model_name, ...)

    と包んでいるが、包んだ理由は「② が未確定」であって「③ が必要と分かった」では
    ない (#378 §6.10)。**未緩和の非 ASCII ``%TEMP%`` で load できるなら wrapper は
    要らない**ので、それを測る。したがってモデル path と ``%TEMP%`` の 2 つが同時に
    非 ASCII になる**実運用条件の計測**である (pass すれば曖昧さは無い)。

    **``%TEMP%`` の残存ファイル数は返さない。** 終了後 0 件でも途中で作られて消された
    可能性があり根拠にならない上、**観測は control と trial で差分比較される**ので、
    返した時点で pass/fail の条件になってしまう。測るのは「未緩和の非 ASCII
    ``%TEMP%`` でも load が成功すること」だけである。
    """
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ProbeSkipped(
            f"qwen_asr 未導入 (`uv sync --extra engines-qwen3asr` が必要): {exc}"
        ) from exc

    from ..artifacts import dominant_mechanism, materialize_tree
    from .utterance_wav import qwen3asr_snapshot_dir

    hf_hub_cache = ctx.payload.get("hf_hub_cache")
    if not hf_hub_cache:
        raise ProbeSkipped("hf_hub_cache が payload に無い (real_model tier 未有効)")
    snapshot = qwen3asr_snapshot_dir(hf_hub_cache)
    if snapshot is None:
        raise ProbeSkipped(
            f"HF hub cache に Qwen3-ASR の snapshot が無い: {ascii(str(hf_hub_cache))}"
        )

    # **%TEMP% が variant root 配下であること。** 別の場所を指していたら、この行が
    # 測ろうとしている「未緩和の %TEMP%」を再現できていない。
    tmpdir = Path(tempfile.gettempdir()).resolve()
    if not tmpdir.is_relative_to(ctx.root.resolve()):
        raise RuntimeError(
            f"%TEMP% が variant root 配下でない: {ascii(str(tmpdir))} "
            f"(root={ascii(str(ctx.root))}) - 未緩和の %TEMP% を測れていない"
        )
    # **trial では非 ASCII でなければ意味が無い。** ASCII に見えるなら
    # ascii_pinned_roots へ TEMP が入ったか、variant が効いていない。
    if not ctx.is_control and str(tmpdir).isascii():
        raise RuntimeError(
            f"trial の %TEMP% が ASCII になっている: {ascii(str(tmpdir))} - "
            "ascii_pinned_roots に TEMP を入れると本行の測る意味が消える"
        )

    dst = ctx.root / "qwen-snapshot"
    mechanisms = materialize_tree(snapshot, dst)
    ctx.stage("materialize")

    # device は CPU 固定。**測るのは load であって推論ではない**ので、GPU にして
    # 他の probe と VRAM を奪い合う理由が無い。
    loaded = Qwen3ASRModel.from_pretrained(str(dst), device_map="cpu")
    ctx.stage("from_pretrained")

    model = getattr(loaded, "model", None)
    processor = getattr(loaded, "processor", None)
    return {
        "materialization": dominant_mechanism(mechanisms),
        "wrapper_class": type(loaded).__name__,
        "model_class": type(model).__name__ if model is not None else None,
        "processor_class": type(processor).__name__ if processor is not None else None,
    }
