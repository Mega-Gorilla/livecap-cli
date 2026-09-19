"""HuggingFace / torch / tokenizer 系の境界プローブ (Issue #378)。

cheap tier は合成アーティファクトとオフライン (``local_files_only``) で完結する。
real_model tier だけがローカルの実モデルを使う (**ネットワークは使わない**)。
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class _MockHubHandler(BaseHTTPRequestHandler):
    """``huggingface_hub`` が snapshot_download で叩く 2 種類の endpoint だけを返す。

    * ``GET /api/models/<repo>/revision/main`` → ``{"sha", "siblings"}``
    * ``HEAD/GET /<repo>/resolve/<sha>/<file>`` → ``ETag`` / ``X-Repo-Commit`` /
      ``Content-Length`` + 本文

    これで **本物の ``huggingface_hub`` の書き込み経路** (``.locks/`` → ``.incomplete``
    → ``blobs/`` → ``snapshots/<sha>/`` → ``refs/main``) がローカルだけで通る
    (hf_hub 0.36.0 で実測、#428 spike A)。
    """

    repo = "livecap-probe/tiny"
    sha = "a" * 40
    files = {
        "config.json": b'{"model_type": "probe"}',
        "README.md": b"probe",
        "vocab.txt": b"a\nb\n",
    }

    def log_message(self, *args) -> None:  # noqa: D401 - 静かにする
        pass

    def _file(self, path: str):
        if "/resolve/" not in path:
            return None
        return self.files.get(path.split("/resolve/", 1)[1].split("/", 1)[1])

    def _file_headers(self, body: bytes) -> None:
        self.send_header("ETag", '"' + hashlib.sha256(body).hexdigest() + '"')
        self.send_header("X-Repo-Commit", self.sha)
        self.send_header("Content-Length", str(len(body)))

    def do_HEAD(self) -> None:
        body = self._file(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self._file_headers(body)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith(f"/api/models/{self.repo}"):
            payload = json.dumps(
                {
                    "sha": self.sha,
                    "id": self.repo,
                    "siblings": [{"rfilename": name} for name in self.files],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = self._file(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self._file_headers(body)
        self.end_headers()
        self.wfile.write(body)


@probe("huggingface_hub.snapshot_download.local_dir.write")
def huggingface_hub_snapshot_download_local_dir_write(ctx: ProbeContext) -> dict:
    """非 ASCII の管理 staging へ ``snapshot_download(local_dir=)`` が**実際に書き込み**、
    flattened dir + manifest として models root へ publish する経路 (#456)。

    Qwen3-ASR / WhisperS2T / Voxtral / ReazonSpeech が共有する production helper
    (``livecap_cli.engines.hf_cache.fetch_repo_dir``) を通す。``endpoint=`` だけを
    ローカルの mock Hub へ向けるため ``huggingface_hub.snapshot_download`` を partial で
    差し替えるが、書き込み (``download/.cache/huggingface/download/*.metadata`` /
    ``.incomplete`` / 本体、``cache_dir=<管理 hub>`` 側の ``models--*``) は本物の
    ``huggingface_hub`` が、staging → payload → ``publish_dir`` は helper が行う。
    ``max_workers=1`` は helper が固定する (hf_hub#4915)。

    観測は「変異で fail する」ものだけ返す: 配置先が manifest 込みで valid なこと、
    本文一致、staging の消去、``.incomplete`` / ``.cache`` の残存 0、既定 cache 不使用。
    """
    import functools
    from unittest.mock import patch

    try:
        import huggingface_hub
        from huggingface_hub import constants
    except ImportError as exc:
        raise ProbeSkipped(f"huggingface_hub 未導入: {exc}") from exc
    if constants.HF_HUB_OFFLINE:
        raise ProbeSkipped("HF_HUB_OFFLINE=1 の環境では mock Hub へも出られない")

    from livecap_cli.engines.hf_cache import fetch_repo_dir
    from livecap_cli.engines.model_store import TRANSIENT_MARKERS, TRANSIENT_SUFFIXES, validate_repo_dir
    from livecap_cli.resources import get_model_manager

    manager = get_model_manager()
    hub = Path(manager.get_huggingface_cache_dir())
    staging_root = Path(manager.get_temp_dir("downloads"))
    models_root = Path(manager.get_models_dir())
    destination = models_root / "livecap-probe--tiny"
    for path in (hub, staging_root, destination):
        if not path.resolve().is_relative_to(ctx.root.resolve()):
            raise RuntimeError(
                f"管理 root が variant root 配下でない: {ascii(str(path))} (root={ascii(str(ctx.root))})"
            )
    ctx.stage("resolve_managed_roots")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    real = huggingface_hub.snapshot_download
    try:
        with patch("huggingface_hub.snapshot_download", functools.partial(real, endpoint=endpoint)):
            placed = fetch_repo_dir(
                _MockHubHandler.repo,
                hub_root=hub,
                staging_root=staging_root,
                destination=destination,
                ignore_patterns=["README.md"],
                required=["config.json", "vocab.txt"],
            )
        ctx.stage("fetch_repo_dir")
    finally:
        server.shutdown()
        server.server_close()

    manifest = validate_repo_dir(placed, repo_id=_MockHubHandler.repo)
    expected = {k: v for k, v in _MockHubHandler.files.items() if k != "README.md"}
    transient = [
        p for p in models_root.rglob("*")
        if p.name in TRANSIENT_MARKERS or p.name.endswith(TRANSIENT_SUFFIXES)
    ]
    return {
        "placed_under_probe_root": placed.resolve().is_relative_to(ctx.root.resolve()),
        "manifest_valid": manifest is not None,
        "manifest_files": sorted(f.path for f in manifest.files) if manifest else None,
        "commit_sha_recorded": bool(manifest and manifest.commit_sha == _MockHubHandler.sha),
        "content_matches": all(
            (placed / n).is_file() and (placed / n).read_bytes() == b for n, b in expected.items()
        ),
        "readme_excluded": not (placed / "README.md").exists(),
        "staging_removed": not (staging_root / destination.name).exists(),
        "incomplete_leftovers": len(list(staging_root.rglob("*.incomplete"))),
        "transient_in_models_root": len(transient),
    }


@probe("huggingface_hub.hf_hub_download.local_dir.write")
def huggingface_hub_hf_hub_download_local_dir_write(ctx: ProbeContext) -> dict:
    """非 ASCII の管理 staging へ ``hf_hub_download(local_dir=)`` が**実際に書き込み**、
    models root へ move する経路 (#447)。

    NeMo (canary / parakeet) の ``.nemo`` 取得と同じ production helper
    (``livecap_cli.engines.hf_cache.download_file``) を通す。``endpoint=`` だけを
    ローカルの mock Hub へ向けるため ``huggingface_hub.hf_hub_download`` を
    partial で差し替えるが、書き込み (``.cache/huggingface/download/*.metadata`` /
    ``.incomplete`` / 本体) と move は本物の ``huggingface_hub`` / helper が行う。
    ``huggingface_hub.snapshot_download.write`` (cache 階層への書き込み) では代用できない —
    ``local_dir=`` は別経路である。

    観測: 配置先の本文一致、staging の消去、``.incomplete`` の残存 0、既定 cache 不使用。
    """
    import functools
    from unittest.mock import patch

    try:
        import huggingface_hub
        from huggingface_hub import constants
    except ImportError as exc:
        raise ProbeSkipped(f"huggingface_hub 未導入: {exc}") from exc
    if constants.HF_HUB_OFFLINE:
        raise ProbeSkipped("HF_HUB_OFFLINE=1 の環境では mock Hub へも出られない")

    from livecap_cli.engines.hf_cache import download_file
    from livecap_cli.resources import get_model_manager

    manager = get_model_manager()
    staging = Path(manager.get_temp_dir("downloads")) / "models--livecap-probe--tiny"
    destination = Path(manager.get_models_dir()) / "livecap-probe--tiny.bin"
    for path in (staging, destination):
        if not path.resolve().is_relative_to(ctx.root.resolve()):
            raise RuntimeError(
                f"管理 root が variant root 配下でない: {ascii(str(path))} (root={ascii(str(ctx.root))})"
            )
    ctx.stage("resolve_managed_roots")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    real = huggingface_hub.hf_hub_download
    try:
        with patch("huggingface_hub.hf_hub_download", functools.partial(real, endpoint=endpoint)):
            placed = download_file(
                _MockHubHandler.repo,
                "vocab.txt",
                hub_root=Path(manager.get_huggingface_cache_dir()),
                staging_dir=staging,
                destination=destination,
            )
        ctx.stage("download_file")
    finally:
        server.shutdown()
        server.server_close()

    incomplete = list(Path(manager.get_temp_dir("downloads")).rglob("*.incomplete"))
    return {
        "placed_under_probe_root": placed.resolve().is_relative_to(ctx.root.resolve()),
        "content_matches": placed.is_file() and placed.read_bytes() == _MockHubHandler.files["vocab.txt"],
        "staging_removed": not staging.exists(),
        "incomplete_leftovers": len(incomplete),
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


def _materialize_model_dir(
    ctx: ProbeContext, source: Path, *, repo_id: str, variant: "str | None" = None
) -> tuple:
    """source の flattened dir (manifest 込み) を worker の ``models_root`` へ実体化する (#456)。

    production は ``<models_root>/<org>--<name>/`` を ``validate_repo_dir`` で確かめてから
    ローカル dir をそのまま engine へ渡す。probe も同じ形にする: 実体化した先が variant root
    配下で、manifest が valid であることを検査してから受け側のネイティブへ渡す。
    """
    from livecap_cli.engines.model_store import validate_repo_dir
    from livecap_cli.resources import get_model_manager

    from ..artifacts import materialize_tree

    if validate_repo_dir(source, repo_id=repo_id, variant=variant) is None:
        raise ProbeSkipped(
            f"source が ModelRoot 契約の形 (manifest 込み) でない: {ascii(str(source))} "
            "(`livecap-cli` で 1 度ロードして migration を通すこと)"
        )
    models_root = Path(get_model_manager().get_models_dir())
    if not models_root.resolve().is_relative_to(ctx.root.resolve()):
        raise RuntimeError(
            f"models_root が variant root 配下でない: {ascii(str(models_root))} "
            f"(root={ascii(str(ctx.root))})"
        )
    dst = models_root / source.name
    mechanisms = materialize_tree(source, dst)
    if validate_repo_dir(dst, repo_id=repo_id, variant=variant) is None:
        raise RuntimeError(f"実体化した dir が manifest と一致しない: {ascii(str(dst))}")
    return dst, mechanisms


@probe("whispers2t.load_model")
def whispers2t_load_model(ctx: ProbeContext) -> dict:
    """``whisper_s2t.load_model(<models_root 内のローカル dir>)`` — CTranslate2 + tokenizers。

    **測るのは受け側のネイティブが非 ASCII path を扱えるかである。**
    ``WhisperModelCT2.__init__`` は同じ path を 2 つのネイティブへ渡す::

        ctranslate2.models.Whisper(self.model_path, ...)          # C++
        tokenizers.Tokenizer.from_file(model_path/"tokenizer.json")  # Rust

    **production と同じ手順である** (#430 / #456)::

        model_dir = <models_root>/Systran--faster-whisper-base/      # flattened dir + manifest
        validate_repo_dir(model_dir, repo_id=..., variant="base")   # cache hit の唯一の条件
        model = whisper_s2t.load_model(model_identifier=str(model_dir), ...)

    source (実 models root の dir) を variant root 配下の ``models_root`` へ実体化してから
    渡す。worker の ``HF_HUB_CACHE`` は空の ASCII scratch + ``HF_HUB_OFFLINE=1`` なので、
    どこかで Hub へ出ようとすれば落ちる。

    ``%TEMP%`` は ASCII へ固定してある (``ascii_pinned_roots``) — モデル path 以外の
    変数を混ぜないため。効いていなければ **fail loud** させる。
    """
    try:
        import whisper_s2t
    except ImportError as exc:
        raise ProbeSkipped(f"whisper-s2t 未導入: {exc}") from exc

    from ..artifacts import dominant_mechanism
    from .utterance_wav import _WHISPERS2T_REPO_ID, _assert_hf_pins_took_effect

    source = _require_model_source(ctx)
    hub_cache_pin = ctx.payload.get("hf_hub_cache_pin")
    if not hub_cache_pin:
        raise ProbeSkipped("hf_hub_cache_pin が payload に無い (real_model tier 未有効)")
    _assert_hf_pins_took_effect(str(hub_cache_pin))
    ctx.stage("verify_hf_pins")

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

    model_dir, mechanisms = _materialize_model_dir(
        ctx, source, repo_id=_WHISPERS2T_REPO_ID, variant="base"
    )
    ctx.stage("materialize")

    model = whisper_s2t.load_model(
        model_identifier=str(model_dir),
        backend="CTranslate2",
        device="cpu",
        compute_type="float32",
    )
    ctx.stage("load_model")

    model_path = Path(model.model_path).resolve()
    # **報告ではなく assert する。** observation に入れても control と trial の
    # **両方**が同じ値になるので差分判定では捕まらない — `os.path.isdir` 分岐に
    # 入り損ねて download_model() が共有 cache を返しても、両側とも False で
    # 一致して **pass になってしまう** (変異で確認済み)。
    if not model_path.is_relative_to(ctx.root.resolve()):
        raise RuntimeError(
            f"モデルが probe root 配下から読まれていない: {ascii(str(model_path))} "
            f"(root={ascii(str(ctx.root))}) - os.path.isdir 分岐に入らず "
            "download_model() へ落ちている。境界を通っていない"
        )
    return {
        "materialization": dominant_mechanism(mechanisms),
        "model_dir_has_manifest": (model_dir / "livecap-manifest.json").is_file(),
        "model_class": type(model).__name__,
        "tokenizer_class": type(model.tokenizer).__name__,
        "is_multilingual": bool(model.model.is_multilingual),
    }


@probe("qwen3asr.from_pretrained")
def qwen3asr_from_pretrained(ctx: ProbeContext) -> dict:
    """``Qwen3ASRModel.from_pretrained(<models_root 内のローカル dir>)`` — **未緩和の %TEMP% で**。

    **本行はローカル dir からの load 境界である** (#387 で再定義した)。download /
    ModelRoot への書き込みは ``huggingface_hub.snapshot_download.local_dir.write`` が持つ
    (#456) — ``ascii_safe_temp_environment()`` が変更するのは ``TEMP`` だけなので、
    両者は独立している。

    **production と同じ手順である** (#428 / #456)::

        model_dir = <models_root>/Qwen--Qwen3-ASR-0.6B/          # flattened dir + manifest
        validate_repo_dir(model_dir, repo_id=...)                # cache hit の唯一の条件
        model = Qwen3ASR.from_pretrained(str(model_dir), device_map=...)

    source (実 models root の dir) を variant root 配下の ``models_root`` へ実体化してから
    渡す。worker の ``HF_HUB_CACHE`` は空の ASCII scratch + ``HF_HUB_OFFLINE=1`` なので、
    ``AutoProcessor`` 側が repo ID で既定 cache へ行こうとすれば落ちる。

    **``%TEMP%`` をあえて緩和しない。** production は
    ``ascii_safe_temp_environment(boundary=..., purpose="download")`` で包んでいるが、
    包んだ理由は「② が未確定」であって「③ が必要と分かった」ではない (#378 §6.10)。
    **未緩和の非 ASCII ``%TEMP%`` で load できるなら wrapper は要らない**ので、それを
    測る。したがってモデル path と ``%TEMP%`` の 2 つが同時に非 ASCII になる
    **実運用条件の計測**である (pass すれば曖昧さは無い)。

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

    from ..artifacts import dominant_mechanism
    from .utterance_wav import _QWEN3ASR_REPO_ID, _assert_hf_pins_took_effect

    source = _require_model_source(ctx)
    hub_cache_pin = ctx.payload.get("hf_hub_cache_pin")
    if not hub_cache_pin:
        raise ProbeSkipped("hf_hub_cache_pin が payload に無い (real_model tier 未有効)")
    _assert_hf_pins_took_effect(str(hub_cache_pin))
    ctx.stage("verify_hf_pins")

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

    model_dir, mechanisms = _materialize_model_dir(ctx, source, repo_id=_QWEN3ASR_REPO_ID)
    ctx.stage("materialize")

    # device は CPU 固定。**測るのは load であって推論ではない**ので、GPU にして
    # 他の probe と VRAM を奪い合う理由が無い。
    loaded = Qwen3ASRModel.from_pretrained(str(model_dir), device_map="cpu")
    ctx.stage("from_pretrained")

    model = getattr(loaded, "model", None)
    processor = getattr(loaded, "processor", None)
    return {
        "materialization": dominant_mechanism(mechanisms),
        "model_dir_has_manifest": (model_dir / "livecap-manifest.json").is_file(),
        "wrapper_class": type(loaded).__name__,
        "model_class": type(model).__name__ if model is not None else None,
        "processor_class": type(processor).__name__ if processor is not None else None,
    }
