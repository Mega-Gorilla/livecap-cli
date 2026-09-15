"""発話ごとの一時 wav を **consumer (ネイティブ ASR) へ渡す**境界 (Issue #413)。

**producer 側は測らない。** `soundfile` は Windows で ``sf_wchar_open`` を使うので
非 ASCII path へ書ける (`lib.soundfile.write` 行で確定済み)。問題があるとすれば
**書いた path をネイティブ ASR に渡す側**であり、それは実モデルでしか測れない。

**モデルは ASCII 側に固定し、一時 wav の置き場所だけを非 ASCII にする。**
両方を同時に非 ASCII にすると、失敗したときに「モデルの path が原因」か
「一時 wav の path が原因」かを切り分けられない。`engine.nemo.restore_path_only`
と `engine.nemo.untar_temp` が同じ理由で分かれているのと同じ設計である。

**worker が既に置き場所を variant root へ向けている** (`runner.py`)。

    TEMP / TMP / TMPDIR      -> root/temp    parakeet / canary / qwen3asr
                                             (NamedTemporaryFile の dir 未指定)
    LIVECAP_CORE_CACHE_DIR   -> root/cache   whispers2t (_tmp_dir) / voxtral
                                             (get_temp_dir())

したがって engine を普通に構築して ``transcribe()`` を呼ぶだけで、**production と
同じ経路**を通る。consumer 呼び出しを自前で再実装すると「実際の経路を測っていない」
ことになるので、そうしない。

**判定はハーネスが行う。** ``runner.derive_verdict`` は control (ASCII) と trial
(非 ASCII) の observation を比較し、一致すれば ``pass``、差があれば ``fail_silent``
とする。ここが返すのは**比較用の観測値だけ**である。
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..artifacts import load_probe_speech
from ..record import ProbeContext, ProbeSkipped
from . import probe

@dataclass(frozen=True)
class _Case:
    """engine ごとの probe 定義。

    ``identity_attr`` / ``source_name`` は「**存在確認した source と、実際に
    ロードされたモデルが同一である**」ことを固定するためにある。engine kwargs を
    省くと `EngineFactory` が metadata の既定値をマージするので、**宣言した source
    とは別のモデルがロードされ得る** — 実際 whispers2t は base の存在を
    確認しながら既定の `large-v3` を読んでいた (レビュー指摘)。そうなると
    「persistent runner に偶然残っていたモデル」で緑になり、fresh runner では
    ダウンロード (real_model tier は**ネットワークを使わない**契約) か失敗になる。
    """

    audio_stem: str
    kwargs: dict
    identity_attr: str
    identity_value: str
    #: identity から、`_HEAVY_SOURCES` / `_REAL_MODEL_SOURCES` が指す名前を導く
    source_name: Callable[[str], str]


#: **言語はモデルに合わせる** — 合わない言語だと転写が空/別言語になり、
#: control の非空要求で落ちる。
_ENGINES: dict[str, _Case] = {
    "parakeet": _Case(
        "en/librispeech_1089-134686-0001", {},
        "model_name", "nvidia/parakeet-tdt-0.6b-v2",
        lambda v: f"{v.replace('/', '--')}.nemo",
    ),
    "canary": _Case(
        "en/librispeech_1089-134686-0001", {"language": "en"},
        "model_name", "nvidia/canary-1b-flash",
        lambda v: f"{v.replace('/', '--')}.nemo",
    ),
    "whispers2t": _Case(
        # **model_size を明示する。** 省くと metadata 既定の large-v3 が読まれ、
        # 存在確認した base とは別のモデルになる。
        # source は models root の marker (#430)。重みは管理 HF cache
        # (`models--Systran--faster-whisper-<size>`) にあり、probe が source の hub cache
        # から実体化してから production の `load_model()` を通す (qwen3asr と同じ)。
        "en/librispeech_1089-134686-0001", {"language": "en", "model_size": "base"},
        "model_size", "base",
        lambda v: f"Systran--faster-whisper-{v}.marker",
    ),
    # Issue #418: auto は processor 境界で [None] に整形されるが、probe では
    # 言語を固定して観測値を安定させる。
    "voxtral": _Case(
        "en/librispeech_1089-134686-0001", {"language": "en"},
        "model_name", "mistralai/Voxtral-Mini-3B-2507",
        lambda v: v.replace("/", "--"),
    ),
    # **kwargs を空にするのが要点である** (#413 PR C)。qwen3asr が一時 wav を書くのは
    # `_transcribe_via_wrapper_fallback()` だけで、そこへ入るのは `_asr_language is None`
    # (auto-detect) のときに限られる。**言語を指定すると `_transcribe_with_scores()` へ
    # 行き、境界を迂回してしまう** — 他の 4 engine とは逆に、ここでは言語を固定しない。
    # 迂回した場合は `_WavRecorder` が「variant root 配下に一時 wav が無い」で落とすので、
    # 黙って緑になることはない。
    #
    # source は models root の marker である。**重みは marker の隣に無く** HF hub cache に
    # ある — #428 以降 production は**管理 cache** (`<cache_root>/huggingface/hub`) から
    # `snapshot_download(cache_dir=)` で解決し、marker にはその snapshot path を書く。
    # probe は source の hub cache (`hf_source_cache`) から管理 cache へ実体化してから
    # production 経路 (`load_model()`) を通す。`qwen3asr_snapshot_dir()` が source を確かめる。
    "qwen3asr": _Case(
        "en/librispeech_1089-134686-0001", {},
        "model_name", "Qwen/Qwen3-ASR-0.6B",
        lambda v: f"{v.replace('/', '--')}.marker",
    ),
}


#: HF hub cache 内での Qwen3-ASR snapshot の位置。
_QWEN3ASR_REPO_ID = "Qwen/Qwen3-ASR-0.6B"
_QWEN3ASR_REPO_DIR = "models--Qwen--Qwen3-ASR-0.6B"
#: WhisperS2T (CTranslate2) の base モデル。CI の warm step が base を温めている。
_WHISPERS2T_REPO_ID = "Systran/faster-whisper-base"
_WHISPERS2T_REPO_DIR = "models--Systran--faster-whisper-base"

#: 重みを**管理 HF cache** (`ModelManager.get_huggingface_cache_dir()`) から
#: `snapshot_download(cache_dir=)` で解決する engine (#428 / #430)。probe は source の
#: hub cache からここへ実体化してから production の `load_model()` を通す。
#: (engine_type → hub 内の repo dir)
_HF_MANAGED_ENGINES = {
    "qwen3asr": _QWEN3ASR_REPO_DIR,
    "whispers2t": _WHISPERS2T_REPO_DIR,
}


def hf_snapshot_dir(hub_cache, repo_dir: str) -> "Path | None":
    """``hub_cache`` (hub 階層) 内の ``repo_dir`` の snapshot。無ければ ``None``。

    **marker の存在だけでは足りない。** models root に置かれているのは snapshot path
    を書いただけのテキストで、重みは HF hub cache にある。marker だけを見て「使える」と
    答えると **real_model tier の「ネットワークを使わない」契約を破ってダウンロードが走る**。

    どの hub cache を見るかは呼び出し側が決める — production は #428 / #430 以降
    ``ModelManager.get_huggingface_cache_dir()`` (``<cache_root>/huggingface/hub``) を
    ``snapshot_download(cache_dir=)`` へ明示的に渡す。ハーネスは source として
    管理 cache → 旧 cache (``huggingface_hub`` 既定 / whisper_s2t の自前 cache) の順に
    探す (``test_probes._source_hub_cache()``)。

    判定をここに置くのは ``sherpa.from_transducer.real`` と同じ理由である —
    ``test_probes.py`` 側にファイル名を書くと二重管理になる。
    """
    snapshots = Path(hub_cache) / repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    return next((p for p in sorted(snapshots.iterdir()) if p.is_dir()), None)


def qwen3asr_snapshot_dir(hf_hub_cache) -> "Path | None":
    """``hf_hub_cache`` 内の Qwen3-ASR snapshot (:func:`hf_snapshot_dir` の特殊化)。"""
    return hf_snapshot_dir(hf_hub_cache, _QWEN3ASR_REPO_DIR)


def materialize_hf_snapshot(source_hub_cache, managed_hub_cache, repo_dir: str) -> tuple:
    """source の snapshot を **production と同じ階層** の管理 cache へ実体化する (#428)。

    ``<managed>/<repo_dir>/{refs/main, snapshots/<sha>/}`` を作る。``blobs/`` は作らない —
    実ファイルを snapshot 直下に置く形は、symlink が使えない Windows で ``huggingface_hub``
    自身が書く形 (degraded mode) と同じで、``snapshot_download(local_files_only=True)`` は
    ``refs/main`` と snapshot dir の実在だけで解決する (実測、#428 spike B)。
    source 側が symlink (Linux 等) でも ``materialize_tree`` は実体を辿る。

    戻り値は ``(snapshot_dir, mechanisms)``。mechanisms は ``materialize_tree`` の
    ファイル別方式 (hardlink / copy)。
    """
    from ..artifacts import materialize_tree

    source = hf_snapshot_dir(source_hub_cache, repo_dir)
    if source is None:
        raise RuntimeError(
            f"source hub cache に {repo_dir} の snapshot が無い: {ascii(str(source_hub_cache))}"
        )
    target_repo = Path(managed_hub_cache) / repo_dir
    (target_repo / "refs").mkdir(parents=True, exist_ok=True)
    (target_repo / "refs" / "main").write_text(source.name, encoding="utf-8")
    dst = target_repo / "snapshots" / source.name
    mechanisms = materialize_tree(source, dst)
    return dst, mechanisms


def _ascii_scratch_models_root() -> Path:
    """``ascii_pinned_roots`` で ASCII 側へ逃がされた root の隣に models root を置く。

    どの root が固定されているかは行ごとに違う (qwen3asr は cache、whispers2t は
    resources / TEMP)。固定された root は ``test_probes._isolation_env`` が
    ``<base>/_ascii_pinned/<boundary>/<leaf>`` に作るので、その親の ``models`` を使う。
    """
    for name in ("LIVECAP_CORE_CACHE_DIR", "LIVECAP_RESOURCE_ROOT", "TEMP"):
        value = os.environ.get(name)
        if value and value.isascii():
            return Path(value).parent / "models"
    raise RuntimeError(
        "ASCII 固定された root が無い - ascii_pinned_roots の前提が崩れている "
        f"(cache={ascii(os.environ.get('LIVECAP_CORE_CACHE_DIR'))})"
    )


def _pin_models_root_to_ascii(models_root: str) -> None:
    """models root だけ ASCII の実体へ戻す。**cache / TEMP は variant root のまま。**

    worker は `LIVECAP_CORE_MODELS_DIR` も variant root へ向けるが、そこに実モデルは
    無い。ここで戻さないとダウンロードを試みてしまい、**測りたいのはモデルの path
    ではない**のに測定が壊れる。singleton は構築時に env を読むので reset も要る。
    """
    from livecap_cli.resources import _reset_resources_for_tests

    os.environ["LIVECAP_CORE_MODELS_DIR"] = models_root
    _reset_resources_for_tests()


def _assert_hf_pins_took_effect(hub_cache_pin: str) -> None:
    """HF hub cache の固定と offline 強制が**実際に効いていること**を確かめる。

    qwen3asr の重みは models root ではなく HF hub cache にある。#428 以降 production は
    管理 cache (``get_huggingface_cache_dir()``) を ``cache_dir=`` で明示するので、
    ``huggingface_hub`` の既定 cache (``HF_HUB_CACHE``) は**使われないはず**である。
    それを確かめるため、worker の ``HF_HUB_CACHE`` は**空の ASCII scratch** へ固定する —
    production が ``cache_dir=`` を落として既定 cache へ silent fallback したら、
    ``HF_HUB_OFFLINE=1`` と合わせて「couldn't find them in the cached files」で落ちる。

    **env を設定するのはここではない。** ``huggingface_hub`` は ``HF_HUB_CACHE`` も
    ``HF_HUB_OFFLINE`` も **import 時に確定する**ので、probe の中で ``os.environ`` を
    書き換えても間に合わない。値は worker の**起動前**に
    ``test_probes._real_model_env()`` が渡しており、ここはそれが効いたかを見るだけである。

    **cache path だけを見ては足りない。** 先行 import があっても親 env から継承した
    ``HF_HUB_CACHE`` が期待値と一致していれば path 検査は通る一方、``HF_HUB_OFFLINE``
    は False のままになる (実測)::

        cache_matches=True  constant_offline=False  env_offline='1'

    したがって**両方の定数**を見る。片方でも欠けたら「ネットワークを使わない」保証が
    消えるので fail loud させる。
    """
    import huggingface_hub.constants as hf

    if Path(hf.HF_HUB_CACHE) != Path(hub_cache_pin):
        raise RuntimeError(
            f"HF hub cache の固定が効いていない: {ascii(hf.HF_HUB_CACHE)} "
            f"(期待 {ascii(str(hub_cache_pin))})。worker の env に HF_HUB_CACHE が "
            "渡っていないか、huggingface_hub がそれより前に import されている"
        )
    if any(Path(hub_cache_pin).glob("models--*")):
        raise RuntimeError(
            f"固定した HF hub cache にモデルがある: {ascii(str(hub_cache_pin))} - "
            "空でなければ「既定 cache へ silent fallback していない」ことを証明できない"
        )
    if not hf.HF_HUB_OFFLINE:
        raise RuntimeError(
            "HF_HUB_OFFLINE が効いていない - ダウンロードが走り得る。worker の env に "
            "HF_HUB_OFFLINE=1 が渡っていないか、huggingface_hub がそれより前に "
            "import されている (cache path が一致していてもこちらは効かない)"
        )


class _WavRecorder:
    """`soundfile.write` が受け取った **filesystem path** を記録する。

    「ゲートは緑だが対象経路を通っていない」を防ぐための経路の証明である。
    ``transformers`` は内部で ``BytesIO`` へも書くので、**path だけ**を数える。
    """

    def __init__(self):
        self.paths: list[str] = []
        import soundfile as sf

        self._sf = sf
        self._real = sf.write

    def __enter__(self):
        def recording(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                self.paths.append(str(file))
            return self._real(file, *args, **kwargs)

        self._sf.write = recording
        return self

    def __exit__(self, *exc):
        self._sf.write = self._real
        return False


def _make_probe(engine_type: str):
    case = _ENGINES[engine_type]

    def impl(ctx: ProbeContext) -> dict:
        try:
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError as exc:
            raise ProbeSkipped(f"numpy/soundfile 未導入: {exc}") from exc

        source = ctx.payload.get("model_source")
        models_root = ctx.payload.get("models_root")
        if not source or not models_root:
            raise ProbeSkipped(
                "model_source / models_root が指定されていない (実モデル tier 未有効)"
            )
        if not Path(source).exists():
            raise ProbeSkipped(f"実モデルが見つからない: {Path(source).name}")
        if not str(models_root).isascii():
            # **前提が崩れている。** models root が非 ASCII だと、一時 wav の path
            # だけを変数にできず、切り分けの意味が無くなる。
            raise ProbeSkipped(f"models root が非 ASCII: {ascii(str(models_root))}")

        if engine_type in _HF_MANAGED_ENGINES:
            # **HF 管理 engine は models root を実体へ戻さない** (#428 / #430)。marker は
            # 管理 cache から導出される記録に過ぎず、production の load_model() が書き直す。
            # 実 models root を向けると probe が実環境の marker を書き換えてしまう。
            # ASCII へ固定された root (行ごとに違う: qwen3asr は cache、whispers2t は
            # resources / TEMP) の隣 (同じ scratch) を models root にする。
            _pin_models_root_to_ascii(str(_ascii_scratch_models_root()))
        else:
            _pin_models_root_to_ascii(str(models_root))
        ctx.stage("pin_models_root")

        # qwen3asr / whispers2t は重みが HF hub cache にある。production は #428 / #430
        # 以降 **管理 cache** (`<cache_root>/huggingface/hub`) から解決するので、source の
        # snapshot をそこへ実体化してから production 経路 (load_model) を通す。
        if engine_type in _HF_MANAGED_ENGINES:
            repo_dir = _HF_MANAGED_ENGINES[engine_type]
            source_cache = ctx.payload.get("hf_source_cache")
            hub_cache_pin = ctx.payload.get("hf_hub_cache_pin")
            if not source_cache or not str(source_cache).isascii():
                raise ProbeSkipped(
                    f"source の HF hub cache が未指定 / 非 ASCII: {ascii(str(source_cache))}"
                )
            if not hub_cache_pin:
                raise ProbeSkipped("hf_hub_cache_pin が payload に無い (real_model tier 未有効)")
            if hf_snapshot_dir(source_cache, repo_dir) is None:
                raise ProbeSkipped(
                    f"source の HF hub cache に {repo_dir} の snapshot が無い: "
                    f"{ascii(str(source_cache))} "
                    "(marker だけでは重みの存在を保証しない)"
                )
            _assert_hf_pins_took_effect(str(hub_cache_pin))
            ctx.stage("verify_hf_pins")

            from livecap_cli.resources import get_model_manager

            managed = Path(get_model_manager().get_huggingface_cache_dir())
            if engine_type == "qwen3asr":
                # **この行の変数は一時 wav だけ。** qwen3asr の一時 wav は %TEMP% なので
                # cache root は ASCII 固定 — 管理 cache もその配下でなければならない。
                if not str(managed).isascii():
                    raise RuntimeError(
                        f"管理 HF cache が非 ASCII: {ascii(str(managed))} - "
                        "LIVECAP_CORE_CACHE_DIR の ASCII 固定が効いていない"
                    )
            else:
                # whispers2t の一時 wav は cache_root 配下 (変数) なので、管理 cache も
                # 同じ variant root 配下になる — trial ではモデル dir も非 ASCII。
                # モデル dir 単独は engine.whispers2t.load_model で確定済みで、失敗の
                # 帰属は stages (load_model で止まるか consumer_returned まで行くか) で
                # 切り分ける (registry の measurement_caveat 参照)。
                if not managed.resolve().is_relative_to(ctx.root.resolve()):
                    raise RuntimeError(
                        f"管理 HF cache が variant root 配下でない: {ascii(str(managed))}"
                    )
            materialize_hf_snapshot(source_cache, managed, repo_dir)
            ctx.stage("materialize_managed_cache")

        from livecap_cli.engines import EngineFactory

        # device は auto。実モデル tier は GPU runner でしか有効化されないが、
        # cuda 決め打ちにすると CPU 環境で probe のバグとして落ちる。
        engine = EngineFactory.create_engine(
            engine_type, device="auto", **case.kwargs
        )
        # **存在確認した source と、実際にロードするモデルを一致させる。**
        # ずれていると「runner に偶然残っていたモデル」で緑になり得る。
        actual = getattr(engine, case.identity_attr, None)
        if actual != case.identity_value:
            raise RuntimeError(
                f"{engine_type}: {case.identity_attr}={actual!r} だが "
                f"{case.identity_value!r} を期待している - 宣言した source と"
                "別のモデルをロードしようとしている"
            )
        expected_source = case.source_name(case.identity_value)
        if Path(source).name != expected_source:
            raise RuntimeError(
                f"{engine_type}: 存在確認した source={Path(source).name!r} と "
                f"ロードするモデル由来の名前 {expected_source!r} が一致しない"
            )
        engine.load_model()
        ctx.stage("load_model")

        sample_rate, audio = load_probe_speech(case.audio_stem)
        try:
            with _WavRecorder() as recorder:
                result = engine.transcribe(audio, sample_rate)
            ctx.stage("consumer_returned")
        finally:
            # モデルは重い。**次の variant / 次の engine のために必ず解放する。**
            # 後始末の失敗で観測そのものを失わない。
            with contextlib.suppress(Exception):
                engine.cleanup()

        # --- 経路の証明 ------------------------------------------------------
        # 一時 wav が **variant root 配下**に書かれたことを確かめる。ここが崩れると
        # 「非 ASCII を通していないのに pass」になる。control 側でこれが落ちれば
        # verdict は error_harness (= ハーネスのバグ) になり、証拠として数えられない。
        under_root = [p for p in recorder.paths if str(ctx.root).lower() in p.lower()]
        if not under_root:
            raise RuntimeError(
                "一時 wav が variant root 配下に書かれなかった。測定対象経路を"
                f"通っていない: root={ascii(str(ctx.root))} paths={[ascii(p) for p in recorder.paths]}"
            )
        # control は ASCII、trial は非 ASCII のはず。ここが揃わないなら
        # **variant が効いていない** (= cjk_kana でも実質 control を 2 回測っている)。
        expected_ascii = str(ctx.root).isascii()
        if any(p.isascii() != expected_ascii for p in under_root):
            raise RuntimeError(
                f"一時 wav の ASCII 性が variant と一致しない (expected_ascii="
                f"{expected_ascii}): {[ascii(p) for p in under_root]}"
            )
        ctx.stage("wrote_under_variant_root")

        text = (result.text or "").strip()

        # **ASCII control が空なら測定不能。** 両方空を pass と数えないための前提。
        # control の例外は error_harness になる (境界のバグではなく probe のバグ)。
        # trial が空の場合は raise せず、observation の差として fail_silent にする。
        if ctx.is_control and not text:
            raise RuntimeError(
                "ASCII control が空の転写を返した - probe は境界を検証できていない"
            )

        # **observation に path を入れない。** control と trial で必ず違うので、
        # 入れると常に fail_silent になる。**token_count も使わない** — engine ごとに
        # 意味 (decoder token / 語数 / subword) が変わり比較できない。
        return {
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_is_nonempty": bool(text),
            "text_char_count": len(text),
        }

    impl.__name__ = f"utterance_wav_{engine_type}"
    impl.__doc__ = (
        f"{engine_type} の発話 wav を非 ASCII な置き場所から consumer へ渡す (#413)。"
    )
    return impl


for _engine in _ENGINES:
    probe(f"asr.utterance_wav.{_engine}")(_make_probe(_engine))
