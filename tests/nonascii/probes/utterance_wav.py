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
    とは別のモデルがロードされ得る** — 実際 whispers2t は `whispers2t_base` の存在を
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
        # 存在確認した whispers2t_base とは別のモデルになる。
        "en/librispeech_1089-134686-0001", {"language": "en", "model_size": "base"},
        "model_size", "base",
        lambda v: f"whispers2t_{v}",
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
    # source は models root の marker (38 バイト) である。**重みは marker の隣に無く**
    # 管理 HF cache (`<cache_root>/huggingface/hub`) にある — `qwen3asr_snapshot_dir()`
    # がそちらまで確かめる。
    "qwen3asr": _Case(
        "en/librispeech_1089-134686-0001", {},
        "model_name", "Qwen/Qwen3-ASR-0.6B",
        lambda v: f"{v.replace('/', '--')}.marker",
    ),
}


#: 管理 HF cache 内での Qwen3-ASR snapshot の位置。
_QWEN3ASR_REPO_DIR = "models--Qwen--Qwen3-ASR-0.6B"


def qwen3asr_snapshot_dir(cache_root) -> "Path | None":
    """管理 HF cache 内の Qwen3-ASR snapshot。無ければ ``None``。

    **marker の存在だけでは足りない。** models root に置かれているのは
    ``model=Qwen/Qwen3-ASR-0.6B`` と書かれただけの 38 バイトのテキストで、重みは
    ``manager.huggingface_cache()`` が指す ``<cache_root>/huggingface`` にある。
    marker だけを見て「使える」と答えると、**real_model tier の「ネットワークを
    使わない」契約を破ってダウンロードが走る**。

    判定をここに置くのは ``sherpa.from_transducer.real`` と同じ理由である —
    ``test_probes.py`` 側にファイル名を書くと二重管理になる。
    """
    hub = Path(cache_root) / "huggingface" / "hub" / _QWEN3ASR_REPO_DIR
    snapshots = hub / "snapshots"
    if not snapshots.is_dir():
        return None
    return next((p for p in sorted(snapshots.iterdir()) if p.is_dir()), None)


def _pin_models_root_to_ascii(models_root: str) -> None:
    """models root だけ ASCII の実体へ戻す。**cache / TEMP は variant root のまま。**

    worker は `LIVECAP_CORE_MODELS_DIR` も variant root へ向けるが、そこに実モデルは
    無い。ここで戻さないとダウンロードを試みてしまい、**測りたいのはモデルの path
    ではない**のに測定が壊れる。singleton は構築時に env を読むので reset も要る。
    """
    from livecap_cli.resources import _reset_resources_for_tests

    os.environ["LIVECAP_CORE_MODELS_DIR"] = models_root
    _reset_resources_for_tests()


def _pin_cache_root_to_ascii(cache_root: str) -> None:
    """cache root だけ ASCII の実体へ戻す。**TEMP は variant root のまま。**

    qwen3asr の重みは models root ではなく **管理 HF cache**
    (``<cache_root>/huggingface``) にある (`manager.huggingface_cache()`)。worker は
    ``LIVECAP_CORE_CACHE_DIR`` も variant root へ向けるので、戻さないと空の cache を
    見てダウンロードへ落ち、**real_model tier の「ネットワークを使わない」契約を破る**。

    **測定対象は変わらない** — qwen3asr の一時 wav は `dir=` 指定なしの
    ``tempfile.NamedTemporaryFile`` で**素の %TEMP%** に書かれるので、変数は TEMP の
    ままである。
    """
    from livecap_cli.resources import _reset_resources_for_tests

    os.environ["LIVECAP_CORE_CACHE_DIR"] = cache_root
    _reset_resources_for_tests()


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

        _pin_models_root_to_ascii(str(models_root))
        ctx.stage("pin_models_root")

        # qwen3asr だけは重みが管理 HF cache 側にあるので cache root も実体へ戻す。
        cache_root = ctx.payload.get("cache_root")
        if engine_type == "qwen3asr":
            if not cache_root or not str(cache_root).isascii():
                raise ProbeSkipped(
                    f"cache root が未指定 / 非 ASCII: {ascii(str(cache_root))}"
                )
            if qwen3asr_snapshot_dir(cache_root) is None:
                raise ProbeSkipped(
                    "管理 HF cache に Qwen3-ASR の snapshot が無い "
                    "(marker だけでは重みの存在を保証しない)"
                )
            _pin_cache_root_to_ascii(str(cache_root))
            ctx.stage("pin_cache_root")

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
