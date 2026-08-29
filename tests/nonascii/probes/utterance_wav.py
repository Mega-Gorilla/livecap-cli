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
from pathlib import Path

from ..artifacts import load_probe_speech
from ..record import ProbeContext, ProbeSkipped
from . import probe

#: engine ごとの (probe 用音声, engine kwargs)。**言語はモデルに合わせる** —
#: 合わない言語だと転写が空/別言語になり、control の非空要求で落ちる。
_ENGINES: dict[str, tuple[str, dict]] = {
    "parakeet": ("en/librispeech_1089-134686-0001", {}),
    "canary": ("en/librispeech_1089-134686-0001", {"language": "en"}),
    "whispers2t": ("en/librispeech_1089-134686-0001", {"language": "en"}),
    # Issue #418: auto は processor 境界で [None] に整形されるが、probe では
    # 言語を固定して観測値を安定させる。
    "voxtral": ("en/librispeech_1089-134686-0001", {"language": "en"}),
}


def _pin_models_root_to_ascii(models_root: str) -> None:
    """models root だけ ASCII の実体へ戻す。**cache / TEMP は variant root のまま。**

    worker は `LIVECAP_CORE_MODELS_DIR` も variant root へ向けるが、そこに実モデルは
    無い。ここで戻さないとダウンロードを試みてしまい、**測りたいのはモデルの path
    ではない**のに測定が壊れる。singleton は構築時に env を読むので reset も要る。
    """
    from livecap_cli.resources import _reset_resources_for_tests

    os.environ["LIVECAP_CORE_MODELS_DIR"] = models_root
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
    audio_stem, engine_kwargs = _ENGINES[engine_type]

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

        from livecap_cli.engines import EngineFactory

        # device は auto。実モデル tier は GPU runner でしか有効化されないが、
        # cuda 決め打ちにすると CPU 環境で probe のバグとして落ちる。
        engine = EngineFactory.create_engine(
            engine_type, device="auto", **engine_kwargs
        )
        engine.load_model()
        ctx.stage("load_model")

        sample_rate, audio = load_probe_speech(audio_stem)
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
