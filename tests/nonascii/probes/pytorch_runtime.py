"""PyTorch CUDA Jiterator の kernel cache 境界 (Issue #422)。

**測るのは production の結末である** — 「非 ASCII な `%TEMP%` の下で、CUDA の
Jiterator 経路に入る演算が control (ASCII) と同じ結果を返すか」。

境界は kernel cache の置き場所であり、既定では ``%TEMP%\\torch\\kernels`` になる。
worker が ``TEMP`` / ``TMP`` / ``TMPDIR`` を variant root へ向けているので、**この
probe が変えるのは TEMP だけ**である (他の root は `ascii_pinned_roots` で ASCII へ
固定する — #413 で複数境界を同時に非 ASCII にして誤帰属しかけた形を繰り返さない)。

**初期化を直接呼ばない。** ``configure_pytorch_runtime()`` をここで呼ぶと、production
の配線 (``BaseEngine.__init__`` 等) が外れても probe だけが緑になる。engine を
**構築するだけ** (``load_model()`` は呼ばない = モデル不要) で同じ経路を通るので、
そちらを使う。

判定はハーネスが行う。``runner.derive_verdict`` が control と trial の observation を
比較し、一致すれば ``pass``、差があれば ``fail_silent`` とする。修正前は trial 側が
``UnicodeDecodeError`` で落ち、**その例外はパスを一切名指ししない**ので ``fail_silent``
になる。
"""

from __future__ import annotations

import hashlib
import os

from ..record import ProbeContext, ProbeSkipped
from . import probe

#: 実測で最初に踏んだ形 (whisper_s2t の前処理が通る)。
#: **Jiterator 経路に入る演算を選ぶ必要がある** — 素の行列積などでは踏まない。
_SIGNAL_LEN = 16381


@probe("framework.pytorch.jiterator_cache")
def jiterator_kernel_cache(ctx: ProbeContext) -> dict:
    """非 ASCII な kernel cache 置き場所で CUDA 複素数演算が通るか (#422)。"""
    try:
        import torch
    except ImportError as exc:
        raise ProbeSkipped(f"torch 未導入: {exc}") from exc

    if not torch.cuda.is_available():
        # **CPU では踏まない** (実測)。CUDA が無い環境では測れない。
        raise ProbeSkipped("CUDA が使えない (Jiterator 経路に入らない)")

    # 前提の明示。ここが崩れていると「TEMP だけを変数にする」が成立しない。
    temp = os.environ.get("TEMP", "")
    if str(temp).isascii() != str(ctx.root).isascii():
        raise ProbeSkipped(
            f"TEMP の ASCII 性が variant と一致しない: {ascii(temp)}"
        )
    ctx.stage("temp_matches_variant")

    # **production の配線を通す。** engine を構築するだけでモデルは要らない
    # (load_model() は呼ばない)。ここで BaseEngine.__init__ ->
    # configure_pytorch_runtime() が走る。
    from livecap_cli.engines import EngineFactory
    from livecap_cli.runtime import current_pytorch_runtime

    if current_pytorch_runtime() is not None:
        raise RuntimeError(
            "engine を作る前に PyTorch ランタイムが設定されている - この probe は "
            "production の配線が実際に走ることを確かめられない (Issue #422)"
        )

    EngineFactory.create_engine(
        "whispers2t", device="auto", model_size="base", language="en"
    )
    ctx.stage("engine_constructed")

    decision = current_pytorch_runtime()
    if decision is None:
        raise RuntimeError(
            "engine を構築したのに PyTorch ランタイムが設定されていない - "
            "production の配線が外れている (Issue #422)"
        )
    ctx.stage("runtime_configured")

    generator = torch.Generator(device="cuda").manual_seed(422)
    signal = torch.randn(_SIGNAL_LEN, device="cuda", generator=generator)
    magnitude = torch.fft.rfft(signal).abs()
    torch.cuda.synchronize()
    ctx.stage("jiterator_ran")

    # **observation に path を入れない** (control と trial で必ず違うため常に
    # fail_silent になる)。数値そのものを fingerprint にする — 破綻すればここへ
    # 到達しないので、比較対象は「同じ値が出たこと」でよい。
    digest = hashlib.sha256(
        magnitude.detach().cpu().numpy().tobytes()
    ).hexdigest()
    return {
        "jiterator_ok": True,
        "magnitude_sha256": digest,
        "kernel_cache": decision.kernel_cache,
    }
