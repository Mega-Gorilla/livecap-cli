"""%TEMP% と temp ヘルパの境界プローブ (Issue #378)。

注入された ``TEMP`` / ``TMP`` / ``TMPDIR`` が非 ASCII のとき、
発話ごとの一時 wav 経路 (parakeet / canary / qwen3asr / whispers2t / voxtral)
がどう振る舞うかを測る。

なお ``%TEMP%`` を移設する API (``livecap_cli.paths.ascii_safe_temp_environment``、
かつての ``unicode_safe_*`` ヘルパ) は**この worker 子プロセスの中でのみ**呼ぶ。
親プロセスで呼んではならない (プロセス全体の ``os.environ`` と
``tempfile.tempdir`` を書き換えるため)。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..record import ProbeContext, ProbeSkipped
from . import probe

_SR = 16000


@probe("tempfile.named_temporary_wav")
def tempfile_named_temporary_wav(ctx: ProbeContext) -> dict:
    """``NamedTemporaryFile(suffix='.wav', delete=False)`` + ``sf.write`` の往復。

    実コード: ``parakeet_engine`` / ``canary_engine`` / ``qwen3asr_engine`` は
    ``dir=`` を指定しないため素の ``%TEMP%`` に落ちる。ここでは worker が
    ``TEMP`` を variant root 配下へ向けているので、その ``%TEMP%`` が非 ASCII の
    ときに何が起きるかを測る。

    **測れるのは producer 側 (soundfile 書き込み) までである。** consumer
    (``model.transcribe([tmp])`` = ネイティブ ASR) は real_model tier の別行。
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise ProbeSkipped(f"soundfile/numpy 未導入: {exc}") from exc

    # 注入された %TEMP% が実際に使われていることを確認する
    resolved_temp = Path(tempfile.gettempdir())
    ctx.stage("resolve_temp")

    audio = np.sin(
        2 * np.pi * 440.0 * np.arange(int(_SR * 0.2), dtype=np.float64) / _SR
    ).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp_name = fh.name
    sf.write(tmp_name, audio, _SR)
    ctx.stage("write_temp_wav")

    back, sr = sf.read(tmp_name, dtype="float32")
    ctx.stage("read_temp_wav")

    size = Path(tmp_name).stat().st_size
    Path(tmp_name).unlink(missing_ok=True)

    return {
        # パスそのものは含めない。「注入した TEMP 配下に作られたか」だけを見る。
        "temp_under_injected_root": str(resolved_temp).startswith(str(ctx.root)),
        "sample_rate": int(sr),
        "n_samples": int(back.size),
        "bytes": size,
    }
