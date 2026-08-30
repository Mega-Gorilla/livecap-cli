"""**mitigated track** — 非 ASCII `%TEMP%` でも WhisperS2T が転写できること (Issue #422)。

`test_pytorch_kernel_cache.py` は **raw track** である — `torch` を直接呼んで
「上流は ACP の外側で壊れる」という基準データを取る。こちらは **production 経路**
(`EngineFactory` → `load_model()` → `transcribe()`) を非 ASCII `%TEMP%` の下で走らせて
**成功する**ことを見る。両方あって初めて「欠陥は実在し、我々の経路では起きない」と
言える (#379 で確立した 2 トラック構成)。

**WhisperS2T を選ぶのは、これが最初に障害が確認された consumer だからである。**
前処理が `torch.fft.rfft(...).abs()` を通るため Jiterator 経路に入る。実際 #413 の
実測では parakeet / canary / voxtral は非 ASCII `%TEMP%` でも通っており、torch を
使えば必ず踏むわけではない。

**variant は `outside_acp` でなければならない。** `ユーザー` は cp932 の内側なので、
日本語 Windows では**壊れないまま通ってしまう** — cjk_kana で書くとこのテストは
修正の有無に関わらず緑になる。

`tests/nonascii` の `engine.whispers2t.utterance_wav` 行は `%TEMP%` を ASCII へ固定
したままにする。あれが測るのは `cache_root` に置かれる**発話 wav** であって
`%TEMP%` ではなく、両方を非 ASCII にすると失敗の帰属ができなくなるためである
(#413 で実際に誤帰属しかけた)。**本 module がその穴を埋める。**
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

#: ACP (cp932 / cp1252) の**外側**。cjk_kana では再現しない。
OUTSIDE_ACP = "한국어Ω"

_SENTINEL = "---LIVECAP-422-JSON---"

_CHILD = textwrap.dedent(
    """
    import json, sys, tempfile, wave
    import numpy as np

    audio_path = sys.argv[1]
    out = {"tempdir": tempfile.gettempdir()}
    try:
        with wave.open(audio_path, "rb") as handle:
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        from livecap_cli.engines import EngineFactory
        from livecap_cli.runtime import current_pytorch_runtime

        engine = EngineFactory.create_engine(
            "whispers2t", device="auto", model_size="base", language="en"
        )
        decision = current_pytorch_runtime()
        out["kernel_cache"] = None if decision is None else decision.kernel_cache
        out["source"] = None if decision is None else decision.source
        engine.load_model()
        result = engine.transcribe(audio, sample_rate)
        text = (result.text or "").strip()
        out["ok"] = bool(text)
        out["char_count"] = len(text)
    except BaseException as exc:
        out["ok"] = False
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:400]

    sys.stdout.write("\\n{s}\\n".format(s="---LIVECAP-422-JSON---"))
    sys.stdout.write(json.dumps(out, ensure_ascii=True, default=str))
    sys.stdout.write("\\n{s}\\n".format(s="---LIVECAP-422-JSON---"))
    """
)

_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "audio"
    / "en"
    / "librispeech_1089-134686-0001.wav"
)


def _run_child(temp_root: Path) -> dict:
    env = dict(os.environ)
    env["TEMP"] = env["TMP"] = env["TMPDIR"] = str(temp_root)
    env["PYTHONIOENCODING"] = "utf-8"
    # **親の cache 設定を継承しない。** 継承すると decision の source が
    # explicit_* になり、**「明示が無ければ既定で無効化される」という production
    # default を検証できない** — 親に設定が残っているだけで緑になってしまう。
    env.pop("USE_PYTORCH_KERNEL_CACHE", None)
    env.pop("PYTORCH_KERNEL_CACHE_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_AUDIO)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(Path(__file__).resolve().parents[3]),
        timeout=1800,
    )
    chunks = proc.stdout.split(_SENTINEL)
    assert len(chunks) >= 3, (
        "子プロセスが結果 JSON を出さなかった:\n"
        f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
    )
    return json.loads(chunks[1])


def test_transcribe_succeeds_with_non_ascii_temp(tmp_path: Path) -> None:
    """非 ASCII `%TEMP%` の下で production 経路の転写が成功する。"""
    if sys.platform != "win32":
        pytest.skip("narrow path 変換は Windows 固有")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA が使えない (Jiterator 経路に入らない)")
    if not _AUDIO.is_file():  # pragma: no cover - 資産は repo に入っている
        pytest.skip(f"音声資産が無い: {_AUDIO}")

    from livecap_cli.resources import get_model_manager

    model_dir = get_model_manager().get_models_dir() / "whispers2t_base"
    if not model_dir.is_dir():
        pytest.skip("whispers2t_base が未取得 (先にモデルを取得すること)")

    temp_root = tmp_path / OUTSIDE_ACP / "temp"
    try:
        temp_root.mkdir(parents=True)
    except (OSError, UnicodeError) as exc:  # pragma: no cover - FS が variant を拒否
        pytest.skip(f"ACP 外の TEMP を作れない: {exc}")

    result = _run_child(temp_root)

    # **前提の確認。** 注入が効いていなければこのテストは何も検証していない。
    assert not str(result["tempdir"]).isascii(), (
        f"子プロセスの %TEMP% が ASCII のまま: {result['tempdir']!r}"
    )
    assert result.get("kernel_cache") == "disabled", (
        "共有初期化が走っていない、または既定が無効化になっていない: "
        f"{result.get('kernel_cache')!r}。この経路が守られていることを"
        "確かめられていない (Issue #422)"
    )
    assert result.get("source") == "default", (
        f"decision の source が {result.get('source')!r} — **明示設定由来**である。"
        "子の環境から 2 変数を消しているはずなので、消し漏れているか、"
        "既定の分岐を通っていない。production default を検証できていない。"
    )
    assert result["ok"], (
        "ACP 外の %TEMP% で WhisperS2T の転写が失敗した (#422 の再発): "
        f"{result.get('error_type')}: {result.get('error')}"
    )
