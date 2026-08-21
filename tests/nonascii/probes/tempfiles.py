"""%TEMP% と temp ヘルパの境界プローブ (Issue #378)。

注入された ``TEMP`` / ``TMP`` / ``TMPDIR`` が非 ASCII のとき、
発話ごとの一時 wav 経路 (parakeet / canary / qwen3asr / whispers2t / voxtral)
がどう振る舞うかを測る。

なお ``livecap_cli.utils`` の ``unicode_safe_*`` ヘルパは**この worker 子プロセス
の中でのみ**呼ぶ。親プロセスで呼んではならない (無ロックでプロセス全体の
``os.environ`` と ``tempfile.tempdir`` を書き換えるため)。
"""

from __future__ import annotations

import tempfile
import threading
import time
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


@probe("utils.temp_helper_is_not_ascii_safe")
def utils_temp_helper_is_not_ascii_safe(ctx: ProbeContext) -> dict:
    """``unicode_safe_temp_directory`` が ASCII 安全ではないことを実測する。

    このヘルパは ``%TEMP%`` を ``cache_root/runtime`` へ移すだけで、その
    ``cache_root`` は appdirs 既定ではユーザー名を含む。つまり **TEMP 移設
    ヘルパであって ASCII 安全ヘルパではない**。

    worker は ``LIVECAP_CORE_CACHE_DIR`` を variant root 配下へ向けているので、
    非 ASCII variant では「ヘルパを通した後の TEMP も非 ASCII のまま」に
    なるはずである。それが観測できれば主張が実証される。
    """
    try:
        from livecap_cli.utils import unicode_safe_temp_directory
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.utils 未 import: {exc}") from exc

    with unicode_safe_temp_directory() as redirected:
        ctx.stage("enter_helper")
        inside = Path(tempfile.gettempdir())
        result = {
            # ヘルパ通過後の TEMP が ASCII かどうか — これが本質
            "redirected_is_ascii": str(redirected).isascii(),
            "tempdir_is_ascii": str(inside).isascii(),
            "tempdir_matches_helper": inside == Path(redirected),
        }
    ctx.stage("exit_helper")
    result["restored"] = str(Path(tempfile.gettempdir())) != str(redirected)
    return result


@probe("utils.download_dir_data_loss")
def utils_download_dir_data_loss(ctx: ProbeContext) -> dict:
    """``unicode_safe_download_directory`` がデータ消失を起こさないことを実測する。

    **#386 の修正前**: ``_cleanup_directory`` が **共有** の
    ``cache_root/downloads`` を ``shutil.rmtree`` していた。スコープが開いている
    間はプロセス内のあらゆる ``NamedTemporaryFile`` がそこへ飛ばされるため、
    **別スレッドが作った一時ファイルがスコープ退出時に消えていた**
    (``victim_survived_scope_exit=False`` を実測)。

    **#386 の修正後**: スコープ退出時に再帰削除しないため victim は生き残る。
    「呼び出しごとの固有ディレクトリにすれば消してよい」は成立しない —
    TEMP はプロセス全体なので、固有ディレクトリにしても無関係なスレッドの
    ファイルはそこへ入る。したがって直したのは**削除しないこと**である。

    なお ``victim_was_redirected_into_downloads`` は **True のまま**であることに
    注意 — #386 は「置き場所がずれる」問題は直していない (消えなくなるだけ)。
    プロセス全体の TEMP 書き換えをやめるのは #375 PR 2 / PR 3。

    これは非 ASCII とは独立した欠陥なので、ASCII / 非 ASCII のどちらでも同じ結果に
    なる (= verdict は pass)。観測値そのものは
    ``test_probes.py::test_download_directory_does_not_delete_unrelated_files``
    が直接 assert する。
    """
    try:
        from livecap_cli.utils import unicode_safe_download_directory
    except ImportError as exc:
        raise ProbeSkipped(f"livecap_cli.utils 未 import: {exc}") from exc

    victim: dict = {}
    entered = threading.Event()
    may_exit = threading.Event()

    def other_thread() -> None:
        """download スコープが開いている間に一時ファイルを作る無関係なコード。"""
        entered.wait(timeout=10)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(b"utterance audio")
            victim["path"] = fh.name
        may_exit.set()

    worker = threading.Thread(target=other_thread, daemon=True)
    worker.start()

    with unicode_safe_download_directory():
        ctx.stage("enter_download_scope")
        entered.set()
        may_exit.wait(timeout=10)
    ctx.stage("exit_download_scope")
    worker.join(timeout=10)

    path = victim.get("path")
    if path is None:
        raise ProbeSkipped("別スレッドが一時ファイルを作れなかった")

    time.sleep(0.05)
    return {
        "victim_survived_scope_exit": Path(path).exists(),
        "victim_was_redirected_into_downloads": "downloads" in str(Path(path).parent),
    }
