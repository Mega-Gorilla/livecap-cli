"""``whisper_s2t`` の import (#474)。

``whisper_s2t`` (1.3.1) は **``whisper_s2t.audio`` の初期化時に** ``os.system('ffmpeg -version')`` で
PATH 上の ffmpeg を確かめ、見つかれば ``silent.mp3`` の変換で resampler (soxr / swr) を決める。
見つからなければ ``Seems 'ffmpeg' is not installed`` を stdout に出して進む。ffmpeg を PATH 以外で
用意しているホスト (``LIVECAP_FFMPEG_BIN`` / managed cache / bundled) では、この確認が必ず失敗していた。

top-level の ``import whisper_s2t`` は ``audio`` を読まない。``audio`` が初期化されるのは
``load_model()`` が backend を import するときで、確認はそのプロセスで 1 回だけ。そこまでに PATH を
整えておけばよいので、``whisper_s2t`` を import する箇所 (engine のロードと preloader の別スレッド)
は必ず :func:`import_whisper_s2t` を通し、最初の import の前に反映する (反映した PATH はプロセス内に残る)。
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

_MODULE = "whisper_s2t"

#: 反映できなかったときの影響 (警告ログの後半)
_IMPACT = "whisper_s2t の ffmpeg の確認は失敗する (16 kHz の PCM WAV を渡す転写には影響しない)"


def expose_ffmpeg_for_whisper_s2t() -> None:
    """解決済みの ffmpeg を PATH に反映する (best-effort、ダウンロードはしない)。

    realtime の WhisperS2T は 16 kHz の PCM WAV を渡すので ffmpeg 無しでも動く。ここで失敗しても
    engine のロードは止めず、警告だけ出す。
    """
    try:
        from livecap_cli.resources import get_ffmpeg_manager

        executable = get_ffmpeg_manager().expose_on_path()
    except Exception as exc:  # noqa: BLE001 - 前処理の失敗で engine のロードを止めない
        logger.warning("FFmpeg を PATH に反映できなかった: %s — %s", exc, _IMPACT)
        return
    if executable is None:
        logger.warning("FFmpeg が見つからない — %s", _IMPACT)
    else:
        logger.debug("whisper_s2t の import の前に FFmpeg を反映: %s", executable)


def import_whisper_s2t() -> ModuleType:
    """ffmpeg を PATH に反映してから ``whisper_s2t`` を import する。

    反映するのは ``whisper_s2t`` が ``sys.modules`` に無いとき (このプロセスで最初の import) だけ。
    ``livecap_cli`` の import はすべてこの helper を通る (AST テストで固定) ので、``sys.modules`` に
    あれば最初の import の前に反映は済んでいる。

    ただし ``sys.modules`` にあることは**初期化が終わったことを意味しない** — Python は module の
    コードを実行する前に登録する。preloader の別スレッドが初期化している途中で ``sys.modules`` の
    値を返すと、``load_model`` がまだ無い module を掴む。返す module は常に
    :func:`importlib.import_module` から得て、初期化中なら Python の import lock で完了を待つ。
    ``ImportError`` はそのまま上げる。
    """
    if _MODULE not in sys.modules:
        expose_ffmpeg_for_whisper_s2t()
    return importlib.import_module(_MODULE)
