"""``whisper_s2t`` の import (#474)。

``whisper_s2t`` (1.3.1) は **import 時に** ``os.system('ffmpeg -version')`` で PATH 上の ffmpeg を
確かめ、見つかれば ``silent.mp3`` の変換で resampler (soxr / swr) を決める。見つからなければ
``Seems 'ffmpeg' is not installed`` を stdout に出して進む。ffmpeg を PATH 以外で用意している
ホスト (``LIVECAP_FFMPEG_BIN`` / managed cache / bundled) では、この確認が必ず失敗していた。

確認は**最初の import の 1 回だけ**なので、``whisper_s2t`` を import する箇所 (engine のロードと
preloader の別スレッド) は必ず :func:`import_whisper_s2t` を通す。
"""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

_MODULE = "whisper_s2t"


def expose_ffmpeg_for_whisper_s2t() -> None:
    """解決済みの ffmpeg を PATH に反映する (best-effort、ダウンロードはしない)。

    realtime の WhisperS2T は 16 kHz の PCM WAV を渡すので ffmpeg 無しでも動く。ここで失敗しても
    engine のロードは止めず、警告だけ出す。
    """
    try:
        from livecap_cli.resources import get_ffmpeg_manager

        executable = get_ffmpeg_manager().expose_on_path()
    except Exception as exc:  # noqa: BLE001 - 前処理の失敗で engine のロードを止めない
        logger.warning(
            "FFmpeg を PATH に反映できなかった: %s — whisper_s2t の import 時の確認は失敗する"
            " (16 kHz の PCM WAV を渡す転写には影響しない)",
            exc,
        )
        return
    if executable is None:
        logger.warning(
            "FFmpeg が見つからない — whisper_s2t の import 時の確認は失敗する"
            " (16 kHz の PCM WAV を渡す転写には影響しない)"
        )
    else:
        logger.debug("whisper_s2t の import の前に FFmpeg を反映: %s", executable)


def import_whisper_s2t() -> ModuleType:
    """ffmpeg を PATH に反映してから ``whisper_s2t`` を import する。

    import 済みなら確認は終わっているので、そのまま返す。``ImportError`` はそのまま上げる。
    """
    module = sys.modules.get(_MODULE)
    if module is not None:
        return module
    expose_ffmpeg_for_whisper_s2t()
    return importlib.import_module(_MODULE)
