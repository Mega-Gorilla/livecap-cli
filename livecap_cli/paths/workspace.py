"""「我々が作るファイル」用の ASCII 保証された作業ディレクトリ (Issue #375 PR 2)。

契約は #378 §6.10。

:func:`ascii_safe_temp_environment` との非対称が要点
--------------------------------------------------
====================================  ==============  ==========================
                                      env を変える    退出時に自分の dir を消す
====================================  ==============  ==========================
``ascii_safe_temp_environment()``     **する**        **しない**
``ascii_safe_workspace()``            **しない**      **する**
====================================  ==============  ==========================

理由は同じ 1 つの事実から出る: **プロセス全体の TEMP を向けている間は、無関係な
スレッドの ``NamedTemporaryFile()`` もそこへ落ちる。** 向けていなければ、その
ディレクトリには**自分が置いたファイルしか無い**ので、消して安全である。

したがって発話ごとの一時 wav の正解はこちらであって、
``ascii_safe_temp_environment()`` ではない — 発話ごとにプロセスグローバル状態を
書き換えるのは現行バグの縮小再生産になる。**最初から ASCII 空間に ASCII 名で作る。**

env を触らないので**自明にスレッド安全・ネスト可**である (ロックも深度カウンタも要らない)。
"""
from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .lease import hold_lease
from .roots import select_staging_root, validate_purpose

logger = logging.getLogger(__name__)

__all__ = ["ascii_safe_workspace"]


@contextmanager
def ascii_safe_workspace(*, boundary: str, purpose: str = "runtime") -> Iterator[Path]:
    """ASCII 保証された**空の**ディレクトリを作り、退出時に消す。

    Args:
        boundary: どのネイティブ境界のためか。失敗メッセージに必ず出すので必須。
        purpose: staging root 配下のサブディレクトリ名。**ASCII slug**
            (``[A-Za-z0-9_-]``、1..16 文字) でなければならない。

    Yields:
        ASCII のみで構成された、呼び出し固有の空ディレクトリ。

    Raises:
        AsciiStagingUnavailableError: ASCII 保証された root を用意できないとき。
        ValueError: ``purpose`` が slug 契約を満たさないとき。

    Note:
        **ここへ置くファイル名も ASCII にすること。** ディレクトリが ASCII でも、
        非 ASCII な葉の名前を付ければ完成した path は非 ASCII になり、目的を失う。

    Example:
        >>> with ascii_safe_workspace(boundary="parakeet.utterance_wav") as work:
        ...     wav = work / "utterance.wav"
        ...     soundfile.write(str(wav), audio, sample_rate)
        ...     model.transcribe([str(wav)])
    """
    validate_purpose(purpose, boundary=boundary)
    root = select_staging_root(boundary=boundary)
    # 12 hex で衝突は事実上起こらない。短くしているのは MAX_PATH の余裕を残すため。
    # 万一衝突したら exist_ok=False で **黙って共有せず落ちる** — 共有すると
    # 「自分のファイルしか無い」という前提が崩れ、退出時の削除が他人のファイルを
    # 巻き込む。
    # **reaper の単位 (entry) と、消費側に見せるディレクトリを分ける。**
    # lease は entry の中に置く必要がある (Windows では開いたハンドルが rmtree を
    # 阻むのが保護の実体) が、消費側には「空のディレクトリ」を渡す契約がある。
    # entry の子を渡せば両立する。
    entry = root / purpose / uuid.uuid4().hex[:12]
    workspace = entry / "w"
    workspace.mkdir(parents=True)
    try:
        # workspace は自分で消すので通常 reaper の出番は無いが、**ハードクラッシュ
        # で残った場合**と、**他プロセスの reaper が使用中に来た場合**に効く。
        with hold_lease(entry):
            yield workspace
    finally:
        # **例外時も消す。** 自分のファイルしか無いので巻き込みは起きない。
        # 失敗しても送出しない — 後始末で本筋の例外を覆い隠さないため。
        # entry ごと消す (lease ファイルも一緒に片付く)。
        shutil.rmtree(entry, ignore_errors=True)
        if entry.exists():  # pragma: no cover - 消せないのは掴まれているとき
            logger.debug(
                "Workspace could not be removed (still in use?): %s", ascii(str(entry))
            )
