"""VADProcessor を file 文字起こしの segmenter として使う adapter (Issue #366 Phase 1)。

`FileTranscriptionPipeline(segmenter=...)` の `Segmenter` 契約
(`Callable[[np.ndarray, int], list[tuple[float, float]]]`) に、streaming 用の
`VADProcessor` を適合させる。CLI file mode の `--vad` 接続と、GUI 等の
offline 一括処理の両方から利用できる。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .processor import VADProcessor


class VADFileSegmenter:
    """`VADProcessor` を offline 音声全体に適用して発話区間 (秒) を返す adapter。

    lifecycle 契約 (Issue #366 Phase 1):

    - 呼び出しの最初に必ず ``reset()`` — 前ファイル・例外後の状態を
      持ち越さない (`process_files()` の複数ファイル処理で安全)
    - `is_final=False` の interim segment は除外する
    - EOF で ``finalize()`` を呼び、末尾の発話を取りこぼさない
    - 音声が検出されなければ空 list を返す (pipeline 側で
      「セグメントなし = ASR 呼び出しゼロ」として扱われる)

    時刻は `VADProcessor` が ``reset()`` 起点の絶対秒で返すため
    (speech_pad 込み・0 clamp)、そのままファイル内タイムスタンプになる。
    """

    _CHUNK_SECONDS = 1.0
    # streaming 契約 (`process_chunk` は任意長受理・内部 buffering) に揃えた
    # 供給単位。長尺ファイルでも一括供給せず一定量ずつ処理する。

    def __init__(self, vad_processor: VADProcessor) -> None:
        self._vad = vad_processor

    def __call__(
        self, audio: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float]]:
        self._vad.reset()  # ファイル開始時 reset (状態非持越の要)

        chunk_samples = max(1, int(sample_rate * self._CHUNK_SECONDS))
        segments: List[Tuple[float, float]] = []
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset:offset + chunk_samples]
            for seg in self._vad.process_chunk(chunk, sample_rate):
                if seg.is_final:
                    segments.append((seg.start_time, seg.end_time))

        tail = self._vad.finalize()
        if tail is not None and tail.is_final:
            segments.append((tail.start_time, tail.end_time))

        return segments


__all__ = ["VADFileSegmenter"]
