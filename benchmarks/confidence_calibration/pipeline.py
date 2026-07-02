"""Corpus loader for the calibration harness (Issue #338 PR-α)。

``benchmarks/non_speech_filter/pipeline.py:load_real_corpus_items()`` を
踏襲し、calibration corpus 用に schema を拡張。Stage 2 (``sweep.py``、PR-β)
が呼び出して各 audio file を 16 kHz mono float32 で load する。

manifest.jsonl schema:

    {"path": "ja_clean/narration_001.wav", "label": "speech",
     "language": "ja", "noise": "clean", "subtype": null,
     "reference_text": "..."}
    {"path": "ja_non_speech/applause_001.wav", "label": "non_speech",
     "subtype": "applause"}

Required fields: ``path``, ``label`` (``"speech"`` / ``"non_speech"`` /
``"noisy_speech"``)
Optional fields: ``language``, ``noise``, ``subtype``, ``reference_text``、
その他 metadata。

Corpus directory layout (推奨、強制ではない):

    $LIVECAP_CALIBRATION_CORPUS_DIR/    # 未 set なら OS 標準 data dir default
      ├── manifest.jsonl
      ├── ja_clean/
      ├── ja_noisy/
      ├── ja_non_speech/
      ├── en_clean/
      └── en_non_speech/

Default corpus directory (env var 未 set 時):

- Windows: ``%LOCALAPPDATA%\\PineLab\\LiveCap\\calibration_corpus``
- Linux: ``~/.local/share/LiveCap/PineLab/calibration_corpus``
- macOS: ``~/Library/Application Support/LiveCap/calibration_corpus``

``manifest.jsonl`` 内の ``path`` は corpus directory からの relative path。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

Label = Literal["speech", "non_speech", "noisy_speech"]


@dataclass(frozen=True)
class CalibrationCorpusItem:
    """1 corpus sample。``sweep.py`` (PR-β) で transcribe() 入力として使う。"""

    path: Path
    label: Label
    audio: np.ndarray  # 16 kHz mono float32
    sample_rate: int  # = 16000 (resampled)
    metadata: dict[str, Any] = field(default_factory=dict)


def _resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 16 kHz に正規化 (``benchmarks/non_speech_filter/pipeline.py:163-208`` 踏襲)。"""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        g = gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr) // g)
    return audio.astype(np.float32, copy=False)


def load_calibration_corpus(
    directory: Path,
    manifest_name: str = "manifest.jsonl",
) -> list[CalibrationCorpusItem]:
    """``directory/manifest.jsonl`` を読んで corpus item list を返す。

    Args:
        directory: corpus root directory。``LIVECAP_CALIBRATION_CORPUS_DIR``
            env var で user 側が指定する想定。
        manifest_name: 通常 ``"manifest.jsonl"``、test fixture で override 可能。

    Returns:
        ``CalibrationCorpusItem`` の list、manifest.jsonl の順序を保持。

    Raises:
        FileNotFoundError: manifest.jsonl or audio file が存在しない場合。
        ImportError: ``soundfile`` 未 install。
        ValueError: required field 欠落 / 不正な label。
    """
    manifest_path = directory / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_name} missing in {directory}; "
            "see docs/research/calibration-corpus-sources.md for the expected schema."
        )

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError(
            "soundfile is required for calibration corpus loading"
        ) from exc

    items: list[CalibrationCorpusItem] = []
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest.jsonl line {line_no} malformed JSON: {exc}"
            ) from exc

        rel_path = entry.get("path")
        if not rel_path:
            raise ValueError(f"manifest.jsonl line {line_no} missing 'path'")
        label = entry.get("label")
        if label not in ("speech", "non_speech", "noisy_speech"):
            raise ValueError(
                f"manifest.jsonl line {line_no} has invalid label={label!r} "
                f"(expected 'speech' / 'non_speech' / 'noisy_speech')"
            )
        audio_path = directory / rel_path
        if not audio_path.exists():
            raise FileNotFoundError(
                f"manifest.jsonl line {line_no}: audio file not found: {audio_path}"
            )

        audio, sr = sf.read(str(audio_path))
        audio = _resample_to_16k_mono(audio, sr)

        metadata = {
            k: v
            for k, v in entry.items()
            if k not in ("path", "label")
        }

        items.append(
            CalibrationCorpusItem(
                path=audio_path,
                label=label,  # type: ignore[arg-type]
                audio=audio,
                sample_rate=16000,
                metadata=metadata,
            )
        )

    if not items:
        logger.warning("manifest.jsonl is empty: %s", manifest_path)

    return items


def _default_corpus_dir() -> Path:
    """OS 標準 data directory 配下の calibration corpus default path。

    ``appdirs.user_data_dir("LiveCap", "PineLab")`` を base として、
    ``calibration_corpus/`` sub directory を追加。 corpus は user が
    build した label + Layer 2/3 augmented data + reports の集合で、
    再生成に時間がかかる **persistent data** のため ``user_cache_dir``
    (OS が自動削除する可能性あり) ではなく ``user_data_dir`` を採用。

    Windows: ``%LOCALAPPDATA%\\PineLab\\LiveCap\\calibration_corpus``
    Linux: ``~/.local/share/LiveCap/PineLab/calibration_corpus``
    macOS: ``~/Library/Application Support/LiveCap/calibration_corpus``

    ``appdirs.user_data_dir()`` は Windows default で ``%LOCALAPPDATA%`` を
    返し (``roaming=True`` opt-in で ``%APPDATA%`` に切替可能)、 大量の
    corpus data (Layer 2/3 augmented ~1 GB) は roaming すべきでないため
    default ``%LOCALAPPDATA%`` のまま使用する (``ModelManager`` precedent と同じ)。

    ``appdirs`` は runtime dep (``pyproject.toml``) だが、 fallback で
    ``~/.livecap/calibration_corpus`` を返す (``ModelManager`` precedent)。
    """
    try:
        from appdirs import user_data_dir  # type: ignore[import-untyped]
        return Path(user_data_dir("LiveCap", "PineLab")) / "calibration_corpus"
    except ImportError:
        return Path.home() / ".livecap" / "calibration_corpus"


def resolve_corpus_dir(env_var: str = "LIVECAP_CALIBRATION_CORPUS_DIR") -> Path:
    """Corpus directory を解決 (env var → OS 標準 data dir default)。

    既存 ``benchmarks/non_speech_filter/conftest.py:58-67`` の
    ``LIVECAP_NON_SPEECH_CORPUS_DIR`` pattern を踏襲しつつ、 env var 未 set
    時は OS 標準 data dir (``user_data_dir("LiveCap", "PineLab")
    / "calibration_corpus"``) に fallback する。

    directory の実存確認や manifest.jsonl の存在確認はしない (呼出側の責務)。

    Returns:
        env var が set されていればその Path (expanduser + resolve)、
        未 set なら OS 標準 data dir の default path (mkdir はしない)。
    """
    import os

    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_corpus_dir()
