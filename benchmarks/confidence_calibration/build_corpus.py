"""Active calibration corpus builder (Issue #338 PR-β、Stage 2 helper)。

YouTube URL or local audio file から:

1. yt-dlp で audio download (URL のみ、local file は skip)
2. ffmpeg で 0:06 trim + 16 kHz mono wav 変換
3. Silero VAD で speech segment 切り出し (``_vad_chunker.chunk_audio_by_vad()``)
4. 各 segment で ASR engine.transcribe() → text 取得
5. 原稿 text と ``difflib.SequenceMatcher`` で fuzzy match → alignment score
6. ``manifest.jsonl`` に entry 追加 (idempotent + resumable、``--force`` で再生成)

CLI usage:

    python -m benchmarks.confidence_calibration.build_corpus \\
        --source "https://www.youtube.com/watch?v=6aJ3jsVeQIg" \\
        --reference-text "https://taltal3014.lsv.jp/little-prince/LittlePrince1.html" \\
        --output-dir "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_clean" \\
        --language ja --label speech

Design (Plan D1, D2, D3, D6):

- VAD chunking は ``_vad_chunker`` (Silero VAD probability + threshold + hysteresis)
- forced alignment 非採用、Whisper transcribe text + difflib fuzzy match で
  alignment score 算出 (低 score segment は warn、drop しない)
- Audio resample は PR-α ``pipeline._resample_to_16k_mono`` を import 再利用
- idempotent: 既に audio download / segment wav が存在すれば skip
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np

from ._vad_chunker import chunk_audio_by_vad
from .pipeline import _resample_to_16k_mono

logger = logging.getLogger(__name__)

# Default VAD parameters (Plan D1)
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_SEC = 0.5
DEFAULT_MAX_SEGMENT_SEC = 3.0
DEFAULT_MIN_SILENCE_SEC = 0.3
DEFAULT_ALIGNMENT_THRESHOLD = 0.5  # 低 score warn しきい値

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class BuildResult:
    """build_corpus の実行結果 summary。"""

    segments_created: int
    segments_skipped: int  # 既存 (resumable)
    low_alignment_warnings: int  # alignment_score < threshold の segment 数
    total_duration_sec: float
    manifest_path: Path


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source))


def download_audio(
    source: str,
    output_path: Path,
    *,
    force: bool = False,
) -> Path:
    """yt-dlp で audio download、idempotent。

    Args:
        source: YouTube URL or local file path。
        output_path: download 先 wav (extension は yt-dlp が決定、wav に統一)。
        force: True なら既存 file 上書き。

    Returns:
        download した wav の Path (extension は ``.wav``)。
    """
    if not is_url(source):
        # Local file: copy or symlink
        src = Path(source).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Local source not found: {src}")
        if output_path.exists() and not force:
            logger.info("Skipping copy (exists, use --force to override): %s", output_path)
            return output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, output_path)
        return output_path

    # URL: yt-dlp で download
    if output_path.exists() and not force:
        logger.info(
            "Skipping download (exists, use --force to override): %s", output_path
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--output",
        str(output_path.with_suffix(".%(ext)s")),
        source,
    ]
    logger.info("Downloading via yt-dlp: %s", source)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return output_path


def ffmpeg_trim_and_resample(
    src: Path,
    dst: Path,
    *,
    start_offset_sec: float = 0.0,
    max_duration_sec: Optional[float] = None,
    force: bool = False,
) -> Path:
    """ffmpeg で 0:06 trim + 16 kHz mono wav 変換、idempotent。

    Returns:
        変換後 wav の Path。
    """
    if dst.exists() and not force:
        logger.info(
            "Skipping ffmpeg (exists, use --force to override): %s", dst
        )
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg_path()]
    if start_offset_sec > 0:
        cmd.extend(["-ss", f"{start_offset_sec:.3f}"])
    cmd.extend(["-i", str(src)])
    if max_duration_sec is not None:
        cmd.extend(["-t", f"{max_duration_sec:.3f}"])
    cmd.extend(
        [
            "-ac",
            "1",  # mono
            "-ar",
            str(SAMPLE_RATE),  # 16 kHz
            "-y",  # overwrite
            str(dst),
        ]
    )
    logger.info("ffmpeg: trim=%s, max_duration=%s → %s", start_offset_sec, max_duration_sec, dst)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\nSTDERR: {result.stderr[:1000]}"
        )
    return dst


def _ffmpeg_path() -> str:
    """ffmpeg path を resolve (env var → ./ffmpeg-bin/ → system PATH の順)。"""
    import os

    env = os.environ.get("LIVECAP_FFMPEG_BIN")
    if env:
        p = Path(env) / "ffmpeg"
        if p.with_suffix(".exe").exists():
            return str(p.with_suffix(".exe"))
        if p.exists():
            return str(p)
    local = Path("ffmpeg-bin/ffmpeg")
    if local.with_suffix(".exe").exists():
        return str(local.with_suffix(".exe"))
    if local.exists():
        return str(local)
    return "ffmpeg"  # system PATH


def load_wav_16k_mono(path: Path) -> np.ndarray:
    """16 kHz mono float32 として load (PR-α `pipeline._resample_to_16k_mono` 再利用)。"""
    import soundfile as sf

    audio, sr = sf.read(str(path))
    return _resample_to_16k_mono(audio, sr)


def fetch_reference_text(source: str) -> str:
    """Reference text を URL or local file から取得、HTML 内本文を抽出。

    HTML 内本文の単純抽出: ``<p>`` / ``<div>`` 内 text を結合、tag を strip。
    """
    if is_url(source):
        import urllib.request

        req = urllib.request.Request(
            source,
            headers={"User-Agent": "Mozilla/5.0 (calibration corpus builder)"},
        )
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    else:
        html = Path(source).expanduser().read_text(encoding="utf-8")

    # HTML tag を strip して text 抽出 (簡素実装、複雑な page で誤抽出する可能性は受容)
    # script / style / nav 等を除去
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # tag を空白に置換
    text = re.sub(r"<[^>]+>", " ", html)
    # HTML entity の主要なものを decode
    import html as html_module

    text = html_module.unescape(text)
    # 連続空白を 1 つに
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_alignment_score(
    transcribed_text: str,
    reference_text: str,
) -> tuple[float, Optional[str]]:
    """Transcribed text が reference text 内に substring として含まれる比率 (coverage)。

    旧実装の ``SequenceMatcher.ratio()`` (= ``2M / (len(a)+len(b))``) は、
    ``reference_text`` が Chapter 1 全文 (~10000 chars) と長い場合、完全一致
    substring でも score ≈ 0.006 と極小になり「短い ASR text が原稿内に
    どれだけ含まれるか」 という本来の目的が崩れていた (PR #340 review 指摘 2)。

    本実装は ``match.size / len(transcribed)`` (= coverage) を返す。これは
    「transcribed の何 % が reference の substring として match したか」 を
    measure する適切な指標:

    - 完全一致 substring ("Once upon a time" が原稿に存在) → 1.0
    - 部分一致 (transcribed の前半だけ match) → 0.0 - 1.0 の比率
    - No match → 0.0

    Returns:
        ``(coverage, matched_span)``。
        coverage は 0.0-1.0 の比率、matched_span は reference 側の matched
        substring (debugging 用、optional)。
    """
    transcribed = transcribed_text.strip()
    if not transcribed:
        return 0.0, None
    # autojunk=False が **必須**: default True だと、reference_text が長い
    # (~数千 chars) かつ頻出 char (日本語ひらがな の/た/し/な/か 等、英語
    # の/the/and 等) が 200 件を超える場合、それらを "junk" として match
    # 候補から除外する heuristic が走り、本来 20-30 chars 連続 substring
    # match できるはずの case が partial match (4-5 chars) に縮小される
    # bug が発覚 (Phase 4 smoke verify 2026-06-29 で確認)。
    #
    # 実例 (JA Chapter 1、reference 6600 chars):
    #   transcribed: "一人でエンジンを修理しなければならなかった" (21 chars)
    #   reference contains: "1人でエンジンを修理しなければならなかった..." (連続 20 chars)
    #   autojunk=True : size=4 ("エンジン") → coverage 0.19 ❌
    #   autojunk=False: size=20 → coverage 0.95 ✅
    #
    # autojunk=False は 公式 docs 推奨の「accurate but slower」設定。
    # alignment_score 計算は build_corpus の per-segment cost で 1 回だけ
    # なので速度懸念なし。
    matcher = SequenceMatcher(None, transcribed, reference_text, autojunk=False)
    match = matcher.find_longest_match(0, len(transcribed), 0, len(reference_text))
    if match.size == 0:
        return 0.0, None
    matched_span = reference_text[match.b : match.b + match.size]
    coverage = match.size / max(1, len(transcribed))
    return coverage, matched_span


def compute_alignment_score_kana(
    transcribed_text: str,
    reference_text: str,
) -> tuple[float, Optional[str], str, str]:
    """kana-level alignment coverage (Issue #338 PR-γ).

    Same coverage semantics as ``compute_alignment_score`` (longest-match
    substring + ``autojunk=False`` regression guard), but both transcribed
    and reference are normalised to a kana-only form via
    ``benchmarks.confidence_calibration._normalize_jp.normalize_for_alignment``
    before comparison.

    Rationale: text-level coverage conflates **acoustic confidence** (was the
    phoneme sequence correct?) with **lexical surface form** (kanji /
    katakana / digit representation choice). The calibration goal is the
    former. Phase 4 smoke verify (PR-β) showed that many JA segments have
    low text coverage purely because of surface-form differences (e.g.
    ``"1人で"`` vs ``"一人で"``, ``"サハラ砂漠"`` vs all-hiragana reference).
    Normalising to kana isolates acoustic-confidence signal from
    surface-form noise.

    Note (PR #341 codex-review fix): the normalization uses per-character
    canonical substitution for kanji digits (一 → 1, 千 → 1000, etc.) rather
    than a blanket mask. This preserves value distinctions across real ASR
    mistakes (e.g. ``"一人"`` and ``"二人"`` still produce different kana)
    while still unifying ``"千マイル"`` and ``"1000マイル"`` to the same
    canonical form. See ``_normalize_jp.py`` for trade-offs.

    This function is **purely additive**: ``compute_alignment_score`` is
    unchanged; this function is computed alongside it and stored as separate
    ``*_kana`` fields in the manifest. PR-γ review consideration: kana
    metric is **dev/benchmark-only** because ``pykakasi`` (GPL-3.0-or-later)
    is a dev dependency.

    For EN text, ``normalize_for_alignment`` degrades to (NFKC + punctuation
    strip) since pykakasi passes ASCII through unchanged — so EN audio
    callers get an approximately text-equivalent metric.

    Returns:
        ``(coverage, matched_kana_span, transcribed_kana, matched_kana_span_or_empty)``
        where:

        - ``coverage`` is ``0.0`` if either side is empty after normalisation
          or no match.
        - ``matched_kana_span`` is the matched substring on the
          (normalised) reference side, or ``None`` if no match.
        - ``transcribed_kana`` is the full normalised transcribed text
          (kept for manifest debugging).
        - The fourth element duplicates ``matched_kana_span`` as ``""`` when
          ``None`` so it can be safely written to manifest JSON.
    """
    # Lazy import keeps pykakasi out of module-load path when only
    # text-level compute_alignment_score is needed.
    from ._normalize_jp import normalize_for_alignment

    transcribed_kana = normalize_for_alignment(transcribed_text)
    reference_kana = normalize_for_alignment(reference_text)
    if not transcribed_kana:
        return 0.0, None, transcribed_kana, ""
    matcher = SequenceMatcher(None, transcribed_kana, reference_kana, autojunk=False)
    match = matcher.find_longest_match(0, len(transcribed_kana), 0, len(reference_kana))
    if match.size == 0:
        return 0.0, None, transcribed_kana, ""
    matched_kana_span = reference_kana[match.b : match.b + match.size]
    coverage = match.size / max(1, len(transcribed_kana))
    return coverage, matched_kana_span, transcribed_kana, matched_kana_span


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def append_manifest(
    manifest_path: Path,
    entry: dict,
) -> None:
    """manifest.jsonl に entry を append (1 行 = 1 JSON)。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_manifest_entries(manifest_path: Path) -> dict[str, dict]:
    """既存 manifest を読み込み、``path`` → entry の map として返す。

    同一 path が複数行ある場合は **last-wins** (後 entry で上書き)。本実装は
    PR #340 review 指摘 1 (`--force` 再実行で同 path 重複 append) への対応
    で、manifest 全体を upsert pattern で書き戻すための土台。
    """
    if not manifest_path.exists():
        return {}
    entries: dict[str, dict] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = entry.get("path")
        if path:
            entries[path] = entry
    return entries


def _load_existing_paths(manifest_path: Path) -> set[str]:
    """既存 manifest から ``path`` field を抽出 (skip 判定用、legacy compatible)。"""
    return set(_load_manifest_entries(manifest_path).keys())


def _write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Entry list を改行区切で manifest に書き戻す (upsert 後の rewrite 用)。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_corpus(
    *,
    source: str,
    reference_text_source: str,
    output_dir: Path,
    language: str,
    label: str = "speech",
    start_offset_sec: float = 0.0,
    max_duration_sec: Optional[float] = None,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    min_speech_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
    min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
    alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    engine_name: str = "whispers2t",
    engine_kwargs: Optional[dict] = None,
    force: bool = False,
    raw_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> BuildResult:
    """Active calibration corpus を 1 source から build。

    Args:
        source: YouTube URL or local audio file path。
        reference_text_source: 原稿 URL or local text path (alignment 用)。
        output_dir: segment wav の出力先 (e.g. ``$DIR/ja_clean/``)。
        language: ISO 639-1 code (e.g. ``ja`` / ``en``)。
        label: ``speech`` / ``noisy_speech`` / ``non_speech``。
        start_offset_sec: source 先頭から trim する秒数 (EN は 0:06)。
        max_duration_sec: source の最大 duration (Chapter 1 抜粋用)。
        vad_threshold/min_speech_sec/max_segment_sec/min_silence_sec: VAD param。
        alignment_threshold: alignment_score < これで warn (drop しない)。
        engine_name: alignment 用 ASR engine (default whispers2t)。
        engine_kwargs: engine **kwargs (e.g. {"model_size": "base"})。
        force: True なら既存 cache 無視で再生成。
        raw_dir: intermediate wav の保存先 (default ``output_dir.parent / "_raw"``)。
        manifest_path: manifest.jsonl path (default ``output_dir.parent / "manifest.jsonl"``)。

    Returns:
        ``BuildResult`` (segment 数 / skip 数 / warning 数 / 総 duration / manifest path)。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    raw_dir = (raw_dir or output_dir.parent / "_raw").expanduser().resolve()
    manifest_path = (
        manifest_path or output_dir.parent / "manifest.jsonl"
    ).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: download
    source_hash = re.sub(r"[^a-zA-Z0-9_]", "_", source)[:80]
    download_target = raw_dir / f"{source_hash}_download.wav"
    download_audio(source, download_target, force=force)

    # Step 2: ffmpeg trim + resample
    normalized = raw_dir / f"{source_hash}_normalized_{language}.wav"
    ffmpeg_trim_and_resample(
        download_target,
        normalized,
        start_offset_sec=start_offset_sec,
        max_duration_sec=max_duration_sec,
        force=force,
    )

    # Step 3: load + VAD chunking
    audio = load_wav_16k_mono(normalized)
    logger.info("Audio loaded: %.1f sec @ 16 kHz mono", len(audio) / SAMPLE_RATE)

    segments = chunk_audio_by_vad(
        audio,
        threshold=vad_threshold,
        min_speech_sec=min_speech_sec,
        max_segment_sec=max_segment_sec,
        min_silence_sec=min_silence_sec,
    )
    logger.info("VAD detected %d speech segments", len(segments))

    # Step 4: reference text 取得
    reference_text = fetch_reference_text(reference_text_source)
    logger.info("Reference text loaded: %d chars", len(reference_text))

    # Step 5: ASR engine 準備
    from livecap_cli.engines.engine_factory import EngineFactory

    engine_kwargs = engine_kwargs or {}
    engine = EngineFactory.create_engine(engine_name, **engine_kwargs)
    engine.load_model()

    # Step 6: 各 segment を transcribe + alignment + manifest upsert
    #
    # PR #340 review 指摘 1 fix: 旧実装は ``append_manifest()`` のみで `--force`
    # 再実行時に既存 entry が残ったまま新 entry が追加され、同一 path が複数
    # 件 manifest に記録されていた (sample 分布が歪む)。本実装は manifest を
    # path → entry の dict として load し upsert、最後に全体を rewrite する
    # ことで、`--force` でも `--force=False` でも path 一意性を保証する。
    existing_entries = _load_manifest_entries(manifest_path)
    segments_created = 0
    segments_skipped = 0
    low_alignment_warnings = 0
    total_duration_sec = 0.0

    for idx, (start_sec, end_sec) in enumerate(segments):
        # File name の生成 (output_dir 相対 path)
        relative_path = f"{output_dir.name}/segment_{idx:04d}.wav"
        if relative_path in existing_entries and not force:
            segments_skipped += 1
            continue

        # Audio 切り出し
        start_sample = int(start_sec * SAMPLE_RATE)
        end_sample = int(end_sec * SAMPLE_RATE)
        segment_audio = audio[start_sample:end_sample]

        # wav 書き出し
        segment_path = output_dir / f"segment_{idx:04d}.wav"
        write_wav(segment_path, segment_audio)

        # Transcribe
        result = engine.transcribe(segment_audio, SAMPLE_RATE)
        text = result.text.strip()

        # Alignment coverage (PR #340 review 指摘 2 fix: ratio → coverage)
        score, matched_span = compute_alignment_score(text, reference_text)
        if score < alignment_threshold:
            logger.warning(
                "Low alignment coverage %.3f < %.3f for segment %s: text=%r",
                score,
                alignment_threshold,
                relative_path,
                text[:80],
            )
            low_alignment_warnings += 1

        # kana-level alignment (Issue #338 PR-γ): purely additive, isolates
        # acoustic confidence from surface-form (kanji / katakana / digit)
        # noise. text-level metric above is retained for production表記評価.
        (
            score_kana,
            matched_span_kana,
            transcribed_kana,
            _matched_dup,
        ) = compute_alignment_score_kana(text, reference_text)

        # Manifest entry (upsert: 既存があれば置換、なければ新規)
        entry = {
            "path": relative_path,
            "label": label,
            "language": language,
            "noise": "clean" if label == "speech" else None,
            "reference_text_matched": matched_span,
            "transcribed_text": text,
            "alignment_score": round(score, 4),
            # PR-γ additive kana fields
            "alignment_score_kana": round(score_kana, 4),
            "reference_text_matched_kana": matched_span_kana,
            "transcribed_text_kana": transcribed_kana,
            "engine_used": engine_name,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(end_sec - start_sec, 3),
        }
        existing_entries[relative_path] = entry
        segments_created += 1
        total_duration_sec += end_sec - start_sec

    # Step 7: manifest 全体を rewrite (upsert の最終 commit、他 source の entry
    # は保持される ─ 同一 path のみ新 entry に置換される)
    _write_manifest(manifest_path, list(existing_entries.values()))

    return BuildResult(
        segments_created=segments_created,
        segments_skipped=segments_skipped,
        low_alignment_warnings=low_alignment_warnings,
        total_duration_sec=total_duration_sec,
        manifest_path=manifest_path,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.confidence_calibration.build_corpus",
        description=(
            "Build calibration corpus from YouTube URL or local audio. "
            "Outputs segment wavs + manifest.jsonl for benchmarks.confidence_calibration.sweep."
        ),
    )
    parser.add_argument("--source", required=True, help="YouTube URL or local audio path")
    parser.add_argument(
        "--reference-text",
        required=True,
        help="Reference transcript URL or local text path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Segment wav output dir (e.g. $LIVECAP_CALIBRATION_CORPUS_DIR/ja_clean)",
    )
    parser.add_argument("--language", required=True, help="ISO 639-1 (e.g. ja, en)")
    parser.add_argument(
        "--label",
        default="speech",
        choices=["speech", "noisy_speech", "non_speech"],
    )
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD)
    parser.add_argument("--min-speech-sec", type=float, default=DEFAULT_MIN_SPEECH_SEC)
    parser.add_argument("--max-segment-sec", type=float, default=DEFAULT_MAX_SEGMENT_SEC)
    parser.add_argument("--min-silence-sec", type=float, default=DEFAULT_MIN_SILENCE_SEC)
    parser.add_argument(
        "--alignment-score-threshold",
        type=float,
        default=DEFAULT_ALIGNMENT_THRESHOLD,
    )
    parser.add_argument(
        "--engine",
        default="whispers2t",
        help=(
            "ASR engine for alignment transcribe (default whispers2t). "
            "Use --engine-kwargs to override engine defaults "
            "(e.g. WhisperS2T default model_size=large-v3 is heavy for "
            "alignment-only use; pass --engine-kwargs model_size=base for CPU)."
        ),
    )
    parser.add_argument(
        "--engine-kwargs",
        nargs="*",
        default=[],
        help=(
            "Extra engine kwargs as key=value for alignment ASR "
            "(e.g. model_size=base compute_type=int8). Recommended for CPU "
            "or quick build: --engine-kwargs model_size=base"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-build even if cached files exist"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # PR #340 review 指摘 3: alignment ASR の重い default (WhisperS2T large-v3)
    # を CLI から override 可能に。sweep._parse_engine_kwargs() を再利用。
    from .sweep import _parse_engine_kwargs

    engine_kwargs = _parse_engine_kwargs(args.engine_kwargs)
    if args.engine == "whispers2t" and not engine_kwargs:
        logger.info(
            "alignment engine=whispers2t with default kwargs. "
            "Default model_size=large-v3 may be heavy; consider "
            "--engine-kwargs model_size=base for faster build on CPU."
        )

    result = build_corpus(
        source=args.source,
        reference_text_source=args.reference_text,
        output_dir=args.output_dir,
        language=args.language,
        label=args.label,
        start_offset_sec=args.start_offset_sec,
        max_duration_sec=args.max_duration_sec,
        vad_threshold=args.vad_threshold,
        min_speech_sec=args.min_speech_sec,
        max_segment_sec=args.max_segment_sec,
        min_silence_sec=args.min_silence_sec,
        alignment_threshold=args.alignment_score_threshold,
        engine_name=args.engine,
        engine_kwargs=engine_kwargs or None,
        force=args.force,
    )

    print(f"Created: {result.segments_created} segments")
    print(f"Skipped: {result.segments_skipped} (already in manifest)")
    print(f"Low alignment warnings: {result.low_alignment_warnings}")
    print(f"Total duration: {result.total_duration_sec:.1f} sec")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
