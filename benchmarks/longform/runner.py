"""長尺の評価の書き起こし (#488)。

**CLI の ``livecap-cli transcribe <file>`` と同じ部品で書き起こす。** 引数の既定値 (VAD の preset、
EnergyGate の閾値、confidence filter) は :func:`livecap_cli.cli.build_parser` から得るので、CLI の既定が
変われば評価も追従する。違いは、モデルを 1 回だけ読み込み、複数の動画を続けて処理することだけ
(VAD の segmenter はファイルごとに状態を戻すので、使い回しても結果は変わらない)。

処理の順序は CLI と同じ: VAD で区切る (1 秒ずつ流す) → EnergyGate → 認識 → confidence filter。

出力: ``<out_dir>/<label>/<audio_id>.json`` (区間ごとの文と時刻、処理時間、drop の内訳) と
``<out_dir>/<label>/run.json`` (使った設定)。書き起こしに成功した動画は飛ばすので、途中から再開できる。
ただし再開するのは、``run.json`` の設定 (CLI の引数、VAD の設定と backend) が今回と同じときだけ。違えば
止まる (違う条件の結果を混ぜない)。失敗した動画と、読めない記録はやり直す。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from benchmarks.common.yodas2 import Yodas2Video

__all__ = ["EngineSpec", "build_transcribe_args", "load_hypotheses", "transcribe_videos"]

logger = logging.getLogger(__name__)

#: run.json に残す CLI の設定 (結果に効くもの)
_RECORDED_ARGS = (
    "engine", "model_size", "language", "device", "vad", "engine_min_rms", "engine_energy_metric",
    "engine_energy_frame_ms", "noise_gate", "transient_filter", "confidence_filter",
)


@dataclass(frozen=True)
class EngineSpec:
    """測る ASR。``model_size`` は ``whispers2t`` だけが使う (CLI の ``--model-size``)。

    ``max_speech_ms`` は VAD の区間の長さの上限 (実験用。CLI には無い)。``None`` なら CLI と同じ
    (preset の値。日本語の preset は 0 = 上限なし)。
    """

    engine_id: str
    model_size: str | None = None
    max_speech_ms: int | None = None

    @property
    def label(self) -> str:
        base = f"{self.engine_id}-{self.model_size}" if self.model_size else self.engine_id
        return base if self.max_speech_ms is None else f"{base}-max{self.max_speech_ms}ms"

    @classmethod
    def parse(cls, text: str, *, max_speech_ms: int | None = None) -> EngineSpec:
        """``"qwen3asr_large"`` や ``"whispers2t:large-v3"`` を読む。"""
        engine_id, _, size = text.partition(":")
        if not engine_id:
            raise ValueError(f"empty engine id in {text!r}")
        if max_speech_ms is not None and max_speech_ms <= 0:
            raise ValueError(f"max_speech_ms must be positive, got {max_speech_ms}")
        return cls(engine_id, size or None, max_speech_ms)


def _with_max_speech(segmenter, max_speech_ms: int):
    """CLI と同じ preset の VAD を、区間の長さの上限だけ変えて作り直す。"""
    from dataclasses import replace

    from livecap_cli.vad import VADFileSegmenter, VADProcessor

    processor = segmenter._vad  # CLI の _build_file_segmenter が作った VADProcessor
    return VADFileSegmenter(
        VADProcessor(config=replace(processor.config, max_speech_ms=max_speech_ms), backend=processor._backend)
    )


def build_transcribe_args(spec: EngineSpec, *, device: str = "auto", language: str = "ja"):
    """CLI の ``transcribe`` と同じ既定値の引数を作る (入力ファイルは呼ぶ側で渡す)。

    Raises:
        ValueError: その ASR が ``language`` に対応していない
    """
    from livecap_cli.cli import build_parser
    from livecap_cli.engines import EngineMetadata

    argv = ["transcribe", "-", "--engine", spec.engine_id, "--language", language, "--device", device]
    if spec.model_size:
        argv += ["--model-size", spec.model_size]
    args = build_parser().parse_args(argv)
    args.language = EngineMetadata.resolve_language(spec.engine_id, args.language)
    return args


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version()}
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        out["livecap-cli"] = version("livecap-cli")
    with contextlib.suppress(Exception):
        import torch

        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    return out


def _reset_peak_gpu() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def _peak_gpu_mb() -> float | None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 2**20
    return None


#: 再開してよいかを決める run.json の項目 (結果に効く設定)
_RESUME_KEYS = ("args", "vad_config", "vad_backend")


def _settings_id(settings: dict[str, Any]) -> str:
    """結果に効く設定の識別子。run.json と動画ごとの記録の両方に入れ、違う設定の記録を見分ける。"""
    canonical = json.dumps({k: settings[k] for k in _RESUME_KEYS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _audio_sha256(path: Path) -> str:
    """入力の音声の識別子。同じ ID でも中身の違う音声で書き起こした記録を使い回さない。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record_problem(record: Any) -> str | None:
    """採点が読む項目と型がそろっているか。問題があれば理由を返す (成功・失敗は別に見る)。"""
    if not isinstance(record, dict):
        return "not a JSON object"
    if not isinstance(record.get("success"), bool):
        return "'success' is missing or not a bool"
    for key in ("elapsed_sec", "duration_sec"):
        if not _is_number(record.get(key)):
            return f"{key!r} is missing or not a number"
    drops = record.get("drop_counts")
    if not isinstance(drops, dict) or not all(isinstance(n, int) and not isinstance(n, bool) for n in drops.values()):
        return "'drop_counts' is missing or not a mapping of counts"
    for key, required in (("segments", True), ("vad_segments", False)):
        if not required and key not in record:
            continue
        items = record.get(key)
        if not isinstance(items, list) or not all(
            isinstance(x, list) and len(x) == 3 and _is_number(x[0]) and _is_number(x[1]) and isinstance(x[2], str)
            for x in items
        ):
            return f"{key!r} is missing or not a list of [start, end, text]"
    return None


def _read_record(path: Path) -> dict[str, Any]:
    """動画ごとの記録を読み、項目と型を確かめる。

    Raises:
        FileNotFoundError: 記録が無い
        ValueError: 読めない、または採点に要る項目・型がそろっていない
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    problem = _record_problem(record)
    if problem:
        raise ValueError(f"{path}: {problem}")
    return record


def _transcribed(path: Path, settings_id: str, audio_sha256: str) -> bool:
    """今回と同じ設定・同じ音声で書き起こしに成功した、そろった記録があるか。

    失敗した記録、読めない・項目の足りない記録、違う設定や違う音声の記録はやり直す。
    """
    try:
        record = _read_record(path)
    except (OSError, ValueError):
        return False
    return (
        record["success"] is True
        and record.get("settings_id") == settings_id
        and record.get("audio_sha256") == audio_sha256
    )


def _check_resumable(dest: Path, settings: dict[str, Any]) -> None:
    """``dest`` に前の結果があるとき、今回の設定で続けてよいか。違えば止める (別の条件の結果を上書きしない)。

    run.json は、その出力先で最後に始めた書き起こしの設定。同じなら、違う設定の記録 (``--force`` を途中で
    止めたときの残り) は ``_transcribed`` がやり直しに回す。

    Raises:
        RuntimeError: 前の設定が分からない、または今回と違う
    """
    has_records = dest.is_dir() and any(p.suffix == ".json" and p.name != "run.json" for p in dest.iterdir())
    try:
        previous = json.loads((dest / "run.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        if not has_records:
            return  # 初めての出力先
        previous = {}
    except (OSError, ValueError):
        previous = {}
    # JSON に直して比べる (tuple と list の違いなどを無視する)
    current = json.loads(json.dumps(settings))
    differ = [k for k in _RESUME_KEYS if previous.get(k) != current[k]]
    if differ:
        raise RuntimeError(
            f"{dest} has results made with different or unknown settings ({', '.join(differ)}). "
            "Use --force to redo every video, or --out-dir for a separate run."
        )


def transcribe_videos(
    spec: EngineSpec,
    videos: Sequence[Yodas2Video],
    *,
    audio_dir: Path,
    out_dir: Path,
    device: str = "auto",
    force: bool = False,
) -> Path:
    """動画を書き起こし、``<out_dir>/<label>/`` に保存する。保存先を返す。

    Raises:
        FileNotFoundError: 音声が無い (``scripts/prepare_yodas2_audio.py`` で取る)
        RuntimeError: 書き起こし済みの結果と設定が違う (``force`` でやり直すか、別の ``out_dir`` にする)。
            または、書き起こしに失敗した動画が残った (記録は保存してあり、同じ呼び出しでやり直せる)
    """
    from livecap_cli import cli
    from livecap_cli.runtime import configure_pytorch_runtime
    from livecap_cli.transcription import FileTranscriptionPipeline

    dest = out_dir / spec.label
    # 書き起こし済みかを音声の中身で確かめるので、全部の動画の音声が要る
    missing = [v.audio_id for v in videos if not (audio_dir / f"{v.audio_id}.wav").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files are missing in {audio_dir} (first: {missing[0]}). "
            "Run: python scripts/prepare_yodas2_audio.py"
        )
    configure_pytorch_runtime()
    args = build_transcribe_args(spec, device=device)
    # CLI の _transcribe_file と同じ部品・同じ順序 (segmenter → preprocessor → engine → transcriber)
    segmenter = cli._build_file_segmenter(args)
    if spec.max_speech_ms is not None:
        segmenter = _with_max_speech(segmenter, spec.max_speech_ms)
    vad = getattr(segmenter, "_vad", None)  # VADFileSegmenter の VADProcessor (テストの偽物には無い)
    settings = {
        "args": {k: getattr(args, k) for k in _RECORDED_ARGS},
        # 実際に使った VAD の設定 (preset の値と、上限を変えたときはその値)
        "vad_config": None if vad is None else {k: getattr(vad.config, k) for k in vad.config.__slots__},
        "vad_backend": None if vad is None else vad._backend.name,
    }
    settings_id = _settings_id(settings)
    if not force:  # 前の結果があれば、設定が同じかを engine を読み込む前に確かめる
        _check_resumable(dest, settings)
    audio_ids = {v.audio_id: _audio_sha256(audio_dir / f"{v.audio_id}.wav") for v in videos}
    todo = [
        v for v in videos
        if force or not _transcribed(dest / f"{v.audio_id}.json", settings_id, audio_ids[v.audio_id])
    ]
    if not todo:
        logger.info("%s: all %d videos already transcribed", spec.label, len(videos))
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    audio_preprocessor = cli._build_audio_preprocessor(args)
    pipeline = None
    engine = None
    try:
        _reset_peak_gpu()
        load_started = time.perf_counter()
        engine = cli._load_engine(args)
        load_sec = time.perf_counter() - load_started
        # VAD が切った区間と、区間ごとの結果 (認識した / EnergyGate / filter で落ちた) を残す。
        # pipeline は区間を順に 1 つずつ処理し、長さ 0 の区間だけを飛ばす (file_pipeline._transcribe_segments)
        spans: list[tuple[float, float, int]] = []

        def recording_segmenter(audio, sample_rate):
            found = list(segmenter(audio, sample_rate))
            spans[:] = [(float(s), float(e), sample_rate) for s, e in found]
            return found

        pipeline = FileTranscriptionPipeline(segmenter=recording_segmenter, audio_preprocessor=audio_preprocessor)
        (dest / "run.json").write_text(
            json.dumps(
                {
                    "label": spec.label,
                    "engine_name": engine.get_engine_name(),
                    **settings,
                    "settings_id": settings_id,
                    "load_sec": round(load_sec, 2),
                    "versions": _versions(),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        for n, video in enumerate(todo, 1):
            path = audio_dir / f"{video.audio_id}.wav"
            base_transcriber = cli._build_segment_transcriber(args, engine, path)
            outcomes: list[str] = []

            def transcriber(audio, sample_rate, _base=base_transcriber, _out=outcomes):
                try:
                    outcome = _base(audio, sample_rate)
                except Exception:
                    _out.append("error")
                    raise
                _out.append(outcome.drop_reason or ("asr" if outcome.text.strip() else "empty"))
                return outcome
            started = time.perf_counter()
            result = pipeline.process_file(
                path, segment_transcriber=transcriber, write_subtitles=False, write_translated_subtitles=False
            )
            elapsed = time.perf_counter() - started
            record = {
                "audio_id": video.audio_id,
                "label": spec.label,
                # どの設定・どの音声の結果か (再開と採点で、違う実行の記録を使わない)
                "settings_id": settings_id,
                "audio_sha256": audio_ids[video.audio_id],
                "success": result.success,
                "error": result.error,
                "elapsed_sec": round(elapsed, 3),
                "duration_sec": video.duration_sec,
                "rtf": round(elapsed / video.duration_sec, 5),
                "drop_counts": result.metadata.get("drop_counts") or {},
                "segments": [[round(s.start, 3), round(s.end, 3), s.text] for s in result.subtitles],
            }
            processed = [(s, e) for s, e, sr in spans if int(e * sr) > int(s * sr)]
            if len(processed) == len(outcomes):
                record["vad_segments"] = [[round(s, 3), round(e, 3), o] for (s, e), o in zip(processed, outcomes)]
            else:  # 対応が取れないときは、取れない旨だけ残す (推測で並べない)
                record["vad_segments_mismatch"] = {"spans": len(processed), "outcomes": len(outcomes)}
            tmp = dest / f"{video.audio_id}.json.tmp"
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(dest / f"{video.audio_id}.json")
            logger.info(
                "%s [%d/%d] %s: %d segments, RTF %.3f%s",
                spec.label, n, len(todo), video.audio_id, len(record["segments"]), record["rtf"],
                "" if result.success else f" (error: {result.error})",
            )
        peak = _peak_gpu_mb()
        if peak is not None:
            run = json.loads((dest / "run.json").read_text(encoding="utf-8"))
            run["peak_gpu_reserved_mb"] = round(peak, 1)
            (dest / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        # CLI と同じ順序で片付ける (pipeline → engine)
        for closer in (getattr(pipeline, "close", None), getattr(engine, "cleanup", None)):
            if closer is not None:
                with contextlib.suppress(Exception):
                    closer()
    failed = [
        v.audio_id for v in videos
        if not _transcribed(dest / f"{v.audio_id}.json", settings_id, audio_ids[v.audio_id])
    ]
    if failed:  # 黙って終えない (採点は失敗を空の出力として扱わず止まるので、ここで先に知らせる)
        raise RuntimeError(
            f"{spec.label}: {len(failed)} of {len(videos)} videos were not transcribed (first: {failed[0]}). "
            "Run the same command again to retry them."
        )
    return dest


def load_hypotheses(run_dir: Path, videos: Sequence[Yodas2Video]) -> dict[str, list[tuple[float, float, str]]]:
    """``transcribe_videos`` の出力を読む。

    どの記録も run.json と同じ設定 (``settings_id``) のものでなければ採点しない。``settings_id`` は記録を
    足す前の出力には無い。その場合は、run.json にも全部の記録にも無いときだけ、確かめられない旨を
    警告して読む (一部の記録にだけあるなら、違う実行が混ざっているので止める)。

    Raises:
        FileNotFoundError: 書き起こしが無い動画がある
        ValueError: 読めない、または採点に要る項目・型がそろっていない記録がある
        RuntimeError: 書き起こしに失敗した動画、または run.json と違う設定の記録がある
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    expected = run.get("settings_id")
    out: dict[str, list[tuple[float, float, str]]] = {}
    failed, foreign = [], []
    for video in videos:
        record = _read_record(run_dir / f"{video.audio_id}.json")
        if record.get("settings_id") != expected:
            foreign.append(video.audio_id)
        if not record["success"]:
            failed.append(f"{video.audio_id}: {record.get('error')}")
        out[video.audio_id] = [(float(s), float(e), t) for s, e, t in record["segments"]]
    if foreign:
        raise RuntimeError(
            f"{len(foreign)} records in {run_dir} were made with settings other than run.json "
            f"(first: {foreign[0]}). Run transcribe again to redo them."
        )
    if failed:
        raise RuntimeError(f"transcription failed for {len(failed)} videos in {run_dir}: {failed[:3]}")
    if expected is None:
        logger.warning("%s: written before settings_id was recorded; cannot check that all records share one run", run_dir)
    return out
