"""長尺の評価の書き起こし (#488)。

**CLI の ``livecap-cli transcribe <file>`` と同じ部品で書き起こす。** 引数の既定値 (VAD の preset、
EnergyGate の閾値、confidence filter) は :func:`livecap_cli.cli.build_parser` から得るので、CLI の既定が
変われば評価も追従する。違いは、モデルを 1 回だけ読み込み、複数の動画を続けて処理することだけ
(VAD の segmenter はファイルごとに状態を戻すので、使い回しても結果は変わらない)。

処理の順序は CLI と同じ: VAD で区切る (1 秒ずつ流す) → EnergyGate → 認識 → confidence filter。

出力: ``<out_dir>/<label>/<audio_id>.json`` (区間ごとの文と時刻、処理時間、drop の内訳) と
``<out_dir>/<label>/run.json`` (使った設定)。すでにある動画は飛ばすので、途中から再開できる。
"""

from __future__ import annotations

import contextlib
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
    """測る ASR。``model_size`` は ``whispers2t`` だけが使う (CLI の ``--model-size``)。"""

    engine_id: str
    model_size: str | None = None

    @property
    def label(self) -> str:
        return f"{self.engine_id}-{self.model_size}" if self.model_size else self.engine_id

    @classmethod
    def parse(cls, text: str) -> EngineSpec:
        """``"qwen3asr_large"`` や ``"whispers2t:large-v3"`` を読む。"""
        engine_id, _, size = text.partition(":")
        if not engine_id:
            raise ValueError(f"empty engine id in {text!r}")
        return cls(engine_id, size or None)


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
    """
    from livecap_cli import cli
    from livecap_cli.runtime import configure_pytorch_runtime
    from livecap_cli.transcription import FileTranscriptionPipeline

    dest = out_dir / spec.label
    todo = [v for v in videos if force or not (dest / f"{v.audio_id}.json").exists()]
    missing = [v.audio_id for v in todo if not (audio_dir / f"{v.audio_id}.wav").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files are missing in {audio_dir} (first: {missing[0]}). "
            "Run: python scripts/prepare_yodas2_audio.py"
        )
    if not todo:
        logger.info("%s: all %d videos already transcribed", spec.label, len(videos))
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    configure_pytorch_runtime()
    args = build_transcribe_args(spec, device=device)
    # CLI の _transcribe_file と同じ部品・同じ順序 (segmenter → preprocessor → engine → transcriber)
    segmenter = cli._build_file_segmenter(args)
    audio_preprocessor = cli._build_audio_preprocessor(args)
    pipeline = None
    engine = None
    try:
        _reset_peak_gpu()
        load_started = time.perf_counter()
        engine = cli._load_engine(args)
        load_sec = time.perf_counter() - load_started
        pipeline = FileTranscriptionPipeline(segmenter=segmenter, audio_preprocessor=audio_preprocessor)
        (dest / "run.json").write_text(
            json.dumps(
                {
                    "label": spec.label,
                    "engine_name": engine.get_engine_name(),
                    "args": {k: getattr(args, k) for k in _RECORDED_ARGS},
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
            transcriber = cli._build_segment_transcriber(args, engine, path)
            started = time.perf_counter()
            result = pipeline.process_file(
                path, segment_transcriber=transcriber, write_subtitles=False, write_translated_subtitles=False
            )
            elapsed = time.perf_counter() - started
            record = {
                "audio_id": video.audio_id,
                "label": spec.label,
                "success": result.success,
                "error": result.error,
                "elapsed_sec": round(elapsed, 3),
                "duration_sec": video.duration_sec,
                "rtf": round(elapsed / video.duration_sec, 5),
                "drop_counts": result.metadata.get("drop_counts") or {},
                "segments": [[round(s.start, 3), round(s.end, 3), s.text] for s in result.subtitles],
            }
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
    return dest


def load_hypotheses(run_dir: Path, videos: Sequence[Yodas2Video]) -> dict[str, list[tuple[float, float, str]]]:
    """``transcribe_videos`` の出力を読む。

    Raises:
        FileNotFoundError: 書き起こしが無い動画がある
        RuntimeError: 書き起こしに失敗した動画がある (失敗を空の出力として採点しない)
    """
    out: dict[str, list[tuple[float, float, str]]] = {}
    failed = []
    for video in videos:
        record = json.loads((run_dir / f"{video.audio_id}.json").read_text(encoding="utf-8"))
        if not record["success"]:
            failed.append(f"{video.audio_id}: {record['error']}")
        out[video.audio_id] = [(float(s), float(e), str(t)) for s, e, t in record["segments"]]
    if failed:
        raise RuntimeError(f"transcription failed for {len(failed)} videos in {run_dir}: {failed[:3]}")
    return out
