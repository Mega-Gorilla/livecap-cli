"""長尺の評価の書き起こし (#488)。

**CLI の ``livecap-cli transcribe <file>`` と同じ部品で書き起こす。** 引数の既定値 (VAD の preset、
EnergyGate の閾値、confidence filter) は :func:`livecap_cli.cli.build_parser` から得るので、CLI の既定が
変われば評価も追従する。違いは、モデルを 1 回だけ読み込み、複数の動画を続けて処理することだけ
(VAD の segmenter はファイルごとに状態を戻すので、使い回しても結果は変わらない)。

処理の順序は CLI と同じ: VAD で区切る (1 秒ずつ流す) → EnergyGate → 認識 → confidence filter。

出力: ``<out_dir>/<label>/<audio_id>.json`` (区間ごとの文と時刻、処理時間、drop の内訳、VAD の区間ごとの
結果、どの設定・どの音声の結果か) と ``<out_dir>/<label>/run.json`` (その出力先で最後に始めた設定と、
動画ごとに正とする音声の SHA-256)。VAD の区間ごとには、EnergyGate が見る音量と、ASR の生の出力 (confidence
filter にかける前の文と confidence の値) も残す (``segment_details``)。EnergyGate や filter の閾値を変えたときの
結果を、書き起こし直さずに計算で出すため (#488 段階 2)。

再開: 今回と同じ設定 (``settings_id``)・同じ音声 (SHA-256) で書き起こしに成功し、項目・型・値の範囲が
そろった記録 (:func:`_read_record`) がある動画だけを飛ばす。それ以外 (失敗、読めない、欠けた、違う設定や
違う音声の記録) はやり直す。``run.json`` の設定が今回と違えば、engine を読む前に止まる (違う条件の結果を
上書きしない。``force`` で全部やり直す)。採点 (:func:`load_hypotheses`) も、run.json と違う設定・違う
音声の記録が混ざっていれば止まる。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import platform
import time
from dataclasses import asdict, dataclass
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


#: ``EngineSpec.vad`` に渡せる VAD の backend (CLI の ``--vad`` のうち ``auto`` 以外)
VAD_BACKENDS = ("silero", "tenvad", "webrtc")


@dataclass(frozen=True)
class EngineSpec:
    """測る ASR。``model_size`` は ``whispers2t`` だけが使う (CLI の ``--model-size``)。

    次の 3 つは実験用で、``None`` なら CLI の既定と同じ。変えたものは ``label`` に付き、結果を分けて保存する。

    - ``max_speech_ms``: VAD の区間の長さの上限 (CLI には無い)。``None`` なら preset の値
    - ``vad``: VAD の backend (CLI の ``--vad``)。``None`` なら ``auto`` (言語と engine で決める)。
      backend を変えても、その backend の、その engine 用の preset を使う
    - ``engine_min_rms``: EnergyGate の閾値 (dBFS、CLI の ``--engine-min-rms``)。``-inf`` で外す
    """

    engine_id: str
    model_size: str | None = None
    max_speech_ms: int | None = None
    vad: str | None = None
    engine_min_rms: float | None = None

    @property
    def label(self) -> str:
        label = f"{self.engine_id}-{self.model_size}" if self.model_size else self.engine_id
        if self.max_speech_ms is not None:
            label += f"-max{self.max_speech_ms}ms"
        if self.vad is not None:
            label += f"-vad{self.vad}"
        if self.engine_min_rms is not None:
            label += "-gateoff" if math.isinf(self.engine_min_rms) else f"-gate{self.engine_min_rms:g}db"
        return label

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        max_speech_ms: int | None = None,
        vad: str | None = None,
        engine_min_rms: float | None = None,
    ) -> EngineSpec:
        """``"qwen3asr_large"`` や ``"whispers2t:large-v3"`` を読む。"""
        engine_id, _, size = text.partition(":")
        if not engine_id:
            raise ValueError(f"empty engine id in {text!r}")
        if max_speech_ms is not None and max_speech_ms <= 0:
            raise ValueError(f"max_speech_ms must be positive, got {max_speech_ms}")
        if vad is not None and vad not in VAD_BACKENDS:
            raise ValueError(f"vad must be one of {VAD_BACKENDS}, got {vad!r}")
        # CLI と同じく、NaN と +inf は受け付けない (-inf は EnergyGate を外す)
        if engine_min_rms is not None and (math.isnan(engine_min_rms) or engine_min_rms == math.inf):
            raise ValueError(f"engine_min_rms must be a number or -inf, got {engine_min_rms}")
        return cls(engine_id, size or None, max_speech_ms, vad, engine_min_rms)


class _RecordingEngine:
    """engine の ``transcribe`` の戻り値 (filter にかける前の文と confidence) を ``last`` に残す。

    CLI の segment transcriber にこれを渡すと、confidence filter が捨てた区間でも、ASR が出した文と
    filter が見た値が分かる。ほかの属性は engine のものをそのまま返す (CLI の処理は変わらない)。
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self.last = None

    def transcribe(self, audio, sample_rate):
        result = self._engine.transcribe(audio, sample_rate)
        self.last = result
        return result

    def __getattr__(self, name: str):
        return getattr(self._engine, name)


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
    from livecap_cli.cli import _resolve_default_model_size, build_parser
    from livecap_cli.engines import EngineMetadata

    argv = ["transcribe", "-", "--engine", spec.engine_id, "--language", language, "--device", device]
    if spec.model_size:
        argv += ["--model-size", spec.model_size]
    # 実験で変える設定も CLI の引数として渡し、CLI と同じ検査と解釈を通す
    if spec.vad is not None:
        argv += ["--vad", spec.vad]
    if spec.engine_min_rms is not None:
        argv.append("--engine-min-rms=" + ("off" if math.isinf(spec.engine_min_rms) else repr(spec.engine_min_rms)))
    args = build_parser().parse_args(argv)
    args.language = EngineMetadata.resolve_language(spec.engine_id, args.language)
    # CLI と同じく、大きさを指定しない whispers2t はハードウェアで決める。決めた値が記録 (run.json) に残る
    _resolve_default_model_size(args)
    # CLI と違い、GPU で動かせないときに CPU へ切り替えさせない。切り替えると、記録する条件 (GPU・大きさ) と
    # 違うモデル・装置の結果になり、再開や採点で同じ条件の結果として混ざる (PR #512 F4)。元のエラーで
    # その動画は失敗として残り、やり直す
    args.whisper_allow_cpu_fallback = False
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


def _json_value(value: Any) -> Any:
    """run.json に残す設定の値。有限でない数 (EnergyGate を外した ``-inf``) は文字列にする。

    JSON の規格に無限大は無く、そのまま書くと ``-Infinity`` になり、Python 以外では読めない
    (run.json の設定は集計の ``summary.json`` にも写る)。
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # "-inf"
    return value


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
    """有限の数 (bool・NaN・無限大は除く)。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_span(item: Any) -> bool:
    """``[start, end, text]`` で、0 <= start <= end の有限の時刻。"""
    return (
        isinstance(item, list) and len(item) == 3 and _is_number(item[0]) and _is_number(item[1])
        and 0 <= item[0] <= item[1] and isinstance(item[2], str)
    )


#: ``segment_details`` の ``confidence`` に入る数の項目 (``EngineConfidence`` の signal)
_CONFIDENCE_FIELDS = ("no_speech_prob", "avg_logprob", "compression_ratio", "token_confidence_mean")


def _is_float_or_none(value: Any) -> bool:
    """``None`` か数 (bool は除く)。confidence には ``-inf`` も入りうるので、有限には限らない。"""
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _is_detail(item: Any, outcome: str) -> bool:
    """``segment_details`` の 1 区間。ASR を呼んだ区間 (EnergyGate で落ちず、ASR が例外を出さなかった) にだけ
    ``text`` と ``confidence`` がある。``outcome`` は同じ位置の ``vad_segments`` の結果。"""
    if not (isinstance(item, dict) and set(item) == {"energy_dbfs", "text", "confidence"}):
        return False
    if not _is_number(item["energy_dbfs"]):
        return False
    asr_called = not (outcome.startswith("energy_gate") or outcome == "error")
    if not asr_called:
        return item["text"] is None and item["confidence"] is None
    conf = item["confidence"]
    return (
        isinstance(item["text"], str)
        and isinstance(conf, dict)
        and set(conf) == {*_CONFIDENCE_FIELDS, "raw"}
        and all(_is_float_or_none(conf[k]) for k in _CONFIDENCE_FIELDS)
        and isinstance(conf["raw"], dict)
        and all(_is_float_or_none(v) for v in conf["raw"].values())
    )


def _record_problem(record: Any) -> str | None:
    """採点が読む項目の有無・型・値の範囲がそろっているか。問題があれば理由を返す (成功・失敗は別に見る)。"""
    if not isinstance(record, dict):
        return "not a JSON object"
    if not isinstance(record.get("success"), bool):
        return "'success' is missing or not a bool"
    # RTF は elapsed_sec ÷ duration_sec なので、長さは正、処理時間は 0 以上の有限の数に限る
    if not (_is_number(record.get("duration_sec")) and record["duration_sec"] > 0):
        return "'duration_sec' is missing or not a finite positive number"
    if not (_is_number(record.get("elapsed_sec")) and record["elapsed_sec"] >= 0):
        return "'elapsed_sec' is missing or not a finite non-negative number"
    drops = record.get("drop_counts")
    if not isinstance(drops, dict) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in drops.values()
    ):
        return "'drop_counts' is missing or not a mapping of non-negative counts"
    for key, required in (("segments", True), ("vad_segments", False)):
        if not required and key not in record:
            continue
        items = record.get(key)
        if not isinstance(items, list) or not all(_is_span(x) for x in items):
            return f"{key!r} is missing or not a list of [start, end, text] with 0 <= start <= end"
    if "segment_details" in record:  # vad_segments と同じ順・同じ数で、区間ごとの音量と ASR の生の出力
        details = record["segment_details"]
        spans = record.get("vad_segments")
        if not (
            isinstance(details, list) and isinstance(spans, list) and len(details) == len(spans)
            and all(_is_detail(d, s[2]) for d, s in zip(details, spans))
        ):
            return "'segment_details' does not match 'vad_segments' or has a malformed entry"
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
    from livecap_cli.audio import _segment_energy_dbfs
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
        "args": {k: _json_value(getattr(args, k)) for k in _RECORDED_ARGS},
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
        # 動画ごとの、いま正とする音声の SHA-256。同じ設定の前の実行の分に、今回の分を上書きして足す。
        # 記録を書き直す前に run.json へ書くので、途中で止めても、古い音声の記録は採点で見分けられる
        try:
            previous = json.loads((dest / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        audio_map = dict(previous.get("audio_sha256") or {}) if previous.get("settings_id") == settings_id else {}
        audio_map.update(audio_ids)
        (dest / "run.json").write_text(
            json.dumps(
                {
                    "label": spec.label,
                    "engine_name": engine.get_engine_name(),
                    **settings,
                    "settings_id": settings_id,
                    "audio_sha256": dict(sorted(audio_map.items())),
                    "load_sec": round(load_sec, 2),
                    "versions": _versions(),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        recorder = _RecordingEngine(engine)
        for n, video in enumerate(todo, 1):
            path = audio_dir / f"{video.audio_id}.wav"
            base_transcriber = cli._build_segment_transcriber(args, recorder, path)
            outcomes: list[str] = []
            # 区間ごとの、EnergyGate が見る音量と、ASR の生の出力 (filter にかける前)。EnergyGate や filter の
            # 閾値を変えたときの結果を、書き起こし直さずに計算で出すため (#488 段階 2)。音量は EnergyGate を
            # 外していても測る
            details: list[dict[str, Any]] = []

            def transcriber(audio, sample_rate, _base=base_transcriber, _out=outcomes, _details=details):
                detail: dict[str, Any] = {
                    "energy_dbfs": _segment_energy_dbfs(
                        audio, sample_rate, metric=args.engine_energy_metric, frame_ms=args.engine_energy_frame_ms
                    ),
                    "text": None,
                    "confidence": None,
                }
                _details.append(detail)
                recorder.last = None
                try:
                    outcome = _base(audio, sample_rate)
                except Exception:
                    _out.append("error")
                    raise
                if recorder.last is not None:  # ASR を呼んだ (EnergyGate で落ちなかった)
                    detail["text"] = recorder.last.text
                    detail["confidence"] = asdict(recorder.last.engine_confidence)
                _out.append(outcome.drop_reason or ("asr" if outcome.text.strip() else "empty"))
                return outcome
            started = time.perf_counter()
            result = pipeline.process_file(
                path, segment_transcriber=transcriber, write_subtitles=False, write_translated_subtitles=False
            )
            elapsed = time.perf_counter() - started
            # 区間の一部で ASR が落ちた動画も、成功として残さない (pipeline は一部の失敗なら成功にする、#362)。
            # 落ちた区間の文字が抜けたまま採点され、再開でもやり直されないため (PR #512 F4)
            failed_segments = outcomes.count("error")
            success = result.success and failed_segments == 0
            error = result.error
            if result.success and failed_segments:
                error = f"{failed_segments} of {len(outcomes)} segments failed in the ASR"
            record = {
                "audio_id": video.audio_id,
                "label": spec.label,
                # どの設定・どの音声の結果か (再開と採点で、違う実行の記録を使わない)
                "settings_id": settings_id,
                "audio_sha256": audio_ids[video.audio_id],
                "success": success,
                "error": error,
                "elapsed_sec": round(elapsed, 3),
                "duration_sec": video.duration_sec,
                "rtf": round(elapsed / video.duration_sec, 5),
                "drop_counts": result.metadata.get("drop_counts") or {},
                "segments": [[round(s.start, 3), round(s.end, 3), s.text] for s in result.subtitles],
            }
            processed = [(s, e) for s, e, sr in spans if int(e * sr) > int(s * sr)]
            if len(processed) == len(outcomes) == len(details):
                record["vad_segments"] = [[round(s, 3), round(e, 3), o] for (s, e), o in zip(processed, outcomes)]
                record["segment_details"] = details
            else:  # 対応が取れないときは、取れない旨だけ残す (推測で並べない)
                record["vad_segments_mismatch"] = {"spans": len(processed), "outcomes": len(outcomes)}
            tmp = dest / f"{video.audio_id}.json.tmp"
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(dest / f"{video.audio_id}.json")
            logger.info(
                "%s [%d/%d] %s: %d segments, RTF %.3f%s",
                spec.label, n, len(todo), video.audio_id, len(record["segments"]), record["rtf"],
                "" if success else f" (error: {error})",
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

    どの記録も、run.json と同じ設定 (``settings_id``) で、run.json がその動画について正とする音声
    (``audio_sha256``) から書き起こしたものでなければ採点しない。``settings_id`` は記録を足す前の出力には
    無い。その場合は、run.json にも全部の記録にも無いときだけ、確かめられない旨を警告して読む
    (一部の記録にだけあるなら、違う実行が混ざっているので止める)。

    Raises:
        FileNotFoundError: 書き起こしが無い動画がある
        ValueError: 読めない、または採点に要る項目・型・値の範囲がそろっていない記録がある
        RuntimeError: 書き起こしに失敗した動画、run.json と違う設定の記録、または run.json と違う音声の記録がある
    """
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    expected = run.get("settings_id")
    audio_map = run.get("audio_sha256") if isinstance(run.get("audio_sha256"), dict) else {}
    out: dict[str, list[tuple[float, float, str]]] = {}
    failed, foreign, stale = [], [], []
    for video in videos:
        record = _read_record(run_dir / f"{video.audio_id}.json")
        if record.get("settings_id") != expected:
            foreign.append(video.audio_id)
        elif expected is not None and (
            audio_map.get(video.audio_id) is None or record.get("audio_sha256") != audio_map[video.audio_id]
        ):
            stale.append(video.audio_id)
        if not record["success"]:
            failed.append(f"{video.audio_id}: {record.get('error')}")
        out[video.audio_id] = [(float(s), float(e), t) for s, e, t in record["segments"]]
    if foreign:
        raise RuntimeError(
            f"{len(foreign)} records in {run_dir} were made with settings other than run.json "
            f"(first: {foreign[0]}). Run transcribe again to redo them."
        )
    if stale:
        raise RuntimeError(
            f"{len(stale)} records in {run_dir} were made from audio other than run.json's "
            f"(first: {stale[0]}). Run transcribe again to redo them."
        )
    if failed:
        raise RuntimeError(f"transcription failed for {len(failed)} videos in {run_dir}: {failed[:3]}")
    if expected is None:
        logger.warning("%s: written before settings_id was recorded; cannot check that all records share one run", run_dir)
    return out
