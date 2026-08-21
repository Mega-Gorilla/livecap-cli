"""プローブ実行と verdict 導出 (Issue #378)。

**differential 方式**: すべてのプローブは同じ操作を 2 回走らせる — ASCII の
control root で 1 回、非 ASCII variant で 1 回 — そして verdict は
**その比較**から導出する。固定の期待値を持たないので、モデルやライブラリを
更新しても壊れない。そして何より、``fail_silent`` を機械的に検出できる
唯一の方法がこれである。

実装 PR (#375 / #379 / #377) はここを ``from tests.nonascii import run_probe``
で直接再利用できる。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from .paths import make_variant_root, variant as get_variant
from .record import ProbeResult, Verdict
from .registry import BOUNDARIES_BY_ID, REPO_ROOT
from .worker import SENTINEL

DEFAULT_TIMEOUT_S = 120.0

#: 真因を握り潰して別のメッセージに差し替える既知の箇所。
#: これらの文言が出たということは、非 ASCII が原因かどうかが**呼び出し側から
#: 判別できない**ことを意味するので fail_silent 扱いにする。
MANGLER_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("NeMo is not installed", "nemo_utils.check_nemo_availability が真因を握り潰す"),
    ("Failed to load engine class", "engine_factory._get_engine_class が型ごと差し替える"),
    ("ダウンロードしたモデルが破損", "base_engine._verify_model_integrity が真因を消す"),
    ("nemo_toolkit", "NeMo import 失敗が汎用メッセージに化ける"),
)


class HarnessError(RuntimeError):
    """ハーネス自体の失敗。境界のバグの証拠ではない。"""


def _child_env(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """子プロセスの環境を組み立てる。

    **親の ``os.environ`` は絶対に触らない** — 触ると測定対象の欠陥
    (``utils/__init__.py`` の無ロック env 書き換え) をハーネス内で再現してしまう。
    """
    env = dict(os.environ)

    # livecap の root 注入点 (既存の env var をそのまま使う)
    env["LIVECAP_CORE_MODELS_DIR"] = str(root / "models")
    env["LIVECAP_CORE_CACHE_DIR"] = str(root / "cache")
    env["LIVECAP_RESOURCE_ROOT"] = str(root / "resources")

    # 素の %TEMP% を踏む境界 (parakeet / canary / qwen3asr の発話 wav、
    # file_pipeline の作業ディレクトリ、NeMo の内部 untar) 用
    temp = root / "temp"
    env["TEMP"] = str(temp)
    env["TMP"] = str(temp)
    env["TMPDIR"] = str(temp)

    # huggingface_hub / transformers
    env["HF_HOME"] = str(root / "hf")

    # 子プロセスが repo を import できるようにする
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )

    # stdio エンコーディングは probe 側の関心事。既定では触らず、
    # 実機の挙動 (日本語 Windows なら cp932) をそのまま測る。
    env.pop("PYTHONIOENCODING", None)

    for key in ("LIVECAP_CORE_MODELS_DIR", "LIVECAP_CORE_CACHE_DIR",
                "LIVECAP_RESOURCE_ROOT", "TEMP", "HF_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    if extra:
        env.update(extra)
    return env


def _parse_framed_json(stdout: str) -> dict | None:
    """ネイティブライブラリの stdout 出力に埋もれた JSON を取り出す。"""
    parts = stdout.split(SENTINEL)
    if len(parts) < 3:
        return None
    try:
        return json.loads(parts[-2].strip())
    except Exception:
        return None


def _run_child(
    probe_id: str,
    *,
    variant_id: str,
    root: Path,
    is_control: bool,
    payload: dict | None,
    timeout_s: float,
    env_extra: dict[str, str] | None,
) -> dict:
    spec = {
        "probe_id": probe_id,
        "variant": variant_id,
        "root": str(root),
        "is_control": is_control,
        "payload": payload or {},
    }
    argv = [sys.executable, "-m", "tests.nonascii.worker"]
    invocation = {
        "command": argv,
        "cwd": str(REPO_ROOT),
        "timeout_s": timeout_s,
        "is_control": is_control,
        "variant": variant_id,
    }
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(spec, ensure_ascii=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
            env=_child_env(root, env_extra),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "timed_out": True,
            "exit_code": None,
            "duration_s": time.perf_counter() - started,
            "stdout": _as_text(exc.stdout),
            "stderr": _as_text(exc.stderr),
            "payload": None,
            "invocation": invocation,
        }
    except OSError as exc:
        # worker を起動できない環境 (実行制約・PYTHONPATH 不備など)。
        # ここで握り潰すと「原因不明の error_harness」になる。
        return {
            "timed_out": False,
            "exit_code": None,
            "duration_s": time.perf_counter() - started,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "payload": None,
            "invocation": {**invocation, "spawn_failed": True},
        }

    return {
        "timed_out": False,
        "exit_code": proc.returncode,
        "duration_s": time.perf_counter() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "payload": _parse_framed_json(proc.stdout),
        "invocation": invocation,
    }


def _as_text(value) -> str:
    """``TimeoutExpired`` の stdout/stderr は bytes のことがある。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def worker_diagnostics(child: dict) -> dict:
    """worker が JSON を返せなかったときの診断情報を組み立てる。

    **制限付き環境 (子プロセス実行やファイル作成に制約のある sandbox など) で
    「90 秒 timeout した」以上のことが分からない状態を避けるためのもの。**
    harness 側のバグなのか実行環境の制約なのかを、記録だけで切り分けられるように
    command / cwd / exit code / stderr を残す。
    """
    stdout = child.get("stdout") or ""
    return {
        **(child.get("invocation") or {}),
        "timed_out": bool(child.get("timed_out")),
        "exit_code": child.get("exit_code"),
        "duration_s": round(float(child.get("duration_s") or 0.0), 3),
        "sentinel_seen": SENTINEL in stdout,
        "stdout_tail": stdout[-1500:],
        "stderr_tail": (child.get("stderr") or "")[-3000:],
    }


def _collect_worker_diagnostics(control: dict, trial: dict) -> dict | None:
    """control / trial のうち JSON を返せなかった側の診断だけを返す。

    正常時は ``None`` — 決定性テストや control/trial の観測比較を汚さない。
    """
    out: dict = {}
    if not (control.get("payload") or {}):
        out["control"] = worker_diagnostics(control)
    if not (trial.get("payload") or {}):
        out["trial"] = worker_diagnostics(trial)
    return out or None


def _mentions_path(text: str, needles: list[str]) -> bool:
    """エラーメッセージが問題のパスに言及しているか。

    言及していれば「診断可能な失敗」= fail_loud。していなければ、
    利用者は何が起きたか分からない。

    **Unicode 正規化を跨いで比較する。** ``nfd`` variant では、ライブラリが
    エラーメッセージ中のパスを NFC 化して返し得る。素の部分文字列比較だと
    「パスを名指しした失敗」を「言及なし」と誤判定し、``fail_loud`` を
    ``fail_silent`` に落としてしまう。判定語彙の意味が変わるので、
    ``paths._roundtrip_ok()`` と同じく NFC に揃えてから比較する。
    """
    if not text:
        return False
    folded = unicodedata.normalize("NFC", text)
    for needle in needles:
        if not needle:
            continue
        candidates = (needle, unicodedata.normalize("NFC", needle))
        for candidate in candidates:
            if candidate in text or candidate in folded:
                return True
            # cp932 コンソール経由で ? に化けているケースも拾う
            escaped = ascii(candidate).strip("'")
            if escaped in text or escaped in folded:
                return True
    return False


def _hits_mangler(message: str) -> list[str]:
    return [why for sig, why in MANGLER_SIGNATURES if sig and sig in (message or "")]


def derive_verdict(
    *,
    control: dict,
    trial: dict,
    variant_id: str,
    variant_segment: str,
) -> tuple[str, list[str], dict]:
    """(verdict, silent_criteria_hit, 詳細) を返す。

    判定順序は上から順に適用する:

    1. ``error_harness`` — **control が失敗**、または worker の JSON が壊れている。
       非 ASCII バグの証拠として数えない。
    2. ``skipped``     — 依存未導入 / tier gate。
    3. ``fail_loud``   — 非ゼロ終了 or 例外、**かつ**問題のパスに言及している。
       ネイティブ ``abort()`` による非ゼロ終了もここ (loud は loud)。
    4. ``fail_silent`` — 下記 4 条件のいずれか。
    5. ``pass``        — control と観測的に等価。
    """
    detail: dict = {}

    cpay = control.get("payload")
    tpay = trial.get("payload")

    # --- 1. harness error -----------------------------------------------
    if cpay is None:
        return (
            Verdict.ERROR_HARNESS.value,
            [],
            {"why": "control の worker が JSON を返さなかった",
             "control_stderr": (control.get("stderr") or "")[-1500:]},
        )
    if cpay.get("harness_error"):
        return Verdict.ERROR_HARNESS.value, [], {"why": cpay["harness_error"]}

    # control 自身が skip なら trial も測れない
    if cpay.get("skipped_reason"):
        return Verdict.SKIPPED.value, [], {"skipped_reason": cpay["skipped_reason"]}

    if not cpay.get("ok"):
        return (
            Verdict.ERROR_HARNESS.value,
            [],
            {
                "why": "control (ASCII パス) の実行が失敗した。プローブが壊れている。",
                "exception_type": cpay.get("exception_type"),
                "exception_message": cpay.get("exception_message"),
            },
        )

    if tpay is None:
        # 子が JSON を出す前に死んだ = ネイティブ abort / hang など。
        # **プロセスが可視的に死ぬこと自体が loud** なので、メッセージが
        # パスを名指ししているかは問わない (loud は loud)。
        stderr = trial.get("stderr") or ""
        detail = {
            "why": "worker が JSON を出す前に終了 (ネイティブ abort の疑い)",
            "trial_stderr": stderr[-1500:],
            "error_mentions_path": _mentions_path(stderr, [variant_segment]),
        }
        if trial.get("timed_out"):
            return Verdict.FAIL_LOUD.value, [], {**detail, "why": "timeout"}
        if trial.get("exit_code"):
            return Verdict.FAIL_LOUD.value, [], detail
        # exit 0 なのに成果物が無い = 最も危険な「黙る」形
        return (
            Verdict.FAIL_SILENT.value,
            ["exit_zero_but_no_result"],
            detail,
        )

    if tpay.get("harness_error"):
        return Verdict.ERROR_HARNESS.value, [], {"why": tpay["harness_error"]}

    # --- 2. skipped ------------------------------------------------------
    if tpay.get("skipped_reason"):
        return Verdict.SKIPPED.value, [], {"skipped_reason": tpay["skipped_reason"]}

    control_obs = cpay.get("observation")
    control_stages = cpay.get("stages") or []
    trial_stages = tpay.get("stages") or []
    detail["control_observation"] = control_obs
    detail["control_stages"] = control_stages
    detail["trial_stages"] = trial_stages

    # --- 3 / 4. trial が例外で終わった場合 --------------------------------
    if not tpay.get("ok"):
        message = tpay.get("exception_message") or ""
        tb = tpay.get("traceback") or ""
        stderr = trial.get("stderr") or ""
        mentions = _mentions_path(
            f"{message}\n{tb}\n{stderr}", [variant_segment]
        )
        detail.update(
            {
                "exception_type": tpay.get("exception_type"),
                "exception_message": message,
                "error_mentions_path": mentions,
                "traceback_tail": tb[-1200:],
            }
        )

        # **パスを名指ししていれば診断可能 = loud。これを最優先する。**
        # 遅延失敗であっても、メッセージが問題のパスを指していれば利用者は
        # 原因に到達できるので silent ではない。
        if mentions:
            return Verdict.FAIL_LOUD.value, [], detail

        # ここから先は「何が起きたか利用者に分からない」形の失敗。
        # なぜ分からないのかを criteria として列挙する。
        criteria: list[str] = []
        # 条件 2: control が成功した地点より後段で落ちた = 遅延失敗
        if trial_stages and len(control_stages) > len(trial_stages):
            criteria.append("deferred_failure_at_later_stage")
        # 条件 3: 既知の mangler 署名に一致 (真因が別メッセージへすり替わっている)
        manglers = _hits_mangler(message)
        if manglers:
            criteria.append("mangled_exception:" + "; ".join(manglers))
        if not criteria:
            criteria.append("exception_does_not_name_path")
        return Verdict.FAIL_SILENT.value, criteria, detail

    # --- 5. trial が「成功」した場合: control と比較 ----------------------
    trial_obs = tpay.get("observation")
    detail["observation"] = trial_obs
    if trial_obs == control_obs:
        return Verdict.PASS.value, [], detail

    criteria = ["no_exception_output_differs_from_control"]
    if not trial_obs:
        criteria.append("exit_zero_but_no_artifact")
    return Verdict.FAIL_SILENT.value, criteria, detail


def run_probe(
    probe_id: str,
    *,
    variant_id: str,
    base_root: Path,
    boundary_id: str | None = None,
    payload: dict | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    env_extra: dict[str, str] | None = None,
    apply_to: str = "dir",
) -> ProbeResult:
    """1 境界 × 1 variant を測定する。control も内部で必ず走らせる。"""
    spec = BOUNDARIES_BY_ID.get(boundary_id or "")
    v = get_variant(variant_id)

    if not str(base_root).isascii():
        raise HarnessError(
            "base_root が非 ASCII。variant を効かせるのは base_root **配下**の "
            f"1 セグメントだけでなければならない: {base_root!r}"
        )

    # ffmpeg の入力側だけを測る等、「片側だけ非 ASCII」にしたい probe のための
    # ASCII 保証されたスクラッチ領域。
    ascii_scratch = base_root / "_ascii_scratch" / probe_id.replace(".", "_")
    ascii_scratch.mkdir(parents=True, exist_ok=True)
    payload = {**(payload or {}), "ascii_scratch": str(ascii_scratch)}

    control_root = make_variant_root(base_root, "control", probe_id)
    trial_root = make_variant_root(base_root, variant_id, probe_id)

    if spec is not None and spec.informational:
        # 照会系: ASCII と非 ASCII で答えが違うこと自体が観測目的なので、
        # 差分判定を行わない (control も走らせない)。
        trial = _run_child(
            probe_id,
            variant_id=variant_id,
            root=trial_root,
            is_control=False,
            payload=payload,
            timeout_s=timeout_s,
            env_extra=env_extra,
        )
        tpay = trial.get("payload") or {}
        if tpay.get("skipped_reason"):
            verdict, obs, skipped = Verdict.SKIPPED.value, None, tpay["skipped_reason"]
        elif tpay.get("ok"):
            verdict, obs, skipped = Verdict.PASS.value, tpay.get("observation"), None
        else:
            verdict, obs, skipped = Verdict.ERROR_HARNESS.value, None, None
        return ProbeResult(
            boundary_id=boundary_id or probe_id,
            probe_id=probe_id,
            variant=variant_id,
            apply_to=apply_to,
            tier=spec.tier,
            evidence_kind="runtime",
            verdict=verdict,
            control_verdict=None,
            exit_code=trial.get("exit_code"),
            timed_out=bool(trial.get("timed_out")),
            duration_s=round(float(trial.get("duration_s") or 0.0), 3),
            exception_type=tpay.get("exception_type"),
            exception_message=tpay.get("exception_message"),
            observation=obs,
            skipped_reason=skipped,
            notes="informational: control との差分判定は行わない",
            worker_diagnostics=(
                None if tpay else worker_diagnostics(trial)
            ),
        )

    control = _run_child(
        probe_id,
        variant_id="control",
        root=control_root,
        is_control=True,
        payload=payload,
        timeout_s=timeout_s,
        env_extra=env_extra,
    )
    trial = _run_child(
        probe_id,
        variant_id=variant_id,
        root=trial_root,
        is_control=False,
        payload=payload,
        timeout_s=timeout_s,
        env_extra=env_extra,
    )

    verdict, criteria, detail = derive_verdict(
        control=control,
        trial=trial,
        variant_id=variant_id,
        variant_segment=v.segment,
    )

    cpay = control.get("payload") or {}
    control_verdict = (
        Verdict.PASS.value if cpay.get("ok") else Verdict.ERROR_HARNESS.value
    )

    return ProbeResult(
        boundary_id=boundary_id or probe_id,
        probe_id=probe_id,
        variant=variant_id,
        apply_to=apply_to,
        tier=spec.tier if spec else "cheap",
        evidence_kind="runtime",
        verdict=verdict,
        control_verdict=control_verdict,
        exit_code=trial.get("exit_code"),
        timed_out=bool(trial.get("timed_out")),
        duration_s=round(float(trial.get("duration_s") or 0.0), 3),
        exception_type=detail.get("exception_type"),
        exception_message=detail.get("exception_message"),
        error_mentions_path=bool(detail.get("error_mentions_path")),
        observation=detail.get("observation"),
        control_observation=detail.get("control_observation"),
        silent_criteria_hit=criteria,
        skipped_reason=detail.get("skipped_reason"),
        # control / trial のどちらかが JSON を返せなかったときだけ診断を残す。
        # 「90 秒 timeout した」以上のことが分からない状態を作らないため。
        worker_diagnostics=_collect_worker_diagnostics(control, trial),
        notes=str(detail.get("why") or ""),
    )


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MANGLER_SIGNATURES",
    "HarnessError",
    "derive_verdict",
    "run_probe",
]
