"""プローブを**子プロセスで**実行する worker (Issue #378)。

``python -m tests.nonascii.worker`` として起動され、stdin から JSON の
実行仕様を受け取り、stdout にセンチネルで囲んだ JSON を 1 個だけ出す。

子プロセスで走らせる理由 (親でやってはいけない理由):

1. sherpa-onnx / NeMo のネイティブコードは ``abort()`` し得る。親で走らせると
   全証拠が消える。
2. **測定対象そのものがプロセス全体の env 書き換え**である。親の ``os.environ`` /
   ``tempfile.tempdir`` を触ると、``livecap_cli/utils/__init__.py`` の
   「ロック無し・refcount 無し」欠陥を測定ツール内で再現してしまう。
3. 汚染された global state は回復不能 — ``nemo_utils`` の ``NEMO_AVAILABLE``
   キャッシュ、``ModelMemoryCache`` の strong ref、ONNX mmap のファイルロック。
4. stdio エンコーディング (cp932 コンソール) を制御できるのは子プロセスだけ。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

SENTINEL = "---LIVECAP-PROBE-JSON---"


def _emit(payload: dict) -> None:
    """ネイティブライブラリの stdout 汚染に耐えるようセンチネルで囲む。

    ``ensure_ascii=True``: 非 ASCII パスを cp932 パイプへ書いても落ちないため。
    """
    sys.stdout.write("\n" + SENTINEL + "\n")
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, default=str))
    sys.stdout.write("\n" + SENTINEL + "\n")
    sys.stdout.flush()


def main() -> int:
    raw = sys.stdin.read()
    try:
        spec = json.loads(raw)
    except Exception as exc:  # pragma: no cover - 親のバグ
        _emit({"harness_error": f"仕様 JSON を読めない: {exc!r}"})
        return 2

    probe_id = spec["probe_id"]
    variant = spec["variant"]
    root = Path(spec["root"])
    is_control = bool(spec["is_control"])
    payload = spec.get("payload") or {}

    # livecap_cli の singleton は構築時に env を取り込むため、
    # env 注入後に必ずリセットする。probe 側で忘れられないようここで一括実行。
    try:
        from livecap_cli.resources import _reset_resources_for_tests

        _reset_resources_for_tests()
    except Exception:  # pragma: no cover - livecap を import しない probe もある
        pass

    from tests.nonascii.probes import load_all
    from tests.nonascii.record import ProbeContext, ProbeSkipped

    impls = load_all()
    impl = impls.get(probe_id)
    if impl is None:
        _emit({"harness_error": f"未登録の probe_id: {probe_id}"})
        return 2

    ctx = ProbeContext(
        probe_id=probe_id,
        variant=variant,
        root=root,
        is_control=is_control,
        payload=payload,
    )

    try:
        observation = impl(ctx)
        # **probe は必ず dict を返す。** None や Path を返すと observation が空のまま
        # control と trial で一致し、**境界を一度も通らずに pass になる**。
        # #387 PR B で実際に踏んだ — `@probe` デコレータが helper に付いてしまい、
        # probe 本体が一度も呼ばれていないのに 2 variant とも緑だった。
        # 境界のバグではないので、ハーネスの不整合として loud に落とす。
        if not isinstance(observation, dict):
            raise TypeError(
                f"probe {probe_id!r} が dict を返さなかった "
                f"({type(observation).__name__})。観測が空のまま control と一致し、"
                "**測っていないのに pass** になる"
            )
        _emit(
            {
                "ok": True,
                "observation": observation,
                "stages": list(ctx.stages),
            }
        )
        return 0
    except ProbeSkipped as skip:
        _emit({"ok": False, "skipped_reason": str(skip), "stages": list(ctx.stages)})
        return 0
    except BaseException as exc:  # noqa: BLE001 - 何が来ても証拠として残す
        _emit(
            {
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=12),
                "stages": list(ctx.stages),
            }
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
