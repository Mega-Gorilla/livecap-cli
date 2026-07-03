"""モデルロード前の VRAM 事前チェック (Issue #96)。

一部 engine (Voxtral ~9.5GB 等) は大容量 VRAM を要求し、 小容量 GPU では
モデルが部分ロードされた後に OOM crash するまで user が気づけない。 本 module
は load 前に利用可能 VRAM を検証し、 不足時に **警告** (default) / **早期 fail**
(strict opt-in) を出す。

方針 (Issue #96、 owner 確認済):

- **default: warn only** — ``translation/impl/riva_instruct.py`` の warn-only
  precedent と一致。 VRAM 要求値は粗い初期推定で、 ``get_available_vram()`` は
  free memory (悲観的) のため、 false-positive で正常 load をブロックしない安全側。
- **strict opt-in**: ``LIVECAP_STRICT_VRAM_CHECK=1`` (env var、 呼出側の
  ``base_engine.load_model()`` が読む) で ``InsufficientVRAMError`` を raise。
  CLI ``--strict-vram-check`` で env var を set。

多段 fail-open (check skip): 非 cuda device / 未知 engine / GPU なし。

**VRAM 要求値は粗い初期推定** (Issue #96 記載値ベース)。 #86 benchmark
(``BenchmarkMetrics.gpu_memory_peak_mb``) での精密化は別 PR。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# 粗い初期推定値 (MB)。 key は ``engine_name`` または ``engine_name:model_size``。
# size 概念を持つ engine (whispers2t) は size 別、 それ以外は engine 単位の代表値。
# Issue #96: #86 benchmark の実測 (gpu_memory_peak_mb) で精密化予定。
MODEL_VRAM_REQUIREMENTS_MB: dict[str, int] = {
    # WhisperS2T (size 別)
    "whispers2t:tiny": 1000,
    "whispers2t:base": 2000,
    "whispers2t:small": 3000,
    "whispers2t:medium": 5000,
    "whispers2t:large-v3": 10000,
    "whispers2t:large-v3-turbo": 6000,
    "whispers2t:distil-large-v3": 6000,
    # engine 単位 (代表値)
    "voxtral": 9500,
    "canary": 4000,
    "parakeet": 3000,
    "parakeet_ja": 3000,
    "reazonspeech": 2000,
    "qwen3asr": 4000,
}


class InsufficientVRAMError(RuntimeError):
    """要求モデルに対して VRAM が不足している場合に raise (strict mode のみ)。

    Attributes:
        required_gb: 要求 VRAM (GB)
        available_gb: 利用可能 VRAM (GB)
        engine_name: 対象 engine 名
    """

    def __init__(
        self,
        message: str,
        *,
        required_gb: float,
        available_gb: float,
        engine_name: str,
    ) -> None:
        super().__init__(message)
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.engine_name = engine_name


def get_vram_requirement_mb(
    engine_name: str, model_size: Optional[str] = None
) -> Optional[int]:
    """engine (+ model_size) の VRAM 要求推定値 (MB) を返す。

    ``engine_name:model_size`` の具体 key を優先し、 なければ ``engine_name``
    単位の代表値に fallback。 どちらも未登録なら ``None`` (= check skip)。
    """
    if model_size:
        specific = MODEL_VRAM_REQUIREMENTS_MB.get(f"{engine_name}:{model_size}")
        if specific is not None:
            return specific
    return MODEL_VRAM_REQUIREMENTS_MB.get(engine_name)


def check_vram_before_load(
    engine_name: str,
    model_size: Optional[str],
    device: Optional[str],
    *,
    strict: bool = False,
) -> None:
    """モデルロード前の VRAM 事前チェック (Issue #96)。

    Args:
        engine_name: ``BaseEngine.engine_name`` (normalized id)。
        model_size: engine が持つ場合の model size (``getattr(self, 'model_size', None)``)。
        device: ``BaseEngine.device`` (raw、 ``None``/``"auto"``/``"cuda"``/``"cpu"``)。
        strict: ``True`` で不足時に ``InsufficientVRAMError`` を raise。
            ``False`` (default) は ``logger.warning`` のみ。

    多段 fail-open (何もしない): 非 cuda device / 未知 engine / GPU なし /
    VRAM 十分。 実際に不足している時のみ warn または raise。
    """
    # 遅延 import (torch 依存 util、 module import 時の副作用を避ける)
    from livecap_cli.utils import detect_device, get_available_vram

    if detect_device(device, engine_name) != "cuda":
        return  # CPU engine は VRAM 無関係

    required_mb = get_vram_requirement_mb(engine_name, model_size)
    if required_mb is None:
        return  # 未知 engine/size → fail-open (安全側で skip)

    available_mb = get_available_vram()
    if available_mb is None:
        return  # GPU/torch なし (CTranslate2 等) → skip

    if available_mb >= required_mb:
        return  # 足りている

    required_gb = required_mb / 1024
    available_gb = available_mb / 1024
    size_suffix = f" ({model_size})" if model_size else ""
    message = (
        f"{engine_name}{size_suffix} は約 {required_gb:.1f}GB VRAM が必要ですが、"
        f" 利用可能なのは約 {available_gb:.1f}GB です。 OOM (メモリ不足) の可能性があります。"
    )
    if strict:
        raise InsufficientVRAMError(
            message,
            required_gb=required_gb,
            available_gb=available_gb,
            engine_name=engine_name,
        )
    logger.warning(
        "%s 停止させるには LIVECAP_STRICT_VRAM_CHECK=1 (or --strict-vram-check)。",
        message,
    )
