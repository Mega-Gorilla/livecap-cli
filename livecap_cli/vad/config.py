"""VAD設定

すべての時間パラメータをミリ秒で統一。
Silero VAD v5/v6 のデフォルト値を基準に設定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class VADConfig:
    """
    VAD設定（すべてミリ秒単位で統一）

    Silero VAD v5/v6 のデフォルト値を基準に設定。

    Silero VAD パラメータ対応:
    - threshold → threshold (0.5)
    - min_speech_duration_ms → min_speech_ms (250)
    - min_silence_duration_ms → min_silence_ms (100)
    - speech_pad_ms → speech_pad_ms (100)
    - neg_threshold → neg_threshold (threshold - 0.15)

    参考: https://github.com/snakers4/silero-vad

    Usage:
        # デフォルト設定
        config = VADConfig()

        # カスタム設定
        config = VADConfig(
            threshold=0.6,
            min_speech_ms=300,
            min_silence_ms=150,
        )

        # 辞書から作成
        config = VADConfig.from_dict({'threshold': 0.6})
    """

    # 音声検出閾値（Silero: threshold）
    threshold: float = 0.5

    # ノイズ閾値（Silero: neg_threshold）
    # None の場合は threshold - 0.15 を使用
    neg_threshold: Optional[float] = None

    # 音声判定に必要な最小継続時間（Silero: min_speech_duration_ms）
    min_speech_ms: int = 250

    # 音声終了判定に必要な無音継続時間（Silero: min_silence_duration_ms）
    min_silence_ms: int = 100

    # 発話前後のパディング（Silero: speech_pad_ms）
    # Silero はデフォルト 30ms だが、livecap-cli では発話区間を
    # より確実に捕捉するため、長めの値を使用（livecap-gui は 300ms）
    # 注: 32ms フレームで最低3フレーム（96ms）を確保
    speech_pad_ms: int = 100

    # 最大発話時間（0 = 無制限）（Silero: max_speech_duration_s）
    max_speech_ms: int = 0

    # 中間結果送信設定（livecap-cli 独自）
    interim_min_duration_ms: int = 2000
    interim_interval_ms: int = 1000

    def get_neg_threshold(self) -> float:
        """有効な neg_threshold を返す"""
        if self.neg_threshold is not None:
            return self.neg_threshold
        return max(0.0, self.threshold - 0.15)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> VADConfig:
        """辞書から設定を作成"""
        return cls(
            threshold=config.get("threshold", 0.5),
            neg_threshold=config.get("neg_threshold"),
            min_speech_ms=config.get("min_speech_ms", 250),
            min_silence_ms=config.get("min_silence_ms", 100),
            speech_pad_ms=config.get("speech_pad_ms", 100),
            max_speech_ms=config.get("max_speech_ms", 0),
            interim_min_duration_ms=config.get("interim_min_duration_ms", 2000),
            interim_interval_ms=config.get("interim_interval_ms", 1000),
        )

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "threshold": self.threshold,
            "neg_threshold": self.neg_threshold,
            "min_speech_ms": self.min_speech_ms,
            "min_silence_ms": self.min_silence_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "max_speech_ms": self.max_speech_ms,
            "interim_min_duration_ms": self.interim_min_duration_ms,
            "interim_interval_ms": self.interim_interval_ms,
        }
