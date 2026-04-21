"""リアルタイムノイズゲート

サンプル単位のエンベロープフォロワーにより、環境ノイズを減衰させる。
VAD の前段処理として使用し、ノイズによる VAD 誤検出を防止する。

livecap-gui v2 の RealtimeNoiseGate から移植。numba で高速化。
"""

from __future__ import annotations

import logging
import math

import numba
import numpy as np

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _process_loop(
    audio: np.ndarray,
    output: np.ndarray,
    envelope: float,
    gate_open: int,
    release_counter: int,
    open_threshold: float,
    close_threshold: float,
    attack_coeff: float,
    release_coeff: float,
    release_samples: int,
    noise_floor: float,
) -> tuple[float, int, int]:
    """numba JIT コンパイルされたノイズゲート処理ループ。

    サンプル単位の逐次処理でエンベロープフォロワーとゲート判定を行う。
    状態管理のためベクトル化ではなくループ処理を採用。

    ヒステリシス:
    - ``open_threshold``: 閉じた状態からゲートを開くには envelope がこれを超える必要がある。
    - ``close_threshold``: 開いた状態でゲートを閉じ始めるには envelope がこれを下回る必要がある。
    - ``open_threshold == close_threshold`` の場合は単一閾値挙動 (ヒステリシスなし)。
    """
    for i in range(len(audio)):
        abs_sample = abs(audio[i])

        # エンベロープフォロワー（音声の包絡線を追跡）
        if abs_sample > envelope:
            # アタック（エンベロープを上昇）
            envelope += attack_coeff * (abs_sample - envelope)
        else:
            # リリース（エンベロープを減衰）
            envelope *= 1.0 - release_coeff

        # ゲート判定（ヒステリシス + リリースカウンターでホールド時間）
        if gate_open:
            # 開いた状態では close_threshold を判定条件に使う
            if envelope > close_threshold:
                # ヒステリシス band 内または上 → hold (counter リセット)
                release_counter = release_samples
            else:
                # close_threshold を下回った → リリース開始
                if release_counter > 0:
                    release_counter -= 1
                else:
                    gate_open = 0
        else:
            # 閉じた状態では open_threshold を超えないと開かない
            if envelope > open_threshold:
                gate_open = 1
                release_counter = release_samples

        # 出力（ゲートが閉じている時は noise_floor レベルに減衰; 0.0 なら hard-mute）
        if gate_open:
            output[i] = audio[i]
        else:
            output[i] = audio[i] * noise_floor

    return envelope, gate_open, release_counter


class NoiseGate:
    """リアルタイムノイズゲート

    サンプル単位のエンベロープフォロワーにより、
    環境ノイズを減衰させる。VAD の前段処理として使用。
    numba JIT で高速化（< 0.1ms/100ms chunk）。
    """

    def __init__(
        self,
        threshold_db: float = -35,
        close_threshold_db: float | None = None,
        attack_ms: float = 0.5,
        release_ms: float = 100,
        sample_rate: int = 16000,
        noise_floor_db: float = float("-inf"),
    ) -> None:
        """NoiseGate を初期化。

        Args:
            threshold_db: ゲートを開く閾値 (dB, ``-80`` ～ ``0``)。
            close_threshold_db: ヒステリシス閉鎖閾値 (dB)。
                ``None`` 既定では ``threshold_db - 6 dB`` に自動解決される。
                単一閾値挙動 (ヒステリシスなし) が欲しい場合は
                ``close_threshold_db=threshold_db`` を明示的に指定する。
            attack_ms: アタック時間 (ms, ``0.1`` ～ ``100``)。
            release_ms: リリース時間 (ms, ``1`` ～ ``1000``)。既定 ``100`` は
                hard-mute + auto hysteresis との組み合わせで whisper 系
                エンジンの fragmentation ハルシネーションを抑制する値
                ([Issue #283] A/B 実測根拠)。短い値を望む場合は明示的に
                ``release_ms=30`` 等を指定する (旧 PR #282 挙動)。
            sample_rate: サンプリングレート (Hz)。
            noise_floor_db: ゲート閉鎖時の出力減衰 (dB)。既定 ``float("-inf")``
                は hard-mute (出力完全ゼロ)。``-60`` のような有限値を明示的に
                指定すると soft-mute になる (旧挙動の再現)。
                有限値の範囲: ``-120`` ～ ``0``。
        """
        # パラメータ検証（v2 から移植）
        if not -80 <= threshold_db <= 0:
            logger.warning(
                "Invalid threshold %sdB, using -35dB", threshold_db
            )
            threshold_db = -35

        # close_threshold_db の解決 (None → auto hysteresis)
        if close_threshold_db is None:
            close_threshold_db = threshold_db - 6.0
        if close_threshold_db > threshold_db:
            logger.warning(
                "close_threshold_db (%sdB) > threshold_db (%sdB); "
                "clamping to threshold_db (no hysteresis)",
                close_threshold_db,
                threshold_db,
            )
            close_threshold_db = float(threshold_db)
        if close_threshold_db < -80:
            logger.warning(
                "close_threshold_db (%sdB) < -80; clamping to -80",
                close_threshold_db,
            )
            close_threshold_db = -80.0

        if not 0.1 <= attack_ms <= 100:
            logger.warning(
                "Invalid attack time %sms, using 0.5ms", attack_ms
            )
            attack_ms = 0.5

        if not 1 <= release_ms <= 1000:
            logger.warning(
                "Invalid release time %sms, using 100ms", release_ms
            )
            release_ms = 100

        # noise_floor_db の検証 + 線形値への変換
        if not math.isfinite(noise_floor_db):
            # -inf (hard-mute) or nan 等
            self._noise_floor = 0.0
            noise_floor_db_resolved: float | None = None  # for logging
        elif -120 <= noise_floor_db <= 0:
            self._noise_floor = 10 ** (noise_floor_db / 20)
            noise_floor_db_resolved = noise_floor_db
        else:
            logger.warning(
                "Invalid noise_floor_db %sdB; using -inf (hard-mute)",
                noise_floor_db,
            )
            self._noise_floor = 0.0
            noise_floor_db_resolved = None

        # 線形スレッショルド
        self._threshold = 10 ** (threshold_db / 20)  # open threshold (互換名)
        self._close_threshold = 10 ** (close_threshold_db / 20)
        self._attack_samples = max(1, int(attack_ms * sample_rate / 1000))
        self._release_samples = max(1, int(release_ms * sample_rate / 1000))

        # エンベロープフォロワーの係数を事前計算（指数平滑化）
        self._attack_coeff = 1.0 - np.exp(-1.0 / self._attack_samples)
        self._release_coeff = 1.0 - np.exp(-1.0 / self._release_samples)

        # ゲート状態
        self._envelope: float = 0.0
        self._gate_open: int = 0  # numba 互換のため int (0/1)
        self._release_counter: int = 0

        logger.info(
            "NoiseGate initialized: open=%.1fdB, close=%.1fdB, "
            "noise_floor=%s, attack=%.1fms, release=%.1fms",
            threshold_db,
            close_threshold_db,
            "-inf (hard-mute)"
            if noise_floor_db_resolved is None
            else f"{noise_floor_db_resolved:.1f}dB",
            attack_ms,
            release_ms,
        )

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """音声チャンクにノイズゲートを適用。

        Args:
            audio_chunk: 入力音声データ（float32, 1次元）

        Returns:
            ゲート処理後の音声データ
        """
        if len(audio_chunk) == 0:
            return audio_chunk

        output = np.empty_like(audio_chunk)

        self._envelope, self._gate_open, self._release_counter = _process_loop(
            audio_chunk,
            output,
            self._envelope,
            self._gate_open,
            self._release_counter,
            self._threshold,
            self._close_threshold,
            self._attack_coeff,
            self._release_coeff,
            self._release_samples,
            self._noise_floor,
        )

        return output

    def reset(self) -> None:
        """内部状態をリセット。"""
        self._envelope = 0.0
        self._gate_open = 0
        self._release_counter = 0
