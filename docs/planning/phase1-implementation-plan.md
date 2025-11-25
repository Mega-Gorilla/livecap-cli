# Phase 1 実装計画: リアルタイム文字起こし

> **作成日:** 2025-11-25
> **ステータス:** 計画中
> **関連:** [livecap-gui-realtime-analysis.md](../reference/livecap-gui-realtime-analysis.md)
> **Issue:** #69

---

## 1. 設計原則

### 1.1 基本方針

| 原則 | 説明 |
|------|------|
| **シンプルさ優先** | 過度な抽象化を避け、必要最小限の設計 |
| **型安全性** | dataclass活用、辞書型の排除 |
| **テスタビリティ** | 小さなクラス、依存性注入 |
| **非同期ファースト** | asyncio ベースの API（同期APIも提供）|
| **既存コードとの共存** | 段階的な移行を可能に |

### 1.2 livecap-gui からの改善点

| 問題 | livecap-gui | livecap-cli |
|------|-------------|-------------|
| 責務過多 | LiveTranscriber 1000行超 | 役割ごとに分離 |
| 型安全性 | 辞書型多用 | dataclass/Protocol |
| VAD設定 | フレーム数/ms/秒 混在 | すべてミリ秒 |
| 結果型 | 二重構造 | 単一の TranscriptionResult |
| 投機的実行 | 複雑な6状態 | 簡素化または Phase 2 |

---

## 2. ディレクトリ構造

```
livecap_core/
├── __init__.py                    # 公開API追加
├── transcription/
│   ├── __init__.py
│   ├── result.py                  # TranscriptionResult [新規]
│   ├── stream.py                  # StreamTranscriber [新規]
│   └── file_pipeline.py           # 既存（変更なし）
├── vad/                           # [新規モジュール]
│   ├── __init__.py
│   ├── config.py                  # VADConfig
│   ├── processor.py               # VADProcessor
│   ├── state_machine.py           # VADStateMachine
│   └── backends/                  # VADバックエンド
│       ├── __init__.py            # VADBackend Protocol
│       └── silero.py              # SileroVAD（デフォルト）
├── audio_sources/                 # [新規モジュール]
│   ├── __init__.py
│   ├── base.py                    # AudioSource ABC
│   ├── microphone.py              # MicrophoneSource
│   └── file.py                    # FileSource
└── ...
```

---

## 3. コンポーネント詳細設計

### 3.1 TranscriptionResult（統一結果型）

**ファイル:** `livecap_core/transcription/result.py`

```python
"""統一された文字起こし結果型"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """
    文字起こし結果を表すイミュータブルなデータクラス

    リアルタイム・ファイル文字起こし両方で使用する統一型。
    """
    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"

    @property
    def duration(self) -> float:
        """セグメントの長さ（秒）"""
        return self.end_time - self.start_time

    def to_srt_entry(self, index: int) -> str:
        """SRT形式のエントリに変換"""
        def format_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        return (
            f"{index}\n"
            f"{format_time(self.start_time)} --> {format_time(self.end_time)}\n"
            f"{self.text}\n"
        )


@dataclass(frozen=True, slots=True)
class InterimResult:
    """
    中間結果（確定前の途中経過）

    TranscriptionResultとは別の型として明示的に区別。
    """
    text: str
    accumulated_time: float
    source_id: str = "default"
```

**設計ポイント:**
- `frozen=True`: イミュータブル（スレッドセーフ）
- `slots=True`: メモリ効率化
- `InterimResult`: 中間結果を別の型として明示的に区別

---

### 3.2 VADConfig（VAD設定）

**ファイル:** `livecap_core/vad/config.py`

```python
"""VAD設定"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class VADConfig:
    """
    VAD設定（すべてミリ秒単位で統一）

    Silero VAD v5/v6 のデフォルト値を基準に設定。

    Silero VAD パラメータ対応:
    - threshold → threshold (0.5)
    - min_speech_duration_ms → min_speech_ms (250)
    - min_silence_duration_ms → min_silence_ms (100)
    - speech_pad_ms → speech_pad_ms (30)
    - neg_threshold → neg_threshold (threshold - 0.15)

    参考: https://github.com/snakers4/silero-vad
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

    @classmethod
    def from_dict(cls, config: dict) -> VADConfig:
        """辞書から設定を作成"""
        return cls(
            threshold=config.get('threshold', 0.5),
            neg_threshold=config.get('neg_threshold'),
            min_speech_ms=config.get('min_speech_ms', 250),
            min_silence_ms=config.get('min_silence_ms', 100),
            speech_pad_ms=config.get('speech_pad_ms', 100),
            max_speech_ms=config.get('max_speech_ms', 0),
            interim_min_duration_ms=config.get('interim_min_duration_ms', 2000),
            interim_interval_ms=config.get('interim_interval_ms', 1000),
        )

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            'threshold': self.threshold,
            'neg_threshold': self.neg_threshold,
            'min_speech_ms': self.min_speech_ms,
            'min_silence_ms': self.min_silence_ms,
            'speech_pad_ms': self.speech_pad_ms,
            'max_speech_ms': self.max_speech_ms,
            'interim_min_duration_ms': self.interim_min_duration_ms,
            'interim_interval_ms': self.interim_interval_ms,
        }
```

---

### 3.3 VADStateMachine（VADステートマシン）

**ファイル:** `livecap_core/vad/state_machine.py`

```python
"""VADステートマシン（簡素化版）"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import time
import numpy as np

from .config import VADConfig


class VADState(Enum):
    """VAD状態"""
    SILENCE = auto()           # 無音
    POTENTIAL_SPEECH = auto()  # 音声の可能性（検証中）
    SPEECH = auto()            # 確定した音声
    ENDING = auto()            # 音声終了処理中


@dataclass(slots=True)
class VADSegment:
    """検出された音声セグメント"""
    audio: np.ndarray
    start_time: float
    end_time: float
    is_final: bool


class VADStateMachine:
    """
    VADステートマシン

    4状態の遷移を管理し、音声セグメントを検出する。
    livecap-guiの実装を簡素化。
    """

    # Silero VAD v5+ のフレーム設定
    # 512 samples (32ms @ 16kHz) 固定
    FRAME_MS: int = 32
    FRAME_SAMPLES: int = 512  # 32ms @ 16kHz

    def __init__(self, config: VADConfig):
        self.config = config
        self._state = VADState.SILENCE
        self._state_entry_time = time.time()

        # フレームカウンタ（ミリ秒から変換）
        self._speech_frames = 0
        self._silence_frames = 0

        # しきい値（フレーム数に変換）
        self._min_speech_frames = config.min_speech_ms // self.FRAME_MS
        self._min_silence_frames = config.min_silence_ms // self.FRAME_MS
        # Silero VAD は speech_pad_ms を前後両方のパディングに使用
        # 最低1フレームを保証（100ms // 32ms = 3フレーム）
        self._padding_frames = max(1, config.speech_pad_ms // self.FRAME_MS)

        # バッファ
        self._pre_buffer: list[np.ndarray] = []
        self._speech_buffer: list[np.ndarray] = []
        self._segment_start_time: float = 0.0

        # 中間結果管理
        self._last_interim_time = 0.0
        self._interim_frame_count = 0

    @property
    def state(self) -> VADState:
        return self._state

    def process_frame(
        self,
        audio_frame: np.ndarray,
        probability: float,
        timestamp: float,
    ) -> Optional[VADSegment]:
        """
        1フレーム（32ms）を処理

        Args:
            audio_frame: 音声データ（float32, 512 samples @ 16kHz）
            probability: VAD確率（0.0-1.0）
            timestamp: 現在のタイムスタンプ

        Returns:
            完成したセグメント（なければNone）
        """
        is_speech = probability >= self.config.threshold

        # 状態別処理
        if self._state == VADState.SILENCE:
            return self._handle_silence(audio_frame, is_speech, timestamp)
        elif self._state == VADState.POTENTIAL_SPEECH:
            return self._handle_potential_speech(audio_frame, is_speech, timestamp)
        elif self._state == VADState.SPEECH:
            return self._handle_speech(audio_frame, is_speech, timestamp)
        elif self._state == VADState.ENDING:
            return self._handle_ending(audio_frame, is_speech, timestamp)

        return None

    def _handle_silence(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """SILENCE状態の処理"""
        # プリバッファに追加（パディング用）
        self._pre_buffer.append(frame)
        if len(self._pre_buffer) > self._padding_frames:
            self._pre_buffer.pop(0)

        if is_speech:
            self._transition_to(VADState.POTENTIAL_SPEECH)
            self._speech_frames = 1
            self._segment_start_time = timestamp - (len(self._pre_buffer) * self.FRAME_MS / 1000)
            # プリバッファをスピーチバッファにコピー
            self._speech_buffer = list(self._pre_buffer)
            self._speech_buffer.append(frame)

        return None

    def _handle_potential_speech(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """POTENTIAL_SPEECH状態の処理"""
        self._speech_buffer.append(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0

            if self._speech_frames >= self._min_speech_frames:
                self._transition_to(VADState.SPEECH)
        else:
            self._silence_frames += 1

            # タイムアウト: 音声として確定せず破棄
            if self._silence_frames >= self._min_silence_frames:
                self._transition_to(VADState.SILENCE)
                self._reset_buffers()

        return None

    def _handle_speech(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """SPEECH状態の処理"""
        self._speech_buffer.append(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

            if self._silence_frames >= self._min_silence_frames:
                self._transition_to(VADState.ENDING)

        # 最大発話時間チェック
        if self.config.max_speech_ms > 0:
            total_ms = len(self._speech_buffer) * self.FRAME_MS
            if total_ms >= self.config.max_speech_ms:
                return self._finalize_segment(timestamp, is_final=True)

        # 中間結果チェック
        return self._check_interim(timestamp)

    def _handle_ending(
        self, frame: np.ndarray, is_speech: bool, timestamp: float
    ) -> Optional[VADSegment]:
        """ENDING状態の処理（ポストパディング）"""
        self._speech_buffer.append(frame)

        if is_speech:
            # 音声再開 → SPEECH に戻る
            self._transition_to(VADState.SPEECH)
            self._silence_frames = 0
            return None

        self._silence_frames += 1

        if self._silence_frames >= self._padding_frames:
            return self._finalize_segment(timestamp, is_final=True)

        return None

    def _transition_to(self, new_state: VADState) -> None:
        """状態遷移"""
        self._state = new_state
        self._state_entry_time = time.time()
        if new_state == VADState.SILENCE:
            self._speech_frames = 0
            self._silence_frames = 0

    def _reset_buffers(self) -> None:
        """バッファリセット"""
        self._speech_buffer = []
        self._pre_buffer = []
        self._last_interim_time = 0.0
        self._interim_frame_count = 0

    def _check_interim(self, timestamp: float) -> Optional[VADSegment]:
        """中間結果の送信チェック"""
        current_time = time.time()
        frames_since_last = len(self._speech_buffer) - self._interim_frame_count
        duration_ms = frames_since_last * self.FRAME_MS

        if (duration_ms >= self.config.interim_min_duration_ms and
            (current_time - self._last_interim_time) * 1000 >= self.config.interim_interval_ms):

            self._last_interim_time = current_time
            self._interim_frame_count = len(self._speech_buffer)

            return VADSegment(
                audio=np.concatenate(self._speech_buffer),
                start_time=self._segment_start_time,
                end_time=timestamp,
                is_final=False,
            )

        return None

    def _finalize_segment(self, timestamp: float, is_final: bool) -> VADSegment:
        """セグメントを確定"""
        segment = VADSegment(
            audio=np.concatenate(self._speech_buffer) if self._speech_buffer else np.array([], dtype=np.float32),
            start_time=self._segment_start_time,
            end_time=timestamp,
            is_final=is_final,
        )

        self._transition_to(VADState.SILENCE)
        self._reset_buffers()

        return segment

    def finalize(self, timestamp: float) -> Optional[VADSegment]:
        """処理を終了し、残っているセグメントを返す"""
        if self._state in (VADState.SPEECH, VADState.ENDING, VADState.POTENTIAL_SPEECH):
            if self._speech_buffer:
                return self._finalize_segment(timestamp, is_final=True)
        return None
```

---

### 3.4 VADProcessor（VADプロセッサ）

**ファイル:** `livecap_core/vad/processor.py`

```python
"""VADプロセッサ"""

from __future__ import annotations
from typing import Optional, Protocol
import numpy as np
import logging

from .config import VADConfig
from .state_machine import VADStateMachine, VADSegment

logger = logging.getLogger(__name__)


class VADBackend(Protocol):
    """VADバックエンドのプロトコル"""
    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（512 samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        ...


class VADProcessor:
    """
    VADプロセッサ

    VADバックエンドとステートマシンを組み合わせて
    音声セグメントを検出する。
    """

    SAMPLE_RATE: int = 16000
    FRAME_SAMPLES: int = 512  # 32ms @ 16kHz (Silero VAD v5+)

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        backend: Optional[VADBackend] = None,
    ):
        self.config = config or VADConfig()
        self._backend = backend
        self._state_machine = VADStateMachine(self.config)
        self._current_time = 0.0

        # Silero VAD 初期化（バックエンドが指定されていない場合）
        if self._backend is None:
            self._backend = self._create_default_backend()

    def _create_default_backend(self) -> VADBackend:
        """デフォルトの Silero VAD バックエンドを作成"""
        try:
            from .backends.silero import SileroVAD
            return SileroVAD(threshold=self.config.threshold, onnx=True)
        except ImportError:
            raise ImportError(
                "silero-vad is required. Install with: pip install silero-vad"
            )

    def process_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[VADSegment]:
        """
        音声チャンクを処理

        Args:
            audio: 音声データ（float32）
            sample_rate: サンプリングレート

        Returns:
            検出されたセグメントのリスト
        """
        # リサンプリング（必要な場合）
        if sample_rate != self.SAMPLE_RATE:
            audio = self._resample(audio, sample_rate)

        segments: list[VADSegment] = []

        # フレーム単位で処理（512 samples = 32ms @ 16kHz）
        for i in range(0, len(audio) - self.FRAME_SAMPLES + 1, self.FRAME_SAMPLES):
            frame = audio[i:i + self.FRAME_SAMPLES]

            # VAD処理（Silero VAD v5+ は float32 を直接受け付ける）
            probability = self._backend.process(frame)

            # ステートマシン更新
            self._current_time += self.FRAME_SAMPLES / self.SAMPLE_RATE
            segment = self._state_machine.process_frame(
                audio_frame=frame,
                probability=probability,
                timestamp=self._current_time,
            )

            if segment is not None:
                segments.append(segment)

        return segments

    def finalize(self) -> Optional[VADSegment]:
        """処理を終了し、残っているセグメントを返す"""
        return self._state_machine.finalize(self._current_time)

    def reset(self) -> None:
        """状態をリセット"""
        self._state_machine = VADStateMachine(self.config)
        self._current_time = 0.0

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """リサンプリング"""
        from scipy import signal

        # 効率的な整数比リサンプリング
        if orig_sr == 48000:
            return signal.resample_poly(audio, up=1, down=3)
        elif orig_sr == 44100:
            return signal.resample_poly(audio, up=160, down=441)
        elif orig_sr == 32000:
            return signal.resample_poly(audio, up=1, down=2)
        else:
            # 汎用フォールバック
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.SAMPLE_RATE)

    @property
    def state(self):
        """現在のVAD状態"""
        return self._state_machine.state
```

---

### 3.5 SileroVAD バックエンド

**ファイル:** `livecap_core/vad/backends/silero.py`

```python
"""Silero VAD バックエンド"""

from __future__ import annotations
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD v5/v6 バックエンド

    VADBackend Protocol を実装。
    512 samples (32ms @ 16kHz) のチャンクを処理して確率を返す。
    """

    def __init__(self, threshold: float = 0.5, onnx: bool = True):
        """
        Args:
            threshold: 音声判定閾値（VADProcessor で使用、ここでは参考値）
            onnx: ONNX ランタイムを使用するか（推奨: True）
        """
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad(onnx=onnx)
            self._onnx = onnx
            logger.info(f"Silero VAD loaded (onnx={onnx})")
        except ImportError:
            raise ImportError(
                "silero-vad is required. Install with: pip install silero-vad"
            )

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（512 samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        import torch

        # numpy → torch tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio.astype(np.float32))

        # Silero VAD は (512,) の 1D tensor を期待
        if audio.dim() > 1:
            audio = audio.squeeze()

        return self.model(audio, 16000).item()

    def reset(self) -> None:
        """内部状態をリセット（新しい音声ストリーム開始時に呼ぶ）"""
        self.model.reset_states()
```

**ファイル:** `livecap_core/vad/backends/__init__.py`

```python
"""VAD バックエンド"""

from typing import Protocol
import numpy as np


class VADBackend(Protocol):
    """VADバックエンドのプロトコル"""

    def process(self, audio: np.ndarray) -> float:
        """
        音声を処理してVAD確率を返す

        Args:
            audio: float32形式の音声データ（512 samples @ 16kHz）

        Returns:
            probability (0.0-1.0)
        """
        ...

    def reset(self) -> None:
        """内部状態をリセット"""
        ...


# デフォルトエクスポート
from .silero import SileroVAD

__all__ = ["VADBackend", "SileroVAD"]
```

---

### 3.6 AudioSource（音声ソース）

**ファイル:** `livecap_core/audio_sources/base.py`

```python
"""音声ソース基底クラス"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional
import numpy as np


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """デバイス情報"""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


class AudioSource(ABC):
    """
    音声ソースの抽象基底クラス

    同期/非同期両方のイテレータインターフェースを提供。
    """

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 100):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @abstractmethod
    def start(self) -> None:
        """キャプチャ開始"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """キャプチャ停止"""
        pass

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        音声チャンクを読み取り（ブロッキング）

        Args:
            timeout: タイムアウト（秒）

        Returns:
            音声データ（float32, mono）またはNone
        """
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        """同期イテレータ"""
        if not self._is_active:
            self.start()
        try:
            while self._is_active:
                chunk = self.read(timeout=1.0)
                if chunk is not None:
                    yield chunk
        finally:
            self.stop()

    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        """非同期イテレータ"""
        import asyncio

        if not self._is_active:
            self.start()
        try:
            while self._is_active:
                # ブロッキング読み取りを別スレッドで実行
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.read(timeout=0.1)
                )
                if chunk is not None:
                    yield chunk
                else:
                    await asyncio.sleep(0.01)
        finally:
            self.stop()

    @classmethod
    @abstractmethod
    def list_devices(cls) -> list[DeviceInfo]:
        """利用可能なデバイス一覧"""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
```

**ファイル:** `livecap_core/audio_sources/microphone.py`

```python
"""マイク入力ソース"""

from __future__ import annotations
from typing import Optional
import queue
import numpy as np
import sounddevice as sd
import logging

from .base import AudioSource, DeviceInfo

logger = logging.getLogger(__name__)


class MicrophoneSource(AudioSource):
    """
    sounddeviceベースのマイク入力
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
    ):
        super().__init__(sample_rate=sample_rate, chunk_ms=chunk_ms)
        self.device_id = device_id
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream: Optional[sd.InputStream] = None

    def start(self) -> None:
        if self._is_active:
            return

        self._stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()
        self._is_active = True
        logger.info(f"MicrophoneSource started (device={self.device_id})")

    def stop(self) -> None:
        if not self._is_active:
            return

        self._is_active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # キューをクリア
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.info("MicrophoneSource stopped")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            data = self._queue.get(timeout=timeout)
            return data.flatten()
        except queue.Empty:
            return None

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.debug(f"Audio callback status: {status}")

        if self._is_active:
            try:
                self._queue.put_nowait(indata.copy())
            except queue.Full:
                # 古いデータを捨てる
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(indata.copy())
                except queue.Empty:
                    pass

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]:
        devices = []
        default_input = sd.default.device[0]

        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append(DeviceInfo(
                    index=i,
                    name=dev['name'],
                    channels=dev['max_input_channels'],
                    sample_rate=int(dev['default_samplerate']),
                    is_default=(i == default_input),
                ))

        return devices
```

**ファイル:** `livecap_core/audio_sources/file.py`

```python
"""ファイル音声ソース（テスト用）"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import logging

from .base import AudioSource, DeviceInfo

logger = logging.getLogger(__name__)


class FileSource(AudioSource):
    """
    ファイルからの音声ストリーム（テスト・デバッグ用）

    リアルタイム処理のシミュレーションに使用。
    """

    def __init__(
        self,
        file_path: Path | str,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        realtime: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, chunk_ms=chunk_ms)
        self.file_path = Path(file_path)
        self.realtime = realtime

        self._audio: Optional[np.ndarray] = None
        self._position = 0
        self._file_sample_rate = 0

    def start(self) -> None:
        if self._is_active:
            return

        # ファイル読み込み
        audio, self._file_sample_rate = sf.read(self.file_path, dtype='float32')

        # モノラル化
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # リサンプリング（必要な場合）
        if self._file_sample_rate != self.sample_rate:
            from scipy import signal
            audio = signal.resample(
                audio,
                int(len(audio) * self.sample_rate / self._file_sample_rate)
            )

        self._audio = audio
        self._position = 0
        self._is_active = True
        logger.info(f"FileSource started: {self.file_path}")

    def stop(self) -> None:
        self._is_active = False
        self._audio = None
        self._position = 0
        logger.info("FileSource stopped")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        if self._audio is None or self._position >= len(self._audio):
            self._is_active = False
            return None

        # チャンク取得
        end = min(self._position + self.chunk_size, len(self._audio))
        chunk = self._audio[self._position:end]
        self._position = end

        # リアルタイムシミュレーション
        if self.realtime and len(chunk) == self.chunk_size:
            import time
            time.sleep(self.chunk_ms / 1000)

        # 短いチャンクはパディング
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

        return chunk

    @classmethod
    def list_devices(cls) -> list[DeviceInfo]:
        return []  # ファイルソースにはデバイスがない
```

---

### 3.7 StreamTranscriber（ストリーミング文字起こし）

**ファイル:** `livecap_core/transcription/stream.py`

#### 設計決定事項（2025-11-25 議論）

**1. 既存エンジンとの統合:**
- 新たな Protocol 定義は不要
- 既存の `BaseEngine.transcribe(audio, sample_rate) -> (str, float)` をそのまま使用
- `BaseEngine.get_required_sample_rate()` でサンプルレートを取得

**2. スレッディングモデル: asyncio + run_in_executor**
```
┌─────────────────────────────────────────────────────────────┐
│  asyncio event loop (メインスレッド)                          │
│                                                             │
│  AudioSource ──async for──▶ VADProcessor ──▶ 結果キュー      │
│       │                         │                           │
│       │                         │ (軽量処理なのでメインスレッド) │
│       ▼                         ▼                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  ThreadPoolExecutor                                     ││
│  │  └── engine.transcribe() (ブロッキング、重い処理)         ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**3. VADProcessor 注入（テスタビリティ向上）:**
```python
def __init__(
    self,
    engine: BaseEngine,
    vad_config: Optional[VADConfig] = None,
    vad_processor: Optional[VADProcessor] = None,  # 追加
    source_id: str = "default",
): ...
```

**4. エラーハンドリング（Phase 1）:**
- 基本的な例外伝播と型付け
- 詳細なログ出力
- 自動リカバリは Phase 2 以降

**5. バッファリング:**
- VAD セグメント単位で即時処理（デフォルト）
- バッチ処理オプションは Phase 2 以降

```python
"""ストリーミング文字起こし"""

from __future__ import annotations
from typing import AsyncIterator, Iterator, Optional, Callable, Protocol, Tuple
import asyncio
import concurrent.futures
import queue
import logging
import numpy as np

from .result import TranscriptionResult, InterimResult
from ..vad import VADProcessor, VADConfig, VADSegment

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """文字起こしエラーの基底クラス"""
    pass


class EngineError(TranscriptionError):
    """エンジン関連のエラー"""
    pass


# エンジンプロトコル（既存のBaseEngineと互換）
class TranscriptionEngine(Protocol):
    """文字起こしエンジンのプロトコル"""
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        ...
    def get_required_sample_rate(self) -> int:
        ...


class StreamTranscriber:
    """
    ストリーミング文字起こし

    VADプロセッサとASRエンジンを組み合わせて
    リアルタイム文字起こしを行う。

    Args:
        engine: 文字起こしエンジン（BaseEngine互換）
        vad_config: VAD設定（vad_processor未指定時に使用）
        vad_processor: VADプロセッサ（テスト用に注入可能）
        source_id: 音声ソース識別子
        max_workers: 文字起こし用スレッド数（デフォルト: 1）
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        vad_config: Optional[VADConfig] = None,
        vad_processor: Optional[VADProcessor] = None,
        source_id: str = "default",
        max_workers: int = 1,
    ):
        self.engine = engine
        self.source_id = source_id
        self._sample_rate = engine.get_required_sample_rate()

        # VADプロセッサ（注入または新規作成）
        if vad_processor is not None:
            self._vad = vad_processor
        else:
            self._vad = VADProcessor(config=vad_config)

        # 文字起こし用スレッドプール
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # 結果キュー
        self._result_queue: queue.Queue[TranscriptionResult | InterimResult] = queue.Queue()

        # コールバック
        self._on_result: Optional[Callable[[TranscriptionResult], None]] = None
        self._on_interim: Optional[Callable[[InterimResult], None]] = None

    def set_callbacks(
        self,
        on_result: Optional[Callable[[TranscriptionResult], None]] = None,
        on_interim: Optional[Callable[[InterimResult], None]] = None,
    ) -> None:
        """コールバックを設定"""
        self._on_result = on_result
        self._on_interim = on_interim

    def feed_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """
        音声チャンクを入力（ノンブロッキング）

        結果は get_result() / get_interim() で取得するか、
        コールバックで受け取る。
        """
        # VAD処理
        segments = self._vad.process_chunk(audio, sample_rate)

        for segment in segments:
            if segment.is_final:
                result = self._transcribe_segment(segment)
                if result:
                    self._result_queue.put(result)
                    if self._on_result:
                        self._on_result(result)
            else:
                # 中間結果
                interim = self._transcribe_interim(segment)
                if interim:
                    self._result_queue.put(interim)
                    if self._on_interim:
                        self._on_interim(interim)

    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]:
        """確定結果を取得（ブロッキング）"""
        try:
            result = self._result_queue.get(timeout=timeout)
            if isinstance(result, TranscriptionResult):
                return result
            # InterimResultは無視して次を待つ
            return self.get_result(timeout=0.001) if timeout else None
        except queue.Empty:
            return None

    def get_interim(self) -> Optional[InterimResult]:
        """中間結果を取得（ノンブロッキング）"""
        try:
            result = self._result_queue.get_nowait()
            if isinstance(result, InterimResult):
                return result
            # TranscriptionResultは戻す
            self._result_queue.put(result)
            return None
        except queue.Empty:
            return None

    def finalize(self) -> Optional[TranscriptionResult]:
        """処理を終了し、残っているセグメントを文字起こし"""
        segment = self._vad.finalize()
        if segment and segment.is_final:
            return self._transcribe_segment(segment)
        return None

    def reset(self) -> None:
        """状態をリセット"""
        self._vad.reset()
        # キューをクリア
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break

    def _transcribe_segment(self, segment: VADSegment) -> Optional[TranscriptionResult]:
        """セグメントを文字起こし（同期）"""
        if len(segment.audio) == 0:
            return None

        try:
            text, confidence = self.engine.transcribe(segment.audio, self._sample_rate)

            if not text or not text.strip():
                return None

            return TranscriptionResult(
                text=text.strip(),
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    async def _transcribe_segment_async(self, segment: VADSegment) -> Optional[TranscriptionResult]:
        """セグメントを文字起こし（非同期、executor使用）"""
        if len(segment.audio) == 0:
            return None

        loop = asyncio.get_event_loop()
        try:
            text, confidence = await loop.run_in_executor(
                self._executor,
                self.engine.transcribe,
                segment.audio,
                self._sample_rate,
            )

            if not text or not text.strip():
                return None

            return TranscriptionResult(
                text=text.strip(),
                start_time=segment.start_time,
                end_time=segment.end_time,
                is_final=True,
                confidence=confidence,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Async transcription error: {e}", exc_info=True)
            raise EngineError(f"Transcription failed: {e}") from e

    def _transcribe_interim(self, segment: VADSegment) -> Optional[InterimResult]:
        """中間結果の文字起こし"""
        if len(segment.audio) == 0:
            return None

        try:
            text, _ = self.engine.transcribe(segment.audio, self._sample_rate)

            if not text or not text.strip():
                return None

            return InterimResult(
                text=text.strip(),
                accumulated_time=segment.end_time - segment.start_time,
                source_id=self.source_id,
            )
        except Exception as e:
            logger.error(f"Interim transcription error: {e}", exc_info=True)
            return None

    def close(self) -> None:
        """リソースを解放"""
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # === 高レベルAPI ===

    def transcribe_sync(
        self,
        audio_source,  # AudioSource
    ) -> Iterator[TranscriptionResult]:
        """
        同期ストリーム処理

        Args:
            audio_source: AudioSourceインスタンス

        Yields:
            TranscriptionResult
        """
        for chunk in audio_source:
            self.feed_audio(chunk, audio_source.sample_rate)

            while True:
                result = self.get_result(timeout=0)
                if result:
                    yield result
                else:
                    break

        # 最終セグメント
        final = self.finalize()
        if final:
            yield final

    async def transcribe_async(
        self,
        audio_source,  # AudioSource
    ) -> AsyncIterator[TranscriptionResult]:
        """
        非同期ストリーム処理

        VAD処理はメインスレッドで実行し、
        文字起こしは ThreadPoolExecutor で実行する。

        Args:
            audio_source: AudioSourceインスタンス

        Yields:
            TranscriptionResult
        """
        async for chunk in audio_source:
            # VAD処理は軽いのでメインスレッドで実行
            segments = self._vad.process_chunk(chunk, audio_source.sample_rate)

            for segment in segments:
                if segment.is_final:
                    # 文字起こしは executor で実行
                    result = await self._transcribe_segment_async(segment)
                    if result:
                        yield result
                # 中間結果は同期で処理（高速なため）
                elif self._on_interim:
                    interim = self._transcribe_interim(segment)
                    if interim:
                        self._on_interim(interim)

            # 他のタスクに制御を譲る
            await asyncio.sleep(0)

        # 最終セグメント
        final_segment = self._vad.finalize()
        if final_segment and final_segment.is_final:
            result = await self._transcribe_segment_async(final_segment)
            if result:
                yield result
```

---

## 4. 公開 API

**ファイル:** `livecap_core/__init__.py` に追加

```python
# Phase 1: リアルタイム文字起こし
from .transcription.result import TranscriptionResult, InterimResult
from .transcription.stream import StreamTranscriber
from .vad import VADProcessor, VADConfig
from .audio_sources import (
    AudioSource,
    MicrophoneSource,
    FileSource,
    DeviceInfo,
)

__all__ = [
    # 既存のエクスポート...

    # Phase 1 追加
    "TranscriptionResult",
    "InterimResult",
    "StreamTranscriber",
    "VADProcessor",
    "VADConfig",
    "AudioSource",
    "MicrophoneSource",
    "FileSource",
    "DeviceInfo",
]
```

---

## 5. 使用例

### 5.1 基本的な使い方

```python
from livecap_core import (
    StreamTranscriber,
    MicrophoneSource,
    VADConfig,
)
from engines import EngineFactory

# エンジン作成
engine = EngineFactory.create_engine("whispers2t_base", device="auto")
engine.load_model()

# StreamTranscriber作成
transcriber = StreamTranscriber(
    engine=engine,
    vad_config=VADConfig(
        min_speech_ms=250,
        min_silence_ms=200,
    ),
)

# マイクからリアルタイム文字起こし
with MicrophoneSource(device_id=0) as mic:
    for result in transcriber.transcribe_sync(mic):
        print(f"[{result.start_time:.2f}s] {result.text}")
```

### 5.2 非同期API

```python
import asyncio
from livecap_core import StreamTranscriber, MicrophoneSource

async def main():
    engine = EngineFactory.create_engine("whispers2t_base")
    engine.load_model()

    transcriber = StreamTranscriber(engine=engine)

    with MicrophoneSource() as mic:
        async for result in transcriber.transcribe_async(mic):
            print(f"{result.text}")

asyncio.run(main())
```

### 5.3 コールバック方式

```python
transcriber = StreamTranscriber(engine=engine)

transcriber.set_callbacks(
    on_result=lambda r: print(f"[確定] {r.text}"),
    on_interim=lambda r: print(f"[途中] {r.text}"),
)

with MicrophoneSource() as mic:
    for chunk in mic:
        transcriber.feed_audio(chunk, mic.sample_rate)
```

---

## 6. テスト計画

### 6.1 テストデータ

既存の `tests/assets/audio/` を使用。追加データは不要。

| ファイル | 言語 | サンプルレート | 長さ | 正解テキスト |
|---------|------|--------------|------|-------------|
| `jsut_basic5000_0001_ja.wav` | 日本語 | 16kHz | 3.19s | 水をマレーシアから買わなくてはならないのです。 |
| `librispeech_..._en.wav` | 英語 | 16kHz | 3.27s | STUFF IT INTO YOU HIS BELLY COUNSELLED HIM |

**利点:**
- 16kHz で VAD/エンジン要求レートと一致（リサンプリング不要）
- 正解テキスト（`.txt`）付きで精度検証可能
- 約3秒で VAD 状態遷移テストに適切

### 6.2 ユニットテスト

| テスト対象 | テストケース |
|-----------|-------------|
| `TranscriptionResult` | 生成、to_srt_entry、frozen性 |
| `VADConfig` | from_dict、to_dict、デフォルト値 |
| `VADStateMachine` | 状態遷移、各状態の処理 |
| `VADProcessor` | process_chunk、finalize |
| `FileSource` | start/stop、read、イテレータ |

### 6.3 統合テスト

| テストケース | 説明 |
|-------------|------|
| FileSource + VADProcessor | ファイルからVADセグメント検出 |
| FileSource + StreamTranscriber | ファイル文字起こしe2e |
| MicrophoneSource（手動） | 実マイクでの動作確認 |

```python
# 統合テスト例
def test_stream_transcriber_japanese():
    """日本語音声のストリーム文字起こし"""
    source = FileSource("tests/assets/audio/jsut_basic5000_0001_ja.wav", realtime=False)
    transcriber = StreamTranscriber(engine)

    results = list(transcriber.transcribe_sync(source))

    assert len(results) >= 1
    assert "マレーシア" in results[0].text  # 部分一致で検証
```

---

## 7. 実装順序

| ステップ | 内容 | 依存 |
|---------|------|------|
| 1 | `TranscriptionResult`, `InterimResult` | なし |
| 2 | `VADConfig` | なし |
| 3 | `VADStateMachine` | VADConfig |
| 4 | `VADProcessor` | VADStateMachine, Silero VAD |
| 5 | `AudioSource`, `FileSource` | なし |
| 6 | `MicrophoneSource` | AudioSource, sounddevice |
| 7 | `StreamTranscriber` | VADProcessor, エンジン |
| 8 | テスト、統合 | 全て |

---

## 8. 将来の拡張

### Phase 2 以降で検討

1. **投機的実行**: シンプルな実装を検討
2. **SystemAudioSource**: Windows/Linux システム音声キャプチャ
3. **翻訳統合**: TranscriptionResult に翻訳テキストを追加

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-11-25 | 初版作成 |
| 2025-11-25 | Silero VAD v5/v6 パラメータに更新（512 samples/32ms フレーム、speech_pad_ms 統一）|
| 2025-11-25 | テスト計画にテストデータ情報を追加 |
| 2025-11-25 | speech_pad_ms を 30→100ms に変更、SileroVAD バックエンド実装を追記 |
| 2025-11-25 | Step 4 設計決定事項を追加: スレッディングモデル、VADProcessor注入、エラー型定義 |
