# livecap-gui リアルタイム文字起こし実装分析

> **目的:** livecap-cli の Phase 1 実装に向けた参考資料
> **作成日:** 2025-11-25
> **対象:** livecap-gui (Live_Cap_v3) のリアルタイム文字起こし機能

---

## 1. アーキテクチャ概要

### 1.1 全体構造

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LiveTranscriber                               │
│  (src/live_transcribe.py)                                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐  │
│  │ AudioSource │ ──▶ │   VAD       │ ──▶ │   ASR Engine        │  │
│  │             │     │ Processing  │     │   (transcribe)      │  │
│  └─────────────┘     └─────────────┘     └─────────────────────┘  │
│        │                   │                       │               │
│        ▼                   ▼                       ▼               │
│  audio_queue         transcribe_queue      TranscriptionResult    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 スレッドモデル

| スレッド | 役割 | 主要メソッド |
|---------|------|-------------|
| **Main Thread** | マイク入力（sounddevice callback）| `audio_callback()` |
| **Audio Processor** | VAD処理、音声セグメント検出 | `process_audio()` |
| **Transcriber** | ASRエンジンで文字起こし | `process_transcription()` |

### 1.3 データフロー

```
マイク/システム音声
       │
       ▼
┌──────────────┐
│ audio_queue  │ ← float32, chunk_size samples
└──────────────┘
       │
       ▼  (AudioProcessor スレッド)
┌──────────────────────────────────────┐
│  VAD Processing                       │
│  ├── リサンプリング (→16kHz)           │
│  ├── TenVAD で音声検出                 │
│  └── ステートマシンで発話区間検出        │
└──────────────────────────────────────┘
       │
       ▼
┌────────────────┐
│transcribe_queue│ ← {audio: np.ndarray, is_final: bool, ...}
└────────────────┘
       │
       ▼  (Transcriber スレッド)
┌──────────────────────────────────────┐
│  ASR Engine (engine.transcribe())    │
│  ├── 投機的実行結果の利用（あれば）      │
│  └── 通常の文字起こし                  │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Callbacks                           │
│  ├── on_progress (中間結果)           │
│  └── on_finished (確定結果)           │
└──────────────────────────────────────┘
```

---

## 2. 主要コンポーネント詳細

### 2.1 LiveTranscriber クラス

**ファイル:** `src/live_transcribe.py`

リアルタイム文字起こしの中核クラス。音声キャプチャ、VAD処理、ASRエンジン呼び出しを統合。

#### コンストラクタパラメータ

```python
class LiveTranscriber:
    def __init__(
        self,
        device=None,              # cuda/cpu
        sample_rate=16000,        # サンプリングレート
        chunk_duration=0.5,       # チャンク長（秒）
        config=None,              # 設定辞書
        engine_type=None,         # ASRエンジンタイプ
        source_id=None,           # ソースID（マルチソース対応）
        on_progress=None,         # 中間結果コールバック
        on_status=None,           # ステータスコールバック
        on_finished=None,         # 完了コールバック
        on_error=None,            # エラーコールバック
    ):
```

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `start(input_device)` | 文字起こし開始 |
| `stop()` | 文字起こし停止 |
| `audio_callback()` | マイク音声受信コールバック |
| `process_audio()` | VAD処理ループ |
| `process_transcription()` | 文字起こしループ |

---

### 2.2 VAD システム

**ファイル:** `src/vad/stream/stream_vad_processor.py`

#### VAD ステートマシン

```
┌──────────┐  音声検出   ┌─────────────────┐  min_speech達成  ┌─────────────────┐
│ SILENCE  │ ──────────▶ │ POTENTIAL_SPEECH│ ───────────────▶ │ CONFIRMED_SPEECH│
└──────────┘             └─────────────────┘                  └─────────────────┘
     ▲                          │                                    │
     │                   タイムアウト                            無音検出
     │                          │                                    │
     │                          ▼                                    ▼
     │                     (破棄)                           ┌───────────────┐
     │                                                      │ ENDING_SPEECH │
     │                                                      └───────────────┘
     │                                                             │
     │                                                    パディング完了
     │                                                             │
     └─────────────────────────────────────────────────────────────┘
                              (セグメント出力)
```

#### VADState 列挙型

```python
class VADState(Enum):
    SILENCE = auto()           # 無音状態
    POTENTIAL_SPEECH = auto()  # 音声の可能性がある状態（検証中）
    CONFIRMED_SPEECH = auto()  # 確定した音声状態
    ENDING_SPEECH = auto()     # 音声終了処理中
```

#### StreamVADConfig

```python
@dataclass
class StreamVADConfig(BaseVADConfig):
    # 基本設定
    sample_rate: int = 16000
    frame_duration_ms: float = 16          # TenVADチャンクサイズ固定
    vad_threshold: float = 0.5

    # 時間設定（ミリ秒）
    vad_min_speech_duration_ms: float = 250    # 音声確定に必要な最小時間
    vad_max_speech_duration_ms: float = 0      # 0 = 無制限
    potential_speech_max_duration_ms: float = 1000

    # フレーム数設定
    potential_speech_timeout_frames: int = 10  # 160ms
    speech_end_threshold_frames: int = 12      # 192ms
    post_speech_padding_frames: int = 18       # 288ms

    # パディング設定（ミリ秒）
    pre_speech_padding_ms: float = 300
    post_speech_padding_ms: float = 300

    # 中間結果設定
    intermediate_result_min_duration_s: float = 2.0  # 中間結果送信の最小蓄積時間
    intermediate_result_interval_s: float = 1.0      # 中間結果送信の最小間隔
```

#### VAD処理の流れ

1. **音声チャンク受信** (16ms @16kHz = 256 samples)
2. **TenVAD処理** (`vad_model.process()`) → 確率値 + 音声フラグ
3. **ステートマシン更新** (`state_machine.process_frame()`)
4. **セグメント完成時** → `transcribe_queue` に追加

---

### 2.3 投機的実行 (Speculative Transcription)

**ファイル:** `src/vad/speculative_transcriber.py`

ENDING_SPEECH 状態に入った時点で、確定前のセグメントの文字起こしを先行して開始する仕組み。

#### 動作原理

```
CONFIRMED_SPEECH → 無音検出 → ENDING_SPEECH
                                    │
                                    ├── 投機的実行開始
                                    │   (バックグラウンドで文字起こし)
                                    │
                              パディング完了
                                    │
                                    ▼
                              投機的結果利用可能？
                                 │       │
                               Yes      No
                                 │       │
                              結果を   通常の
                              即返却   文字起こし
```

#### メリット

- 実質的なレイテンシ削減（最大300ms程度）
- ポストスピーチパディング中の時間を有効活用

---

### 2.4 AudioSource 抽象化

**ファイル:** `src/audio/sources/base_source.py`

```python
class AudioSource(ABC):
    """すべての音声ソースの基底クラス"""

    def __init__(self, source_id: str, config: Dict[str, Any]):
        self.source_id = source_id
        self.config = config
        self.is_active = False
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.chunk_duration = config.get('chunk_duration', 0.5)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

    @abstractmethod
    def start(self) -> bool:
        """音声キャプチャ開始"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """音声キャプチャ停止"""
        pass

    @abstractmethod
    def read_chunk(self, timeout: float = None) -> Optional[np.ndarray]:
        """音声チャンク読み取り"""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """ソース情報取得"""
        pass
```

#### 実装クラス

| クラス | ファイル | 説明 |
|--------|---------|------|
| `MicrophoneSource` | `microphone_source.py` | sounddevice ベースのマイク入力 |
| `PyWACSource` | `pywac_source.py` | Windows プロセス音声キャプチャ |

---

### 2.5 TranscriptionResult

**ファイル:** `src/transcription/events.py`

```python
@dataclass(slots=True)
class TranscriptionProgress:
    """中間結果用"""
    current: int
    total: int
    status: str = ""
    context: Optional[Dict[str, Any]] = None  # is_final, text, source_id など


@dataclass(slots=True)
class TranscriptionResult:
    """確定結果用"""
    text: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # timestamp, source_id, is_final
```

---

### 2.6 BaseEngine

**ファイル:** `src/engines/base_engine.py`

```python
class BaseEngine(ABC):
    @abstractmethod
    def transcribe(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> TranscriptionResult:
        """
        音声データを文字起こし

        Returns:
            TranscriptionResult (text / confidence / engine_confidence)
        """
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        pass

    @abstractmethod
    def get_supported_languages(self) -> list:
        pass

    @abstractmethod
    def get_required_sample_rate(self) -> int:
        pass

    def load_model(self) -> None:
        """6段階フェーズでモデルロード（GUI向け）"""
        pass

    def cleanup(self) -> None:
        pass
```

---

## 3. 重要な実装詳細

### 3.1 リサンプリング

TenVAD は 16kHz 固定のため、異なるサンプリングレートの音声はリサンプリングが必要。

```python
# 高速リサンプリング（scipy）
if self.sample_rate == 48000 and self.vad_sample_rate == 16000:
    self.resample_up = 1
    self.resample_down = 3  # 48000/16000 = 3

vad_chunk_resampled = signal.resample_poly(
    vad_chunk_original,
    up=self.resample_up,
    down=self.resample_down
)
```

### 3.2 RingBuffer

効率的な音声バッファ管理のための循環バッファ実装。

```python
class RingBuffer:
    def __init__(self, maxsize: int):
        self.data = np.zeros(maxsize, dtype=np.float32)
        self.maxsize = maxsize
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0

    def extend(self, data: np.ndarray):
        """データ追加（オーバーフロー時は古いデータを上書き）"""
        ...

    def get_chunk(self, size: int) -> Optional[np.ndarray]:
        """指定サイズのチャンク取得"""
        ...
```

### 3.3 ノイズゲート

オプションのノイズゲート処理。livecap-cli #278 / #279 以降は `livecap_cli.audio.NoiseGate` として本体に統合済み（numba JIT 実装）。詳細は [API リファレンス](api.md#noisegate) を参照。

```python
from livecap_cli.audio import NoiseGate

# 推奨閾値は livecap-cli levels コマンドまたは analyze_noise_samples() で算出
# 他のパラメータ (close_threshold_db / release_ms / noise_floor_db 等) は既定値で問題なし
self.noise_gate = NoiseGate(threshold_db=-35)

# audio_callback内で適用
if self.noise_gate:
    processed_data = self.noise_gate.process(flat_data)
```

> **GUI 較正キャリブレーション** (issue [#291]): GUI 側で独自に閾値を算出している場合は、`analyze_noise_samples(rms_db_list, peak_db_list)` (新 API; per-chunk RMS と peak を両方渡す) に委譲すべきです。これにより NoiseGate の envelope follower と単位が揃い、無音時 hallucination の root-cause が解消されます。`AudioLevelWorker` 等で per-chunk RMS と一緒に `|x|.max()` を emit すれば本 API を直接呼べます。

---

## 4. livecap-cli への適用方針

### 4.1 採用すべき設計

1. **スレッドモデル**: Audio/Transcription の2スレッド分離
2. **VADステートマシン**: 4状態モデル（SILENCE/POTENTIAL_SPEECH/CONFIRMED_SPEECH/ENDING_SPEECH）
3. **投機的実行**: レイテンシ削減に有効
4. **AudioSource抽象化**: MicrophoneSource, FileSource の基底クラス

### 4.2 簡素化すべき部分

1. **TranscriptionResult**:
   - livecap-gui: `segments` + `metadata` で二重構造
   - livecap-cli: 単一の `TranscriptionResult` dataclass に統一

2. **BaseEngine**:
   - livecap-gui: 6段階フェーズ管理（GUI進捗表示向け）
   - livecap-cli: シンプルな `load_model()` / `transcribe()` インターフェース

3. **Config**:
   - livecap-gui: GUI設定が混在
   - livecap-cli: コア設定のみ

### 4.3 livecap-cli での新設計案

```python
# TranscriptionResult（統一型）
@dataclass
class TranscriptionResult:
    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"

# StreamTranscriber（新API）
class StreamTranscriber:
    def __init__(self, engine: BaseEngine, vad: Optional[VADProcessor] = None):
        ...

    def feed_audio(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> None:
        """音声チャンク入力（ノンブロッキング）"""
        ...

    def get_result(self) -> Optional[TranscriptionResult]:
        """確定結果取得"""
        ...

    def get_interim(self) -> Optional[TranscriptionResult]:
        """中間結果取得"""
        ...

    async def transcribe_stream(
        self,
        audio_source: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[TranscriptionResult]:
        """非同期ストリーム処理"""
        ...
```

---

## 5. 参照ファイル一覧

| ファイル | 説明 |
|---------|------|
| `src/live_transcribe.py` | リアルタイム文字起こしメインクラス |
| `src/vad/__init__.py` | VADモジュールエクスポート |
| `src/vad/stream/stream_vad_processor.py` | ストリームVADステートマシン |
| `src/vad/speculative_transcriber.py` | 投機的実行マネージャー |
| `src/audio/sources/base_source.py` | AudioSource抽象基底クラス |
| `src/audio/sources/microphone_source.py` | マイク入力実装 |
| `src/audio/sources/pywac_source.py` | プロセス音声キャプチャ |
| `src/transcription/events.py` | TranscriptionResult等の型定義 |
| `src/engines/base_engine.py` | ASRエンジン基底クラス |

---

## 6. 関連ドキュメント

- [refactoring-plan.md](../planning/refactoring-plan.md) - リファクタリング計画
- [feature-inventory.md](./feature-inventory.md) - 現在の機能一覧
- Issue #69 - [Phase1] リアルタイム文字起こし実装
