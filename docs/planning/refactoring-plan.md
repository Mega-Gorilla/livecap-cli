# LiveCap Core リファクタリング計画

> **作成日:** 2025-11-25
> **ステータス:** 計画中
> **関連:** [feature-inventory.md](../reference/feature-inventory.md)

---

## 1. 背景と目的

### 1.1 本リポジトリの目的

**可能な限り高速なリアルタイム文字起こしを、CLIでユーザーに提供すること。**

そのために：
- モデルのロード効率化
- 文字起こし（VAD含む）の効率化
- 低レイテンシなストリーミング処理

### 1.2 現状の課題

現在の設計は LiveCap-GUI からの分離抽出という経緯から、以下の問題を抱えている：

| 課題 | 影響度 | 詳細 |
|------|--------|------|
| リアルタイム文字起こし未実装 | 致命的 | 本来の目的機能が存在しない |
| パッケージ境界の不整合 | 高 | `engines/`, `config/` が分散 |
| API の二重構造 | 高 | イベント型とセグメント型が混在 |
| Config の肥大化 | 中 | GUI 由来の不要パラメータ |
| BaseEngine の過剰な複雑さ | 中 | GUI 向け6段階フェーズ管理 |
| 翻訳機能の未実装 | 中 | 型定義のみで実装なし |
| 依存関係の重さ | 中 | 基本インストールが重い |

---

## 2. リファクタリング方針

### 2.1 設計原則

1. **リアルタイム優先**: ストリーミング処理を第一級市民として設計
2. **ハードウェア抽象化**: コアは `np.ndarray` を受け取り、デバイス管理は外部
3. **プラグイン拡張性**: エンジン・翻訳サービスの追加が容易
4. **最小依存**: 基本機能は軽量、重い依存はオプショナル
5. **単一責務**: GUI 固有の機能はコアに含めない

### 2.2 ターゲットアーキテクチャ

```
livecap-core/
├── livecap_core/                  # 単一パッケージに統合
│   ├── __init__.py                # 公開 API
│   ├── cli.py                     # CLI エントリーポイント
│   │
│   ├── engines/                   # ASR エンジン（統合）
│   │   ├── __init__.py
│   │   ├── base.py                # シンプル化した基底クラス
│   │   ├── factory.py
│   │   ├── metadata.py
│   │   └── impl/                  # 各エンジン実装
│   │       ├── whisper.py
│   │       ├── reazonspeech.py
│   │       ├── parakeet.py
│   │       ├── canary.py
│   │       └── voxtral.py
│   │
│   ├── transcription/             # 文字起こしパイプライン
│   │   ├── __init__.py
│   │   ├── stream.py              # StreamTranscriber（新規・最重要）
│   │   ├── file.py                # FileTranscriber（既存を簡素化）
│   │   └── result.py              # 統一された結果型
│   │
│   ├── translation/               # 翻訳機能（新規）
│   │   ├── __init__.py
│   │   ├── base.py                # BaseTranslator
│   │   ├── factory.py
│   │   └── impl/
│   │       ├── google.py          # deep-translator
│   │       └── riva.py            # NVIDIA Riva
│   │
│   ├── vad/                       # VAD（新規モジュール化）
│   │   ├── __init__.py
│   │   ├── processor.py           # VADProcessor
│   │   └── buffer.py              # AudioBuffer
│   │
│   ├── audio_sources/             # オーディオソース（オプショナル）
│   │   ├── __init__.py
│   │   ├── base.py                # AudioSource ABC
│   │   ├── microphone.py          # sounddevice
│   │   ├── system.py              # システム音声キャプチャ
│   │   └── file.py                # ファイルストリーム
│   │
│   ├── config/                    # 設定（統合・簡素化）
│   │   ├── __init__.py
│   │   ├── schema.py              # 最小限のスキーマ
│   │   └── defaults.py            # コア設定のみ
│   │
│   ├── resources/                 # リソース管理（維持）
│   │   ├── model_manager.py
│   │   ├── ffmpeg_manager.py
│   │   └── resource_locator.py
│   │
│   ├── i18n.py                    # 国際化（維持）
│   └── languages.py               # 言語定義（維持）
│
├── tests/
├── docs/
└── pyproject.toml
```

---

## 3. 優先順位付きタスク

### Phase 1: リアルタイム文字起こし実装（最優先）

#### 1.1 StreamTranscriber の設計・実装

**目的:** リアルタイム文字起こしの中核クラス

```python
class StreamTranscriber:
    """リアルタイム文字起こしの中核"""

    def __init__(
        self,
        engine: BaseEngine,
        vad: Optional[VADProcessor] = None,
        config: Optional[dict] = None,
    ):
        ...

    def feed_audio(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> None:
        """音声チャンクを入力（ノンブロッキング）"""
        ...

    def get_result(self) -> Optional[TranscriptionResult]:
        """確定した結果を取得（ノンブロッキング）"""
        ...

    def get_interim(self) -> Optional[TranscriptionResult]:
        """中間結果を取得"""
        ...

    async def transcribe_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[TranscriptionResult]:
        """非同期ストリーム処理"""
        ...

    def reset(self) -> None:
        """状態をリセット"""
        ...

    def close(self) -> None:
        """リソース解放"""
        ...
```

**設計ポイント:**
- VAD 統合（オプショナル）
- 中間結果と確定結果の区別
- 非同期 API 対応
- 低レイテンシ設計

#### 1.2 VAD モジュールの分離

**目的:** VAD 処理を独立モジュール化

```python
class VADProcessor:
    """Voice Activity Detection プロセッサ"""

    def __init__(self, config: VADConfig):
        self._threshold = config.threshold
        self._min_speech_ms = config.min_speech_ms
        ...

    def process(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        """音声を分析し、発話区間を検出"""
        ...

    def get_speech_segments(self) -> List[Tuple[float, float]]:
        """検出された発話区間を取得"""
        ...
```

#### 1.3 統一された結果型

**目的:** API の二重構造を解消

```python
@dataclass
class TranscriptionResult:
    """統一された文字起こし結果"""
    text: str
    start_time: float
    end_time: float
    is_final: bool
    confidence: float = 1.0
    language: str = ""
    source_id: str = "default"

    def to_srt_entry(self, index: int) -> str:
        """SRT 形式のエントリに変換"""
        ...

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        ...
```

---

### Phase 2: API 統一と Config 簡素化

#### 2.1 Config の整理

**削除すべきセクション:**
- `multi_source`: GUI 専用
- `silence_detection.vad_state_machine`: GUI 状態管理
- `audio.processing`: GUI キュー管理
- `queue`: GUI 専用

**新しいコア設定:**

```python
CORE_CONFIG = {
    "engine": {
        "type": "auto",
        "device": None,           # cuda/cpu/auto
        "language": "ja",
    },
    "vad": {
        "enabled": True,
        "threshold": 0.5,
        "min_speech_ms": 250,
        "min_silence_ms": 100,
    },
    "translation": {
        "enabled": False,
        "service": "google",
        "target_language": "en",
    },
}
```

#### 2.2 Config 統合

- `config/core_config_builder.py` を `livecap_core/config/` に統合
- GUI 変換ロジックは削除（GUI 側で対応）

---

### Phase 3: パッケージ構造整理

#### 3.1 engines/ の統合

```bash
# 現在
engines/
├── base_engine.py
├── engine_factory.py
└── ...

# 移行後
livecap_core/engines/
├── __init__.py
├── base.py
├── factory.py
└── impl/
    └── ...
```

#### 3.2 インポートパスの移行

```python
# 旧（廃止）
from engines import EngineFactory

# 新
from livecap_core.engines import EngineFactory
```

**互換性:** 旧パスからのインポートは `DeprecationWarning` を出して新パスにリダイレクト。

---

### Phase 4: 翻訳機能実装

#### 4.1 BaseTranslator と Factory

```python
class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, source: str, target: str) -> str:
        ...

    @abstractmethod
    async def translate_async(self, text: str, source: str, target: str) -> str:
        ...

    @abstractmethod
    def get_supported_pairs(self) -> List[Tuple[str, str]]:
        """サポートする言語ペアを取得"""
        ...

class TranslatorFactory:
    @classmethod
    def create(cls, service: str, config: dict) -> BaseTranslator:
        ...
```

#### 4.2 実装

- `GoogleTranslator`: deep-translator ベース
- `RivaTranslator`: NVIDIA Riva NMT

---

### Phase 5: エンジン最適化

#### 5.1 BaseEngine 簡素化

**現在（6段階フェーズ）:**
```python
def load_model(self):
    # Phase 1: CHECK_DEPENDENCIES
    # Phase 2: PREPARE_DIRECTORY
    # Phase 3: CHECK_FILES
    # Phase 4: DOWNLOAD_MODEL
    # Phase 5: LOAD_TO_MEMORY
    # Phase 6: APPLY_SETTINGS
```

**簡素化後:**
```python
class BaseEngine(ABC):
    def load_model(self, progress_callback: Optional[Callable] = None) -> None:
        """モデルをロード。進捗報告はオプショナル"""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        ...

    @abstractmethod
    def get_required_sample_rate(self) -> int:
        ...

    def cleanup(self) -> None:
        ...
```

#### 5.2 各エンジンの効率化

- 不要なロギング削除
- 推論パスの最適化
- メモリ使用量の削減

---

### Phase 6: 依存関係整理

#### 6.1 pyproject.toml 再構成

```toml
[project]
dependencies = [
    # 最小限のコア
    "numpy",
    "appdirs",
    "tqdm",
]

[project.optional-dependencies]
# VAD
vad = ["ten-vad"]

# エンジン（個別選択可能）
engine-whisper = ["whisper-s2t"]
engine-sherpa = ["sherpa-onnx"]
engine-torch = ["torch", "torchaudio", "reazonspeech-k2-asr"]
engine-nemo = ["nemo-toolkit[asr]"]

# 翻訳
translation-google = ["deep-translator"]
translation-riva = ["nvidia-riva-client"]

# オーディオキャプチャ（オプショナル）
audio = ["sounddevice"]

# 開発
dev = ["pytest"]

# 推奨セット
recommended = [
    "livecap-core[vad,engine-whisper,translation-google]"
]

# フルセット
all = [
    "livecap-core[vad,engine-whisper,engine-sherpa,engine-torch,engine-nemo,translation-google,translation-riva,audio]"
]
```

---

## 4. オーディオソースの設計方針

### 4.1 コアの責務

コアは **音声データ（`np.ndarray`）を受け取る** ことに専念。

```python
# コアの使用例
transcriber = StreamTranscriber(engine=engine)

# 呼び出し側がオーディオソースを管理
while True:
    audio_chunk = get_audio_from_somewhere()  # 外部
    transcriber.feed_audio(audio_chunk)
    if result := transcriber.get_result():
        print(result.text)
```

### 4.2 オプショナルなオーディオソース

`livecap_core.audio_sources` として提供（optional extra）:

```python
class AudioSource(ABC):
    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

class MicrophoneSource(AudioSource):
    """sounddevice ベースのマイク入力"""
    def __init__(self, device_id: Optional[int] = None, sample_rate: int = 16000):
        ...

class SystemAudioSource(AudioSource):
    """システム音声キャプチャ"""
    # Windows: PyWAC
    # Linux: PulseAudio
    ...

class FileSource(AudioSource):
    """ファイルからのストリーム（テスト用にも有用）"""
    def __init__(self, file_path: Path, chunk_duration_ms: int = 100):
        ...
```

**理由:**
- テスト容易性（`FileSource` でハードウェアなしテスト可能）
- クロスプラットフォーム対応を分離
- 依存関係を軽量に保てる

---

## 5. マイルストーン

| マイルストーン | 内容 | 依存 |
|---------------|------|------|
| M1 | StreamTranscriber + VAD モジュール | - |
| M2 | 統一結果型 + API 整理 | M1 |
| M3 | Config 簡素化 | M2 |
| M4 | パッケージ構造統合 | M3 |
| M5 | 翻訳機能実装 | M4 |
| M6 | エンジン最適化 | M4 |
| M7 | 依存関係整理 | M5, M6 |
| M8 | ドキュメント整備 | M7 |

---

## 6. 移行戦略

### 6.1 破壊的変更の管理

**現状:** 本リポジトリは外部で利用されておらず、動作可能な状態ではない。

**方針:** 互換性維持は不要。破壊的変更を積極的に行い、クリーンな API 設計を優先する。

- 旧 API の非推奨期間は設けない
- 不要なコードは即座に削除
- インポートパスの変更も躊躇なく実施

### 6.2 テスト戦略

- 既存テストは必要に応じて更新・削除
- 新機能には必ずユニットテスト追加
- リアルタイム処理のベンチマーク追加

---

## 7. 決定事項・未決定事項

### 7.1 決定済み

1. **パッケージ名変更**
   - `livecap-core` → `livecap-cli` にリネーム予定
   - **優先度:** 低（リファクタリング完了後に対応）

2. **GPU/CPU デバイス選択**
   - `auto` / `gpu` / `cpu` の3オプションを提供
   - `auto`: CUDA 利用可能なら GPU、なければ CPU
   - 設定例: `--device auto`

### 7.2 検討中

1. **CLI の機能範囲（オーディオソース取得）**
   - CLI でマイク/システム音声からの直接リアルタイム文字起こしを提供するか？
   - **選択肢:**
     - A) 含める: `livecap-cli transcribe --realtime --device 0`
     - B) 含めない: CLI はファイル処理のみ、リアルタイムはライブラリ API として提供
   - **影響:** `audio_sources/` モジュールの位置づけが変わる

---

## 8. 関連ドキュメント

- [feature-inventory.md](../reference/feature-inventory.md) - 現在の機能一覧
- [core-api-spec.md](../architecture/core-api-spec.md) - 現在の API 仕様

---

## 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-11-25 | 初版作成 |
| 2025-11-25 | 7章: パッケージ名・デバイス選択を決定済みに移動 |
| 2025-11-25 | 6章: 互換性維持不要の方針を追記 |
