# LiveCap Core リファクタリング計画

> **Status**: ✅ COMPLETED
> **作成日:** 2025-11-25
> **完了日:** 2025-11-28
> **関連:** [feature-inventory.md](../../reference/feature-inventory.md)
> **実装:** Phase 1 完了 (#69), Phase A/B/C 完了 (#86), Phase 2 完了 (#158)

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
│   │   # config/ は Phase 2 で完全廃止
│   │   # VADConfig は livecap_core/vad/config.py に定義
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

### Phase 2: Config 廃止と API 簡素化 ✅ 完了 (#158)

> **実装詳細:** [phase2-api-config-simplification.md](./phase2-api-config-simplification.md)

#### 2.1 Config の完全廃止

**当初の計画:** Config の簡素化
**実際の実装:** Config システムの完全廃止

Phase 1 のアーキテクチャが Config なしで動作することが判明したため、簡素化ではなく完全廃止に方針転換。

**削除されたもの:**
- `config/` ディレクトリ全体
- `livecap_core/config/` ディレクトリ全体
- `DEFAULT_CONFIG`, `get_default_config()` 等
- `engine_type="auto"` サポート（`ValueError` を発生）

**新しい API:**

```python
from engines import EngineFactory, EngineMetadata

# エンジン検索
engines = EngineMetadata.get_engines_for_language("ja")
# → ["reazonspeech", "parakeet_ja", "whispers2t_base", ...]

# エンジン作成（明示的に指定）
engine = EngineFactory.create_engine("reazonspeech", device="cuda")

# パラメータ上書き
engine = EngineFactory.create_engine(
    "reazonspeech",
    use_int8=True,
    num_threads=8
)
```

#### 2.2 EngineMetadata.default_params

エンジン固有パラメータは `EngineMetadata.default_params` で一元管理：

```python
"reazonspeech": {
    "use_int8": False,
    "num_threads": 4,
    "decoding_method": "greedy_search",
}
```

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
    # コア依存
    "numpy",
    "appdirs",
    "tqdm",
    # VAD（コア機能）
    "ten-vad",
    # オーディオキャプチャ（コア機能）
    "sounddevice",
]

[project.optional-dependencies]
# エンジン（個別選択可能）
engine-whisper = ["whisper-s2t"]
engine-sherpa = ["sherpa-onnx"]
engine-torch = ["torch", "torchaudio", "reazonspeech-k2-asr"]
engine-nemo = ["nemo-toolkit[asr]"]

# 翻訳
translation-google = ["deep-translator"]
translation-riva = ["nvidia-riva-client"]

# 開発
dev = ["pytest"]

# 推奨セット（Whisper + Google翻訳）
recommended = [
    "livecap-cli[engine-whisper,translation-google]"
]

# フルセット
all = [
    "livecap-cli[engine-whisper,engine-sherpa,engine-torch,engine-nemo,translation-google,translation-riva]"
]
```

---

## 4. オーディオソースの設計方針

### 4.1 コア機能としてのオーディオソース

`audio_sources/` はコア機能として提供（通常インストールに含む）。

**CLI 使用例:**
```bash
# マイクからリアルタイム文字起こし
livecap-cli transcribe --realtime --mic 0

# システム音声からリアルタイム文字起こし
livecap-cli transcribe --realtime --system

# ファイル文字起こし
livecap-cli transcribe input.mp4 -o output.srt
```

### 4.2 オーディオソース実装

`livecap_core.audio_sources` として提供:

```python
class AudioSource(ABC):
    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[np.ndarray]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @classmethod
    @abstractmethod
    def list_devices(cls) -> List[DeviceInfo]:
        """利用可能なデバイス一覧を取得"""
        ...

class MicrophoneSource(AudioSource):
    """sounddevice ベースのマイク入力"""
    def __init__(self, device_id: Optional[int] = None, sample_rate: int = 16000):
        ...

class SystemAudioSource(AudioSource):
    """システム音声キャプチャ"""
    # Windows: PyWAC / WASAPI
    # Linux: PulseAudio
    ...

class FileSource(AudioSource):
    """ファイルからのストリーム（テスト用にも有用）"""
    def __init__(self, file_path: Path, chunk_duration_ms: int = 100):
        ...
```

### 4.3 ライブラリ API としての使用

CLI を使わず、ライブラリとして直接使用することも可能:

```python
from livecap_core import StreamTranscriber, EngineFactory
from livecap_core.audio_sources import MicrophoneSource

engine = EngineFactory.create_engine("whispers2t_base", device="auto")
engine.load_model()

transcriber = StreamTranscriber(engine=engine)
mic = MicrophoneSource(device_id=0)

async for result in transcriber.transcribe_stream(mic):
    print(f"{result.text}")
```

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

3. **CLI の機能範囲（オーディオソース取得）**
   - CLI でマイク/システム音声からの直接リアルタイム文字起こしを提供する
   - 使用例: `livecap-cli transcribe --realtime --mic 0`
   - `audio_sources/` モジュールはコア機能として含める（オプショナルではない）
   - 通常インストールでリアルタイム文字起こしが利用可能

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
| 2025-11-25 | 7章: CLI オーディオソース取得を決定済みに移動 |
| 2025-12-02 | Phase 2 完了: Config 簡素化から Config 完全廃止に方針転換・実装完了 (#158) |
| 2025-12-02 | アーキテクチャ図更新: config/ ディレクトリ削除を反映 |
