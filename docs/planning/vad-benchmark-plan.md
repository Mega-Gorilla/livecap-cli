# 統合ベンチマークフレームワーク実装計画

> **作成日:** 2025-11-25
> **関連 Issue:** #86
> **ステータス:** 計画確定
> **最終更新:** 2025-11-26

---

## 1. 概要

### 1.1 目的

livecap-cli の音声認識パイプライン全体を評価するための**統合ベンチマークフレームワーク**を構築する。

**VAD ベンチマーク + ASR ベンチマークを同時実装**し、以下を実現：

- 複数の VAD バックエンド（10構成）を比較評価
- 全 ASR エンジン（10種類）の単体性能を評価
- VAD × ASR の最適な組み合わせを発見

### 1.2 背景

- Phase 1 で Silero VAD をデフォルトとして採用
- `docs/reference/vad-comparison.md` の調査により、他の VAD（JaVAD, TenVAD）が優れている可能性
- 本リポジトリには **10種類の ASR エンジン**が実装済み
- VAD × ASR の最適な組み合わせを発見する必要がある

### 1.3 同時実装の理由

```
┌─────────────────────────────────────────────────────────────────┐
│ コード共有率分析                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  共通部分 (80%):                                                 │
│  ├── metrics.py      # WER/CER/RTF計算                          │
│  ├── datasets.py     # 音声ファイル + Ground Truth管理           │
│  ├── engines.py      # 10エンジンの統一管理                      │
│  └── reports.py      # JSON/Markdown出力                        │
│                                                                  │
│  固有部分 (20%):                                                 │
│  ├── vad/runner.py   # VAD処理 + ASR呼び出し                    │
│  └── asr/runner.py   # ASR呼び出しのみ（VADスキップ）           │
│                                                                  │
│  → 共通基盤を作る時点で、ASRベンチマークは実質完成               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**工数比較:**

| アプローチ | 共通基盤 | VAD固有 | ASR固有 | 合計 |
|-----------|---------|---------|---------|------|
| VADのみ先行 | 5日 | 2日 | - | 7日 |
| 後からASR追加 | - | - | 3日 | **10日** |
| **同時実装** | 5日 | 2日 | 1日 | **8日** |

### 1.4 スコープ

| 含む | 含まない |
|------|----------|
| VAD × 全ASR評価（10 VAD × 10 ASR） | VAD/ASR の本番切り替え |
| ASR 単体評価（10エンジン） | 新エンジン実装 |
| 日本語・英語での評価 | 全言語での評価 |
| CLI ベンチマークツール | GUI |
| Windows self-hosted runner (RTX 4090) | GitHub-hosted GPU |

---

## 2. アーキテクチャ

### 2.1 モジュラー設計

**共通基盤 + 個別ベンチマーク**の設計で、コード再利用を最大化。

```
benchmarks/
├── common/                    # 共通モジュール (80%)
│   ├── __init__.py
│   ├── metrics.py             # WER/CER/RTF 計算
│   ├── datasets.py            # データセット管理
│   ├── engines.py             # ASR エンジン管理
│   └── reports.py             # レポート生成
├── asr/                       # ASR ベンチマーク
│   ├── __init__.py
│   ├── runner.py              # ASR ベンチマーク実行
│   └── cli.py                 # CLI エントリポイント
└── vad/                       # VAD ベンチマーク
    ├── __init__.py
    ├── runner.py              # VAD ベンチマーク実行
    ├── cli.py                 # CLI エントリポイント
    ├── factory.py             # VADFactory (engines/ パターン踏襲)
    └── backends/              # VAD バックエンド
        ├── __init__.py
        ├── base.py            # VADBackend Protocol
        ├── silero.py          # SileroVADBackend
        ├── javad.py           # JaVADBackend
        ├── webrtc.py          # WebRTCVADBackend
        └── tenvad.py          # TenVADBackend
```

### 2.2 評価フローの関係

```
┌─────────────────────────────────────────────────────────────────┐
│ ASR ベンチマーク（基礎）                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  テスト音声 (.wav)                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ASR Engine (10 engines)                                 │    │
│  │ → 音声全体を文字起こし                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 評価 (common/metrics.py)                                │    │
│  │ → WER/CER 計算                                          │    │
│  │ → RTF 計測                                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              VAD ベンチマークは「VAD処理を前に挿入」するだけ
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ VAD ベンチマーク（ASRの上に構築）                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  テスト音声 (.wav)                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VAD Backend (11 configurations)  ← 追加部分             │    │
│  │ → 音声セグメント検出                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ASR Engine (共通部分を再利用)                            │    │
│  │ → 各セグメントを文字起こし                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 評価 (common/metrics.py を再利用)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 共通コンポーネント

#### metrics.py

```python
from dataclasses import dataclass
from jiwer import wer, cer
import time

@dataclass
class BenchmarkMetrics:
    """ベンチマーク評価指標"""
    wer: float                      # Word Error Rate
    cer: float                      # Character Error Rate
    rtf: float                      # Real-Time Factor
    latency_ms: float               # 処理遅延
    memory_mb: float                # ピークRAM使用量
    gpu_memory_model_mb: float      # モデルロード後のVRAM使用量
    gpu_memory_peak_mb: float       # 推論中のピークVRAM使用量

def calculate_wer(reference: str, hypothesis: str) -> float:
    """WER を計算"""
    return wer(reference, hypothesis)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """CER を計算（日本語向け）"""
    return cer(reference, hypothesis)

def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    """Real-Time Factor を計算"""
    return processing_time / audio_duration if audio_duration > 0 else 0.0
```

#### engines.py

```python
from engines.metadata import EngineMetadata
from engines.engine_factory import EngineFactory

class BenchmarkEngineManager:
    """ベンチマーク用 ASR エンジン管理"""

    @staticmethod
    def get_engines_for_language(language: str) -> list[str]:
        """言語に対応するエンジン一覧を取得"""
        return EngineMetadata.get_engines_for_language(language)

    @staticmethod
    def create_engine(engine_id: str, device: str = "cuda", language: str = "ja"):
        """ベンチマーク用エンジンを作成（VAD無効化）"""
        config = {"transcription": {"input_language": language}}

        # WhisperS2T のみ内蔵 VAD を無効化
        if engine_id.startswith("whispers2t_"):
            config["whispers2t"] = {"use_vad": False}

        engine = EngineFactory.create_engine(engine_id, device=device, config=config)
        engine.load_model()
        return engine

    @staticmethod
    def get_all_engines() -> dict:
        """全エンジン情報を取得"""
        return EngineMetadata.get_all()
```

---

## 3. 評価マトリクス

### 3.1 ASR ベンチマーク

```
┌─────────────────────────────────────────────────────────────────┐
│ ASR 評価マトリクス                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ASR (10エンジン)              言語 (2+)                         │
│  ┌─────────────────┐          ┌──────────┐                      │
│  │ reazonspeech    │──────ja──│ Japanese │                      │
│  │ parakeet_ja     │──────ja──│          │                      │
│  │ parakeet        │──────en──│ English  │                      │
│  │ canary          │──────en──│          │                      │
│  │ voxtral         │──────en──│ (Future) │                      │
│  │ whispers2t_*    │──────all─│ de,fr,es │                      │
│  └─────────────────┘          └──────────┘                      │
│                                                                  │
│  Total: 10 ASR × 2 Lang = 20 tests (言語対応分のみ)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 VAD × ASR ベンチマーク

```
┌─────────────────────────────────────────────────────────────────┐
│ VAD × ASR 評価マトリクス                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VAD (10構成)          ASR (10エンジン)         言語 (2+)       │
│  ┌─────────────┐      ┌─────────────────┐      ┌──────────┐    │
│  │ Silero v6   │      │ reazonspeech    │──ja──│ Japanese │    │
│  │ TenVAD      │      │ parakeet_ja     │──ja──│          │    │
│  │ JaVAD tiny  │  ×   │ parakeet        │──en──│ English  │    │
│  │ JaVAD bal.  │      │ canary          │──en──│          │    │
│  │ JaVAD prec. │      │ voxtral         │──en──│          │    │
│  │ WebRTC 0-3  │      │ whispers2t_*    │──all─│          │    │
│  └─────────────┘      └─────────────────┘      └──────────┘    │
│                                                                  │
│  Full Matrix: 10 VAD × 10 ASR × 2 Lang = 200 combinations       │
│  Practical:   10 VAD × 3-4 ASR/lang × 2 Lang ≈ 60-80 tests     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 言語別推奨エンジン

| 言語 | 推奨エンジン | 代替エンジン |
|------|-------------|-------------|
| **Japanese (ja)** | reazonspeech, parakeet_ja | whispers2t_base |
| **English (en)** | parakeet, canary | whispers2t_base, voxtral |
| **German (de)** | canary | whispers2t_base, voxtral |
| **French (fr)** | canary | whispers2t_base, voxtral |
| **Spanish (es)** | canary | whispers2t_base, voxtral |

### 3.4 実行戦略

**Quick Mode** (CI デフォルト):
- ASR: 言語別デフォルト 2 エンジン（約 4 テスト）
- VAD: Silero v6, JaVAD precise, WebRTC mode 3（約 12 テスト）
- 推定時間: ~5分

**Standard Mode**:
- ASR: 言語別全エンジン（約 10 テスト）
- VAD: 全 11 構成 × 言語別 2-3 エンジン（約 44-66 テスト）
- 推定時間: ~20分

**Full Mode** (手動実行):
- ASR: 全エンジン × 全対応言語（約 20 テスト）
- VAD: 全 11 構成 × 全対応エンジン（約 88+ テスト）
- 推定時間: ~60分

---

## 4. ASR エンジン一覧

### 4.1 実装済みエンジン

`engines/metadata.py` から取得した全 10 エンジン：

| ID | 表示名 | 対応言語 | サイズ | VAD内蔵 |
|----|--------|---------|--------|---------|
| `reazonspeech` | ReazonSpeech K2 v2 | ja | 159MB | ❌ |
| `parakeet` | NVIDIA Parakeet TDT 0.6B | en | 1.2GB | ❌ |
| `parakeet_ja` | NVIDIA Parakeet TDT CTC JA | ja | 600MB | ❌ |
| `canary` | NVIDIA Canary 1B Flash | en,de,fr,es | 1.5GB | ❌ |
| `voxtral` | MistralAI Voxtral Mini 3B | en,es,fr,pt,hi,de,nl,it | 3GB | ❌ |
| `whispers2t_tiny` | WhisperS2T Tiny | 多言語 | 39MB | ✅ |
| `whispers2t_base` | WhisperS2T Base | 多言語 | 74MB | ✅ |
| `whispers2t_small` | WhisperS2T Small | 多言語 | 244MB | ✅ |
| `whispers2t_medium` | WhisperS2T Medium | 多言語 | 769MB | ✅ |
| `whispers2t_large_v3` | WhisperS2T Large-v3 | 多言語 | 1.55GB | ✅ |

### 4.2 VAD 無効化

WhisperS2T のみ内蔵 VAD を持つ。ベンチマーク時は `use_vad=False` で無効化：

```python
# engines/whispers2t_engine.py:52
self.use_vad = self.engine_config.get('use_vad', True)

# engines/whispers2t_engine.py:290-304
if self.use_vad:
    outputs = self.model.transcribe_with_vad(...)
else:
    outputs = self.model.transcribe(...)  # VAD なし
```

他のエンジン（Parakeet, ReazonSpeech, Canary, Voxtral）は VAD 関連コードがないため、そのまま使用可能。

---

## 5. VAD 構成一覧

### 5.1 ベンチマーク対象

**合計 10 構成**（Silero v5 は v6 の上位互換のため除外）:

| VAD | モデル/設定 | ライセンス | 特徴 |
|-----|------------|-----------|------|
| Silero VAD v6 | ONNX | MIT | 現在のデフォルト、高精度 |
| TenVAD | - | 独自 | 最軽量・最高速（評価のみ） |
| JaVAD | tiny | MIT | 0.64s window、即時検出向け |
| JaVAD | balanced | MIT | 1.92s window、バランス型 |
| JaVAD | precise | MIT | 3.84s window、最高精度 |
| WebRTC VAD | mode 0 | BSD | 最も寛容、誤検出少 |
| WebRTC VAD | mode 1 | BSD | やや厳格 |
| WebRTC VAD | mode 2 | BSD | 厳格 |
| WebRTC VAD | mode 3 | BSD | 最も厳格、見逃し多 |

### 5.2 VAD バックエンド設計 (engines/ パターン踏襲)

`engines/` の設計パターン（Protocol + Factory）を踏襲し、一貫した API を提供。

#### VADBackend Protocol

```python
# benchmarks/vad/backends/base.py
from typing import Protocol, List, Tuple
import numpy as np

class VADBackend(Protocol):
    """VAD バックエンドの共通インターフェース"""

    @property
    def name(self) -> str:
        """バックエンド名 (例: 'silero_v6', 'javad_precise')"""
        ...

    def load(self) -> None:
        """モデルをロード"""
        ...

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        """
        音声から発話区間を検出

        Returns:
            [(start_sec, end_sec), ...] のリスト
        """
        ...

    def cleanup(self) -> None:
        """リソース解放"""
        ...
```

#### バックエンド実装例

```python
# benchmarks/vad/backends/silero.py
class SileroVADBackend:
    def __init__(self, version: str = "v6", threshold: float = 0.5):
        self._version = version
        self._threshold = threshold
        self._model = None

    @property
    def name(self) -> str:
        return f"silero_{self._version}"

    def load(self) -> None:
        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float]]:
        from silero_vad import get_speech_timestamps
        timestamps = get_speech_timestamps(audio, self._model, threshold=self._threshold)
        return [(t['start'] / sample_rate, t['end'] / sample_rate) for t in timestamps]

    def cleanup(self) -> None:
        self._model = None


# benchmarks/vad/backends/javad.py
class JaVADBackend:
    def __init__(self, model_name: str = "balanced"):  # tiny, balanced, precise
        self._model_name = model_name
        self._processor = None

    @property
    def name(self) -> str:
        return f"javad_{self._model_name}"

    def load(self) -> None:
        from javad import Processor
        self._processor = Processor(model_name=self._model_name)
    # ...
```

#### VADFactory

```python
# benchmarks/vad/factory.py
class VADFactory:
    """VAD バックエンド生成ファクトリ (EngineFactory パターン踏襲)"""

    _REGISTRY = {
        "silero_v6": lambda: SileroVADBackend(version="v6"),
        "javad_tiny": lambda: JaVADBackend(model_name="tiny"),
        "javad_balanced": lambda: JaVADBackend(model_name="balanced"),
        "javad_precise": lambda: JaVADBackend(model_name="precise"),
        "webrtc_0": lambda: WebRTCVADBackend(mode=0),
        "webrtc_1": lambda: WebRTCVADBackend(mode=1),
        "webrtc_2": lambda: WebRTCVADBackend(mode=2),
        "webrtc_3": lambda: WebRTCVADBackend(mode=3),
        "tenvad": lambda: TenVADBackend(),
    }

    @classmethod
    def create(cls, vad_id: str) -> VADBackend:
        if vad_id not in cls._REGISTRY:
            raise ValueError(f"Unknown VAD: {vad_id}")
        backend = cls._REGISTRY[vad_id]()
        backend.load()
        return backend

    @classmethod
    def get_available(cls) -> List[str]:
        return list(cls._REGISTRY.keys())
```

**設計のメリット:**
- `engines/` と一貫した設計パターン
- 新しいVADバックエンドの追加が容易
- テストしやすい（モック化しやすい）

**既存 `livecap_core/vad/backends/silero.py` との関係:**
- 既存: ストリーミング処理向け（チャンク → 確率）
- ベンチマーク用: バッチ処理向け（音声全体 → セグメントリスト）
- 用途が異なるため、別クラスとして実装

---

## 6. 評価指標

### 6.1 ASR 精度指標

| 指標 | 説明 | 計算方法 |
|------|------|----------|
| **WER** | Word Error Rate | `(S + D + I) / N` |
| **CER** | Character Error Rate | 文字単位の WER（日本語向け） |

- `S`: 置換数、`D`: 削除数、`I`: 挿入数、`N`: 参照単語数
- ライブラリ: [jiwer](https://github.com/jitsi/jiwer)

### 6.2 性能指標

| 指標 | 説明 | 単位 |
|------|------|------|
| **RTF** | Real-Time Factor | 比率（低いほど高速） |
| **Latency** | 入力→出力の遅延 | ms |
| **Memory (RAM)** | ピークRAM使用量 | MB |
| **GPU VRAM (Model)** | モデルロード後のVRAM使用量 | MB |
| **GPU VRAM (Peak)** | 推論中のピークVRAM使用量 | MB |

#### RAM メモリ測定方法

Python 標準ライブラリの `tracemalloc` を使用:

```python
import tracemalloc

def measure_ram_usage(func):
    """RAM 使用量を測定"""
    tracemalloc.start()
    result = func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "memory_current_mb": current / 1024**2,
        "memory_peak_mb": peak / 1024**2,
    }
```

**注意:**
- `tracemalloc` は Python ヒープのみを測定（ネイティブメモリは含まない）
- 相対比較には十分な精度
- 必要に応じて `psutil` への切り替えを検討（Phase B で検証）

#### GPU メモリ測定方法

```python
import torch

def measure_gpu_memory(func):
    """GPU メモリ使用量を測定するデコレータ"""
    if not torch.cuda.is_available():
        return {"gpu_memory_model_mb": None, "gpu_memory_peak_mb": None}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # モデルロード後の使用量
    result = func()
    torch.cuda.synchronize()
    model_memory = torch.cuda.memory_allocated() / 1024**2

    # 推論後のピーク使用量
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "gpu_memory_model_mb": model_memory,
        "gpu_memory_peak_mb": peak_memory,
    }
```

**注意:**
- CPU モードの場合は `None` を記録
- ONNX Runtime (GPU) の場合は `nvidia-smi` 経由で測定が必要な場合がある
- 測定の再現性のため、warm-up 実行後に測定する

### 6.3 VAD 固有指標

| 指標 | 説明 |
|------|------|
| **Segments** | 検出セグメント数 |
| **Avg Duration** | 平均セグメント長 |
| **Speech Ratio** | 音声区間の割合 |

---

## 7. データセット

### 7.1 ディレクトリ構造

**決定事項:** 言語別フォルダ構造に統一（`audio/` と `prepared/` で一貫性を持たせる）

```
tests/assets/
├── audio/                    # git追跡（quickモード用、数ファイル）
│   ├── ja/
│   │   ├── jsut_basic5000_0001.wav
│   │   └── jsut_basic5000_0001.txt
│   └── en/
│       ├── librispeech_1089-134686-0001.wav
│       └── librispeech_1089-134686-0001.txt
│
├── prepared/                 # git無視（スクリプトで生成）
│   ├── ja/                   # 変換済み日本語データ
│   │   ├── jsut_basic5000_0002.wav
│   │   └── ...
│   └── en/                   # 変換済み英語データ
│       └── ...
│
├── source/                   # git無視（ソースコーパス）
│   ├── jsut/jsut_ver1.1/     # 3.4GB - JSUT v1.1
│   └── librispeech/test-clean/  # 358MB - LibriSpeech
│
└── README.md
```

### 7.2 構造変更の影響

現在の `tests/assets/audio/` はフラット構造のため、言語別フォルダへの移行が必要。

**現在の構造:**
```
tests/assets/audio/
├── jsut_basic5000_0001_ja.wav
├── jsut_basic5000_0001_ja.txt
├── librispeech_test-clean_1089-134686-0001_en.wav
└── librispeech_test-clean_1089-134686-0001_en.txt
```

**変更後:**
```
tests/assets/audio/
├── ja/
│   ├── jsut_basic5000_0001.wav
│   └── jsut_basic5000_0001.txt
└── en/
    ├── librispeech_1089-134686-0001.wav
    └── librispeech_1089-134686-0001.txt
```

**影響を受けるファイル:**

| ファイル | 変更内容 |
|---------|---------|
| `tests/audio_sources/test_file_source.py` | パス更新 |
| `tests/core/test_text_normalization.py` | パス更新 |
| `tests/integration/engines/test_smoke_engines.py` | パス更新 |
| `tests/integration/realtime/test_mock_realtime_flow.py` | パス更新 |
| `tests/integration/realtime/test_e2e_realtime_flow.py` | パス更新 |
| `examples/realtime/basic_file_transcription.py` | パス更新 |
| `examples/realtime/custom_vad_config.py` | パス更新 |
| `examples/realtime/callback_api.py` | パス更新 |
| `examples/README.md` | パス更新 |
| `tests/assets/README.md` | ドキュメント更新 |
| `tests/utils/text_normalization.py` | `get_language_from_filename()` 更新 |

**実装タスク:** Phase A-2 に含める

### 7.3 ソースデータセット

| データセット | 言語 | 形式 | トランスクリプト | ライセンス |
|-------------|------|------|-----------------|-----------|
| JSUT v1.1 | ja | WAV | `ID:テキスト` | 非商用 |
| LibriSpeech test-clean | en | FLAC | `ID TEXT` | CC BY 4.0 |

### 7.4 統一フォーマット仕様

変換スクリプトで以下のフォーマットに統一：

| 項目 | 仕様 |
|------|------|
| 音声形式 | WAV, 16kHz, mono, 16bit |
| 正規化 | ピーク -1dBFS |
| ファイル名 | `{corpus}_{subset}_{id}.wav` |
| トランスクリプト | 同名 `.txt`、UTF-8、1行、末尾改行 |
| フォルダ | `{lang}/` (ja, en) |

### 7.5 実行モードとデータセット

| モード | データソース | ファイル数 | 用途 |
|--------|-------------|-----------|------|
| `quick` | `audio/` | ja:2, en:2 | CI smoke test |
| `standard` | `prepared/` | ja:100, en:100 (調整可) | ローカル開発 |
| `full` | `prepared/` | 全ファイル | 本格ベンチマーク |

### 7.6 変換スクリプト

```bash
# standard モード（各言語100ファイル）
python scripts/prepare_benchmark_data.py --mode standard

# full モード（全ファイル）
python scripts/prepare_benchmark_data.py --mode full

# カスタム
python scripts/prepare_benchmark_data.py --ja-limit 500 --en-limit 200
```

スクリプトの処理内容：

1. **JSUT**: `transcript_utf8.txt` 読み込み → WAV 正規化 → `prepared/ja/` へ出力
2. **LibriSpeech**: `*.trans.txt` 読み込み → FLAC→WAV 変換 + 正規化 → `prepared/en/` へ出力

### 7.7 .gitignore 設定

```gitignore
# ソースコーパス（大規模）
tests/assets/source/

# 生成データ（ライセンス問題）
tests/assets/prepared/
```

---

## 8. 実装ステップ

### 概要: Phase A/B/C アプローチ

```
┌─────────────────────────────────────────────────────────────┐
│ Phase A: 基盤構築                                            │
├─────────────────────────────────────────────────────────────┤
│ A-1. エンジン動作確認 (self-hosted runner)                   │
│ A-2. データセット管理実装 (builtin + 外部参照)               │
│ A-3. 共通モジュール (metrics.py, reports.py)                │
│ A-4. pyproject.toml に benchmark extra 追加                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase B: ASR ベンチマーク                                    │
├─────────────────────────────────────────────────────────────┤
│ B-1. ASR runner 実装                                         │
│ B-2. CLI 実装 (python -m benchmarks.asr)                    │
│ B-3. 動作するエンジンでベンチマーク実行                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase C: VAD ベンチマーク                                    │
├─────────────────────────────────────────────────────────────┤
│ C-1. VAD バックエンド実装 (JaVAD, WebRTC)                    │
│ C-2. VAD runner 実装                                         │
│ C-3. CI ワークフロー設定                                     │
└─────────────────────────────────────────────────────────────┘
```

**実装理由:**
1. **動作確認が先**: 壊れたエンジンでベンチマークしても無意味
2. **データセットが先**: ベンチマークの価値はデータ品質で決まる
3. **ASR が先**: VAD は ASR の上に構築（依存関係）

---

### Phase A: 基盤構築

#### A-1. エンジン動作確認

self-hosted runner で全10エンジンの smoke test を実行:

```bash
uv run pytest tests/integration/engines -m engine_smoke
```

動作しないエンジンを特定し、ベンチマーク対象から除外または修正。

#### A-2. データセット管理実装

**対象:**
- `scripts/prepare_benchmark_data.py` - 変換スクリプト
- `tests/assets/audio/` の再構成（言語別フォルダ）
- 既存コードのパス参照修正

**実装内容:**

1. **audio/ 再構成** (Section 7.2 参照)
   - `tests/assets/audio/ja/`, `tests/assets/audio/en/` フォルダ作成
   - 既存ファイルを言語別フォルダに移動
   - ファイル名から `_ja`/`_en` サフィックスを削除

2. **既存コード修正** (~11ファイル)
   - テストコード: パス参照更新
   - サンプルコード: パス参照更新
   - `tests/utils/text_normalization.py`: `get_language_from_filename()` をフォルダベースに更新
   - `tests/assets/README.md`: ドキュメント更新

3. **変換スクリプト作成** (`scripts/prepare_benchmark_data.py`)
   ```python
   def prepare_jsut(source_dir, output_dir, limit=None):
       """JSUT → 統一フォーマット変換"""
       # transcript_utf8.txt 読み込み
       # WAV 正規化 (16kHz mono, -1dBFS)
       # prepared/ja/ へ出力

   def prepare_librispeech(source_dir, output_dir, limit=None):
       """LibriSpeech → 統一フォーマット変換"""
       # *.trans.txt 読み込み
       # FLAC → WAV 変換 + 正規化
       # prepared/en/ へ出力
   ```

4. **.gitignore 更新**
   - `tests/assets/prepared/` を追加

#### A-3. 共通モジュール

**対象:** `benchmarks/common/`

1. **metrics.py** - WER/CER/RTF/VRAM 計算
   - jiwer ライブラリ活用
   - GPU メモリ測定（torch.cuda）
   - RAM メモリ測定（tracemalloc）

2. **text_normalization.py** - テキスト正規化（独立モジュール）
   - `tests/utils/text_normalization.py` からコピー・独立化
   - 理由: `tests/` からのインポートを避け、ベンチマークを独立させる
   - `normalize_en()`, `normalize_ja()`, `normalize_text()` を提供

3. **datasets.py** - データセット管理
   ```python
   class DatasetManager:
       def get_dataset(self, mode: str = "auto") -> Dataset:
           """
           mode:
           - "quick": audio/ (git追跡)
           - "standard": prepared/ (100ファイル/言語) - prepared/ 必須
           - "full": prepared/ (全ファイル) - prepared/ 必須
           - "auto": prepared > audio の順で自動選択
           """
   ```

   **実装方針:**
   - AudioFile は **Lazy ロード**（イテレート時に音声読み込み）
   - standard/full モードで `prepared/` が存在しない場合は**エラー**
     - `scripts/prepare_benchmark_data.py` の実行を促すメッセージを表示

4. **engines.py** - ASR エンジン管理
   - EngineFactory ラッパー
   - VAD 無効化設定（WhisperS2T の `use_vad=False`）

   **エンジンキャッシュ戦略:**
   ```python
   class BenchmarkEngineManager:
       _cache: dict[str, BaseEngine] = {}

       def create_engine(self, engine_id: str, device: str, language: str):
           """エンジンを取得（キャッシュがあれば再利用）"""
           cache_key = f"{engine_id}_{device}_{language}"
           if cache_key not in self._cache:
               engine = EngineFactory.create_engine(...)
               engine.load_model()
               self._cache[cache_key] = engine
           return self._cache[cache_key]

       def clear_cache(self):
           """ベンチマーク終了時に全エンジンをクリーンアップ"""
           for engine in self._cache.values():
               engine.cleanup()
           self._cache.clear()
   ```

   **GPU メモリ測定との整合:**
   - Model memory: 初回ロード時に1回測定
   - Peak memory: 各推論前に `reset_peak_memory_stats()` でリセット

5. **reports.py** - レポート生成
   - JSON 出力
   - Markdown 出力
   - コンソール表形式出力（`tabulate` ライブラリ使用）
   - **出力フォーマットの詳細は Phase B で調整**

---

### Phase B: ASR ベンチマーク

#### B-1. ASR runner 実装

**対象:** `benchmarks/asr/`

```python
# benchmarks/asr/runner.py
class ASRBenchmarkRunner:
    def __init__(self, engines: list[str], languages: list[str], device: str = "cuda"):
        self.engine_manager = BenchmarkEngineManager()
        self.engines = engines
        self.languages = languages
        self.device = device

    def run(self, dataset: Dataset) -> list[ASRBenchmarkResult]:
        results = []
        for engine_id in self.engines:
            for audio_file in dataset.get_files_for_engine(engine_id):
                result = self._benchmark_single(engine_id, audio_file)
                results.append(result)
        return results

    def _benchmark_single(self, engine_id: str, audio_file: AudioFile) -> ASRBenchmarkResult:
        import torch

        # GPU メモリ測定準備
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        engine = self.engine_manager.create_engine(engine_id, self.device, audio_file.language)

        # モデルロード後の VRAM 使用量
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_model = torch.cuda.memory_allocated() / 1024**2
        else:
            gpu_memory_model = None

        try:
            start_time = time.perf_counter()
            transcript, _ = engine.transcribe(audio_file.audio, audio_file.sample_rate)
            elapsed = time.perf_counter() - start_time

            # 推論後のピーク VRAM 使用量
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**2
            else:
                gpu_memory_peak = None

            return ASRBenchmarkResult(
                engine=engine_id,
                language=audio_file.language,
                audio_file=audio_file.path,
                transcript=transcript,
                reference=audio_file.transcript,
                wer=calculate_wer(audio_file.transcript, transcript),
                cer=calculate_cer(audio_file.transcript, transcript),
                rtf=calculate_rtf(audio_file.duration, elapsed),
                gpu_memory_model_mb=gpu_memory_model,
                gpu_memory_peak_mb=gpu_memory_peak,
            )
        finally:
            engine.cleanup()
```

---

### Phase C: VAD ベンチマーク

#### C-1. VAD バックエンド実装

**対象:** `benchmarks/vad/backends/` および `benchmarks/vad/factory.py`

Section 5.2 の設計（engines/ パターン踏襲）に従って実装:

1. **base.py** - VADBackend Protocol（load, process, cleanup メソッド）
2. **silero.py** - SileroVADBackend（`get_speech_timestamps()` 使用、バッチ処理向け）
3. **javad.py** - JaVADBackend（tiny/balanced/precise）
4. **webrtc.py** - WebRTCVADBackend（mode 0-3）
5. **tenvad.py** - TenVADBackend
6. **factory.py** - VADFactory（VADBackend 生成ファクトリ）

#### C-2. VAD runner 実装

**対象:** `benchmarks/vad/`

```python
# benchmarks/vad/runner.py
class VADBenchmarkRunner:
    def __init__(
        self,
        vad_backends: list[VADBackend],
        asr_engines: list[str],
        device: str = "cuda"
    ):
        self.vad_backends = vad_backends
        self.asr_runner = ASRBenchmarkRunner(asr_engines, [], device)

    def run(self, dataset: Dataset) -> list[VADBenchmarkResult]:
        results = []
        for vad in self.vad_backends:
            for engine_id in self.asr_runner.engines:
                for audio_file in dataset.get_files_for_engine(engine_id):
                    result = self._benchmark_single(vad, engine_id, audio_file)
                    results.append(result)
        return results

    def _benchmark_single(
        self,
        vad: VADBackend,
        engine_id: str,
        audio_file: AudioFile
    ) -> VADBenchmarkResult:
        # VAD処理
        vad.reset()
        segments = vad.process(audio_file.audio, audio_file.sample_rate)

        # セグメントごとにASR実行
        transcripts = []
        engine = self.asr_runner.engine_manager.create_engine(...)
        try:
            for start, end in segments:
                segment_audio = audio_file.audio[int(start*sr):int(end*sr)]
                text, _ = engine.transcribe(segment_audio, sr)
                transcripts.append(text)
        finally:
            engine.cleanup()

        full_transcript = " ".join(transcripts)
        return VADBenchmarkResult(
            vad=vad.name,
            asr=engine_id,
            # ... metrics ...
        )
```

#### B-2. CLI 統合 (ASR)

```bash
# ASR ベンチマーク
python -m benchmarks.asr --all
python -m benchmarks.asr --engine reazonspeech parakeet_ja --language ja
python -m benchmarks.asr --mode quick --output results.json
```

#### C-3. CLI 統合 (VAD) + CI ワークフロー

```bash
# VAD ベンチマーク
python -m benchmarks.vad --all
python -m benchmarks.vad --vad silero_v6 javad_precise --asr reazonspeech
python -m benchmarks.vad --mode standard --format markdown

# 両方実行
python -m benchmarks --all --output report.md
```

GitHub Actions ワークフロー作成（詳細は Section 11）。

---

## 9. CLI インターフェース

### 9.1 ASR ベンチマーク

```bash
# 全エンジン比較
python -m benchmarks.asr --all

# 言語別比較
python -m benchmarks.asr --language ja

# 特定エンジン
python -m benchmarks.asr --engine reazonspeech parakeet_ja whispers2t_base

# 実行モード
python -m benchmarks.asr --mode quick    # デフォルト2エンジン
python -m benchmarks.asr --mode standard # 言語別全エンジン
python -m benchmarks.asr --mode full     # 全エンジン×全言語

# 出力形式
python -m benchmarks.asr --output results.json --format json
python -m benchmarks.asr --output report.md --format markdown
```

### 9.2 VAD ベンチマーク

```bash
# クイック実行（デフォルト VAD + デフォルト ASR）
python -m benchmarks.vad

# 全 VAD × 言語別推奨 ASR
python -m benchmarks.vad --all-vad

# 特定 VAD + 全対応 ASR
python -m benchmarks.vad --vad silero_v6 javad_precise --all-asr

# 全 VAD × 全 ASR（フルモード）
python -m benchmarks.vad --full

# 言語指定
python -m benchmarks.vad --language ja --all-vad

# 特定エンジン指定
python -m benchmarks.vad --asr reazonspeech parakeet_ja whispers2t_base

# 出力形式
python -m benchmarks.vad --output results.json --format json
python -m benchmarks.vad --output report.md --format markdown
```

### 9.3 統合実行

```bash
# 両方のベンチマークを実行
python -m benchmarks --type both --mode standard

# ASRのみ
python -m benchmarks --type asr --mode quick

# VADのみ
python -m benchmarks --type vad --mode full
```

---

## 10. 出力フォーマット

### 10.1 ASR ベンチマーク結果

```
=== ASR Benchmark Results ===

Dataset: tests/assets/audio (2 files)
Device: cuda (RTX 4090)

┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Japanese Results                                                                       │
├─────────────────┬────────┬────────┬────────┬──────────┬───────────────┬───────────────┤
│ Engine          │ WER    │ CER    │ RTF    │ RAM      │ VRAM (Model)  │ VRAM (Peak)   │
├─────────────────┼────────┼────────┼────────┼──────────┼───────────────┼───────────────┤
│ reazonspeech    │ 3.2%   │ 1.1%   │ 0.08   │ 245 MB   │ 412 MB        │ 523 MB        │
│ parakeet_ja     │ 4.5%   │ 1.8%   │ 0.12   │ 1.2 GB   │ 1.8 GB        │ 2.1 GB        │
│ whispers2t_base │ 5.8%   │ 2.3%   │ 0.15   │ 312 MB   │ 890 MB        │ 1.1 GB        │
└─────────────────┴────────┴────────┴────────┴──────────┴───────────────┴───────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────┐
│ English Results                                                                        │
├─────────────────┬────────┬────────┬────────┬──────────┬───────────────┬───────────────┤
│ Engine          │ WER    │ CER    │ RTF    │ RAM      │ VRAM (Model)  │ VRAM (Peak)   │
├─────────────────┼────────┼────────┼────────┼──────────┼───────────────┼───────────────┤
│ parakeet        │ 3.8%   │ 2.1%   │ 0.10   │ 1.4 GB   │ 2.2 GB        │ 2.5 GB        │
│ canary          │ 4.2%   │ 2.5%   │ 0.14   │ 1.8 GB   │ 2.8 GB        │ 3.2 GB        │
│ whispers2t_base │ 5.1%   │ 3.0%   │ 0.12   │ 312 MB   │ 890 MB        │ 1.1 GB        │
└─────────────────┴────────┴────────┴────────┴──────────┴───────────────┴───────────────┘

=== Summary ===
Best for Japanese: reazonspeech (CER: 1.1%)
Best for English:  parakeet (WER: 3.8%)
Fastest overall:   reazonspeech (RTF: 0.08)
Lowest VRAM:       reazonspeech (Peak: 523 MB)
```

### 10.2 VAD ベンチマーク結果

```
=== VAD Benchmark Results ===

Dataset: tests/assets/audio (2 files)
Mode: Standard (10 VAD × 3 ASR/lang)

┌─────────────────────────────────────────────────────────────────────────┐
│ Japanese Results (reazonspeech)                                          │
├─────────────┬────────┬────────┬────────┬──────────┬──────────┬─────────┤
│ VAD         │ WER    │ CER    │ RTF    │ Segments │ Memory   │ Status  │
├─────────────┼────────┼────────┼────────┼──────────┼──────────┼─────────┤
│ Silero v6   │ 5.2%   │ 2.1%   │ 0.012  │ 3        │ 245 MB   │ ✓       │
│ JaVAD prec. │ 4.1%   │ 1.5%   │ 0.015  │ 3        │ 312 MB   │ ✓       │
│ WebRTC m3   │ 8.3%   │ 4.2%   │ 0.003  │ 6        │ 128 MB   │ ✓       │
└─────────────┴────────┴────────┴────────┴──────────┴──────────┴─────────┘

=== Summary ===
Best VAD for Japanese: JaVAD precise (CER: 1.5%)
Best VAD for English:  JaVAD precise (WER: 4.2%)
Fastest VAD:           WebRTC mode 3 (RTF: 0.003)
```

### 10.3 JSON 出力

```json
{
  "metadata": {
    "timestamp": "2025-11-25T12:00:00Z",
    "device": "cuda (RTX 4090)",
    "benchmark_type": "both",
    "mode": "standard"
  },
  "asr_results": [
    {
      "engine": "reazonspeech",
      "language": "ja",
      "audio_file": "jsut_basic5000_0001_ja.wav",
      "metrics": {
        "wer": 0.032,
        "cer": 0.011,
        "rtf": 0.08,
        "memory_mb": 245,
        "gpu_memory_model_mb": 412,
        "gpu_memory_peak_mb": 523
      }
    }
  ],
  "vad_results": [
    {
      "vad": "silero_v6",
      "asr": "reazonspeech",
      "language": "ja",
      "metrics": {
        "wer": 0.052,
        "cer": 0.021,
        "rtf": 0.012,
        "segments": 3
      }
    }
  ],
  "summary": {
    "best_asr_by_language": {
      "ja": {"engine": "reazonspeech", "cer": 0.011},
      "en": {"engine": "parakeet", "wer": 0.038}
    },
    "best_vad_by_language": {
      "ja": {"vad": "javad_precise", "cer": 0.015},
      "en": {"vad": "javad_precise", "wer": 0.042}
    },
    "lowest_vram": {
      "engine": "reazonspeech",
      "gpu_memory_peak_mb": 523
    },
    "fastest": {
      "engine": "reazonspeech",
      "rtf": 0.08
    }
  }
}
```

---

## 11. CI ワークフロー

### 11.1 ワークフロー設計

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on:
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Benchmark type'
        required: true
        default: 'both'
        type: choice
        options:
          - asr
          - vad
          - both
      mode:
        description: 'Execution mode'
        required: true
        default: 'quick'
        type: choice
        options:
          - quick
          - standard
          - full
      language:
        description: 'Target language (empty for all)'
        required: false
        default: ''

jobs:
  benchmark-gpu:
    name: GPU Benchmark (Windows RTX 4090)
    runs-on: [self-hosted, windows]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup FFmpeg
        run: |
          $ffmpegBinDir = Join-Path $env:GITHUB_WORKSPACE "ffmpeg-bin"
          # ... (既存の FFmpeg セットアップ)

      - name: Setup Python environment
        run: |
          uv sync --extra vad --extra engines-torch --extra engines-nemo --extra benchmark

      - name: Run Benchmark
        env:
          LIVECAP_FFMPEG_BIN: ${{ github.workspace }}\ffmpeg-bin
          LIVECAP_DEVICE: cuda
        run: |
          $type = "${{ github.event.inputs.benchmark_type }}"
          $mode = "${{ github.event.inputs.mode }}"
          $lang = "${{ github.event.inputs.language }}"

          $args = @("--mode", $mode, "--output", "results.json", "--format", "json")
          if ($lang) { $args += @("--language", $lang) }

          uv run python -m benchmarks --type $type @args

      - name: Generate Report
        run: |
          uv run python -m benchmarks.common.reports --input results.json --output report.md

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.run_id }}
          path: |
            results.json
            report.md

      - name: Post Summary
        run: |
          Get-Content report.md | Add-Content $env:GITHUB_STEP_SUMMARY
```

### 11.2 実行モード詳細

| モード | ASR テスト | VAD テスト | 推定時間 |
|--------|-----------|-----------|---------|
| `quick` | 4 (言語別2) | 12 (3 VAD × 2 ASR × 2 lang) | ~5分 |
| `standard` | 10 (言語別全) | 40-60 (10 VAD × 2-3 ASR × 2 lang) | ~20分 |
| `full` | 20 (全組み合わせ) | 80+ (10 VAD × 全ASR × 2 lang) | ~60分 |

---

## 12. 依存関係

### 12.1 pyproject.toml 追加

```toml
[project.optional-dependencies]
benchmark = [
    # VAD backends
    "silero-vad>=5.1",
    "webrtcvad>=2.0.10",
    "javad",
    # Metrics
    "jiwer>=3.0",
    # Reporting
    "matplotlib",
    "pandas",
    "tabulate",
    # Profiling
    "memory_profiler",
]

benchmark-full = [
    "livecap-cli[benchmark]",
    "ten-vad",  # ライセンス注意
]
```

### 12.2 既存依存の活用

- `engines/` - ASR エンジン実装
- `engines/metadata.py` - エンジンメタデータ
- `engines/engine_factory.py` - エンジン生成
- `tests/utils/text_normalization.py` - テキスト正規化

---

## 13. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| TenVAD ライセンス問題 | 商用利用不可 | 評価のみに使用、`benchmark-full` で分離 |
| 大規模モデルのメモリ不足 | テスト失敗 | エンジンごとにメモリ解放、順次実行 |
| 全組み合わせの実行時間 | CI タイムアウト | モード分離（quick/standard/full） |
| エンジン依存関係の競合 | インストール失敗 | `engines-nemo` と `engines-torch` を分離 |

---

## 14. 将来の拡張

### 14.1 ノイズ耐性評価

- DEMAND ノイズデータセットとの混合
- SNR 別の精度評価

### 14.2 リアルタイム性能評価

- ストリーミング処理のレイテンシ測定
- メモリ使用量の時系列分析

### 14.3 多言語拡張

- de, fr, es などの追加言語
- 言語検出精度の評価

---

## 15. 参考資料

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [JaVAD GitHub](https://github.com/skrbnv/javad)
- [TenVAD GitHub](https://github.com/TEN-framework/ten-vad)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- `engines/metadata.py` - エンジンメタデータ定義
- `docs/reference/vad-comparison.md` - VAD 比較調査
- `tests/integration/engines/test_smoke_engines.py` - 既存エンジンテスト
