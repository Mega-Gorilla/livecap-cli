# 統合ベンチマークフレームワーク実装計画

> **作成日:** 2025-11-25
> **関連 Issue:** #86
> **ステータス:** 計画中

---

## 1. 概要

### 1.1 目的

livecap-cli の音声認識パイプライン全体を評価するための**統合ベンチマークフレームワーク**を構築する。

**VAD ベンチマーク + ASR ベンチマークを同時実装**し、以下を実現：

- 複数の VAD バックエンド（11構成）を比較評価
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
| VAD × 全ASR評価（11 VAD × 10 ASR） | VAD/ASR の本番切り替え |
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
    └── backends/              # VAD バックエンド
        ├── __init__.py
        ├── base.py            # VADBackend Protocol
        ├── silero.py
        ├── tenvad.py
        ├── javad.py
        └── webrtc.py
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
│  VAD (11構成)          ASR (10エンジン)         言語 (2+)       │
│  ┌─────────────┐      ┌─────────────────┐      ┌──────────┐    │
│  │ Silero v5   │      │ reazonspeech    │──ja──│ Japanese │    │
│  │ Silero v6   │      │ parakeet_ja     │──ja──│          │    │
│  │ TenVAD      │  ×   │ parakeet        │──en──│ English  │    │
│  │ JaVAD tiny  │      │ canary          │──en──│          │    │
│  │ JaVAD bal.  │      │ voxtral         │──en──│          │    │
│  │ JaVAD prec. │      │ whispers2t_*    │──all─│          │    │
│  │ WebRTC 0-3  │      └─────────────────┘      └──────────┘    │
│  └─────────────┘                                                 │
│                                                                  │
│  Full Matrix: 11 VAD × 10 ASR × 2 Lang = 220 combinations       │
│  Practical:   11 VAD × 3-4 ASR/lang × 2 Lang ≈ 66-88 tests     │
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

**合計 11 構成**:

| VAD | モデル/設定 | ライセンス | 特徴 |
|-----|------------|-----------|------|
| Silero VAD v5 | ONNX | MIT | 旧バージョン、比較用 |
| Silero VAD v6 | ONNX | MIT | 現在のデフォルト |
| TenVAD | - | 独自 | 最軽量・最高速（評価のみ） |
| JaVAD | tiny | MIT | 0.64s window、即時検出向け |
| JaVAD | balanced | MIT | 1.92s window、バランス型 |
| JaVAD | precise | MIT | 3.84s window、最高精度 |
| WebRTC VAD | mode 0 | BSD | 最も寛容、誤検出少 |
| WebRTC VAD | mode 1 | BSD | やや厳格 |
| WebRTC VAD | mode 2 | BSD | 厳格 |
| WebRTC VAD | mode 3 | BSD | 最も厳格、見逃し多 |

### 5.2 VAD バックエンド Protocol

```python
# benchmarks/vad/backends/base.py
from typing import Protocol, List, Tuple
import numpy as np

class VADBackend(Protocol):
    """VAD バックエンドの共通インターフェース"""

    @property
    def name(self) -> str:
        """バックエンド名"""
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

    def reset(self) -> None:
        """状態をリセット"""
        ...
```

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

### 7.1 既存テストアセット

```
tests/assets/audio/
├── jsut_basic5000_0001_ja.wav      # 日本語（約3秒）
├── jsut_basic5000_0001_ja.txt      # トランスクリプト
├── librispeech_test-clean_1089-134686-0001_en.wav  # 英語（約4秒）
└── librispeech_test-clean_1089-134686-0001_en.txt  # トランスクリプト
```

### 7.2 拡張データセット（オプション）

```bash
export LIVECAP_JSUT_DIR=/path/to/jsut/jsut_ver1.1
export LIVECAP_LIBRISPEECH_DIR=/path/to/librispeech/test-clean
```

---

## 8. 実装ステップ

### 概要: 効率的な実装順序

```
Step 1: 共通基盤 (common/)     ─┐
                                ├─ ASRベンチマーク完成
Step 2: ASR ランナー (asr/)    ─┘

Step 3: VAD バックエンド        ─┐
                                ├─ VADベンチマーク完成
Step 4: VAD ランナー (vad/)    ─┘

Step 5: CLI 統合
Step 6: CI 統合
```

### Step 1: 共通基盤構築 (2-3日)

**対象:** `benchmarks/common/`

1. **metrics.py** - WER/CER/RTF 計算
   - jiwer ライブラリ活用
   - テキスト正規化（既存 `tests/utils/text_normalization.py` 活用）

2. **datasets.py** - データセット管理
   - 音声ファイル + トランスクリプト読み込み
   - 言語自動検出

3. **engines.py** - ASR エンジン管理
   - EngineFactory ラッパー
   - VAD 無効化設定

4. **reports.py** - レポート生成
   - JSON 出力
   - Markdown 出力
   - コンソール表形式出力

### Step 2: ASR ベンチマーク実装 (1日)

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

### Step 3: VAD バックエンド実装 (2日)

**対象:** `benchmarks/vad/backends/`

1. **base.py** - VADBackend Protocol
2. **silero.py** - Silero VAD v5/v6
3. **javad.py** - JaVAD tiny/balanced/precise
4. **webrtc.py** - WebRTC VAD mode 0-3
5. **tenvad.py** - TenVAD

### Step 4: VAD ベンチマーク実装 (1日)

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

### Step 5: CLI 統合 (0.5日)

```bash
# ASR ベンチマーク
python -m benchmarks.asr --all
python -m benchmarks.asr --engine reazonspeech parakeet_ja --language ja
python -m benchmarks.asr --mode quick --output results.json

# VAD ベンチマーク
python -m benchmarks.vad --all
python -m benchmarks.vad --vad silero_v6 javad_precise --asr reazonspeech
python -m benchmarks.vad --mode standard --format markdown

# 両方実行
python -m benchmarks --all --output report.md
```

### Step 6: CI 統合 (0.5日)

GitHub Actions ワークフロー作成（詳細は後述）。

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
Mode: Standard (11 VAD × 3 ASR/lang)

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
| `standard` | 10 (言語別全) | 44-66 (11 VAD × 2-3 ASR × 2 lang) | ~20分 |
| `full` | 20 (全組み合わせ) | 88+ (11 VAD × 全ASR × 2 lang) | ~60分 |

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
