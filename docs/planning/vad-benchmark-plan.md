# VAD ベンチマーク実装計画

> **作成日:** 2025-11-25
> **関連 Issue:** #86
> **ステータス:** 計画中

---

## 1. 概要

### 1.1 目的

複数の VAD（Voice Activity Detection）バックエンドを比較評価し、livecap-cli に最適な VAD を選定するためのベンチマークモジュールを実装する。

### 1.2 背景

- Phase 1 で Silero VAD をデフォルトとして採用
- `docs/reference/vad-comparison.md` の調査により、他の VAD（JaVAD, TenVAD）が優れている可能性が判明
- ベンチマーク結果はデータセットに強く依存するため、実際の使用環境での評価が必要

### 1.3 スコープ

| 含む | 含まない |
|------|----------|
| End-to-End 評価（VAD → ASR → WER/CER） | 本番環境への VAD 切り替え |
| 4つの VAD の比較 | 新しい VAD の実装 |
| 日本語・英語での評価 | 全言語での評価 |
| CLI ベンチマークツール | GUI |

---

## 2. 評価方法

### 2.1 End-to-End 評価（メイン）

ASR の最終精度で VAD を評価する方法。**追加アノテーション不要**。

```
┌─────────────────────────────────────────────────────────────┐
│ End-to-End VAD 評価フロー                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  テスト音声 (.wav)                                           │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ VAD (Silero / TenVAD / JaVAD / WebRTC)              │    │
│  │ → 音声セグメント検出                                  │    │
│  │ → [(start, end, audio), ...]                        │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ WhisperS2T (use_vad=False)                          │    │
│  │ → 各セグメントを文字起こし                            │    │
│  │ → 結果を結合                                         │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 評価                                                 │    │
│  │ → WER/CER 計算 (vs transcript Ground Truth)         │    │
│  │ → RTF 計測                                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Ground Truth**: 既存の `tests/assets/audio/*.txt` トランスクリプトファイル

### 2.2 フレームレベル評価（オプション）

VAD 自体の音声検出精度を評価する方法。

```
┌─────────────────────────────────────────────────────────────┐
│ Ground Truth 生成                                            │
├─────────────────────────────────────────────────────────────┤
│ クリーン音声 → エネルギーベース VAD → 音声区間アノテーション    │
│            → JSON ファイルとして保存                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ フレームレベル評価                                            │
├─────────────────────────────────────────────────────────────┤
│ テスト音声 → VAD → フレーム単位の speech/non-speech 判定      │
│           → Ground Truth と比較                              │
│           → Precision / Recall / F1 / ROC-AUC               │
└─────────────────────────────────────────────────────────────┘
```

**参考**: [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics) では 31.25ms フレームで評価

---

## 3. 評価指標

### 3.1 ASR 精度指標

| 指標 | 説明 | 計算方法 |
|------|------|----------|
| **WER** | Word Error Rate | `(S + D + I) / N` |
| **CER** | Character Error Rate | 文字単位の WER（日本語向け） |

- `S`: 置換数、`D`: 削除数、`I`: 挿入数、`N`: 参照単語数
- ライブラリ: [jiwer](https://github.com/jitsi/jiwer)

### 3.2 VAD 性能指標

| 指標 | 説明 | 単位 |
|------|------|------|
| **RTF** | Real-Time Factor | 比率（低いほど高速） |
| **Latency** | 入力→出力の遅延 | ms |
| **Memory** | ピークメモリ使用量 | MB |
| **Segments** | 検出セグメント数 | 個 |
| **Avg Duration** | 平均セグメント長 | 秒 |

### 3.3 フレームレベル指標（オプション）

| 指標 | 説明 |
|------|------|
| **Precision** | 検出音声のうち実際に音声だった割合 |
| **Recall** | 実際の音声のうち検出された割合 |
| **F1** | Precision と Recall の調和平均 |
| **ROC-AUC** | 閾値を変えた時の検出率 vs 誤検出率 |

---

## 4. 比較対象 VAD

### 4.1 一覧

| VAD | バージョン | ライセンス | 採用可否 | 備考 |
|-----|-----------|-----------|---------|------|
| **Silero VAD** | v6 | MIT | ✅ | 現在のデフォルト |
| **TenVAD** | latest | 独自 | ❌ | 評価のみ（ライセンス問題） |
| **JaVAD** | precise | MIT | ✅ | AVA-Speech で最高精度 |
| **WebRTC VAD** | 2.0.10 | BSD | ✅ | ベースライン |

### 4.2 インストール

```toml
# pyproject.toml
[project.optional-dependencies]
"benchmark-vad" = [
    "silero-vad>=5.1",
    "javad",
    "webrtcvad",
    "ten-vad",
    "matplotlib",
    "pandas",
    "jiwer",
]
```

### 4.3 各 VAD の特徴

#### Silero VAD v6

```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad(onnx=True)
vad = VADIterator(
    model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
)
```

- **チャンクサイズ**: 512 samples (32ms) @ 16kHz
- **出力**: `{'start': float, 'end': float}` または `None`

#### TenVAD

```python
import tenvad

vad = tenvad.create()
# 16kHz, 16-bit PCM
result = vad.process(audio_bytes)
```

- **特徴**: 最軽量（300-500KB）、最高速
- **注意**: ライセンスが複雑

#### JaVAD

```python
from javad import JaVAD

vad = JaVAD(model="precise")  # tiny, balanced, precise
segments = vad.get_speech_timestamps(audio, sampling_rate=16000)
```

- **モデル**: tiny, balanced, precise
- **特徴**: AVA-Speech で最高精度

#### WebRTC VAD

```python
import webrtcvad

vad = webrtcvad.Vad(mode=3)  # 0-3 (aggressive)
is_speech = vad.is_speech(frame, sample_rate)
```

- **フレームサイズ**: 10, 20, or 30ms
- **特徴**: 軽量、実績あり、精度は低め

---

## 5. データセット

### 5.1 既存テストアセット

```
tests/assets/audio/
├── jsut_basic5000_0001_ja.wav      # 日本語（約3秒）
├── jsut_basic5000_0001_ja.txt      # トランスクリプト
├── librispeech_test-clean_1089-134686-0001_en.wav  # 英語（約4秒）
└── librispeech_test-clean_1089-134686-0001_en.txt  # トランスクリプト
```

### 5.2 拡張データセット（オプション）

環境変数で大規模コーパスを指定可能:

```bash
export LIVECAP_JSUT_DIR=/path/to/jsut/jsut_ver1.1
export LIVECAP_LIBRISPEECH_DIR=/path/to/librispeech/test-clean
```

### 5.3 ノイズ混合（将来対応）

```python
# DEMAND ノイズデータセットとの混合
noisy_audio = mix_with_noise(clean_audio, noise, snr_db=10)
```

---

## 6. 実装計画

### 6.1 ディレクトリ構造

```
benchmarks/
└── vad/
    ├── __init__.py
    ├── runner.py           # ベンチマーク実行
    ├── metrics.py          # WER/CER/RTF 計算
    ├── datasets.py         # データセット管理
    ├── cli.py              # CLI エントリポイント
    ├── backends/
    │   ├── __init__.py
    │   ├── base.py         # VADBackend Protocol
    │   ├── silero.py
    │   ├── tenvad.py
    │   ├── javad.py
    │   └── webrtc.py
    └── reports/
        ├── __init__.py
        └── generate.py     # Markdown/HTML レポート生成
```

### 6.2 コアインターフェース

#### VADBackend Protocol

```python
from typing import Protocol, List, Tuple
import numpy as np

class VADBackend(Protocol):
    """VAD バックエンドのインターフェース"""

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

        Args:
            audio: 音声データ (float32, mono)
            sample_rate: サンプリングレート

        Returns:
            [(start_sec, end_sec), ...] のリスト
        """
        ...

    def reset(self) -> None:
        """状態をリセット"""
        ...
```

#### BenchmarkRunner

```python
@dataclass
class BenchmarkResult:
    vad_name: str
    audio_file: str
    language: str

    # ASR 結果
    transcript: str
    reference: str
    wer: float
    cer: float

    # VAD 性能
    rtf: float
    num_segments: int
    avg_segment_duration: float
    peak_memory_mb: float

class BenchmarkRunner:
    def __init__(
        self,
        backends: List[VADBackend],
        asr_engine: TranscriptionEngine,
        dataset: Dataset,
    ):
        ...

    def run(self) -> List[BenchmarkResult]:
        ...
```

### 6.3 CLI 使用例

```bash
# 全 VAD をベンチマーク
python -m benchmarks.vad --all

# 特定 VAD のみ
python -m benchmarks.vad --vad silero javad

# 言語指定
python -m benchmarks.vad --language ja

# 拡張データセット
python -m benchmarks.vad --dataset jsut-full

# レポート出力
python -m benchmarks.vad --output report.md --format markdown

# JSON 出力（CI 用）
python -m benchmarks.vad --output results.json --format json
```

---

## 7. 出力フォーマット

### 7.1 コンソール出力

```
=== VAD Benchmark Results ===

Dataset: tests/assets/audio (2 files)
ASR Engine: WhisperS2T Base (use_vad=False)

| VAD         | WER (ja) | CER (ja) | WER (en) | CER (en) | RTF    | Segments |
|-------------|----------|----------|----------|----------|--------|----------|
| Silero v6   | 5.2%     | 2.1%     | 4.8%     | 3.2%     | 0.012  | 3        |
| TenVAD      | 4.9%     | 1.9%     | 4.5%     | 2.9%     | 0.008  | 4        |
| JaVAD       | 4.1%     | 1.5%     | 4.2%     | 2.7%     | 0.015  | 3        |
| WebRTC      | 8.3%     | 4.2%     | 7.1%     | 5.1%     | 0.003  | 6        |

Best overall: JaVAD (lowest average WER)
Fastest: WebRTC (RTF: 0.003)
```

### 7.2 Markdown レポート

```markdown
# VAD Benchmark Report

Generated: 2025-11-25 12:00:00

## Summary

| Metric | Silero v6 | TenVAD | JaVAD | WebRTC |
|--------|-----------|--------|-------|--------|
| Avg WER | 5.0% | 4.7% | 4.2% | 7.7% |
| Avg RTF | 0.012 | 0.008 | 0.015 | 0.003 |

## Detailed Results

### Japanese (jsut_basic5000_0001_ja.wav)
...

### English (librispeech_test-clean_1089-134686-0001_en.wav)
...
```

---

## 8. 実装ステップ

### Step 1: 基盤構築

1. `benchmarks/vad/` ディレクトリ作成
2. `VADBackend` Protocol 定義
3. データセットローダー実装
4. 基本的な CLI 実装

### Step 2: VAD ラッパー実装

1. Silero VAD ラッパー（既存コード活用）
2. WebRTC VAD ラッパー
3. JaVAD ラッパー
4. TenVAD ラッパー

### Step 3: 評価実装

1. WER/CER 計算（jiwer 使用）
2. RTF 計測
3. メモリ使用量計測

### Step 4: レポート生成

1. コンソール出力
2. Markdown 出力
3. JSON 出力

### Step 5: CI 統合（オプション）

1. GitHub Actions でのベンチマーク実行
2. 結果の自動コミット

---

## 9. 依存関係

### 9.1 必須

```
silero-vad>=5.1
webrtcvad>=2.0.10
javad
jiwer>=3.0
numpy
```

### 9.2 オプション

```
ten-vad          # TenVAD（ライセンス注意）
matplotlib       # グラフ生成
pandas           # データ集計
memory_profiler  # メモリ計測
```

---

## 10. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| TenVAD ライセンス問題 | 商用利用不可 | 評価のみに使用、採用候補から除外 |
| JaVAD インストール失敗 | ベンチマーク不完全 | PyTorch 依存を明記、スキップ可能に |
| データセットが少ない | 統計的信頼性低い | 拡張データセット対応を実装 |
| ASR 精度がボトルネック | VAD 差異が見えにくい | 複数 ASR エンジンでの評価検討 |

---

## 11. 参考資料

- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [Silero VAD Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [JaVAD GitHub](https://github.com/skrbnv/javad)
- [TenVAD GitHub](https://github.com/TEN-framework/ten-vad)
- [Anwarvic/VAD_Benchmark](https://github.com/Anwarvic/VAD_Benchmark)
- [AVA-Speech Dataset](https://research.google.com/ava/download.html)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- `docs/reference/vad-comparison.md`
