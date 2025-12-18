# VAD バックエンドリファレンス

> **最終更新:** 2025-12-01
> **関連:** [VADConfig リファレンス](./config.md), [VAD バックエンド比較](./comparison.md)

livecap-core で利用可能な VAD（Voice Activity Detection）バックエンドの詳細リファレンス。

---

## 概要

livecap-core は 3 つの VAD バックエンドをサポートしています：

| バックエンド | クラス | 特徴 | 推奨用途 |
|-------------|--------|------|---------|
| Silero VAD | `SileroVAD` | 高精度、ディープラーニングベース | デフォルト、汎用 |
| WebRTC VAD | `WebRTCVAD` | 軽量、C拡張ベース | 低リソース環境 |
| TenVAD | `TenVAD` | 高速、軽量 | 低レイテンシ環境 |

### 言語別推奨バックエンド

Bayesian 最適化によるパラメータチューニング結果（#126）に基づく推奨：

| 言語 | 推奨バックエンド | スコア | 備考 |
|------|-----------------|--------|------|
| 日本語 | **TenVAD** | 7.2% CER | パラメータ調整で最適スコア |
| 英語 | **WebRTC** | 3.3% WER | パラメータ調整で最適スコア |

> **注意**: これらの推奨は各 VAD バックエンドが言語固有に最適化されているわけではなく、Bayesian 最適化によるパラメータチューニングの結果、当該言語で最も良いスコアを記録したバックエンド + パラメータの組み合わせです。

```python
# 言語に最適化された VAD を自動選択（推奨）
from livecap_cli.vad import VADProcessor

processor = VADProcessor.from_language("ja")  # → TenVAD
processor = VADProcessor.from_language("en")  # → WebRTC
```

---

## Silero VAD

### 概要

Silero VAD は PyTorch/ONNX ベースのディープラーニング VAD です。高精度で汎用性が高く、livecap-core のデフォルトバックエンドです。

| 項目 | 値 |
|------|-----|
| フレームサイズ | 512 samples (32ms @ 16kHz) |
| 出力 | 確率値 (0.0 - 1.0) |
| 依存パッケージ | `silero-vad`, `torch` |
| サンプルレート | 16kHz |

### コンストラクタ

```python
from livecap_cli.vad.backends import SileroVAD

vad = SileroVAD(
    threshold=0.5,  # 音声判定閾値（参考値、実際は VADConfig で制御）
    onnx=True,      # ONNX ランタイム使用（推奨）
)
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `threshold` | `float` | `0.5` | 音声判定閾値（VADConfig で上書き可能） |
| `onnx` | `bool` | `True` | ONNX ランタイムを使用するか |

### 使用例

```python
from livecap_cli.vad import VADProcessor
from livecap_cli.vad.backends import SileroVAD

# デフォルト設定で使用
processor = VADProcessor()  # Silero VAD がデフォルト

# 明示的に指定
processor = VADProcessor(backend=SileroVAD(onnx=True))
```

### 注意事項

- 初回ロード時にモデルダウンロードが発生します（約 2MB）
- GPU 使用時は `onnx=False` でネイティブ PyTorch を使用可能
- メモリ使用量は他のバックエンドより大きい

---

## WebRTC VAD

### 概要

WebRTC VAD は Google WebRTC プロジェクトの C 拡張ベース VAD です。軽量で高速、バイナリ（音声/非音声）出力を返します。

| 項目 | 値 |
|------|-----|
| フレームサイズ | 160/320/480 samples (10/20/30ms @ 16kHz) |
| 出力 | バイナリ (0.0 or 1.0) |
| 依存パッケージ | `webrtcvad` |
| サンプルレート | 8kHz, 16kHz, 32kHz, 48kHz |

### コンストラクタ

```python
from livecap_cli.vad.backends import WebRTCVAD

vad = WebRTCVAD(
    mode=3,              # 積極性レベル (0-3)
    frame_duration_ms=20, # フレーム長 (10, 20, 30ms)
)
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `mode` | `int` | `3` | 積極性レベル（下記参照） |
| `frame_duration_ms` | `int` | `20` | フレーム長（10, 20, 30ms のいずれか） |

#### mode パラメータ

| mode | 説明 | 特徴 |
|------|------|------|
| `0` | 最も寛容 | 誤検出少、見逃し多 |
| `1` | やや厳格 | バランス型 |
| `2` | 厳格 | やや厳格 |
| `3` | 最も厳格 | 誤検出多、見逃し少 |

### 使用例

```python
from livecap_cli.vad import VADProcessor, VADConfig
from livecap_cli.vad.backends import WebRTCVAD

# mode=0（寛容）で英語向けに最適化
processor = VADProcessor(
    backend=WebRTCVAD(mode=0, frame_duration_ms=30),
    config=VADConfig(
        min_speech_ms=450,
        min_silence_ms=280,
        speech_pad_ms=200,
    ),
)
```

### 注意事項

- 出力がバイナリ（0.0 or 1.0）のため、`threshold` パラメータは効果がありません
- 状態を持たないため、`reset()` は何もしません
- C 拡張のため、プラットフォームによってはビルドが必要

---

## TenVAD

### 概要

TenVAD は TEN Framework の軽量 VAD です。高速で低レイテンシが特徴です。

| 項目 | 値 |
|------|-----|
| フレームサイズ | 160/256 samples (10/16ms @ 16kHz) |
| 出力 | 確率値 (0.0 - 1.0) |
| 依存パッケージ | `ten-vad` |
| サンプルレート | 16kHz |

### コンストラクタ

```python
from livecap_cli.vad.backends import TenVAD

vad = TenVAD(
    hop_size=256,    # フレームサイズ (160 or 256)
    threshold=0.5,   # 音声判定閾値
)
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `hop_size` | `int` | `256` | フレームサイズ（160 or 256 samples） |
| `threshold` | `float` | `0.5` | 音声判定閾値 |

#### hop_size パラメータ

| hop_size | 時間 | 特徴 |
|----------|------|------|
| `160` | 10ms | より細かい粒度 |
| `256` | 16ms | デフォルト、バランス型 |

### 使用例

```python
from livecap_cli.vad import VADProcessor, VADConfig
from livecap_cli.vad.backends import TenVAD
import warnings

# ライセンス警告を抑制（内容を理解した上で）
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    processor = VADProcessor(
        backend=TenVAD(hop_size=256),
        config=VADConfig(
            threshold=0.204,
            min_speech_ms=400,
            min_silence_ms=180,
            speech_pad_ms=90,
        ),
    )
```

### 注意事項

> **ライセンス警告**: TenVAD は商用利用に制限があります。使用前にライセンスを確認してください。
> https://github.com/TEN-framework/ten-vad

- Linux では `libc++` が必要: `sudo apt-get install libc++1`
- インスタンス作成時にライセンス警告が表示されます
- `reset()` は内部的にインスタンスを再作成します

---

## カスタムバックエンドの実装

`VADBackend` Protocol を実装することで、独自の VAD バックエンドを追加できます。

### VADBackend Protocol

```python
from typing import Protocol
import numpy as np

class VADBackend(Protocol):
    def process(self, audio: np.ndarray) -> float:
        """音声を処理して VAD 確率を返す (0.0-1.0)"""
        ...

    def reset(self) -> None:
        """内部状態をリセット"""
        ...

    @property
    def frame_size(self) -> int:
        """16kHz での推奨フレームサイズ（samples）"""
        ...

    @property
    def name(self) -> str:
        """バックエンド識別子"""
        ...
```

### 実装例

```python
import numpy as np
from livecap_cli.vad import VADProcessor

class MyVAD:
    """カスタム VAD の実装例"""

    def process(self, audio: np.ndarray) -> float:
        # 簡易的なエネルギーベース VAD
        energy = np.sqrt(np.mean(audio ** 2))
        return min(1.0, energy * 10)

    def reset(self) -> None:
        pass

    @property
    def frame_size(self) -> int:
        return 512  # 32ms @ 16kHz

    @property
    def name(self) -> str:
        return "my_vad"

# 使用
processor = VADProcessor(backend=MyVAD())
```

---

## 関連ドキュメント

- [VADConfig リファレンス](./config.md) - 共通パラメータの詳細
- [VAD バックエンド比較](./comparison.md) - ベンチマーク結果
- [リアルタイム文字起こしガイド](../../guides/realtime-transcription.md) - 使い方ガイド

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
