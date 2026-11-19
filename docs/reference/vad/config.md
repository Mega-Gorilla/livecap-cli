# VADConfig リファレンス

> **最終更新:** 2025-12-01
> **関連:** [VAD バックエンドリファレンス](./backends.md), [VAD バックエンド比較](./comparison.md)

`VADConfig` は VAD の共通パラメータを設定するためのデータクラスです。

---

## 概要

```python
from livecap_cli.vad import VADConfig

# デフォルト設定
config = VADConfig()

# カスタム設定
config = VADConfig(
    threshold=0.6,
    min_speech_ms=300,
    min_silence_ms=150,
    speech_pad_ms=100,
)
```

---

## パラメータ一覧

### 基本パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `threshold` | `float` | `0.5` | 音声判定閾値 (0.0-1.0) |
| `neg_threshold` | `float \| None` | `None` | 非音声判定閾値（`None` = `threshold - 0.15`） |
| `min_speech_ms` | `int` | `250` | 音声と判定する最小継続時間（ms） |
| `min_silence_ms` | `int` | `100` | 音声終了と判定する無音継続時間（ms） |
| `speech_pad_ms` | `int` | `100` | 発話前後のパディング（ms） |
| `max_speech_ms` | `int` | `0` | 最大発話時間（0 = 無制限）（ms）。日本語の preset は `15000` (下の注) |

> **Note**: ミリ秒の値は、backend の 1 フレームの長さでフレーム数に直してから使います (Silero 32 ms、TenVAD 16 ms (`hop_size` 256) / 10 ms (160)、WebRTC 10 / 20 / 30 ms)。フレームの長さで割り切れない分は切り捨てで、最低 1 フレームです ([#507](https://github.com/Mega-Gorilla/livecap-cli/issues/507) より前は、どの backend でも 32 ms として数えていました)。
>
> **Note**: `max_speech_ms` は、前後の余白を含めた区間の長さの上限です。ほかの時間の設定と同じくフレーム数に切り捨てて数えるので、区間はこの長さを超えません (Silero の 32 ms では 15,000 ms は 468 フレーム = 14.976 秒)。上限で切ったときに話し続けていれば、次の区間も確定した発話として続けます (最短の発話の判定をやり直さないので、切った直後に短く終わる発話も ASR に渡ります)。次の区間は、切った時刻から隙間なく始まります。ただし、切った瞬間に話し終えて、続きの区間に発話のフレームが 1 つも無いまま終われば、その区間は出しません (余白と背景音だけの区間を ASR に渡さない)。
>
> **Note**: 日本語の preset (`--vad auto` など) は、区間を最長 15 秒で区切ります (`livecap_cli.vad.presets.MAX_SPEECH_MS_BY_LANGUAGE`)。上限が無いと、配信のような長い音声で区間が数分に伸び、言葉が落ちて、リアルタイムでは字幕が数分確定しませんでした ([#488](https://github.com/Mega-Gorilla/livecap-cli/issues/488))。

### 中間結果パラメータ（livecap-core 独自）

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `interim_min_duration_ms` | `int` | `2000` | 中間結果送信を開始する最小発話時間（ms） |
| `interim_interval_ms` | `int` | `1000` | 中間結果の送信間隔（ms） |

---

## パラメータ詳細

### threshold

音声と判定する確率の閾値です。VAD バックエンドの出力がこの値以上の場合、音声として判定されます。

```
threshold ↑ → 音声検出が厳格に → 短い発話やノイズを除外
threshold ↓ → 音声検出が緩く   → ノイズも音声として検出
```

| 値 | 推奨用途 |
|----|---------|
| `0.2-0.3` | 静かな環境、小声の検出 |
| `0.4-0.5` | 一般的な環境（デフォルト） |
| `0.6-0.7` | ノイズの多い環境 |

> **Note**: WebRTC VAD はバイナリ出力（0.0 or 1.0）のため、`threshold` は実質的に効果がありません。

### neg_threshold

音声から非音声への遷移を判定する閾値です。`None` の場合、`threshold - 0.15` が使用されます。

```python
# 明示的に設定
config = VADConfig(threshold=0.5, neg_threshold=0.35)

# 自動計算を使用
config = VADConfig(threshold=0.5)  # neg_threshold は 0.35
```

### min_speech_ms

音声として認識される最小継続時間です。この時間より短い音声は無視されます。

```
min_speech_ms ↑ → 短い発話（「はい」「うん」）を無視
min_speech_ms ↓ → 短い音も検出 → ノイズ誤検出の可能性
```

| 値 | 推奨用途 |
|----|---------|
| `100-200ms` | 短い応答も検出したい場合 |
| `250-350ms` | 一般的な発話（デフォルト） |
| `400-500ms` | 完全な文のみ検出したい場合 |

### min_silence_ms

発話終了と判定する無音継続時間です。この時間無音が続くと、発話区間が確定されます。

```
min_silence_ms ↑ → 長い間（ポーズ）も同一発話として扱う
min_silence_ms ↓ → 短い間で発話を分割 → セグメントが細かくなる
```

| 値 | 推奨用途 |
|----|---------|
| `50-100ms` | 細かいセグメント分割 |
| `100-200ms` | 一般的な発話（デフォルト） |
| `200-300ms` | 長めの文章をまとめたい場合 |

### speech_pad_ms

確定した発話区間の前後に追加するパディングです。文頭・文末の切れを防ぎます。

```
speech_pad_ms ↑ → 発話前後に余裕 → 文頭・文末の欠落防止
speech_pad_ms ↓ → タイトな切り出し → レイテンシ改善
```

| 値 | 推奨用途 |
|----|---------|
| `50-80ms` | 低レイテンシ重視 |
| `100-150ms` | 一般的な発話（デフォルト） |
| `150-200ms` | 文頭・文末の精度重視 |

---

## 最適化済みプリセット

Bayesian 最適化（Optuna）によって調整されたプリセットが利用可能です。

```python
from livecap_cli.vad.presets import get_optimized_preset, get_available_presets
from livecap_cli.vad import VADConfig

# 特定の VAD + 言語 + エンジンのプリセットを取得
preset = get_optimized_preset("silero", "ja", "parakeet_ja")
if preset:
    config = VADConfig.from_dict(preset["vad_config"])
    print(f"Score: {preset['metadata']['score']}")  # CER/WER

# エンジン指定なし → 全エンジン中の最良スコアを返す
preset = get_optimized_preset("silero", "ja")

# 利用可能なプリセット一覧
for vad_type, lang, engine in get_available_presets():
    print(f"{vad_type}/{lang}/{engine}")
```

### 利用可能なプリセット

プリセットはエンジンごとに最適化されています（全21件: 3 VAD x 2 言語 x 複数エンジン）。

> **推奨**: `VADProcessor.from_language()` を使用すると、言語に最適な VAD とプリセットが自動選択されます。
> エンジン固有の最適化には `VADProcessor.from_language("ja", engine="parakeet_ja")` を使用します。

---

## ユーティリティメソッド

### from_dict

辞書から `VADConfig` を作成します。

```python
config = VADConfig.from_dict({
    "threshold": 0.6,
    "min_speech_ms": 300,
})
```

### to_dict

`VADConfig` を辞書に変換します。

```python
config = VADConfig(threshold=0.6)
config_dict = config.to_dict()
# {'threshold': 0.6, 'neg_threshold': None, 'min_speech_ms': 250, ...}
```

### get_neg_threshold

有効な `neg_threshold` を返します（`None` の場合は計算値）。

```python
config = VADConfig(threshold=0.5)
neg = config.get_neg_threshold()  # 0.35 (0.5 - 0.15)
```

---

## 使用例

### 基本的な使用

```python
from livecap_cli.vad import VADProcessor, VADConfig

# カスタム設定で VADProcessor を作成
config = VADConfig(
    threshold=0.5,
    min_speech_ms=250,
    min_silence_ms=100,
    speech_pad_ms=100,
)
processor = VADProcessor(config=config)
```

### 環境別設定

```python
# ノイズ環境向け（厳しめ）
noisy_config = VADConfig(
    threshold=0.7,
    min_speech_ms=400,
    min_silence_ms=300,
    speech_pad_ms=50,
)

# 静かな環境向け（緩め）
quiet_config = VADConfig(
    threshold=0.3,
    min_speech_ms=150,
    min_silence_ms=80,
    speech_pad_ms=150,
)

# 低レイテンシ向け
fast_config = VADConfig(
    threshold=0.5,
    min_speech_ms=150,
    min_silence_ms=50,
    speech_pad_ms=30,
)
```

### StreamTranscriber との統合

```python
from livecap_cli import StreamTranscriber, MicrophoneSource, EngineFactory
from livecap_cli.vad import VADProcessor, VADConfig

# エンジン初期化
engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

# カスタム VAD 設定
config = VADConfig(threshold=0.6, min_speech_ms=300)
vad_processor = VADProcessor(config=config)

# StreamTranscriber に注入
with StreamTranscriber(engine=engine, vad_processor=vad_processor) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(result.text)
```

---

## 関連ドキュメント

- [VAD バックエンドリファレンス](./backends.md) - バックエンド固有パラメータ
- [VAD バックエンド比較](./comparison.md) - ベンチマーク結果
- [VAD Bayesian 最適化ガイド](../../guides/benchmark/vad-optimization.md) - カスタムパラメータチューニング
- [リアルタイム文字起こしガイド](../../guides/realtime-transcription.md) - 使い方ガイド

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
