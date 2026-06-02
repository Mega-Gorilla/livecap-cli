# API リファレンス

livecap-cli をライブラリとして使用するための API リファレンスです。

## 目次

- [インポート](#インポート)
- [EngineFactory](#enginefactory)
- [MicrophoneSource](#microphonesource)
- [DeviceInfo](#deviceinfo)
- [StreamTranscriber](#streamtranscriber)
- [TranscriptionResult](#transcriptionresult)
- [VADConfig](#vadconfig)
- [NoiseGate](#noisegate)
- [NoiseAnalysis / analyze_noise_samples](#noiseanalysis--analyze_noise_samples)
- [FileTranscriptionPipeline](#filetranscriptionpipeline)

---

## インポート

```python
from livecap_cli import (
    # エンジン
    EngineFactory,
    EngineMetadata,
    EngineInfo,

    # 音声ソース
    MicrophoneSource,
    FileSource,
    AudioSource,
    DeviceInfo,

    # 文字起こし
    StreamTranscriber,
    TranscriptionResult,
    InterimResult,
    FileTranscriptionPipeline,

    # VAD
    VADConfig,
    VADProcessor,
    VADSegment,
    VADState,

    # エラー
    TranscriptionError,
    EngineError,
)

# Audio utilities (ノイズゲート / キャリブレーション)
from livecap_cli.audio import (
    NoiseGate,
    NoiseAnalysis,
    analyze_noise_samples,
)
```

---

## EngineFactory

ASR エンジンの作成・管理を行うファクトリークラス。

### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `get_available_engines()` | `Dict[str, Dict[str, str]]` | 利用可能なエンジン一覧を取得 |
| `get_engine_info(engine_type)` | `Optional[Dict[str, Any]]` | 特定エンジンの詳細情報を取得 |
| `get_engines_for_language(lang_code)` | `Dict[str, Dict[str, Any]]` | 指定言語に対応したエンジン一覧を取得 |
| `create_engine(engine_type, device, **options)` | `BaseEngine` | エンジンインスタンスを作成 |

### 使用例

```python
from livecap_cli import EngineFactory

# エンジン一覧を取得
engines = EngineFactory.get_available_engines()
for engine_id, info in engines.items():
    print(f"{engine_id}: {info['name']}")
    # 出力例: whispers2t: WhisperS2T

# 特定エンジンの詳細情報を取得
info = EngineFactory.get_engine_info("whispers2t")
print(info)
# {
#     'name': 'WhisperS2T',
#     'description': 'Multilingual ASR model with selectable model sizes...',
#     'supported_languages': ['en', 'ja', 'zh', ...],
#     'default_params': {'model_size': 'large-v3', ...},
#     'available_model_sizes': ['tiny', 'base', 'small', ...]
# }

# 日本語対応エンジンを取得
ja_engines = EngineFactory.get_engines_for_language("ja")
for engine_id in ja_engines:
    print(engine_id)  # reazonspeech, parakeet_ja, qwen3asr, whispers2t

# エンジンを作成
engine = EngineFactory.create_engine(
    "whispers2t",
    device="cuda",        # "cuda", "cpu", または None (自動検出)
    model_size="base",    # whispers2t 固有オプション
)
engine.load_model()       # モデルをロード（必須）
```

### `get_engine_info()` の戻り値

| キー | 型 | 説明 |
|-----|-----|------|
| `name` | `str` | エンジン表示名 |
| `description` | `str` | エンジンの説明 |
| `supported_languages` | `List[str]` | 対応言語コード一覧 |
| `default_params` | `Dict[str, Any]` | デフォルトパラメータ |
| `available_model_sizes` | `Optional[List[str]]` | 選択可能なモデルサイズ（whispers2t のみ） |

### 利用可能なエンジン

| ID | モデル | 言語 | 備考 |
|----|--------|------|------|
| `reazonspeech` | ReazonSpeech K2 v2 | ja | 日本語特化 |
| `parakeet` | Parakeet TDT 0.6B v2 | en | 英語特化 |
| `parakeet_ja` | Parakeet TDT CTC 0.6B JA | ja | 日本語特化 |
| `canary` | Canary 1B Flash | en, de, fr, es | 多言語 |
| `voxtral` | Voxtral Mini 3B | en, es, fr 等 | 多言語 |
| `qwen3asr` | Qwen3-ASR 0.6B | 30言語 | 多言語 |
| `whispers2t` | WhisperS2T | 99言語 | モデルサイズ選択可 |

---

## MicrophoneSource

マイク入力からの音声キャプチャを行うクラス。

### コンストラクタ

```python
MicrophoneSource(
    device: Optional[int | str] = None,  # デバイスインデックスまたは名前（None=デフォルト）
    sample_rate: int = 16000,            # サンプリングレート
    chunk_ms: int = 100,                 # チャンクサイズ（ミリ秒）
)
```

### クラスメソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `list_devices()` | `List[DeviceInfo]` | 利用可能なマイクデバイス一覧を取得 |

### 使用例

```python
from livecap_cli import MicrophoneSource

# デバイス一覧を取得
devices = MicrophoneSource.list_devices()
for dev in devices:
    default_mark = " (default)" if dev.is_default else ""
    print(f"[{dev.index}] {dev.name} (ch:{dev.channels}){default_mark}")

# マイクから音声をキャプチャ
with MicrophoneSource(device=0) as mic:
    for chunk in mic:  # 同期イテレータ
        process(chunk)  # numpy.ndarray (float32)

# 非同期使用
async with MicrophoneSource() as mic:
    async for chunk in mic:
        await process(chunk)
```

> **Note**: MicrophoneSource は PortAudio に依存しています。`sudo apt-get install libportaudio2` (Ubuntu) または `brew install portaudio` (macOS) が必要です。

---

## DeviceInfo

オーディオデバイス情報を格納する dataclass。

### 属性

| 属性 | 型 | 説明 |
|-----|-----|------|
| `index` | `int` | デバイスインデックス |
| `name` | `str` | デバイス名 |
| `channels` | `int` | 入力チャンネル数 |
| `sample_rate` | `int` | デフォルトサンプリングレート |
| `is_default` | `bool` | デフォルトデバイスかどうか |

### 使用例

```python
from livecap_cli import MicrophoneSource

devices = MicrophoneSource.list_devices()
for dev in devices:
    print(f"Index: {dev.index}")
    print(f"Name: {dev.name}")
    print(f"Channels: {dev.channels}")
    print(f"Sample Rate: {dev.sample_rate}")
    print(f"Is Default: {dev.is_default}")
```

---

## StreamTranscriber

VAD + ASR を組み合わせたストリーミング文字起こしクラス。

### コンストラクタ

```python
StreamTranscriber(
    engine: TranscriptionEngine,              # ASR エンジン（必須）
    translator: Optional[BaseTranslator] = None,  # 翻訳エンジン
    source_lang: Optional[str] = None,        # ソース言語（translator 使用時必須）
    target_lang: Optional[str] = None,        # ターゲット言語（translator 使用時必須）
    vad_config: Optional[VADConfig] = None,   # VAD 設定
    vad_processor: Optional[VADProcessor] = None,  # VAD プロセッサ（テスト用）
    source_id: str = "default",               # ソース識別子
    max_workers: int = 1,                     # ワーカースレッド数
    result_coalescer: Optional[ResultCoalescer] = None,  # 短文結合
    noise_gate: Optional[NoiseGate] = None,   # NoiseGate (pre-VAD per-sample peak)
    # === EnergyGate (#292; engine-input low-energy guard) ===
    engine_min_rms_dbfs: float = -45.0,       # threshold (dBFS); float("-inf") で opt-out
    engine_energy_metric: str = "max_frame_rms",  # 4 metric から選択
    engine_energy_frame_ms: float = 32.0,     # frame size (ms)
)
```

### パラメータ詳細

| パラメータ | 必須 | デフォルト | 説明 |
|-----------|:---:|-----------|------|
| `engine` | ✅ | - | 文字起こしエンジン（`load_model()` 済みであること） |
| `translator` | - | `None` | 翻訳エンジン（設定時は source_lang/target_lang 必須） |
| `source_lang` | - | `None` | 翻訳元言語コード（例: `"ja"`） |
| `target_lang` | - | `None` | 翻訳先言語コード（例: `"en"`） |
| `vad_config` | - | `None` | VAD 設定（vad_processor 未指定時に使用） |
| `vad_processor` | - | `None` | カスタム VAD プロセッサ（テスト用） |
| `source_id` | - | `"default"` | 音声ソースの識別子 |
| `max_workers` | - | `1` | 文字起こし用スレッドプールのワーカー数 |
| `noise_gate` | - | `None` | NoiseGate (pre-VAD)。per-sample peak envelope。 |
| `engine_min_rms_dbfs` | - | `-45.0` | EnergyGate threshold (per-segment RMS dBFS)。`float("-inf")` で opt-out。 |
| `engine_energy_metric` | - | `"max_frame_rms"` | EnergyGate の energy 指標。後述「Energy metric の選択」参照。 |
| `engine_energy_frame_ms` | - | `32.0` | frame-based metrics の窓長 (ms)。`whole_rms` では無視。 |

> ⚠ **物理量の警告**: `engine_min_rms_dbfs` は **per-segment / per-frame RMS** unit。
> `NoiseGate.threshold_db` (per-sample peak envelope) とは物理量が異なるため、
> 同じ値を共有してはいけません。`levels` コマンドはそれぞれ別の suggested 値
> (`suggested_threshold_db` / `suggested_engine_min_rms_dbfs`) を出力します。

### Energy metric の選択 (`engine_energy_metric`)

`_segment_energy_dbfs(audio, sample_rate, metric, frame_ms) -> float` (公開: `livecap_cli.audio._segment_energy_dbfs`) が per-segment energy を測定します。

| Metric | 説明 | Trade-off |
|---|---|---|
| `max_frame_rms` (default) | `frame_ms` 窓ごとの RMS の max | VAD padding 希釈に強い (短文発話/ささやきでも実 speech 部分を捕捉)。単発 transient で false-pass する可能性。 |
| `whole_rms` | segment 全体の RMS | Aggressive (最も多く drop)。padding 希釈リスクあり (短文発話で false-drop)。 |
| `p95_frame_rms` | `frame_ms` 窓ごとの RMS の 95%ile | max と whole の中庸。 |
| `top3_frame_rms` | `frame_ms` 窓ごとの RMS の top-3 の mean | 単発 transient false-pass に耐性。 |

`audio` が 1 frame に満たない、あるいは `frame_ms <= 0` の場合は `whole_rms` に fallback します。

### メソッド

| メソッド | 説明 |
|---------|------|
| `transcribe_sync(audio_source)` | 同期ストリーム処理（Iterator を返す） |
| `transcribe_async(audio_source)` | 非同期ストリーム処理（AsyncIterator を返す） |
| `feed_audio(audio, sample_rate)` | 音声チャンクを入力（低レベル API） |
| `get_result(timeout)` | 確定結果を取得 |
| `set_callbacks(on_result, on_interim)` | コールバックを設定 |
| `finalize()` | 残りのセグメントを文字起こし（`list[TranscriptionResult]` を返す） |
| `reset()` | 状態をリセット |
| `close()` | リソースを解放 |

### 使用例

```python
from livecap_cli import StreamTranscriber, MicrophoneSource, EngineFactory

# エンジンを準備
engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="base")
engine.load_model()

# 基本的な使い方（同期）
with StreamTranscriber(engine=engine) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.start_time:.2f}s] {result.text}")

# 非同期使用
async with MicrophoneSource() as mic:
    async for result in transcriber.transcribe_async(mic):
        print(result.text)

# コールバック方式
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

## TranscriptionResult

文字起こし結果を格納する dataclass。

### 属性

| 属性 | 型 | 説明 |
|-----|-----|------|
| `text` | `str` | 文字起こしテキスト |
| `start_time` | `float` | 開始時間（秒） |
| `end_time` | `float` | 終了時間（秒） |
| `is_final` | `bool` | 確定結果かどうか |
| `confidence` | `float` | 信頼度スコア |
| `language` | `str` | 検出された言語コード（= 翻訳元言語） |
| `source_id` | `str` | 音声ソース識別子 |
| `translated_text` | `Optional[str]` | 翻訳テキスト |
| `target_language` | `Optional[str]` | 翻訳先言語 |

### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `to_srt_entry(index)` | `str` | SRT 形式の字幕エントリに変換 |
| `duration` | `float` | 発話時間（秒）をプロパティとして取得 |

### 使用例

```python
for result in transcriber.transcribe_sync(mic):
    print(f"Text: {result.text}")
    print(f"Time: {result.start_time:.2f}s - {result.end_time:.2f}s")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Confidence: {result.confidence:.2f}")

    if result.translated_text:
        print(f"Translation: {result.translated_text}")

    # SRT 出力
    print(result.to_srt_entry(index=1))
    # 1
    # 00:00:00,000 --> 00:00:02,500
    # こんにちは
```

---

## VADConfig

VAD（音声活動検出）の設定を格納する dataclass。

### コンストラクタ

```python
VADConfig(
    threshold: float = 0.5,              # 音声検出閾値
    neg_threshold: Optional[float] = None,  # ノイズ閾値（None=自動）
    min_speech_ms: int = 250,            # 最小音声継続時間
    min_silence_ms: int = 100,           # 音声終了判定の無音時間
    speech_pad_ms: int = 100,            # 発話前後のパディング
    max_speech_ms: int = 0,              # 最大発話時間（0=無制限）
    interim_min_duration_ms: int = 2000, # 中間結果の最小時間
    interim_interval_ms: int = 1000,     # 中間結果の送信間隔
)
```

### パラメータ詳細

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `threshold` | `float` | `0.5` | 音声検出閾値（0.0-1.0） |
| `neg_threshold` | `float` | `None` | ノイズ閾値（None = threshold - 0.15） |
| `min_speech_ms` | `int` | `250` | 音声判定に必要な最小継続時間（ミリ秒） |
| `min_silence_ms` | `int` | `100` | 音声終了判定に必要な無音継続時間（ミリ秒） |
| `speech_pad_ms` | `int` | `100` | 発話前後のパディング（ミリ秒） |
| `max_speech_ms` | `int` | `0` | 最大発話時間（0 = 無制限） |
| `interim_min_duration_ms` | `int` | `2000` | 中間結果送信の最小発話時間 |
| `interim_interval_ms` | `int` | `1000` | 中間結果の送信間隔 |

### 使用例

```python
from livecap_cli import VADConfig, StreamTranscriber, EngineFactory

# デフォルト設定
config = VADConfig()

# カスタム設定（より敏感な検出）
config = VADConfig(
    threshold=0.4,        # 低い閾値でより敏感に
    min_speech_ms=200,    # 短い発話も検出
    min_silence_ms=150,   # 少し長い無音で区切る
)

# 辞書から作成
config = VADConfig.from_dict({
    'threshold': 0.6,
    'min_speech_ms': 300,
})

# StreamTranscriber に設定
engine = EngineFactory.create_engine("whispers2t", model_size="base")
engine.load_model()

transcriber = StreamTranscriber(
    engine=engine,
    vad_config=config,
)
```

---

## NoiseGate

`livecap_cli.audio.NoiseGate` — サンプル単位のエンベロープフォロワーで環境ノイズを減衰させる音量ベースのノイズゲート。VAD の前段処理として使用すると、VAD 誤検出（ハルシネーションの原因）を抑制できます。numba JIT で高速化（< 0.1 ms / 100 ms chunk）。

### コンストラクタ

```python
NoiseGate(
    threshold_db: float = -35,              # 開放閾値 (dB)
    close_threshold_db: float | None = None,  # 閉鎖閾値 (dB), None = threshold_db - 6
    attack_ms: float = 0.5,                 # アタック時間 (ms)
    release_ms: float = 100,                # リリース時間 (ms)
    sample_rate: int = 16000,               # サンプリングレート (Hz)
    noise_floor_db: float = float("-inf"),  # ゲート閉鎖時の減衰 (dB), 既定 = hard-mute
)
```

### パラメータ詳細

| パラメータ | 型 | デフォルト | 有効範囲 | 説明 |
|-----------|-----|-----------|---------|------|
| `threshold_db` | `float` | `-35` | `-80` ～ `0` | 開放閾値。envelope がこれを超えるとゲートが開く |
| `close_threshold_db` | `float \| None` | `None` (= `threshold_db - 6`) | `-80` ～ `threshold_db` | 閉鎖閾値。ゲート開放中、envelope がこれを下回ると閉じ始める。`None` は自動ヒステリシス (6 dB 下) |
| `attack_ms` | `float` | `0.5` | `0.1` ～ `100` | エンベロープ上昇時定数 |
| `release_ms` | `float` | `100` | `1` ～ `1000` | エンベロープ減衰時定数。PR C ([#283](https://github.com/Mega-Gorilla/livecap-cli/issues/283)) で既定を `30 → 100` に変更し、aggressive 閾値での fragmentation ハルシネーションを抑制 |
| `sample_rate` | `int` | `16000` | - | 音声のサンプリングレート |
| `noise_floor_db` | `float` | `float("-inf")` (hard-mute) | `-120` ～ `0` または `-inf` | ゲート閉鎖時の減衰量。`-inf` で完全無音 (出力ゼロ)、有限値で `× 10^(dB/20)` 減衰 |

### メソッド

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `process(audio_chunk)` | `np.ndarray` | `float32` 1 次元チャンクにゲートを適用 |
| `reset()` | `None` | 内部状態（envelope / gate_open / release_counter）をリセット |

### ヒステリシスと hard-mute (Issue #280 C-1 / C-2)

#### なぜヒステリシスが必要か

単一閾値 (open == close) の場合、envelope が threshold 付近で振動するとゲートが急速に開閉を繰り返します (flicker)。断片化された音声が ASR エンジンに渡り、特に whisper 系ではハルシネーションを誘発します。

**2 閾値方式** (open > close):
- 閉状態では `envelope > open_threshold` で開く
- 開状態では `envelope < close_threshold` で閉じ始める
- `close_threshold < envelope < open_threshold` の「死のゾーン」では現在の状態を維持

既定で `close_threshold_db = open_threshold - 6` を採用することで、6 dB 幅の hysteresis band を確保します。

#### なぜ hard-mute が既定か

従来 `-60 dB` の soft-mute (出力 × 0.001) は、ゲート閉鎖時にも残留信号が残ります。この残留が whisper の YouTube dataset バイアス (「ご視聴ありがとうございました」等) のトリガーになることが実測で確認されました ([PR #281 A/B 結果](https://github.com/Mega-Gorilla/livecap-cli/pull/281#issuecomment-4286562884))。

`noise_floor_db = float("-inf")` を既定とし、ゲート閉鎖時は出力を完全ゼロにします。従来の soft-mute 挙動を望む場合は `noise_floor_db=-60` を明示的に指定してください。

### 使用例

```python
from livecap_cli import StreamTranscriber, MicrophoneSource, EngineFactory
from livecap_cli.audio import NoiseGate

engine = EngineFactory.create_engine("whispers2t", device="cuda")
engine.load_model()

# 既定: 自動ヒステリシス + hard-mute (推奨)
gate = NoiseGate(threshold_db=-49)

# 単一閾値挙動を明示的に望む場合 (pre-PR-B 互換)
gate_legacy = NoiseGate(
    threshold_db=-49,
    close_threshold_db=-49,  # open == close で single-threshold
    noise_floor_db=-60,      # soft-mute
)

transcriber = StreamTranscriber(engine=engine, noise_gate=gate)
with MicrophoneSource() as mic:
    for result in transcriber.transcribe_sync(mic):
        print(result.text)
```

### 閾値決定のガイドライン

環境ノイズに近い閾値（±5 dB の「死のゾーン」）は、ヒステリシスを入れても避けるべきです。推奨値は `levels` コマンドまたは [`analyze_noise_samples()`](#noiseanalysis--analyze_noise_samples) で算出してください（`noise_peak + 10 dB` の保守的マージン）。

**攻撃的な閾値 (speech peak 付近) での tuning tips**:
- `close_threshold_db` を下げると hysteresis band が広がり、より安定 (例: open=-20, close=-30)
- `release_ms` は **既定 100 ms** で多くの状況をカバー済み ([Issue #283](https://github.com/Mega-Gorilla/livecap-cli/issues/283) で `30 → 100` に変更)。さらに緩める場合は 150-200 ms を試す

---

## NoiseAnalysis / analyze_noise_samples

`livecap_cli.audio.analysis` — 録音したノイズサンプル列から推奨閾値・危険ゾーンを算出する純関数。CLI `levels` コマンドと GUI キャリブレーション UI から共通 API として使用されます。

`NoiseGate` (`livecap_cli/audio/noise_gate.py`) は **per-sample envelope follower** で判定するため、calibration も **per-chunk peak (`|x|.max()`)** を入力にして単位を揃えます。`samples_db` (chunk RMS) は noise floor / RMS p95 の diagnostic としてのみ使用し、`suggested_threshold_db` は `peak_p95 + PEAK_SAFETY_MARGIN_DB` で求めます (issue [#291])。

### `NoiseAnalysis` dataclass

```python
PEAK_SAFETY_MARGIN_DB = 6.0  # module-level 公開

@dataclass(frozen=True)
class NoiseAnalysis:
    noise_floor_db: float          # RMS p25 (RMS-unit, diagnostic)
    noise_rms_p95_db: float        # RMS p95 (RMS-unit, diagnostic)
    peak_p95_db: float             # per-chunk |x|.max() の 95%ile (peak-unit)
    suggested_threshold_db: float  # = peak_p95_db + PEAK_SAFETY_MARGIN_DB
    danger_zone: tuple[float, float]  # floor ± 5 (RMS-unit diagnostic)
    sample_count: int
    duration_s: float
```

`danger_zone` は **RMS-unit の diagnostic**: 手動で閾値をこの RMS 範囲に設定すると floor の揺らぎで gate がフリッカーするため避けるべき領域です。`suggested_threshold_db` は peak-unit のため直接比較できません。

### `analyze_noise_samples()`

```python
def analyze_noise_samples(
    samples_db: Sequence[float] | np.ndarray,
    peak_samples_db: Sequence[float] | np.ndarray,
    sample_rate_hz: float = 10.0,
) -> NoiseAnalysis:
```

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `samples_db` | `Sequence[float]` / `np.ndarray` | chunk RMS の dB 列 (`20*log10(rms(chunk))`) |
| `peak_samples_db` | `Sequence[float]` / `np.ndarray` | chunk peak の dB 列 (`20*log10(|chunk|.max())`)。`len(peak_samples_db) == len(samples_db)` でなければならない |
| `sample_rate_hz` | `float` | chunk 取得レート (`duration_s` の計算用) |

**例外**:
- `ValueError` — `samples_db` / `peak_samples_db` が空、長さ不一致、または `sample_rate_hz <= 0`

### 使用例

```python
import numpy as np
from livecap_cli.audio import analyze_noise_samples

# マイクから 5 秒分のレベル（10 Hz で 50 chunk × RMS + peak）
rms_db_list  = [-72.3, -71.8, -70.5, ..., -58.2]   # 20*log10(rms(chunk))
peak_db_list = [-58.0, -57.5, -57.1, ..., -45.0]   # 20*log10(|chunk|.max())

analysis = analyze_noise_samples(rms_db_list, peak_db_list, sample_rate_hz=10.0)
print(f"Suggested threshold: {analysis.suggested_threshold_db:.1f} dB")
print(f"Peak p95: {analysis.peak_p95_db:.1f} dB")
print(f"Danger zone (RMS): {analysis.danger_zone}")

# NoiseGate に直接適用 (単位が揃っているため安全に渡せる)
from livecap_cli.audio import NoiseGate
gate = NoiseGate(threshold_db=analysis.suggested_threshold_db)
```

### `PEAK_SAFETY_MARGIN_DB` の根拠

`+6 dB` は NoiseGate 既定 (`attack_ms=0.5`, `release_ms=100`, `sample_rate=16000`) に対する実測ベースのマージン (livecap-gui [#331] root-cause 調査)。`attack_ms` を大幅に短くすると envelope の peak 追従が鋭くなるため margin の見直しが必要。

> **Note (issue [#291])**: 旧実装は `noise_peak (chunk RMS p95) + 10 dB` を推奨値としており、White noise の crest factor ≈ 11 dB が偶然 `+10` で吸収されていただけでした。impulsive noise (キーボード/呼吸/breath bursts) では crest factor が大きくなり threshold が peak の下に潜り、envelope follower が瞬間超え → 無音時 hallucination ("あ"/"うん"/"ピッ") を引き起こす根本原因となっていました。本 API は per-chunk peak を入力にすることで NoiseGate と単位を揃え、この root-cause を解消します。

**将来 follow-up** ([#283] と組): NoiseGate の envelope follower filter を calibration 入力に対して simulate し envelope の 95%ile を取れば margin を 1-2 dB に縮められる可能性があります。

---

## FileTranscriptionPipeline

ファイルからの一括文字起こしを行うパイプライン。

### 使用例

```python
from livecap_cli import FileTranscriptionPipeline, EngineFactory

engine = EngineFactory.create_engine("whispers2t", device="cuda", model_size="base")
engine.load_model()

pipeline = FileTranscriptionPipeline()
result = pipeline.process_file(
    file_path="audio.wav",
    segment_transcriber=lambda audio, sr: engine.transcribe(audio, sr)[0],
)

print(f"Output: {result.output_path}")
print(f"Subtitles: {len(result.subtitles)}")
```

---

## 関連ドキュメント

- [CLI リファレンス](cli.md) - コマンドライン操作
- [リアルタイム文字起こしガイド](../guides/realtime-transcription.md) - 詳細なガイド
- [サンプル README](../../examples/README.md) - 実行可能なサンプルコード
