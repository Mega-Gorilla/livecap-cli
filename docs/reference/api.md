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
    # === Confidence filter (PR-A.1 / Issue #308) ===
    filter_config: Optional[FilterConfig] = None,  # None → FilterConfig() (= mode="on" default)
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
| `filter_config` | - | `None` | Engine confidence filter 設定 (PR-A.1 [#308])。`None` 渡しで内部的に `FilterConfig()` (= `mode="on"`、default ON) を構築。`FilterConfig(mode="off")` で PR-A.0 相当の挙動 (filter 無効) に戻せる。`FilterConfig(mode="observe")` は judge を JSON log するが reject しない (PR-A.3 calibration 用)。詳細は後述「Confidence filter (PR-A.1)」参照。 |

> ⚠ **物理量の警告**: `engine_min_rms_dbfs` は **per-segment / per-frame RMS** unit。
> `NoiseGate.threshold_db` (per-sample peak envelope) とは物理量が異なるため、
> 同じ値を共有してはいけません。`levels` コマンドはそれぞれ別の suggested 値
> (`suggested_threshold_db` / `suggested_engine_min_rms_dbfs`) を出力します。

> 💡 **Calibration の推奨**: `engine_min_rms_dbfs=-45.0` (default) は whisper
> recording / 遠距離マイクを壊さないための保守値です。実音源プローブで
> hallucination が顕在化する環境では:
>
> 1. `livecap-cli levels --mic <id> --duration 5 --json` で calibration
> 2. `suggested_engine_min_rms_dbfs` フィールドを取得
> 3. `StreamTranscriber(engine_min_rms_dbfs=<calibrated>)` で渡す
>
> 実測ベースで default 26 % → calibrated 78 % の hallucination 削減差があります
> (詳細: `docs/reference/cli.md` 「EnergyGate の限界と推奨運用」)。

### Confidence filter (PR-A.1)

PR-A.0 ([#309](https://github.com/Mega-Gorilla/livecap-cli/pull/309)) で expose した `TranscriptionResult.engine_confidence` を読み、字幕に出る前に「非音声」判定 output を弾く post-ASR filter。直接 API 利用でも default で `mode="on"` が適用されます。

```python
from livecap_cli import StreamTranscriber
from livecap_cli.transcription.confidence_filter import FilterConfig

# default: filter on (production 推奨)
transcriber = StreamTranscriber(engine=engine)
#   ↑ 内部で FilterConfig() (= mode="on") を構築

# PR-A.0 相当 (filter なし) に戻す
transcriber = StreamTranscriber(
    engine=engine,
    filter_config=FilterConfig(mode="off"),
)

# observe (judge を JSON log するが reject しない; PR-A.3 calibration 用)
transcriber = StreamTranscriber(
    engine=engine,
    filter_config=FilterConfig(mode="observe"),
)

# threshold をプログラム的に override (sweep harness / 実験用)
transcriber = StreamTranscriber(
    engine=engine,
    filter_config=FilterConfig(
        mode="on",
        no_speech_threshold=0.85,       # WhisperS2T 用、default 0.71 (Phase 2)
        token_conf_threshold=0.0005,    # Parakeet_ja 用、default 0.001 (Phase 2)
    ),
)
```

| `FilterConfig` 引数 | デフォルト | 説明 |
|---|---|---|
| `mode` | `"on"` | `"off"` / `"observe"` / `"on"` のいずれか。 |
| `no_speech_threshold` | `0.71` | WhisperS2T の `no_speech_prob` がこれより上なら reject。 Phase 2 report ([#334] PR-4) §2.3 Pareto relaxed_B (clean 2.67%、 SNR≥5 全て ≤ 2%、 F1=0.901)、 Whisper 公式 0.6 近傍。 旧 default 0.5 は PR-A.0 実機 verify 値。 |
| `token_conf_threshold` | `0.001` | Parakeet (ja/en) / Canary の `token_confidence_mean` がこれより下なら reject。 Phase 2 report §2.4 Pareto strict pass (Parakeet_ja: F1=0.961、 false reject 39→11 で 72% 削減)。 Parakeet_en / Canary は Phase 2 未 calibrate だが speech mean 実測 (Parakeet_en 0.2452 / Canary 0.0724) から margin 十分。 旧 default 0.005 は PR-A.0 実機 verify 値。 |
| `avg_logprob_threshold` | `-1.0` | **global default** `avg_logprob` threshold (PR-A.4.1 [#311] Voxtral smoke verify 値、Whisper 慣習値とも一致)。`avg_logprob_thresholds` dict に entry がない engine で fallback。**strict-gated**: `no_speech_prob` と `token_confidence_mean` が両方 None の時のみ評価される (WhisperS2T 退行回避)。`None` を渡すと **global fallback のみ off**。engine-specific threshold (`avg_logprob_thresholds`) も含めて完全に off にする場合は `avg_logprob_threshold=None, avg_logprob_thresholds={}` を指定する (PR-A.5.1 [#317] codex-review Point 4 で仕様明示)。 |
| `avg_logprob_thresholds` | `{"reazonspeech": -0.40, "qwen3-asr": -0.42}` | **engine-specific** `avg_logprob` threshold dict。 Phase 2 report §2.1/§2.2 で 4 engine の Pareto gate 適用値。 ReazonSpeech (relaxed_B、 clean 2.9%、 int8/float32 完全同一、 現 default -0.20 の FRR 42.5% 実害を 5.4% に改善)、 qwen3-asr (relaxed_C、 JA/EN 両方に適用、 SNR 10 borderline は Layer 4 で再確認)。 `engine_name` で lookup (display string → `_engine_id_from_name()` で normalize)、entry なし時は `avg_logprob_threshold` (global) fallback。 |
| `compression_ratio_threshold` | `None` | 予約 field、Finding 5 で継続検討中 (別 PR)。 |

Voxtral は PR-A.4.1 から `engine_confidence.avg_logprob` を populate するため filter 対象 (上記 strict-gated)。**Canary は PR-A.4.2 から `engine_confidence.token_confidence_mean` を populate するため filter 対象** (greedy decoding 経由、Parakeet_ja と同じ `token_conf_threshold` を共用)。**Parakeet 英語は PR-A.4.3 [#316] から同 `token_confidence_mean` を populate するため filter 対象** (TDT + `preserve_alignments` 経由)。**ReazonSpeech は PR-A.5.1 [#317] から `engine_confidence.avg_logprob` を populate するため filter 対象** (sherpa-onnx `OfflineRecognitionResult.ys_log_probs` mean を Voxtral と同 semantics で、Phase 2 report ([#334] PR-4) の engine-specific threshold `-0.40` で評価)。**qwen3asr は PR-A.5.2 [Issue #318] から wrapper bypass + `output_scores=True / repetition_penalty=1.1 / no_repeat_ngram_size=3` 経由で `avg_logprob` を populate するため filter 対象** (両言語 en/ja で confirmed、Phase 2 report §2.2 で engine-specific threshold `-0.42` に更新、**7 engine 対応で PR-A 系列完成**)。CLI の `--confidence-filter` / `LIVECAP_CONFIDENCE_FILTER` env var を経由せず、直接 `filter_config` を渡せばユーザー側が完全に制御できます。詳細は [`audio-filter-reference.md`](../audio-filter-reference.md) §5。

#### Qwen3-ASR auto-detect mode の fail-open caveat (Issue [#334] Finding 6)

`Qwen3ASREngine` を `language=None` (auto-detect mode) で初期化した場合、`_transcribe_via_wrapper_fallback` path に入り **`engine_confidence` が全 None** になります。confidence filter は `engine_confidence.is_available is False` を **fail-open 規約** で pass-through するため、`filter_config.mode="on"` を指定しても **実質的に filter は無効** になります (= reject は 1 件も起きない)。

**Programmatic API 利用者向けの動作変更 (PR #336)**: 上記の組合せが発生すると `StreamTranscriber.__init__` で 1 回 `logger.warning(...)` が出ます (filter / engine の交差点で notify、Issue #334 reviewer 指摘の architectural separation に従い engine init 層ではなく stream 層で実装)。

| filter mode | engine | language | warning |
|---|---|---|---|
| `"off"` | (任意) | (任意) | ❌ なし (filter 不要) |
| `"on"` / `"observe"` | 非 qwen3asr | (任意) | ❌ なし |
| `"on"` / `"observe"` | qwen3asr | `"Japanese"` / `"English"` 等 | ❌ なし (wrapper bypass で filter active) |
| **`"on"` / `"observe"`** | **qwen3asr** | **`None`** (auto-detect) | **✅ 1 回 warn** |

**警告メッセージ例**:
```
WARNING  livecap_cli.transcription.stream:stream.py:431 Qwen3-ASR auto-detect mode
(language=None): confidence filter is effectively disabled (engine_confidence
unavailable in this path). Specify language explicitly to enable filtering
(e.g., language='Japanese'). See Issue #334 Finding 6.
```

**filter を有効化する場合**: `Qwen3ASREngine(language="Japanese")` (or `"English"` 等の Qwen3-ASR が受け入れる言語名) を明示的に指定すると wrapper bypass path で `compute_transition_scores` 経由の `avg_logprob` が populate され、threshold `-0.3` (engine-specific) で reject 判定が機能します。

```python
from livecap_cli import StreamTranscriber, EngineFactory
from livecap_cli.transcription.confidence_filter import FilterConfig

# ❌ Anti-pattern: filter on にしても reject されない (warning が 1 回出る)
engine = EngineFactory.create_engine("qwen3asr", language=None)  # auto-detect
transcriber = StreamTranscriber(
    engine=engine,
    filter_config=FilterConfig(mode="on"),  # ← 効かない
)

# ✅ Recommended: language を明示して filter を有効化
engine = EngineFactory.create_engine("qwen3asr", language="Japanese")
transcriber = StreamTranscriber(
    engine=engine,
    filter_config=FilterConfig(mode="on"),  # ← active、threshold -0.3 で reject 判定
)
```

**CLI users への影響**: CLI default は `--language ja` のため `livecap-cli transcribe ... --engine qwen3asr ...` 形式の利用者は通常通り protected。`--language auto` を明示指定した場合のみ本警告に該当します。

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
| `set_callbacks(on_result, on_interim, on_utterance_settled)` | コールバックを設定（後述の `set_callbacks` セクション参照） |
| `finalize()` | 残りのセグメントを文字起こし（`list[TranscriptionResult]` を返す） |
| `reset()` | 状態をリセット |
| `close()` | リソースを解放 |

### `set_callbacks`

```python
def set_callbacks(
    self,
    on_result: Optional[Callable[[TranscriptionResult], None]] = None,
    on_interim: Optional[Callable[[InterimResult], None]] = None,
    on_utterance_settled: Optional[Callable[[UtteranceSettledEvent], None]] = None,
) -> None
```

`**kwargs` swallow なし、未知 kwarg は `TypeError` で即時 fail（policy「不要な後方互換は廃する」、Issue [#332]）。

| 引数 | 型 | 説明 |
|---|---|---|
| `on_result` | `Optional[Callable[[TranscriptionResult], None]]` | 確定結果のコールバック |
| `on_interim` | `Optional[Callable[[InterimResult], None]]` | 中間結果のコールバック |
| `on_utterance_settled` | `Optional[Callable[[UtteranceSettledEvent], None]]` | 1 論理 utterance の処理確定 (emit/drop どちらでも) を観測する hook (Issue [#332])。Interim 字幕を出した後の silent drop で consumer 側 state が残置する architectural gap の根治に使用。 |

`UtteranceSettledEvent` (frozen dataclass、`livecap_cli` から re-export):

| 属性 | 型 | 説明 |
|---|---|---|
| `emitted` | `bool` | `True` = final result が producer 側 delivery boundary に到達確定 (consumer 受領は保証外、generator + break で未受領あり)、`False` = drop |
| `reason` | `Optional[str]` | drop 理由 (closed set、後述)。`emitted=True` 時は `None` |
| `source_id` | `str` | 1 source : N events |
| `utterance_start_time` | `float` | 発話開始時刻 (秒、stream timeline) |
| `utterance_end_time` | `float` | 発話終了時刻 (秒) |

#### `reason` 一覧

Public `Final[str]` 定数 (`from livecap_cli import REASON_FILTER_REJECT` 等で import 可)。

| 定数 | 値 | 発火条件 |
|---|---|---|
| `REASON_EMPTY_AUDIO` | `"segment:empty_audio"` | VAD final segment の audio が空 (edge case) |
| `REASON_ENERGY_GATE` | `"energy_gate:low_rms"` | per-segment EnergyGate (#292) が drop |
| `REASON_FILTER_REJECT` | `"confidence_filter:reject"` | confidence filter mode=on で reject (GUI #362 主因) |
| `REASON_ENGINE_EMPTY` | `"engine:empty_text"` | engine が空 text / whitespace のみを返却 |

**動的 reason**: `engine_error:<ExceptionType>` (例: `"engine_error:RuntimeError"`、`"engine_error:CudaOutOfMemoryError"`)。`raise EngineError(...) from e` で chain された場合は `__cause__` の型名、chain なし時は `EngineError` 自身の型名 (`"NoneType"` 出力を回避)。

#### Delivery ordering

API ごとに settled callback の発火タイミングが異なる:

| API | Settled 発火タイミング |
|---|---|
| `feed_audio()` (callback / polling path) | `_emit_result()` (queue + on_result) **完了直後**、同期実行 (consumer は result 受信 → settle 通知の順で観測) |
| `transcribe_sync()` (Iterator yield) | 内部で `feed_audio` を呼ぶため、上記同様 `on_result` 後に settled |
| `transcribe_async()` (AsyncIterator yield) | **`yield` の直前** (yield 後の code は caller が次の `__anext__()` を呼ぶまで実行されないため、caller break で永久未発火になるのを回避) |
| `finalize()` (list return) | 各 result を list append する **直前** (generator path と整合) |

drop 時は result delivery が発生せず、`on_utterance_settled(emitted=False, reason=...)` のみ発火。

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

# Utterance lifecycle observation hook (Issue #332)
# Interim を出した後 final が drop された時に consumer 側 state を clear する
from livecap_cli import UtteranceSettledEvent, REASON_FILTER_REJECT

def on_settled(event: UtteranceSettledEvent) -> None:
    if not event.emitted and event.reason == REASON_FILTER_REJECT:
        gui.clear_interim()  # consumer 側 state を即時 clear

transcriber.set_callbacks(
    on_result=on_result,
    on_interim=on_interim,
    on_utterance_settled=on_settled,
)
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

環境ノイズに近い閾値（±5 dB の「死のゾーン」）は、ヒステリシスを入れても避けるべきです。推奨値は `levels` コマンドまたは [`analyze_noise_samples()`](#noiseanalysis--analyze_noise_samples) で算出してください（`peak_p95 + peak_safety_margin_db` の保守的マージン、default `+6 dB`、issue [#291] / [#327]）。studio コンデンサーマイク (AT4040 等) では `--noise-gate-margin -5` 等の負値で更に下げられます ([#327])。

**攻撃的な閾値 (speech peak 付近) での tuning tips**:
- `close_threshold_db` を下げると hysteresis band が広がり、より安定 (例: open=-20, close=-30)
- `release_ms` は **既定 100 ms** で多くの状況をカバー済み ([Issue #283](https://github.com/Mega-Gorilla/livecap-cli/issues/283) で `30 → 100` に変更)。さらに緩める場合は 150-200 ms を試す

---

## NoiseAnalysis / analyze_noise_samples

`livecap_cli.audio.analysis` — 録音したノイズサンプル列から推奨閾値・危険ゾーンを算出する純関数。CLI `levels` コマンドと GUI キャリブレーション UI から共通 API として使用されます。

`NoiseGate` (`livecap_cli/audio/noise_gate.py`) は **per-sample envelope follower** で判定するため、calibration も **per-chunk peak (`|x|.max()`)** を入力にして単位を揃えます。`samples_db` (chunk RMS) は noise floor / RMS p95 の diagnostic としてのみ使用し、`suggested_threshold_db` は `peak_p95 + peak_safety_margin_db` で求めます (default `PEAK_SAFETY_MARGIN_DB = 6.0`、issue [#291] / [#327])。`peak_safety_margin_db` keyword 引数で user-tunable、負値も valid (高 SNR studio mic 向け、[#327])。

### `NoiseAnalysis` dataclass

```python
PEAK_SAFETY_MARGIN_DB = 6.0              # module-level 公開、peak-unit (NoiseGate)
ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0    # module-level 公開、RMS-unit (#292 EnergyGate)

@dataclass(frozen=True)
class NoiseAnalysis:
    noise_floor_db: float                # RMS p25 (RMS-unit, diagnostic)
    noise_rms_p95_db: float              # RMS p95 (RMS-unit, diagnostic)
    peak_p95_db: float                   # per-chunk |x|.max() の 95%ile (peak-unit)
    suggested_threshold_db: float        # = peak_p95_db + peak_safety_margin_db (default: PEAK_SAFETY_MARGIN_DB = 6.0)
    suggested_engine_min_rms_dbfs: float # = noise_rms_p95_db + engine_min_rms_margin_db (#292)
    danger_zone: tuple[float, float]     # floor ± 5 (RMS-unit diagnostic)
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
    *,
    engine_min_rms_margin_db: float = ENGINE_MIN_RMS_SAFETY_MARGIN_DB,
    peak_safety_margin_db: float = PEAK_SAFETY_MARGIN_DB,
) -> NoiseAnalysis:
```

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `samples_db` | `Sequence[float]` / `np.ndarray` | chunk RMS の dB 列 (`20*log10(rms(chunk))`) |
| `peak_samples_db` | `Sequence[float]` / `np.ndarray` | chunk peak の dB 列 (`20*log10(|chunk|.max())`)。`len(peak_samples_db) == len(samples_db)` でなければならない |
| `sample_rate_hz` | `float` | chunk 取得レート (`duration_s` の計算用) |
| `engine_min_rms_margin_db` | `float` (keyword-only) | `suggested_engine_min_rms_dbfs = noise_rms_p95_db + engine_min_rms_margin_db` (RMS-unit、[#292] EnergyGate)。default `6.0`、CLI `--engine-min-rms-margin` |
| `peak_safety_margin_db` | `float` (keyword-only) | `suggested_threshold_db = peak_p95_db + peak_safety_margin_db` (peak-unit、NoiseGate)。default `PEAK_SAFETY_MARGIN_DB = 6.0`、**負値も valid** (高 SNR studio mic 向け、AT4040 等)。CLI `--noise-gate-margin` ([#327]) |

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
