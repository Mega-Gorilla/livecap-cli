# 新規 ASR engine 実装ガイド

> **対象**: livecap-cli に新規 ASR engine を追加する contributor 向け。
> このドキュメント **1 つで Quickstart から testing まで完結する** よう設計されている。

本 doc は [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit の成果物として作成された。6 engine の confidence_filter 統合を audit した結果、**docs と code の乖離が構造的に発生していた**こと (stale docstring / scale 誤認 / silent fail-open / 量子化 calibration audit 抜け) が判明し、これらを **anti-pattern として codify** することで再発を防止するのが本 doc の目的。

---

## 目次

1. [Quickstart 10-step checklist](#1-quickstart-10-step-checklist)
2. [Engine 契約 (BaseEngine)](#2-engine-契約-baseengine)
3. [登録 flow](#3-登録-flow)
4. [Confidence signal extraction (EngineConfidence)](#4-confidence-signal-extraction-engineconfidence)
5. [Threshold calibration](#5-threshold-calibration)
6. [Reference table (既存 7 engine)](#6-reference-table-既存-7-engine)
7. [Anti-patterns (Issue #334 audit からの教訓)](#7-anti-patterns-issue-334-audit-からの教訓)
8. [Testing 慣用 pattern](#8-testing-慣用-pattern)
9. [CHANGELOG / docs update checklist (PR description 用)](#9-changelog--docs-update-checklist-pr-description-用)

---

## 1. Quickstart 10-step checklist

新規 engine 追加は以下 10 step を順に進める。各 step の詳細は後続 section に。

| # | Step | 触る file | 詳細 section |
|---|---|---|---|
| 1 | `metadata.py` の `_ENGINES` dict に `EngineInfo` entry 追加 | `livecap_cli/engines/metadata.py` | §3.1 |
| 2 | `livecap_cli/engines/{engine_id}_engine.py` を新規作成、`BaseEngine` を subclass | `livecap_cli/engines/{engine_id}_engine.py` | §2, §3.2 |
| 3 | 必須 attribute (`engine_name`, `device`) を `__init__` で set | 同上 | §2.1 |
| 4 | Abstract method 4 個を実装 (`transcribe` / `get_engine_name` / `get_supported_languages` / `get_required_sample_rate`) | 同上 | §2.2 |
| 5 | Hook method (`get_model_metadata` / `_check_dependencies` / `_get_local_model_path` / `_load_model_from_path` / `_download_model` / `_configure_model`) を override | 同上 | §2.3 |
| 6 | (engine が信頼度 signal を expose する場合) `engine_confidence` を populate する純関数 helper を作成 | 同上 (`_extract_engine_confidence`) | §4 |
| 7 | (該当時) `confidence_filter.py:FilterConfig` に threshold を登録、calibration data を docstring に明記 | `livecap_cli/transcription/confidence_filter.py` | §5 |
| 8 | engine_smoke test を追加 (`tests/integration/engines/test_smoke_engines.py` に entry 追加、必要なら専用 test file) | `tests/integration/engines/` | §8.1 |
| 9 | `docs/contributor/adding-an-engine.md` §6 reference table に新 engine 追加 (本 doc を update、anti-pattern AP-5 で必須) | `docs/contributor/adding-an-engine.md` | §7 (AP-5) |
| 10 | `CHANGELOG.md` `[Unreleased] → ### Added` に entry 追加 | `CHANGELOG.md` | §9 |

`engine_factory.py` は **触らない** — metadata.py 経由で dynamic import するため、user 側の変更は不要。

---

## 2. Engine 契約 (BaseEngine)

`livecap_cli/engines/base_engine.py` の `BaseEngine` (line 85-) を subclass する。Template Method パターンで、`load_model()` が hook method を順に呼ぶ。

### 2.1 必須 attribute (子クラスの `__init__` で set)

| Attribute | 型 | 内容 | 例 |
|---|---|---|---|
| `engine_name` | `str` | 小文字 ID、`confidence_filter` lookup key、`metadata.py` の `id` と一致 | `"whispers2t"` / `"qwen3asr"` / `"reazonspeech"` |
| `device` | `Optional[str]` | `"cpu"` / `"cuda"` / `None` (auto detect) | — |

```python
class MyEngine(BaseEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        self.engine_name = "myengine"       # ← 必須
        self.device = detect_device(device, "MyEngine")  # 既存 helper
        super().__init__(self.device, **kwargs)
```

### 2.2 Abstract method 4 個 (必ず override)

| Method | Signature | 目的 |
|---|---|---|
| `transcribe` | `(audio: np.ndarray, sample_rate: int) -> TranscriptionResult` | 音声 → text 変換、本 engine の主要 entry point |
| `get_engine_name` | `() -> str` | **display 名** (user-facing、internal `engine_name` ID とは別)。例: `"WhisperS2T base"` / `"Qwen3-ASR 0.6B"` |
| `get_supported_languages` | `() -> list` | サポート言語の ISO 639-1 code list (例: `["ja"]`、`["en", "de", "fr"]`) |
| `get_required_sample_rate` | `() -> int` | engine が要求する sample rate (Hz)。通常 `16000` |

### 2.3 Hook method 6 個 (Template Method の流れに参加、override 推奨)

`load_model()` (`base_engine.py:139-185`) が以下を順に呼ぶ:

| Step | Method | default 動作 | override の必要性 |
|---|---|---|---|
| Step 1 (0-10%) | `_check_dependencies()` | `pass` | 依存 library 存在確認等、必要に応じ override |
| Step 2 (10-15%) | `_prepare_model_directory()` (concrete) | — | override 不要 |
| Step 3 (15-20%) | `_get_local_model_path(models_dir)` | `models_dir / f"{model_name}.bin"` | engine 固有の model path 規約があれば override |
| Step 4 (20-70%) | `_get_or_download_model()` → `_download_model()` (abstract) | `NotImplementedError` | **必ず実装** (HuggingFace / sherpa-onnx 等の download logic) |
| Step 5 (70-90%) | `_load_model_from_path(model_path)` (abstract) | `NotImplementedError` | **必ず実装** (model 読込) |
| Step 6 (90-100%) | `_configure_model()` | `pass` | decoding strategy 設定等、必要に応じ override |
| —  | `get_model_metadata()` | `{}` | model name / version / size を返す、log / progress 表示で使用 |

### 2.4 Template Method の流れ (`load_model()`)

```
load_model() の internal flow:
  ├── 0-10%   _check_dependencies()           # 依存確認
  ├── 10-15%  _prepare_model_directory()      # dir 作成
  ├── 15-20%  _get_local_model_path()         # path 決定
  ├── 20-70%  _get_or_download_model()        # cache hit or download
  │            └── _download_model()           # 子で実装
  ├── 70-90%  _load_model_from_path()         # 子で実装、model を memory に読込
  ├── 90-100% _configure_model()              # decoding 設定
  └── _initialized = True
```

---

## 3. 登録 flow

### 3.1 `metadata.py:_ENGINES` dict に `EngineInfo` entry を追加

`livecap_cli/engines/metadata.py:41-178` の `_ENGINES` dict に新 entry を追加する。

```python
"my_engine": EngineInfo(
    id="my_engine",
    display_name="My ASR Engine 1B",
    description="Short description shown in CLI / GUI",
    supported_languages=["ja"],            # ISO 639-1
    requires_download=True,
    model_size="1.2GB",                    # 表示用
    device_support=["cpu", "cuda"],
    streaming=True,                        # realtime 対応か
    module=".my_engine_engine",            # relative import path
    class_name="MyEngineEngine",
    default_params={                       # engine __init__ kwargs
        "model_name": "...",
        "use_int8": False,
    },
)
```

### 3.2 実装ファイル `livecap_cli/engines/{engine_id}_engine.py` 作成

ファイル名は `{engine_id}_engine.py` (例: `my_engine_engine.py`)。`BaseEngine` を subclass し、§2 の契約を満たす。

### 3.3 `engine_factory.py` は触らない (自動 dynamic import)

`engine_factory.py:41-53` が `metadata.py` を読んで dynamic import する。**user 側で `engine_factory.py` を編集する必要なし**。

### 3.4 既存 engine の registration 例 (whispers2t と reazonspeech の対比)

| Field | `whispers2t` | `reazonspeech` |
|---|---|---|
| `id` | `"whispers2t"` | `"reazonspeech"` |
| `display_name` | `"WhisperS2T"` | `"ReazonSpeech K2 v2"` |
| `supported_languages` | 99 languages (`list(WHISPER_LANGUAGES)`) | `["ja"]` |
| `device_support` | `["cpu", "cuda"]` | `["cpu", "cuda"]` |
| `streaming` | `True` | `True` |
| `module` | `".whispers2t_engine"` | `".reazonspeech_engine"` |
| `class_name` | `"WhisperS2TEngine"` | `"ReazonSpeechEngine"` |
| `available_model_sizes` | 9 sizes (tiny ~ distil-large-v3) | (なし、単一 model) |
| 特徴的な `default_params` | `compute_type`, `batch_size`, `use_vad` | `use_int8`, `num_threads`, `decoding_method` |

---

## 4. Confidence signal extraction (EngineConfidence)

新 engine が信頼度 signal を expose する場合、`engine_confidence` を populate することで confidence_filter の reject 判定に参加できる。

### 4.1 Signal family decision tree

```
新 engine の confidence signal は何を出している?
│
├── Whisper convention の no_speech_prob (0.0-1.0、高いほど非音声)?
│   → no_speech_prob field、Whisper family
│   → reference: livecap_cli/engines/whispers2t_engine.py:18-79
│
├── NeMo の token confidence (Hypothesis.frame_confidence)?
│   → token_confidence_mean field、NeMo token family
│   → reference: livecap_cli/engines/parakeet_engine.py:38-78
│                livecap_cli/engines/canary_engine.py:18-83
│   ⚠ scale 確認必須 (典型 NeMo 0.85+ ではなく、emission probability 系で
│      Parakeet ja ≈ 0.0504、Parakeet en ≈ 0.2452、Canary en ≈ 0.0724)
│
├── HF compute_transition_scores or sherpa-onnx ys_log_probs?
│   → avg_logprob field、AvgLogprob family
│   → reference: livecap_cli/engines/voxtral_engine.py:32-105
│                livecap_cli/engines/reazonspeech_engine.py:19-79
│                livecap_cli/engines/qwen3asr_engine.py:73-130
│   ⚠ 負の log probability、低いほど engine が出力に自信なし
│
└── engine が信号 expose せず?
    → engine_confidence は default (全 None)、filter は fail-open で pass-through
    ⚠ ただし AP-1 / AP-4 (§7) の anti-pattern に該当する設計選択になるため要注意
```

### 4.2 各 family の populate pattern (5-10 行 code 例)

**Pattern A: Whisper convention (no_speech_prob)** (`whispers2t_engine.py:18-79`)
```python
def _extract_engine_confidence(result: Any) -> EngineConfidence:
    if not isinstance(result, dict):
        return EngineConfidence()  # fail-open
    return EngineConfidence(
        no_speech_prob=_mean([float(result.get("no_speech_prob"))]),
        avg_logprob=_mean([float(result.get("avg_logprob"))]),
        compression_ratio=_mean([float(result.get("compression_ratio"))]),
    )
```

**Pattern B: NeMo token confidence** (`canary_engine.py:18-83`、`parakeet_engine.py:38-78`)
```python
def _extract_engine_confidence(hypothesis: Any) -> EngineConfidence:
    token_conf = getattr(hypothesis, "token_confidence", None)
    if token_conf is None:
        return EngineConfidence()
    if hasattr(token_conf, "tolist"):    # GPU tensor (Canary)
        token_conf = token_conf.tolist()
    return EngineConfidence(
        token_confidence_mean=sum(token_conf) / len(token_conf),
    )
```

**Pattern C: AvgLogprob (HF compute_transition_scores)** (`voxtral_engine.py:32-105`、`qwen3asr_engine.py:73-130`)
```python
def _extract_engine_confidence(transition_scores, gen_tokens, special_ids) -> EngineConfidence:
    if transition_scores is None or gen_tokens is None:
        return EngineConfidence()
    # special token を除外して mean
    masked = [
        s for s, t in zip(transition_scores, gen_tokens)
        if t not in special_ids
    ]
    if not masked:
        return EngineConfidence()
    return EngineConfidence(avg_logprob=sum(masked) / len(masked))
```

**Pattern D: AvgLogprob (sherpa-onnx ys_log_probs)** (`reazonspeech_engine.py:19-79`)
```python
def _extract_engine_confidence(result: Any) -> EngineConfidence:
    ys = getattr(result, "ys_log_probs", None)
    if ys is None:
        return EngineConfidence()
    numeric = [float(v) for v in ys if v is not None]
    if not numeric:
        return EngineConfidence()
    mean_lp = sum(numeric) / len(numeric)
    return EngineConfidence(
        avg_logprob=mean_lp,
        raw={"ys_log_probs_mean": mean_lp, "ys_log_probs_n": len(numeric)},
    )
```

### 4.3 fail-open の規約

`engine_confidence.is_available is False` (= 全 field None) の result は `confidence_filter` が **無条件 pass-through** する。これは「信号不在で silently 全 reject」を防ぐための意図的設計。signal 取得に失敗した case は **必ず default の `EngineConfidence()` を返す** こと (例外を raise しない)。

### 4.4 EngineConfidence dataclass docstring への link

`livecap_cli/engines/base_engine.py` の `EngineConfidence` dataclass docstring (line 22-60) に **各 field の scale / populate engine / filter 取扱** を documenting 済 (Issue #334 PR-1)。本 doc と齟齬が出た場合は **EngineConfidence docstring が source of truth**、本 doc を update。

---

## 5. Threshold calibration

新 engine の threshold を決定する手順。**盲目的な値設定は anti-pattern (AP-2 / AP-3)** に該当するため、必ず以下を実施。

### 5.1 clean speech corpus で speech mean 測定 (N ≥ 10)

短時間 (1-3 秒) の clean speech sample を 10 件以上用意し、各 sample で `engine.transcribe()` を実行、`result.engine_confidence` の関連 field を集計。

### 5.2 non-speech corpus で non-speech mean 測定 (N ≥ 5)

silence / applause / desk_tap / 音楽 等の non-speech sample で同様に測定。

### 5.3 noisy_speech も含めて margin verify (Issue #334 Finding 3 教訓)

`noisy_speech` (= 環境音や音響歪み込みの正常音声) を含めて margin を verify する。**clean speech だけで calibration すると noisy_speech が false reject される regression が起きる** (Issue #334 Finding 3、ReazonSpeech `avg_logprob -0.235/-0.262` での false reject 実例)。

### 5.4 量子化 (int8 / float32 等) 別に分布測定 (Issue #334 Finding 8 教訓)

量子化は logits 計算精度に影響し、`avg_logprob` 等の signal 分布を変える可能性がある。**両量子化で smoke verify** すること。同分布なら 1 threshold 共用、分布差大なら量子化別 threshold 分離 (`avg_logprob_thresholds["engine:int8"]` 等の key 拡張) を検討。

### 5.5 `confidence_filter.py:FilterConfig` に entry 追加

- **engine-specific threshold** (Voxtral 以外の avg_logprob 系):
  ```python
  avg_logprob_thresholds: Dict[str, float] = field(default_factory=lambda: {
      ...
      "my_engine": -0.3,  # PR-X.Y.Z smoke verify YYYY-MM-DD
  })
  ```
- **global threshold** (Voxtral 系で margin が広く 1 値で十分):
  ```python
  # avg_logprob_threshold (line 124) の default を変える必要は通常なし
  # entry なしの engine は -1.0 fallback、ただし engine-specific dict 推奨
  ```

### 5.6 docstring に数値 + 日付 + smoke verify reference を記載

例 (`confidence_filter.py:125-136` の reazonspeech / qwen3-asr entry 参照):
```python
"my_engine": -0.3,  # PR-X.Y.Z smoke verify YYYY-MM-DD で speech mean -0.05
                    # vs non-speech mean -0.45 → margin +0.40 → threshold -0.3 で分類
```

---

## 6. Reference table (既存 7 engine)

| engine_name (ID) | display_name | signal family | populate condition | threshold | 公式 default | 量子化 |
|---|---|---|---|---|---|---|
| `whispers2t` | WhisperS2T | Whisper (no_speech_prob + avg_logprob + compression_ratio) | dict result で各 key 存在時 (compression_ratio は base backend で None) | `no_speech_prob > 0.5` | `0.6` (公式は AND logic) | int8 / int8_float16 / float16 / float32 |
| `parakeet_ja` | NVIDIA Parakeet TDT CTC 0.6B JA | NeMo token (`token_confidence_mean`) | CTC decoder + `frame_confidence` (Path 1、`decoding_strategy="greedy_batch"`) | `< 0.005` (speech ≈ 0.0504) | なし (application-layer) | (NeMo default) |
| `parakeet` | NVIDIA Parakeet TDT 0.6B v2 | NeMo token | TDT + `preserve_alignments` + `confidence_cfg` (Path 1.5) | `< 0.005` (speech ≈ 0.2452) | なし | (NeMo default) |
| `canary` | NVIDIA Canary 1B Flash | NeMo token | AED + `preserve_token_confidence` (greedy decoding) | `< 0.005` (speech ≈ 0.0724) | なし | (NeMo default) |
| `voxtral` | MistralAI Voxtral Mini 3B | AvgLogprob | HF `compute_transition_scores(normalize_logits=True)` | `< -1.0` (global、margin +1.0) | なし | (HF default) |
| `reazonspeech` | ReazonSpeech K2 v2 | AvgLogprob | sherpa-onnx `ys_log_probs` mean | `< -0.2` (engine-specific) | なし | **int8 / float32** ⚠ Finding 8 |
| `qwen3asr` | Qwen3-ASR 0.6B | AvgLogprob | HF `compute_transition_scores`、**language-specified 時のみ** | `< -0.3` (engine-specific、両言語同一) | なし (training-time non-speech rejection) ⚠ Finding 7 hypothesis | (HF default) |

> **⚠ AP-5 (§7 参照)**: 新 engine 追加時は本 table に entry を必ず追加すること。table が stale 化すると本 doc 全体の信頼性が失われる。

詳細は `livecap_cli/transcription/confidence_filter.py:79-138` (`FilterConfig` docstring) 参照。

---

## 7. Anti-patterns (Issue #334 audit からの教訓)

本 doc で禁止する 5 つの anti-pattern。すべて [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit で発見された concrete findings から導出。

### AP-1: 「engine_confidence は常に全 None」を docstring に書く (F2 由来)

**問題**: 初期実装時に「engine_confidence は populate しない」と decision した場合、`transcribe()` docstring に「engine_confidence は常に全 None」と書きがち。しかし将来 populate を追加した時に docstring update が漏れ、**新規 consumer が「filter は無効」と誤読する** 深刻な mis-info になる。

**実例**: `livecap_cli/engines/reazonspeech_engine.py:443-451` (PR #335 で fix) — PR-A.0 era に書かれた「常に全 None / filter fail-open」記述が PR-A.5.1 (avg_logprob populate 開始) と矛盾していた。

**正解**:
- 現状の populate field を `Returns:` section に **enumerate** し、populate される **条件** も明記する
- populate 状況が変わったら **必ず同じ commit で docstring update**
- "常に全 None" 等の **時間軸に依存する表現** を避ける ("PR-X 時点では populate なし" 等の言い方も避ける)

### AP-2: `token_confidence_mean` threshold を直感で 0.5 等に変更する (F2 由来)

**問題**: `token_conf_threshold = 0.005` を見て「低すぎる、0.5 にしよう」と直感で変更すると、**全 speech が false reject** される regression が起きる。

**理由**: `token_confidence_mean` は典型 NeMo confidence (0.85+) ではなく **raw emission probability** (Parakeet ja ≈ 0.0504、Parakeet en ≈ 0.2452、Canary en ≈ 0.0724) で、scale が一桁以上小さい。

**正解**:
- `EngineConfidence` docstring (`base_engine.py:22-60`) の engine 別 scale 表 を確認
- threshold 変更時は **必ず実機 verify** (speech mean / non-speech mean / margin を測定)
- 変更内容を docstring + CHANGELOG に明記

### AP-3: 量子化 (int8 / float32) を smoke verify せず threshold 採用する (F8 由来)

**問題**: 量子化は logits 計算精度に影響し、`avg_logprob` 等の signal 分布を変える可能性がある。片方の量子化だけで calibration して採用すると、もう片方で false reject が起きる risk あり。

**実例**: ReazonSpeech の PR-A.5.1 calibration (speech `-0.11` vs non-speech `-0.45`) が **どちらの量子化で測定されたか docstring に明記なし**、Issue #334 Finding 8 で audit gap として記録。

**正解**:
- §5.4 に従い **両量子化で smoke verify**
- 同分布なら 1 threshold 共用、分布差大なら量子化別 threshold (`avg_logprob_thresholds` dict の key を `"engine:int8"` 等に拡張) を検討
- calibration data の量子化情報を docstring に明記

### AP-4: auto-detect / fail-open path を user 通知なしで残す (F6 由来)

**問題**: engine が auto-detect mode 等の特殊 path で `engine_confidence` 全 None を返す設計の場合、filter が **silently fail-open する** ことを user が気付けない。「filter on にしたのに reject 0 件」silent failure が発生。

**実例**: `livecap_cli/engines/qwen3asr_engine.py:487-490` — `language=None` で wrapper fallback path に入り、`engine_confidence` 全 None で fail-open する path が user 通知なしで残存していた (Issue #334 Finding 6、PR #336 で warn 追加で対応)。

**正解**:
- `StreamTranscriber.__init__` で **filter mode + engine + fail-open 条件** の 3 つを check し、組合せが silent fail-open に該当する場合は 1 回 `logger.warning(...)` で notify する pattern を踏襲
- **参考実装**: `livecap_cli/transcription/stream.py:_maybe_warn_qwen3_auto_detect_fail_open` (PR #336 で追加)
- warn message は **actionable** にする (「どうすれば filter を有効化できるか」を含む)

### AP-5: 本 doc の reference table (§6) を engine 追加時に update しない (本 doc 自身に対する meta-rule)

**問題**: 新 engine を追加したが本 doc の §6 reference table を update しなかった場合、将来の contributor が読む doc が **stale 状態** になる。AP-1 と同じ構造の問題が doc レベルで再発する。

**正解**:
- 新 engine 追加 PR の checklist (§9) に「§6 reference table に新 engine を追加」を含める
- PR description で本 doc の update を **必須項目** として明示
- code review で table update の有無を確認

---

## 8. Testing 慣用 pattern

### 8.1 engine_smoke marker

`tests/integration/engines/test_smoke_engines.py:25` で `pytestmark = pytest.mark.engine_smoke` (module level) を attach。CI では `engine_smoke` marker は別 job で gated 実行される (実 model load + 実音声で smoke 検証)。

```python
import pytest

pytestmark = pytest.mark.engine_smoke

KEYWORD_HINTS: dict[str, dict[str, list[str]]] = {
    "en/librispeech_1089-134686-0001": {
        "en": ["stuff", "belly"],   # expected keyword tokens
    },
    "ja/my_engine_clip_001": {
        "ja": ["こんにちは", "テスト"],
    },
}
```

新 engine 追加時は **KEYWORD_HINTS に新 entry を追加** + 必要なら専用 test file (`test_my_engine_smoke.py` 等) で個別 case を pin。

### 8.2 Mock pattern

実 model load を avoid したい test (filter / warn 経路の verification 等) では、`tests/transcription/test_stream.py:MockEngine` を再利用 or subclass する。

**例**: `tests/transcription/test_qwen3_warn.py:MockQwen3LikeEngine` (PR #336 で追加)
```python
class MockQwen3LikeEngine(MockEngine):
    """Qwen3ASREngine の identifying attribute だけ持つ minimal Mock。

    実 Qwen3ASREngine の attribute (engine_name="qwen3asr"、_asr_language=...) を
    模擬して、StreamTranscriber 層の warn 経路を実 model なしで test する。
    """
    def __init__(self, language: Optional[str] = None) -> None:
        super().__init__()
        self.engine_name = "qwen3asr"
        self._asr_language = language
```

別 pattern: `tests/integration/engines/test_reazonspeech_split.py:26-30` のように `__new__` で engine instance を作って attribute を差し込む方法 (model load 完全 skip、split 制御 path のみ test)。

### 8.3 Confidence populate test

engine が `engine_confidence` を populate する場合、その field 値を assert で pin する pure-function unit test を **実 model 不要** で書ける (extract logic を module-level pure function として export している場合)。

例: `tests/core/engines/test_reazonspeech_confidence_extraction.py` (実 sherpa-onnx 不要に schema 抽出 logic を pin)。

---

## 9. CHANGELOG / docs update checklist (PR description 用)

新 engine 追加 PR で description に含めるべき項目:

- [ ] `CHANGELOG.md` `[Unreleased] → ### Added` に entry 追加 (engine 概要 / supported_languages / device_support / model size)
- [ ] `docs/reference/api.md` の engine 一覧 (line ~130 周辺) に新 engine 行を追加
- [ ] **`docs/contributor/adding-an-engine.md` §6 reference table に新 engine 追加 (AP-5、必須)**
- [ ] (該当時) `confidence_filter.py:FilterConfig` の `avg_logprob_thresholds` dict + docstring に entry 追加 + smoke verify 数値を documenting
- [ ] (該当時) `tests/integration/engines/test_smoke_engines.py:KEYWORD_HINTS` に entry 追加
- [ ] `transcribe()` docstring の `Returns:` に **現状の populate field + 条件** を明記 (AP-1 防止)
- [ ] (該当時) auto-detect / fail-open path がある場合は `StreamTranscriber` 層で warn pattern を追加 (AP-4 防止)
- [ ] (該当時) 量子化 (int8 / float32 等) を `use_*` parameter で expose する場合は **両量子化で smoke verify** (AP-3 防止)

## 関連 / Related

- [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) — 本 doc の起点となった audit issue (6 PR + 2 Epic plan、9 findings)
- [`livecap_cli/engines/base_engine.py`](../../livecap_cli/engines/base_engine.py) — `BaseEngine` class docstring (engine 契約の literal source) + `EngineConfidence` dataclass docstring (signal field の literal source)
- [`livecap_cli/transcription/confidence_filter.py`](../../livecap_cli/transcription/confidence_filter.py) — `FilterConfig` docstring (threshold registration の literal source)
- [`livecap_cli/engines/metadata.py`](../../livecap_cli/engines/metadata.py) — `_ENGINES` dict (engine registry の literal source)
- [`docs/reference/api.md`](../reference/api.md) — user-facing API reference (engine 一覧 / Confidence filter section)
