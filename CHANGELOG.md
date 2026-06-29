# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Added

#### Confidence threshold calibration harness — Stage 1 (Issue [#338] PR-α)

新規 `benchmarks/confidence_calibration/` sub-package を追加。observe mode で
蓄積した JSON log + user 提供 label から threshold sweep を実行する CLI
tooling (Stage 1)。Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334)
PR-2 / PR-3 / PR-4 (observe mode 1-2 月運用に依存) を ~1-2 週に短縮する path。

- **`benchmarks/confidence_calibration/_core.py`**: signal-agnostic な sweep
  logic。confusion matrix (TP/FP/TN/FN)、F1 / precision / recall / Youden's J、
  false_reject_rate を計算。direction (`reject_if_less` / `reject_if_greater`)
  と criterion (`f1` / `youden_j` / `precision` / `recall`) を arg 化、
  `LabeledSample` で input 一般化、`SweepReport` で output 標準化。
- **`benchmarks/confidence_calibration/parse_observe.py`**: Stage 1 CLI。
  ``confidence_filter[observe]: <JSON>`` の jsonl + user 提供 `labels.jsonl`
  (source_id → label mapping) を input、`_core.sweep_threshold()` 経由で
  sweep report を生成。
- **`benchmarks/confidence_calibration/pipeline.py`**: `manifest.jsonl`
  corpus loader、`LIVECAP_CALIBRATION_CORPUS_DIR` env var pattern (既存
  `LIVECAP_NON_SPEECH_CORPUS_DIR` を踏襲)、16 kHz mono float32 への
  自動 resample。
- **`benchmarks/confidence_calibration/README.md`**: Quickstart + signal
  direction / confusion matrix の解釈 / criterion 選択指針 / corpus 準備方針。

**Stage 2 (PR-β、未実装)**: user 提供 audio corpus + engine.transcribe() で
直接 active calibration、yt-dlp + Silero VAD + 原稿 fuzzy match による
corpus build helper を提供予定。

**design 判断**:
- 既存 `benchmarks/non_speech_filter/sweep.py` の argparse + grid sweep
  canonical pattern を踏襲、sibling として位置付け
- `_core` を共通化、Stage 1 / Stage 2 は input 経路のみ異なる (DRY)
- code 挙動の変更ゼロ — 既存 `FilterConfig` / engine 実装は touch しない、
  新 harness のみ追加 (Issue #334 PR-4 で本 output を活用して default を update)

**Tests**: 40 passed (`tests/benchmark_tests/confidence_calibration/`)、
unit test 中心、実 model 不要、Mock + synthetic data で confusion matrix /
edge case / log parse / corpus loader を pin。

#### Qwen3-ASR auto-detect fail-open warning (Issue [#334] Finding 6)

`StreamTranscriber.__init__` で、`filter_config.mode != "off"` かつ engine が
**Qwen3-ASR + `language=None` (auto-detect)** の組合せの時に `logger.warning(...)`
で 1 回通知する。auto-detect path (`Qwen3ASREngine._transcribe_via_wrapper_fallback`)
は ``engine_confidence`` 全 None で fail-open するため filter mode "on" でも
実質無効になる現象を、**programmatic API 利用者** が早期に気付けるようにする。

- **検出 logic**: duck typing (`engine.engine_name == "qwen3asr"` + `engine._asr_language
  is None`) で識別、`isinstance` は循環 import / Mock false negative 回避のため不採用。
- **発火条件 matrix**:
  - `filter=on` + qwen3asr + `language=None` → **warn 1 回**
  - `filter=off` + 同上 → warn なし (filter 不要のため)
  - `filter=on` + qwen3asr + `language="Japanese"` → warn なし (filter active)
  - `filter=on` + 非 qwen3asr engine → warn なし
  - `filter=observe` + qwen3asr + `language=None` → warn 1 回 (filter active)
- **実装場所** (reviewer 2nd round 指摘): `Qwen3ASREngine.__init__` は `FilterConfig`
  を受けないため、両方を知る `StreamTranscriber.__init__` で警告するのが
  architectural separation 上正しい。
- **CLI default は `--language ja`** のため CLI 利用者は通常通り保護される。`language`
  引数を明示すれば warn は出ない (actionable message)。
- **Migration**: 既存 caller は変更不要。`language=None` で auto-detect mode を
  programmatic に利用していた user は warn を 1 回受け取る (silent fail-open の
  解消、行動は変更されない)。
- **Tests** (`tests/transcription/test_qwen3_warn.py`、5 test):
  warn 発火条件 matrix を pin、`MockQwen3LikeEngine` で `Qwen3ASREngine` の
  identifying attribute を模擬。

#### Utterance lifecycle observation hook (Issue [#332])

`StreamTranscriber` の post-processing 経路には 5 種類の silent drop
(filter reject / energy_gate / engine error / engine 空 text / empty audio)
があり、interim 字幕を出した後 final が drop されると consumer 側 state
が clear されず残置する問題があった (livecap-gui#362、ReazonSpeech
`avg_logprob ≈ -0.2` 境界で正常音声断片の false reject が頻発)。本機能で
1 論理 utterance の処理確定 (emit / drop どちらでも) を観測する callback を
追加し、consumer が `emitted=False` 時に interim state を clear できるよう
にした。

- **`on_utterance_settled` callback** を `StreamTranscriber.set_callbacks`
  に追加 (3 番目の kwarg、optional)。`**kwargs` swallow なし、未知 kwarg は
  `TypeError` で即時 fail (policy「不要な後方互換は廃する」、pre-1.0
  cleanup 系列と整合)。
- **`UtteranceSettledEvent` dataclass** (`livecap_cli.transcription.utterance`、
  top-level `livecap_cli` から re-export): 5 field
  (`emitted` / `reason` / `source_id` / `utterance_start_time` /
  `utterance_end_time`)、`frozen=True`。
- **`REASON_*` 静的 reason 定数** (`Final[str]`、public re-export):
  - `REASON_EMPTY_AUDIO = "segment:empty_audio"`
  - `REASON_ENERGY_GATE = "energy_gate:low_rms"`
  - `REASON_FILTER_REJECT = "confidence_filter:reject"` ← GUI #362 主因
  - `REASON_ENGINE_EMPTY = "engine:empty_text"`
- **動的 reason**: `engine_error:<ExceptionType>` (例:
  `"engine_error:RuntimeError"`)。`raise EngineError(...) from e` で chain
  された場合は `__cause__` の型名、chain なし (`__cause__ is None`) の
  場合は `EngineError` 自身の型名 (`"NoneType"` 出力を回避)。
- **Tier 1 の 7 hook point** が settled event を発火: empty_audio /
  energy_gate / filter reject / engine_empty / engine_error /
  coalescer push emission (per output、0-2 件) / coalescer flush emission
  (periodic / force / finalize)。
- **Delivery ordering**:
  - `feed_audio` (callback path): `on_result` 完了 **後** に
    `on_utterance_settled` 発火 (同期実行、stack frame 内で順序保証)
  - `transcribe_async` (async generator): `yield` **直前** に発火
    (yield 後の code は caller が次の `__anext__()` を呼ぶまで実行されない
    ため、break で永久未発火を回避)
  - `finalize` (list return): list append **直前** に発火 (generator path
    と整合)

Consumer example:

```python
from livecap_cli import (
    StreamTranscriber, UtteranceSettledEvent, REASON_FILTER_REJECT,
)

def on_settled(event: UtteranceSettledEvent) -> None:
    if not event.emitted and event.reason == REASON_FILTER_REJECT:
        gui.clear_interim()  # consumer 側 state を即時 clear

transcriber.set_callbacks(
    on_result=on_result,
    on_interim=on_interim,
    on_utterance_settled=on_settled,
)
```

Migration: 既存 caller は不変 (default `on_utterance_settled=None`、
発火コストゼロ)。新 consumer は `set_callbacks` で opt-in。

Out of scope (別 issue で defer): interim path informational reject signal、
coalescer periodic flush で utterance なし時の event、multi-source 内部統合、
`coalescer:discarded` reason (現行実装に該当 branch なし)。

#### NoiseGate `PEAK_SAFETY_MARGIN_DB` user-tunable (Issue [#327])

`analyze_noise_samples` の `suggested_threshold_db` 計算に
`peak_safety_margin_db` keyword 引数を追加、CLI `levels` コマンドに
`--noise-gate-margin <dB>` flag を追加。`engine_min_rms_margin_db` (#292)
と並列の API 対称性を回復。

- **`analyze_noise_samples(peak_safety_margin_db=...)`** (keyword-only):
  `suggested_threshold_db = peak_p95_db + peak_safety_margin_db`。
  default = `PEAK_SAFETY_MARGIN_DB = 6.0`、既存呼び出しは bit-identical。
- **CLI `levels --noise-gate-margin <dB>`**: user が任意の margin を渡せる。
  - 高 SNR studio コンデンサーマイク (AT4040、SM7B 等、self-noise <15 dBA):
    `2` 〜 **負値** (例: `-5` で `suggested = peak_p95 - 5`、`peak_p95 ≈
    -60 dB` の AT4040 で `-65 dB` が得られる)
  - 高ノイズ環境 / 低品質 USB マイク: `10` 程度
- **CLI 出力**: 旧 hardcoded `(= peak_p95 + 6.0)` を user value 反映
  `(= peak_p95 + {margin})` に変更、user が `--noise-gate-margin -5` を
  渡した時に正確な値を表示。

scope: `transcribe` には flag 追加しない (現状 auto-calibration なし、
parse-only effectless flag は anti-pattern。auto-calibration mode は
別 issue 候補)。

Workflow (既存 `levels → --noise-gate-threshold` 手動 pass を維持):

```pwsh
# AT4040 等 studio mic
livecap-cli levels --mic 0 --duration 10 --noise-gate-margin -5 --json
# → {"suggested_threshold_db": -64.6, ...}
livecap-cli transcribe --realtime --mic 0 --noise-gate --noise-gate-threshold -64.6
```

#### Engine confidence signal schema (Issue [#308] PR-A.0 / Phase 1 Layer 3)

- **`EngineConfidence` / `TranscriptionResult` dataclasses** added to
  `livecap_cli/engines/base_engine.py`:
  - `EngineConfidence`: `Optional[float]` fields for `no_speech_prob`,
    `avg_logprob`, `compression_ratio`, `token_confidence_mean`, plus a
    `raw: dict[str, float]` overflow bucket for engine-specific signals.
    `is_available` property returns `True` when at least one signal field
    is non-`None` (PR-A.1 filter precondition).
  - `TranscriptionResult`: `text`, `confidence`, `engine_confidence`
    (default = all-None `EngineConfidence()`). `__iter__` yields
    `(text, confidence)` so the legacy `text, confidence = result` tuple
    unpacking pattern continues to work — no caller migration required
    for existing engine adapters.
  - Both dataclasses are `frozen=True` (immutable) and re-exported via
    `livecap_cli.engines.__all__`.
- **Engine adapter expose paths** (engine-by-engine breakdown):
  - `whispers2t_engine.py`: extracts `no_speech_prob`, `avg_logprob`, and
    `compression_ratio` from the CTranslate2 backend result dict via the
    new pure-function `_extract_engine_confidence()`. Handles both
    top-level signals (current backend shape) and per-segment lists
    (legacy / future shape). Real-machine smoke verify (`normal_speech_neko`
    vs `desk_tap` vs `applause_5_claps`): `no_speech_prob` = 0.036 (speech)
    vs 0.63-0.66 (non-speech) — clean separation usable by the PR-A.1
    filter. Existing `confidence: float` calculation untouched.
  - `parakeet_engine.py`: hybrid `EncDecHybridRNNTCTCBPEModel` is now
    explicitly switched to the CTC decoder via the new
    `_configure_decoding_with_confidence()` helper (see "Changed"
    section below). Real-machine `token_confidence_mean` separation:
    0.01-0.10 (speech) vs 0.0000029-0.0003 (non-speech) — 3-4 orders
    of magnitude, usable by the PR-A.1 filter with a `> 0.005` threshold.
  - `reazonspeech_engine.py`: returns `EngineConfidence()` (all `None`)
    with a docstring `Note` explaining that sherpa-onnx Python bindings
    for transducer models do not expose per-token scores. Users who
    require engine-level hallucination defense are pointed to Silero /
    TenVAD backends. Real-machine smoke verify confirmed
    `is_available is False` on all corpus clips.
  - `qwen3asr_engine.py`, `voxtral_engine.py`, `canary_engine.py`,
    `benchmarks/non_speech_filter/mock_engine.py` (`MockEngine`): no-op
    migration — return `TranscriptionResult(text=..., confidence=...)`
    with default empty `EngineConfidence`. PR-A.1 filter will treat
    these engines as fail-open (`is_available is False` → pass-through).
- **Caller migration** (defensive `hasattr`-based dispatch retained for
  legacy `Tuple[str, float]` mocks):
  - `shared_engine_manager.py` `_process_request` uses `hasattr(result,
    'text')` primary branch, keeps tuple/dict legacy branches.
  - `benchmarks/non_speech_filter/mock_engine.py` `InstrumentedEngine`
    accepts both `TranscriptionResult` and the historical tuple shape so
    benchmark harnesses keep working unmodified.
  - `livecap_cli/transcription/stream.py` `TranscriptionEngine` Protocol
    return type updated to `EngineTranscriptionResult` (runtime import
    alias from `engines.base_engine` — kept out of `TYPE_CHECKING` so
    `typing.get_type_hints()` resolves it). Stream call sites at lines
    546 / 618 / 767 use `text, confidence = ...` unpacking and continue
    to work via the dataclass `__iter__`.
- **New unit tests** (do not require ASR models — pure-function pins
  the extraction logic):
  - `tests/core/engines/test_engine_confidence_schema.py` (17 cases):
    default values, `is_available` semantics across all four signal
    fields, frozen-mutation rejection, `__iter__` yields exactly two
    items (engine_confidence excluded from tuple unpacking), public
    re-export coverage.
  - `tests/core/engines/test_whispers2t_confidence_extraction.py`
    (17 cases): mock CTranslate2 result dicts covering top-level + per-
    segment mean aggregation, missing fields, non-numeric / `None`
    values, and non-dict segment entries.
  - `tests/core/engines/test_parakeet_confidence_extraction.py`
    (12 cases): `FakeHypothesis` mock pinning the token-confidence
    primary path and edge cases (string input, empty list, non-numeric
    values, completely empty hypothesis).
  - `tests/core/engines/test_parakeet_return_hypotheses.py` (5 cases):
    pins that `return_hypotheses=True` is passed to NeMo and that the
    legacy `score / len(y_sequence)` fallback is no longer populated.
  - `tests/core/engines/test_parakeet_decoding_strategy.py` (5 cases):
    pins hybrid-model detection, CTC switch via `decoder_type='ctc'`,
    fallback to strategy-only on `TypeError`, and exception resilience.
  - `tests/transcription/test_transcription_engine_protocol.py`
    (2 cases): pins that `typing.get_type_hints()` resolves
    `EngineTranscriptionResult` to `engines.base_engine.TranscriptionResult`
    rather than the `transcription.result` dataclass of the same name.

The new signals feed into PR-A.1 (`--confidence-filter {off,observe,on}`
post-filter) and PR-A.3 (calibration + production default). Together with
PR-B calibration (PR [#304]) and the PR #307 audio-filter-reference
rewrite, this lands the Phase 1 Layer 3 schema required to close Issue
[#295].

### Documentation

#### 新規 ASR engine 実装 contributor guide 追加 (Issue [#334] PR-6)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit で
発見した「engine 追加時の docstring stale 化」「signal scale 誤認」「silent
fail-open」、および「量子化 calibration 観点の明文化」を構造的に行うため、
新規 ASR engine 追加 contributor 向けの **single source of truth doc** を
新設。本 audit の findings (F2 / F5 / F6 / F8) を anti-pattern として codify
(F8 は既存 ReazonSpeech では PR-A.5.1 で int8 / float32 両方 verify 済、本
codify は新 engine への一般則として位置付け)。

- **`docs/contributor/adding-an-engine.md` 新規**: 9 section (Quickstart 10-step
  checklist / Engine 契約 / 登録 flow / Confidence signal extraction / Threshold
  calibration / 既存 7 engine の reference table / Anti-patterns AP-1 ~ AP-5 /
  Testing 慣用 pattern / CHANGELOG・docs update checklist) を 1 doc で完結
  (~444 行)。
- **`livecap_cli/engines/base_engine.py` `BaseEngine` class docstring 拡張**:
  必須 attribute (`engine_name` / `device`) / Abstract method 4 個 / Hook method
  6 個 / Optional contract (`engine_confidence` populate) を明文化、
  `docs/contributor/adding-an-engine.md` への link。
- **`CLAUDE.md` / `AGENTS.md` cross-reference**: engine adapter section に
  「新規 engine 追加時は `docs/contributor/adding-an-engine.md` 参照」を 1 行
  ずつ追加。
- **Codified anti-patterns** (Issue #334 audit 由来):
  - **AP-1** (F2): 「engine_confidence は常に全 None」 docstring → 後で populate
    追加時に stale 化、新規 consumer が誤読
  - **AP-2** (F2): `token_confidence_mean` threshold を直感で 0.5 等に変更
    → engine 別 scale (Parakeet ja 0.0504 / en 0.2452 / Canary 0.0724) を
    知らないと全 speech false reject regression
  - **AP-3** (F8、一般則): 新 engine 追加時に量子化 (int8 / float32) を smoke
    verify せず threshold 採用 → 量子化で signal 分布が変わる可能性。
    既存 ReazonSpeech は PR-A.5.1 で両量子化 verified (margin +0.13 / +0.10)、
    本 codify は新 engine への一般則。
  - **AP-4** (F6): auto-detect / fail-open path を user 通知なしで残す
    → 「filter on にしたのに reject 0 件」silent failure。
    `StreamTranscriber._maybe_warn_qwen3_auto_detect_fail_open` (PR #336) が
    参考実装
  - **AP-5** (本 doc 自身に対する meta-rule): 新 engine 追加時に本 doc の
    reference table を update しない → doc が stale 化

#### Engine confidence signal semantics clarified (Issue [#334] Findings 1 / 2 / 5)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit
で発見した既存 docstring と実装の乖離 + signal semantics の誤認 risk を
docstring/comment レベルで解消。code 挙動の変更なし、low-risk な
documentation cleanup。

- **`EngineConfidence` の各 field 説明を `Attributes:` section に拡充**
  (`livecap_cli/engines/base_engine.py:22-44`):
  - 各 field の **scale / populate engine / filter 取扱**を明記
  - `token_confidence_mean` の **低 scale (Parakeet ja ≈ 0.0504、
    Parakeet en ≈ 0.2452、Canary en ≈ 0.0724、典型 NeMo confidence 0.85+
    ではない)** を明示 (Issue #334 Finding 2)
  - 「ReazonSpeech / qwen3asr は未対応で全 None」という冒頭の stale 記述を
    削除 (PR-A.5.1 / PR-A.5.2 で対応済)
- **`ReazonSpeechEngine.transcribe()` docstring を PR-A.5.1 反映**
  (`livecap_cli/engines/reazonspeech_engine.py:443-454`):
  - 以前は「`engine_confidence` は **常に全 None**、filter fail-open」と
    読めたが、現在は `avg_logprob` populate 済 (sherpa-onnx 1.12.39+ の
    `ys_log_probs` mean、engine-specific threshold `-0.2`)
- **`FilterConfig.no_speech_threshold` の公式 Whisper 0.6 との差を明記**
  (`livecap_cli/transcription/confidence_filter.py:86-101`):
  - livecap-cli は ``0.5`` (公式より ``0.1`` strict)、PR-A.0 data-calibrated
  - Speech margin / non-speech margin の数値も明記 (Issue #334 Finding 1)
- **`FilterConfig.token_conf_threshold` の docstring に engine 別 scale 追加**
  (`livecap_cli/transcription/confidence_filter.py:102-120`):
  - 「threshold を高い値に変更すると全 speech が false reject される深刻
    regression」を明示 (Issue #334 Finding 2)
- **`FilterConfig.compression_ratio_threshold` の「未使用予約 field」を実態
  に書き換え** (`livecap_cli/transcription/confidence_filter.py:121-128`):
  - extract logic は実装済だが、**現 CTranslate2 backend (WhisperS2T base)
    では `compression_ratio` は常に `None`** (`whispers2t_engine.py:31-33`
    smoke verify 済)
  - forward-compatibility 用、enable には populate verify + calibration の
    2 段階が必要 (Issue #334 Finding 5)

### Removed

#### `SharedEngineManager` orphan module 削除 (Issue [#326])

[Issue #321 PR #3](https://github.com/Mega-Gorilla/livecap-cli/pull/325) の
API contract cleanup 中に発見した orphan code (`livecap_cli/engines/shared_engine_manager.py`、
**467 行**) を完全削除。pre-1.0 cleanup。

**削除対象** (3 symbols すべて zero caller、`__all__` 非 export、production / tests
からの参照ゼロを grep で確認):

- `ProgressCallback` Protocol
- `TranscriptionRequest` dataclass (`__lt__` 比較含む)
- `SharedEngineManager` class (threading + queue + 進捗 callback)

**Migration**: production / tests から未参照のため影響なし。仮に第三者
plugin が import していた場合は git history (`git log -- livecap_cli/engines/shared_engine_manager.py`)
から復元可能。

**reviewer feedback で追加 scope** (本 PR で実施):

- `livecap_cli/transcription/stream.py` の `TranscriptionEngine` Protocol
  docstring 2 箇所 (line 118 / 153) から `SharedEngineManager._process_request`
  の挙動説明を削除、`apply_filter` 単一 consumer 記述に整理
- `AGENTS.md:5` の repo guidance を更新、共有 tooling 説明を
  `shared_engine_manager.py` → `model_memory_cache.py` / `library_preloader.py` /
  `nemo_utils.py` (actually active な shared utility) に置換

**Verification** (本 PR merge 後):

```pwsh
git grep -n "SharedEngineManager\|shared_engine_manager" -- `
  livecap_cli tests AGENTS.md docs/reference docs/guides
# → 0 件 (CHANGELOG.md と docs/planning/archive/* の歴史的言及は許容)

uv run python -c "from livecap_cli.engines import EngineFactory, BaseEngine; print('OK')"
# → OK
```

### Changed

#### Engine API contract — fallback adapter cleanup (Issue [#321] PR #3、3-PR 系列完成)

[Issue #321](https://github.com/Mega-Gorilla/livecap-cli/issues/321) の
**3-PR 系列 (PR #1 wording + Canary `beam_size` / PR #2 NeMo fallback chain /
PR #3 本) を完成**させる最終 PR。PR #320 (qwen3asr) / PR #322 / PR #323 で
確立した「framework contract を trust、silent degradation より hard fail」
方針を `TranscriptionEngine` Protocol contract に最終適用。

##### Engine I/O 契約の明文化

`TranscriptionEngine` Protocol (`livecap_cli/transcription/stream.py`) の
docstring を厳格化:

- 実装者は `transcribe()` から **必ず `TranscriptionResult` を返すこと**
- tuple / dict / str / None は契約違反、`apply_filter` (StreamTranscriber
  経路) 側で `AttributeError` が caller に propagate して fail-fast
- 別 path の `SharedEngineManager._process_request` も bare attribute
  access に整理したが、module-level の `except Exception` で contract
  violation も "request failure" として log + `None` 返却 (orphan code、
  Issue [#326] で本 file 削除予定のため fail-fast 化は scope 外)
- pre-1.0 cleanup の方針を明示、precedent (PR #320/#322/#323) を docstring
  で reference

##### `apply_filter()` — `hasattr` legacy guard 削除

`confidence_filter.py:386-390` の `hasattr(result, "engine_confidence")`
guard を削除、bare attribute access に統一:

- **Before**: `if not hasattr(result, "engine_confidence"): return result`
  (旧 mock の tuple 返却互換)
- **After**: bare `result.engine_confidence` access、契約違反時は
  `AttributeError` propagate
- **Audit verify**: 全 test MockEngines (6 件) + 全 `apply_filter()` test
  caller が既に `TranscriptionResult` 返却済を grep + read で確認、guard は
  dead code

##### `SharedEngineManager._process_request` — tuple/dict adapter 削除

`shared_engine_manager.py:437-490` の `hasattr` tuple branch + `isinstance(result, dict)` branch を削除、direct attribute access only に rewrite:

- **Before**: `hasattr(result, 'text')` 主、tuple `(text, conf)` fallback、
  dict `{"text": ..., "confidence": ...}` fallback の 3 path
- **After**: `result.text` / `result.confidence` の bare access。
  ただし method-level の `except Exception as e` は維持されているため、
  契約違反 (`AttributeError`) は **caller に propagate せず** "request
  failure" として log + `None` 返却。`apply_filter` 側 (fail-fast) と
  挙動が異なる点に注意
- **Caveat**: `SharedEngineManager` 自体は production / tests から完全に
  未参照の **orphan code** (`__all__` にも非 export)。本 PR では契約
  整合のみ実施、`except Exception` を狭めて fail-fast 化することは
  scope 外。**ファイル削除自体は [Issue #326]** で対応予定

##### Stale docstring 整理

- `tests/transcription/test_stream.py::FilteringMockEngine` docstring の
  「`MockEngine` は legacy tuple を返す」記述を削除 (実態は `TranscriptionResult`
  返却、stale comment)
- `CLAUDE.md:78` の TranscriptionEngine Protocol 例を `Tuple[str, float]` →
  `TranscriptionResult` に修正 (AI agent guidance と code 契約の乖離を解消)

##### Audit findings (本 PR scope を絞った根拠)

| Audit item | 結果 |
|---|---|
| 全 test MockEngines (6 件) | ✅ 既に `TranscriptionResult` 返却済 |
| `apply_filter()` 全 test caller | ✅ 既に `TranscriptionResult` 渡し済 |
| `SharedEngineManager` の production/test caller | ✅ **0 件** (orphan code) |
| `CLAUDE.md:78` 旧 `Tuple[str, float]` 型 | ⚠ stale、本 PR で修正 |
| `FilteringMockEngine` docstring | ⚠ stale、本 PR で修正 |

→ test fixture 統一 phase は不要、PR scope は contract tightening +
stale comment 整理に絞れた。

##### Migration

- **既存 production engine (WhisperS2T/Parakeet/Voxtral/Canary/ReazonSpeech/
  qwen3asr) は影響なし**: 既に `TranscriptionResult` を返却済
- **既存 test mocks も影響なし**: 既に `TranscriptionResult` 返却 (audit verify)
- **第三者 plugin / custom engine 実装者** (もしいれば): `transcribe()` の
  戻り値を `TranscriptionResult` に統一する必要あり。tuple/dict/str/None
  返却は `AttributeError` で fail-fast

##### Tests (退行ゼロ、712 baseline 維持)

- `tests/transcription/test_confidence_filter.py`: 全 pass (`apply_filter`
  caller 全て `TranscriptionResult`)
- `tests/transcription/test_stream.py`: 全 pass (MockEngine / FilteringMockEngine
  共に `TranscriptionResult`)
- Full local regression: 712 passed

##### Out of scope (本 PR では行わない)

- `livecap_cli/engines/shared_engine_manager.py` orphan file 自体の削除 →
  別 issue "audit unused engine infrastructure" (`SharedEngineManager` +
  `TranscriptionRequest` + `ProgressCallback` の orphan 確認 + 削除提案)
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 → 別 issue
- 他 engine の `__init__` `**kwargs` 削除 → 別 issue
- `docs/planning/archive/*.md` の旧型 reference → archive 性質上 触らない

##### Issue #321 完成宣言

| PR | scope | 状況 |
|---|---|---|
| **PR #1 ([#322])** | wording cleanup + Canary `beam_size` fail-fast | ✅ merged |
| **PR #2 ([#323])** | Canary/Parakeet NeMo fallback chain | ✅ merged |
| **PR #3 (本)** | API contract cleanup | ✅ |

3-PR 系列完成、本 PR merge 後に Issue #321 を close。

#### Engine confidence — Canary / Parakeet NeMo fallback chain cleanup (Issue [#321] PR #2)

`CanaryEngine._configure_decoding_with_confidence` と
`ParakeetEngine._configure_decoding_with_confidence` から、`token_confidence`
取得 path (Path 1) が失敗した場合に **token_confidence なしで継続する silent
degradation を生む fallback chain** を削除して fail-fast 化。加えて
`ParakeetEngine._transcribe` の `return_hypotheses=True` TypeError silent
fallback も削除。

PR #320 (qwen3asr) / PR #322 (Canary `**kwargs`) の precedent と整合、
「framework contract を trust、silent degradation より hard fail」方針を
NeMo 系 engine にも適用。

##### Before / After

- **Before**:
  - Canary: Path 1 (greedy + confidence_cfg) → Path 2 (greedy のみ、confidence
    なし) → Path 3 (argument-less)。Path 1 失敗時に silent fallback、
    `token_confidence_mean` が None になり confidence filter は pass-through に degrade
  - Parakeet: Path 1 (Hybrid CTC) → Path 1.5 (Pure RNNT/TDT) → Path 2
    (strategy-only) → Path 3 (argument-less)。同様に Path 1/1.5 失敗時に
    silent fallback
  - `ParakeetEngine._transcribe`: `transcribe(return_hypotheses=True)` が
    TypeError なら kwarg なしで再 transcribe、`engine_confidence` 全 None
    で filter が pass-through に degrade
- **After**:
  - Canary: Path 1 のみ、bare 呼出 (try/except 削除)。失敗時は NeMo native
    `TypeError` / `ValueError` 等が propagate
  - Parakeet: Path 1 (Hybrid model-family dispatch、temporary fallback で
    Path 1.5 へ) + Path 1.5 (Pure RNNT/TDT primary、bare 呼出)。Path 1.5
    失敗時は NeMo error が propagate
  - `ParakeetEngine._transcribe`: bare 呼出、`return_hypotheses=True` 失敗時
    は TypeError が propagate (`nemo-toolkit>=2.3,<2.5` の supported range で
    公式安定 API)

##### Migration

- `nemo-toolkit>=2.3,<2.5` (lockfile `2.3.0`) の supported range では既存
  挙動と完全に同じ (Path 1 / Path 1.5 / `return_hypotheses=True` は常に成功)
- supported range 外の旧 nemo build を使う user は、Path 1 が拒否された時点で
  従来の silent fallback ではなく `TypeError`/`KeyError`/`ValueError` 等が
  直接 raise されるため、具体的な NeMo error message から nemo version を
  確認する actionable hint を得る
- Parakeet **Path 1 (Hybrid CTC)** と **Path 1.5 (Pure RNNT/TDT)** は
  **model-family dispatch** (hybrid vs pure decoder の正規 dispatch) として
  温存。legacy fallback ではないため reviewer 承認の上で温存

##### Verification (merge gate)

`tests/integration/engines/test_smoke_engines.py::test_token_confidence_populated`
で実機 GPU verify (RTX 4090 self-hosted runner):

| Case | Expected token_confidence_mean (probe baseline) |
|---|---|
| `canary_gpu_en` (LibriSpeech 英語) | > 0.05 (PR-A.4.2 で 0.0724) |
| `parakeet_gpu_en` (LibriSpeech 英語) | > 0.10 (PR-A.4.3 で 0.2452) |
| `parakeet_ja_gpu_ja` (jsut 日本語) | > 0.02 (PR-A.0 で 0.0504) |

新 test は `@pytest.mark.engine_smoke` で hosted CI から除外、self-hosted
GPU runner でのみ実行。失敗時は merge を blocking する design。

##### Out of scope (本 PR では行わない)

- `confidence_filter.py:386` `hasattr(result, "engine_confidence")` guard
  削除 — **Issue #321 PR #3** で `shared_engine_manager.py` tuple fallback
  + `TranscriptionEngine` Protocol cleanup とセットで扱う
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 — 別 issue
- nemo-toolkit version の pin 化 / 上限拡大 — 別 issue (本 PR では現状の
  `>=2.3,<2.5` range を contract として扱う)

#### Engine confidence filter — qwen3asr support via wrapper bypass (Issue [#318] PR-A.5.2)

PR-A 系列の **7 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
最終形に到達させる PR。Phase 1 probe (Issue #318 で User 意向「EN/JP 両言語
対応出来なければ close」に対する go condition 達成) を受けて実装、両言語
verified で qwen3asr を追加対応。

##### Engine integration — wrapper bypass で avg_logprob 抽出

qwen3asr は qwen-asr wrapper が ``output_scores=True`` を渡さない設計の
ため confidence filter 非対応だったが、**`Qwen3ASRModel.model` (= 内部
``Qwen3ASRForConditionalGeneration``) を直接呼ぶ wrapper bypass** で対応:

- ``transcribe()`` を rewrite、``self.model.model.generate(output_scores=True,
  repetition_penalty=1.1, no_repeat_ngram_size=3)`` を直接呼出
- ``compute_transition_scores(normalize_logits=True)`` 経由で per-token
  logprob を取得、Voxtral PR-A.4.1 と完全同形の ``_extract_engine_confidence``
  helper で mean → ``EngineConfidence.avg_logprob`` populate
- ``_asr_language is None`` (auto-detect mode) は旧 wrapper.transcribe()
  path に fail-open (engine_confidence 全 None、filter pass-through)

##### `repetition_penalty=1.1 + no_repeat_ngram_size=3` で両言語 failure mode 解消

Phase 1 probe で確認した critical finding:

- **Japanese**: desk_tap の 256-token repetition loop ("うんうんうん...") を
  4-token "うん。" に短縮、avg_logprob -0.13 → -0.65、**margin -0.02 (逆転)
  → +0.27 (filter 可能)**
- **English**: applause の system prompt leak ("You are a speech
  recognition model.") を avg_logprob -0.04 → -1.08 に低下 + "You are an AI."
  に短縮、**margin -0.03 (逆転) → +0.21 (filter 可能)**

→ **両言語で同じ generation parameter** で対応可能、言語別実装不要。

##### Section 1 smoke (両言語 6 clip)、Phase 1 probe 値を上回る margin 確認

- **English**: speech -0.05、non-speech -0.71、**margin +0.65** (Phase 1
  probe +0.21 を大幅に上回る) → Case A
- **Japanese**: speech -0.12、non-speech -0.63、**margin +0.42** (Phase 1
  probe +0.27 を上回る) → Case A
- 両言語で threshold ``-0.3`` で 100% 分類成功

##### Section 2 (12 cell stream pipeline benchmark)

- **Hall.(pre) = 0% 全 cell** — qwen3asr は `repetition_penalty` 適用後
  本 corpus で hallucinate しない (Canary PR-A.4.2 と同 engine 固有
  fail-safe pattern)
- SR(post) = 100% real corpus 全 cell (legit speech は 1 件も drop されず)
- Latency 影響なし

##### Engine-specific threshold = `-0.3` (両言語 safe)

- `FilterConfig.avg_logprob_thresholds["qwen3-asr"] = -0.3` を default load
- dict key を ``"qwen3-asr"`` (ハイフン含む) にしているのは、
  ``_engine_id_from_name("Qwen3-ASR 0.6B")`` が ``"qwen3-asr"`` に normalize
  するため (PR-A.5.1 codex Point 1 の learning を pre-empt)
- Phase 4 unit test で **production display string** での threshold lookup
  を pin (6 件 + helper mapping 拡張で qwen3-asr 追加)

##### Migration

| Engine | Before | After |
|---|---|---|
| qwen3asr (language 指定あり) | fail-open (engine_confidence 全 None) | ``avg_logprob < -0.3`` で reject、`repetition_penalty=1.1 + no_repeat_ngram_size=3` で failure mode 解消 |
| qwen3asr (auto-detect mode) | (不変) | (不変、wrapper fallback で fail-open) |
| WhisperS2T / Parakeet (ja/en) / Voxtral / Canary / ReazonSpeech | (不変) | 退行ゼロ |

##### Caveats (production user 向け)

1. **WER 軽微退行 (LLM typical 0.5-1%)**: ``repetition_penalty=1.1 +
   no_repeat_ngram_size=3`` で稀に正常 token も抑制可能性。Voxtral
   PR-A.4.1 / Canary PR-A.4.2 と同 framing で filter benefit を優先。
   ``--confidence-filter off`` は **post-ASR reject のみ**無効化し、
   generation 側変更 (``repetition_penalty=1.1`` / ``no_repeat_ngram_size=3``)
   は固定で残る (Voxtral greedy / Canary greedy と同 design)
2. **多言語 verify (28+ 言語) は本 PR scope 外**: en/ja のみ verified、
   他言語は user feedback ベース (Voxtral / Canary と同 framing)
3. **`_asr_language is None`** (auto-detect mode) は fail-open、production
   user は ``--language en/ja/...`` を明示推奨
4. **wrapper internal attribute 依存**: ``self.model.model`` (=
   ``Qwen3ASRForConditionalGeneration``) の private structure に依存。
   qwen-asr が future update でこの構造を変更すると AttributeError で hard
   fail する (Voxtral PR-A.4.1 と同じく framework contract を trust する design)

##### Tests (新規 +20 件、合計 703 passed)

- `tests/core/engines/test_qwen3asr_confidence_extraction.py` (新規 14 件) —
  Voxtral pattern 流用、masking ロジック + tensor/numpy 互換 + Phase 1
  probe 値再現を pin
- `tests/transcription/test_confidence_filter.py` (+6 件) —
  `TestQwen3AsrEngineSpecificThreshold` + `TestEngineIdNormalization` に
  `"Qwen3-ASR 0.6B"` / `"Qwen3-ASR 1.7B"` display string pin
- `tests/transcription/test_stream.py` (+1 件) — banner test に
  ``qwen3-asr`` + ``-0.3`` assertion

##### Docs

- ``docs/research/qwen3asr-confidence-smoke-2026-06-12.md`` (新規 decision doc)
- ``docs/audio-filter-reference.md``: Engine support table を **7 engine
  対応**に拡大、Property table / Decision section / 完成サマリ / Comparison
  table の全 section で 6 → 7 engine 整合
- ``docs/reference/cli.md`` + ``docs/reference/api.md``: qwen3asr 行追加 +
  `avg_logprob_thresholds` 例に qwen3-asr 追記
- ``base_engine.py`` ``EngineConfidence`` docstring の populate status table
  に qwen3asr 追加

##### Out of scope (本 PR では行わない)

- 多言語 verify (en/ja 以外の 28+ 言語) — user feedback ベース
- ``Qwen3ASRModel.transcribe()`` の auto-detect mode (``force_language=None``)
  — scope 外、必要なら follow-up
- 長尺音声 (>50s) の auto-chunking — stream pipeline では不要 (VAD segments
  典型 <30s)
- CLI flag ``--qwen3asr-repetition-penalty`` 等の generation param 制御 —
  不要 (hardcoded、Voxtral / Canary と整合)

#### Engine confidence filter — ReazonSpeech support + engine-specific threshold (Issue [#317] PR-A.5.1)

PR-A 系列の **6 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
完成形に到達させる PR。reviewer feedback (Issue #317) で 7 件の critical
指摘を受領、本 PR で全反映。

##### Production bug fix (reviewer Point 6、CRITICAL)

`reazonspeech_engine.py:430` の ``text, confidence = self._transcribe_single(...)``
unpack が PR #314 で削除済 ``TranscriptionResult.__iter__`` で TypeError を
投げるが、外側の ``except Exception`` で silent swallow + ``continue`` して
いたため、**長尺音声 (>30s、``auto_split_duration`` 経路) で全 segment が
silently dropped されていた production critical bug** を Phase 1 で独立
commit で修正。3 件 regression test (mock-based) で pin。

旧挙動: 30s 超え audio → 空 transcription (production reach 中)
新挙動: 各 segment の text が正しく combined_text に積まれる

これは breaking change ではなく **production bug 修正**。

##### ReazonSpeech confidence filter integration (Issue [#317] core)

旧 docs ([Issue #308 close 時点]) では「sherpa-onnx Python bindings に
per-token score API なし、PR #2897 closed/not-merged、Python 未対応」を
理由に **PR-A.5 candidate (heavy refactor)** としていたが、本 PR plan
段階の実機 probe で **sherpa-onnx 1.12.39 で
``OfflineRecognitionResult.ys_log_probs`` が既に exposed されている**こと
が判明、standard integration work で対応:

- **Before**: ReazonSpeech の ``transcribe()`` は ``engine_confidence =
  EngineConfidence()`` (全 None) で fail-open
- **After**:
  - ``reazonspeech_engine.py`` に module-level ``_extract_engine_confidence(result)``
    helper 追加 (Canary / Voxtral pattern 流用、実 sherpa-onnx 不要に unit
    test で schema pin)
  - ``_transcribe_single()`` で sherpa-onnx ``result.ys_log_probs`` を抽出、
    mean を **``EngineConfidence.avg_logprob`` field** に populate
    (Voxtral と同 semantics、reviewer Point 1/2 で確定設計)
  - ``raw["ys_log_probs_mean"]`` + ``raw["ys_log_probs_n"]`` に metadata 保存
  - ``_transcribe_with_split()`` で segment 別 engine_confidence を
    **weighted mean** で aggregate (token 数 weight)

reviewer Point 1 (CRITICAL): ``token_confidence_mean`` field 再利用は
**probability (0-1 range) vs log probability (負の値) semantics 不整合**で
全 reject になる critical bug。``avg_logprob`` field 使用が正解。

##### Engine-specific threshold (reviewer Point 3、HIGH)

ReazonSpeech (margin +0.10-0.13) と Voxtral (margin +1.0) は同 ``avg_logprob``
field を共用するが分布が桁違い。global ``avg_logprob_threshold = -1.0`` は
ReazonSpeech に機能しないため、**``FilterConfig.avg_logprob_thresholds:
Dict[str, float]``** を追加:

- ReazonSpeech default ``-0.2``
- Voxtral は dict に load しない → ``avg_logprob_threshold`` (global) fallback
  で ``-1.0`` 維持 (**backward compat ゼロ regression**)
- ``should_reject()`` の signature を ``(result, config, engine_name=None)``
  に refactor、``apply_filter()`` から engine_name pass-through
- engine-specific lookup → global fallback の 2 段 fallback で scalable

##### Findings (詳細は ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md``)

- **Phase 1 bug fix** ✅ — 3 件 regression test で long-audio silent drop bug pin
- **Section 1 (engine smoke、5 clip × int8/float32)** ✅ —
  - int8: speech mean -0.14、non_speech mean -0.30、margin +0.13、Case A
  - float32: speech mean -0.16、non_speech mean -0.45、margin +0.10、Case A
  - 両 model で threshold -0.2 が 100% 分類成功 (reviewer Point: int8
    availability も確認済)
- **Section 2 (12 cell stream pipeline benchmark)** ✅ —
  - **``webrtc × reazonspeech × real × on``: Hall.(post) 50% → 0%**
    (Issue #295 元 motivation の最後の cell 完了)
  - ``webrtc × synthetic × on``: 62.5% → **0%** (完全解消、codex-review
    #319 1st round の engine_name normalize fix 後の 2nd run で確認、初版
    は 25% 残)
  - silero / tenvad × real: 0% → 0% (VAD で除去済、filter は冗長安全網)
  - Latency 影響ゼロ
- **Section 3 (language coverage)** — ReazonSpeech は日本語 native only、
  Canary PR-A.4.2 と同 framing

##### Migration

- WhisperS2T / Parakeet (ja/en) / Voxtral / Canary 退行ゼロ
- ReazonSpeech user は ``--confidence-filter on`` (default) で hallucination
  が自動 drop されるようになる
- 長尺音声 (>30s) user は production bug 修正で正しい transcription を
  受け取れるようになる

##### Out of scope (qwen3asr は Issue [#318] で research-phase)

reviewer Point 5 (HIGH): qwen3asr の avg_logprob 単独 filter は危険
(confidence filter ≠ hallucination content guard、英語 mode で system
prompt leak、日本語 mode で repetition loop)。Issue [#318] で probe +
hallucination guard 設計を別 PR で扱う (本 PR scope 外)。

##### docs update (6 engine 対応に整合)

- ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md`` (新規)
- ``docs/audio-filter-reference.md`` Engine support table を 6 engine 対応、
  Property table / Decision section / 完成サマリ / Comparison table 全
  section で 5 → 6 engine 整合
- ``docs/reference/cli.md`` / ``docs/reference/api.md``: ReazonSpeech 追加、
  ``avg_logprob_thresholds`` field 明記
- ``base_engine.py`` EngineConfidence docstring の populate status table に
  ReazonSpeech 追加
- ``confidence_filter.py`` module docstring を engine-specific threshold
  framing に整合

#### Engine confidence filter — Parakeet 英語 support (Issue [#311] PR-A.4.3)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) / PR-A.4.2 ([#315]) で whispers2t /
parakeet_ja / Voxtral / Canary に対応した confidence filter を **Parakeet
英語** (`nvidia/parakeet-tdt-0.6b-v2`) にも拡張。本 PR で **Parakeet 英語が
構造的限界ではなく PR #309 時点の設定漏れだった**ことが判明、新規 **Path
1.5** で対応:

- **Before**: Parakeet 英語の `transcribe()` は `engine_confidence =
  EngineConfidence()` (全 None) で fail-open。decoding は strategy-only
  (旧 Path 2)。
- **After**:
  - `parakeet_engine.py::_configure_decoding_with_confidence` に **Path 1.5**
    追加 (Path 1 Hybrid CTC と Path 2 strategy-only の間に挿入):
    - pure RNNT/TDT model 用、`preserve_alignments=True` + `confidence_cfg`
      + `greedy.preserve_frame_confidence=True`
    - NeMo の制約「preserve_frame_confidence は preserve_alignments と同時
      設定必須」(`rnnt_decoding.py:280-282`) を満たす形で構成
    - Path 1.5 が rejected された場合は Path 2 (strategy-only) に fail-open
      fallback
  - `_extract_engine_confidence()` helper は Parakeet_ja と同じものを共用
    (Canary PR-A.4.2 で Tensor / List / numpy 全部扱えるよう拡張済)
  - `_log_filter_banner()` の表現を `parakeet_ja / canary` → **`parakeet (ja/en) / canary`**
    に整合
- **Migration**:
  - WhisperS2T / Parakeet_ja / Voxtral / Canary 退行ゼロ
  - Parakeet 英語 user は `--confidence-filter on` (default) で英語 audio の
    hallucination が自動 drop される
  - **非英語入力時の false reject リスク**: Parakeet 英語は English-only model、
    非英語音声には低 confidence で false reject の可能性。`--confidence-filter off`
    で opt-out 可能
- **Findings (詳細は `docs/research/parakeet-english-confidence-smoke-2026-06-11.md`)**:
  - **Phase 1 probe** ✅ — `hypothesis.token_confidence` は **List[float]** で
    populate。LibriSpeech 英語 → token_confidence_mean = **0.2452**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.2452 vs threshold 0.005
    で **49× margin** (Case A、3 engine 中で最大)。非音声は engine 自体が
    empty text を返す fail-safe
  - **Section 2 (stream pipeline、12 cell)** ✅ — `webrtc × synthetic × on`
    で **Hall.(post) 75% → 12.5%** を実証 (filter で hallucination の 5/6 を drop)
  - **Section 3 (language coverage)** — English native validate、非英語入力時
    の language mismatch も実機確認 (production user 注意点として docs 記載)
- **Tests** (新規 +3 件、合計 655 passed):
  - `tests/core/engines/test_parakeet_decoding_strategy.py` の pin を新挙動
    (Path 1.5 で confidence_cfg 試行) に整合、CTC failure fallback も Path 1.5
    に整合
- **Docs update**:
  - `docs/research/parakeet-english-confidence-smoke-2026-06-11.md` (新規)
  - `docs/audio-filter-reference.md`: Engine support table を Parakeet 英語
    ✅ Production に更新、PR-A 系列完成サマリ 5 engine 対応に拡大
  - `docs/reference/cli.md` / `docs/reference/api.md`: Parakeet 英語追加
  - `livecap_cli/cli.py` / `confidence_filter.py` / `base_engine.py` /
    `stream.py`: 全 layer で Parakeet (ja/en) 一貫表示

#### PR-A 系列完成 docs 整合 (Issue [#311] PR-A.4.docs)

Issue #311 v2.1 plan の最終 PR。PR-A.4.1 ([#313 MERGED]) で Voxtral、PR-A.4.2
([#315 MERGED]) で Canary の filter 対応を完了した後の **docs 整合 sweep**。

- **Before**: 一部 docs に stale な「voxtral / canary は fail-open」記述が
  残存:
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md:177` (PR-B calibration
    時の残作業 list、voxtral/canary がまだ fail-open とされていた)
  - `docs/research/voxtral-confidence-smoke-2026-06-11.md:108` (他 engine
    の挙動 section に Canary が fail-open list で残存)
- **After**:
  - `pr-a-calibration-2026-06-10.md`: PR-A.4.1/A.4.2 完了状況を反映、qwen3asr
    のみ PR-A.5 candidate として残存する旨を明示
  - `voxtral-confidence-smoke-2026-06-11.md`: Canary を populate engine list
    に移動 (PR-A.4.2 整合)
  - `docs/audio-filter-reference.md`:
    - Property table の Production-ready statement を 4 engine (WhisperS2T /
      Parakeet_ja / Voxtral / Canary) に拡張
    - Comparison table の Confidence Filter 行を「4 engine 対応」+ 50%→0%
      実測実証を反映
    - **新 section: PR-A 系列 完成サマリ** (2026-06-11 時点) を追加 — Engine
      support table の最終状態 / production user 選択ガイド / Phase 1 多段
      防御 5 layer 到達点を 1 section に集約
- **Side effects**:
  - 全 docs 層 (audio-filter-reference / cli.md / api.md / feature-inventory
    / decision doc × 2 / CHANGELOG / Engine support table / source docstring)
    で **Canary が `token_confidence_mean` populate engine** として一貫表示
    完了
  - Issue #311 v2.1 plan の Core scope (PR-A.4.1 + PR-A.4.2 + PR-A.4.docs)
    が完了、close 候補に
- **Out of scope (PR-A.5 candidate に申し送り)**:
  - qwen3asr: qwen-asr wrapper が内部で ``output_scores=True`` を渡さず、
    ``text_ids = model.generate(...)`` のみ実行 ([source 確認済](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py))。
    wrapper bypass or vLLM logprobs 移行が必要 (heavy)。
  - ReazonSpeech: sherpa-onnx Python bindings に per-token score API
    なし。upstream [PR #2897](https://github.com/k2-fsa/sherpa-onnx/pull/2897)
    が C/Dart で `getVocabLogProbs()` 追加したが closed (not merged)、
    Python 未対応。upstream PR or PyTorch native 実装切替が必要。
- **🆕 PR-A.4.3 candidate (Issue #311 v2.2 へ申し送り)**:
  - **Parakeet 英語** (`parakeet-tdt-0.6b-v2`): 旧 docs では「NeMo RNNT
    path に token_confidence 未実装」を根拠に PR-A.5 candidate としていた
    が、本 PR 作業中の調査で **「構造的限界ではなく PR #309 時点の設定漏れ」**
    と判明。NeMo source (`rnnt_decoding.py:95-106`, `tdt_loop_labels_computer.py`)
    で `preserve_token_confidence` documented + 実装あり、`preserve_alignments
    =True` 同時設定で populate される。**実機 probe (本 PR docs 作業中)**
    で `token_confidence_mean = 0.2452` を確認 (LibriSpeech 英語、threshold
    0.005 の 49x)。実装は別 PR (PR-A.4.3) で対応 — 本 PR は docs scope の
    ため probe 修正は revert 済、PR-A.4.3 で改めて Path 1.5 実装 + smoke
    verify + 完全 docs 整合を実施予定。
  - **PR-A.4.3 acceptance criteria** (codex-review PR #316 3rd round 提示):
    1. ``parakeet_engine.py::_configure_decoding_with_confidence`` の pure
       RNNT/TDT path に ``preserve_alignments=True`` と
       ``confidence_cfg.preserve_token_confidence=True`` を含む dedicated
       path (Path 1.5) を追加
    2. その path が失敗した場合は現行の strategy-only path に fail-open
       fallback (Path 2 既存)
    3. ``tests/core/engines/test_parakeet_decoding_strategy.py:113-138``
       を更新し、Parakeet English で confidence cfg を試行することを pin
       (現状は pure RNNT で confidence_cfg を含めないことを pin している
       ため、PR-A.4.3 で挙動変更に合わせ更新必須)
    4. ``tests/core/engines/test_parakeet_confidence_extraction.py`` は
       既存 helper が list/tuple の ``token_confidence`` を扱えているため
       大枠流用可能。Parakeet English の hypothesis shape が異なる場合
       (Tensor / numpy 等) は fixture 追加
    5. 実機 smoke で ``token_confidence_mean`` populate + speech が
       threshold ``0.005`` を十分上回ること確認 (本 PR probe で 0.2452 =
       49× を確認済、smoke verify で再現性確保)

#### Engine confidence filter — Canary support (Issue [#311] PR-A.4.2)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) で whispers2t / parakeet_ja / Voxtral に
対応した confidence filter を **Canary 1B Flash** にも拡張。NeMo
``EncDecMultiTaskModel`` の **beam → greedy decoding 切替** +
``confidence_cfg.preserve_token_confidence=True`` で
``hypothesis.token_confidence`` (torch.Tensor) を取得、Parakeet 同様の
``token_confidence_mean`` を ``EngineConfidence`` に populate。

- **Before**: Canary の ``transcribe()`` は ``engine_confidence = EngineConfidence()``
  (全 None) で fail-open、``confidence = 1.0`` ハードコード。
- **After**:
  - ``canary_engine.py::_configure_model()`` で ``_configure_decoding_with_confidence()``
    呼出 (3-fallback path、Parakeet pattern 流用): Greedy + confidence_cfg →
    Greedy only → argument-less。いずれも raise しない。
  - ``_transcribe_single_chunk()`` に ``return_hypotheses=True`` 追加、
    Hypothesis から ``_extract_engine_confidence()`` 経由で
    ``token_confidence_mean`` 取得。
  - **新 helper**: ``_extract_engine_confidence(hypothesis)`` — Canary は
    ``token_confidence: torch.Tensor`` (Parakeet は ``List[float]``) で
    返すため ``hasattr(token_conf, 'tolist')`` 防御で GPU tensor / numpy /
    list を統一処理。
  - ``confidence = float(token_confidence_mean)`` で UI display 意味化。
  - **filter logic 変更なし**: ``confidence_filter.py::should_reject()``
    の ``token_conf_threshold = 0.005`` path を Parakeet_ja と共用。
  - **新 doc**: ``docs/research/canary-confidence-smoke-2026-06-11.md`` に
    Phase 1 probe + Section 1/2/3 を永続化。
- **Migration**:
  - **WhisperS2T / Parakeet_ja / Voxtral 退行ゼロ** (filter logic 不変)。
  - **Canary user** は ``--confidence-filter on`` (default) で対応言語
    (en/de/fr/es) の hallucination が自動 drop される。
  - decoding strategy が beam → greedy に切替 (NeMo AED の confidence 取得
    のため)。Parakeet_ja TDT→CTC 同様、軽微 WER 退行可能性あるが filter
    benefit を優先。**``--confidence-filter off`` は post-ASR の reject を
    止めるだけで、decoding は常に greedy** (filter logic と decoding strategy
    を独立に管理)。旧 beam decoding に戻す option は本 PR では非提供。
  - **``beam_size`` parameter 削除 (silent no-op cleanup、PR-A.4.2 →
    Issue #321 PR #1 で完成)**: 旧 ``CanaryEngine`` constructor の
    ``beam_size: int = 1`` および ``metadata.default_params`` の
    ``beam_size`` は ``_configure_decoding_with_confidence()`` で常に
    greedy 切替されるため silent no-op だった。pre-1.0 cleanup 方針に従い
    削除。
    - **Before**: ``CanaryEngine(beam_size=N)`` は warn + silent ignore
    - **After** (Issue #321 PR #1): ``CanaryEngine.__init__`` から
      ``**kwargs`` を完全削除、``CanaryEngine(beam_size=N)`` は
      ``TypeError: unexpected keyword argument 'beam_size'`` で fail-fast
    - **Migration**: caller は ``beam_size`` 指定を削除する必要がある
      (greedy が常に有効、beam search に戻す option は本 PR では非提供)
- **Findings**:
  - **Phase 1 probe** ✅ — ``hypothesis.token_confidence`` は **torch.Tensor**
    で populate (Parakeet と型差分、helper で吸収)。LibriSpeech 英語 →
    token_confidence_mean = **0.0724**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.0724 vs threshold
    0.005 で **14.5x margin** (Case A)。非音声は engine 自体が empty text を
    返す **fail-safe 挙動** (Voxtral/Parakeet と異なり Canary は元々
    hallucinate しない)
  - **Section 2 (stream pipeline benchmark、12 cell)**:
    - Hall.(pre) = 0% 全 cell (Canary 固有の robustness)
    - SR(post) = 0% all real cells — PR-B corpus は日本語、Canary 非対応
      で engine が empty text を返す fail-safe
    - filter on/off で同一結果 (pre-filter hallucination が 0% のため filter
      効果が観察不可、Section 1 で margin 検証済)
    - Latency 影響なし
  - **Section 3 (language coverage)** ✅ — English native で margin 確認済、
    Japanese は Canary 非対応で empty text 返却 (Voxtral の translation
    regime と異なり誤訳混入なし)
- **Out of scope**: qwen3asr / reazonspeech / parakeet_en は **PR-A.5**
  (heavy refactor)。Canary 他言語 verify は user feedback ベース。
  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は本 PR の probe で
  > 「実は populate 可能 (PR #309 時点の `preserve_alignments` 併設漏れ)」と
  > 判明、**PR-A.4.3 candidate に格上げ**。PR-A.5 は qwen3asr / reazonspeech
  > の 2 engine に縮減 (上の PR-A.4.docs entry 参照)。

#### ``TranscriptionResult.__iter__`` 削除 (pre-1.0 cleanup)

PR-A.0 ([#309]) で導入した ``TranscriptionResult.__iter__`` (旧
``Tuple[str, float]`` 戻り値との後方互換 shim) を削除。pre-1.0 (
`1.0.0.dev0`) では legacy compat shim は不要との CLAUDE.md / AGENTS.md
方針に従う。

- **Before**: ``text, confidence = engine.transcribe(audio, sr)`` の
  tuple unpacking 形が動作 (``TranscriptionResult.__iter__`` 経由)。
- **After**: ``result = engine.transcribe(audio, sr); text =
  result.text; confidence = result.confidence`` の attribute access に
  統一。``__iter__`` 削除により tuple unpacking は ``TypeError`` で fail。
- **Migration**:
  - `livecap_cli/transcription/stream.py`: 3 path (sync / async /
    interim) を attribute access に migration
  - `benchmarks/asr/runner.py` / `benchmarks/common/engines.py` /
    `benchmarks/optimization/objective.py`: 同 migration
  - `tests/`: 4 mock engine fixture (test_stream.py /
    test_stream_translation.py / test_mock_realtime_flow.py /
    test_from_language_integration.py) を ``Tuple[str, float]`` 返却 →
    ``TranscriptionResult`` 返却に統一
  - `tests/core/engines/test_engine_confidence_schema.py`:
    ``__iter__`` 互換を pin する 3 件を削除、代わりに
    ``test_tuple_unpacking_no_longer_supported`` で ``TypeError`` 化を
    pin
  - `livecap_cli/engines/{base,whispers2t,parakeet,canary,qwen3asr}_engine.py`
    docstring から「tuple unpacking 互換」記述削除
  - `livecap_cli/transcription/stream.py` / `confidence_filter.py`
    docstring も同様に整合
- **Side effects**:
  - `livecap_cli/engines/shared_engine_manager.py:443-449` の defensive
    3-branch (attribute / ``__getitem__`` / fallback) は不変。
    ``__iter__`` に依存しない別 path のため独立 cleanup PR で対応可。
  - `livecap_cli/engines/reazonspeech_engine.py:430` の internal helper
    ``_transcribe_single()`` も不変 (``TranscriptionResult`` ではなく
    raw tuple を返す内部関数、cleanup scope 外)。
- **Tests**: 508 → 506 passed (削除 3 - 新 1 = -2)。

#### Engine confidence filter — Voxtral support (Issue [#311] PR-A.4.1)

PR-A.0 ([#309]) / PR-A.1 ([#310]) で whispers2t / parakeet_ja に対応した
confidence filter を **Voxtral** にも拡張。`transformers.GenerationMixin.compute_transition_scores(normalize_logits=True)` 経由で per-token logprob
を復元、special token (EOS/PAD/BOS) 除外平均を `EngineConfidence.avg_logprob`
field に populate する。

- **Before**: Voxtral の `transcribe()` は `engine_confidence = EngineConfidence()`
  (全 None) で返却 → `is_available=False` → filter 常に fail-open。`confidence`
  field は `1.0` ハードコード。
- **After**:
  - `livecap_cli/engines/voxtral_engine.py` の `_transcribe_single_chunk()`
    を `model.generate(output_scores=True, return_dict_in_generate=True)` 化、
    `compute_transition_scores` で score 復元、`_extract_engine_confidence()`
    helper で special token 除外平均を計算。
  - `confidence = exp(avg_logprob)` で UI confidence display を意味化
    (PR-A.0 の WhisperS2T と整合)。
  - **新 helper**: `_extract_engine_confidence(transition_scores, gen_tokens, special_ids) -> EngineConfidence`
    — pure function として export、PR-A.0 の whispers2t / parakeet 同パターン。
  - **filter 拡張**: `confidence_filter.py::should_reject()` に **strict-gated**
    `avg_logprob` 分岐追加。`no_speech_prob is None` AND
    `token_confidence_mean is None` の時のみ評価 (WhisperS2T 退行回避)。
  - **新 default**: `FilterConfig.avg_logprob_threshold = -1.0` (PR-A.4.1
    smoke verify 結果に基づく決定)。Voxtral smoke (2026-06-11, RTX 4090) で
    speech 4 clip mean=-0.420 vs non-speech mean=-1.525、margin +1.002、
    midpoint -1.024 → `-1.0` で 100% 分類可能。
  - **新 doc**: `docs/research/voxtral-confidence-smoke-2026-06-11.md` に
    decision (Setup / 4 Hypotheses / Results / Decision / Implications /
    Reproducibility) を永続化。
  - **docs update**: `docs/audio-filter-reference.md` の Engine support
    table を Voxtral ✅ avg_logprob (gated) に更新。
- **Migration**:
  - **WhisperS2T / Parakeet_ja 退行ゼロ** (strict gate により avg_logprob
    経路に到達しない、unit test で pin)。
  - **Voxtral user** は `--confidence-filter on` (default) で hallucination
    が自動 drop されるようになる (production behavior 変化、PR-A.4.1 smoke で
    実証)。
  - Python API で `FilterConfig(avg_logprob_threshold=None)` を明示すれば
    avg_logprob 判定経路を opt-out 可能 (e.g., Voxtral debugging 時)。
  - ReazonSpeech / qwen3asr / Canary / mock は `engine_confidence` 全 None
    のまま (fail-open 不変)。
- **Findings (詳細は `docs/research/voxtral-confidence-smoke-2026-06-11.md`)**:
  - **Section 1 (engine-level smoke、6 clip)**:
    - H1 ✅ — Voxtral speech 4 clip max=-0.354、min=-0.523、worst でも -1.0 上
    - H2 ✅ — Voxtral non-speech `applause_5_claps`: -1.525 << -1.0、
      `desk_tap`: empty (全 EOS) → fail-open
    - H3 ✅ — clear margin +1.002 (Case A 確定)
    - H4 ✅ — special token 除外 logic で `desk_tap` (全 EOS) は
      `EngineConfidence()` を返す (filter fail-open)
  - **Section 2 (stream pipeline benchmark、12 cell sweep)**:
    - F2.1 ✅ — **`webrtc × voxtral × real × filter on`: post-filter
      hallucination 50% → 0%**、post-filter speech recall 100% 維持。
      PR-A.4.1 核心 claim を stream pipeline 経由で実機実証。
    - F2.2 ✅ — silero / tenvad × voxtral × real は filter on/off 関係なく
      0% 維持 (副作用ゼロ、VAD 段階で既に non-speech 除去)
    - F2.3 — synthetic positive (formant proxy) は filter on で SR(post)
      40-60% drop。PR-A.3 H3.b と同じ意図通り挙動 (real speech ではない)
    - F2.4 — synthetic Hall.(post) は partial drop (75% → 25% on webrtc)、
      残存は threshold -1.0 と real corpus 100% 維持の trade-off
    - F2.5 ✅ — latency 影響なし (p50/p95 は filter off と同等)
  - **Section 3 (language-stratified follow-up)**: 旧 Section 1 は
    **日本語音声 × language="en"** で実行されていたが Voxtral は en/es/fr/
    pt/hi/de/nl/it の 8 言語のみサポート (ja は対象外)、結果として旧 smoke
    は **translation regime** (ja→en) を測定していた。native English
    transcription (LibriSpeech) で再検証:
    - F3.1 ✅ — Native transcription: avg_logprob -0.115 (translation
      mean -0.420 より 0.305 高信頼度)
    - F3.2 ✅ — Threshold -1.0 は translation regime の lower bound に
      calibrate されていた → native regime では margin +1.410 (translation
      +1.002 から拡大)、両 regime で validate 完了
    - F3.3 — 言語 coverage: en (native + translation) のみ、他 7 言語は
      merge 後 user feedback で順次検証 (false reject 報告時は
      `FilterConfig(avg_logprob_threshold=None)` opt-out 可能)
- **Out of scope (次の handle)**:
  - **Canary** filter 対応: **PR-A.4.2** (NeMo `EncDecMultiTaskModel`、
    beam→greedy decoding 切替が gate)
  - **qwen3asr / reazonspeech / parakeet_en**: **PR-A.5** (wrapper bypass /
    vLLM 移行 / sherpa-onnx 構造的限界などの heavy refactor 系)
  - **Voxtral non-English language** での smoke verify: user feedback で
    順次対応 (本 PR は `language="en"` で実施)

  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は PR-A.5 から外され
  > **PR-A.4.3 candidate** に格上げ済 (probe で `token_confidence_mean = 0.2452`
  > 確認、threshold 0.005 の 49×)。PR-A.5 は qwen3asr / reazonspeech の 2
  > engine に縮減。詳細は最上段 PR-A.4.docs entry を参照。

#### Confidence filter calibration sweep + new `post_filter_hallucination_rate` metric (Issue [#308] PR-A.3)

PR-A.1 ([#310]) で実装した confidence filter を 54 cell sweep
(1 preset × 3 backend × 3 engine × 2 corpus × 3 filter_mode) で validate
し、Phase 1 epic ([#295]) closure を数値証拠付きで achievable な状態に。

- **Before**: PR-A.1 で filter 本体は実装済だが、production 全 cell での効果は
  smoke verify (6 clip × 2 engine = 12 ケース) のみで実証。`benchmarks/
  non_speech_filter/` の既存 metric (`false_asr_trigger_rate` / `non_empty_
  hallucination_rate`) は **engine の生出力** を測定しており、filter 適用後
  の user の subtitle stream に届く text は計測できなかった。
- **After**:
  - `benchmarks/non_speech_filter/runner.py` の `NonSpeechFilterBenchmark
    Config` に `filter_config: Optional[FilterConfig] = None` を追加、
    `_make_pipeline_factory()` で `build_pipeline()` に pass-through。
  - `benchmarks/non_speech_filter/sweep.py` の `run_sweep()` に
    `filter_mode ∈ {off, observe, on}` の 3 段 nested loop を追加。
    `SweepCellResult` に `filter_mode` field 追加、CSV / Markdown 出力に
    column 追加。
  - **新 metric `post_filter_hallucination_rate`** を `evaluate_pipeline()`
    に追加。`transcriber.finalize()` 戻り値 + `_result_queue` の直接 drain
    (`InterimResult` を明示的に skip し `TranscriptionResult` のみ収集)
    を合算することで、user の subtitle stream に実際に届く text を計測する。
    `non_empty_hallucination_rate` (pre-filter engine 直出力) と並列で出力。
    旧版 (initial commit) は `finalize()` 戻り値を取りこぼし、queue drain も
    interim 先頭で停止する 2 件の bug があったため、codex-review on #312
    1st + 2nd round で修正済。
  - **新 metric `post_filter_speech_recall` / `post_filter_short_utterance_recall`**
    を追加 (codex-review on #312 3rd round Item 1 HIGH)。旧 `speech_recall`
    は engine call の counter で計測されており、filter が legit speech を
    drop しても 1.0 のままだった。新 metric は user の subtitle stream に
    届く speech 比率を直接測定。`measure_hallucination=True` 時のみ意味あり。
  - `_collect_post_filter_texts` helper を追加 (codex-review #312 3rd round
    Item 2 MED)。`_result_queue` 直接 access を helper に閉じ込め、将来の
    StreamTranscriber queue 実装変更時の修正箇所を 1 箇所に集約。
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md` 新規 — PR-A 系列
    (A.0/A.1/A.3) の calibration 総括 doc を PR-B (2026-06-07) と同じ
    Setup / Hypotheses / Findings / Decision / Implications / Reproducibility
    構造で執筆。
- **Migration**: 既存 sweep を本 PR の harness で再実行すると、新 column
  `post_filter_hallucination_rate` が CSV / Markdown に追加されている。
  既存の `non_empty_hallucination_rate` semantics は不変 (pre-filter
  engine 出力を測定)、解釈時には 2 列を比較して filter 効果を測る。
- **Findings (詳細は `docs/benchmarks/pr-a-calibration-2026-06-10.md`)**:
  - H1 ✅ — `webrtc × parakeet_ja × real desk_tap` filter `on` で
    50% → 0%、`synthetic` でも 75% → 0% を実証。Issue #295 の元 motivation を
    実機で完全解決。
  - H1.b ✅ — synthetic corpus で WhisperS2T 内部 `no_speech_prob` filter
    を bypass する edge case (25%) も filter `on` で 0% に。重複防御として
    実効的に機能。
  - H2 ✅ — `silero / tenvad × all engines` で filter mode に関係なく
    0% 維持 (production user の副作用ゼロ)。
  - H3 ✅ (v3 refined) — **real corpus** で post-filter SR = 100% 維持
    (filter は legit speech を 1 件も drop していない)。synthetic positive
    の SR(post) drop は filter が formant proxy を正しく低信頼度として
    drop している = 期待挙動。production user は real speech を扱うため
    real corpus の結果が production 挙動。
  - H4 — `BASELINE_INVARIANTS` は不変判断。CI test は synthetic + Mock
    Engine で filter は fail-open のため tighten 不要。
  - ReazonSpeech — `engine_confidence` 全 None で filter fail-open。
    `post_filter = pre_filter` で filter 効果なし (sherpa-onnx 構造的限界、
    PR-A.5 で長期対応)。
- **Side effects**:
  - `benchmarks/non_speech_filter/report.py::NonSpeechFilterRunRecord`
    に `post_filter_hallucination_rate: float | None = None` field 追加。
    既存 caller は default None で動作するため後方互換。
  - 既存 sweep test 全 pass 維持。新規 sweep axis test 5 件追加で
    filter_mode 軸の挙動を pin。
  - `CHANGELOG.md` の本 entry で PR-B との比較を可能に。
  - Phase 1 epic ([#295]) close 候補状態に到達 — 残作業は PR-A.4
    ([#311] qwen3asr/voxtral/canary の filter 拡張、別 track)。

#### Engine confidence filter — default ON (Issue [#308] PR-A.1)

Adds the post-ASR `livecap_cli.transcription.confidence_filter` module
that watches `engine_confidence` (PR-A.0) and silently drops outputs the
engine itself judged as non-speech, before they reach the subtitle stream.
Default is **on** for all realtime sessions.

- **Before** (PR-A.0): every engine output reached the subtitle stream,
  even when `no_speech_prob` was high (WhisperS2T) or
  `token_confidence_mean` was near zero (Parakeet_ja). The PR-B 144-cell
  matrix showed `webrtc × parakeet_ja × desk_tap` hallucinated 50 % of
  the time.
- **After** (PR-A.1): `StreamTranscriber` calls `apply_filter()` on the
  3 call sites (sync L566 / async L638 / interim L787). For
  `--confidence-filter on` (default), rejected outputs become `None`
  drops with a structured INFO log; the subtitle stream sees nothing.
  Real-machine smoke verify on the 6-clip PR-B corpus produced 100 %
  classification (all speech clips passed, both non-speech clips dropped
  on both whispers2t and parakeet_ja).
- **Migration**: existing scripts keep the same flags. To restore the
  previous behavior, pass `--confidence-filter off` or export
  `LIVECAP_CONFIDENCE_FILTER=off`. Engines that do not expose confidence
  signals (`reazonspeech`, `qwen3asr`, `voxtral`, `canary`) are
  pass-through regardless of the flag (fail-open by design).
- **Side effects**:
  - Per-engine thresholds (`whispers2t no_speech_prob > 0.5`,
    `parakeet_ja token_confidence_mean < 0.005`) are baked in from the
    PR-A.0 verify values; PR-A.3 will revisit them after a full 144-cell
    sweep.
  - The `--confidence-filter observe` mode emits the same structured
    decision log **for both pass and reject decisions** (codex-review
    #310 Item 4) — PR-A.3 calibration needs the speech-side
    `engine_confidence` distribution as well, not just the reject side,
    to evaluate threshold margins and speech-recall safety. The `on`
    mode keeps logging reject only to avoid production log spam. Log
    payload is JSON (stable schema documented in `_decision_to_dict()`
    of `confidence_filter.py`) so PR-A.3 parsers can read it as JSONL.
  - The `StreamTranscriber.__init__` gained a `filter_config:
    Optional[FilterConfig] = None` parameter; `None` constructs the
    default (on) at instantiation time. Direct API users who want the
    old behavior should pass `FilterConfig(mode="off")` explicitly.
  - `benchmarks/non_speech_filter/pipeline.py::build_pipeline` defaults
    to `FilterConfig(mode="off")` so existing sweep baselines remain
    bit-identical; PR-A.3 will pass `FilterConfig(mode="on")` to
    measure filter impact on the cell matrix.
  - **Scope clarification (codex-review #310 Item 3)**: this PR exposes
    `filter_config` on `build_pipeline()` only. Adding a
    `confidence_filter` axis to `benchmarks/non_speech_filter/sweep.py`
    (so that the existing preset/backend/engine matrix is multiplied by
    `{off, observe, on}`) is **deferred to PR-A.3**, together with the
    full 144-cell sweep run. The pipeline-level hook here is sufficient
    for PR-A.3 to construct sweep cells programmatically.
  - A startup INFO log line (`"Confidence filter: ON (...)"`) makes the
    active mode visible at every session start.

#### Parakeet_ja decoder strategy: RNNT greedy → CTC greedy_batch (Issue [#308] PR-A.0)

Investigation during PR #309 smoke verify uncovered that the
`nvidia/parakeet-tdt_ctc-0.6b-ja` checkpoint is an
`EncDecHybridRNNTCTCBPEModel` whose RNNT decoder (NeMo default) does not
implement `token_confidence`. The CTC decoder does. To make
`engine_confidence` actually populated for `parakeet_ja`, the adapter now
switches to the CTC decoder on `load_model`.

- **Before**: `parakeet_ja` used the RNNT decoder with `strategy=greedy`
  and the old `confidence_cfg` block — which NeMo silently rejected on
  the current version (`preserve_frame_confidence=True` requires
  `preserve_alignments=True`), so `token_confidence` was always `None`
  and the old `score / len(y_sequence)` fallback returned an
  empirically-inverted signal (speech `-71.5` vs applause `-47.3`).
- **After**: `_configure_decoding_with_confidence()` detects the hybrid
  model via `hasattr(self.model, 'cur_decoder')` and switches to
  `decoder_type='ctc'` with `strategy=greedy_batch`,
  `greedy.preserve_frame_confidence=True`, and a full `confidence_cfg`.
  `token_confidence_mean` is now populated with clean speech-vs-noise
  separation (0.01-0.10 vs 0.0000029-0.0003). A 3-stage fallback path
  protects against older NeMo versions and the non-hybrid English
  `parakeet` model.
- **Migration**: `EngineMetadata.default_params["parakeet_ja"]
  ["decoding_strategy"]` updated from `"greedy"` to `"greedy_batch"`
  (single source of truth, surfaced by GUI / diagnostics / docs). The
  English `parakeet` (pure RNNT) default remains `"greedy"`. Users who
  hard-coded `decoding_strategy="greedy"` on `parakeet_ja` will keep
  working (CTC greedy is slower but functional, NeMo emits a
  `greedy_batch` recommendation warning).
- **Side effects** measured on RTX 4090 with the
  `.tmp/non_speech_corpus/` 6-clip set:
  - **Latency** improves: CTC + `greedy_batch` runs 1.83× faster on the
    speech clip than the old RNNT `greedy` path (p50 81.4 ms vs
    149.8 ms).
  - **Transcription text** is preserved on 4/6 clips, slightly improved
    on 1/6 (`applause_5_claps` hallucinates fewer characters), and
    differs by 1 hiragana on 2/6 (e.g. 「とんと」 → 「とんど」). Not
    a regression for production usage; documented in
    `docs/research/parakeet-ja-confidence-spec-2026-06-10.md`.
- **Score fallback removed**: the previous
  `score / len(y_sequence) → avg_logprob` path inside
  `_extract_engine_confidence` is gone. When `token_confidence` is not
  available (older NeMo, non-hybrid model, or fallback path), the
  function returns `EngineConfidence()` honestly. The smoke-verified
  signal inversion made the old fallback actively harmful for the
  PR-A.1 filter.

This is an **engine behavior change** for `parakeet_ja`, but it
strengthens (not regresses) production behavior on every measured axis:
faster, comparable text, and a now-usable confidence signal.

#### Phase 2 SED model evaluation harness (Issue [#305] PR-D0)

- **New `benchmarks/sed/` package (research-only off-line evaluation;
  does not touch `livecap_cli/`)**:
  - `class_mapping.py` — pins the AudioSet 527-class taxonomy mapping for
    livecap reject signals. Defines `TARGET_CLASSES` (Hands / Finger
    snapping / Clapping / Applause / Door / Sliding door / Slam / Knock
    / Tap / Thump, thud — 10 classes) and `SPEECH_LIKE_CLASSES` (Speech
    family + Singing — 7 classes). Implements the three Issue #305 v3
    threshold policies (`max`, `sum`, `target_minus_speech`).
  - `inference.py` — loads EfficientAT pretrained models
    (`mn04_as` / `dymn04_as` / `dymn10_as`), resamples 16 kHz audio to
    32 kHz, slices into 1-second windows (Issue #305 v3 primary metric
    unit), and returns per-window 527-dim sigmoid probability matrices.
  - `metrics.py` — clip-level confusion-matrix metrics with hand-pinned
    semantics: class-level + reject-signal-level (Issue #305 v3 two-axis
    report), provisional-gate verdict (`precision ≥ 0.70` AND
    `recall ≥ 0.50` AND target clip flagged at the chosen threshold).
  - `latency.py` — 5-axis runtime measurement (checkpoint size, installed
    dep delta vs `engines-torch` baseline, runtime peak memory via
    `tracemalloc`, CPU + GPU p50/p95 latency, cold-start) per Issue #305
    v3 Dimension 3 refinement.
  - `orchestrator.py` — full evaluation pipeline (corpus → inference →
    CSVs + NPZ + JSON metadata).
  - `analyze.py` — post-hoc analysis: threshold sweep, class-level
    summary tables, provisional-gate verdict, `analysis.{json,md}` for
    decision-doc paste-in.
  - `cli.py` + `__main__.py` — `python -m benchmarks.sed` entry point.
  - `README.md` — manual EfficientAT setup, env vars, command reference.
- **New `tests/integration/sed/` (23 tests, `sed_evaluation` marker)**:
  - `test_class_mapping.py` (12 tests) — AudioSet index integrity, three
    policy semantics, validation; the
    `test_indices_match_efficientat_csv` test cross-checks the pinned
    indices against the canonical AudioSet CSV when EfficientAT is
    cloned.
  - `test_metrics.py` (10 tests) — synthetic 4-clip corpus with
    hand-derived precision/recall, gate truth-table (pass / precision
    fail / recall fail / target-not-flagged).
  - `test_inference_smoke.py` (1 test) — env-gated 1-clip smoke
    verifying `(n_windows, 527)` output shape; skipped automatically
    when the EfficientAT clone is absent.
  - New `sed_evaluation` pytest marker declared in `pyproject.toml`.
- **New `docs/research/phase2-sed-evaluation-2026-06-10.md` (~430
  lines)** — 4-dimension decision document covering Accuracy / Safety /
  Runtime / License (verdicts honest after codex-review on #306):
  - Accuracy: PASS (provisional). `target_minus_speech` policy at
    threshold ~0.10 yields precision=1.0, recall=1.0 on the 6-clip
    corpus; the critical `overlapping_applause_speech` case is correctly
    retained (`max(target)=0.16` would over-fire, but
    `target − speech_like = -0.66` correctly suppresses).
  - Safety: PASS. speech_recall = 1.00, short_utterance_recall = 1.00.
  - Runtime: **Conditional PASS** (CPU production-device path). CPU p95
    = 29.0 ms (3.4× under the 100 ms budget), checkpoint 4.07 MB
    (12× under 50 MB), runtime peak 6.68 MB (30× under 200 MB).
    **GPU p95 = 32.8 ms misses the original 30 ms ceiling by 9 %** —
    documented honestly rather than papered over; CPU runs faster than
    GPU at this 3.9 M-parameter scale, so production device = CPU and
    the CPU budget is satisfied.
  - License: PASS at the **Auto-download OK tier** (not Bundle OK —
    corrected after codex-review). The upstream EfficientAT release
    does not explicitly grant a license on the model weights, so the
    integration ships the checkpoint via `torch.hub` auto-download
    rather than packaging the `.pt` file; this matches both the legal
    evidence we have and the implementation already in use.
    Attribution stub recorded for PR-D1's `THIRD_PARTY_NOTICES.md`.
- **New `benchmark_results/sed/2026-06-10/` (committed per Issue #305 v3
  artifact policy)**: `probabilities.csv`, `probabilities_full.npz`,
  `latency.csv`, `metadata.json`, `analysis.json`, `analysis.md`.
- **`.gitignore` update**: changed `benchmark_results/` to
  `benchmark_results/*` with `!benchmark_results/sed/` exception so the
  PR-D0 evidence is committed while other benchmark outputs remain
  ignored. Added `.tmp/` to ignore the EfficientAT clone and any
  research scratch.
- **Issue #305 v2 → v3 body update** with six clarifications: metric
  calculation unit (window primary / clip-level max decision unit),
  license outcome 4-classification (Bundle OK / Auto-download OK /
  Manual user-provided only / NG), artifact commit policy, accuracy
  provisional-gate disclaimer, runtime constraint detailing
  (checkpoint / installed dep / runtime peak memory split), class-level
  + reject-signal-level two-axis metric report.
- **Scope discipline**: this PR does **not** modify any file under
  `livecap_cli/`. SED pipeline integration is PR-D1; default-decision
  is PR-D2; DSP detector disposition is PR-D2.

Verification: `pytest tests/integration/sed/` → 23 passed, 0 failed.
PR-relevant regression
(`tests/audio tests/integration/non_speech_filter tests/integration/vad
tests/transcription tests/core/cli tests/audio_sources`) → 307 passed,
5 skipped (env-gated), 0 failed — identical to the pre-PR baseline.

[#305]: https://github.com/Mega-Gorilla/livecap-cli/issues/305

#### Calibration follow-up: real-engine sweep + threshold tuning (Issue [#295] PR-B follow-up)

- **3 new hypothesis-driven candidate presets** appended to
  `benchmarks/non_speech_filter/sweep.py::default_named_presets()`:
  - `on_relaxed_rms` — drop `rms_min_db` floor from -35 to -45 to admit
    quieter real-corpus frames (real recordings sit at -41 to -46 dBFS
    overall, so the default floor was rejecting > 95 % of frames before
    the AND combination could fire).
  - `on_low_freq_aware` — widen the spectral centroid window
    (`centroid_min_hz` 2500 → 500) and tighten `voiced_max` (0.25 →
    0.15) to test whether `desk_tap`-style low-frequency thumps can be
    caught without dropping low-pitched speech.
  - `on_speech_safe` — tightest preset (`flatness_min` 0.45,
    `centroid_min_hz` 3000, `onset_ratio` 5.0) as a safety ceiling that
    confirms short-utterance recall stays at 100 % under aggressive
    filtering.
- **New `benchmarks/non_speech_filter/calibration.py` (~430 lines)**:
  reads the CSV emitted by `sweep.py` and produces a structured Markdown
  report containing (1) per-engine hallucination delta vs `baseline_off`
  (segmented by backend and corpus), (2) recall-regression flags for any
  preset/cell pair that dropped recall below the baseline, (3) a Pareto
  summary across presets with explicit dominance markers, and (4) a
  structured recommendation driven by Issue #295 PR-B follow-up plan
  rule D4 (≥30 % hallucination drop on `webrtc × parakeet_ja × real`
  with no recall regression → promote that preset; otherwise default
  off, document gap, propose Phase 2 SED).
- **Calibration findings** (full record in
  `docs/benchmarks/calibration-results-2026-06-07.md`):
  - 144 cells (8 presets × 3 backends × 3 engines × 2 corpora) ran in
    ~16.5 min on a single RTX 4090 with engine-load amortisation.
  - **`parakeet_ja × WebRTC × real desk_tap` hallucination unchanged
    at 50 % across all 8 presets** (the PR-B v4 AC target).
  - Same on `reazonspeech × WebRTC × real desk_tap` (50 % → 50 %).
  - `parakeet_ja × WebRTC × synthetic burst` hallucination drops
    75 % → 62.5 % (one item out of eight) on the 4 Pareto-dominant
    presets (`on_moderate`, `on_aggressive`, `on_relaxed_rms`,
    `on_low_freq_aware`).
  - **Zero recall regressions** in any of the 144 cells.
- **Default mode decision: `--transient-filter=off` is maintained.**
  Rule D4's headline criterion (≥30 % hallucination drop on the AC
  target cell) is unmet, so no preset earns a promotion to default.
- **`on_moderate` is documented as the best observed DSP preset for
  synthetic rapid-burst tests only** — explicitly **not** a production
  hallucination mitigation recommendation. Calibration showed zero
  improvement on the real-corpus target cell.
- **The DSP transient detector layer is positioned as `experimental`
  going forward**: not deprecated (no replacement exists yet) but not a
  production-hallucination-mitigation candidate. CLI invocations of
  `--transient-filter observe/on` now emit a one-line experimental
  notice to make the status visible at the moment of opt-in. Phase 2
  SED (sound-event detection) is the planned successor for
  `desk_tap`-style low-frequency transients.
- **New `docs/audio-filter-reference.md`**: user-facing reference for
  every audio filter in the pipeline (NoiseGate / TransientDetector /
  VAD / EnergyGate) — purpose, pipeline position, CLI surface, default
  state, measured effectiveness with citations, recommendation, known
  limitations. Single doc users can scan to decide which filter to
  enable.
- **No detector code change**. The sweep + analysis is pure data
  collection; this PR does not modify
  `livecap_cli/audio/transient_detector.py` or any production pipeline.
- **Issue #295 v6** reframes the PR-B AC line `WebRTC × desk_tap (real)
  false_trigger 50 % → 0 %` with the empirically demonstrated bound
  ("0.0 pp achievable with 6-feature AND DSP detector — Phase 2 SED is
  the correct route").
- **BASELINE_INVARIANTS bounds remain unchanged** (default unchanged
  → no tightening warranted).
- **Out of scope** (separate follow-ups): Phase 2 SED epic for low-
  frequency / non-broadband transient detection; detector architecture
  changes (AND → OR, weighted-sum, new features); `#302` lookahead
  delay (still gated on a future reject default ON decision that this
  calibration data argues against).
- Verification: full PR-relevant suite (`pytest tests/audio/
  tests/integration/non_speech_filter/ tests/integration/vad/
  tests/transcription/ tests/core/cli/ tests/audio_sources/`) → **300
  passed, 5 skipped (env-gated), 0 failed**.

#### Fixed: PR-B follow-up — async path bypass + causal best-effort spec (Issue [#295] PR-B follow-up)

- **`StreamTranscriber.transcribe_async()` が transient detector を bypass**
  していた問題を修正。`feed_audio()` と `transcribe_async()` の pre-VAD
  処理を `_apply_pre_vad_processing()` 共通 helper に集約し、両 path が
  必ず NoiseGate → Layer 1 detector → VAD の順で走るよう pin。
- **`tests/transcription/test_stream.py::TestTransientDetectorWiring`**
  を追加 (3 cases): sync/async 両 path の detector 起動、両 path の
  telemetry 完全一致を assert。再発防止層。
- **`TransientDetector.process()` docstring** に causal / no-lookahead
  仕様を明文化。`on` mode の chunked output は best-effort upper bound で
  あり、bit-exact reconstruction ではないことを記載。32 ms lookahead-
  delay 拡張は別 issue で track。
- **`tests/audio/test_transient_detector.py::TestStreamingEquivalence::
  test_on_mode_chunked_is_causal_best_effort`** を追加: telemetry が
  full / chunked で一致することと、`on` mode で flagged frame があれば
  energy が入力より低下することを assert (full vs chunked output の
  bit-exact equality は意図的に検証しない)。
- **default mode を `off` のまま維持** + ドキュメント整合: Issue #295
  / docs / CHANGELOG では旧 v3/v4 で「default observe」と記載していた
  が、PR-B 実装は安全側の `default off` を採用。calibration は
  `--transient-filter observe` で明示的に opt-in する運用を明記。
- 既存挙動への影響: なし (default `off` で detector 構築されないため)。
- Verification: `pytest tests/audio/test_transient_detector.py
  tests/transcription/test_stream.py tests/integration/non_speech_filter/`
  → 86 passed, 8 skipped (env-gated).

#### Layer 1: DSP Transient/Applause Detector (Issue [#295] PR-B)

- **新規 `livecap_cli/audio/transient_detector.py`**: 6 DSP feature
  (`spectral_flatness` / `spectral_centroid_hz` / `zero_crossing_rate` /
  `onset_strength` / `voiced_ratio` / `rms_db`) を AND 結合して
  applause-like フレームを検出する frame-based stateful detector。
  3 mode: `off` (構築されない) / `observe` (telemetry のみ、audio 不変) /
  `on` (applause-flag frame を zero-out)。
- **`StreamTranscriber` 統合**: 新引数
  `transient_detector: Optional[TransientDetector] = None`、`feed_audio`
  の NoiseGate 後 / VAD 前で起動。`reset()` / `close()` テレメトリにも
  対応 (EnergyGate と同じ pattern)。
- **CLI flags** (`transcribe` サブコマンド): `--transient-filter`
  (`off`/`observe`/`on`、default `off`) + 6 threshold flag。
- **Benchmark CLI flags** 同名で揃え、`build_pipeline()` に
  `transient_config` kwarg を追加。`None` を渡せば baseline pipeline は
  bit-identical に保たれる (PR-0 BASELINE_INVARIANTS regression なし)。
- **新規 `benchmarks/non_speech_filter/sweep.py`**: 5 named preset
  (`baseline_off` / `observe_defaults` / `on_conservative` /
  `on_moderate` / `on_aggressive`) を回す threshold sweep harness。
  CSV + Markdown 出力。
- **テスト**:
  - `tests/audio/test_transient_detector.py`: 26 unit test
    (feature 算出 / AND 決定 / streaming 等価性 / mode semantics /
    config validation)。
  - `tests/integration/non_speech_filter/test_transient_detector_integration.py`:
    7 integration test (observe = no-op on metrics, on-mode positive
    preservation, WebRTC burst no-regression)。
- **検証結果 (private real corpus + synthetic、mock engine)**:
  - observe mode は BASELINE_INVARIANTS と完全一致 (silero 0/0/0 %,
    tenvad 25/100/100 %, webrtc 75/100/100 %)。
  - on mode (moderate/aggressive) で **WebRTC × synthetic burst の
    false_trigger 75 % → 62.5 %**。
  - WebRTC × real desk_tap は default 閾値で 50 % のまま (per-clip 観測で
    `centroid_min_hz=2500` が desk_tap の低域成分を弾いていることを確認、
    calibration follow-up で対応)。
- **限界 (docs に明示)**:
  - 既定閾値は **synthetic rapid burst 想定**で、private real corpus の
    個別 clip (desk_tap、scattered 拍手) は未較正。
  - reject default ON 化は PR-B のスコープ外 — calibration sweep の
    結果が出てから別 PR で実施する設計。
- **Out of scope**:
  - PR-C で予定する Layer 2 (VADStateMachine cooldown) との signaling は
    実装しない (検出器は event を emit するが消費側は別 PR)。
  - PR-A 系の confidence filter / prompt reset は対象外。

#### Non-speech filter evaluation harness (Issue [#295] PR-0)

- **新規 `tests/integration/non_speech_filter/`**: Phase 1 多段防御
  (DSP transient detector / VADStateMachine cooldown 拡張 / Confidence filter /
  Prompt reset) の **baseline + regression 検出基盤** を導入。3 VAD backend
  (Silero / TenVAD / WebRTC) × 13 件の synthetic corpus (negative 8 + positive 5、
  うち短発話 2) で現状 pipeline (NoiseGate + VAD + EnergyGate) を計測し、
  `baselines/{backend}.json` に schema v1 で永続化。後続 PR-B/C/A はこの JSON を
  比較基準にする。
- **新規 `benchmarks/non_speech_filter/`**:
  `python -m benchmarks.non_speech_filter` で ad-hoc 評価可能な runner +
  Markdown/JSON レポート。`--engine whispers2t` 等を指定すれば
  `non_empty_hallucination_rate` (engine が非空 text を返した負例の割合) も計測。
  実音源は `LIVECAP_NON_SPEECH_CORPUS_DIR` で manifest+WAV を渡すと自動 load。
- **指標**: `false_asr_trigger_rate` / `speech_recall` /
  **`short_utterance_recall`** (最重要) / `non_empty_hallucination_rate` (opt-in) /
  `added_latency_p50_ms` / `_p95_ms`。
- **新 marker**: `evaluation_harness` (pyproject.toml に登録)。`-m evaluation_harness`
  で opt-in 実行、CI baseline tests のみ拾う。
- **既存コード touch ゼロ**: `livecap_cli/` 配下は無変更。`benchmarks/common/`
  (`DatasetManager` / `BenchmarkEngineManager`) は real-engine 経路でのみ利用。
- **動機**: Issue #295 v2 のレビュー指摘
  「**評価ハーネス先行整備** + pre-engine 優先 + DSP detector default off-by-default
  + 実装前 corpus 整備」を満たすため、Phase 1 PR-B/C/A の前提として独立着地させる。
- **限界 (docs/benchmarks/non-speech-filter.md に明記)**:
  - 合成 speech proxy は Silero VAD (実音声学習) で recall=0 になる構造的限界。
    Silero baseline を意味のある形で測るには実音源 fixture が必要。
  - WebRTC backend は binary 出力 (0.0/1.0) のため、Phase 1 PR-C で導入予定の
    hysteresis は no-op (duration-based cooldown のみ機能)。
- **検証**:
  - `uv run pytest tests/integration/non_speech_filter/ -m evaluation_harness`
    → 6 passed (3 backend × 2 tests), 6 skipped (env-var gated)。
  - `uv run python -m benchmarks.non_speech_filter --mode quick --backend silero,tenvad,webrtc`
    → JSON + Markdown 出力、Silero / TenVAD / WebRTC の baseline 差を可視化。
  - 既存 `tests/integration/vad/` + `tests/audio/` の 74 test に regression なし。

### Changed

#### **BREAKING** `StreamTranscriber` に engine-input low-energy gate (EnergyGate) を追加 (Issue [#292])

- **Before**: VAD segment は energy 不問で全て `engine.transcribe()` に渡され、低 RMS / 純ノイズ segment で hallucination ("うん"/"ピッ"/"え?"/"どうぞ" 等) が発生。
- **After**: `StreamTranscriber` の 3 callsites (`_transcribe_segment` / `_transcribe_segment_async` / `_transcribe_interim`) で共通 helper `_should_skip_low_energy(audio, kind)` を呼び、per-segment energy が threshold 未満なら `engine.transcribe()` を skip。
- **動機**: `#291` (NoiseGate 単位ミスマッチ) は NoiseGate 有効時の primary fix だが、NoiseGate は opt-in (default off) で大半のユーザーは VAD のみで防御。VAD false-positive segment が engine に渡って hallucination する経路が残っていた (Mega-Gorilla/livecap-gui#331 の root-cause の一つ)。実音源 pre-evaluation で parakeet_ja は **silent audio に対して 100% hallucination** を確認、EnergyGate を経由すると -45 dBFS threshold で 26% 削減 (stress test)。production 条件 (VAD default threshold) では VAD が一次防御として効き、本機能は副次防御として機能する。
- **API 変更点 (`StreamTranscriber.__init__`)**:
  - 新引数: `engine_min_rms_dbfs: float = -45.0` — threshold (dBFS)。`float("-inf")` で完全 opt-out。
  - 新引数: `engine_energy_metric: str = "max_frame_rms"` — 4 metric から選択 (`max_frame_rms` / `whole_rms` / `p95_frame_rms` / `top3_frame_rms`)。default は VAD padding 希釈に耐性の `max_frame_rms`。
  - 新引数: `engine_energy_frame_ms: float = 32.0` — frame-based metric の窓長 (ms)。
- **CLI 変更**:
  - `transcribe` に `--engine-min-rms` / `--engine-energy-metric` / `--engine-energy-frame-ms` の 3 flag を追加。`--engine-min-rms` には custom type を実装し `off` / `disabled` / `none` 文字列を `float("-inf")` に map (argparse の leading-`-` value 制約のため bare `-inf` は不可、`=-inf` か `off` を使う)。
  - `levels` に `--engine-min-rms-margin` flag を追加 (default `+6 dB`)。`suggested_engine_min_rms_dbfs` の margin を user 任意に調整可能。
- **`NoiseAnalysis` 変更**:
  - 新 field: `suggested_engine_min_rms_dbfs: float` (= `noise_rms_p95_db + engine_min_rms_margin_db`)。CLI `levels` で 1 回の calibration から peak-unit / RMS-unit 両方の suggested 値が得られる。
  - `analyze_noise_samples()` に optional `engine_min_rms_margin_db` キーワード引数を追加 (default `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0`)。
- **新公開 API (`livecap_cli.audio`)**:
  - `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0` (定数)
  - `ENERGY_METRICS = ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms")`
  - `_segment_energy_dbfs(audio, sample_rate, metric, frame_ms) -> float` (helper、user-configurable metric/frame で per-segment energy を測定)
- **Telemetry**: `StreamTranscriber.close()` 時に drop counter の内訳 (`final_sync` / `final_async` / `interim`) を `logger.info` で 1 行サマリ。silent failure 防止。
- **Migration**:
  - 既存挙動を完全に維持したい場合: `StreamTranscriber(engine=..., engine_min_rms_dbfs=float("-inf"))` または CLI `--engine-min-rms off` で opt-out。
  - 通常はデフォルト (`-45.0`) で問題なく動作 (synthetic regression + 実音源プローブで通常会話・小声・ささやきレベル speech は pass を確認)。whisper 録音など特殊用途は閾値を下げる (`--engine-min-rms -50` 等) か opt-out。
- **検証**:
  - `tests/audio/test_analysis.py::TestSegmentEnergyDbfs` (10 cases) — 4 metric ごとの動作 / fallback / 物理的妥当性。
  - `tests/audio/test_energy_gate_regression.py` (新規 7 cases) — synthetic fixture (silent noise / speech-like burst / padded short utterance) で default threshold の drop/pass を assert。VAD padding 希釈に max_frame が耐性を持ち、whole_rms は希釈で false-drop することを documentation。
  - `tests/transcription/test_stream.py::TestEnergyGate` (10 cases) — 3 callsites (sync / async / interim) で `engine.transcribe()` が呼ばれない / 呼ばれることを mock の `call_count` で検証 + opt-out + invalid arg validation + close() log。
  - `tests/core/cli/test_cli.py::TestEnergyGateFlags` (12 cases) — `--engine-min-rms` の 4 parse パターン (numeric / off / disabled / =-inf) + invalid raises + metric choices + frame-ms parse + help text 可視性。
- **限界**:
  - EnergyGate は **silver bullet ではない**。実音源プローブで parakeet_ja は 73 silent windows で 100% hallucinate し、-45 dBFS threshold (max_frame_rms) で削減できるのは 26% のみ (transient を含む noisy silence は max frame で pass する)。完全防御には `--noise-gate` との併用 + VAD threshold 適切化が必要。
  - Engine choice が hallucination 耐性に大きく影響 (Parakeet 100% vs ReazonSpeech は known 低い)。
- **将来 follow-up**:
  - `BaseEngine` 側へのガード追加 (`StreamTranscriber` を経由しない advanced user 向け)
  - top-k metric の k=3 から user-configurable に拡張

#### **BREAKING** `NoiseAnalysis` / `analyze_noise_samples()` を peak-based calibration に置換 (Issue [#291])

- **Before**: `analyze_noise_samples(samples_db, sample_rate_hz)` →
  `suggested_threshold_db = noise_peak (chunk RMS p95) + 10 dB`
- **After**: `analyze_noise_samples(samples_db, peak_samples_db, sample_rate_hz)` →
  `suggested_threshold_db = peak_p95 (per-chunk |x|.max() p95) + 6 dB`
- **動機**: `NoiseGate` (`livecap_cli/audio/noise_gate.py`) の envelope follower
  は per-sample peak を追跡するが、calibration は chunk RMS を計測していた → unit
  mismatch により impulsive noise (キーボード/呼吸/breath bursts) で threshold が
  peak の下に潜り、無音時 hallucination ("あ"/"うん"/"ピッ") を引き起こしていた
  (Mega-Gorilla/livecap-gui#331 root-cause)。White noise の crest factor ≈ 11 dB
  が偶然 `+10` で吸収されていたが、impulsive noise では crest factor がより大きく
  破綻する。
- **API 変更点**:
  - `NoiseAnalysis` 新 field: `peak_p95_db: float` (per-chunk `|x|.max()` の 95%ile)
  - `NoiseAnalysis` 改名: `noise_peak_db` → `noise_rms_p95_db` (unit を field 名に明示)
  - `NoiseAnalysis` 削除: `safe_zone_min_db` (新 `suggested_threshold_db` と 1 dB 差で意味崩壊)
  - `analyze_noise_samples()` 新 required 引数: `peak_samples_db` (Optional=None の
    legacy default は無し: pre-1.0 backward-compat policy で旧バグ温存の flag は不可)
  - 新 module-level 定数: `livecap_cli.audio.PEAK_SAFETY_MARGIN_DB = 6.0`
  - `danger_zone` は据え置き (RMS-unit diagnostic として docstring で明記)
- **Migration**:
  ```python
  # 旧
  rms_db_list = [20*log10(rms(chunk)) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list)

  # 新
  rms_db_list  = [20*log10(rms(chunk))    for chunk in chunks]
  peak_db_list = [20*log10(|chunk|.max()) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list, peak_db_list)
  # a.noise_peak_db    -> a.noise_rms_p95_db
  # a.safe_zone_min_db -> (削除; suggested_threshold_db を直接使用)
  # a.peak_p95_db      -> 新 field; threshold の基準
  ```
  CLI `levels` は内部で per-chunk peak を収集するように移行済みのため、
  外部から CLI を呼ぶ場合の変更は不要 (JSON schema のみ変更)。
- **検証**: `tests/audio/test_noise_gate_calibration.py` (新規) で synthetic
  impulsive noise を旧 / 新 threshold それぞれで NoiseGate に通し、旧で gate
  が開く / 新で閉じ続けることを assert する end-to-end 回帰テストを追加。
- **GUI ペア PR**: livecap-gui 側は `core/noise_statistics.py` を削除し本 API
  に委譲する PR を別 issue (Mega-Gorilla/livecap-gui#335 の対) で受ける。
- **将来 follow-up**: NoiseGate の envelope follower を calibration 入力に対して
  simulate し envelope の 95%ile を取れば margin を 1-2 dB に縮められる ([#283]
  と組で別 issue 化予定)。

#### **BREAKING** `NoiseGate` デフォルト `release_ms` 変更 (Issue [#283] PR C)

- **Before** (PR #279 / PR #281 / PR #282): `release_ms=30`
- **After**: `release_ms=100`
- **動機**: PR #282 で導入された hard-mute と短い release の組み合わせが、aggressive な閾値 (-25/-17 dB) で whisper 系エンジンの fragmentation ハルシネーション (「んんん...」loop) を引き起こす。A/B 実測で `release_ms=100` により完全解消を確認 (316→102, 299→96 chars)
- **Migration**: 旧挙動を明示的に再現するには `release_ms=30` を直接渡す:
  ```python
  NoiseGate(release_ms=30)  # pre-PR-C default
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate --noise-gate-release 30 ...
  ```
- **検証結果**: `docs/benchmarks/noise-gate-ab.md` に更新後のテーブル掲載

#### **BREAKING** `NoiseGate` 既定挙動変更 (Issue [#280] PR B)

- **Before** (PR [#279] / PR [#281]): 単一閾値 + `-60 dB` soft-mute
- **After** (PR B): 自動ヒステリシス (`threshold_db - 6 dB`) + hard-mute (出力ゼロ)
- **動機**: PR #281 の A/B 検証で、PR #281 までの挙動が whisper 系エンジンで flicker によるハルシネーション暴走 ("どうもどうも..." 等) を引き起こすことが実証された
- **Migration**: 過去挙動を明示的に再現するには以下を指定:
  ```python
  NoiseGate(
      threshold_db=-35,
      close_threshold_db=-35,  # single-threshold (ヒステリシス無効)
      noise_floor_db=-60,      # soft-mute
  )
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate \
      --noise-gate-threshold -35 \
      --noise-gate-close-threshold -35 \
      --noise-gate-floor -60 \
      ...
  ```

新規オプション (既存呼び出しは無変更で動作、挙動のみ変化):

- `NoiseGate` / `transcribe` CLI に `close_threshold_db` / `--noise-gate-close-threshold` を追加 (ヒステリシス制御)
- `NoiseGate` / `transcribe` CLI に `noise_floor_db` / `--noise-gate-floor` を追加 (ゲート閉鎖時の減衰量制御)
- 初期化ログが resolved 値 (open/close/noise_floor) を出力するように改善 (ポリシー準拠)

既知の follow-up:

- `release_ms=30` は PR B の新しい gate 挙動 (hard-mute による clean silence) に対して短すぎるため、攻撃的な閾値で fragmentation hallucination が発生することがあります。`--noise-gate-release 100` または `200` で回避可能。デフォルト値の変更は別 issue で対応予定。

### Added

#### Noise Gate & Calibration ([#278], [#279], [#280], [#281])

- `livecap_cli.audio.NoiseGate` — 音量ベースのリアルタイムノイズゲート（サンプル単位エンベロープフォロワー、numba JIT で < 0.1 ms / 100 ms chunk）。VAD 前段に挿入してハルシネーションを抑制。
- `transcribe` サブコマンドに `--noise-gate` / `--noise-gate-threshold` / `--noise-gate-attack` / `--noise-gate-release` オプションを追加。
- `livecap-cli levels` サブコマンド — マイク入力レベルを dB 単位でリアルタイム表示し、環境ノイズから推奨閾値を算出。
  - `--duration N` — N 秒後に自動停止（非対話モード）。
  - `--json` — `NoiseAnalysis` を JSON で stdout に出力（GUI / スクリプト連携向け）。
- `livecap_cli.audio.analysis` モジュール — `NoiseAnalysis` dataclass と `analyze_noise_samples()` 関数（CLI / GUI 共通キャリブレーション API）。
- 推奨閾値アルゴリズム: `noise_peak (95%ile) + 10 dB`（[livecap-gui PR #294](https://github.com/Mega-Gorilla/livecap-gui/pull/294) の実測に基づく保守的マージン）。「死のゾーン」(`noise_floor ± 5 dB`) を回避する設計。

**段階導入について**: PR #281 は **キャリブレーション API 基盤の先行導入** (Issue #280 C-3 + C-4) です。NoiseGate 本体の安定化 ([Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) の C-1 ヒステリシス + C-2 hard-mute) は follow-up PR で提供予定。現行実装 (単一閾値 + `-60 dB` soft-mute) では、閾値が speech peak 付近の場合に flicker で逆にハルシネーションを誘発することがあります。特に `whispers2t` エンジンで影響が大きく、`reazonspeech` / `parakeet_ja` / `qwen3asr` は影響を受けにくいことが A/B テストで確認されています (PR #281 comments 参照)。暫定対応として、低 SNR 環境では `levels` の推奨値より保守的な値の使用、または別エンジンの利用を推奨します。

#### Phase 6: CLI Subcommand Structure ([#74], [#201])

New CLI with subcommand architecture:

| Command | Description |
|---------|-------------|
| `livecap-cli info` | Display installation diagnostics |
| `livecap-cli devices` | List audio input devices |
| `livecap-cli engines` | List available ASR engines |
| `livecap-cli translators` | List available translators |
| `livecap-cli transcribe` | Transcribe audio (file or realtime) |

**transcribe options:**
- `<file> -o <output.srt>` - File transcription to SRT
- `--realtime --mic <id>` - Realtime microphone transcription
- `--translate <id> --target-lang <lang>` - Translation support
- `--vad <auto|silero|tenvad|webrtc>` - VAD backend selection
- `--engine <id>` - ASR engine selection
- `--device <auto|gpu|cpu>` - Device selection

**Package extras:**
- `recommended`: Google translation (deep-translator)
- `all`: All optional dependencies

#### Phase 5: Engine Optimizations ([#73], [#194], [#196], [#197])

- Template Method pattern for `BaseEngine` with standardized lifecycle
- Progress reporting during model loading (0-100%)
- Model memory caching for faster subsequent loads
- Library preloading for reduced import time
- Standardized cleanup and resource management

#### Phase 4: Translation Support ([#72], [#180], [#181], [#182], [#184], [#186])

**Translators:**
- `google` - Google Translate ([#180])
- `opus_mt` - Helsinki-NLP Opus-MT local models ([#181])
- `riva_instruct` - NVIDIA Riva Translate 4B Instruct ([#182])

**Features:**
- Context-aware translation with sentence buffering
- `StreamTranscriber` translation integration ([#184])
- `FileTranscriptionPipeline` translation integration ([#186])
- Configurable timeout via `LIVECAP_TRANSLATION_TIMEOUT`
- Async translation deadlock prevention ([#189])

#### Phase 3: Package Structure ([#71])

- Reorganized module structure under `livecap_cli/`
- Clear separation: `engines/`, `vad/`, `transcription/`, `translation/`
- Unified public API exports in `__init__.py`

#### Phase 2: API Unification ([#70])

- `TranscriptionResult` dataclass replacing `TranscriptionEventDict`
- `VADConfig` dataclass for VAD parameters
- `EngineFactory.create_engine(engine_type, device, **options)` API
- Consistent error handling with `TranscriptionError`, `EngineError`

#### Phase 1: Realtime Transcription ([#69], [#65], [#66], [#67], [#68])

**Core components:**
- `StreamTranscriber` - VAD + ASR streaming orchestration ([#65])
- `VADProcessor` - Pluggable VAD with state machine ([#66])
- `TranscriptionResult` / `InterimResult` - Unified result types ([#67])
- `AudioSource` / `FileSource` / `MicrophoneSource` - Audio abstraction ([#68])

**VAD backends:**
- Silero VAD (default, neural network-based)
- WebRTC VAD (fast, low memory)
- TenVAD (optimized for Japanese)

**Language optimization:**
- `VADProcessor.from_language("ja")` - Auto-select optimal VAD
- Benchmark-based presets for Japanese and English

#### File Transcription

- `FileTranscriptionPipeline` for batch processing
- SRT subtitle output format
- FFmpeg integration for audio extraction
- Translation integration for bilingual subtitles

### Changed

#### Breaking Changes

| Before | After |
|--------|-------|
| Package: `livecap-core` | Package: `livecap-cli` |
| Module: `livecap_core` | Module: `livecap_cli` |
| CLI: `livecap-core --info` | CLI: `livecap-cli info` |
| CLI: `livecap-core --as-json` | CLI: `livecap-cli info --as-json` |

#### API Changes

- `TranscriptionEventDict` → `TranscriptionResult` dataclass
- Engine creation unified to `EngineFactory.create_engine()`
- VAD configuration via `VADConfig` instead of dict
- `detect_device()` returns `str` instead of `Tuple` ([#175])

### Deprecated

- `TranscriptionEventDict` (use `TranscriptionResult`)
- `languages.py` module (use `langcodes` for BCP-47) ([#173])

### Removed

- Old flag-based CLI interface (`--info`, `--ensure-ffmpeg`, `--as-json`)
- `livecap-core` entry point
- `livecap_core` module name
- `Languages.get_engines_for_language()` (use `EngineMetadata`) ([#171])

### Fixed

- GitHub Actions workflows updated for module rename ([#201])
- Integration test path filters updated
- Async translation deadlock in concurrent scenarios ([#189])
- Translation timeout handling improvements ([#187])
- OPUS-MT context disabled by default for stability ([#191])

### Security

- No security issues in this release

---

## Migration Guide

### From `livecap-core` to `livecap-cli`

#### 1. Update package installation

```bash
# Before
pip install livecap-core[engines-torch]

# After
pip install livecap-cli[engines-torch]

# Or use the recommended bundle:
pip install livecap-cli[recommended]
```

#### 2. Update imports

```python
# Before
from livecap_core import StreamTranscriber, EngineFactory
from livecap_core.vad import VADProcessor, VADConfig

# After
from livecap_cli import StreamTranscriber, EngineFactory
from livecap_cli.vad import VADProcessor, VADConfig
```

#### 3. Update CLI commands

```bash
# Before
livecap-core --info
livecap-core --as-json

# After
livecap-cli info
livecap-cli info --as-json

# New commands
livecap-cli devices
livecap-cli engines
livecap-cli translators
livecap-cli transcribe input.mp4 -o output.srt
livecap-cli transcribe --realtime --mic 0
```

#### 4. Update result handling (if using old dict API)

```python
# Before (TranscriptionEventDict)
result = {"text": "...", "start": 0.0, "end": 1.0}
print(result["text"])

# After (TranscriptionResult dataclass)
# result is now a dataclass with attributes
print(result.text)
print(result.start_time)
print(result.end_time)
print(result.to_srt_entry(index=1))
```

---

## Issue References

- Epic: [#64] - livecap-cli リファクタリング
- Phase 1: [#69] - リアルタイム文字起こし実装
- Phase 2: [#70] - API 統一と Config 簡素化
- Phase 3: [#71] - パッケージ構造整理
- Phase 4: [#72] - 翻訳機能実装
- Phase 5: [#73] - エンジン最適化
- Phase 6: [#74] - 依存関係整理・CLI・パッケージ名変更
- Docs: [#75] - ドキュメント更新

---

[Unreleased]: https://github.com/Mega-Gorilla/livecap-cli/compare/main...HEAD

[#64]: https://github.com/Mega-Gorilla/livecap-cli/issues/64
[#65]: https://github.com/Mega-Gorilla/livecap-cli/issues/65
[#66]: https://github.com/Mega-Gorilla/livecap-cli/issues/66
[#67]: https://github.com/Mega-Gorilla/livecap-cli/issues/67
[#68]: https://github.com/Mega-Gorilla/livecap-cli/issues/68
[#69]: https://github.com/Mega-Gorilla/livecap-cli/issues/69
[#70]: https://github.com/Mega-Gorilla/livecap-cli/issues/70
[#71]: https://github.com/Mega-Gorilla/livecap-cli/issues/71
[#72]: https://github.com/Mega-Gorilla/livecap-cli/issues/72
[#73]: https://github.com/Mega-Gorilla/livecap-cli/issues/73
[#74]: https://github.com/Mega-Gorilla/livecap-cli/issues/74
[#75]: https://github.com/Mega-Gorilla/livecap-cli/issues/75
[#171]: https://github.com/Mega-Gorilla/livecap-cli/pull/171
[#173]: https://github.com/Mega-Gorilla/livecap-cli/pull/173
[#175]: https://github.com/Mega-Gorilla/livecap-cli/pull/175
[#180]: https://github.com/Mega-Gorilla/livecap-cli/pull/180
[#181]: https://github.com/Mega-Gorilla/livecap-cli/pull/181
[#182]: https://github.com/Mega-Gorilla/livecap-cli/pull/182
[#184]: https://github.com/Mega-Gorilla/livecap-cli/pull/184
[#186]: https://github.com/Mega-Gorilla/livecap-cli/pull/186
[#187]: https://github.com/Mega-Gorilla/livecap-cli/pull/187
[#189]: https://github.com/Mega-Gorilla/livecap-cli/pull/189
[#191]: https://github.com/Mega-Gorilla/livecap-cli/pull/191
[#194]: https://github.com/Mega-Gorilla/livecap-cli/pull/194
[#196]: https://github.com/Mega-Gorilla/livecap-cli/pull/196
[#197]: https://github.com/Mega-Gorilla/livecap-cli/pull/197
[#201]: https://github.com/Mega-Gorilla/livecap-cli/pull/201
[#278]: https://github.com/Mega-Gorilla/livecap-cli/issues/278
[#279]: https://github.com/Mega-Gorilla/livecap-cli/pull/279
[#280]: https://github.com/Mega-Gorilla/livecap-cli/issues/280
[#281]: https://github.com/Mega-Gorilla/livecap-cli/pull/281
[#283]: https://github.com/Mega-Gorilla/livecap-cli/issues/283
[#291]: https://github.com/Mega-Gorilla/livecap-cli/issues/291
[#292]: https://github.com/Mega-Gorilla/livecap-cli/issues/292
[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
