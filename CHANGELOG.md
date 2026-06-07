# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Added

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
