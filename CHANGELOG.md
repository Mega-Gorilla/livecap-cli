# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Changed

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
