# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

### Added

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
