# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-18

Initial release as `livecap-cli` (formerly `livecap-core`).

### Added

#### CLI Subcommand Structure (Phase 6)
- `livecap-cli info` - Display installation diagnostics
- `livecap-cli devices` - List audio input devices
- `livecap-cli engines` - List available ASR engines
- `livecap-cli translators` - List available translators
- `livecap-cli transcribe` - Transcribe audio from file or microphone
  - File transcription: `livecap-cli transcribe <file> -o <output.srt>`
  - Realtime transcription: `livecap-cli transcribe --realtime --mic <id>`
  - Translation support: `--translate <translator_id> --target-lang <lang>`
  - VAD backend selection: `--vad <auto|silero|tenvad|webrtc>`

#### Package Extras
- `recommended` extra: Includes `engines-torch` and `translation` for common use cases
- `all` extra: Includes all optional dependencies

#### Translation Support (Phase 4)
- Google Translate integration (`translation` extra)
- Opus-MT local translation (`translation-local` extra)
- NVIDIA Riva translation (`translation-riva` extra)
- Context-aware translation with sentence buffering
- Translation timeout configuration via `LIVECAP_TRANSLATION_TIMEOUT`

#### Engine Optimizations (Phase 5)
- Template Method pattern for `BaseEngine`
- Progress reporting during model loading
- Standardized cleanup and resource management
- Model memory caching for faster subsequent loads

#### Realtime Transcription (Phase 1)
- `StreamTranscriber` for VAD-based streaming transcription
- `VADProcessor` with pluggable backends (Silero, WebRTC, TenVAD)
- `MicrophoneSource` for live audio capture
- `FileSource` for file-based testing
- Language-optimized VAD presets via `VADProcessor.from_language()`

#### File Transcription
- `FileTranscriptionPipeline` for batch processing
- SRT subtitle output format
- Translation integration for bilingual subtitles

### Changed

#### Breaking Changes
- Package renamed from `livecap-core` to `livecap-cli`
- Module renamed from `livecap_core` to `livecap_cli`
- Entry point changed from `livecap-core` to `livecap-cli`
- Old CLI flags removed (`--info`, `--ensure-ffmpeg`, `--as-json`)
  - Use `livecap-cli info` and `livecap-cli info --as-json` instead

#### API Changes
- `TranscriptionEventDict` deprecated in favor of `TranscriptionResult` dataclass
- Engine creation via `EngineFactory.create_engine(engine_type, device, **options)`
- VAD configuration via `VADConfig` dataclass

### Deprecated

- None

### Removed

- Old flag-based CLI interface
- `livecap-core` entry point
- `livecap_core` module name

### Fixed

- GitHub Actions workflows updated for new module name
- Integration test path filters updated

### Security

- None

---

## Migration Guide

### From `livecap-core` to `livecap-cli`

1. **Update imports:**
   ```python
   # Before
   from livecap_core import StreamTranscriber, EngineFactory

   # After
   from livecap_cli import StreamTranscriber, EngineFactory
   ```

2. **Update CLI commands:**
   ```bash
   # Before
   livecap-core --info

   # After
   livecap-cli info
   ```

3. **Update installation:**
   ```bash
   # Before
   pip install livecap-core[engines-torch]

   # After
   pip install livecap-cli[engines-torch]
   # Or use the recommended bundle:
   pip install livecap-cli[recommended]
   ```

---

[Unreleased]: https://github.com/Mega-Gorilla/livecap-cli/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mega-Gorilla/livecap-cli/releases/tag/v0.1.0
