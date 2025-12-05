# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

livecap-core is a high-performance speech transcription library providing real-time transcription, file processing, and multi-ASR engine support. It supports 16 languages including Japanese, English, Chinese, and Korean.

## Build & Development Commands

```bash
# Environment setup (uv preferred)
uv sync --extra engines-torch --extra dev

# Without uv
pip install -e ".[engines-torch,dev]"

# Run tests
uv run pytest tests                           # Full suite
uv run pytest tests/core                      # Unit tests only
uv run pytest tests/integration               # Integration tests (requires FFmpeg)
uv run pytest tests/integration/engines -m engine_smoke  # Engine smoke tests

# CLI validation
uv run livecap-core --info                    # Show installation diagnostics
uv run livecap-core --as-json                 # JSON output
```

**FFmpeg for integration tests**: Place `ffmpeg`/`ffprobe` in `./ffmpeg-bin/` or set `LIVECAP_FFMPEG_BIN`.

## Architecture

### Core Pipeline Flow

```
AudioSource (mic/file) → VADProcessor → StreamTranscriber → TranscriptionResult
                              ↓
                      VADSegment (speech chunks)
                              ↓
                      ASR Engine (transcribe)
```

### Module Structure

- **`livecap_core/`**: Public API surface
  - `transcription/stream.py`: `StreamTranscriber` - VAD + ASR orchestration
  - `audio_sources/`: `AudioSource`, `FileSource`, `MicrophoneSource`
  - `vad/`: VAD processor with pluggable backends (`backends/silero.py`)
  - `transcription/file_pipeline.py`: Batch file transcription to SRT

- **`livecap_core/engines/`**: ASR engine adapters implementing `BaseEngine`
  - `base_engine.py`: Abstract base with Template Method pattern
  - `engine_factory.py`: `EngineFactory.create_engine(engine_type, device, **engine_options)`
  - `metadata.py`: Engine registry with `EngineMetadata.default_params` for each engine
  - `whispers2t_engine.py`: Has `use_vad` parameter to disable built-in VAD

### Key Interfaces

```python
# StreamTranscriber usage
from livecap_core import StreamTranscriber, MicrophoneSource, EngineFactory

engine = EngineFactory.create_engine("whispers2t_base", device="cuda")
engine.load_model()

with StreamTranscriber(engine=engine) as transcriber:
    with MicrophoneSource() as mic:
        for result in transcriber.transcribe_sync(mic):
            print(f"[{result.start_time:.2f}s] {result.text}")

# TranscriptionEngine protocol (engines must implement)
class TranscriptionEngine(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float]: ...
    def get_required_sample_rate(self) -> int: ...
    def get_engine_name(self) -> str: ...
    def cleanup(self) -> None: ...
```

### Configuration

- Engine defaults defined in `livecap_core/engines/metadata.py` via `EngineMetadata.default_params`
- Engine options passed via `**kwargs` to `EngineFactory.create_engine()`
- VAD config via `VADConfig(threshold, min_speech_ms, min_silence_ms, speech_pad_ms)`

## Testing Markers

```bash
pytest -m engine_smoke    # Real audio + engine tests
pytest -m gpu            # CUDA-required tests
pytest -m realtime_e2e   # Requires LIVECAP_ENABLE_REALTIME_E2E=1
```

## CI Notes

- Self-hosted runners (Windows RTX 4090, Linux) used for GPU tests
- Non-ephemeral runners: avoid `sudo`, use portable installations
- `.github/actions/setup-livecap-ffmpeg`: Custom FFmpeg setup action
- `uv.lock` is source of truth for dependencies

## Coding Conventions

- Type hints required: `from __future__ import annotations`
- PEP 8, 4-space indents, dataclasses for structured data
- Update `__all__` exports when adding public APIs
- Commit prefixes: `feat:`, `fix:`, `chore:`, `ci:`, `docs:`
