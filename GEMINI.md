# GEMINI.md

This file serves as the primary context and instruction set for AI agents (Gemini) interacting with the `livecap-core` repository.

## Project Overview

`livecap-core` is the standalone runtime engine for the LiveCap project, handling real-time and file-based speech transcription. It abstracts various speech-to-text engines (Whisper, ReazonSpeech, NVIDIA NeMo, etc.) into a unified pipeline.

**Key Technologies:**
- **Python**: 3.10 – 3.12
- **Dependency Management**: `uv` (preferred)
- **Core Libraries**: `ffmpeg-python`, `numpy`, `torch` (optional), `sherpa-onnx`
- **Testing**: `pytest`

## Repository Structure

- **`livecap_core/`**: Main runtime logic.
    - `transcription/`: Streaming and file pipelines.
    - `resources/`: Management of external binaries (FFmpeg) and models.
    - `config/`: Configuration schemas and validation.
    - `cli.py`: Entry point.
- **`engines/`**: Adapter layer for different ASR engines.
    - `base_engine.py`: The interface all engines must implement.
- **`config/`**: Top-level configuration builders.
- **`tests/`**: Mirrors the source structure.
    - `tests/integration`: End-to-end tests requiring real audio processing and FFmpeg.
- **`.github/`**: CI/CD configurations.
    - `actions/setup-livecap-ffmpeg`: Custom action to setup portable FFmpeg.

## Development Workflow

### 1. Environment Setup

The project uses `uv` for fast dependency resolution.

```bash
# Install dependencies including dev tools and translation extras
uv sync --extra translation --extra dev

# Install with heavy engine support (Torch/NeMo)
uv sync --extra translation --extra dev --extra engines-torch
```

### 2. Running the CLI

```bash
# Dump default configuration
uv run livecap-core --dump-config

# (Future) Run transcription
# uv run livecap-core transcribe input.wav
```

### 3. Testing

Tests are split into unit and integration suites.

```bash
# Run full suite (default)
uv run pytest tests

# Run specific subset
uv run pytest tests/core
uv run pytest tests/integration
```

**FFmpeg Setup for Tests:**
Integration tests (`tests/integration`) require a working FFmpeg binary.
- **CI**: Handled automatically by `.github/actions/setup-livecap-ffmpeg`.
- **Local**:
    1. Download generic/static FFmpeg binaries (e.g., from [ffbinaries](https://github.com/ffbinaries/ffbinaries-prebuilt/releases) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)).
    2. Place `ffmpeg` and `ffprobe` executables in a `./ffmpeg-bin/` directory in the project root.
    3. Run tests (the test harness automatically detects `./ffmpeg-bin`).
    4. Alternatively, set `LIVECAP_FFMPEG_BIN` environment variable to the directory containing the binaries.

## Architecture & Conventions

- **Type Hinting**: Strictly enforced. Use `from __future__ import annotations`.
- **Configuration**: All configuration is data-driven, defined in `livecap_core/config/schema.py` using `dataclasses`.
- **Engine Loading**: Engines are loaded lazily via `engines/engine_factory.py`. New engines should be registered there.
- **Platform Support**:
    - Primary: Linux (Ubuntu)
    - Secondary: Windows (Support actively being improved, see `.github/workflows/core-tests-windows.yml`)
    - **Self-Hosted Runners**: Used for GPU-dependent tests. CI flows are designed to be runner-agnostic where possible (avoiding `sudo`, `choco`, or `shell: bash` strict dependencies).

## Critical Context for AI Agents

- **Self-Hosted CI**: When modifying CI workflows, remember that self-hosted runners (especially Windows) might be non-ephemeral and lack admin privileges. Use portable tool installation strategies (like the `setup-livecap-ffmpeg` action) rather than system package managers.
- **Windows Paths**: Always handle path separators carefully. Python's `pathlib` is preferred over string manipulation.
- **Lockfile**: `uv.lock` is the source of truth for dependencies. Update it via `uv sync` or `uv lock` when changing `pyproject.toml`.
