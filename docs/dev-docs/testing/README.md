# Testing LiveCap Core

This guide explains how the repository's test suites are organized, how to
install the right extras for each scope, and when to run integration scenarios
that require network access or large downloads.

## Directory Layout

| Path | Scope |
| --- | --- |
| `tests/core/cli` | CLI entrypoints, configuration dumps, human I/O |
| `tests/core/config` | Shared config builders and defaults |
| `tests/core/engines` | Engine factory wiring and adapter registration |
| `tests/core/i18n` | Translation tables and locale fallbacks |
| `tests/core/resources` | Resource managers (FFmpeg, model cache, uv profiles) |
| `tests/transcription` | Pure transcription helper/unit tests (legacy path kept for Live_Cap_v3 compatibility) |
| `tests/integration/transcription` | End-to-end pipelines that touch audio, disk, or model downloads |

Add new tests beside the module they validate. If a test requires real model
artifacts or external binaries, move it under `tests/integration/` and guard it
with `pytest.mark.skipif` so CI stays green.

## Dependency Profiles

`pyproject.toml` exposes extras that toggle optional engines and tooling:

| Extra | Description |
| --- | --- |
| `translation` | Language packs and text processing dependencies |
| `dev` | Pytest, typing, linting utilities |
| `engines-torch` | Torch-based engines such as Whisper or ReazonSpeech |
| `engines-nemo` | NVIDIA NeMo engines such as Parakeet or Canary |

Most day-to-day development uses `translation` + `dev`. Add engine extras when
you need to exercise specific adapters.

## Running the Test Suites

Clone the repo, then install dependencies using uv:

```bash
uv sync --extra translation --extra dev
```

Run the default suite (matching the CI workflow):

```bash
uv run python -m pytest tests
```

Targeted executions:

```bash
# Only CLI/config tests
uv run python -m pytest tests/core/cli tests/core/config

# Engine wiring (requires corresponding extras)
uv sync --extra translation --extra dev --extra engines-torch
uv run python -m pytest tests/core/engines
```

## Integration Tests & FFmpeg setup

Integration tests live under `tests/integration/` and now run as part of the
default `pytest tests` invocation. To keep the suite offline-friendly, prepare a
local FFmpeg build and point `LIVECAP_FFMPEG_BIN` at it:

```bash
mkdir -p ffmpeg-bin
# Download ffmpeg/ffprobe from e.g. https://github.com/ffbinaries/ffbinaries-prebuilt/releases
# and copy the binaries into ./ffmpeg-bin/
export LIVECAP_FFMPEG_BIN="$PWD/ffmpeg-bin"           # Linux/macOS
# PowerShell:
# $env:LIVECAP_FFMPEG_BIN = "$(Get-Location)\ffmpeg-bin"

uv sync --extra translation --extra dev
uv run python -m pytest tests
```

CI copies the system ffmpeg and ffprobe into the same directory before running
tests so we avoid runtime downloads. When adding new integration suites (e.g.
requiring optional extras or models), make sure they continue to respect the
`ffmpeg-bin` contract and keep network access to explicit workflows only.

## CI Mapping

- `Core Tests` workflow: runs `pytest tests` (integration tests included) on Python 3.10/3.11/3.12 with `translation`+`dev` extras and a prepared `ffmpeg-bin/`.
- `Optional Extras` job: validates `engines-torch` / `engines-nemo` installs.
- `Integration Tests` workflow: manual or scheduled opt-in that runs the same
  suite with additional extras/models when required.

Keep this document updated whenever the workflows or extras change so local
developers can reproduce CI faithfully.
