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

Add new tests beside the module they validate. Put scenarios that hit real
artifacts or external binaries under `tests/integration/`. These suites now run
as part of `pytest tests`, so keep them deterministic and use explicit flags
only when absolutely necessary (for example, the MKV fixtures from Issue #21).

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
default `pytest tests` invocation. Most scenarios still rely on a stub
`FFmpegManager`, but Issue #21 adds a lightweight MKV-based regression test
that exercises the real FFmpeg extraction path.

To run the MKV extraction test, you need real `ffmpeg/ffprobe` executables
available. Download an FFmpeg build (for example from
[ffbinaries-prebuilt](https://github.com/ffbinaries/ffbinaries-prebuilt/releases)),
place `ffmpeg`/`ffprobe` under `./ffmpeg-bin/`, and set
`LIVECAP_FFMPEG_BIN="$PWD/ffmpeg-bin"` (PowerShell:
`$env:LIVECAP_FFMPEG_BIN="$(Get-Location)\ffmpeg-bin"`).

The `ffmpeg-bin/` directory is ignored by git so contributors can supply their
own binaries without committing them.

## CI Mapping

- `Core Tests` workflow: runs `pytest tests` (integration tests included) on Python 3.10/3.11/3.12 with `translation`+`dev` extras. The workflow prepares a `./ffmpeg-bin/` directory and exposes it via `LIVECAP_FFMPEG_BIN` so the MKV regression test can resolve FFmpeg offline using the system binaries available on the runner.
- `Optional Extras` job: validates `engines-torch` / `engines-nemo` installs.
- `Integration Tests` workflow: manual or scheduled opt-in that runs the same
  suite with additional extras/models when required and also prepares
  `ffmpeg-bin` for MKV coverage.
- `Windows Debug Tests` workflow: manual, Windows-only variant of the core test
  suite that mirrors `pytest tests` on `windows-latest` with `translation`+`dev`
  extras and a `ffmpeg-bin` directory configured via `LIVECAP_FFMPEG_BIN`.
- `Verify Self-Hosted Linux Runner` workflow: manual verification workflow that
  validates self-hosted Linux runner environment setup (Python, uv, FFmpeg)
  by downloading portable binaries.
- `Verify Self-Hosted Windows Runner` workflow: manual verification workflow that
  validates self-hosted Windows runner environment setup using portable FFmpeg
  from gyan.dev.

### Runner inventory (current state)

| Workflow | Runner | FFmpeg provisioning |
| --- | --- | --- |
| `Core Tests` / `Integration Tests` | GitHub-hosted `ubuntu-latest` | `apt-get install ffmpeg`, binaries copied into `./ffmpeg-bin/` for deterministic paths |
| `Windows Debug Tests` | GitHub-hosted `windows-latest` | `choco install -y ffmpeg`, then copy the real binaries from `$env:ChocolateyInstall\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe` / `ffprobe.exe` into `.\ffmpeg-bin\` so `ffmpeg-python` resolves the correct executable |

For local Windows debugging we currently mirror the workflow by extracting the
official 6.1 builds from [ffbinaries-prebuilt](
https://github.com/ffbinaries/ffbinaries-prebuilt/releases) into `ffmpeg-bin/`
and pointing `LIVECAP_FFMPEG_BIN` at that directory. Self-hosted runners
(Linux/Windows + GPU) will be documented in a follow-up once those workflows
land.

Keep this document updated whenever the workflows or extras change so local
developers can reproduce CI faithfully.
