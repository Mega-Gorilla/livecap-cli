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

## Integration Tests

Integration tests live under `tests/integration/` and download FFmpeg binaries
or speech models. Opt in explicitly to avoid surprise network traffic:

```bash
export LIVECAP_ENABLE_INTEGRATION=true
uv sync --extra translation --extra dev --extra engines-torch
uv run python -m pytest tests/integration
```

Unset the variable (or leave it absent) to keep integration tests skipped. Use
this flow before releasing binaries or when validating hardware/engine combos.
The same `LIVECAP_ENABLE_INTEGRATION` flag is referenced by README and the
integration workflow, so keep the name consistent if it ever changes.

## CI Mapping

- `Core Tests` workflow: runs `pytest tests` on Python 3.10/3.11/3.12 with `translation`+`dev` extras.
- `Optional Extras` job: validates `engines-torch` / `engines-nemo` installs.
- `Integration Tests` workflow: manual or scheduled opt-in that sets
  `LIVECAP_ENABLE_INTEGRATION=true`.

Keep this document updated whenever the workflows or extras change so local
developers can reproduce CI faithfully.
