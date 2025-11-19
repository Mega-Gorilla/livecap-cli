# Repository Guidelines

## Project Structure & Module Organization
- `livecap_core/` hosts the runtime pipeline: `transcription/` orchestrates streaming flows, `resources/` wraps FFmpeg and model management, and `config/` exposes defaults consumed by the CLI.
- `engines/` contains engine adapters (Whisper, ReazonSpeech, Parakeet, etc.) that implement `base_engine.py` and share tooling via `shared_engine_manager.py`.
- Top-level `config/` provides higher-level builders used by entrypoints; mirror updates here and under `livecap_core/config`.
- `tests/` mirrors runtime modules (`tests/core`, `tests/transcription`) with pytest suites; extend alongside new features.
- `docs/` stores architecture and strategy notes—consult when modifying pipeline boundaries or licensing touchpoints.

## Build, Test, and Development Commands
- `uv sync --extra translation --extra dev` creates `.venv` with runtime, engine, and dev dependencies (CI mirrors this step).
- `uv run livecap-core --dump-config` validates the CLI script and prints generated defaults; add `--as-json` for machine output.
- `uv run pytest tests` executes the full unit suite; target subsets (`pytest tests/core`) during rapid iterations.
- Without `uv`, `python -m venv .venv && source .venv/bin/activate` followed by `pip install -e .[dev,translation]` reproduces the environment.

## Coding Style & Naming Conventions
- Stick to PEP 8 with 4-space indents; keep modules typed (`from __future__ import annotations`) and prefer dataclasses for structured payloads.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and refresh `__all__` exports whenever public APIs change.
- Keep configuration constants under `config/` namespaces and document engine-specific options near their adapters.

## Testing Guidelines
- Pytest is the canonical framework; name files `test_*.py` and co-locate fixtures beside the target module (`tests/core`, `tests/transcription`).
- Add regression coverage for new engines by stubbing resource managers rather than hitting network downloads.
- Update CLI diagnostics expectations in `tests/core/test_cli.py` whenever configuration fields or JSON output changes.

## Commit & Pull Request Guidelines
- Follow the existing conventional prefixes (`feat:`, `fix:`, `chore:`, `ci:`) with an imperative summary; keep commits scoped to one concern.
- Reference impacted modules in the body and call out compatibility or migration steps for engine consumers.
- PRs should summarize intent, list verification steps (`uv run pytest …`, CLI snapshots), link issues/docs, and request runtime maintainers when touching `engines/` or shared resources.
