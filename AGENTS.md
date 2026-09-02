# Repository Guidelines

## Project Structure & Module Organization
- `livecap_cli/` hosts the runtime pipeline: `transcription/` orchestrates streaming flows, `resources/` wraps FFmpeg and model management, and `vad/` provides voice activity detection.
- `livecap_cli/engines/` contains engine adapters (Whisper, ReazonSpeech, Parakeet, etc.) that implement `base_engine.py` and share tooling via `model_memory_cache.py` (VRAM-aware model cache), `library_preloader.py` (background dependency warm-up), and `nemo_utils.py` (NeMo runtime setup).
- `livecap_cli/engines/metadata.py` defines `EngineMetadata.default_params` as the single source of truth for engine defaults.
- **新規 engine 追加 guide**: `docs/contributor/adding-an-engine.md` (Issue #334 audit findings から codify した anti-patterns 含む、Quickstart 10-step + signal family decision tree + threshold calibration template)。
- `scripts/` hosts developer utilities. `scripts/benchmarks/` holds reproducible evaluation harnesses (A/B comparisons, perf probes) that are not run in CI — invoke them manually when validating behavior changes.
- `tests/` mirrors runtime modules (`tests/core`, `tests/transcription`) with pytest suites; extend alongside new features.
- `docs/` stores architecture and strategy notes. `docs/benchmarks/` holds empirical evaluation summaries paired with `scripts/benchmarks/` harnesses (raw data is regenerable and excluded from the repo).

## Build, Test, and Development Commands
- `uv sync --extra translation --extra dev` creates `.venv` with runtime, engine, and dev dependencies (CI mirrors this step).
- `uv run livecap-core --info` shows installation diagnostics (FFmpeg, CUDA, VAD backends, ASR engines); add `--as-json` for machine output.
- `uv run pytest tests` executes the full unit suite; target subsets (`pytest tests/core`) during rapid iterations.
- Without `uv`, `python -m venv .venv && source .venv/bin/activate` followed by `pip install -e .[dev,translation]` reproduces the environment.

## Coding Style & Naming Conventions
- Stick to PEP 8 with 4-space indents; keep modules typed (`from __future__ import annotations`) and prefer dataclasses for structured payloads.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and refresh `__all__` exports whenever public APIs change.
- Document engine-specific options in `livecap_cli/engines/metadata.py` via `EngineMetadata.default_params`.

## Testing Guidelines
- Pytest is the canonical framework; name files `test_*.py` and co-locate fixtures beside the target module (`tests/core`, `tests/transcription`).
- Add regression coverage for new engines by stubbing resource managers rather than hitting network downloads.
- Update CLI diagnostics expectations in `tests/core/test_cli.py` whenever configuration fields or JSON output changes.

## Commit & Pull Request Guidelines
- Follow the existing conventional prefixes (`feat:`, `fix:`, `chore:`, `ci:`) with an imperative summary; keep commits scoped to one concern.
- Reference impacted modules in the body and call out compatibility or migration steps for engine consumers.
- PRs should summarize intent, list verification steps (`uv run pytest …`, CLI snapshots), link issues/docs, and request runtime maintainers when touching `livecap_cli/engines/` or shared resources.

## Backward Compatibility Policy (pre-1.0)
`livecap-cli` is currently versioned `1.0.0.dev0`. Its only known consumer is the sibling project `livecap-gui`, which is developed in lockstep. Until we ship `1.0.0`, breaking internal behavior in service of correctness is acceptable — **preserving buggy defaults as "backward compatibility" is not**.

When you change a default, rename a parameter, or adjust observable behavior:
1. Document the change under `CHANGELOG.md` → `## [Unreleased]` → **the section matching the nature of the change** (see "CHANGELOG sections" below) with a concrete **Before / After / Migration** note. Observable behavior changes require Before / After / Migration **regardless of which section they land in**.
2. Update any affected `docs/` page (especially `docs/reference/cli.md` and `docs/reference/api.md`).
3. Keep the initialization / diagnostic logs informative enough that an affected user can see which mode is active (e.g., log the resolved threshold values, not just the raw args).
4. Do not add `Optional[T] = None` "legacy mode" flags whose only purpose is to preserve pre-existing bugs. If a caller genuinely wants the old behavior, they can pass the old value explicitly (e.g., `close_threshold_db=threshold_db`, `noise_floor_db=-60`) — surface it as opt-in, not opt-out.

Re-evaluate this policy before the first tagged `1.0.0` release.

## CHANGELOG sections

`CHANGELOG.md` → `## [Unreleased]` uses exactly these H3 sections, **in this order**. Each name appears **at most once** — `tests/core/test_changelog_structure.py` fails otherwise.

| Section | What belongs there |
|---|---|
| `Added` | New features / new APIs |
| `Changed` | Changes to existing behavior, structure, or defaults |
| `Deprecated` | Scheduled for removal |
| `Removed` | Deleted features, APIs, or dead code |
| `Fixed` | Bug fixes |
| `Documentation` | Docs-only changes with no runtime effect |
| `Security` | Security fixes |

**This format is Keep a Changelog plus three local extensions**, not stock Keep a Changelog: the `Documentation` section, the H4 detail blocks inside each section, and the summary table at the top of `[Unreleased]`. Stock Keep a Changelog defines six categories (no `Documentation`) and Common Changelog defines four, filing docs-only work under the functional categories. We keep `Documentation` because this CHANGELOG preserves investigation results and design decisions as well as user-facing changes, and forcing docs-only work into `Added` / `Changed` loses that distinction (#436).

**Choose the section by the primary change as a user sees it — the H4 heading.** Do **not** decide by counting `- **Type**:` bullets inside the entry: a `Fixed` entry's bullets describe *what was added or changed in order to fix it*, so counting moves correctly-placed bug fixes out of `Fixed` (measured on 5 existing entries, #436).

The summary table at the top of `[Unreleased]` is organized by **what changed for a user**, not by epic — epic and issue links belong in it as pointers to detail, not as its axis. Avoid entry counts there; they go stale.

**The test only checks structure** (a single `[Unreleased]` as the first H2; duplicate / unknown / empty sections; ordering; unclosed code fences; that parsing stops at the next H2; and that every `[#N]` reference has a link definition). It cannot tell that a `Removed`-heavy entry was filed under `Added` — that is what this table and code review are for.

### Writing an entry

Each entry is an H4 under one of the sections above. Write it for a **reader upgrading the package**, and link out for everything else:

```markdown
#### What changed, from the user's point of view ([#123])

- **Before**: ...
- **After**: ...
- **Migration**: ... (or "none")
- **Details**: `docs/research/...`
```

**No line limit is imposed** — physical line counts shift with wrapping and link placement, so they make a poor rule. The structure above is the rule (#438).

Where each kind of information belongs:

| | |
|---|---|
| `CHANGELOG.md` | What changed for a user, and what they must do about it |
| `docs/research/` | Measurements, evidence, and design decisions |
| Issue / PR | How the decision was reached; review history; mutation results |
| Migration guide | Step-by-step migration procedures |

**Docs-only entries are exempt from Before / After / Migration.** Under `### Documentation`, forcing that shape produces filler; a short summary plus a pointer is enough.

**Everything else is not exempt**: a change to observable behavior needs Before / After / Migration **whatever section it lands in** — a `Fixed` entry that changes a default still needs them.

Reference links (`[#123]`) are defined at the bottom of the file under `## Issue References`. **GitHub does not autolink `#123` inside repository files**, so an undefined reference renders as literal text — the test catches that.
