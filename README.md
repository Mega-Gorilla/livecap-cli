# LiveCap Core

[日本語 READMEはこちら](README.ja.md)

LiveCap Core is the standalone runtime used by the LiveCap GUI and headless
deployments. It ships the streaming transcription pipeline, engine adapters
(Whisper/ReazonSpeech/NeMo…), resource managers, and configuration helpers that
other projects embed.

## What's inside?

- `livecap_core/config`: default config builders, schema validation, and
  helpers for CLI/GUI integration.
- `livecap_core/transcription`: streaming/file transcription pipelines plus
  event normalization utilities.
- `livecap_core/engines`: adapters for Whisper, ReazonSpeech, Parakeet, etc.
  Optional extras extend these engines with heavier dependencies.

> ℹ️ **Project status** – We are still extracting modules from the monolithic
> LiveCap repository. Public APIs may change until the first 1.0.0 release
> candidate.

## Requirements

- Python **3.10 – 3.12** (match `pyproject.toml`).
- POSIX-like environment (Linux/macOS). Windows support is planned and tracked
  in the Live_Cap_v3 repository.
- [uv](https://github.com/astral-sh/uv) (recommended) or pip for dependency
  management.
- `sherpa-onnx>=1.12.15` ships with the base install so the ReazonSpeech engine
  works out of the box (same floor version as Live_Cap_v3).

## Installation

```bash
# Clone the repo
git clone https://github.com/Mega-Gorilla/livecap-core
cd livecap-core

# Recommended: uv
uv sync --extra translation --extra dev

# Optional: traditional pip/venv
python -m venv .venv && source .venv/bin/activate
pip install -e .[translation,dev]
```

### Optional extras

| Extra name       | Installs …                             | When to use                                    |
| ---------------- | -------------------------------------- | ---------------------------------------------- |
| `engines-torch`  | `reazonspeech-k2-asr`, `torch`, `torchaudio`, `torchvision` (ensures ReazonSpeech stack including sherpa-onnx) | ReazonSpeech / Torch-based engines             |
| `engines-nemo`   | `nemo-toolkit`, `hydra-core`, etc.     | NVIDIA NeMo engine support                     |
| `translation`    | `deep-translator`                      | Translation pipeline extras                    |
| `dev`            | `pytest`, tooling for contributors     | Running tests and lint locally                 |

Install with `uv sync --extra engines-torch` or
`pip install "livecap-core[engines-torch]"`.

## Usage

### CLI diagnostics

```bash
uv run livecap-core --dump-config > default-config.json
```

### Programmatic example

```python
from livecap_core import FileTranscriptionPipeline, normalize_to_event_dict
from livecap_core.config.defaults import get_default_config

config = get_default_config()
pipeline = FileTranscriptionPipeline(config=config)

# Minimal stub transcriber (replace with a real engine such as SharedEngineManager)
def stub_transcriber(audio_data, sample_rate):
    # Return a list of subtitle events (start, end, text)
    return [(0.0, 1.2, "Hello world")]

result = pipeline.process_file(
    file_path="sample.wav",
    segment_transcriber=stub_transcriber,
    write_subtitles=False,
)

# Normalise an event dictionary for downstream consumers
event = normalize_to_event_dict({"text": "Hello", "offset": 0.0, "duration": 1.2})
print("Transcription success:", result.success, "sample text:", event["text"])
```

This prints the normalized transcription event that downstream consumers expect.

## Testing

```bash
# Run all tests (matches CI)
uv sync --extra translation --extra dev
uv run python -m pytest tests

# Engine adapters (add the extra you want to validate)
uv sync --extra translation --extra dev --extra engines-torch
uv run python -m pytest tests/core/engines

# Integration tests (opt-in: downloads FFmpeg/models)
LIVECAP_ENABLE_INTEGRATION=true uv run python -m pytest tests/integration
```

See [`docs/dev-docs/testing/README.md`](docs/dev-docs/testing/README.md) for the
full test matrix, optional extras, and troubleshooting tips.

## Documentation & Further Reading

- Pre-release workflow:
  [`docs/dev-docs/releases/pre-release-tag-workflow.md`](docs/dev-docs/releases/pre-release-tag-workflow.md)
- Testing guide:
  [`docs/dev-docs/testing/README.md`](docs/dev-docs/testing/README.md)
- Architecture notes and roadmap:
  [`docs/dev-docs/architecture/livecap-core-extraction.md`](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/docs/dev-docs/architecture/livecap-core-extraction.md)
- Licensing and compliance checklist:
  [`docs/dev-docs/licensing/README.md`](docs/dev-docs/licensing/README.md)

## License & Contact

LiveCap Core ships under the GNU Affero General Public License v3.0 (AGPL-3.0).
The full text is provided in `LICENSE`. Questions about permissible usage or
collaboration can be shared via the
[LiveCap Discord community](https://discord.gg/hdSV4hJR8Y). Security or
commercial inquiries are listed in `LICENSE-COMMERCIAL.md`.
