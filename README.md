# LiveCap Core

Core runtime components for LiveCap, extracted into a standalone Python package.
This repository will host the streaming transcription pipeline, engine adapters,
and shared utilities that power both the LiveCap GUI and headless deployments.

## Licensing

LiveCap Core ships under the GNU Affero General Public License v3.0 (AGPL-3.0).
The full text is provided in `LICENSE`, and any derivative work that is
distributed or offered as a network service must comply with the AGPL terms.
Questions about permissible usage or collaboration can be shared via the
[LiveCap Discord community](https://discord.gg/hdSV4hJR8Y).

## Status

> ⚠️ **Early extraction phase**  
> The codebase is migrating from the monolithic LiveCap repository. API surface,
> packaging metadata, and CI workflows are still being finalized. Expect rapid
> iteration until the first 1.0.0 release candidate.

## Roadmap (condensed)

1. Bootstrap packaging (`pyproject.toml`, `uv.lock`, minimal CI).
2. Migrate `livecap_core/` runtime modules and accompanying tests.
3. Publish pre-release artifacts to TestPyPI (`1.0.0a*`, `1.0.0b*`, `1.0.0rc*`).
4. Coordinate 1.0.0 GA with the LiveCap GUI repository.

For the full migration plan, refer to
[`docs/dev-docs/architecture/livecap-core-extraction.md`](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/docs/dev-docs/architecture/livecap-core-extraction.md).

## Getting Involved

- Issues & feature requests: use the tracker in this repository once it opens
  for public contributions.
- Security or usage inquiries: see `LICENSE-COMMERCIAL.md` for Discord contact
  details.

Stay tuned for contributor guidelines and API documentation as the split
progresses.
