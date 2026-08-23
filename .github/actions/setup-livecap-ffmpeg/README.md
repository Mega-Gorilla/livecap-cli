# setup-livecap-ffmpeg

Installs a pinned, checksum-verified `ffmpeg` / `ffprobe` pair for CI (Issue #395).

```yaml
- name: Setup LiveCap FFmpeg
  uses: ./.github/actions/setup-livecap-ffmpeg
  with:
    ffmpeg-bin-dir: 'ffmpeg-bin'
    cache: 'true'          # 'false' on self-hosted runners with a persistent dir
```

## How it decides what to do

Verification is an **invariant, not a step**. Bytes can reach `ffmpeg-bin-dir` from
three places — an `actions/cache` restore, a self-hosted persistent directory, or a
fresh download — and all three go through the same check: **exists / SHA-256 matches
/ runs / reports the pinned version**. Only if that fails does anything download.

Do **not** add a `Check FFmpeg existence` gate in a workflow. That pattern used to
exist in five workflows and skipped this action entirely, which skipped verification
with it — so a stale binary in `C:\LiveCap\Cache\ffmpeg-bin` was never noticed.
The skip decision belongs here, after verification.

## Caching

The cache key is `ffmpeg-<version>-<platform>-<manifest sha256 prefix>-g<generation>`
and there are **no `restore-keys`**. A broad restore-key resurrects a cache built for
an older pinned version, and the action's own "already installed" branch then keeps
it alive forever — pinning silently stops working.

If a run warns about a **poisoned cache**, the exact key restored bytes that failed
verification. `actions/cache` entries are immutable, so the fix is not to delete it:
bump `cache_generation` in `ffmpeg-manifest.json`. Until then, every run re-downloads
(and stays green).

`cache: 'false'` disables restore/save only. Verification still runs, which is how a
stale binary in a persistent directory gets replaced.

## Pinning is not behavioural equivalence

The Linux and Windows builds have different `configuration:` lines and different
enabled codecs. Pinning makes runs **comparable**; it does not make them behave
identically.

## SHA-256 refresh

Never hand-type a hash. To move to a new version:

1. Point `base_url` / `release_tag` / `version` at the new release and update the four
   `archives[*].asset` names. ffbinaries names assets
   `<tool>-<version>-<platform>.zip` — the version is part of the name, and the
   platform tokens are `win-64`, `linux-64`, `macos-64` (**not** `windows-64` / `osx-64`;
   that mistake is what makes the runtime downloader 404 — see #398).
2. Regenerate all eight hashes from the archives themselves:

   ```bash
   python - <<'PY'
   import hashlib, urllib.request, zipfile, io
   BASE = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1"
   for asset in ("ffmpeg-6.1-linux-64.zip", "ffprobe-6.1-linux-64.zip",
                 "ffmpeg-6.1-win-64.zip", "ffprobe-6.1-win-64.zip"):
       blob = urllib.request.urlopen(f"{BASE}/{asset}").read()
       print(f"{asset}\n  archive {hashlib.sha256(blob).hexdigest()}")
       with zipfile.ZipFile(io.BytesIO(blob)) as z:
           for info in z.infolist():
               member = z.read(info)
               print(f"  binary  {info.filename}: {hashlib.sha256(member).hexdigest()}")
   PY
   ```

3. Run it **twice** and confirm the values reproduce before committing.
4. Bump `cache_generation`.
5. `uv run pytest tests/ci -q -m network` to confirm every pinned URL resolves.
6. Note the change in `CHANGELOG.md` — CI now tests a different FFmpeg.
