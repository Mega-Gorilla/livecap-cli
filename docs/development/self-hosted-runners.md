# Self-Hosted Runners 運用 runbook

GPU 必須テスト (engine_smoke、transcription-pipeline) を回す self-hosted GitHub Actions runner の運用手順をまとめる。

## なぜ self-hosted か

GitHub-hosted runner では CUDA GPU が使えないため、以下を self-hosted で実行している:

- `engine-smoke-gpu`: Canary / Parakeet / Voxtral / Whisper 等 GPU engine の transcribe 動作確認 (`tests/integration/engines/test_smoke_engines.py`)
  - **PR #323 (Issue #321 PR #2) で merge gate 化**: `test_token_confidence_populated` が NeMo fallback chain 削除後の silent degradation を検出
- `transcription-pipeline` の self-hosted matrix entry: `FileTranscriptionPipeline` の実 GPU pipeline 動作確認

該当 workflow: [`.github/workflows/integration-tests.yml`](../../.github/workflows/integration-tests.yml)

## 現状のセットアップ

| Runner | OS | GPU | 役割 |
|---|---|---|---|
| `windows self host runner` | Windows | NVIDIA GeForce RTX 4090 (24GB) | `engine-smoke-gpu (self-hosted, windows)` + `transcription-pipeline (self-hosted, windows)` |
| Linux runner | (未設定または expired) | RTX 4090 | (将来必要なら) `engine-smoke-gpu (self-hosted, linux)` + `transcription-pipeline (self-hosted, linux)` |

Runner repo 変数:

- `LIVECAP_ENABLE_GPU_SMOKE` = `"1"` (`engine-smoke-gpu` job の `if` 条件、設定済)
- `LIVECAP_REQUIRE_ENGINE_SMOKE` = `"1"` (skip ではなく failure として扱う)

## 症状 — Runner registration が消えている

GitHub は **long offline (~14 日以上) の runner を自動 unregister** する。再起動しても以下のエラーで listen 開始できなくなる:

```
√ Connected to GitHub
Failed to create a session. The runner registration has been deleted
from the server, please re-configure. Runner registrations are
automatically deleted for runners that have not connected to the
service recently.
"Runner listener exit with terminated error, stop the service, no retry needed."
"Exiting runner..."
```

CI 上では以下のように見える:

- PR の `engine-smoke-gpu (self-hosted, linux/windows)` が **永久 pending** のまま
- 新 push のたびに `cancelled` になる (concurrency cancel)
- `gh api repos/Mega-Gorilla/livecap-cli/actions/runners` で **`status: "offline"`** または entry なし

## 復旧手順 (Windows runner)

### 0. 確認 — registration の状態を見る

```pwsh
gh api repos/Mega-Gorilla/livecap-cli/actions/runners
```

`"status": "offline"` で `version: null` なら、registration は server 側で削除されている (= 復旧手順が必要)。`"status": "online"` なら起動するだけで OK。

### 1. 旧 registration を削除 (config に残骸があれば)

```pwsh
cd C:\actions-runner

# 既存 config がある場合、まず remove (失敗しても無視可)
.\config.cmd remove --token <REMOVAL_TOKEN>
```

`REMOVAL_TOKEN` は以下で取得 (admin 権限の PAT 必要):

```pwsh
gh api -X POST repos/Mega-Gorilla/livecap-cli/actions/runners/remove-token
```

または GitHub UI: `Settings → Actions → Runners → 該当 runner → ⋯ → Remove` (UI 経由は自動)。

### 2. 新規 registration token を取得

GitHub UI:

`https://github.com/Mega-Gorilla/livecap-cli/settings/actions/runners/new`

→ "Configure" セクションの `.\config.cmd --url ... --token <TOKEN>` をコピー。token は **1 時間有効**。

CLI でも可 (admin 権限の PAT 必要):

```pwsh
gh api -X POST repos/Mega-Gorilla/livecap-cli/actions/runners/registration-token
```

### 3. Configure + 起動

```pwsh
cd C:\actions-runner

# Configure (token は上記で取得した値、name と labels はお好み)
.\config.cmd --url https://github.com/Mega-Gorilla/livecap-cli `
             --token <REGISTRATION_TOKEN> `
             --name "windows self host runner" `
             --labels self-hosted,X64,Windows `
             --work _work `
             --unattended

# Interactive 起動 (foreground、Ctrl+C で停止)
.\run.cmd
```

長時間運用するなら **Windows service 化** がおすすめ:

```pwsh
# Service として install
.\svc install

# 起動
.\svc start

# 状態確認
.\svc status

# 停止 / 削除
.\svc stop
.\svc uninstall
```

Windows service にすると logon 不要・OS 起動時に auto start。

### 4. CI 上で reflection を確認

新 PR を push するか既存 PR を re-run、`engine-smoke-gpu (self-hosted, windows)` が pending → running → pass に推移すれば成功。

## Linux runner を追加する場合

Linux runner も同様の手順だが、以下が異なる:

- インストーラ: [actions-runner-linux-x64](https://github.com/actions/runner/releases)
- Service 化: `sudo ./svc.sh install <USER>` → `sudo ./svc.sh start`
- labels: `self-hosted,X64,Linux`

Workflow 側 (`integration-tests.yml`) の matrix は両方を含む形で記述されているため、Linux runner が registered であれば自動で job が pickup される。

## Timeout (PR #324 [Issue #321 follow-up] で導入)

`integration-tests.yml` の self-hosted job には `timeout-minutes: 60` を設定。

- runner offline で running 状態に到達できない場合: GitHub の queue 経由で eventually cancel される (workflow run cancel policy 依存)
- runner online で test が hang した場合: 60 min で job-level hard fail、明確な error signal を CI に残す

cold model cache の場合 engine_smoke は 20-30 min かかるため、60 min は十分な margin (実機 verify では 47.90s で完了)。

## 監視 / アラート (提案、未実装)

- `gh api .../actions/runners` を polling して `status: "offline"` を Slack/Discord に通知する scheduled workflow
- 別の workflow `.github/workflows/verify-self-hosted-windows.yml` を定期実行 (`schedule: cron`) して runner health check

これらは本 issue scope 外、必要なら別 PR で対応。

## Troubleshooting

### `Configure persistent paths` step で permission denied

Windows runner が UAC 環境下にある場合、`C:\LiveCap\Cache\...` への write permission が無いことがある。runner を **administrator として起動** するか、cache root を user-writable な path (例: `%USERPROFILE%\LiveCap\Cache`) に変更する。

### `Failed to copy ffmpeg-bin` (Linux)

runner ホストに `ffmpeg` / `ffprobe` が system install されていれば自動 detect → copy される。`apt-get install ffmpeg` または手動でバイナリを置く (`~/.local/bin/ffmpeg` 等)。

### `Insufficient VRAM` で skip される

`tests/integration/engines/test_smoke_engines.py` の `_guard_gpu` が VRAM 要件 (e.g. Voxtral は 16GB) を check。RTX 4090 (24GB) なら全 engine OK。

## 関連

- 該当 workflow: [`integration-tests.yml`](../../.github/workflows/integration-tests.yml)
- Runner verify workflow:
  - [`verify-self-hosted-windows.yml`](../../.github/workflows/verify-self-hosted-windows.yml)
  - [`verify-self-hosted-linux.yml`](../../.github/workflows/verify-self-hosted-linux.yml)
- Issue #321 PR #2 ([#323](https://github.com/Mega-Gorilla/livecap-cli/pull/323)) — `engine-smoke-gpu` の merge gate 化、本 doc の動機
