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

起動方式: **ログオン時スケジュールタスク** (`GitHub Actions Runner (livecap-cli)`)。
管理者権限が無く Windows service 化できないための代替 (手順は [3-b](#3-b-代替-ログオン時自動起動-管理者権限が不要))。

**Linux runner は運用していない**。未登録 runner 宛の job は実行されないまま
**24 時間のキュー滞留上限で cancel** され、PR check が恒久的に赤くなるため、
`integration-tests.yml` の matrix から `[self-hosted, linux]` を除外している
(`timeout-minutes` は job 開始後のみ計測されるため、この滞留は防げない)。
再導入する場合は「[Linux runner を追加する場合](#linux-runner-を追加する場合)」を参照。

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

- PR の `engine-smoke-gpu (self-hosted, windows)` が **pending** のまま進まず、
  **24 時間後に `cancelled` (fail 表示)** になる — GitHub のキュー滞留上限。
  `timeout-minutes` は job 開始後のみ計測するためこの経路では効かない
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

長時間運用するなら **Windows service 化** がおすすめ (logon 不要・OS 起動時に auto start)。

> **重要**: Windows には `svc.cmd` / `svc install` は**存在しない**。service 化は
> **`config.cmd` の構成処理そのもので行う**。runner パッケージ
> (`actions-runner-win-x64-*.zip`) に同梱される service script は
> `bin/systemd.svc.sh.template` (Linux) と `bin/darwin.svc.sh.template` (macOS)
> のみで、Windows 用の script は生成されない (service 本体は `bin\RunnerService.exe`)。
> service 化には**管理者権限が必須**。取得できない場合は
> [3-b](#3-b-代替-ログオン時自動起動-管理者権限が不要) を使う。

**初回構成時に service 化する** (管理者 PowerShell):

```pwsh
cd C:\actions-runner

.\config.cmd --url https://github.com/Mega-Gorilla/livecap-cli `
             --token <REGISTRATION_TOKEN> `
             --name "windows self host runner" `
             --labels self-hosted,X64,Windows `
             --work _work `
             --unattended `
             --runasservice
```

**すでに service なしで構成済みの場合**は、一度 remove してから service 指定で
再構成する (§1-2 の手順で新しい token を取得):

```pwsh
.\config.cmd remove --token <REMOVAL_TOKEN>   # server 側 registration が既に消えていれば remove --local
.\config.cmd --url ... --runasservice ...      # 上記と同じ
```

構成後の管理は **Windows Services** または PowerShell の service コマンドで行う
(service 名は `actions.runner.<owner>-<repo>.<runner 名>`):

```pwsh
Get-Service actions.runner.*                    # 状態確認 / 正式名の確認
Start-Service actions.runner.*
Stop-Service  actions.runner.*
```

### 3-b. 代替: ログオン時自動起動 (管理者権限が不要)

service 化できない環境では、**ログオン時に起動するスケジュールタスク**で
永続化できる。

**service との差**: トリガーが `-AtLogOn` のため、**OS 再起動後は対象ユーザーが
ログオンするまで runner は起動しない** (service は logon 不要)。したがって
「再起動後にそのユーザーが日常的にログオンする」運用であれば runner は自動再接続し、
**14 日 offline による registration 自動削除を防げる**が、再起動後 14 日以上
ログオンしない運用では削除条件が成立し得る。常時稼働が要るなら service 化する。

```pwsh
$taskName = "GitHub Actions Runner (livecap-cli)"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument '-NoProfile -WindowStyle Hidden -Command "Set-Location C:\actions-runner; .\run.cmd"'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Force

# 即時起動 (次回ログオンを待たない)
Start-ScheduledTask -TaskName $taskName
```

運用コマンド:

```pwsh
# 状態確認 (LastTaskResult 267009 = 0x41301 "実行中" が正常)
Get-ScheduledTask -TaskName "GitHub Actions Runner (livecap-cli)" | Get-ScheduledTaskInfo

# 停止 / 削除
Stop-ScheduledTask   -TaskName "GitHub Actions Runner (livecap-cli)"
Unregister-ScheduledTask -TaskName "GitHub Actions Runner (livecap-cli)" -Confirm:$false
```

`ExecutionTimeLimit` を `[TimeSpan]::Zero` にしないと既定 3 日で kill される点に注意。

### 4. CI 上で reflection を確認

新 PR を push するか既存 PR を re-run、`engine-smoke-gpu (self-hosted, windows)` が pending → running → pass に推移すれば成功。

## Linux runner を追加する場合

Linux runner も同様の手順だが、以下が異なる:

- インストーラ: [actions-runner-linux-x64](https://github.com/actions/runner/releases)
- Service 化: `sudo ./svc.sh install <USER>` → `sudo ./svc.sh start`
- labels: `self-hosted,X64,Linux`

**Workflow 側の matrix 追加が必要**: 現在 `integration-tests.yml` の
`transcription-pipeline` / `engine-smoke-gpu` はいずれも
`[self-hosted, windows]` のみを列挙しているため、Linux runner を registered に
しただけでは job は生成されない。matrix に `[self-hosted, linux]` を戻し、
Linux 用 step (`Set up Python (Linux)` / `Sync dependencies (Linux)` /
`Warm engine caches` / `Run engine smoke tests` の Linux 版) を復活させること
(削除時の内容は git 履歴を参照)。

## Timeout (PR #324 [Issue #321 follow-up] で導入)

`integration-tests.yml` の self-hosted job には `timeout-minutes: 60` を設定。

- runner online で test が hang した場合: 60 min で job-level hard fail、明確な error signal を CI に残す
- **runner offline で pickup されない場合: `timeout-minutes` は効かない** — job timeout は
  runner が job を pickup してから計測されるため、queue 滞留中はカウントが始まらない。
  この場合は GitHub の**キュー滞留 24 時間上限**で cancel される (= PR check が
  丸 1 日 pending のまま、その後 fail 表示)。runner 側の復旧が唯一の対処

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
- Issue #321 PR #2 ([#323](https://github.com/Mega-Gorilla/livecap-cli/pull/323)) — `engine-smoke-gpu` の merge gate 化、本 doc の動機
