# ModelRoot 契約 — 永続モデル資産の置き場所 (Issue #456)

**この文書は「どのモデル資産がどこに置かれ、何が一時データか」の SSoT です。** 実装は
`livecap_cli/engines/model_store.py` (manifest / validate / publish) と
`livecap_cli/engines/hf_cache.py` (`fetch_repo_dir` / `download_file`)、契約を固定するテストは
`tests/core/engines/test_model_store.py` / `test_hf_cache.py` / `test_model_store_contract.py` と
各 engine の `test_*_managed_cache.py`。食い違ったら実装が正で、本書を直す。

## 1. 2 つの root

| root | 役割 | 消してよいか |
|---|---|---|
| **`models_root`** (`configure_resources(models_dir=)` / `LIVECAP_CORE_MODELS_DIR`) | 推論を再起動するために必要な永続資産の**唯一の正本**。重み・tokenizer・processor・vocabulary・SentencePiece・変換済みモデル・manifest | 消すと再ダウンロードになる |
| **`cache_root`** (`configure_resources(cache_dir=)` / `LIVECAP_CORE_CACHE_DIR`) | staging (`downloads/<repo>/{download,payload}`)、lock、`.incomplete`、HTTP metadata、展開途中、publish 用 temp、ffmpeg / runtime の一時ファイル | いつ消してもモデルは失われない (途中の download が最初からになるだけ) |

standalone (設定なし) の既定は `appdirs.user_cache_dir("LiveCap", "PineLab")` 配下 (`%LOCALAPPDATA%\PineLab\LiveCap\Cache\{models,cache}`、Linux は `~/.cache/LiveCap/...`) で、**ユーザープロファイル配下である**。本契約が禁止するのは「設定した root を無視した別 root への保存」であり、既定 root の場所ではない。`livecap-cli info` の `Models root` / `Cache root` / `HF cache` が実効値。

## 2. 保存方式

| 方式 | 形 | engine |
|---|---|---|
| **flattened dir + manifest** | `<models_root>/<org>--<name>/` に repo の必要ファイルと `livecap-manifest.json` | Qwen3-ASR / WhisperS2T / Voxtral / ReazonSpeech / Riva |
| **single file** | `<models_root>/<org>--<name>.nemo` | Parakeet / Parakeet JA / Canary |
| **CTranslate2 dir + manifest** | `<models_root>/opus-mt/<org>--<name>/` に CT2 model + tokenizer + manifest | OPUS-MT |

flattened dir の取得は `hf_cache.fetch_repo_dir()`:

```
<cache_root>/downloads/<org>--<name>/
  download/    snapshot_download(local_dir=ここ, cache_dir=<cache_root>/huggingface/hub, allow/ignore_patterns, max_workers=1)
               huggingface_hub は local_dir 直下に .cache/huggingface/download/*.metadata / .lock / *.incomplete を作る → ここに閉じる
  payload/     download/ から必要ファイルだけを rename で集め、manifest を書く
<models_root>/<org>--<name>/   payload/ だけを publish_dir() で原子的に配置 (同一 volume の temp → os.replace)
```

- `cache_dir=<cache_root>/huggingface/hub` は `local_dir` モードでも lookup / lock に使われるので**明示する** (省略すると既定 `HF_HUB_CACHE` から silent fallback する)。ここは transient で、正本は置かれない
- `HF_HUB_OFFLINE=1` で staging に完了済みファイルが無ければ `LocalEntryNotFoundError` (既定 cache は見ない)
- repo 単位の `FileLock` で download → publish → cleanup を直列化。後続は destination が valid なら取得を skip
- 成功したら staging を消す。**失敗時は残す** (resume)。destination はどの段階で失敗しても作られない

## 3. manifest (`livecap-manifest.json`、`schema_version` 1)

| key | 内容 |
|---|---|
| `repo_id` / `revision` / `commit_sha` | どの snapshot か (`commit_sha` / `etag` は HF の `.metadata` から。無ければ `null`) |
| `variant` | engine 固有の識別 (WhisperS2T の size、ReazonSpeech の int8 / float32、OPUS-MT の quantization) |
| `source` | `download` / `adopted` (既存 dir をその場で採用) / `migrated` (旧配置から実体化) |
| `files[]` | 相対 path、size、etag |

**cache hit は `validate_repo_dir()` だけで決まる**: manifest があり、`repo_id` / `variant` が一致し、`files[]` の全てが存在してサイズ一致し、symlink が dir の外を指していない。「非空 dir」は hit ではない。hash 照合は既定では行わない (起動コスト)。load に失敗した engine は `invalidate_manifest()` で manifest を **`files: []` + `source: invalidated` に書き換え**、次回 miss → 再取得 (self-heal)。消すのではなく書き換えるのは、消すと `adopt_dir()` が同じ壊れた内容を「manifest の無い完全な dir」として再採用してしまうため。manifest があるのに invalid な dir は `adopt_dir()` の対象外で、次の取得時に `publish_dir()` が `<name>.invalid-<ts>` へ隔離する。隔離された dir は `livecap-cli info` の `Legacy model layouts` に出る (削除は利用者の判断)。

## 4. publish (`publish_dir`)

1. destination が valid → skip
2. destination が存在するが invalid → 同じ dir 内の `<name>.invalid-<ts>` へ rename して**隔離** (削除しない)
3. payload を destination と同じ volume の sibling temp `.<name>.<uuid>.part/` へ (同一 volume は rename、別 volume は copytree)
4. temp 上で validate → `os.replace(temp, destination)` (destination はこの時点で無いので Windows でも原子的)
5. 失敗: temp を消す (rename 済みなら payload へ戻す)、隔離した旧 destination を元に戻す

## 5. 明示例外 (契約の対象外)

| 資産 | 所在 | 理由 |
|---|---|---|
| Silero VAD | `site-packages/silero_vad/data/*.onnx`, `*.jit` | wheel 同梱の install asset。`importlib.resources` で読む。**runtime の書き込み無し**。`load_silero_vad()` に path を渡す口が無く、複製すると正本が 2 つになるだけ |
| TenVAD | `site-packages/ten_vad_library/ten_vad.dll` | wheel 同梱の native library。**runtime の書き込み無し** |
| PyTorch CUDA Jiterator kernel cache (#422 / #425)、FFmpeg binary (`<cache_root>/ffmpeg`) | — | model asset ではない |
| Google Translate / WebRTC VAD | — | ローカル重みなし |

この表は `model_store.MODEL_STORE_EXEMPT_ASSETS` と `tests/core/engines/test_model_store_contract.py` で固定する。

## 6. 禁止事項

- repo ID を `from_pretrained()` / converter / `load_model()` へ直接渡し、上流既定 cache に保存させる
- 完成済みの同一 weight を `cache_root` と `models_root` に二重保持する
- 必要ファイル以外を取得して残す (repo 全体の snapshot、archive)
- import 後の `HF_HOME` 書き換えに依存する (`huggingface_hub` は import 時に cache path を確定する)
- 設定した `models_root` にモデルが無いとき、既定 HF cache へ silent fallback する
- `models_root` に `.cache/huggingface/`、`.locks/`、`*.lock`、`*.incomplete`、`*.metadata`、`*.part` を残す

## 7. migration (旧配置からの取り込み)

`<cache_root>` / `<models_root>` の**中**にある旧配置 (0.1.0 / 0.2.0 の HF hub 階層、Voxtral の transformers cache、engine subdir の重複、canary の nested `.nemo`) は、cold load 時に自動で正本へ取り込む: snapshot から必要ファイルを **symlink を dereference して**実体化 → manifest (`source: migrated`) → validate → publish → **成功して検証を通った後にだけ**旧側を削除。root の**外** (`~/.cache/huggingface/hub`、`%LOCALAPPDATA%\whisper_s2t`) は #453 の範囲で、削除はしない。

## 関連

- #456 (本契約) / #428 / #430 / #447 (既定 HF cache を止めた) / #453 (root 外の旧 cache) / #375 (root の設定と readback)
- livecap-gui `docs/architecture/resource-policy-and-cleanup.md` — GUI 側は `<install>\data\models` / `<install>\data\cache` を同じ意味論で扱う
