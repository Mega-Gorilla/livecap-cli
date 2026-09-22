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
| **flattened dir + manifest** | `<models_root>/<org>--<name>/` に repo の必要ファイルと `livecap-manifest.json`。同じ repo の variant を分けるときだけ `<org>--<name>-<variant>` (ReazonSpeech int8 = `reazon-research--reazonspeech-k2-v2-int8`) | Qwen3-ASR / WhisperS2T / Voxtral / ReazonSpeech / Riva |
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
| `files[]` | 相対 path (空でない正規化済み POSIX、`.` / `..` / 絶対 / drive / UNC / backslash は parse 時に拒否)、size、etag |

**cache hit は `validate_repo_dir()` だけで決まる**: manifest があり、`repo_id` / `variant` が一致し、呼び出し側が**今**要求する `required` が manifest に記録され通常ファイルとして実在し、`files[]` の全てが存在してサイズ一致し、**全 entry の実体 (`resolve()`) が dir の中にある** (最終要素の symlink だけでなく親 dir の symlink や `..` も拒否)。「非空 dir」は hit ではない。hash 照合は既定では行わない (起動コスト)。load に失敗した engine は `invalidate_manifest()` で manifest を **`files: []` + `source: invalidated` に書き換え**、次回 miss → 再取得 (self-heal)。消すのではなく書き換えるのは、消すと `adopt_dir()` が同じ壊れた内容を「manifest の無い完全な dir」として再採用してしまうため。manifest があるのに invalid な dir は `adopt_dir()` の対象外で、次の取得時に `publish_dir()` が `<name>.invalid-<ts>` へ隔離する。隔離された dir は `livecap-cli info` の `Legacy model layouts` に出る (削除は利用者の判断)。

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

## 7. engine 側の実装

flattened dir が正本の engine (Qwen3-ASR / WhisperS2T / Voxtral / ReazonSpeech) は `livecap_cli/engines/repo_dir_engine.py` の `RepoDirModelMixin` を `BaseEngine` の前に継承し、`_repo_dir_spec()` で `RepoDirSpec` (repo_id / required / variant / allow_patterns / ignore_patterns / legacy_subdirs) を返す。cache hit (`validate_repo_dir` + required)、旧配置の取り込み (`migrate_dir`)、取得 (`fetch_repo_dir`)、fail loud (`_require_model_dir`)、self-heal (`_invalidate_model_dir`) はすべて mixin の 1 実装で、engine 側では override しない (`tests/core/engines/test_repo_dir_engine.py`)。単一 `.nemo` の engine (Parakeet / Canary) は `download_file` + `migrate_nemo_file(validate=)` を使う。

翻訳 translator (`livecap_cli/translation/impl/`) は `BaseTranslator.model_manager` で同じ root を見る。Riva は `migrate_dir` (adopt) → `fetch_repo_dir` で `<models_root>/nvidia--Riva-Translate-4B-Instruct/` へ、OPUS-MT は変換元 repo を staging (`<cache_root>/downloads/opus-mt-source/`) に取って (root の外の既定 HF cache に 0.2.0 までの変換元 snapshot が残っていれば `migrate_dir` で copy、無ければ `fetch_repo_dir`) `TransformersConverter` で変換し、tokenizer を `save_pretrained` で同梱してから `publish_dir` で `<models_root>/opus-mt/<org>--<name>/` へ (変換元は消す)。#456 以前の変換済み dir (tokenizer 無し) は tokenizer だけ取って adopt する。`ctranslate2.Translator` / `AutoTokenizer` / `AutoModelForCausalLM` にはローカル dir だけを渡す (repo id は既定 HF cache へ行く)。

## 8. migration (旧配置からの取り込み)

`<cache_root>` / `<models_root>` の**中**にある旧配置 (0.1.0 / 0.2.0 の HF hub 階層、Voxtral の transformers cache、engine subdir の重複、canary の nested `.nemo`) は、cold load 時に自動で正本へ取り込む: snapshot から必要ファイルを **symlink を dereference して**実体化 → manifest (`source: migrated`) → validate → publish → **成功して検証を通った後にだけ**旧側を削除。root の**外** (既定 HF cache `huggingface_hub.constants.HF_HUB_CACHE` = `~/.cache/huggingface/hub`、whisper_s2t の自前 cache `platformdirs.user_cache_dir("whisper_s2t")/models`) の hub snapshot も候補にする (#453) — ただし root の中の候補の**後**に並べ、取り込みは hardlink / copy だけで**削除は絶対にしない** (他アプリと共用)。`livecap-cli info` の `External model caches` が cli が使う repo (`legacy_model_layouts.KNOWN_MODEL_REPOS`) の残骸を `adopted` 付きで列挙する。`adopted` は **その repo から作られる正本 (`KnownRepo.destinations`。ReazonSpeech は float32 と int8 の 2 つ) が全部 `models_root` にある** = livecap-cli としては外の copy が要らない、までを意味し、**共有 cache としての削除可否 (他アプリの利用、hardlink による解放量) は含まない**。既定 HF cache が root の中に解決する環境 (`HF_HOME` を `cache_root` に向けている) では root の中の旧配置として扱う (二重に列挙しない)。これは §6 の「既定 HF cache への silent fallback」ではない: 外から読むのは取り込みの入力としてだけで、ロードは常に `models_root` の正本から行い、取り込みはログに出る。

単一ファイル (`.nemo`) も同じ契約 (`legacy_model_layouts.migrate_nemo_file`): engine の validator (`_verify_model_integrity`) を通る候補だけを正本にし、配置後にもう一度 validate してから旧側を削除する。validator を通らない root (truncated) や nested `.nemo/` dir、同名ファイルの無い `.nemo/` dir は `<name>.nemo.invalid-<ts>` へ隔離する (download が publish できる形にする)。 validator を通らなかった旧候補は、別の候補が採用された後も**消さない** (残骸として `livecap-cli info` に出る。隔離された file / dir も同様に列挙する)。nested dir の un-nest に失敗したときは退避 dir を元の名前へ戻す。ReazonSpeech の旧 int8 tarball (`<cache_root>/downloads/*.tar.bz2`) は int8 の正本が validator を通った後にだけ削除する。

取り込みと取得は **destination 単位の同じ lock** (`model_store.model_lock`: `<cache_root>/downloads/<destination 名>.lock`) を共有する。2 process が同時に cold load しても、旧配置の rename / delete と download / publish が競合しない。

`<cache_root>/huggingface/**/models--*/` のうち `snapshots/` / `blobs/` にファイルが無いもの (新方式の `snapshot_download(local_dir=, cache_dir=)` が `cache_dir` 側に残す `refs/main` だけの metadata) は**許可された transient** であり、旧配置として列挙も取り込みもしない。

## 関連

- #456 (本契約) / #428 / #430 / #447 (既定 HF cache を止めた) / #453 (root 外の旧 cache) / #375 (root の設定と readback)
- livecap-gui `docs/architecture/resource-policy-and-cleanup.md` — GUI 側は `<install>\data\models` / `<install>\data\cache` を同じ意味論で扱う
