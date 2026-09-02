# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Epic #64 (livecap-cli refactoring) - completion of all 6 phases.

This represents the completion of a major refactoring effort spanning 6 phases.
Package renamed from `livecap-core` to `livecap-cli`.

**この Unreleased の概要** — 詳細は下の各節にある。**ここは入口であって、内容の要約ではない。**
エントリは **変更の性質** で分類してある (epic 単位ではない) ので、1 つの取り組みは複数の節にまたがる。

| 取り組み | 主な内容 | 節 |
|---|---|---|
| **非 ASCII パス耐性** (epic [#380] / 14 エントリ) | ネイティブライブラリのパス契約を **境界の棚卸し + runtime 実測**で確定した。sherpa-onnx / NeMo / PyTorch Jiterator の **黙って壊れる** 経路を特定し、対策と再評価条件を記録している ([#378] / [#375] / [#377] / [#379] / [#413] / [#387] / [#422]) | Added / Changed / Removed / Fixed |
| **confidence の較正** ([#338] / 8 エントリ、[#308] / [#334] / [#351]) | engine ごとの confidence signal の意味論を監査し、閾値較正ハーネスと corpus を整備した | Added / Changed / Documentation |
| **resource / モデル管理** ([#375] / [#409] / [#386]) | `configure_resources()` と共有 resource graph、ASCII path 保証 API、cache identity | Added / Changed / Fixed |
| **公開 API の整理** ([#363] / [#365] / [#366] / [#286]) | SRT serializer、言語解決 metadata、VAD 分割 adapter、engine の推薦 API | Added / Changed |
| **realtime / 翻訳の不具合** ([#402] / [#403] / [#407] / [#418]) | 黙って原文になる翻訳、無視される `--translate`、呼ばれない `cleanup()` | Fixed |
| **`livecap-core` からの改名** | パッケージ名 / entry point / CLI の刷新 | Changed / Removed / Fixed |

> **節の使い分けは `AGENTS.md` に定義がある。** 迷ったら **利用者から見た主要な変更**で決めること — bullet の種類を数えて決めない ([#436])。

### Added

#### load 境界 2 行を ②wide-path で確定 — **複合境界を分割してから測った** (Issue [#387] PR B、epic [#380])

**実測で確定 36 → 38 行 / 未確定 9 → 7 行。** [#387] の所有は 4 → 2 行になった。

- **Changed**: `engine.whispers2t.load_model` を**「ローカル snapshot から CTranslate2 / tokenizers へロードする境界」へ再定義**し、**②wide-path** で確定した。
  - **Before**: `path_desc` は「HF repo id (パスではない) + 既定 HF cache ディレクトリ」、`receiver` は「whisper_s2t → huggingface_hub → CTranslate2 + tokenizers」。**HF repo id / cache / huggingface_hub まで束ねた複合境界**だった
  - **After**: `ctranslate2.models.Whisper(dir)` (C++) と `tokenizers.Tokenizer.from_file()` (Rust) の **2 つのネイティブ境界**に絞った。`cjk_kana` / `outside_acp` の両方で pass
  - **ローカル dir を渡す測定は cache 経路を通らない。** 複合のまま確定すると**測っていない範囲まで検証済みに見える**ので、cache の所在と書き込みは [#430] へ分離した
  - `rationale` の「この engine だけ `manager.huggingface_cache()` で包まれていないため、既定の HF cache が実世界の経路になる」も**事実と違った** — 実際は `platformdirs.user_cache_dir("whisper_s2t")` 配下の**自前 cache**である
- **Changed**: `engine.qwen3asr.from_pretrained` を**「ローカル snapshot からの load 境界」へ再定義**し、**②wide-path** で確定した。
  - **Before**: 「初回ダウンロード境界」。`real_model` tier の「ネットワークを使わない」契約と衝突していた
  - **After**: download / cache への書き込みは [#428] が持つ。**`%TEMP%` はあえて緩和せず**に測るので、モデル path と `%TEMP%` が同時に非 ASCII になる**実運用条件の計測**である。両 variant で pass
  - **`ascii_safe_temp_environment()` wrapper 撤去の「load 層の」根拠が揃った** — (a) `huggingface_hub` の download は system `%TEMP%` を使わない (実測)、(b) 未緩和の非 ASCII `%TEMP%` で**ローカル snapshot を**load できる (本行)。**production は repo ID を渡すので、この 2 つだけでは撤去できない** — 撤去 PR で `HF_HUB_OFFLINE=1` + 既存 cache のまま **production と同じ repo ID** を未緩和の `outside_acp` な `%TEMP%` でロードする smoke test を行う。撤去は production 変更なので別 PR とする
- **Fixed**: **worker が probe の戻り値を検証していなかった。** `dict` 以外を返すと `observation` が `None` のまま control と一致し、**境界を一度も通らずに `pass` になる**。
  - **Before**: `observation = impl(ctx)` をそのまま emit
  - **After**: `dict` でなければ `TypeError` で loud に落とす。仕込みプローブ `selftest.returns_none` で固定した
  - **Migration**: なし (テストハーネス内部)
  - **実際に踏んだ** — `@probe` デコレータを helper に付けてしまい probe 本体が一度も呼ばれなかったのに、2 variant とも緑だった (実行時間 0.7s の不自然さで気付いた)
- **Fixed**: `asr.utterance_wav.whispers2t` の source ガードが**空洞**だった。`whispers2t_base` は models root の**空 dir** で、`_real_model_is_usable()` は `path.is_dir()` しか見ないため**この precondition は落ちようがなかった**。実体は whisper_s2t 自前の cache にある。[#413] で qwen3asr に入れた「snapshot の実在まで確かめる」ガードを両行へ入れた
- **Added**: 両行の worker へ **`HF_HUB_OFFLINE=1` を起動前に渡す**。whispers2t にも要る — `os.path.isdir` 分岐に入り損ねると `download_model()` が**ネットワークへ出る** ([#413] PR C で qwen が実際にそうなっていた)。offline にしておけば黙ってダウンロードせず落ちる
- **Added**: `benchmark_results/nonascii/2026-09-02/results.json` — clean tree (`0ae123e`) から **cheap / real_model / heavy / gpu を 1 セッションで**生成した証拠 (54 passed, **skip 0** / 125 レコード / 39 probe)。**これまで skip していた 2 行が走るようになった**
- **Changed**: CI ゲートへ `engine.whispers2t.load_model` / `engine.qwen3asr.from_pretrained` の PASSED 要求を追加した
- **Note**: **変異テストが 2 つの設計ミスを暴いた。** (1) `model_path_under_probe_root` を observation で返していたが、**control と trial の両方が同じ値になる**ので差分判定で捕まらない → **報告ではなく assert** へ。(2) `%TEMP%` の判定を `isascii()` にしていたが、**control の root は常に ASCII** なので control では発火せず、trial だけ落ちて **`fail_loud` (境界が壊れた) に見える** → 「variant root の外に逃がされているか」で判定し control でも落ちるようにした

#### 未実測境界のうち 2 行を ②wide-path で確定 (Issue [#387] PR A、epic [#380])

**実測で確定 34 → 36 行 / 未確定 11 → 9 行。** [#387] の所有は 6 → 4 行になった。

- **Changed**: `engine.voxtral.autoprocessor` を **②wide-path** へ確定した。
  - **Before**: `verified_method=None`。`unmeasured_reason` は「processor の optional 依存 mistral-common が未導入のため skip された」
  - **After**: `cjk_kana` / `outside_acp` の両方で ASCII control と processor / tokenizer のクラスが一致。**tokenizers (Rust native) は非 ASCII を通す**。旧理由は嘘になっていた — `mistral-common 1.8.5` は導入済みで、実際には**測れていたのに 1 variant しか回っていなかった**
- **Changed**: `engine.voxtral.autoprocessor` に `required_variants=("cjk_kana", "outside_acp")` を設定した。**`cjk_kana` だけでは ② を名乗れない** — `ユーザー` は cp932 の内側なので、tokenizers が narrow path (ACP 変換) で実装されていても**日本語 Windows なら通ってしまう**。ACP の外側まで通して初めて narrow path を排除できる
- **Changed**: `resources.resource_locator.source_root` を `tier="none"` / source-check から **runtime 実測**へ格上げし、**②wide-path** で確定した。4 variant すべてで探索 root が非 ASCII のまま解決され、同梱 resource を読み戻せた
  - **Before**: `unmeasured_reason` は「非 ASCII パス配下への第二 install tree が必要 (site-packages を丸ごと複製する)。本 issue のコストに見合わないため未実測」
  - **After**: **その前提が誤りだった** — `livecap_cli/` (2.9 MB) だけを非 ASCII 側へ物理コピーし `PYTHONPATH` 経由で孫プロセスに import させれば足りる。依存は venv の site-packages に残る
- **Added**: `resources.source_root` probe。**symlink は使わない** (`resolve()` が ASCII 側へ戻り、測ったつもりで何も測っていない状態になる)。**元 package を import したら fail loud させる** — editable install が `PYTHONPATH` に勝つと非 ASCII を一度も通さないまま緑になるため (変異で確認: `PYTHONPATH` を外すと exit=3 で元 package の path を名指しして落ちる)。cwd は ASCII scratch に固定する
- **Fixed**: `materialize_file()` の観測が**呼び出し回数に依存**していた。**ASCII control の root は variant を跨いで共有される**ので、2 回目以降に `"existing"` を返すと `dominant_mechanism()` の答えが `hardlink` → `mixed` と変わり、差分判定が「path と無関係な非決定性」を検出して **`error_harness`** に落ちる。境界のバグでもないのに証拠が取れない
  - **Before**: `if dst.exists(): return "existing"`
  - **After**: `(st_dev, st_ino)` の一致で hardlink を判定し、「今回コピーしたか」ではなく「**どう実体化されているか**」を返す。Windows でも `os.stat` は file index を `st_ino` に載せる (実測で確認)。`st_ino` が 0 の FS では**すべて同一に見えてしまう**ので copy 扱いにする
  - **Migration**: なし (テストハーネス内部)
  - **`required_variants` を 1 件から 2 件へ増やした瞬間に表面化した** — 1 variant しか回っていなかったので control が 1 回しか走らず、これまで見えていなかった
  - **`os.link()` の成功だけで `"hardlink"` と答えないこと** (レビュー指摘)。判定を既存ファイル側と揃えないと、`st_ino` を返さない FS で「1 回目 = hardlink / 2 回目 = copy」となり**同じドリフトが別の環境で再発する**。両分岐で同じ述語を通し、「inode が使えないなら常に copy と答える」契約を閉じた
- **Changed**: `engine.reazonspeech.sherpa_narrow_path_signature` に**再評価 trigger** を記録した。4 variant すべて pass するが、不正 ONNX が `tokens.txt` より先に検証されるため**境界に届いていない**。`covers_boundary=False` を維持し、**この pass を確定に使ってはならない**ことを明記した
- **Added**: `benchmark_results/nonascii/2026-09-01b/results.json` — clean tree (`0740fb5`) から **cheap / real_model / heavy / gpu を 1 セッションで**生成した証拠 (52 passed, 2 skipped / 121 レコード / 37 probe)
- **Added**: `tests/nonascii/README.md` に**同じ日に 2 回目の run を出すときは接尾辞で分ける**規約を明記した。上書きすると、先の PR の CHANGELOG が実在しない内容を指すことになる
- **Changed**: CI ゲートへ `test_real_model_boundary[engine.voxtral.autoprocessor] PASSED` を追加した。`required_variants` があるので node の PASSED を要求すれば両 variant の完走まで保証される
- **Note**: skip した 2 件 (`whispers2t.load_model` / `qwen3asr.from_pretrained`) は `_REAL_MODEL_SOURCES` に source 定義が無い [#387] PR B の対象で、どちらも `verified_method=None` なのでゲートには影響しない

#### ネイティブ境界向けの ASCII path 保証 API (Issue [#375] PR 2、epic [#380])

Windows のユーザー名が非 ASCII だと、既定の `cache_root` (appdirs 由来で**ユーザー名を含む**) は「ASCII が必要なネイティブ境界」の役に立たない。**ASCII だと保証できる場所**を選ぶ API を `livecap_cli/paths/` に新設した。

- **Added**: `ascii_safe_temp_environment()` — ネイティブが**自前で `%TEMP%` へ展開する**境界向け (NeMo の untar 等)。`TEMP` / `TMP` / `TMPDIR` / `tempfile.tempdir` を ASCII 保証ディレクトリへ向ける
- **Added**: `ascii_safe_workspace()` — **我々がファイルを作る**境界向け (発話ごとの wav 等)。env を触らないので自明にスレッド安全・ネスト可
- **2 つの非対称が設計の核**である。`ascii_safe_temp_environment()` は**退出時に自分のディレクトリを消さない**が、`ascii_safe_workspace()` は**消す**。同じ 1 つの事実から出る — **プロセス全体の TEMP を向けている間は無関係なスレッドの `NamedTemporaryFile()` もそこへ落ちる**ので、消すと [#386] のデータ消失が再発する。向けていなければ自分のファイルしか無い。したがって発話ごとの wav の正解は workspace であり、**発話ごとにプロセスグローバル状態を書き換えてはならない**
- **staging root を配る操作は configuration を freeze する。** preview を読んでいた頃は、初回利用の後に `configure_resources(staging_root=...)` が成功してしまい、**既に配った root と食い違う設定が黙って受け入れられた** — 「API > env > default」と「ホストの指定を黙って無視しない」契約の両方に反する。resolved 値を配る操作は確定させなければならない (後から取り消せる余地が無いため、graph 構築のように「成功してから freeze」にはできない)
- **明示指定が実行時に使えなくなっても候補へ降りない** (R2)。configure 時には有効でも、その後 ACL 変更・削除・容量で使えなくなり得る。降りると「運用者が指定した場所を黙って使わない」ことになる
- **staging root の候補 ladder**: 明示指定 (`configure_resources(staging_root=...)` / `LIVECAP_CORE_ASCII_STAGING_DIR`) → `%ProgramData%` → `%SystemDrive%` → `%PUBLIC%` → cache root → system temp。述語は **ASCII → 長さ → 作成 → 書き込み probe**。**明示指定が不正なら候補へ降りず fail loud** (運用者の明示指示を黙って無視しない)。全滅時も**元の非 ASCII path へ黙って fallback しない**
- **`%ProgramData%` 候補にユーザー名そのものを使わない** — `sha256(username)[:8]` で分離する。非 ASCII なユーザー名を候補 path に混ぜたら、ASCII 保証という目的自体が壊れる
- **Added**: `purpose` の slug 契約 (`[A-Za-z0-9_-]`、1..16 文字)。公開 API が受け取った値をそのまま path へ連結するので、**検証しないと保証が 3 つとも破れる** — `"日本語"` は完成した path を非 ASCII にし (この API の存在意義が消える)、`"../outside"` は staging root の外へ出て、長い文字列は path 予算の計算を崩す。予算計算が「purpose <= 16」を前提にしているのだから、**前提は契約として強制する**
- **Added**: **所有権マーカー兼 lease** (`.livecap-entry`)。1 つのファイルが 2 つの役割を持つ — **存在**が「この entry は LiveCap が作った」(entry と同じ寿命)、**開いていること**が「いま使っている」(スコープの間だけ)。
  - **所有権が要る理由**: 明示 staging root には運用者が**既存のディレクトリ**を指定できる。その配下に無関係なデータがあっても TTL だけで回収すると**それを消す** — [#386] のデータ消失そのもの。reaper は**印のある entry にしか触らない**
  - **lease が要る理由**: 「TTL 超過かつ `rmtree` が通る」は生存判定ではない。使用中のプロセスがその瞬間ハンドルを開いていなければ消せてしまう。**スコープの全期間 開いたまま**にすることで、開いていること自体が証明になる。Windows では `rmtree` の `PermissionError` が、POSIX では `flock` が判定になる
  - **entry の中に置く**のが Windows の保護そのもの。外に置くと `rmtree(entry)` を妨げない。reaper の単位 (entry) と消費側に見せるディレクトリ (その子) を分けて「空のディレクトリを返す」契約と両立させた
  - **退出時にマーカーを unlink しない**。消すと ①所有権の印が失われて残骸が回収対象から外れ、②POSIX で他者が lock を保持している path を消してしまい、その holder の entry が次の reaper から「印無し」に見えて削除される
  - **確立できなければ `AsciiPathError` を送出する**。保護なしで進めると reaper から見て使用中と区別のつかない entry が生まれる。TTL は猶予であって安全性ではない
- **Added**: TTL ベースの孤児回収 (既定 14 日)。`ascii_safe_temp_environment()` が消さない残骸を回収する。対象は**所有権マーカーを持つ entry に限る** (印の無いディレクトリは、どれだけ古くても他人のもの)。**PID 生存判定は使わない** (子プロセスは親の TEMP を継承するがディレクトリ名は親 pid のままで、pid は再利用される)。代わりに **OS に判定させる** — 掴まれていれば削除が失敗するので、そのエントリを飛ばす。best-effort で、失敗しても例外にしない
- **契約**: `ascii_safe_temp_environment()` が支えるのは**スコープ内で完了する同期境界だけ**。Python のハンドルは既定で非継承 (PEP 446) なので、**親のスコープより長生きする子プロセスは lease で保護されない**。この context の中で spawn した子は、抜ける前に終了 / join させること。lease は `TEMP` を書き換える**前**に確立し、**復元し終わるまで**保持する (逆順だと「TEMP が向いているのに lease が無い」区間ができる)
- **Added**: **staging 発生ログ** (Issue [#375] の AC)。staging のたびに `boundary` / `mechanism` (`temp-environment` / `workspace`) / `resolved_root` / `root_source` / `fallbacks` を 1 行の構造化ログへ出す。**root が cache hit でも出す** — 「なぜこの root か」は 2 回目以降こそ分からなくなる。**`(boundary, mechanism, root)` ごとに初回だけ INFO**、以降は DEBUG (`ascii_safe_workspace()` は発話ごとに呼ばれるので毎回 INFO だと realtime 転写でログが埋まる。一方 DEBUG だけでは通常の CLI / GUI ログで観測できない)
- **Changed**: `StagingRootStatus.mechanism` → **`root_source`**、および **`fallbacks` を追加**。旧 field は root の**選択元** (`%ProgramData%` 等) を入れていたが、本 repo では "mechanism" を hardlink / copy の materialization の意味で使っており (`tests/nonascii/artifacts.py`)、読み手が誤解する。どの staging API を通ったかは root ではなく**呼び出しごと**の属性なので、型ではなくログへ出す。`fallbacks` は**後続候補が成功すると失われる**拒否理由を保持する
  - **Migration**: `staging_roots[i].mechanism` を読んでいた host は `root_source` へ。PR 1 の時点で常に空 tuple だったため、実際に読めた host は存在しない
- **Fixed**: `StagingRootStatus.source_volume` に**採用された root の drive** が入っていた。field の定義は「**staging 元**のボリューム」なので、`D:` から staging して `C:\ProgramData\...` へ降りると入力の `"D:"` が失われ、**fallback の関係が説明できなく**なる。現行 2 API は `source_volume=None` で呼ぶのに Windows では `"C:"` 等が入る食い違いもあった。入力を `RootSelection` に保持してそのまま記録する。採用先の drive が要るなら `path` から求められる。あわせて重複判定を **`(path, source_volume)`** 単位にした — 同じ root でも staging 元が違えば別の関係
- **Added**: `AsciiPathError(RuntimeError)` 階層。**`OSError` 派生にしない** — 呼び出し側が `except OSError` で握り潰すと「ASCII を保証できなかった」が黙って握られる。`boundary` を**必須キーワード引数**にして、失敗メッセージに**境界名 → 問題の path → 何を試して各々なぜ失敗したか → env var を名指しした対処**が必ず載るようにした
- **`logger.warning` を出して非 ASCII の path を返すことはしない。`strict=False` も作らない**
- **Changed**: `STAGING_ROOT_MAX_LEN` を **160 → 120**。PR 1 では staged path の形が未定だったための暫定値で、コメントに「PR 2 で締め直す」と書いていた。本 PR で `<root>\<purpose>\<uuid12>\...` に確定したので計算できる — 120 なら消費側のサブツリーに約 109 残り、NeMo の入れ子展開に足りる (160 では 69 しか残らない)。これにより `\?\` 接頭辞を一切使わずに MAX_PATH 260 に収まる
- **Fixed**: `source_volume` 候補が Windows で**ドライブ相対 path** になっていた。`Path("D:") / x` は `D:\x` ではなく **`D:x`** (そのドライブのカレントディレクトリ基準) で、`normalize_path()` が process の cwd で解決する。実測では `D:` が **`D:\Codes\livecap-cli\LiveCapStaging`** (リポジトリの中) に落ちており、**同じ入力が起動場所次第で別の場所を指していた**。`%SystemDrive%` 候補が `+ os.sep` しているのと同じ対策を入れ、あわせて `validate_source_volume()` で**相対値を拒否**する (黙って cwd 基準で解決しない)。現行 2 API は `source_volume=None` なので production 経路には出ておらず、`select_staging_root()` も**内部 selector** (top-level export ではない) だが、**consumer が現れる前に出荷した引数**なので契約としては既に存在していた — これ自体が「消費者ゼロの先行実装が後から欠陥を生む」例である
- **Changed**: `livecap_cli.paths` の**公開面を境界 API 2 つと例外だけに絞った**。`select_staging_root` / `RootSelection` / `reset_staging_root_cache` / `reap_staging_root` / `DEFAULT_TTL_HOURS` は **production consumer が 1 つも無い**まま top-level export されており、`core-api-spec.md` §3.4 が公開 API として挙げているのも境界 API 2 つだけだった。root 選定と回収は内部実装とし、ホストは `get_resource_configuration().staging_roots` を読む (**selector を直接呼ぶと configuration を freeze する副作用**があり、readback には無い)
  - **Migration**: 内部 module が要る場合は `from livecap_cli.paths import roots` のように明示 import する
- **Changed**: **`fork()` を支えないことを明記**し、「子で `reset_staging_root_cache()` を呼べばよい」という案内を撤回した。同関数が戻すのは選定キャッシュとログ抑制だけで、**temp-environment の `RLock` (別スレッドが保持したまま fork するとデッドロック) と深度カウンタ、lease の file descriptor (親子が同じ open file description を共有するので子が閉じると親の lease が外れる)、reaper の once-state、freeze 済み configuration は対象外**だった。**安全になるように読めてしまう記述は、無いより悪い**。一括で戻す API は使う consumer が居ないので作らない — `spawn` を使うか、本 API を親でだけ使うこと
- **Removed**: **`unicode_safe_temp_directory()`**。**本番呼び出しがゼロ**で (4 engine が import しているだけ)、cli 側・livecap-gui 側とも利用が無かった。名前に反して ASCII 保証も無い (`cache_root` はユーザー名を含む)。移行先が要らないデッドコードなので、置換を伴う PR 3 を待たずに削除する。4 engine の未使用 import・nonascii probe・registry 行・棚卸し表の行 (③staging 10 -> 9) も同時に除去した
  - **Migration**: 呼び出しが存在しないため影響なし。ASCII 保証が要る場合は `ascii_safe_temp_environment()`、要らない場合は `get_temp_dir()` を使う
- **Removed**: `temp_environment()` の **`unique` 引数**。上記削除で `unique=False` の呼び出しがゼロになった。**旧挙動を保つためだけの分岐を残さない** (pre-1.0 方針)。常に固有ディレクトリを作り、常に lease を取る — `ExitStack` の条件分岐も消えた
- **Removed**: **`is_ascii_safe()`**。`str(path).isascii()` の 1 行 wrapper で**消費者が 0 件**、しかも `_reject_reason()` が同じ判定を直接持っていた (判定が 2 箇所)。加えて #378 §6.9 が要求する「`\?\` 付き入力を `ValueError` で拒否する」を**満たしておらず、設計書を読んだホストの期待と食い違う**状態だった。`ascii_safe_path()` を実装するときに §6 のとおり作る
- **Changed**: `resolve_staging_root()` を **`select_staging_root()` へ統合**。`path` だけを返す薄い wrapper は**本番呼び出しが 0 件**だった (2 つの呼び出し側はどちらもログ用に経緯を必要とする)。**同じ操作に 2 つの名前を与えても保守対象が増えるだけ**なので 1 本にした。戻り値の `RootSelection` は `roots` module 内で定義される型であり、**selector ともども top-level の公開 export ではない** (同 Unreleased の「公開面を境界 API 2 つと例外だけに絞った」を参照)
- **Changed**: TEMP 移設の実装を `utils/__init__.py` から `paths/temp_env.py` へ移した。**ロック実装を 2 つ保守しない**ため、**`unicode_safe_download_directory()`** (PR 3 で削除) は移設先へ委譲する薄い層になった。挙動は変えていない (base は従来どおり `cache_root`)。**`unicode_safe_temp_directory()` は本 PR で削除済み**なので、残る旧 helper はこの 1 つだけである
- **`ascii_safe_path()` (既存ツリーの staging) は実装していない。** 設計は #378 §6 に確定しているが、**必要とする境界が現時点で 0 件**である — 唯一の候補だった sherpa-onnx は 1.13.6 への version bump で ②wide-path になった ([#377])。残る ③staging **9 行**はすべて `%TEMP%` 移設 (4) か workspace (5) で、上記 2 API で足りる (`unicode_safe_temp_directory()` の削除前は 10 行だった)。消費者が現れた時点で実装する
- **Docs**: `docs/reference/api.md` に**リソース探索順**と `reset_resource_graph()` の契約を追加。探索順は直感的でなく、**`extra_resource_roots` は project / source root より後ろ**なので、override 目的で渡しても既存リソースに負ける — 省略すると誤用につながる。`ResourceSearchResolution` の全 field と、API 指定時に `LIVECAP_RESOURCE_ROOT` を検索対象から除外して `overridden_env` へ記録すること (「上書き」を「優先 fallback」にしないため) も明記した。あわせて**ホスト向けの入口は configure / get / reset の 3 つ**と範囲を示した
- **Fixed**: readback の例が **immutable snapshot を再取得していなかった**。`ResourceConfiguration` は frozen dataclass なので、境界利用前に取得したインスタンスは**後から staging が起きても更新されない**。例のとおり書くとホストは永遠に空の `staging_roots` を読む。境界呼び出しを挟んで**取り直す**例に差し替え、`get_resource_configuration()` の説明にも immutable を明記した
- **Fixed**: 棚卸し文書に**存在しない fork 復旧 API** (`reset_ascii_staging_state()`) の案内が残っていた。同関数は存在せず、仮に用意しても roots の選定キャッシュ / reaper の once-state / freeze 済み configuration / lease の fd を一貫して戻せない。**「呼べば安全になる」と読める記述は無いより悪い**ので撤回し、「`spawn` を使うか親でだけ使う」へ揃えた (併せて前回の追記が壊していた Markdown も修復)
- **Fixed**: async の案内を **`paths/__init__.py` / `paths/temp_env.py` / `core-api-spec.md`** でも「**enter・境界処理・exit を 1 つの同期関数にまとめ、その全体を 1 回の `asyncio.to_thread()` で実行する**」に限定した。`api.md` だけ直すと**コード側の docstring と契約が分裂**し、enter/exit を別々の `to_thread()` へ渡す危険な実装を防げない
- **Fixed**: 「**無関係な境界を直列化しない**」という記述が実装と食い違っていた。`ascii_safe_temp_environment()` は `TEMP` がプロセス全体の状態であるため**排他をスコープ全期間保持**し、**別スレッドの呼び出しは `boundary` / `purpose` に関係なく直列化される** (実測: 別 boundary + 別 purpose のスレッドが 0.60 秒待機)。待たされるのであって `TempEnvironmentConflictError` にはならない — 同エラーは**同一スレッドで別 purpose をネスト**したときに出る。`ascii_safe_workspace()` は直列化されない。`paths/__init__.py` / `core-api-spec.md` / `api.md` の 3 箇所を修正
- **Docs**: `docs/reference/api.md` に **resource 設定 / readback の節**を追加 (`configure_resources()` の 6 引数と `API > env > default` / `data_root` が派生させるのは models・cache だけであること / freeze 契約 / `ResourceConfiguration`・`RootResolution`・`StagingPolicy`・`StagingRootStatus` の field / 例外)。**公開 API を追加した PR の責務はアーキテクチャ仕様ではなくホスト向けリファレンス**という本 PR の根拠は、PR #408 の resource API にもそのまま当てはまるため
- **Docs**: 例外メッセージの「境界名 → path → 試行理由 → env var」順は、**候補 ladder 全滅時の `AsciiStagingUnavailableError` に限定**した。`configure_resources()` 時の検証は `boundary=None` / `attempts=()`、`TempEnvironmentConflictError` は候補一覧を持たない。型ごとの `boundary` / `attempts` の有無を表で示す形にした
- **Docs**: `docs/reference/api.md` に**ホスト向け公開 API の節**を追加 (`ascii_safe_temp_environment()` / `ascii_safe_workspace()` / 例外 3 種 / 使い分けと非対称 / `boundary`・`purpose` の契約 / fail-loud / staging root の決まり方と readback / 明示的な非保証)。PR 2 では `core-api-spec.md` (アーキテクチャ仕様) しか更新しておらず、**公開 API を追加したのにホスト向けリファレンスが無い**状態だった
- **Tests**: 新規 **101 件** (`tests/core/paths/` 全体。`pytest tests/core/paths --collect-only` で実測)。**`source_volume` 候補が絶対 path になり cwd で動かないこと** / **相対 `source_volume` の拒否** / **`source_volume` が staging 元を保持し、別ボリュームへ降りても失われないこと** / **`(path, source_volume)` 単位の重複判定** / **cache hit でも staging ログが出ること** / **拒否候補と理由が後続候補の成功後も残ること** / **初回 INFO・以降 DEBUG** / **`mechanism` と `root_source` を混ぜないこと** / **印の無い古いディレクトリを reaper が消さないこと** (レビューで実測された [#386] 同種のデータ消失の回帰) / **マーカーがスコープを越えて残ること** / **lease を確立できないとき yield 前に落ちること** / **POSIX で取得に失敗した側が既存 holder のマーカーを消さないこと** / **lease が env 書き換えより先に確立されること** / 候補 ladder の順序 / 述語 (非 ASCII・長すぎ・作成不可・書き込み不可) / **書き込み probe が既存ファイルを壊さない** / **`%ProgramData%` 候補にユーザー名が現れない** / 全滅時のメッセージ契約 / `%TEMP%` の移設と復元 (正常・例外) / ネストの reentrant 性と purpose 衝突 / **並行スコープを途中復元しない** / **temp env は消さず workspace は消す** (非対称を 1 つのテストで並べる) / reaper の TTL と**使用中を消さない**こと / **例外クラスが 1 つだけであること** / **初回利用が freeze すること** / **後続の configure が黙って無視されず落ちること** / **明示 root が使えなくなったら候補へ降りないこと** / **不正な `purpose` 9 種の拒否** / **lease 中のエントリを reaper が消さないこと** / **workspace が空のまま entry が leased であること**

#### ホスト設定可能な resource API と共有 resource graph (Issue [#375] PR 1、epic [#380])

**ホストアプリが root を設定しても黙って効かない**状態の解消。3 つの manager が独立した singleton として生成され、**それぞれが構築時に env を直読み**していたため、`FFmpegManager` が使う cache root と `get_model_manager()` の cache root が**別物になり得た**。ホストにはそれを観測する手段も無かった。

- **`configure_resources()` / `get_resource_configuration()`** を追加。優先順位は **API > env > built-in default**。`data_root` から派生するのは `data_root/"models"` と `data_root/"cache"` **だけ**で、静的 resource の検索 root は派生しない (書き込み用 root と読み取り用 root は別物であるため)
- **明示された入力が使えないときは候補へ黙って落ちず送出する。** 「ホストが渡した root が使えない」ことは「別の場所を勝手に使ってよい」という意味ではない。判定は root 種別ごとに定義した — models/cache/data は **作成 + 書き込み probe**、resource/extra は**存在する読み取り可能な directory** (書き込みは要求しない — read-only なインストール先を指すのは正当な使い方)、staging は **ASCII + 長さ + 書き込み可能**
- **API が設定済みの env を上書きするときは `WARNING` を出し、readback の `overridden_env` にも載せる。** 非 ASCII パス問題を `LIVECAP_CORE_MODELS_DIR` で回避しているユーザーのホストが `data_root` を渡すと、env が無視されて**数 GB の再ダウンロード**が起きる。優先順位を決めるだけでは readback を見ない限り観測できない
- **静的 resource の検索順は API 指定の有無で 2 分岐し、混在しない**: API 指定時は `API → project → source → extra → package fallback` とし、**`LIVECAP_RESOURCE_ROOT` を検索順から除外**して `overridden_env` に記録する。env root を fallback として残すとそれは「上書き」ではなく「優先 fallback」であり、上記の記録と食い違う
- **central factory** `resources/graph.py` が manager 一式を組み立て、`FFmpegManager` へ locator と model manager を**必須注入する** (既定値を与えて暗黙に shared graph から取ることはしない — 無引数で構築できると、その instance が `reset_resource_graph()` の管理外に残り、片方だけ注入すれば custom と shared が混ざった graph も作れてしまう)。以前は `FFmpegManager.__init__` が private に `ResourceLocator()` / `ModelManager()` を作っていた。**`livecap_cli/` の他の場所で構築していないことを AST で検査する test** を追加 — 直接構築するとその instance だけが frozen configuration の外側に立ち、本 issue の不具合が再発するため
- **`get_resource_configuration()` は freeze せず、filesystem も一切触らない。** 起動ログに readback を出すホストが、参照しただけで root を実体化してしまうのを避ける。したがって `is_frozen=False` の preview は**利用可能性が未検証**である
- **env は freeze 時点で固定**し、以後の変更を無視する。manager は env を読まない
- **再設定は静的 configuration 全体が一致するときのみ no-op 成功**。resolved path だけを比べると「`data_root` を渡した」と「`models_dir`/`cache_dir` を個別に渡した」を区別できず、意図が違うのに成功してしまう
- **path 正規化は `expanduser` → `abspath` → `normpath`。`Path.resolve()` は使わない** — symlink を追跡するとホストが渡した path と別の場所を指し始め、readback が「渡していない path」を返すことになる
- **Added**: `ResourceConfiguration` / `RootResolution` / `ResourceSearchResolution` / `StagingPolicy` / `StagingRootStatus` / `ConfiguredPath` / `OverriddenEnv` / `ResourceConfigurationError` / `AsciiStagingUnavailableError`、環境変数名の定数 (`ENV_MODELS_DIR` 等)
- **Changed**: **`ModelManager.__init__` / `ResourceLocator.__init__` が解決済みの値を要求する**
  - **Before**: `ModelManager()` / `ResourceLocator()` が env と既定値を自分で解決していた
  - **After**: `ModelManager(models_root=..., cache_root=...)` / `ResourceLocator(search_roots=...)`。解決は `resources/configuration.py` の責務
  - **Migration**: 直接構築せず `get_model_manager()` / `get_resource_locator()` を使う。**構築は `build_resource_graph()` のみが行う**
- **Changed**: **manager getter から `force_reset` を削除**
  - **Before**: `get_model_manager(force_reset=True)` で 1 つだけ作り直せた
  - **After**: 引数なし。graph 全体を作り直すには `reset_resource_graph()`
  - **Migration**: 一部だけ差し替えられると **graph の一部が古い configuration を参照する**状態が作れてしまうため撤去した。テストで env を読み直したい場合は `_reset_resources_for_tests()`
- **Removed**: **`reset_resource_managers()`**。`reset_resource_graph()` (frozen configuration を維持して graph を再生成) と `_reset_resources_for_tests()` (configuration も消して env 再読込) に分かれた。前者は「graph だけ作り直すのか configuration も消すのか」が未定義だった。shim は残さない (pre-1.0)
- **書き込み probe は `mkstemp()` で一意なファイルを作る。** 固定名にすると、同名のファイルがあったときに内容を truncate したうえで削除することになり (symlink ならリンク先まで)、複数プロセスの同時 configure も同じ probe を奪い合う
- **Tests**: 新規 70 件。優先順位の全組み合わせ / 検索順の 3 ケース (**API あり + env あり で env が除外され記録されること**を含む) / root 種別ごとの fail loud / 上書きの WARNING と記録 / **preview が directory を 1 つも作らないこと** / freeze 境界と env 固定 / 再設定の同一性判定 / 正規化 (**symlink を追跡しないこと**) / **AST による直接構築の検査** / configure と初期アクセスの競合 / **既存 sentinel が書き込み probe を生き延びること** / **env の resource root も fail loud すること** / **graph 構築に失敗しても freeze が成立せず、正しい設定で復旧できること**

**本 PR は #375 の PR 1** であり、`ascii_safe_path()` 等の staging core (PR 2)、旧 `unicode_safe_*` の削除 (PR 3)、`utterance_wav` の移行 (PR 4) は含まない。`staging_policy` は**明示指定の有無**だけを保持し、候補 ladder は PR 2 が実装する。

#### 非 ASCII パス境界の棚卸しと検証ハーネス (Issue [#378]、epic [#380] Phase 0)

非 ASCII パス起因の不具合 (sherpa-onnx / NeMo で確認済み 2 件) の**露出範囲を確定**するための調査成果。実際の修正は #375 / #379 / #377 が担当し、本変更は `livecap_cli/` の production code を変更しない。

- **棚卸し表** `docs/research/nonascii-path-boundary-inventory-2026-08.md` — ネイティブ / 第三者ライブラリへパスを渡す **47 境界**を列挙し、境界ごとに ①buffer / ②wide-path / ③ASCII staging / ④fail-fast / 非該当 のいずれかへ分類 (**未分類ゼロ**)。内訳は applicable 44 行 (runtime 実測で確定 30 / 未確定 14) + 非該当 3 行、実測レコード 132 件。**「決定」と「実測で確定」を別の列に分離**しており、未実測を ② として数えない。`ascii_safe_path()` の契約 (シグネチャ / 生存期間 / 機構 ladder / staging root / 並行ロード契約 / fail-loud / 却下した代替案) もここに確定
- **検証ハーネス** `tests/nonascii/` — 実装 PR がそのまま回帰テストとして再利用できる。全プローブを子プロセスで実行 (ネイティブ `abort()` 耐性 + 親プロセスの env を汚さない)、ASCII control との **differential 判定**で `fail_silent` を機械的に検出。仕込み欠陥による自己検証付き
- **受け入れ条件の機械化** `tests/nonascii/test_registry.py` — 未分類ゼロ / silent-failure 行への「現状維持」割り当て禁止 / callsite の生存 / 証拠の有無を CI で強制する
- pytest marker `nonascii_paths` を追加 (cheap tier は既定スイートで実行、実モデル tier は `slow` + `LIVECAP_NONASCII_REAL_MODELS=1`)
- `docs/contributor/adding-an-engine.md` に §10 パス境界チェックリストと AP-6 を追加 (新規 engine 追加時の必須確認項目化)

**新規に判明した事実**:

- Windows で stdio がパイプの場合、`sys.stdout` は `surrogateescape`、`sys.stderr` は `backslashreplace` で、**stdout だけが `UnicodeEncodeError` で落ちる**。CLI の SRT stdout 出力が該当し、別 issue で対応する
- `unicode_safe_temp_directory` / `unicode_safe_download_directory` は `%TEMP%` を `cache_root` へ移設するだけで、その `cache_root` はユーザー名を含むため **ASCII 安全ではない** (全 variant で `fail_silent` を実測)。加えて後者は**共有ディレクトリを rmtree** し、スコープ中に他スレッドが作った一時ファイルを消す (実測で確認)
- sherpa-onnx 1.12.39 は `tokens_buf` / `hotwords_buf` を持たない → 方式①は利用不可 (#361 の hotwords も同じ narrow path を踏む)
- **NeMo が壊れる主因は `.nemo` のパスではなく NeMo 内部の `%TEMP%` 展開先だった** — 片側ずつ固定して実測した結果 (`.nemo` だけ非 ASCII → pass / `%TEMP%` だけ非 ASCII → fail_silent)、`restore_from(restore_path=...)` は非 ASCII を正しく扱えると確定。#379 のレバーは `ascii_safe_temp_environment()` だけで、`.nemo` の staging は不要

#### `VADFileSegmenter` — offline 音声の VAD 分割 adapter (Issue [#366] Phase 1)

`livecap_cli.vad.VADFileSegmenter`: streaming 用 `VADProcessor` を `FileTranscriptionPipeline` の `Segmenter` 契約 (`Callable[[np.ndarray, int], list[tuple[float, float]]]`) に適合させる公開 adapter。呼び出し毎に `reset()` (ファイル間・例外後の状態非持越)、interim segment 除外、EOF で `finalize()` 回収。CLI file mode の `--vad` 接続の基盤で、GUI 等の offline 一括処理からも利用可能。あわせて `FileProcessingResult.metadata` に `detected_segment_count` (検出区間数) / `segmentation_empty` (注入 segmenter がセグメントなしと判定) を追加。

#### EngineInfo 言語解決 metadata + `EngineMetadata.resolve_language()` (Issue [#365])

- `EngineInfo` に末尾 default 付き field を追加 (後方互換、#286 と同パターン): `cli_default_language: str` (`--language` 未指定時の実効値。導入時の名称 `default_language` は [#230] で改名) / `supports_language_auto: bool` (native 自動検出対応、voxtral / qwen3asr のみ True)
- `EngineMetadata.resolve_language(engine_id, requested) -> str`: CLI `--language` の単一解決点。未指定 → `cli_default_language`、明示 → 正規化 + 検証、`auto` → 対応 engine のみ許可。全拒否は `ValueError` (不正形式コードの `langcodes.LanguageTagError` も friendly な `ValueError` に変換)

#### Public SRT serializer `build_srt` / `write_srt` (Issue [#363])

`livecap_cli.transcription.srt` に公開 SRT serializer を追加 (top-level `livecap_cli` からも import 可)。`FileTranscriptionPipeline` の private serializer (`_build_srt` / `_build_translated_srt` / `_format_timestamp`) を module 関数として抽出したもので、pipeline の SRT 書き出しはこれらへ委譲 (出力はバイト単位で同一、`tests/core/transcription/test_srt_serializer.py` で固定)。

- `build_srt(subtitles, *, translated=False) -> str` / `write_srt(path, subtitles, *, translated=False) -> Path`
- `translated=True` は `translated_text` を持つ segment のみ出力 (index は renumber しない — 従来の翻訳 SRT と同一挙動)
- caller が `process_file(write_subtitles=False)` と組み合わせて出力先を制御できる (CLI `-o` / stdout 出力の基盤、入力横への不要な sidecar 生成を回避)

#### EngineMetadata: capability/quality metadata + `recommend()` API (Issue [#286])

`livecap_cli.engines.EngineMetadata` に **言語 × ハードウェアに最適な ASR engine を優先度順で推奨** する `recommend()` API を追加。動機は livecap-gui setup wizard ([gui#326](https://github.com/Mega-Gorilla/livecap-gui/issues/326))。従来 `get_engines_for_language()` は宣言順の list を返すのみで順位付けが無かった。

- **`EngineInfo` 拡張 (後方互換、末尾 default 付き field)**:
  - `quality_tier: Dict[str, LanguageQuality]` — 言語別の品質階層 (`best`/`good`/`fallback`) + 根拠レベル (`measured`/`model_card`/`heuristic`)
  - `vram_required_mb: Optional[int]` — GPU 推論の代表 VRAM 要件 (**VRAM 要件の正本**)。実測値は `docs/planning/issue-73` 由来 (parakeet 2417 / canary 6830 / voxtral 8923)。軽量/未計測/size 依存は `None` (filter で除外しない)
  - capability flags: `cpu_supported` / `cpu_recommended` / `gpu_recommended` / `realtime_on_cpu`
- **新規 public 型**: `LanguageQuality` (NamedTuple)、`ReasonCode` (str Enum、i18n 対応の根拠コード)、`EngineRecommendation` (NamedTuple: `engine_id`/`params`/`rank`/`quality`/`reason_codes`/`scores`)。`livecap_cli` と `livecap_cli.engines` から export
- **`EngineMetadata.recommend(language, gpu_available=False, vram_gb=None)`**:
  - `to_iso639_1()` で言語正規化 (既存 `get_engines_for_language` と parity)
  - hard 除外は「言語非対応」のみ。VRAM 超過は除外せず `EXCEEDS_VRAM` code + 低 score で沈める (wizard が全選択肢を提示 / gray-out できる)
  - sort は分解 score (quality → hardware_fit → latency → streaming) の辞書式多段。同 score の tie は登録順で安定 (「重いモデル優先」は不採用)
  - whispers2t は hardware に応じた `params["model_size"]` を返す (`create_engine(id, device, **params)` で一発起動)。他 engine は空 dict
- **VRAM 要件の正本を `EngineInfo` に一元化** — 専用 `model_vram.py` は作らず、Issue [#96] (VRAM 事前 check) を supersede
- **注意 (値の精度)**: `vram_required_mb` の一部 (parakeet_ja 2500 / qwen3asr None) と `quality_tier` の tier は `heuristic`/`model_card` タグ止まりの推定値。純追加 API で既定挙動は不変、精緻化は #86 benchmark / 別 PR で予定。qwen3asr は v1 draft の「30 言語すべて best」を過剰主張として **good** に是正
- **Tests**: `tests/core/engines/test_metadata.py` 新規 19 件 (recommend の言語/CPU-GPU/VRAM filter/rank/reason_codes/whispers2t params/後方互換)、torch 非依存

#### EnergyGate 限界寄与 ablation harness (Issue [#357])

`benchmarks/confidence_calibration/energygate_ablation.py` を追加。EnergyGate(#292)/ ConfidenceFilter(#334)/ 空text guard の 3 層を独立判定し、4 config (baseline/+energy/+confidence/both) で非音声抑制率・speech FRR・EnergyGate の marginal 寄与を測る。simulate ロジックは engine 非依存で単体 test 可能 (`tests/benchmark_tests/confidence_calibration/test_energygate_ablation.py` 13 件)。

結果 (`docs/research/energygate-effectiveness-2026-07.md`、星の王子さま JA corpus 1375 sample、3 engine): EnergyGate の必要性は **signal 依存だが Whisper の `no_speech_prob` だけが例外**。実際の信頼度量を出す engine (avg_logprob=reazonspeech unique=0 / token_confidence=parakeet_ja unique=1) では ConfidenceFilter が無音幻聴を捕捉し EnergyGate は品質冗長。一方 no_speech_prob engine (whispers2t) では ConfidenceFilter が無音幻聴を見逃し、**EnergyGate だけが 76/676 を捕捉 → 相補的で必要**。speech への false-drop は全 engine で 0 件。→ 既定 ON 維持が妥当。**production コード変更なし (調査のみ)**。

#### Confidence filter observe log: `is_interim` field 追加 (Issue [#351] PR 1)

`livecap_cli/transcription/confidence_filter.py:apply_filter()` に `is_interim: bool = False` kwarg を追加、 `FilterDecision` dataclass と `_decision_to_dict()` の JSON schema に `"is_interim": bool` field を追加。 observe mode log entry の interim path 由来 (`_transcribe_interim`) と final segment path 由来 (`_transcribe_segment` / async 版) を区別可能にする ([Issue #351](https://github.com/Mega-Gorilla/livecap-cli/issues/351))。

**Backward compat 完全維持** — default `False` で既存 caller は kwarg 未指定で動作、 legacy log (`is_interim` field なし) は parser 側で final 相当扱いとする compromise (caveat: legacy log の interim/final は本質的に区別不能)。

- **`apply_filter()` signature 拡張**: `is_interim: bool = False` kwarg 追加、 `FilterDecision` 構築時に反映
- **`FilterDecision` dataclass**: `is_interim: bool = False` field を末尾に追加 (frozen dataclass、 default 値で backward compat)
- **`_decision_to_dict()` JSON schema**: `"is_interim": bool` field を `"reason"` の後に固定位置で追加、 docstring example も更新
- **`stream.py` 3 call site から適切な値を明示 pass**:
  - `_transcribe_segment` (sync final): `is_interim=False`
  - `_transcribe_segment_async` (async final): `is_interim=False`
  - `_transcribe_interim` (interim): `is_interim=True`
- **Docs**: `apply_filter()` docstring に `is_interim` の意味論と backward compat / legacy log caveat を明記、 `benchmarks/confidence_calibration/README.md` の observe log JSON example に `"is_interim": false` 追加
- **Tests**: 11 新 test (`test_confidence_filter.py` に 8 = `TestFilterDecisionDataclass` +2 + `TestIsInterim` +6、 `test_stream.py` integration に 3 = sync final / async final / interim path 各 log field 検証)、 backward compat + interim/final 各 path の log field 検証、 全 268 pass in transcription/cli suite (退行ゼロ)

**下流の 別 PR (Issue #351 PR 2、 CLI merge 後)**: `benchmarks/confidence_calibration/parse_observe.py` の consumer 側対応 — default で `is_interim=True` entry を occurrence counter 前に除外、 `--include-interim` flag で opt-in。 これにより Layer 4 replay pipeline ([Task #393]) で production observe log を calibration に安全に使用可能に。

**関連**: Issue [#334] Finding F6 (Qwen3-ASR auto-detect fail-open) の calibration accuracy 向上、 [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) Layer 4 前提整備の 1 つ。

#### Confidence calibration corpus: OS 標準 data dir に永続化 default 化 (Issue [#338] follow-up)

`benchmarks/confidence_calibration/` の corpus directory を **`.tmp/calibration_corpus_full/` (session-local temp) から OS 標準 `user_data_dir` に移行**。 Phase 1/2 で `.tmp/` 配下に手動 mkdir が必要だった導入摩擦を解消、 コーパス構築後も次 sweep で自動的に再利用できるようにする。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

- **`benchmarks/confidence_calibration/pipeline.py:resolve_corpus_dir()` の semantics 変更**:
  - Before: env var 未 set 時 `None` return、 呼出側が `--corpus-dir` 必須 or `return 1` で fail
  - After: env var 未 set 時 **`_default_corpus_dir()` fallback** — `appdirs.user_data_dir("LiveCap", "PineLab") / "calibration_corpus"`
- **OS 別 default path**:
  - Windows: `%LOCALAPPDATA%\PineLab\LiveCap\calibration_corpus`
  - Linux: `~/.local/share/LiveCap/calibration_corpus` (or `$XDG_DATA_HOME/LiveCap/calibration_corpus`、 appauthor は `appdirs` 仕様上 Windows 専用)
  - macOS: `~/Library/Application Support/LiveCap/calibration_corpus`
  - appdirs 欠損 fallback: `~/.livecap/calibration_corpus` (`ModelManager` precedent)
- **`user_data_dir` (persistent) を採用、 `user_cache_dir` (`ModelManager` precedent) ではない理由**: corpus は user が build した label + Layer 2/3 augmented data + reports の集合で、 model cache (再 download 可) と異なり **再生成に時間がかかる persistent data**。 OS の cache 自動削除で消えるリスクを回避。
- **CLI default 変更 (backward compat)**:
  - `sweep.py --corpus-dir`: 未指定時 fail → OS default fallback + directory 存在確認 (存在しない場合は明確な error message + `return 1`)
  - `gen_esc50_non_speech.py --output-dir` / `gen_musan_noise.py --output-dir` / `gen_mixed_noisy_speech.py --output-dir`: `required=True` → `default=None` (env var + OS default fallback)
  - `build_corpus.py --output-dir`: sub directory (e.g. `<root>/ja_clean`) を指定する用途のため `required=True` のまま、 help に typical usage 例を追記
- **docs 更新** (canonical のみ、 research/* Phase 1/2 report は historical artifact として不変):
  - `benchmarks/confidence_calibration/README.md`: Stage 2 quickstart + Corpus 準備方針で OS default fallback を明示
  - `docs/architecture/core-api-spec.md`: 環境変数 table に `LIVECAP_CALIBRATION_CORPUS_DIR` 追加
  - `docs/reference/cli.md`: 環境変数 table に追加
  - `docs/testing/README.md`: env var table に追加
- **Tests**: `tests/benchmark_tests/confidence_calibration/test_pipeline.py:TestResolveCorpusDir` 4 test を新 semantics (fallback) 用に更新 (env unset → OS default、 空文字 → OS default、 tilde expand 既存挙動)、 新規 sentinel assertion (`resolved.name == "calibration_corpus"`) 追加。 `test_sweep.py:test_main_returns_error_when_no_corpus_dir` を `test_main_returns_error_when_corpus_dir_not_exists` にリネーム + semantics 変更 (env unset → directory 存在しない場合の fail-close verify)。
- **Migration**: 旧挙動を維持したい場合は明示的に `--corpus-dir <path>` を指定するか、 `LIVECAP_CALIBRATION_CORPUS_DIR=<path>` で override。 既存 `.tmp/calibration_corpus_full/` を使い続けたい user は env var を設定すれば従来通り。

#### Confidence threshold calibration harness — `sweep.py --breakdown-by` per-metadata-key 混同行列 (Issue [#338] Phase 6a)

Issue #338 Phase 5 の sweep 実行 **前に必須の infrastructure**。 現行 `sweep.py:measure_signals()` は `item.metadata` から `language` の 1 field のみ cherry-pick して `LabeledSample.metadata` に copy しており、 Layer 3 で追加した `snr_db` / `subtype` / `noise_source_dataset` 等が sweep report まで到達しなかった。 このため Phase 5 で 5 engine × ~1 hour GPU sweep を実行しても SNR 別 FRR / noise category 別 pass rate が計算不能で、 Issue #334 PR-4 の Pareto gate 「`clean_frr ≤ 1%` かつ `noisy_frr(SNR 別) ≤ 5%` かつ `known_probe を却下`」を validate できない状態だった。 本 PR は additive schema 拡張で 1 sweep から SNR 別 / subtype 別 / dataset 別の混同行列を全部取り出せるようにする。

- **`benchmarks/confidence_calibration/_core.py`** (拡張、 +100 行): `BreakdownReport` dataclass 追加 (`key` / `value_counts: dict[str, int]` / `sweep_by_value: dict[str, list[ThresholdMetrics]]`)。 `SweepReport.breakdown: dict[str, BreakdownReport]` field を default_factory=dict で **additive 追加** (未指定時は空 dict、 Phase 1 report / `parse_observe.py` output と backward compat 完全維持)。 `_breakdown_key(value) -> str` helper で `None` → `"__none__"` sentinel + float/int/str/bool の `str()` 変換を deterministic に処理。 `compute_breakdowns(samples, key, thresholds, direction) -> BreakdownReport` 関数で per-value サンプル分類 + 既存 `_confusion_matrix()` を各 bucket で reuse (stateless で安全)。 `sweep_threshold()` に `breakdown_by: Optional[list[str]] = None` param 追加、 `None` 時は 100% 現行挙動 fallback。 `report_to_dict()` に breakdown シリアライズ追加、 空時も `"breakdown": {}` として schema 上位互換維持。
- **`benchmarks/confidence_calibration/sweep.py`** (拡張、 +50 行): `measure_signals()` の `LabeledSample.metadata` 構築を **`item.metadata` full merge + 既存 3 key (`text` / `is_available`) override** に変更 (main path + error path 両方、 error 時も `snr_db` 等が保持されて後追い分類可能)。 `breakdown_list(value)` argparse type 新規追加 (comma-separated key + 空文字 / duplicate reject、 `positive_int` / `snr_list` の pattern 踏襲)。 `--breakdown-by` CLI flag 追加、 未指定時は空 list → `sweep_threshold(breakdown_by=None)` として backward compat。 対象 key が全 sample に存在しない場合は warning log で継続 (typo detection、 fail-close ではない)。 `SweepReport.metadata` に `"breakdown_by": [...]` を record して sweep report で実施 key を追跡可能。
- **設計 highlight** (Phase 6a plan D1-D10): **完全 additive schema** (Phase 1 report / PR #342 / `parse_observe.py` output を壊さない)、 `_confusion_matrix()` 再利用で計算誤り抑制、 `None` → `"__none__"` bucket で clean speech (`snr_db` field なし) を SNR 別 breakdown に含めても分類可能、 float 値の str 変換は Layer 3 output の全 integer float 保証で precision 問題回避、 O(N_samples × N_thresholds × N_breakdown_keys) の計算量で GPU 不要 (679 sample × 100 threshold × 3 key で < 1 sec)。
- **タイミング**: 本 PR **は Phase 5 sweep 実行前に merge 必須**。 逆順だと 5 engine × ~1 hour GPU sweep を再走する必要が発生する。 本 PR merge 後、 user は `--breakdown-by snr_db,subtype,noise_source_dataset` 付き 1 回の sweep 実行で Phase 2 report / PR-4 Pareto gate の全 evidence を取得可能。
- **Production runtime 影響**: なし (`livecap_cli/` に `breakdown` / `BreakdownReport` / `compute_breakdowns` / `breakdown_by` / `breakdown_list` 一切 import されない、 grep verify 済)。 dev 依存追加なし (numpy 既存 dep のみ、 stdlib で完結)。
- **テスト**: 新規 33 test (`test_core.py` +21 = `TestBreakdownKey` 5 / `TestComputeBreakdowns` 11 / `TestSweepThresholdWithBreakdown` 4 / serialization empty breakdown 1、 `test_sweep.py` +12 = `TestBreakdownList` 8 / `TestMeasureSignals` full metadata pass-through + error path 2 / `TestMainE2E` breakdown E2E + backward compat 2)。 既存 374 test 全 retain、 zero regression、 full calibration suite total 407 pass。 Phase 1 report / #342 の JSON schema 互換 test で pin。

#### Confidence threshold calibration harness — Layer 3 SNR-mixed noisy_speech corpus CLI (Issue [#338])

PR #343 codex-review 2nd round で確認された **Layered evaluation** framework の Layer 3 (Layer 1 clean baseline + Layer 2 ESC-50/MUSAN hard negative の下流) として、 clean speech に Layer 2 noise を目標 SNR で混合した `label=noisy_speech` corpus を生成する CLI を追加。 Issue #334 PR-4 の Pareto gate `noisy_speech_frr by SNR ≤ 5%` に対する direct evidence 収集用で、 default threshold を下げた際に production の実 mic で背景 noise 混在会話が過剰 reject される trade-off を SNR 別に定量化する土台。

- **`benchmarks/confidence_calibration/_mix_snr.py`** (新規、 ~130 行): RMS-based SNR mixing helper。 `mix_at_snr(speech, noise, snr_db)` は closed-form scale = `sqrt(P_speech / (P_noise * 10^(snr_db/10)))` で合成し、 `match_length` は tile / truncate で noise 長を speech に合わせ、 `check_and_renorm` は `|mix|.max() > 1.0` 時に 0.95 peak へ scale down (全 sample 均一 scaling で SNR ratio 保持)。 numpy-only、 `lhotse` / `torchaudio` 依存追加なし、 `compute_snr_db` は back-compute で test 内実測 verify に使用。
- **`benchmarks/confidence_calibration/gen_mixed_noisy_speech.py`** (新規、 ~360 行、 CLI): default 50 speech × 5 SNR = 250 mixed entry 生成。 `--snr-db-list "-5,0,5,10,20"` (default、 comma-separated float + NaN/inf reject + **raw / formatted duplicate reject** の `snr_list` argparse type) / `--samples` (`positive_int` 再利用) / `--noise-datasets esc50,musan` (default) / `--speech-language ja` (default、 **output manifest entry の `language` は speech-language を継承**、 別引数だと mismatch で `sweep.py --filter-by-language` を汚染するため独立 `--language` は排除) / `--force` で `source_dataset="layer3_mix"` の safe re-augment / `--dry-run`。 speech は `label=speech + language=<speech-language>` filter → filename sort 先頭 N 件、 noise は Layer 2 output (`source_dataset in {esc50, musan}`) から deterministic rotation (`noise_pool[i % len]`)、 同一 speech を全 SNR で mix (paired within-subject 比較で SNR effect 純粋化)。 CLI 起動時に prerequisite check (`manifest.jsonl` 存在 / speech 数十分 / Layer 2 output 非空) で loud fail、 error message で `build_corpus` / `gen_esc50_non_speech` / `gen_musan_noise` を明示誘導。
- **Manifest schema additive**: Phase 2 の 17 field を保持し、 Layer 3 で `snr_db: float` / `noise_source_dataset: "esc50"|"musan"` / `noise_source_file: str` / `noise_source_path: str` の 4 field を additive 追加。 既存 entry (field なし) は `pipeline.load_calibration_corpus()` が `CalibrationCorpusItem.metadata: dict` にそのまま格納するため backward compat 完全維持 (verify 済)。 `label="noisy_speech"` は `_core.py:_normalize_label()` で `"speech"` 扱い → filter reject 時に FRR contribution となり confusion matrix に統合 (既存 test で pin 済)。
- **Output layout (multi-language 対応)**: filename pattern `{speech_stem}_snr{db_str}dB_{noise_subtype}.wav` (例: `segment_0000_snr10dB_clapping.wav`、 負 SNR は `snr-5dB`)、 `{corpus_root}/{speech_language}_noisy_speech/` に配置 (JA なら `ja_noisy_speech/`、 EN なら `en_noisy_speech/`)、 JA と EN を同 corpus で augment しても path 衝突なし。 既存 `.tmp/` gitignore rule で保護。 `--speech-language` は language の **single source of truth** (clean speech filter + output entry `language` field + output subdir 全て一元制御)。
- **Sweep report metadata gap** (Phase 6a で解消済): Layer 3 (本 entry) 時点では `sweep.py:measure_signals()` は `item.metadata` から 3 field (`text` / `language` / `is_available`) のみ cherry-pick しており、 `snr_db` は sweep report まで到達しない gap が残った状態で merge した。 **Phase 6a (PR #345) で `measure_signals()` を `item.metadata` full pass-through + `sweep.py --breakdown-by` CLI flag 追加として解消** され、 1 sweep から SNR 別 / subtype 別 / noise_source_dataset 別の混同行列を report で取得可能に。
- **License**: 出力音声は derivative (clean speech 朗読 + Layer 2 noise)、 raw audio は `.tmp/` 配下のみ、 production runtime 依存追加なし (numpy 既存 dep のみ)。
- **テスト**: 新規 74 test (`test_mix_snr.py` 37 / `test_gen_mixed_noisy_speech.py` 37) で SNR 精度 ±0.5 dB (5 SNR × 3 signal pair pin) / length matching / clip renormalization / determinism / edge case (zero-power / empty) / CLI 引数 validation (`snr_list` NaN/inf reject、 `positive_int` 0 reject) / prerequisite fail case / E2E augment + real SNR back-computation verify を pin。 実 dataset なしで synthetic sine + white noise で covered。 既存 288 test 全 retain、 zero regression。

#### Confidence threshold calibration harness — ESC-50 / MUSAN augmentation CLIs (Issue [#338] Phase 2)

Phase 1 report ([`docs/research/calibration-japan-engines-2026-07.md`](docs/research/calibration-japan-engines-2026-07.md)) の最重要 caveat (synthetic silence + low-level noise では production の applause / laughing / engine 等より easier で、 data-driven threshold が probe を pass してしまう) を解消するため、 **ESC-50** (CC BY-NC 4.0 / dev-only) と **MUSAN noise** (CC BY 4.0 / dev-only) を calibration corpus に augment する CLI 2 本を追加。 Issue #334 PR-4 (default threshold 変更) の直接入力となる Phase 2 report 執筆に load-bearing。 なお本 CLI は Layered evaluation の **Layer 2** (production-realistic non_speech hard negative augmentation) に位置し、 PR-4 default threshold 確定には Layer 3 (SNR-mixed noisy_speech corpus、 別 PR) + Layer 4 (production observe replay) + Layer 5 (VAD + noise gate + confidence_filter + ASR realtime E2E) の follow-up が必要 (PR #343 codex-review 方針 review 反映)。

- **`benchmarks/confidence_calibration/_augment_common.py`** (新規、 ~250 行): ESC-50 / MUSAN 共通の resample (16 kHz mono、 `pipeline._resample_to_16k_mono` 再利用) + deterministic uniform-stride chunking (default 1.5 sec × up to 3-5 chunks) + manifest upsert (`build_corpus._load_manifest_entries` / `_write_manifest` を import 再利用、 `source_dataset` filter で選択的削除 support) + optional dataset download (SHA-256 検証 support)。
- **`benchmarks/confidence_calibration/gen_esc50_non_speech.py`** (新規、 ~280 行、 CLI): ESC-50 の 15 production-realistic category (`laughing` / `sneezing` / `coughing` / `breathing` / `clapping` / `footsteps` / `rain` / `door_wood_knock` / `mouse_click` / `keyboard_typing` / `clock_tick` / `glass_breaking` / `engine` / `car_horn` / `siren`) から default 10 file/category × 3 chunk = ~450 sample augment。 `--categories` で override、 `--samples-per-category` で件数調整、 `--force` で safe re-augment、 `--dry-run` で preview、 `--download` (~600 MB) 自動 fetch。 `meta/esc50.csv` deterministic 選択 (filename sort、 先頭 N 件)。 音楽 (BGM) は controversial として除外。
- **`benchmarks/confidence_calibration/gen_musan_noise.py`** (新規、 ~250 行、 CLI): MUSAN の `noise/{free-sound,sound-bible}/` sub-directory から default 50 file × up to 5 chunk = ~150-250 sample augment。 `music/` と `speech/` は意図的に除外 (music は BGM 判断が別問題、 speech は false positive)。 `--samples` で file 選択総数 (uniform stride、 deterministic)、 `--max-chunks-per-file` で file 当たり chunk 数、 `--force` で safe re-augment、 `--dry-run` で preview、 `--download` (~11 GB) 自動 fetch。
- **Manifest schema additive fields**: Phase 2 augmented entry は Phase 1 の 14 field に加えて `source_dataset` (`"esc50"` / `"musan"`)、 `source_file` (元 filename、 attribution)、 `source_license` (`"CC BY-NC 4.0"` / `"CC BY 4.0"`) の 3 field を持つ。 既存 entry (field なし) は `pipeline.load_calibration_corpus()` が `CalibrationCorpusItem.metadata: dict` にそのまま格納するため forward-compat 完全維持。
- **License safety** (Plan D1): 両 dataset とも raw audio data のため production code から import しようがなく、 `.tmp/` 配下は既存 `.gitignore` rule で git push 事故を物理的に防止。 CI では dataset unavailable のため実行 skip、 Phase 4 (user 環境で dataset download + augment) と Phase 5 (5 engine 再 sweep) の後 Phase 2 report を生成し PR-4 threshold candidate を最終確定する運用。
- **テスト**: 新規 56 test (`test_augment_common.py` 24 / `test_gen_esc50_non_speech.py` 16 / `test_gen_musan_noise.py` 16) で resample / chunking / manifest upsert / force filter / dry-run / CLI 引数 / license attribution / determinism / forward-compat を pin。 実 dataset download なしで synthetic mini-corpus fixture で覆う。

#### Confidence threshold calibration harness — kana-level alignment metric (Issue [#338] PR-γ)

PR-β (Stage 2) で発覚した **表記揺れ起因の偽 low coverage** 問題への対応として、
calibration alignment metric に **kana-level coverage** を並列追加。`pykakasi`
(`GPL-3.0-or-later`、dev 限定) で kanji / katakana を hiragana に正規化し、
acoustic confidence (音素列が正しいか) と lexical surface form (kanji 変換差) を
分離。Phase 4 smoke verify で観測された「1人で vs 一人で」 (text 0.95) / 「サハラ砂漠
vs さはらさばく」 (text 0.21) 等の表記差を kana 化で吸収しつつ、「真っ先 vs さっき」
のような真の音響誤認識は低 score として保持。`sweep.py` は **不変** (PR #340 codex-
review 3rd round の scope minimization 訂正反映、kana ベース sweep は Phase 4 で
必要性確認後の別 PR に分離)。

- **`benchmarks/confidence_calibration/_normalize_jp.py`** (新規): `pykakasi`
  (GPL-3.0-or-later) と `kanjize` (MIT) の lazy import + ``to_hiragana()`` +
  ``normalize_for_alignment()`` (NFKC → CJK 隣接の Arabic 数字 run → kanjize
  で 漢数字化 (`1200 → 千二百`、 `1人 → 一人`) → hiragana → 句読点 strip)。
  両 lib とも **dev 限定 import**、 production runtime は一切 import しない
  (`tests/test_production_no_pykakasi.py` で parametrize した grep guard)。
  正規化は PR #341 codex-review で 4 段階を経た: v1 blanket mask (`一人` と
  `二人` を同一視する false-high) → v2 per-char canonical substitution
  (compound `千二百` を `10002100` と誤変換) → v3 kanji→arabic via
  `kanji2number` (compound numeral は OK だが `一緒` / `十分` / `一番` /
  `一人` 等の idiom で pykakasi の自然な読みを壊す) → **v4 arabic→kanji via
  `number2kanji`** で全方位対応 (idiom は無変更、 EN の `Chapter 1` 等も
  CJK 非隣接なので無変更、 cross-form `1人 ↔ 一人` は kanjize で kanji 化
  後 pykakasi の compound rules で `ひとり` に統一)。
- **`benchmarks/confidence_calibration/build_corpus.py`**: 新規
  ``compute_alignment_score_kana()`` を ``compute_alignment_score()`` と並列に
  追加 (既存関数の signature / 挙動は **不変**)。build_corpus main loop で text +
  kana を並列計算、manifest entry に 3 つの kana field を additive で追記:
  ``alignment_score_kana`` / ``reference_text_matched_kana`` /
  ``transcribed_text_kana``。
- **`benchmarks/confidence_calibration/recompute_alignment.py`** (新規): 既存
  Phase 4 manifest を **audio 再 transcribe なしで** kana metric に migrate する
  CLI。``--manifest`` + ``--reference-text-ja`` / ``--reference-text-en`` +
  ``--force``、idempotent (既に kana field がある entry は skip)、forensic safe
  (text-level field を一切変更しない)。
- **`tests/test_production_no_pykakasi.py`** (新規): livecap_cli/ の .py に
  pykakasi 文字列が現れないことを static grep で assert。本 repo の
  AGPL-3.0-only と pykakasi GPL-3.0-or-later の dev 限定整合性を CI で常時
  guard。
- **`pyproject.toml`**: ``[project.optional-dependencies] dev`` に
  ``pykakasi>=2.3.0`` を追加 (`yt-dlp` と同 group、`uv sync --extra dev` で
  インストール)。runtime ``dependencies`` には一切影響なし。
- **`benchmarks/confidence_calibration/README.md`**: §4.5 (任意) section
  追加、kana metric の動機 / recompute_alignment quickstart / license note。
- 新規 test: ``test_normalize_jp.py`` (28) + kana 専用 test in ``test_build_corpus.py``
  (7) + ``test_recompute_alignment.py`` (13) + production guard (1) = 49 件追加。

#### Confidence threshold calibration harness — Stage 2 (Issue [#338] PR-β)

PR-α (Stage 1) で landed した signal-agnostic sweep core (`_core.py`) を base に、
**user 提供 audio corpus から直接 calibration する Stage 2 CLI** を追加。Stage 1
は observe log 経由、Stage 2 は audio + engine.transcribe() を直接呼ぶ active
calibration。Issue #334 PR-2 / PR-3 / PR-4 の 1-2 月待ちを **~2-3 週** に短縮可能。

- **`benchmarks/confidence_calibration/sweep.py`**: Stage 2 CLI、corpus loader
  (`pipeline.load_calibration_corpus()`) → 各 sample で `engine.transcribe()` →
  `engine_confidence` の signal 抽出 → `_core.sweep_threshold()` で sweep。
  `--engine-kwargs key=value` で engine 個別 kwargs (例: `use_int8=true`) 対応、
  `--quantization` / `--filter-by-language` を report metadata に embed。
- **`benchmarks/confidence_calibration/build_corpus.py`**: corpus build CLI、
  yt-dlp で audio download (URL / local file 両対応) + ffmpeg で 0:06 trim +
  16kHz mono 変換 + Silero VAD で speech segment 切り出し + 各 segment で
  Whisper transcribe + `difflib.SequenceMatcher.find_longest_match()` で 原稿
  fuzzy match (**coverage** = matched substring 長 / transcribed 長、0.0-1.0)
  → `manifest.jsonl` 生成。**idempotent + upsert** (path 単位で同 entry を
  rewrite、`--force` で再生成しても重複なし、他 source の entry は保持)。CLI
  に `--engine-kwargs` 追加 (alignment 用 WhisperS2T の重い default
  `model_size=large-v3` を CPU 環境向けに `model_size=base` 等に override 可能、
  PR #340 review 反映)。`SequenceMatcher` の **autojunk=False を明示**
  (Phase 4 smoke verify で発覚、default `autojunk=True` は長文 reference +
  頻出 char で partial match に縮小される bug、両言語 coverage 値を大幅
  改善: JA 高一致 6/15 → 9/15、EN 1/24 → 12/24)。
- **`benchmarks/confidence_calibration/_vad_chunker.py`**: Silero VAD probability
  stream → speech segment list の pure logic (threshold + hysteresis + min/max
  duration)。`SileroVAD.process()` の 32ms frame probability を sliding 評価、
  `min_silence_sec` 連続 silence で boundary 確定、`max_segment_sec` 超過は
  均等 split。
- **`docs/research/calibration-corpus-sources.md`**: JA / EN リトル・プリンス
  Chapter 1 corpus (user 提供) + PD alternative (Common Voice ja / JSUT /
  LibriSpeech / ESC-50) URL 一覧、著作権 caveat (raw audio は repo 外、private
  利用に限定)。
- **`docs/contributor/adding-an-engine.md` §5** 更新: threshold calibration template
  から本 harness の invoke 例を link (Issue #334 PR-6 で landed した template を
  operational 化、AP-3 量子化 verify の手動 procedure を automated harness で代替)。
- **`benchmarks/confidence_calibration/README.md`** 拡張: Stage 2 quickstart
  (JA / EN Chapter 1 corpus build → 5 engine sweep の流れ)、`--engine-kwargs`
  + metadata embed の説明。
- **`pyproject.toml`**: `[project.optional-dependencies] dev` に `yt-dlp>=2024.0.0`
  追加 (runtime 不要、build_corpus 用)。

Design (Plan D1, D2, D3, D6, D7, D8, D9):
- VAD chunking は Silero probability + simple boundary detection (forced
  alignment lib 追加せず、`difflib` で代替)
- Audio resample は PR-α `pipeline._resample_to_16k_mono` を import 再利用
- Engine.transcribe() は 1 sample 1 回のみ呼ぶ (cache、sweep は cached value)
- Calibration target: ReazonSpeech (P0、int8/float32 両方) / Qwen3-ASR (P1、
  ja+en) / Parakeet_ja (P2) / WhisperS2T (P2)

Tests (`tests/benchmark_tests/confidence_calibration/`): 73 (Stage 1) + 56 new
(Stage 2) = **129 passed** (vad_chunker: 16、build_corpus: 27、sweep: 13)。実
model + 実 yt-dlp / ffmpeg 不要 (MockEngine + subprocess.run mock + 合成 audio
で全 unit test)。

#### Confidence threshold calibration harness — Stage 1 (Issue [#338] PR-α)

新規 `benchmarks/confidence_calibration/` sub-package を追加。observe mode で
蓄積した JSON log + user 提供 label から threshold sweep を実行する CLI
tooling (Stage 1)。Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334)
PR-2 / PR-3 / PR-4 (observe mode 1-2 月運用に依存) を ~1-2 週に短縮する path。

- **`benchmarks/confidence_calibration/_core.py`**: signal-agnostic な sweep
  logic。confusion matrix (TP/FP/TN/FN)、F1 / precision / recall / Youden's J、
  false_reject_rate を計算。direction (`reject_if_less` / `reject_if_greater`)
  と criterion (`f1` / `youden_j` / `precision` / `recall`) を arg 化、
  `LabeledSample` で input 一般化、`SweepReport` で output 標準化。
- **`benchmarks/confidence_calibration/parse_observe.py`**: Stage 1 CLI。
  ``confidence_filter[observe]: <JSON>`` の jsonl + user 提供 `labels.jsonl`
  (source_id → label mapping) を input、`_core.sweep_threshold()` 経由で
  sweep report を生成。
- **`benchmarks/confidence_calibration/pipeline.py`**: `manifest.jsonl`
  corpus loader、`LIVECAP_CALIBRATION_CORPUS_DIR` env var pattern (既存
  `LIVECAP_NON_SPEECH_CORPUS_DIR` を踏襲)、16 kHz mono float32 への
  自動 resample。
- **`benchmarks/confidence_calibration/README.md`**: Quickstart + signal
  direction / confusion matrix の解釈 / criterion 選択指針 / corpus 準備方針。

**Stage 2 (PR-β、未実装)**: user 提供 audio corpus + engine.transcribe() で
直接 active calibration、yt-dlp + Silero VAD + 原稿 fuzzy match による
corpus build helper を提供予定。

**design 判断**:
- 既存 `benchmarks/non_speech_filter/sweep.py` の argparse + grid sweep
  canonical pattern を踏襲、sibling として位置付け
- `_core` を共通化、Stage 1 / Stage 2 は input 経路のみ異なる (DRY)
- code 挙動の変更ゼロ — 既存 `FilterConfig` / engine 実装は touch しない、
  新 harness のみ追加 (Issue #334 PR-4 で本 output を活用して default を update)

**Tests**: 40 passed (`tests/benchmark_tests/confidence_calibration/`)、
unit test 中心、実 model 不要、Mock + synthetic data で confusion matrix /
edge case / log parse / corpus loader を pin。

#### Qwen3-ASR auto-detect fail-open warning (Issue [#334] Finding 6)

`StreamTranscriber.__init__` で、`filter_config.mode != "off"` かつ engine が
**Qwen3-ASR + `language=None` (auto-detect)** の組合せの時に `logger.warning(...)`
で 1 回通知する。auto-detect path (`Qwen3ASREngine._transcribe_via_wrapper_fallback`)
は ``engine_confidence`` 全 None で fail-open するため filter mode "on" でも
実質無効になる現象を、**programmatic API 利用者** が早期に気付けるようにする。

- **検出 logic**: duck typing (`engine.engine_name == "qwen3asr"` + `engine._asr_language
  is None`) で識別、`isinstance` は循環 import / Mock false negative 回避のため不採用。
- **発火条件 matrix**:
  - `filter=on` + qwen3asr + `language=None` → **warn 1 回**
  - `filter=off` + 同上 → warn なし (filter 不要のため)
  - `filter=on` + qwen3asr + `language="Japanese"` → warn なし (filter active)
  - `filter=on` + 非 qwen3asr engine → warn なし
  - `filter=observe` + qwen3asr + `language=None` → warn 1 回 (filter active)
- **実装場所** (reviewer 2nd round 指摘): `Qwen3ASREngine.__init__` は `FilterConfig`
  を受けないため、両方を知る `StreamTranscriber.__init__` で警告するのが
  architectural separation 上正しい。
- **CLI default は `--language ja`** のため CLI 利用者は通常通り保護される。`language`
  引数を明示すれば warn は出ない (actionable message)。
- **Migration**: 既存 caller は変更不要。`language=None` で auto-detect mode を
  programmatic に利用していた user は warn を 1 回受け取る (silent fail-open の
  解消、行動は変更されない)。
- **Tests** (`tests/transcription/test_qwen3_warn.py`、5 test):
  warn 発火条件 matrix を pin、`MockQwen3LikeEngine` で `Qwen3ASREngine` の
  identifying attribute を模擬。

#### Utterance lifecycle observation hook (Issue [#332])

`StreamTranscriber` の post-processing 経路には 5 種類の silent drop
(filter reject / energy_gate / engine error / engine 空 text / empty audio)
があり、interim 字幕を出した後 final が drop されると consumer 側 state
が clear されず残置する問題があった (livecap-gui#362、ReazonSpeech
`avg_logprob ≈ -0.2` 境界で正常音声断片の false reject が頻発)。本機能で
1 論理 utterance の処理確定 (emit / drop どちらでも) を観測する callback を
追加し、consumer が `emitted=False` 時に interim state を clear できるよう
にした。

- **`on_utterance_settled` callback** を `StreamTranscriber.set_callbacks`
  に追加 (3 番目の kwarg、optional)。`**kwargs` swallow なし、未知 kwarg は
  `TypeError` で即時 fail (policy「不要な後方互換は廃する」、pre-1.0
  cleanup 系列と整合)。
- **`UtteranceSettledEvent` dataclass** (`livecap_cli.transcription.utterance`、
  top-level `livecap_cli` から re-export): 5 field
  (`emitted` / `reason` / `source_id` / `utterance_start_time` /
  `utterance_end_time`)、`frozen=True`。
- **`REASON_*` 静的 reason 定数** (`Final[str]`、public re-export):
  - `REASON_EMPTY_AUDIO = "segment:empty_audio"`
  - `REASON_ENERGY_GATE = "energy_gate:low_rms"`
  - `REASON_FILTER_REJECT = "confidence_filter:reject"` ← GUI #362 主因
  - `REASON_ENGINE_EMPTY = "engine:empty_text"`
- **動的 reason**: `engine_error:<ExceptionType>` (例:
  `"engine_error:RuntimeError"`)。`raise EngineError(...) from e` で chain
  された場合は `__cause__` の型名、chain なし (`__cause__ is None`) の
  場合は `EngineError` 自身の型名 (`"NoneType"` 出力を回避)。
- **Tier 1 の 7 hook point** が settled event を発火: empty_audio /
  energy_gate / filter reject / engine_empty / engine_error /
  coalescer push emission (per output、0-2 件) / coalescer flush emission
  (periodic / force / finalize)。
- **Delivery ordering**:
  - `feed_audio` (callback path): `on_result` 完了 **後** に
    `on_utterance_settled` 発火 (同期実行、stack frame 内で順序保証)
  - `transcribe_async` (async generator): `yield` **直前** に発火
    (yield 後の code は caller が次の `__anext__()` を呼ぶまで実行されない
    ため、break で永久未発火を回避)
  - `finalize` (list return): list append **直前** に発火 (generator path
    と整合)

Consumer example:

```python
from livecap_cli import (
    StreamTranscriber, UtteranceSettledEvent, REASON_FILTER_REJECT,
)

def on_settled(event: UtteranceSettledEvent) -> None:
    if not event.emitted and event.reason == REASON_FILTER_REJECT:
        gui.clear_interim()  # consumer 側 state を即時 clear

transcriber.set_callbacks(
    on_result=on_result,
    on_interim=on_interim,
    on_utterance_settled=on_settled,
)
```

Migration: 既存 caller は不変 (default `on_utterance_settled=None`、
発火コストゼロ)。新 consumer は `set_callbacks` で opt-in。

Out of scope (別 issue で defer): interim path informational reject signal、
coalescer periodic flush で utterance なし時の event、multi-source 内部統合、
`coalescer:discarded` reason (現行実装に該当 branch なし)。

#### NoiseGate `PEAK_SAFETY_MARGIN_DB` user-tunable (Issue [#327])

`analyze_noise_samples` の `suggested_threshold_db` 計算に
`peak_safety_margin_db` keyword 引数を追加、CLI `levels` コマンドに
`--noise-gate-margin <dB>` flag を追加。`engine_min_rms_margin_db` (#292)
と並列の API 対称性を回復。

- **`analyze_noise_samples(peak_safety_margin_db=...)`** (keyword-only):
  `suggested_threshold_db = peak_p95_db + peak_safety_margin_db`。
  default = `PEAK_SAFETY_MARGIN_DB = 6.0`、既存呼び出しは bit-identical。
- **CLI `levels --noise-gate-margin <dB>`**: user が任意の margin を渡せる。
  - 高 SNR studio コンデンサーマイク (AT4040、SM7B 等、self-noise <15 dBA):
    `2` 〜 **負値** (例: `-5` で `suggested = peak_p95 - 5`、`peak_p95 ≈
    -60 dB` の AT4040 で `-65 dB` が得られる)
  - 高ノイズ環境 / 低品質 USB マイク: `10` 程度
- **CLI 出力**: 旧 hardcoded `(= peak_p95 + 6.0)` を user value 反映
  `(= peak_p95 + {margin})` に変更、user が `--noise-gate-margin -5` を
  渡した時に正確な値を表示。

scope: `transcribe` には flag 追加しない (現状 auto-calibration なし、
parse-only effectless flag は anti-pattern。auto-calibration mode は
別 issue 候補)。

Workflow (既存 `levels → --noise-gate-threshold` 手動 pass を維持):

```pwsh
# AT4040 等 studio mic
livecap-cli levels --mic 0 --duration 10 --noise-gate-margin -5 --json
# → {"suggested_threshold_db": -64.6, ...}
livecap-cli transcribe --realtime --mic 0 --noise-gate --noise-gate-threshold -64.6
```

#### Engine confidence signal schema (Issue [#308] PR-A.0 / Phase 1 Layer 3)

- **`EngineConfidence` / `TranscriptionResult` dataclasses** added to
  `livecap_cli/engines/base_engine.py`:
  - `EngineConfidence`: `Optional[float]` fields for `no_speech_prob`,
    `avg_logprob`, `compression_ratio`, `token_confidence_mean`, plus a
    `raw: dict[str, float]` overflow bucket for engine-specific signals.
    `is_available` property returns `True` when at least one signal field
    is non-`None` (PR-A.1 filter precondition).
  - `TranscriptionResult`: `text`, `confidence`, `engine_confidence`
    (default = all-None `EngineConfidence()`). `__iter__` yields
    `(text, confidence)` so the legacy `text, confidence = result` tuple
    unpacking pattern continues to work — no caller migration required
    for existing engine adapters.
  - Both dataclasses are `frozen=True` (immutable) and re-exported via
    `livecap_cli.engines.__all__`.
- **Engine adapter expose paths** (engine-by-engine breakdown):
  - `whispers2t_engine.py`: extracts `no_speech_prob`, `avg_logprob`, and
    `compression_ratio` from the CTranslate2 backend result dict via the
    new pure-function `_extract_engine_confidence()`. Handles both
    top-level signals (current backend shape) and per-segment lists
    (legacy / future shape). Real-machine smoke verify (`normal_speech_neko`
    vs `desk_tap` vs `applause_5_claps`): `no_speech_prob` = 0.036 (speech)
    vs 0.63-0.66 (non-speech) — clean separation usable by the PR-A.1
    filter. Existing `confidence: float` calculation untouched.
  - `parakeet_engine.py`: hybrid `EncDecHybridRNNTCTCBPEModel` is now
    explicitly switched to the CTC decoder via the new
    `_configure_decoding_with_confidence()` helper (see "Changed"
    section below). Real-machine `token_confidence_mean` separation:
    0.01-0.10 (speech) vs 0.0000029-0.0003 (non-speech) — 3-4 orders
    of magnitude, usable by the PR-A.1 filter with a `> 0.005` threshold.
  - `reazonspeech_engine.py`: returns `EngineConfidence()` (all `None`)
    with a docstring `Note` explaining that sherpa-onnx Python bindings
    for transducer models do not expose per-token scores. Users who
    require engine-level hallucination defense are pointed to Silero /
    TenVAD backends. Real-machine smoke verify confirmed
    `is_available is False` on all corpus clips.
  - `qwen3asr_engine.py`, `voxtral_engine.py`, `canary_engine.py`,
    `benchmarks/non_speech_filter/mock_engine.py` (`MockEngine`): no-op
    migration — return `TranscriptionResult(text=..., confidence=...)`
    with default empty `EngineConfidence`. PR-A.1 filter will treat
    these engines as fail-open (`is_available is False` → pass-through).
- **Caller migration** (defensive `hasattr`-based dispatch retained for
  legacy `Tuple[str, float]` mocks):
  - `shared_engine_manager.py` `_process_request` uses `hasattr(result,
    'text')` primary branch, keeps tuple/dict legacy branches.
  - `benchmarks/non_speech_filter/mock_engine.py` `InstrumentedEngine`
    accepts both `TranscriptionResult` and the historical tuple shape so
    benchmark harnesses keep working unmodified.
  - `livecap_cli/transcription/stream.py` `TranscriptionEngine` Protocol
    return type updated to `EngineTranscriptionResult` (runtime import
    alias from `engines.base_engine` — kept out of `TYPE_CHECKING` so
    `typing.get_type_hints()` resolves it). Stream call sites at lines
    546 / 618 / 767 use `text, confidence = ...` unpacking and continue
    to work via the dataclass `__iter__`.
- **New unit tests** (do not require ASR models — pure-function pins
  the extraction logic):
  - `tests/core/engines/test_engine_confidence_schema.py` (17 cases):
    default values, `is_available` semantics across all four signal
    fields, frozen-mutation rejection, `__iter__` yields exactly two
    items (engine_confidence excluded from tuple unpacking), public
    re-export coverage.
  - `tests/core/engines/test_whispers2t_confidence_extraction.py`
    (17 cases): mock CTranslate2 result dicts covering top-level + per-
    segment mean aggregation, missing fields, non-numeric / `None`
    values, and non-dict segment entries.
  - `tests/core/engines/test_parakeet_confidence_extraction.py`
    (12 cases): `FakeHypothesis` mock pinning the token-confidence
    primary path and edge cases (string input, empty list, non-numeric
    values, completely empty hypothesis).
  - `tests/core/engines/test_parakeet_return_hypotheses.py` (5 cases):
    pins that `return_hypotheses=True` is passed to NeMo and that the
    legacy `score / len(y_sequence)` fallback is no longer populated.
  - `tests/core/engines/test_parakeet_decoding_strategy.py` (5 cases):
    pins hybrid-model detection, CTC switch via `decoder_type='ctc'`,
    fallback to strategy-only on `TypeError`, and exception resilience.
  - `tests/transcription/test_transcription_engine_protocol.py`
    (2 cases): pins that `typing.get_type_hints()` resolves
    `EngineTranscriptionResult` to `engines.base_engine.TranscriptionResult`
    rather than the `transcription.result` dataclass of the same name.

The new signals feed into PR-A.1 (`--confidence-filter {off,observe,on}`
post-filter) and PR-A.3 (calibration + production default). Together with
PR-B calibration (PR [#304]) and the PR #307 audio-filter-reference
rewrite, this lands the Phase 1 Layer 3 schema required to close Issue
[#295].

#### Noise Gate & Calibration ([#278], [#279], [#280], [#281])

- `livecap_cli.audio.NoiseGate` — 音量ベースのリアルタイムノイズゲート（サンプル単位エンベロープフォロワー、numba JIT で < 0.1 ms / 100 ms chunk）。VAD 前段に挿入してハルシネーションを抑制。
- `transcribe` サブコマンドに `--noise-gate` / `--noise-gate-threshold` / `--noise-gate-attack` / `--noise-gate-release` オプションを追加。
- `livecap-cli levels` サブコマンド — マイク入力レベルを dB 単位でリアルタイム表示し、環境ノイズから推奨閾値を算出。
  - `--duration N` — N 秒後に自動停止（非対話モード）。
  - `--json` — `NoiseAnalysis` を JSON で stdout に出力（GUI / スクリプト連携向け）。
- `livecap_cli.audio.analysis` モジュール — `NoiseAnalysis` dataclass と `analyze_noise_samples()` 関数（CLI / GUI 共通キャリブレーション API）。
- 推奨閾値アルゴリズム: `noise_peak (95%ile) + 10 dB`（[livecap-gui PR #294](https://github.com/Mega-Gorilla/livecap-gui/pull/294) の実測に基づく保守的マージン）。「死のゾーン」(`noise_floor ± 5 dB`) を回避する設計。

**段階導入について**: PR #281 は **キャリブレーション API 基盤の先行導入** (Issue #280 C-3 + C-4) です。NoiseGate 本体の安定化 ([Issue #280](https://github.com/Mega-Gorilla/livecap-cli/issues/280) の C-1 ヒステリシス + C-2 hard-mute) は follow-up PR で提供予定。現行実装 (単一閾値 + `-60 dB` soft-mute) では、閾値が speech peak 付近の場合に flicker で逆にハルシネーションを誘発することがあります。特に `whispers2t` エンジンで影響が大きく、`reazonspeech` / `parakeet_ja` / `qwen3asr` は影響を受けにくいことが A/B テストで確認されています (PR #281 comments 参照)。暫定対応として、低 SNR 環境では `levels` の推奨値より保守的な値の使用、または別エンジンの利用を推奨します。

#### Phase 6: CLI Subcommand Structure ([#74], [#201])

New CLI with subcommand architecture:

| Command | Description |
|---------|-------------|
| `livecap-cli info` | Display installation diagnostics |
| `livecap-cli devices` | List audio input devices |
| `livecap-cli engines` | List available ASR engines |
| `livecap-cli translators` | List available translators |
| `livecap-cli transcribe` | Transcribe audio (file or realtime) |

**transcribe options:**
- `<file> -o <output.srt>` - File transcription to SRT
- `--realtime --mic <id>` - Realtime microphone transcription
- `--translate <id> --target-lang <lang>` - Translation support
- `--vad <auto|silero|tenvad|webrtc>` - VAD backend selection
- `--engine <id>` - ASR engine selection
- `--device <auto|gpu|cpu>` - Device selection

**Package extras:**
- `recommended`: Google translation (deep-translator)
- `all`: All optional dependencies

#### Phase 5: Engine Optimizations ([#73], [#194], [#196], [#197])

- Template Method pattern for `BaseEngine` with standardized lifecycle
- Progress reporting during model loading (0-100%)
- Model memory caching for faster subsequent loads
- Library preloading for reduced import time
- Standardized cleanup and resource management

#### Phase 4: Translation Support ([#72], [#180], [#181], [#182], [#184], [#186])

**Translators:**
- `google` - Google Translate ([#180])
- `opus_mt` - Helsinki-NLP Opus-MT local models ([#181])
- `riva_instruct` - NVIDIA Riva Translate 4B Instruct ([#182])

**Features:**
- Context-aware translation with sentence buffering
- `StreamTranscriber` translation integration ([#184])
- `FileTranscriptionPipeline` translation integration ([#186])
- Configurable timeout via `LIVECAP_TRANSLATION_TIMEOUT`
- Async translation deadlock prevention ([#189])

#### Phase 3: Package Structure ([#71])

- Reorganized module structure under `livecap_cli/`
- Clear separation: `engines/`, `vad/`, `transcription/`, `translation/`
- Unified public API exports in `__init__.py`

#### Phase 2: API Unification ([#70])

- `TranscriptionResult` dataclass replacing `TranscriptionEventDict`
- `VADConfig` dataclass for VAD parameters
- `EngineFactory.create_engine(engine_type, device, **options)` API
- Consistent error handling with `TranscriptionError`, `EngineError`

#### Phase 1: Realtime Transcription ([#69], [#65], [#66], [#67], [#68])

**Core components:**
- `StreamTranscriber` - VAD + ASR streaming orchestration ([#65])
- `VADProcessor` - Pluggable VAD with state machine ([#66])
- `TranscriptionResult` / `InterimResult` - Unified result types ([#67])
- `AudioSource` / `FileSource` / `MicrophoneSource` - Audio abstraction ([#68])

**VAD backends:**
- Silero VAD (default, neural network-based)
- WebRTC VAD (fast, low memory)
- TenVAD (optimized for Japanese)

**Language optimization:**
- `VADProcessor.from_language("ja")` - Auto-select optimal VAD
- Benchmark-based presets for Japanese and English

#### File Transcription

- `FileTranscriptionPipeline` for batch processing
- SRT subtitle output format
- FFmpeg integration for audio extraction
- Translation integration for bilingual subtitles

### Changed

#### 重複した棚卸し行を削除し、SymbolTable 境界を実モデル行へ一本化 (Issue [#387] PR D、epic [#380])

**棚卸し 48 → 47 行 / applicable 45 → 44 行 / 未確定 6 → 5 行** (実測で確定は **39 行**のまま)。[#387] の所有は 1 → **0 行**になった。

- **Removed**: `engine.reazonspeech.sherpa_narrow_path_signature` 行。**独立した production 境界ではなかった。**
  - **Before**: 「実モデル行の cheap tier 裏付け」として 4 variant を回し、`covers_boundary=False` で未確定に留めていた (測定限界と再評価 trigger を記録済み)
  - **After**: 行ごと削除。`engine.reazonspeech.sherpa_from_transducer` と**同じ `from_transducer()` 呼び出し**が対象で、`tokens=` はその引数である
  - **4 variant の pass は情報を持っていなかった** — `2026-09-02b` の観測では control を含む 5 通りすべてが `mentions_parse_failure=true` / `mentions_open_failure=false` で**完全に同一**だった。不正な ONNX が `tokens.txt` より先に検証されるため、**path に依存する情報が出力に現れない**
  - **Migration**: なし (テストハーネス内部)
- **Removed**: `sherpa.from_transducer.diff` probe と、唯一の利用者が消えた `write_invalid_onnx()` / `write_tokens_txt()`。行を消すと `test_probe_ids_are_all_referenced` が probe 本体の削除まで要求する
- **Changed**: `engine.reazonspeech.sherpa_from_transducer` に `required_variants=("cjk_kana", "outside_acp")` を設定した。**削除だけでは裏付けが痩せる** — 削除した行は (境界に届かないまま) 4 variant を回していたのに対し、実モデル行は real_model tier の既定である**代表 1 variant しか記録が無かった**。`ユーザー` は cp932 の内側なので、SymbolTable が narrow path へ戻っても日本語 Windows なら通ってしまう
- **Changed**: sherpa-onnx bump 時の再測定 trigger を実モデル行へ移した。上流が narrow path へ戻れば decode が token を返さなくなり、probe が `ProbeSkipped` で落ちる
- **Note**: **「測れなかったので諦める」でも「有効な ONNX を用意して格上げする」でもない。** 前者は closed な issue を追跡先に持つ未確定行を残し、後者は**既存 real-model probe の重複実装**になる。棚卸しモデル上の重複だったので削除した

#### `FFMPEG_BINARY` / `FFPROBE_BINARY` の env export を削除 (Issue [#387] PR C、epic [#380])

**監査の結論は「削除」だった。** [#387] で唯一 production コードを変更する PR である。

- **Removed**: `FileTranscriptionPipeline._initialise_ffmpeg_environment()` の `os.environ.setdefault("FFMPEG_BINARY", ...)` / `("FFPROBE_BINARY", ...)`。
  - **Before**: パイプラインの初期化時に、解決済みの ffmpeg / ffprobe path をプロセス env へ流していた
  - **After**: 流さない。`livecap_cli` 自身は使っておらず、実際の抽出は `ffmpeg.run(stream, cmd=self._ffmpeg_path, ...)` で **path を直接渡している**
  - **Migration**: moviepy 経由で本 package の ffmpeg を拾っていた host は、`FFMPEG_BINARY` を自分で設定するか `moviepy.config` を明示指定すること。**ただし moviepy を本 package の初期化より先に import していた場合は元々効いていない** — moviepy は `os.getenv("FFMPEG_BINARY", "ffmpeg-imageio")` を **module import 時に 1 度だけ**読むため
  - **`FFPROBE_BINARY` は読み手が 1 つも無かった。** venv 全体を grep しても 0 件で、moviepy が読むのは `FFMPEG_BINARY` と `FFPLAY_BINARY` だけである。棚卸し行の「消費者は pydub / moviepy 系」という記述も誤りで、**pydub はこれらを読まない**
- **Removed**: 上記に伴い**読み手のいなくなった連鎖**を削除した — `_initialise_ffmpeg_environment()` の `resolve_probe()` 呼び出しと `shutil.which("ffprobe")` fallback、`__init__` の `self._ffprobe_path` 属性、未使用になった `import os`。`file_pipeline` は `ffmpeg.probe()` を呼ばないので、**ffprobe を解決する意味がそもそも無かった**
  - `FFmpegManager.resolve_probe()` は**残している** — 公開 API であり、manager は ffprobe を DL / 検証して他の consumer へ提供している
- **Changed**: 棚卸しから `transcription.file_pipeline.ffmpeg_env_export` の行を削除し、**`resources.ffmpeg_manager.path_env` を新設**した。
  - 行の削除は [#375] PR 2 / PR 3 で `utils.unicode_safe_*` を消したときと同じ扱いである (呼び出し箇所が無くなった行は表からも消す)
  - **ただし「env へ path を流す箇所が無くなった」わけではない** — `FFmpegManager._finalise_environment()` は Windows で **`PATH` の先頭へ bin ディレクトリを挿している**。この実在の境界に行が無いままだと、次に読む人が「env へ path を流す箇所は無い」と誤読する。`candidate_method=②wide-path` / source-check で新設した (`PATH` は str、消費側は `CreateProcessW`。解決後の実行ファイル path は `transcription.file_pipeline.ffmpeg_binary` が実測済み)
  - **棚卸しは 48 行 / applicable 45 のまま**で、**実測で確定 38 → 39 / 未確定 7 → 6** になった。削除 1・追加 1 で行数は相殺され、追加した行を実測で確定させたぶん確定数が増えた
- **Added**: `ffmpeg.path_env` probe と、それによる `resources.ffmpeg_manager.path_env` の **②wide-path 確定** (4 variant すべて pass)。**`PATH` に挿した非 ASCII の bin ディレクトリから、basename だけで実行ファイルを解決できるか**を測る (`shutil.which()` で **staged 側が解決された**ことも確認する — システムの ffmpeg を拾っていたら非 ASCII を一度も通っていない)。**probe は production の `_finalise_environment()` を直接呼ぶ** — 手書きで同じ mutation を再現すると、**production 側の挿入条件が壊れても probe は pass し続ける** (レビュー指摘。変異で確認: 挿入を止めると probe が落ちる)。公開入口の `configure_environment()` は `ensure_executable()` 経由で**ダウンロードを起こし得る**ので使わない — `_finalise_environment()` 自体は PATH を触るだけで I/O が無い。**Windows 限定**なので他 OS では skip する。`transcription.file_pipeline.ffmpeg_binary` は**フルパスを渡して**起動するので PATH 探索を通らず、**あちらの pass では代用できない** (レビュー指摘)。`cjk_kana` / `outside_acp` を必須にした
- **Added**: `benchmark_results/nonascii/2026-09-02b/results.json` — clean tree (`c56aa9a`) から全 tier を 1 セッションで生成した証拠 (55 passed, skip 0 / 129 レコード / 40 probe)。**`ffmpeg.path_env` を新設し、さらに probe を production の関数経由へ直したので取り直した**
- **Added**: env export 削除の回帰テスト (`TestFfmpegEnvIsNotExported`)。`FFMPEG_BINARY` / `FFPROBE_BINARY` が設定されないこと、host の値を上書きしないこと、解決した path は引き続き抽出へ使われることを固定する。stub の `resolve_probe()` は**呼ばれたら `AssertionError`** を投げるので、ffprobe 解決の再導入も検出する (変異で確認: export を戻すと 1 件だけ落ちる)

#### CI の FFmpeg 取得をバージョン固定 + チェックサム検証へ (Issue [#395])

PR #394 の Windows CI が **gyan.dev の 503 だけで 2 job 落ち**、既定の PR ゲートがブロックされた件への対応。調査の結果、単発の障害ではなく `.github/actions/setup-livecap-ffmpeg` の構造的な弱点だった。**`livecap_cli/` の production code は変更しない** (runtime 側の同種問題は [#398])。

- **Before**: Windows は `ffmpeg-release-essentials.zip` (ローリング URL) を `Invoke-WebRequest` 1 回。Linux は `curl -L` 1 回で `--fail` 無し。チェックサム検証もリトライも無く、**両 OS で違う FFmpeg バージョンをテスト**していた。キャッシュは 5 workflow 中 1 つだけ、しかも広い `restore-keys` 付きで、**バージョンを上げても古いキャッシュが復元され続ける**状態だった
- **After**: 両 OS とも **ffbinaries v6.1 に固定**し、アーカイブ 4 資産と展開後バイナリ 4 つの **SHA-256 を manifest に固定して検証**する (この manifest は後に [#398] で `livecap_cli/resources/ffmpeg_manifest.json` へ移り、runtime と共有する単一の正本になった)。リトライ・分類・検証は `setup_ffmpeg.py` の 1 実装を両 OS が共有し、**再試行するのは timeout / DNS・transport error / 408 / 429 / 5xx のみ** (指数バックオフ)。permanent 4xx とチェックサム不一致は fail loud。キャッシュ (`restore-keys` 無しの exact key) の所有者を action へ移し、**既定の PR ゲートも初めてキャッシュの恩恵を受ける**
- **検証はバイトの出所に依存しない** — cache restore / self-hosted の永続ディレクトリ / 新規ダウンロードのいずれでも、存在・SHA-256・実行可能性・期待バージョンを同じ経路で検査する。これにより **self-hosted に残った古いバイナリも検出され再取得される** (従来は workflow 側の「`ffmpeg.exe` があれば action をスキップ」ゲート 5 箇所により、一度も検証されなかった)
- **取得元と期待ハッシュは毎回記録される** — 固定 URL・期待 archive SHA-256・期待 binary SHA-256 は manifest 由来なので、ダウンロードが発生しない cache hit でもログと job summary に出る。取得経路は `source` (`download` / `cache` / `existing directory`)、キャッシュ状態は `cache-state` (`hit` / `miss` / `disabled`) として別々に報告する (`cache: 'false'` は miss ではない)
- **Migration**: workflow から `Check FFmpeg existence` ゲートと `Cache ffmpeg-bin` step を削除済み。新たに action を使う場合、ワークスペース外の永続ディレクトリを使う job では `cache: 'false'` を渡す (検証は引き続き実行される)
- **同一バージョン ≠ 同一挙動**: Linux と Windows のビルドは `configuration:` が異なる。固定はあくまで**比較可能性**を上げるものである

あわせて `core-tests-windows.yml` の `LIVECAP_FFMPEG_BIN: "${{ github.workspace }}\ffmpeg-bin"` を修正。YAML の二重引用符では `\f` が**フォームフィード (U+000C)** になるため、action が配置した直後にテストへ存在しないパスを渡していた。

#### CLI file mode に NoiseGate を接続 + `AudioPreprocessor` 注入点 (Issue [#366] Phase 3)

#366 の最終実装 Phase。NoiseGate parity には「VAD 判定と ASR 入力が同じ処理後音声を見る」ことが必要 (file pipeline は時刻範囲で元音声を slice するため、segmenter 前処理だけでは VAD 判定のみ処理後になる)。

- **Before**: file mode では `--noise-gate` 系 6 option を警告して無視
- **After**: `--noise-gate` 指定時、NoiseGate が **VAD 前処理として音声全体へ 1 回**適用され、VAD 判定・EnergyGate・ASR 入力の**すべてが処理後音声**を見る (realtime と同一の意味論)。有効化時は resolved 値 (open/close threshold・floor・attack/release) を stderr 表示。**既定 (off) のため未指定ユーザーの挙動は不変**
- **Migration**: 従来どおり無加工で処理する場合は `--noise-gate` を指定しない
- **実装**: `FileTranscriptionPipeline(audio_preprocessor=...)` 注入点を新設 (`AudioPreprocessor = Callable[[np.ndarray, int], np.ndarray]`、公開 export)。戻り値契約は fail-fast (`np.ndarray` / 1 次元 / shape・dtype とも入力と同一 = float32)、preprocessor の例外・契約違反は **file-level failure** (#362 経路で `process_files` が failed result へ変換)。CLI は **per-file factory** (ファイル毎に新 NoiseGate 生成) で `process_files` のファイル間・例外後の状態非共有を構造的に保証 — pipeline は `reset()` の暗黙契約を持たない
- **Tests**: 注入点契約 11 件 (厳密 1 回 / segmenter・ASR 同一配列 / identity 等価 / fail-fast 5 種 / file-level failure) + NoiseGate の全配列一括処理と chunk 逐次処理の bit-identical 等価性 + CLI 8 件 (per-file 生成 / 状態隔離 / **層の合成**: 静音 + hard-mute + EnergyGate で ASR 未呼出 / resolved 値ログ)。realtime NoiseGate suite は無改修 green (realtime 経路のコード変更なし)

#### CLI file mode で EnergyGate・confidence filter を既定有効化 + `SegmentTranscriber` 契約拡張 (Issue [#366] Phase 2)

Phase 1 (VAD 分割) に続き、EnergyGate と confidence filter を file mode に接続。realtime と同一の判定式を共有 module (`should_drop_low_energy` / `apply_filter`) で使う。

**1. `SegmentTranscriber` 契約拡張 (公開契約)**

- **Before**: `Callable[[np.ndarray, int], str]` — str のみ
- **After**: `str | SegmentOutcome` を受理。**str 返却は意味不変で継続受理** (既存 caller は無改修で動作)。`SegmentOutcome(text, drop_reason, asr_called)` は caller 側 filter の判定結果を運び、reason 別統計が `metadata["drop_counts"]` (新設、常時格納) に集計される。`asr_calls` は「engine を実際に呼んだ数」の意味を厳密化 — EnergyGate drop (`asr_called=False`) は数えず、confidence reject / engine empty は数える。drop は `empty_results` / `asr_errors` に混ざらない。全 segment drop でも `success=True` (全滅判定 #362 は不変 — 「gate drop + 残り全例外」は正しく `success=False`)
- **Migration**: 統計が必要な caller のみ `SegmentOutcome` へ移行。`drop_reason` の語彙は realtime と共通の `REASON_*` 定数

**2. file mode で filter 群が既定実効 (CLI)**

- **Before**: `--engine-min-rms` 系 / `--confidence-filter` / `LIVECAP_CONFIDENCE_FILTER` は file mode で warning 付き無視
- **After**:
  - **EnergyGate** (既定 `-45.0` dBFS) が VAD 分割された**各 segment 単位**で実効 — threshold 未満は **`engine.transcribe()` を呼ばずに** drop (`--engine-min-rms off` で無効化)。判定式は realtime と単一の `should_drop_low_energy` (strict `<`、equality pass、`-inf` は energy 計算ごと skip)
  - **confidence filter** (既定 `on`) が ASR 後に実効 — mode 3 種 + `LIVECAP_CONFIDENCE_FILTER` env の precedence を realtime と共有。observe log の `source_id` は入力 file path (複数ファイルの区別が可能)
  - drop があれば stderr に `Dropped segments: <reason>=<count>, ...` を表示
  - `LIVECAP_CONFIDENCE_FILTER` の file mode warning (#363) を撤去 (env が実効になったため)
- **Migration**: 従来の file mode 挙動 (filter なし) が必要な場合は `--engine-min-rms off --confidence-filter off`。無音区間が drop され字幕が減る場合は `Dropped segments:` の stderr 表示と `metadata["drop_counts"]` で診断可能。signal を出さない engine では confidence filter は fail-open (realtime と同一)

**3. Added**: `SegmentOutcome` (`livecap_cli` / `livecap_cli.transcription` から export、不変条件を `__post_init__` で強制) / `should_drop_low_energy` (`livecap_cli.audio` — realtime `StreamTranscriber._should_skip_low_energy` も同 helper へ委譲、挙動不変) / `metadata["drop_counts"]`

- **Tests**: helper 単体 7 件 (equality pass / `-inf` skip 未計算 / validation) / pipeline 正規化集計 12 件 (混在ケースの期待値固定、gate drop + 全例外 → `success=False`、legacy 空と structured 空の別勘定) / CLI 7 件 (無音 file で `transcribe` 未呼出 / per-segment 評価 / observe caplog + source_id / env precedence) / realtime `TestEnergyGate` は無改修 green (委譲の受け入れ条件)

#### CLI file mode の VAD 分割を既定有効化 + 注入 segmenter の空 fallback 廃止 (Issue [#366] Phase 1)

file mode は音声全体を 1 segment として ASR に渡しており長尺 file で品質・メモリに不利だった。`--vad` を file mode に接続する (#366 Phase 1)。

**1. file mode の VAD 分割 (CLI)**

- **Before**: `--vad` は realtime 専用 (file mode では warning 付き無視)。file transcription は常に音声全体を 1 segment として処理
- **After**: file mode は**既定 (`--vad auto`) で VAD により音声区間を分割**してから ASR へ渡す (resolved language (#365) の preset を適用)。音声セグメント 0 件の場合は **exit 0・出力は空** (`-o` 時は空 SRT) + stderr に `No speech segments detected.`。新設の **`--vad off`** で従来の全音声 1 segment 処理へ opt-out。`--realtime --vad off` は**モデルロード前に明確なエラー** (realtime は VAD 必須 — VAD なし realtime は別機能)
- **Migration**: 従来の file mode 挙動が必要な場合は `--vad off` を指定。無音 file が「空 SRT + exit 0」になる点に依存する script は `segmentation_empty` metadata / stderr 表示で判別可能

**2. `FileTranscriptionPipeline` の注入 segmenter 空 fallback 廃止 (公開契約)**

- **Before**: 注入 segmenter が `[]` を返すと**全音声 1 segment へ fallback** — VAD が正しく無音判定した場合ほど全音声が ASR へ流れ hallucination を招く逆転があった
- **After**: 注入 segmenter の `[]` は「音声セグメントなし」= **ASR 呼び出しゼロ** (`success=True` / `subtitles=[]` / metadata `segmentation_empty=True`)。`segmenter=None` (未注入) の全音声 fallback は従来どおり
- **Migration**: 全音声 1 segment 処理が必要な caller は `segmenter=None` で構築する

- **Tests**: `tests/vad/test_file_segmenter.py` 新規 7 件 (MockVADBackend で torch-free: final のみ採用 / 全無音 `[]` / finalize 回収 / reset lifecycle / 例外後回復 / 複数ファイル非持越)、`TestSegmentationEmpty` (pipeline 契約 3 件)、`TestVadFileMode` (CLI 7 件: off / no-speech semantics / realtime off 拒否 / preset 連携)、実 Silero integration 3 件
- **Docs**: `docs/reference/cli.md` (--vad 行 / file mode note / realtime-only 一覧から --vad 除去)、`docs/reference/api.md` (segmenter 意味論 + `VADFileSegmenter`)

#### 言語データ正本の一元化 + `supported_languages` の不変化 + `cli_default_language` 改名 (Issue [#230])

engine の対応言語データが metadata と adapter に二重定義されており、[#365] で `EngineMetadata.resolve_language()` が CLI の受理判定の正本になったことで、乖離が「対応言語の誤拒否 / 非対応言語の誤受理」という correctness バグに昇格していた。また `EngineMetadata.get()` が内部 `EngineInfo` を直接返すため、外部の list 変更が受理判定を書き換え可能だった (実証済み)。

- **Before**:
  - canary / voxtral / reazonspeech / parakeet の `get_supported_languages()` と metadata が独立 hardcode (一致は偶然維持)。qwen3asr は engine 内こそ #229 で単一化済みだが metadata の 30 言語は独立 hardcode
  - `EngineInfo.supported_languages` は mutable `List[str]` — `EngineMetadata.get("canary").supported_languages.append("ja")` の後 `resolve_language("canary", "ja")` が誤受理
  - field 名 `default_language` (constructor default と混同しやすい)
- **After**:
  - **言語データの正本を engine 特性別に一元化**: canary / voxtral / reazonspeech / parakeet(_ja) は `EngineMetadata` 正本 (adapter は defensive copy を返す) / whispers2t は `whisper_languages.py` 正本 (従来どおり) / qwen3asr は新設 `qwen3asr_languages.py` (data-only) 正本で metadata と adapter の両方が派生。全 engine の対応言語の**集合・順序は不変**
  - `EngineInfo.supported_languages` を `__post_init__` で **tuple 化** (構築時は list 可) + **`EngineInfo` 自体を frozen dataclass 化** (field 再代入 `info.supported_languages += (...)` の経路も封鎖)。qwen3asr の正本 map は **`MappingProxyType`** (item 代入で adapter だけが新言語を受理する split-brain を防止)、`SUPPORTED_LANGUAGES` class 属性も tuple。`EngineFactory.get_engine_info()` の `supported_languages` は毎回独立した list copy
  - `EngineInfo.default_language` → **`cli_default_language`** に改名 (未リリース field のため単純 rename)。CLI の未指定時ポリシーであり **constructor default との一致は要求しない**ことを明文化 (qwen3asr は意図的に constructor=auto / CLI=ja — PR-A.5.2)
- **Migration**: `supported_languages` を list として mutate していた外部 code は copy を取ること (`list(info.supported_languages)`)。`get_engine_info()` の戻り値は従来どおり `List[str]` (ただし独立 copy)。`default_language` 参照は `cli_default_language` へ (同一 Unreleased 内の rename)
- **Tests**: `tests/core/engines/test_language_authority.py` 新規 (adapter↔metadata 一致 ×5 / data module 同源 ×2 / qwen3asr 意図的差 pin / auto adapter のモデルロードなし変換)、`TestLanguageDataAuthority` (正本派生・golden 順序・tuple 不変・factory copy 独立性)
- **Docs**: `adding-an-engine.md` に「言語データの正本規約」(hardcode 禁止・パターン表・新 engine 追加時のテスト行追加) を新設

#### CLI `--language` を全 engine に適用 + 言語解決の一元化 (Issue [#365])

従来 `--language` は **qwen3asr にのみ** routing され、whispers2t は常に constructor default (`ja`) で動作 (英語音声を ja 設定で認識する silent 精度劣化)、canary は `en` 固定、voxtral は明示指定不能、単一言語 engine (reazonspeech/parakeet/parakeet_ja) は不一致を silent 無視していた。#363 で file 経路が実働化したため実害化していた。

- **Before**:
  - parser default `--language ja`。qwen3asr 以外の engine へは渡らない
  - 非対応言語・不正コード・単一言語 engine の不一致は silent 無視 (誤設定のまま認識)
  - VAD preset は常に args 生値 (実質 ja) で選択
- **After**:
  - parser default を **未指定 sentinel (None)** に変更し、`EngineMetadata.resolve_language(engine_id, requested)` が起動時に一度だけ engine 別へ解決。**未指定時の実効言語は全 engine 現状維持** (whispers2t/qwen3asr/reazonspeech/parakeet_ja: ja、canary/parakeet: en、voxtral: auto)
  - 明示指定は BCP-47 → primary language subtag 正規化 (`ja-JP` → `ja`、`yue-HK` → `yue`) + `supported_languages` 検証。非対応言語・不正形式コード・auto 非対応 engine への `auto`・単一言語 engine の不一致は**モデルロード前に exit 1** (silent fallback しない)
  - resolved 値を engine kwargs (whispers2t/canary/voxtral/qwen3asr へ routing)・VAD preset・翻訳 `source_lang`・起動ログ (`Language: requested=..., resolved=...`) へ一貫配布
  - `--translate` 併用時は resolved が具体言語であることを必須化 (voxtral の未指定/auto は拒否。default whispers2t + 翻訳は resolved=ja のため従来どおり動作)
- **Migration**: 言語未指定の利用は無影響。従来 silent 無視されていた不正組合せ (`--language ja --engine canary` / `--language en --engine reazonspeech` / `--language auto --engine whispers2t` 等) はエラーになるため、正しい言語または対応 engine を指定すること。realtime の VAD preset が engine 別 resolved 値で選択される (canary 未指定時は en preset 等。preset 不在言語は default VAD へ fallback)
- **Tests**: `TestLanguageResolutionMetadata`/`TestResolveLanguage` (metadata、engine 別解決表を固定)、`TestLanguageResolutionE2E` (CLI file 経路 9 件: routing / fail-fast 4 種 / 翻訳併用 2 種 / 起動ログ)、pin テスト 2 件を新契約へ書き換え (誤った「whisper は VAD 経由」docstring を是正)、VAD auto-fallback (integration)

#### CLI `transcribe <file>`: 出力・翻訳失敗の意味論を仕様化 (Issue [#363])

CLI file 経路の復旧 (上記 Fixed) に伴い、出力先と失敗時の観測可能な挙動を確定。旧実装は全滅していたため実利用上の互換影響はないが、pre-1.0 policy に従い Before/After を記録する。

- **Before** (旧コードの意図した挙動、実際には未動作):
  - `-o` はファイルモード時必須扱い (docs 記載)。未指定時は `[start - end] text` 形式の独自行を stdout 出力
  - 翻訳失敗時の仕様なし
  - realtime 専用オプションをファイルモードで指定しても silent no-op
- **After**:
  - `-o` は任意。**未指定時は SRT content を stdout へ出力** (進捗・警告は stderr のみ — pipe/リダイレクトで安全)
  - `--translate` 指定で**全 segment の翻訳が失敗** → exit 1 + 出力ファイル非生成 (**原文 SRT への silent fallback はしない**)。**一部失敗** → 翻訳成功 segment のみ出力 + 件数付き stderr warning (segment index は元の値を維持)
  - realtime 専用オプション (`--vad` / noise-gate 系 / EnergyGate 系 / transient 系 / `--confidence-filter` / `--mic`) を**既定値から変更して**指定した場合、および `LIVECAP_CONFIDENCE_FILTER` 設定時に stderr warning (「既定値と同値の明示指定」の検出は [#366])
  - file mode は VAD 分割なしで音声全体を 1 segment として処理 (pipeline の従来契約どおり。segmenter 接続は [#366])
- **Migration**: `-o` 必須を前提にした script はそのまま動作。stdout を parse していた script は SRT 形式に更新すること (旧形式はそもそも全滅バグで出力されなかったため実影響なし)

#### FileTranscriptionPipeline: 全 ASR segment 例外を `success=False` に格上げ (Issue [#362])

livecap-gui v3.1.0 の「常に空 SRT」障害 ([gui#392](https://github.com/Mega-Gorilla/livecap-gui/issues/392)) の増幅要因を修正。`FileTranscriptionPipeline._transcribe_segments` は segment 単位の transcriber 例外を fail-soft で握り潰すため、**全 segment が例外** (契約不整合・モデル破損等の異常状態) でも `FileProcessingResult(success=True)` + 0 byte SRT を正常出力し、障害が caller (GUI) から不可視だった。segment 単位 fail-soft は維持しつつ、「全滅」と「部分失敗」を区別する集計を導入。

- **Before**:
  - 全 ASR 呼び出しが例外でも `success=True` で完走し、空 SRT を書き出す (既存 SRT があれば空で上書き)
  - `metadata` に ASR 呼び出し件数の内訳なし
  - `process_files` の `error_callback` は `process_file` が例外を**送出**した場合のみ発火
- **After**:
  - ASR 呼び出しが 1 件以上かつ全件例外 → **`success=False`**、`error` に `"All N ASR segment calls failed; first error: <Type>: <msg>"`、**`output_path=None` で SRT を新規作成も上書きもしない**
  - `metadata` に `asr_calls` / `asr_errors` / `empty_results` を**成功・失敗を問わず常時**格納 (caller の警告表示・診断用)
  - `process_files` は `success=False` **返却**時も `error_callback(error_msg, None)` を発火 (送出経路と契約を一貫)
  - 部分失敗 (例外 < 全件) と正常な空 (無音・全件正常空認識・ASR 呼び出し 0 件) は従来どおり `success=True` (gui#392 レビュー合意: 件数ヒューリスティック誤発火の回避)
  - `FileTranscriptionCancelled` は従来どおり集計せず再送出
- **Migration**: `result.success` のみを見る caller は挙動改善のみ (全滅が正しく失敗として見える)。全滅時の 0 byte SRT 生成に依存する caller は想定なし。`error_callback` を「送出例外専用」として使っていた caller は `exc=None` のケース (返却された failure) を許容すること。
- **Tests**: `tests/core/transcription/test_file_pipeline_outcome.py` 新規 12 件 (全滅/既存 SRT 保護/部分失敗/3 種混在/全件正常空/ASR 0 件/Cancelled 再送出/metadata 常時格納/process_files callback 契約)、engine/torch/FFmpeg 不要
- **Docs**: `docs/reference/api.md` — FileTranscriptionPipeline 使用例の `engine.transcribe(audio, sr)[0]` を `.text` に修正 (#314 以降 `[0]` は TypeError、gui#392 と同種の誤実装を誘発するサンプルだった) + 「失敗の意味論」節を新設。`docs/reference/feature-inventory.md` — `error_callback` の `Exception | None` 契約・metadata 件数内訳を反映、実在しない `config=` 引数をサンプルから除去

**関連**: [Issue #362](https://github.com/Mega-Gorilla/livecap-cli/issues/362) / gui#392 (GUI 側 hotfix v3.1.1) / [#314] (`TranscriptionResult.__iter__` 削除 — 障害トリガー) / [#363] (CLI file 経路の API 乖離 — 別 issue、本修正の scope 外)

#### Calibration harness: `parse_observe.py` が interim entry を default 除外 (Issue [#351] PR 2)

[Issue #351](https://github.com/Mega-Gorilla/livecap-cli/issues/351) の PR 2。 PR 1 ([#352] で observe log entry に `is_interim` field 追加) の **consumer 側対応**。 `benchmarks/confidence_calibration/parse_observe.py` が `is_interim` field を認識し、 threshold sweep から **interim path 由来 entry を default で除外** (final のみで calibration) するよう変更。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

**背景**: `parse_observe_log()` は従来、 同一発話の interim n 回 + final 1 回を独立 sample として counting していた。 interim は蓄積途中の temporary UI feedback で threshold tuning の「正解」ではないため、 final のみを使うのが正しい。 Layer 4 replay pipeline ([Issue #338] Task) で production observe log を calibration に使う前提整備。

- **`parse_observe.py:parse_observe_log()` の default 挙動変更**:
  - `include_interim: bool = False` param 追加、 default で `is_interim=True` entry を除外
  - **occurrence 採番の critical fix**: interim 除外は occurrence counter increment の**前**に行う。 counter を先に進めてから除外すると final の occurrence が `1, 3, 5...` と skip 交じりになり、 user の `occurrence_index` label (final 前提) が match しなくなる
  - `skipped_interim` counter + `logger.info` 報告 (observability、 silent drop 回避)
  - sample metadata に `is_interim` field 追加 (`--include-interim` 時の下流分析用)
- **`LogEntry` / `parse_log_line()`**: `is_interim` field 追加、 他 optional field と同じ `.get(..., default)` pattern で読む (key 欠落は final 扱い)
- **`main()` CLI**: `--include-interim` flag で opt-in (interim hallucination filter tuning 用 advanced analysis)

**Migration (advanced analysis 用途)**:

```bash
# default: interim 除外、 final のみで sweep (推奨)
uv run python -m benchmarks.confidence_calibration.parse_observe \
    --log observe.log --labels labels.jsonl --engine reazonspeech --signal avg_logprob

# opt-in: interim も含める (occurrence_index label は final-only 前提だと mismatch、 caveat あり)
uv run python -m benchmarks.confidence_calibration.parse_observe \
    --log observe.log --labels labels.jsonl --engine reazonspeech --signal avg_logprob \
    --include-interim
```

**Tests**: 8 新 test (`TestIsInterimConsumer`) 追加 — `parse_log_line` の is_interim 読取 (True/False/key 欠落→False の parse robustness)、 default final-only 除外、 `--include-interim` opt-in で全 entry 保持、 **occurrence 採番が final entries 内で連続 (interim skip 後 0, 1)** の critical verify、 skipped_interim logging、 sample metadata、 CLI flag E2E。 全 53 pass in `test_parse_observe.py` (元 45 + 新規 8)、 退行ゼロ。 `_make_log_line` helper は PR 1 schema に合わせ `is_interim` を常時 emit。

**関連**: 上流 PR 1 ([#352]、 CLI schema 追加)、 [Issue #338](https://github.com/Mega-Gorilla/livecap-cli/issues/338) Layer 4 replay pipeline の前提、 [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) Finding F6 (Qwen3-ASR auto-detect) の calibration accuracy 向上。

#### Confidence filter 既定閾値を Phase 2 report 反映で更新 (Issue [#334] PR-4)

[Phase 2 report](docs/research/calibration-japan-engines-phase2-2026-07.md) で Layer 2 (ESC-50/MUSAN hard negative) + Layer 3 (SNR-mixed noisy_speech) 込みの augmented corpus 1375 sample を用いて 5 engine を再 calibration した結果、 **Pareto gate 「`clean_frr ≤ 3%` かつ `noisy_frr(SNR≥5) ≤ 5%` かつ known probe reject」適用値** を新 default として採用。 [Issue #334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit の **main deliverable**。 主要効果: **ReazonSpeech で現 default `-0.2` の FRR 42.5% 実害を新 default `-0.40` で 5.4% に改善** (Phase 2 report §2.1 実測)。 全 engine で Pareto gate を実測 evidence に基づき採用。

- **`livecap_cli/transcription/confidence_filter.py:FilterConfig` の default 4 値を Phase 2 recommended に変更**:
  - `no_speech_threshold`: `0.5 → 0.71` (WhisperS2T、 Pareto relaxed_B、 Whisper 公式 0.6 近傍、 Phase 2 report §2.3)
  - `token_conf_threshold`: `0.005 → 0.001` (Parakeet_ja Pareto strict pass、 F1=0.961、 false reject 39→11 = 72% 削減 with non_speech recall 97.9%→94.1% の trade-off、 Phase 2 report §2.4、 collateral: Parakeet_en / Canary にも適用、 いずれも speech margin 十分)
  - `avg_logprob_thresholds["reazonspeech"]`: `-0.2 → -0.40` (Pareto relaxed_B、 現 default -0.20 の FRR 42.5% 実害を 5.4% に改善、 Phase 2 report §2.1)
  - `avg_logprob_thresholds["qwen3-asr"]`: `-0.3 → -0.42` (Pareto relaxed_C、 JA/EN 両方に適用、 EN は Phase 1 probe で safety verify 済、 Phase 2 report §2.2、 SNR 10 borderline は Layer 4 で再確認予定)
  - `avg_logprob_threshold`: **不変** (`-1.0`、 Voxtral 用 global fallback として維持)

**Migration**:

```python
# 旧 default (PR-A.0/A.5.X で verify された値)
FilterConfig(
    no_speech_threshold=0.5,
    token_conf_threshold=0.005,
    avg_logprob_threshold=-1.0,  # 変更なし
    avg_logprob_thresholds={"reazonspeech": -0.2, "qwen3-asr": -0.3},
)

# 新 default (Unreleased / Issue #334 PR-4 以降、 Phase 2 report Pareto gate 適用値)
FilterConfig(
    no_speech_threshold=0.71,
    token_conf_threshold=0.001,
    avg_logprob_threshold=-1.0,  # 変更なし
    avg_logprob_thresholds={"reazonspeech": -0.40, "qwen3-asr": -0.42},
)

# 旧挙動を維持したい場合 (明示 override)
FilterConfig(
    no_speech_threshold=0.5,
    token_conf_threshold=0.005,
    avg_logprob_thresholds={"reazonspeech": -0.2, "qwen3-asr": -0.3},
)
```

**Trade-off** (Phase 2 report §2.4 / §4.1):

- **Parakeet_ja** (`0.005 → 0.001`): false reject **39 → 11 (72% 削減)** の引き換えに non_speech recall 97.9% → 94.1% (-3.9pt)。 clean speech user 体感で顕著な改善、 non_speech 側の recall 低下は許容範囲判定。
- **Parakeet_en / Canary** (collateral、 Phase 2 未 calibrate): scalar `token_conf_threshold` 変更で共用のため collateral 適用、 speech `token_confidence_mean` 実測 mean が Parakeet_en 0.2452 / Canary 0.0724 と新 threshold `0.001` から margin 十分 (245× / 72×)、 実害なし想定。
- **Qwen3-ASR EN** (collateral、 単一 dict key で JA/EN 共用): EN Phase 1 probe (speech `-0.05` / applause `-1.08`) で `-0.42` が safety verify 済 (speech pass 維持 / applause reject 維持)。 engine+language subkey 化は [Finding F9] の対応、 別 PR で。

**Test 更新**:

- `test_default_thresholds_from_pr_a0_verify` を **削除**、 新規 `test_default_thresholds_from_phase2_report` を追加 (Phase 2 report §4.1 の Pareto gate 適用値を pin)。
- `TestRegressionPrA0Values.test_pr_b_corpus_classification` の WhisperS2T 2 rows (`0.635` desk_tap / `0.662` applause) で `expected_reject: True → False` に flip + Pareto trade-off コメント (Phase 2 report §2.3 relaxed_B の直接 evidence として code 内に記録)。
- 12+ test で threshold literal (`0.5` / `0.005` / `-0.2` / `-0.3`) を新値に update。
- 既存 65 test 全 retain、 全 pass。

**Documentation**:

- `docs/reference/api.md` / `cli.md` の閾値表 + code sample 更新
- `docs/audio-filter-reference.md` の signal-family table + banner example + prose margin 説明更新
- `livecap_cli/cli.py` argparse help text 更新
- `livecap_cli/engines/base_engine.py` / `reazonspeech_engine.py` / `qwen3asr_engine.py` docstring 更新

**関連**:

- Historical smoke report (`reazonspeech-confidence-smoke-2026-06-11.md` / `qwen3asr-confidence-smoke-2026-06-12.md` / Phase 1 report) は frozen artifact として **不変**、 最新値は Phase 2 report で記録。
- livecap-gui 側 release note で閾値変更を告知 (本 PR merge 後の別 timeline)。
- Voxtral の `avg_logprob_threshold = -1.0` global fallback は Phase 2 未 calibrate のため不変。 dict 化 schema change は [Finding F9] の対応で別 PR。

#### Engine API contract — fallback adapter cleanup (Issue [#321] PR #3、3-PR 系列完成)

[Issue #321](https://github.com/Mega-Gorilla/livecap-cli/issues/321) の
**3-PR 系列 (PR #1 wording + Canary `beam_size` / PR #2 NeMo fallback chain /
PR #3 本) を完成**させる最終 PR。PR #320 (qwen3asr) / PR #322 / PR #323 で
確立した「framework contract を trust、silent degradation より hard fail」
方針を `TranscriptionEngine` Protocol contract に最終適用。

##### Engine I/O 契約の明文化

`TranscriptionEngine` Protocol (`livecap_cli/transcription/stream.py`) の
docstring を厳格化:

- 実装者は `transcribe()` から **必ず `TranscriptionResult` を返すこと**
- tuple / dict / str / None は契約違反、`apply_filter` (StreamTranscriber
  経路) 側で `AttributeError` が caller に propagate して fail-fast
- 別 path の `SharedEngineManager._process_request` も bare attribute
  access に整理したが、module-level の `except Exception` で contract
  violation も "request failure" として log + `None` 返却 (orphan code、
  Issue [#326] で本 file 削除予定のため fail-fast 化は scope 外)
- pre-1.0 cleanup の方針を明示、precedent (PR #320/#322/#323) を docstring
  で reference

##### `apply_filter()` — `hasattr` legacy guard 削除

`confidence_filter.py:386-390` の `hasattr(result, "engine_confidence")`
guard を削除、bare attribute access に統一:

- **Before**: `if not hasattr(result, "engine_confidence"): return result`
  (旧 mock の tuple 返却互換)
- **After**: bare `result.engine_confidence` access、契約違反時は
  `AttributeError` propagate
- **Audit verify**: 全 test MockEngines (6 件) + 全 `apply_filter()` test
  caller が既に `TranscriptionResult` 返却済を grep + read で確認、guard は
  dead code

##### `SharedEngineManager._process_request` — tuple/dict adapter 削除

`shared_engine_manager.py:437-490` の `hasattr` tuple branch + `isinstance(result, dict)` branch を削除、direct attribute access only に rewrite:

- **Before**: `hasattr(result, 'text')` 主、tuple `(text, conf)` fallback、
  dict `{"text": ..., "confidence": ...}` fallback の 3 path
- **After**: `result.text` / `result.confidence` の bare access。
  ただし method-level の `except Exception as e` は維持されているため、
  契約違反 (`AttributeError`) は **caller に propagate せず** "request
  failure" として log + `None` 返却。`apply_filter` 側 (fail-fast) と
  挙動が異なる点に注意
- **Caveat**: `SharedEngineManager` 自体は production / tests から完全に
  未参照の **orphan code** (`__all__` にも非 export)。本 PR では契約
  整合のみ実施、`except Exception` を狭めて fail-fast 化することは
  scope 外。**ファイル削除自体は [Issue #326]** で対応予定

##### Stale docstring 整理

- `tests/transcription/test_stream.py::FilteringMockEngine` docstring の
  「`MockEngine` は legacy tuple を返す」記述を削除 (実態は `TranscriptionResult`
  返却、stale comment)
- `CLAUDE.md:78` の TranscriptionEngine Protocol 例を `Tuple[str, float]` →
  `TranscriptionResult` に修正 (AI agent guidance と code 契約の乖離を解消)

##### Audit findings (本 PR scope を絞った根拠)

| Audit item | 結果 |
|---|---|
| 全 test MockEngines (6 件) | ✅ 既に `TranscriptionResult` 返却済 |
| `apply_filter()` 全 test caller | ✅ 既に `TranscriptionResult` 渡し済 |
| `SharedEngineManager` の production/test caller | ✅ **0 件** (orphan code) |
| `CLAUDE.md:78` 旧 `Tuple[str, float]` 型 | ⚠ stale、本 PR で修正 |
| `FilteringMockEngine` docstring | ⚠ stale、本 PR で修正 |

→ test fixture 統一 phase は不要、PR scope は contract tightening +
stale comment 整理に絞れた。

##### Migration

- **既存 production engine (WhisperS2T/Parakeet/Voxtral/Canary/ReazonSpeech/
  qwen3asr) は影響なし**: 既に `TranscriptionResult` を返却済
- **既存 test mocks も影響なし**: 既に `TranscriptionResult` 返却 (audit verify)
- **第三者 plugin / custom engine 実装者** (もしいれば): `transcribe()` の
  戻り値を `TranscriptionResult` に統一する必要あり。tuple/dict/str/None
  返却は `AttributeError` で fail-fast

##### Tests (退行ゼロ、712 baseline 維持)

- `tests/transcription/test_confidence_filter.py`: 全 pass (`apply_filter`
  caller 全て `TranscriptionResult`)
- `tests/transcription/test_stream.py`: 全 pass (MockEngine / FilteringMockEngine
  共に `TranscriptionResult`)
- Full local regression: 712 passed

##### Out of scope (本 PR では行わない)

- `livecap_cli/engines/shared_engine_manager.py` orphan file 自体の削除 →
  別 issue "audit unused engine infrastructure" (`SharedEngineManager` +
  `TranscriptionRequest` + `ProgressCallback` の orphan 確認 + 削除提案)
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 → 別 issue
- 他 engine の `__init__` `**kwargs` 削除 → 別 issue
- `docs/planning/archive/*.md` の旧型 reference → archive 性質上 触らない

##### Issue #321 完成宣言

| PR | scope | 状況 |
|---|---|---|
| **PR #1 ([#322])** | wording cleanup + Canary `beam_size` fail-fast | ✅ merged |
| **PR #2 ([#323])** | Canary/Parakeet NeMo fallback chain | ✅ merged |
| **PR #3 (本)** | API contract cleanup | ✅ |

3-PR 系列完成、本 PR merge 後に Issue #321 を close。

#### Engine confidence — Canary / Parakeet NeMo fallback chain cleanup (Issue [#321] PR #2)

`CanaryEngine._configure_decoding_with_confidence` と
`ParakeetEngine._configure_decoding_with_confidence` から、`token_confidence`
取得 path (Path 1) が失敗した場合に **token_confidence なしで継続する silent
degradation を生む fallback chain** を削除して fail-fast 化。加えて
`ParakeetEngine._transcribe` の `return_hypotheses=True` TypeError silent
fallback も削除。

PR #320 (qwen3asr) / PR #322 (Canary `**kwargs`) の precedent と整合、
「framework contract を trust、silent degradation より hard fail」方針を
NeMo 系 engine にも適用。

##### Before / After

- **Before**:
  - Canary: Path 1 (greedy + confidence_cfg) → Path 2 (greedy のみ、confidence
    なし) → Path 3 (argument-less)。Path 1 失敗時に silent fallback、
    `token_confidence_mean` が None になり confidence filter は pass-through に degrade
  - Parakeet: Path 1 (Hybrid CTC) → Path 1.5 (Pure RNNT/TDT) → Path 2
    (strategy-only) → Path 3 (argument-less)。同様に Path 1/1.5 失敗時に
    silent fallback
  - `ParakeetEngine._transcribe`: `transcribe(return_hypotheses=True)` が
    TypeError なら kwarg なしで再 transcribe、`engine_confidence` 全 None
    で filter が pass-through に degrade
- **After**:
  - Canary: Path 1 のみ、bare 呼出 (try/except 削除)。失敗時は NeMo native
    `TypeError` / `ValueError` 等が propagate
  - Parakeet: Path 1 (Hybrid model-family dispatch、temporary fallback で
    Path 1.5 へ) + Path 1.5 (Pure RNNT/TDT primary、bare 呼出)。Path 1.5
    失敗時は NeMo error が propagate
  - `ParakeetEngine._transcribe`: bare 呼出、`return_hypotheses=True` 失敗時
    は TypeError が propagate (`nemo-toolkit>=2.3,<2.5` の supported range で
    公式安定 API)

##### Migration

- `nemo-toolkit>=2.3,<2.5` (lockfile `2.3.0`) の supported range では既存
  挙動と完全に同じ (Path 1 / Path 1.5 / `return_hypotheses=True` は常に成功)
- supported range 外の旧 nemo build を使う user は、Path 1 が拒否された時点で
  従来の silent fallback ではなく `TypeError`/`KeyError`/`ValueError` 等が
  直接 raise されるため、具体的な NeMo error message から nemo version を
  確認する actionable hint を得る
- Parakeet **Path 1 (Hybrid CTC)** と **Path 1.5 (Pure RNNT/TDT)** は
  **model-family dispatch** (hybrid vs pure decoder の正規 dispatch) として
  温存。legacy fallback ではないため reviewer 承認の上で温存

##### Verification (merge gate)

`tests/integration/engines/test_smoke_engines.py::test_token_confidence_populated`
で実機 GPU verify (RTX 4090 self-hosted runner):

| Case | Expected token_confidence_mean (probe baseline) |
|---|---|
| `canary_gpu_en` (LibriSpeech 英語) | > 0.05 (PR-A.4.2 で 0.0724) |
| `parakeet_gpu_en` (LibriSpeech 英語) | > 0.10 (PR-A.4.3 で 0.2452) |
| `parakeet_ja_gpu_ja` (jsut 日本語) | > 0.02 (PR-A.0 で 0.0504) |

新 test は `@pytest.mark.engine_smoke` で hosted CI から除外、self-hosted
GPU runner でのみ実行。失敗時は merge を blocking する design。

##### Out of scope (本 PR では行わない)

- `confidence_filter.py:386` `hasattr(result, "engine_confidence")` guard
  削除 — **Issue #321 PR #3** で `shared_engine_manager.py` tuple fallback
  + `TranscriptionEngine` Protocol cleanup とセットで扱う
- `BaseEngine.__init__` の `**kwargs` swallowing 削除 — 別 issue
- nemo-toolkit version の pin 化 / 上限拡大 — 別 issue (本 PR では現状の
  `>=2.3,<2.5` range を contract として扱う)

#### Engine confidence filter — qwen3asr support via wrapper bypass (Issue [#318] PR-A.5.2)

PR-A 系列の **7 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
最終形に到達させる PR。Phase 1 probe (Issue #318 で User 意向「EN/JP 両言語
対応出来なければ close」に対する go condition 達成) を受けて実装、両言語
verified で qwen3asr を追加対応。

##### Engine integration — wrapper bypass で avg_logprob 抽出

qwen3asr は qwen-asr wrapper が ``output_scores=True`` を渡さない設計の
ため confidence filter 非対応だったが、**`Qwen3ASRModel.model` (= 内部
``Qwen3ASRForConditionalGeneration``) を直接呼ぶ wrapper bypass** で対応:

- ``transcribe()`` を rewrite、``self.model.model.generate(output_scores=True,
  repetition_penalty=1.1, no_repeat_ngram_size=3)`` を直接呼出
- ``compute_transition_scores(normalize_logits=True)`` 経由で per-token
  logprob を取得、Voxtral PR-A.4.1 と完全同形の ``_extract_engine_confidence``
  helper で mean → ``EngineConfidence.avg_logprob`` populate
- ``_asr_language is None`` (auto-detect mode) は旧 wrapper.transcribe()
  path に fail-open (engine_confidence 全 None、filter pass-through)

##### `repetition_penalty=1.1 + no_repeat_ngram_size=3` で両言語 failure mode 解消

Phase 1 probe で確認した critical finding:

- **Japanese**: desk_tap の 256-token repetition loop ("うんうんうん...") を
  4-token "うん。" に短縮、avg_logprob -0.13 → -0.65、**margin -0.02 (逆転)
  → +0.27 (filter 可能)**
- **English**: applause の system prompt leak ("You are a speech
  recognition model.") を avg_logprob -0.04 → -1.08 に低下 + "You are an AI."
  に短縮、**margin -0.03 (逆転) → +0.21 (filter 可能)**

→ **両言語で同じ generation parameter** で対応可能、言語別実装不要。

##### Section 1 smoke (両言語 6 clip)、Phase 1 probe 値を上回る margin 確認

- **English**: speech -0.05、non-speech -0.71、**margin +0.65** (Phase 1
  probe +0.21 を大幅に上回る) → Case A
- **Japanese**: speech -0.12、non-speech -0.63、**margin +0.42** (Phase 1
  probe +0.27 を上回る) → Case A
- 両言語で threshold ``-0.3`` で 100% 分類成功

##### Section 2 (12 cell stream pipeline benchmark)

- **Hall.(pre) = 0% 全 cell** — qwen3asr は `repetition_penalty` 適用後
  本 corpus で hallucinate しない (Canary PR-A.4.2 と同 engine 固有
  fail-safe pattern)
- SR(post) = 100% real corpus 全 cell (legit speech は 1 件も drop されず)
- Latency 影響なし

##### Engine-specific threshold = `-0.3` (両言語 safe)

- `FilterConfig.avg_logprob_thresholds["qwen3-asr"] = -0.3` を default load
- dict key を ``"qwen3-asr"`` (ハイフン含む) にしているのは、
  ``_engine_id_from_name("Qwen3-ASR 0.6B")`` が ``"qwen3-asr"`` に normalize
  するため (PR-A.5.1 codex Point 1 の learning を pre-empt)
- Phase 4 unit test で **production display string** での threshold lookup
  を pin (6 件 + helper mapping 拡張で qwen3-asr 追加)

##### Migration

| Engine | Before | After |
|---|---|---|
| qwen3asr (language 指定あり) | fail-open (engine_confidence 全 None) | ``avg_logprob < -0.3`` で reject、`repetition_penalty=1.1 + no_repeat_ngram_size=3` で failure mode 解消 |
| qwen3asr (auto-detect mode) | (不変) | (不変、wrapper fallback で fail-open) |
| WhisperS2T / Parakeet (ja/en) / Voxtral / Canary / ReazonSpeech | (不変) | 退行ゼロ |

##### Caveats (production user 向け)

1. **WER 軽微退行 (LLM typical 0.5-1%)**: ``repetition_penalty=1.1 +
   no_repeat_ngram_size=3`` で稀に正常 token も抑制可能性。Voxtral
   PR-A.4.1 / Canary PR-A.4.2 と同 framing で filter benefit を優先。
   ``--confidence-filter off`` は **post-ASR reject のみ**無効化し、
   generation 側変更 (``repetition_penalty=1.1`` / ``no_repeat_ngram_size=3``)
   は固定で残る (Voxtral greedy / Canary greedy と同 design)
2. **多言語 verify (28+ 言語) は本 PR scope 外**: en/ja のみ verified、
   他言語は user feedback ベース (Voxtral / Canary と同 framing)
3. **`_asr_language is None`** (auto-detect mode) は fail-open、production
   user は ``--language en/ja/...`` を明示推奨
4. **wrapper internal attribute 依存**: ``self.model.model`` (=
   ``Qwen3ASRForConditionalGeneration``) の private structure に依存。
   qwen-asr が future update でこの構造を変更すると AttributeError で hard
   fail する (Voxtral PR-A.4.1 と同じく framework contract を trust する design)

##### Tests (新規 +20 件、合計 703 passed)

- `tests/core/engines/test_qwen3asr_confidence_extraction.py` (新規 14 件) —
  Voxtral pattern 流用、masking ロジック + tensor/numpy 互換 + Phase 1
  probe 値再現を pin
- `tests/transcription/test_confidence_filter.py` (+6 件) —
  `TestQwen3AsrEngineSpecificThreshold` + `TestEngineIdNormalization` に
  `"Qwen3-ASR 0.6B"` / `"Qwen3-ASR 1.7B"` display string pin
- `tests/transcription/test_stream.py` (+1 件) — banner test に
  ``qwen3-asr`` + ``-0.3`` assertion

##### Docs

- ``docs/research/qwen3asr-confidence-smoke-2026-06-12.md`` (新規 decision doc)
- ``docs/audio-filter-reference.md``: Engine support table を **7 engine
  対応**に拡大、Property table / Decision section / 完成サマリ / Comparison
  table の全 section で 6 → 7 engine 整合
- ``docs/reference/cli.md`` + ``docs/reference/api.md``: qwen3asr 行追加 +
  `avg_logprob_thresholds` 例に qwen3-asr 追記
- ``base_engine.py`` ``EngineConfidence`` docstring の populate status table
  に qwen3asr 追加

##### Out of scope (本 PR では行わない)

- 多言語 verify (en/ja 以外の 28+ 言語) — user feedback ベース
- ``Qwen3ASRModel.transcribe()`` の auto-detect mode (``force_language=None``)
  — scope 外、必要なら follow-up
- 長尺音声 (>50s) の auto-chunking — stream pipeline では不要 (VAD segments
  典型 <30s)
- CLI flag ``--qwen3asr-repetition-penalty`` 等の generation param 制御 —
  不要 (hardcoded、Voxtral / Canary と整合)

#### Engine confidence filter — ReazonSpeech support + engine-specific threshold (Issue [#317] PR-A.5.1)

PR-A 系列の **6 engine 対応** を達成、Confidence Filter (Phase 1 Layer 5) を
完成形に到達させる PR。reviewer feedback (Issue #317) で 7 件の critical
指摘を受領、本 PR で全反映。

##### Production bug fix (reviewer Point 6、CRITICAL)

`reazonspeech_engine.py:430` の ``text, confidence = self._transcribe_single(...)``
unpack が PR #314 で削除済 ``TranscriptionResult.__iter__`` で TypeError を
投げるが、外側の ``except Exception`` で silent swallow + ``continue`` して
いたため、**長尺音声 (>30s、``auto_split_duration`` 経路) で全 segment が
silently dropped されていた production critical bug** を Phase 1 で独立
commit で修正。3 件 regression test (mock-based) で pin。

旧挙動: 30s 超え audio → 空 transcription (production reach 中)
新挙動: 各 segment の text が正しく combined_text に積まれる

これは breaking change ではなく **production bug 修正**。

##### ReazonSpeech confidence filter integration (Issue [#317] core)

旧 docs ([Issue #308 close 時点]) では「sherpa-onnx Python bindings に
per-token score API なし、PR #2897 closed/not-merged、Python 未対応」を
理由に **PR-A.5 candidate (heavy refactor)** としていたが、本 PR plan
段階の実機 probe で **sherpa-onnx 1.12.39 で
``OfflineRecognitionResult.ys_log_probs`` が既に exposed されている**こと
が判明、standard integration work で対応:

- **Before**: ReazonSpeech の ``transcribe()`` は ``engine_confidence =
  EngineConfidence()`` (全 None) で fail-open
- **After**:
  - ``reazonspeech_engine.py`` に module-level ``_extract_engine_confidence(result)``
    helper 追加 (Canary / Voxtral pattern 流用、実 sherpa-onnx 不要に unit
    test で schema pin)
  - ``_transcribe_single()`` で sherpa-onnx ``result.ys_log_probs`` を抽出、
    mean を **``EngineConfidence.avg_logprob`` field** に populate
    (Voxtral と同 semantics、reviewer Point 1/2 で確定設計)
  - ``raw["ys_log_probs_mean"]`` + ``raw["ys_log_probs_n"]`` に metadata 保存
  - ``_transcribe_with_split()`` で segment 別 engine_confidence を
    **weighted mean** で aggregate (token 数 weight)

reviewer Point 1 (CRITICAL): ``token_confidence_mean`` field 再利用は
**probability (0-1 range) vs log probability (負の値) semantics 不整合**で
全 reject になる critical bug。``avg_logprob`` field 使用が正解。

##### Engine-specific threshold (reviewer Point 3、HIGH)

ReazonSpeech (margin +0.10-0.13) と Voxtral (margin +1.0) は同 ``avg_logprob``
field を共用するが分布が桁違い。global ``avg_logprob_threshold = -1.0`` は
ReazonSpeech に機能しないため、**``FilterConfig.avg_logprob_thresholds:
Dict[str, float]``** を追加:

- ReazonSpeech default ``-0.2``
- Voxtral は dict に load しない → ``avg_logprob_threshold`` (global) fallback
  で ``-1.0`` 維持 (**backward compat ゼロ regression**)
- ``should_reject()`` の signature を ``(result, config, engine_name=None)``
  に refactor、``apply_filter()`` から engine_name pass-through
- engine-specific lookup → global fallback の 2 段 fallback で scalable

##### Findings (詳細は ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md``)

- **Phase 1 bug fix** ✅ — 3 件 regression test で long-audio silent drop bug pin
- **Section 1 (engine smoke、5 clip × int8/float32)** ✅ —
  - int8: speech mean -0.14、non_speech mean -0.30、margin +0.13、Case A
  - float32: speech mean -0.16、non_speech mean -0.45、margin +0.10、Case A
  - 両 model で threshold -0.2 が 100% 分類成功 (reviewer Point: int8
    availability も確認済)
- **Section 2 (12 cell stream pipeline benchmark)** ✅ —
  - **``webrtc × reazonspeech × real × on``: Hall.(post) 50% → 0%**
    (Issue #295 元 motivation の最後の cell 完了)
  - ``webrtc × synthetic × on``: 62.5% → **0%** (完全解消、codex-review
    #319 1st round の engine_name normalize fix 後の 2nd run で確認、初版
    は 25% 残)
  - silero / tenvad × real: 0% → 0% (VAD で除去済、filter は冗長安全網)
  - Latency 影響ゼロ
- **Section 3 (language coverage)** — ReazonSpeech は日本語 native only、
  Canary PR-A.4.2 と同 framing

##### Migration

- WhisperS2T / Parakeet (ja/en) / Voxtral / Canary 退行ゼロ
- ReazonSpeech user は ``--confidence-filter on`` (default) で hallucination
  が自動 drop されるようになる
- 長尺音声 (>30s) user は production bug 修正で正しい transcription を
  受け取れるようになる

##### Out of scope (qwen3asr は Issue [#318] で research-phase)

reviewer Point 5 (HIGH): qwen3asr の avg_logprob 単独 filter は危険
(confidence filter ≠ hallucination content guard、英語 mode で system
prompt leak、日本語 mode で repetition loop)。Issue [#318] で probe +
hallucination guard 設計を別 PR で扱う (本 PR scope 外)。

##### docs update (6 engine 対応に整合)

- ``docs/research/reazonspeech-confidence-smoke-2026-06-11.md`` (新規)
- ``docs/audio-filter-reference.md`` Engine support table を 6 engine 対応、
  Property table / Decision section / 完成サマリ / Comparison table 全
  section で 5 → 6 engine 整合
- ``docs/reference/cli.md`` / ``docs/reference/api.md``: ReazonSpeech 追加、
  ``avg_logprob_thresholds`` field 明記
- ``base_engine.py`` EngineConfidence docstring の populate status table に
  ReazonSpeech 追加
- ``confidence_filter.py`` module docstring を engine-specific threshold
  framing に整合

#### Engine confidence filter — Parakeet 英語 support (Issue [#311] PR-A.4.3)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) / PR-A.4.2 ([#315]) で whispers2t /
parakeet_ja / Voxtral / Canary に対応した confidence filter を **Parakeet
英語** (`nvidia/parakeet-tdt-0.6b-v2`) にも拡張。本 PR で **Parakeet 英語が
構造的限界ではなく PR #309 時点の設定漏れだった**ことが判明、新規 **Path
1.5** で対応:

- **Before**: Parakeet 英語の `transcribe()` は `engine_confidence =
  EngineConfidence()` (全 None) で fail-open。decoding は strategy-only
  (旧 Path 2)。
- **After**:
  - `parakeet_engine.py::_configure_decoding_with_confidence` に **Path 1.5**
    追加 (Path 1 Hybrid CTC と Path 2 strategy-only の間に挿入):
    - pure RNNT/TDT model 用、`preserve_alignments=True` + `confidence_cfg`
      + `greedy.preserve_frame_confidence=True`
    - NeMo の制約「preserve_frame_confidence は preserve_alignments と同時
      設定必須」(`rnnt_decoding.py:280-282`) を満たす形で構成
    - Path 1.5 が rejected された場合は Path 2 (strategy-only) に fail-open
      fallback
  - `_extract_engine_confidence()` helper は Parakeet_ja と同じものを共用
    (Canary PR-A.4.2 で Tensor / List / numpy 全部扱えるよう拡張済)
  - `_log_filter_banner()` の表現を `parakeet_ja / canary` → **`parakeet (ja/en) / canary`**
    に整合
- **Migration**:
  - WhisperS2T / Parakeet_ja / Voxtral / Canary 退行ゼロ
  - Parakeet 英語 user は `--confidence-filter on` (default) で英語 audio の
    hallucination が自動 drop される
  - **非英語入力時の false reject リスク**: Parakeet 英語は English-only model、
    非英語音声には低 confidence で false reject の可能性。`--confidence-filter off`
    で opt-out 可能
- **Findings (詳細は `docs/research/parakeet-english-confidence-smoke-2026-06-11.md`)**:
  - **Phase 1 probe** ✅ — `hypothesis.token_confidence` は **List[float]** で
    populate。LibriSpeech 英語 → token_confidence_mean = **0.2452**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.2452 vs threshold 0.005
    で **49× margin** (Case A、3 engine 中で最大)。非音声は engine 自体が
    empty text を返す fail-safe
  - **Section 2 (stream pipeline、12 cell)** ✅ — `webrtc × synthetic × on`
    で **Hall.(post) 75% → 12.5%** を実証 (filter で hallucination の 5/6 を drop)
  - **Section 3 (language coverage)** — English native validate、非英語入力時
    の language mismatch も実機確認 (production user 注意点として docs 記載)
- **Tests** (新規 +3 件、合計 655 passed):
  - `tests/core/engines/test_parakeet_decoding_strategy.py` の pin を新挙動
    (Path 1.5 で confidence_cfg 試行) に整合、CTC failure fallback も Path 1.5
    に整合
- **Docs update**:
  - `docs/research/parakeet-english-confidence-smoke-2026-06-11.md` (新規)
  - `docs/audio-filter-reference.md`: Engine support table を Parakeet 英語
    ✅ Production に更新、PR-A 系列完成サマリ 5 engine 対応に拡大
  - `docs/reference/cli.md` / `docs/reference/api.md`: Parakeet 英語追加
  - `livecap_cli/cli.py` / `confidence_filter.py` / `base_engine.py` /
    `stream.py`: 全 layer で Parakeet (ja/en) 一貫表示

#### PR-A 系列完成 docs 整合 (Issue [#311] PR-A.4.docs)

Issue #311 v2.1 plan の最終 PR。PR-A.4.1 ([#313 MERGED]) で Voxtral、PR-A.4.2
([#315 MERGED]) で Canary の filter 対応を完了した後の **docs 整合 sweep**。

- **Before**: 一部 docs に stale な「voxtral / canary は fail-open」記述が
  残存:
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md:177` (PR-B calibration
    時の残作業 list、voxtral/canary がまだ fail-open とされていた)
  - `docs/research/voxtral-confidence-smoke-2026-06-11.md:108` (他 engine
    の挙動 section に Canary が fail-open list で残存)
- **After**:
  - `pr-a-calibration-2026-06-10.md`: PR-A.4.1/A.4.2 完了状況を反映、qwen3asr
    のみ PR-A.5 candidate として残存する旨を明示
  - `voxtral-confidence-smoke-2026-06-11.md`: Canary を populate engine list
    に移動 (PR-A.4.2 整合)
  - `docs/audio-filter-reference.md`:
    - Property table の Production-ready statement を 4 engine (WhisperS2T /
      Parakeet_ja / Voxtral / Canary) に拡張
    - Comparison table の Confidence Filter 行を「4 engine 対応」+ 50%→0%
      実測実証を反映
    - **新 section: PR-A 系列 完成サマリ** (2026-06-11 時点) を追加 — Engine
      support table の最終状態 / production user 選択ガイド / Phase 1 多段
      防御 5 layer 到達点を 1 section に集約
- **Side effects**:
  - 全 docs 層 (audio-filter-reference / cli.md / api.md / feature-inventory
    / decision doc × 2 / CHANGELOG / Engine support table / source docstring)
    で **Canary が `token_confidence_mean` populate engine** として一貫表示
    完了
  - Issue #311 v2.1 plan の Core scope (PR-A.4.1 + PR-A.4.2 + PR-A.4.docs)
    が完了、close 候補に
- **Out of scope (PR-A.5 candidate に申し送り)**:
  - qwen3asr: qwen-asr wrapper が内部で ``output_scores=True`` を渡さず、
    ``text_ids = model.generate(...)`` のみ実行 ([source 確認済](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py))。
    wrapper bypass or vLLM logprobs 移行が必要 (heavy)。
  - ReazonSpeech: sherpa-onnx Python bindings に per-token score API
    なし。upstream [PR #2897](https://github.com/k2-fsa/sherpa-onnx/pull/2897)
    が C/Dart で `getVocabLogProbs()` 追加したが closed (not merged)、
    Python 未対応。upstream PR or PyTorch native 実装切替が必要。
- **🆕 PR-A.4.3 candidate (Issue #311 v2.2 へ申し送り)**:
  - **Parakeet 英語** (`parakeet-tdt-0.6b-v2`): 旧 docs では「NeMo RNNT
    path に token_confidence 未実装」を根拠に PR-A.5 candidate としていた
    が、本 PR 作業中の調査で **「構造的限界ではなく PR #309 時点の設定漏れ」**
    と判明。NeMo source (`rnnt_decoding.py:95-106`, `tdt_loop_labels_computer.py`)
    で `preserve_token_confidence` documented + 実装あり、`preserve_alignments
    =True` 同時設定で populate される。**実機 probe (本 PR docs 作業中)**
    で `token_confidence_mean = 0.2452` を確認 (LibriSpeech 英語、threshold
    0.005 の 49x)。実装は別 PR (PR-A.4.3) で対応 — 本 PR は docs scope の
    ため probe 修正は revert 済、PR-A.4.3 で改めて Path 1.5 実装 + smoke
    verify + 完全 docs 整合を実施予定。
  - **PR-A.4.3 acceptance criteria** (codex-review PR #316 3rd round 提示):
    1. ``parakeet_engine.py::_configure_decoding_with_confidence`` の pure
       RNNT/TDT path に ``preserve_alignments=True`` と
       ``confidence_cfg.preserve_token_confidence=True`` を含む dedicated
       path (Path 1.5) を追加
    2. その path が失敗した場合は現行の strategy-only path に fail-open
       fallback (Path 2 既存)
    3. ``tests/core/engines/test_parakeet_decoding_strategy.py:113-138``
       を更新し、Parakeet English で confidence cfg を試行することを pin
       (現状は pure RNNT で confidence_cfg を含めないことを pin している
       ため、PR-A.4.3 で挙動変更に合わせ更新必須)
    4. ``tests/core/engines/test_parakeet_confidence_extraction.py`` は
       既存 helper が list/tuple の ``token_confidence`` を扱えているため
       大枠流用可能。Parakeet English の hypothesis shape が異なる場合
       (Tensor / numpy 等) は fixture 追加
    5. 実機 smoke で ``token_confidence_mean`` populate + speech が
       threshold ``0.005`` を十分上回ること確認 (本 PR probe で 0.2452 =
       49× を確認済、smoke verify で再現性確保)

#### Engine confidence filter — Canary support (Issue [#311] PR-A.4.2)

PR-A.0 ([#309]) / PR-A.4.1 ([#313]) で whispers2t / parakeet_ja / Voxtral に
対応した confidence filter を **Canary 1B Flash** にも拡張。NeMo
``EncDecMultiTaskModel`` の **beam → greedy decoding 切替** +
``confidence_cfg.preserve_token_confidence=True`` で
``hypothesis.token_confidence`` (torch.Tensor) を取得、Parakeet 同様の
``token_confidence_mean`` を ``EngineConfidence`` に populate。

- **Before**: Canary の ``transcribe()`` は ``engine_confidence = EngineConfidence()``
  (全 None) で fail-open、``confidence = 1.0`` ハードコード。
- **After**:
  - ``canary_engine.py::_configure_model()`` で ``_configure_decoding_with_confidence()``
    呼出 (3-fallback path、Parakeet pattern 流用): Greedy + confidence_cfg →
    Greedy only → argument-less。いずれも raise しない。
  - ``_transcribe_single_chunk()`` に ``return_hypotheses=True`` 追加、
    Hypothesis から ``_extract_engine_confidence()`` 経由で
    ``token_confidence_mean`` 取得。
  - **新 helper**: ``_extract_engine_confidence(hypothesis)`` — Canary は
    ``token_confidence: torch.Tensor`` (Parakeet は ``List[float]``) で
    返すため ``hasattr(token_conf, 'tolist')`` 防御で GPU tensor / numpy /
    list を統一処理。
  - ``confidence = float(token_confidence_mean)`` で UI display 意味化。
  - **filter logic 変更なし**: ``confidence_filter.py::should_reject()``
    の ``token_conf_threshold = 0.005`` path を Parakeet_ja と共用。
  - **新 doc**: ``docs/research/canary-confidence-smoke-2026-06-11.md`` に
    Phase 1 probe + Section 1/2/3 を永続化。
- **Migration**:
  - **WhisperS2T / Parakeet_ja / Voxtral 退行ゼロ** (filter logic 不変)。
  - **Canary user** は ``--confidence-filter on`` (default) で対応言語
    (en/de/fr/es) の hallucination が自動 drop される。
  - decoding strategy が beam → greedy に切替 (NeMo AED の confidence 取得
    のため)。Parakeet_ja TDT→CTC 同様、軽微 WER 退行可能性あるが filter
    benefit を優先。**``--confidence-filter off`` は post-ASR の reject を
    止めるだけで、decoding は常に greedy** (filter logic と decoding strategy
    を独立に管理)。旧 beam decoding に戻す option は本 PR では非提供。
  - **``beam_size`` parameter 削除 (silent no-op cleanup、PR-A.4.2 →
    Issue #321 PR #1 で完成)**: 旧 ``CanaryEngine`` constructor の
    ``beam_size: int = 1`` および ``metadata.default_params`` の
    ``beam_size`` は ``_configure_decoding_with_confidence()`` で常に
    greedy 切替されるため silent no-op だった。pre-1.0 cleanup 方針に従い
    削除。
    - **Before**: ``CanaryEngine(beam_size=N)`` は warn + silent ignore
    - **After** (Issue #321 PR #1): ``CanaryEngine.__init__`` から
      ``**kwargs`` を完全削除、``CanaryEngine(beam_size=N)`` は
      ``TypeError: unexpected keyword argument 'beam_size'`` で fail-fast
    - **Migration**: caller は ``beam_size`` 指定を削除する必要がある
      (greedy が常に有効、beam search に戻す option は本 PR では非提供)
- **Findings**:
  - **Phase 1 probe** ✅ — ``hypothesis.token_confidence`` は **torch.Tensor**
    で populate (Parakeet と型差分、helper で吸収)。LibriSpeech 英語 →
    token_confidence_mean = **0.0724**
  - **Section 1 (engine smoke、3 clip)** ✅ — speech 0.0724 vs threshold
    0.005 で **14.5x margin** (Case A)。非音声は engine 自体が empty text を
    返す **fail-safe 挙動** (Voxtral/Parakeet と異なり Canary は元々
    hallucinate しない)
  - **Section 2 (stream pipeline benchmark、12 cell)**:
    - Hall.(pre) = 0% 全 cell (Canary 固有の robustness)
    - SR(post) = 0% all real cells — PR-B corpus は日本語、Canary 非対応
      で engine が empty text を返す fail-safe
    - filter on/off で同一結果 (pre-filter hallucination が 0% のため filter
      効果が観察不可、Section 1 で margin 検証済)
    - Latency 影響なし
  - **Section 3 (language coverage)** ✅ — English native で margin 確認済、
    Japanese は Canary 非対応で empty text 返却 (Voxtral の translation
    regime と異なり誤訳混入なし)
- **Out of scope**: qwen3asr / reazonspeech / parakeet_en は **PR-A.5**
  (heavy refactor)。Canary 他言語 verify は user feedback ベース。
  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は本 PR の probe で
  > 「実は populate 可能 (PR #309 時点の `preserve_alignments` 併設漏れ)」と
  > 判明、**PR-A.4.3 candidate に格上げ**。PR-A.5 は qwen3asr / reazonspeech
  > の 2 engine に縮減 (上の PR-A.4.docs entry 参照)。

#### ``TranscriptionResult.__iter__`` 削除 (pre-1.0 cleanup)

PR-A.0 ([#309]) で導入した ``TranscriptionResult.__iter__`` (旧
``Tuple[str, float]`` 戻り値との後方互換 shim) を削除。pre-1.0 (
`1.0.0.dev0`) では legacy compat shim は不要との CLAUDE.md / AGENTS.md
方針に従う。

- **Before**: ``text, confidence = engine.transcribe(audio, sr)`` の
  tuple unpacking 形が動作 (``TranscriptionResult.__iter__`` 経由)。
- **After**: ``result = engine.transcribe(audio, sr); text =
  result.text; confidence = result.confidence`` の attribute access に
  統一。``__iter__`` 削除により tuple unpacking は ``TypeError`` で fail。
- **Migration**:
  - `livecap_cli/transcription/stream.py`: 3 path (sync / async /
    interim) を attribute access に migration
  - `benchmarks/asr/runner.py` / `benchmarks/common/engines.py` /
    `benchmarks/optimization/objective.py`: 同 migration
  - `tests/`: 4 mock engine fixture (test_stream.py /
    test_stream_translation.py / test_mock_realtime_flow.py /
    test_from_language_integration.py) を ``Tuple[str, float]`` 返却 →
    ``TranscriptionResult`` 返却に統一
  - `tests/core/engines/test_engine_confidence_schema.py`:
    ``__iter__`` 互換を pin する 3 件を削除、代わりに
    ``test_tuple_unpacking_no_longer_supported`` で ``TypeError`` 化を
    pin
  - `livecap_cli/engines/{base,whispers2t,parakeet,canary,qwen3asr}_engine.py`
    docstring から「tuple unpacking 互換」記述削除
  - `livecap_cli/transcription/stream.py` / `confidence_filter.py`
    docstring も同様に整合
- **Side effects**:
  - `livecap_cli/engines/shared_engine_manager.py:443-449` の defensive
    3-branch (attribute / ``__getitem__`` / fallback) は不変。
    ``__iter__`` に依存しない別 path のため独立 cleanup PR で対応可。
  - `livecap_cli/engines/reazonspeech_engine.py:430` の internal helper
    ``_transcribe_single()`` も不変 (``TranscriptionResult`` ではなく
    raw tuple を返す内部関数、cleanup scope 外)。
- **Tests**: 508 → 506 passed (削除 3 - 新 1 = -2)。

#### Engine confidence filter — Voxtral support (Issue [#311] PR-A.4.1)

PR-A.0 ([#309]) / PR-A.1 ([#310]) で whispers2t / parakeet_ja に対応した
confidence filter を **Voxtral** にも拡張。`transformers.GenerationMixin.compute_transition_scores(normalize_logits=True)` 経由で per-token logprob
を復元、special token (EOS/PAD/BOS) 除外平均を `EngineConfidence.avg_logprob`
field に populate する。

- **Before**: Voxtral の `transcribe()` は `engine_confidence = EngineConfidence()`
  (全 None) で返却 → `is_available=False` → filter 常に fail-open。`confidence`
  field は `1.0` ハードコード。
- **After**:
  - `livecap_cli/engines/voxtral_engine.py` の `_transcribe_single_chunk()`
    を `model.generate(output_scores=True, return_dict_in_generate=True)` 化、
    `compute_transition_scores` で score 復元、`_extract_engine_confidence()`
    helper で special token 除外平均を計算。
  - `confidence = exp(avg_logprob)` で UI confidence display を意味化
    (PR-A.0 の WhisperS2T と整合)。
  - **新 helper**: `_extract_engine_confidence(transition_scores, gen_tokens, special_ids) -> EngineConfidence`
    — pure function として export、PR-A.0 の whispers2t / parakeet 同パターン。
  - **filter 拡張**: `confidence_filter.py::should_reject()` に **strict-gated**
    `avg_logprob` 分岐追加。`no_speech_prob is None` AND
    `token_confidence_mean is None` の時のみ評価 (WhisperS2T 退行回避)。
  - **新 default**: `FilterConfig.avg_logprob_threshold = -1.0` (PR-A.4.1
    smoke verify 結果に基づく決定)。Voxtral smoke (2026-06-11, RTX 4090) で
    speech 4 clip mean=-0.420 vs non-speech mean=-1.525、margin +1.002、
    midpoint -1.024 → `-1.0` で 100% 分類可能。
  - **新 doc**: `docs/research/voxtral-confidence-smoke-2026-06-11.md` に
    decision (Setup / 4 Hypotheses / Results / Decision / Implications /
    Reproducibility) を永続化。
  - **docs update**: `docs/audio-filter-reference.md` の Engine support
    table を Voxtral ✅ avg_logprob (gated) に更新。
- **Migration**:
  - **WhisperS2T / Parakeet_ja 退行ゼロ** (strict gate により avg_logprob
    経路に到達しない、unit test で pin)。
  - **Voxtral user** は `--confidence-filter on` (default) で hallucination
    が自動 drop されるようになる (production behavior 変化、PR-A.4.1 smoke で
    実証)。
  - Python API で `FilterConfig(avg_logprob_threshold=None)` を明示すれば
    avg_logprob 判定経路を opt-out 可能 (e.g., Voxtral debugging 時)。
  - ReazonSpeech / qwen3asr / Canary / mock は `engine_confidence` 全 None
    のまま (fail-open 不変)。
- **Findings (詳細は `docs/research/voxtral-confidence-smoke-2026-06-11.md`)**:
  - **Section 1 (engine-level smoke、6 clip)**:
    - H1 ✅ — Voxtral speech 4 clip max=-0.354、min=-0.523、worst でも -1.0 上
    - H2 ✅ — Voxtral non-speech `applause_5_claps`: -1.525 << -1.0、
      `desk_tap`: empty (全 EOS) → fail-open
    - H3 ✅ — clear margin +1.002 (Case A 確定)
    - H4 ✅ — special token 除外 logic で `desk_tap` (全 EOS) は
      `EngineConfidence()` を返す (filter fail-open)
  - **Section 2 (stream pipeline benchmark、12 cell sweep)**:
    - F2.1 ✅ — **`webrtc × voxtral × real × filter on`: post-filter
      hallucination 50% → 0%**、post-filter speech recall 100% 維持。
      PR-A.4.1 核心 claim を stream pipeline 経由で実機実証。
    - F2.2 ✅ — silero / tenvad × voxtral × real は filter on/off 関係なく
      0% 維持 (副作用ゼロ、VAD 段階で既に non-speech 除去)
    - F2.3 — synthetic positive (formant proxy) は filter on で SR(post)
      40-60% drop。PR-A.3 H3.b と同じ意図通り挙動 (real speech ではない)
    - F2.4 — synthetic Hall.(post) は partial drop (75% → 25% on webrtc)、
      残存は threshold -1.0 と real corpus 100% 維持の trade-off
    - F2.5 ✅ — latency 影響なし (p50/p95 は filter off と同等)
  - **Section 3 (language-stratified follow-up)**: 旧 Section 1 は
    **日本語音声 × language="en"** で実行されていたが Voxtral は en/es/fr/
    pt/hi/de/nl/it の 8 言語のみサポート (ja は対象外)、結果として旧 smoke
    は **translation regime** (ja→en) を測定していた。native English
    transcription (LibriSpeech) で再検証:
    - F3.1 ✅ — Native transcription: avg_logprob -0.115 (translation
      mean -0.420 より 0.305 高信頼度)
    - F3.2 ✅ — Threshold -1.0 は translation regime の lower bound に
      calibrate されていた → native regime では margin +1.410 (translation
      +1.002 から拡大)、両 regime で validate 完了
    - F3.3 — 言語 coverage: en (native + translation) のみ、他 7 言語は
      merge 後 user feedback で順次検証 (false reject 報告時は
      `FilterConfig(avg_logprob_threshold=None)` opt-out 可能)
- **Out of scope (次の handle)**:
  - **Canary** filter 対応: **PR-A.4.2** (NeMo `EncDecMultiTaskModel`、
    beam→greedy decoding 切替が gate)
  - **qwen3asr / reazonspeech / parakeet_en**: **PR-A.5** (wrapper bypass /
    vLLM 移行 / sherpa-onnx 構造的限界などの heavy refactor 系)
  - **Voxtral non-English language** での smoke verify: user feedback で
    順次対応 (本 PR は `language="en"` で実施)

  > _Superseded by PR-A.4.docs_ ([#316]): `parakeet_en` は PR-A.5 から外され
  > **PR-A.4.3 candidate** に格上げ済 (probe で `token_confidence_mean = 0.2452`
  > 確認、threshold 0.005 の 49×)。PR-A.5 は qwen3asr / reazonspeech の 2
  > engine に縮減。詳細は最上段 PR-A.4.docs entry を参照。

#### Confidence filter calibration sweep + new `post_filter_hallucination_rate` metric (Issue [#308] PR-A.3)

PR-A.1 ([#310]) で実装した confidence filter を 54 cell sweep
(1 preset × 3 backend × 3 engine × 2 corpus × 3 filter_mode) で validate
し、Phase 1 epic ([#295]) closure を数値証拠付きで achievable な状態に。

- **Before**: PR-A.1 で filter 本体は実装済だが、production 全 cell での効果は
  smoke verify (6 clip × 2 engine = 12 ケース) のみで実証。`benchmarks/
  non_speech_filter/` の既存 metric (`false_asr_trigger_rate` / `non_empty_
  hallucination_rate`) は **engine の生出力** を測定しており、filter 適用後
  の user の subtitle stream に届く text は計測できなかった。
- **After**:
  - `benchmarks/non_speech_filter/runner.py` の `NonSpeechFilterBenchmark
    Config` に `filter_config: Optional[FilterConfig] = None` を追加、
    `_make_pipeline_factory()` で `build_pipeline()` に pass-through。
  - `benchmarks/non_speech_filter/sweep.py` の `run_sweep()` に
    `filter_mode ∈ {off, observe, on}` の 3 段 nested loop を追加。
    `SweepCellResult` に `filter_mode` field 追加、CSV / Markdown 出力に
    column 追加。
  - **新 metric `post_filter_hallucination_rate`** を `evaluate_pipeline()`
    に追加。`transcriber.finalize()` 戻り値 + `_result_queue` の直接 drain
    (`InterimResult` を明示的に skip し `TranscriptionResult` のみ収集)
    を合算することで、user の subtitle stream に実際に届く text を計測する。
    `non_empty_hallucination_rate` (pre-filter engine 直出力) と並列で出力。
    旧版 (initial commit) は `finalize()` 戻り値を取りこぼし、queue drain も
    interim 先頭で停止する 2 件の bug があったため、codex-review on #312
    1st + 2nd round で修正済。
  - **新 metric `post_filter_speech_recall` / `post_filter_short_utterance_recall`**
    を追加 (codex-review on #312 3rd round Item 1 HIGH)。旧 `speech_recall`
    は engine call の counter で計測されており、filter が legit speech を
    drop しても 1.0 のままだった。新 metric は user の subtitle stream に
    届く speech 比率を直接測定。`measure_hallucination=True` 時のみ意味あり。
  - `_collect_post_filter_texts` helper を追加 (codex-review #312 3rd round
    Item 2 MED)。`_result_queue` 直接 access を helper に閉じ込め、将来の
    StreamTranscriber queue 実装変更時の修正箇所を 1 箇所に集約。
  - `docs/benchmarks/pr-a-calibration-2026-06-10.md` 新規 — PR-A 系列
    (A.0/A.1/A.3) の calibration 総括 doc を PR-B (2026-06-07) と同じ
    Setup / Hypotheses / Findings / Decision / Implications / Reproducibility
    構造で執筆。
- **Migration**: 既存 sweep を本 PR の harness で再実行すると、新 column
  `post_filter_hallucination_rate` が CSV / Markdown に追加されている。
  既存の `non_empty_hallucination_rate` semantics は不変 (pre-filter
  engine 出力を測定)、解釈時には 2 列を比較して filter 効果を測る。
- **Findings (詳細は `docs/benchmarks/pr-a-calibration-2026-06-10.md`)**:
  - H1 ✅ — `webrtc × parakeet_ja × real desk_tap` filter `on` で
    50% → 0%、`synthetic` でも 75% → 0% を実証。Issue #295 の元 motivation を
    実機で完全解決。
  - H1.b ✅ — synthetic corpus で WhisperS2T 内部 `no_speech_prob` filter
    を bypass する edge case (25%) も filter `on` で 0% に。重複防御として
    実効的に機能。
  - H2 ✅ — `silero / tenvad × all engines` で filter mode に関係なく
    0% 維持 (production user の副作用ゼロ)。
  - H3 ✅ (v3 refined) — **real corpus** で post-filter SR = 100% 維持
    (filter は legit speech を 1 件も drop していない)。synthetic positive
    の SR(post) drop は filter が formant proxy を正しく低信頼度として
    drop している = 期待挙動。production user は real speech を扱うため
    real corpus の結果が production 挙動。
  - H4 — `BASELINE_INVARIANTS` は不変判断。CI test は synthetic + Mock
    Engine で filter は fail-open のため tighten 不要。
  - ReazonSpeech — `engine_confidence` 全 None で filter fail-open。
    `post_filter = pre_filter` で filter 効果なし (sherpa-onnx 構造的限界、
    PR-A.5 で長期対応)。
- **Side effects**:
  - `benchmarks/non_speech_filter/report.py::NonSpeechFilterRunRecord`
    に `post_filter_hallucination_rate: float | None = None` field 追加。
    既存 caller は default None で動作するため後方互換。
  - 既存 sweep test 全 pass 維持。新規 sweep axis test 5 件追加で
    filter_mode 軸の挙動を pin。
  - `CHANGELOG.md` の本 entry で PR-B との比較を可能に。
  - Phase 1 epic ([#295]) close 候補状態に到達 — 残作業は PR-A.4
    ([#311] qwen3asr/voxtral/canary の filter 拡張、別 track)。

#### Engine confidence filter — default ON (Issue [#308] PR-A.1)

Adds the post-ASR `livecap_cli.transcription.confidence_filter` module
that watches `engine_confidence` (PR-A.0) and silently drops outputs the
engine itself judged as non-speech, before they reach the subtitle stream.
Default is **on** for all realtime sessions.

- **Before** (PR-A.0): every engine output reached the subtitle stream,
  even when `no_speech_prob` was high (WhisperS2T) or
  `token_confidence_mean` was near zero (Parakeet_ja). The PR-B 144-cell
  matrix showed `webrtc × parakeet_ja × desk_tap` hallucinated 50 % of
  the time.
- **After** (PR-A.1): `StreamTranscriber` calls `apply_filter()` on the
  3 call sites (sync L566 / async L638 / interim L787). For
  `--confidence-filter on` (default), rejected outputs become `None`
  drops with a structured INFO log; the subtitle stream sees nothing.
  Real-machine smoke verify on the 6-clip PR-B corpus produced 100 %
  classification (all speech clips passed, both non-speech clips dropped
  on both whispers2t and parakeet_ja).
- **Migration**: existing scripts keep the same flags. To restore the
  previous behavior, pass `--confidence-filter off` or export
  `LIVECAP_CONFIDENCE_FILTER=off`. Engines that do not expose confidence
  signals (`reazonspeech`, `qwen3asr`, `voxtral`, `canary`) are
  pass-through regardless of the flag (fail-open by design).
- **Side effects**:
  - Per-engine thresholds (`whispers2t no_speech_prob > 0.5`,
    `parakeet_ja token_confidence_mean < 0.005`) are baked in from the
    PR-A.0 verify values; PR-A.3 will revisit them after a full 144-cell
    sweep.
  - The `--confidence-filter observe` mode emits the same structured
    decision log **for both pass and reject decisions** (codex-review
    #310 Item 4) — PR-A.3 calibration needs the speech-side
    `engine_confidence` distribution as well, not just the reject side,
    to evaluate threshold margins and speech-recall safety. The `on`
    mode keeps logging reject only to avoid production log spam. Log
    payload is JSON (stable schema documented in `_decision_to_dict()`
    of `confidence_filter.py`) so PR-A.3 parsers can read it as JSONL.
  - The `StreamTranscriber.__init__` gained a `filter_config:
    Optional[FilterConfig] = None` parameter; `None` constructs the
    default (on) at instantiation time. Direct API users who want the
    old behavior should pass `FilterConfig(mode="off")` explicitly.
  - `benchmarks/non_speech_filter/pipeline.py::build_pipeline` defaults
    to `FilterConfig(mode="off")` so existing sweep baselines remain
    bit-identical; PR-A.3 will pass `FilterConfig(mode="on")` to
    measure filter impact on the cell matrix.
  - **Scope clarification (codex-review #310 Item 3)**: this PR exposes
    `filter_config` on `build_pipeline()` only. Adding a
    `confidence_filter` axis to `benchmarks/non_speech_filter/sweep.py`
    (so that the existing preset/backend/engine matrix is multiplied by
    `{off, observe, on}`) is **deferred to PR-A.3**, together with the
    full 144-cell sweep run. The pipeline-level hook here is sufficient
    for PR-A.3 to construct sweep cells programmatically.
  - A startup INFO log line (`"Confidence filter: ON (...)"`) makes the
    active mode visible at every session start.

#### Parakeet_ja decoder strategy: RNNT greedy → CTC greedy_batch (Issue [#308] PR-A.0)

Investigation during PR #309 smoke verify uncovered that the
`nvidia/parakeet-tdt_ctc-0.6b-ja` checkpoint is an
`EncDecHybridRNNTCTCBPEModel` whose RNNT decoder (NeMo default) does not
implement `token_confidence`. The CTC decoder does. To make
`engine_confidence` actually populated for `parakeet_ja`, the adapter now
switches to the CTC decoder on `load_model`.

- **Before**: `parakeet_ja` used the RNNT decoder with `strategy=greedy`
  and the old `confidence_cfg` block — which NeMo silently rejected on
  the current version (`preserve_frame_confidence=True` requires
  `preserve_alignments=True`), so `token_confidence` was always `None`
  and the old `score / len(y_sequence)` fallback returned an
  empirically-inverted signal (speech `-71.5` vs applause `-47.3`).
- **After**: `_configure_decoding_with_confidence()` detects the hybrid
  model via `hasattr(self.model, 'cur_decoder')` and switches to
  `decoder_type='ctc'` with `strategy=greedy_batch`,
  `greedy.preserve_frame_confidence=True`, and a full `confidence_cfg`.
  `token_confidence_mean` is now populated with clean speech-vs-noise
  separation (0.01-0.10 vs 0.0000029-0.0003). A 3-stage fallback path
  protects against older NeMo versions and the non-hybrid English
  `parakeet` model.
- **Migration**: `EngineMetadata.default_params["parakeet_ja"]
  ["decoding_strategy"]` updated from `"greedy"` to `"greedy_batch"`
  (single source of truth, surfaced by GUI / diagnostics / docs). The
  English `parakeet` (pure RNNT) default remains `"greedy"`. Users who
  hard-coded `decoding_strategy="greedy"` on `parakeet_ja` will keep
  working (CTC greedy is slower but functional, NeMo emits a
  `greedy_batch` recommendation warning).
- **Side effects** measured on RTX 4090 with the
  `.tmp/non_speech_corpus/` 6-clip set:
  - **Latency** improves: CTC + `greedy_batch` runs 1.83× faster on the
    speech clip than the old RNNT `greedy` path (p50 81.4 ms vs
    149.8 ms).
  - **Transcription text** is preserved on 4/6 clips, slightly improved
    on 1/6 (`applause_5_claps` hallucinates fewer characters), and
    differs by 1 hiragana on 2/6 (e.g. 「とんと」 → 「とんど」). Not
    a regression for production usage; documented in
    `docs/research/parakeet-ja-confidence-spec-2026-06-10.md`.
- **Score fallback removed**: the previous
  `score / len(y_sequence) → avg_logprob` path inside
  `_extract_engine_confidence` is gone. When `token_confidence` is not
  available (older NeMo, non-hybrid model, or fallback path), the
  function returns `EngineConfidence()` honestly. The smoke-verified
  signal inversion made the old fallback actively harmful for the
  PR-A.1 filter.

This is an **engine behavior change** for `parakeet_ja`, but it
strengthens (not regresses) production behavior on every measured axis:
faster, comparable text, and a now-usable confidence signal.

#### Phase 2 SED model evaluation harness (Issue [#305] PR-D0)

- **New `benchmarks/sed/` package (research-only off-line evaluation;
  does not touch `livecap_cli/`)**:
  - `class_mapping.py` — pins the AudioSet 527-class taxonomy mapping for
    livecap reject signals. Defines `TARGET_CLASSES` (Hands / Finger
    snapping / Clapping / Applause / Door / Sliding door / Slam / Knock
    / Tap / Thump, thud — 10 classes) and `SPEECH_LIKE_CLASSES` (Speech
    family + Singing — 7 classes). Implements the three Issue #305 v3
    threshold policies (`max`, `sum`, `target_minus_speech`).
  - `inference.py` — loads EfficientAT pretrained models
    (`mn04_as` / `dymn04_as` / `dymn10_as`), resamples 16 kHz audio to
    32 kHz, slices into 1-second windows (Issue #305 v3 primary metric
    unit), and returns per-window 527-dim sigmoid probability matrices.
  - `metrics.py` — clip-level confusion-matrix metrics with hand-pinned
    semantics: class-level + reject-signal-level (Issue #305 v3 two-axis
    report), provisional-gate verdict (`precision ≥ 0.70` AND
    `recall ≥ 0.50` AND target clip flagged at the chosen threshold).
  - `latency.py` — 5-axis runtime measurement (checkpoint size, installed
    dep delta vs `engines-torch` baseline, runtime peak memory via
    `tracemalloc`, CPU + GPU p50/p95 latency, cold-start) per Issue #305
    v3 Dimension 3 refinement.
  - `orchestrator.py` — full evaluation pipeline (corpus → inference →
    CSVs + NPZ + JSON metadata).
  - `analyze.py` — post-hoc analysis: threshold sweep, class-level
    summary tables, provisional-gate verdict, `analysis.{json,md}` for
    decision-doc paste-in.
  - `cli.py` + `__main__.py` — `python -m benchmarks.sed` entry point.
  - `README.md` — manual EfficientAT setup, env vars, command reference.
- **New `tests/integration/sed/` (23 tests, `sed_evaluation` marker)**:
  - `test_class_mapping.py` (12 tests) — AudioSet index integrity, three
    policy semantics, validation; the
    `test_indices_match_efficientat_csv` test cross-checks the pinned
    indices against the canonical AudioSet CSV when EfficientAT is
    cloned.
  - `test_metrics.py` (10 tests) — synthetic 4-clip corpus with
    hand-derived precision/recall, gate truth-table (pass / precision
    fail / recall fail / target-not-flagged).
  - `test_inference_smoke.py` (1 test) — env-gated 1-clip smoke
    verifying `(n_windows, 527)` output shape; skipped automatically
    when the EfficientAT clone is absent.
  - New `sed_evaluation` pytest marker declared in `pyproject.toml`.
- **New `docs/research/phase2-sed-evaluation-2026-06-10.md` (~430
  lines)** — 4-dimension decision document covering Accuracy / Safety /
  Runtime / License (verdicts honest after codex-review on #306):
  - Accuracy: PASS (provisional). `target_minus_speech` policy at
    threshold ~0.10 yields precision=1.0, recall=1.0 on the 6-clip
    corpus; the critical `overlapping_applause_speech` case is correctly
    retained (`max(target)=0.16` would over-fire, but
    `target − speech_like = -0.66` correctly suppresses).
  - Safety: PASS. speech_recall = 1.00, short_utterance_recall = 1.00.
  - Runtime: **Conditional PASS** (CPU production-device path). CPU p95
    = 29.0 ms (3.4× under the 100 ms budget), checkpoint 4.07 MB
    (12× under 50 MB), runtime peak 6.68 MB (30× under 200 MB).
    **GPU p95 = 32.8 ms misses the original 30 ms ceiling by 9 %** —
    documented honestly rather than papered over; CPU runs faster than
    GPU at this 3.9 M-parameter scale, so production device = CPU and
    the CPU budget is satisfied.
  - License: PASS at the **Auto-download OK tier** (not Bundle OK —
    corrected after codex-review). The upstream EfficientAT release
    does not explicitly grant a license on the model weights, so the
    integration ships the checkpoint via `torch.hub` auto-download
    rather than packaging the `.pt` file; this matches both the legal
    evidence we have and the implementation already in use.
    Attribution stub recorded for PR-D1's `THIRD_PARTY_NOTICES.md`.
- **New `benchmark_results/sed/2026-06-10/` (committed per Issue #305 v3
  artifact policy)**: `probabilities.csv`, `probabilities_full.npz`,
  `latency.csv`, `metadata.json`, `analysis.json`, `analysis.md`.
- **`.gitignore` update**: changed `benchmark_results/` to
  `benchmark_results/*` with `!benchmark_results/sed/` exception so the
  PR-D0 evidence is committed while other benchmark outputs remain
  ignored. Added `.tmp/` to ignore the EfficientAT clone and any
  research scratch.
- **Issue #305 v2 → v3 body update** with six clarifications: metric
  calculation unit (window primary / clip-level max decision unit),
  license outcome 4-classification (Bundle OK / Auto-download OK /
  Manual user-provided only / NG), artifact commit policy, accuracy
  provisional-gate disclaimer, runtime constraint detailing
  (checkpoint / installed dep / runtime peak memory split), class-level
  + reject-signal-level two-axis metric report.
- **Scope discipline**: this PR does **not** modify any file under
  `livecap_cli/`. SED pipeline integration is PR-D1; default-decision
  is PR-D2; DSP detector disposition is PR-D2.

Verification: `pytest tests/integration/sed/` → 23 passed, 0 failed.
PR-relevant regression
(`tests/audio tests/integration/non_speech_filter tests/integration/vad
tests/transcription tests/core/cli tests/audio_sources`) → 307 passed,
5 skipped (env-gated), 0 failed — identical to the pre-PR baseline.

[#305]: https://github.com/Mega-Gorilla/livecap-cli/issues/305

#### Calibration follow-up: real-engine sweep + threshold tuning (Issue [#295] PR-B follow-up)

- **3 new hypothesis-driven candidate presets** appended to
  `benchmarks/non_speech_filter/sweep.py::default_named_presets()`:
  - `on_relaxed_rms` — drop `rms_min_db` floor from -35 to -45 to admit
    quieter real-corpus frames (real recordings sit at -41 to -46 dBFS
    overall, so the default floor was rejecting > 95 % of frames before
    the AND combination could fire).
  - `on_low_freq_aware` — widen the spectral centroid window
    (`centroid_min_hz` 2500 → 500) and tighten `voiced_max` (0.25 →
    0.15) to test whether `desk_tap`-style low-frequency thumps can be
    caught without dropping low-pitched speech.
  - `on_speech_safe` — tightest preset (`flatness_min` 0.45,
    `centroid_min_hz` 3000, `onset_ratio` 5.0) as a safety ceiling that
    confirms short-utterance recall stays at 100 % under aggressive
    filtering.
- **New `benchmarks/non_speech_filter/calibration.py` (~430 lines)**:
  reads the CSV emitted by `sweep.py` and produces a structured Markdown
  report containing (1) per-engine hallucination delta vs `baseline_off`
  (segmented by backend and corpus), (2) recall-regression flags for any
  preset/cell pair that dropped recall below the baseline, (3) a Pareto
  summary across presets with explicit dominance markers, and (4) a
  structured recommendation driven by Issue #295 PR-B follow-up plan
  rule D4 (≥30 % hallucination drop on `webrtc × parakeet_ja × real`
  with no recall regression → promote that preset; otherwise default
  off, document gap, propose Phase 2 SED).
- **Calibration findings** (full record in
  `docs/benchmarks/calibration-results-2026-06-07.md`):
  - 144 cells (8 presets × 3 backends × 3 engines × 2 corpora) ran in
    ~16.5 min on a single RTX 4090 with engine-load amortisation.
  - **`parakeet_ja × WebRTC × real desk_tap` hallucination unchanged
    at 50 % across all 8 presets** (the PR-B v4 AC target).
  - Same on `reazonspeech × WebRTC × real desk_tap` (50 % → 50 %).
  - `parakeet_ja × WebRTC × synthetic burst` hallucination drops
    75 % → 62.5 % (one item out of eight) on the 4 Pareto-dominant
    presets (`on_moderate`, `on_aggressive`, `on_relaxed_rms`,
    `on_low_freq_aware`).
  - **Zero recall regressions** in any of the 144 cells.
- **Default mode decision: `--transient-filter=off` is maintained.**
  Rule D4's headline criterion (≥30 % hallucination drop on the AC
  target cell) is unmet, so no preset earns a promotion to default.
- **`on_moderate` is documented as the best observed DSP preset for
  synthetic rapid-burst tests only** — explicitly **not** a production
  hallucination mitigation recommendation. Calibration showed zero
  improvement on the real-corpus target cell.
- **The DSP transient detector layer is positioned as `experimental`
  going forward**: not deprecated (no replacement exists yet) but not a
  production-hallucination-mitigation candidate. CLI invocations of
  `--transient-filter observe/on` now emit a one-line experimental
  notice to make the status visible at the moment of opt-in. Phase 2
  SED (sound-event detection) is the planned successor for
  `desk_tap`-style low-frequency transients.
- **New `docs/audio-filter-reference.md`**: user-facing reference for
  every audio filter in the pipeline (NoiseGate / TransientDetector /
  VAD / EnergyGate) — purpose, pipeline position, CLI surface, default
  state, measured effectiveness with citations, recommendation, known
  limitations. Single doc users can scan to decide which filter to
  enable.
- **No detector code change**. The sweep + analysis is pure data
  collection; this PR does not modify
  `livecap_cli/audio/transient_detector.py` or any production pipeline.
- **Issue #295 v6** reframes the PR-B AC line `WebRTC × desk_tap (real)
  false_trigger 50 % → 0 %` with the empirically demonstrated bound
  ("0.0 pp achievable with 6-feature AND DSP detector — Phase 2 SED is
  the correct route").
- **BASELINE_INVARIANTS bounds remain unchanged** (default unchanged
  → no tightening warranted).
- **Out of scope** (separate follow-ups): Phase 2 SED epic for low-
  frequency / non-broadband transient detection; detector architecture
  changes (AND → OR, weighted-sum, new features); `#302` lookahead
  delay (still gated on a future reject default ON decision that this
  calibration data argues against).
- Verification: full PR-relevant suite (`pytest tests/audio/
  tests/integration/non_speech_filter/ tests/integration/vad/
  tests/transcription/ tests/core/cli/ tests/audio_sources/`) → **300
  passed, 5 skipped (env-gated), 0 failed**.

#### Fixed: PR-B follow-up — async path bypass + causal best-effort spec (Issue [#295] PR-B follow-up)

- **`StreamTranscriber.transcribe_async()` が transient detector を bypass**
  していた問題を修正。`feed_audio()` と `transcribe_async()` の pre-VAD
  処理を `_apply_pre_vad_processing()` 共通 helper に集約し、両 path が
  必ず NoiseGate → Layer 1 detector → VAD の順で走るよう pin。
- **`tests/transcription/test_stream.py::TestTransientDetectorWiring`**
  を追加 (3 cases): sync/async 両 path の detector 起動、両 path の
  telemetry 完全一致を assert。再発防止層。
- **`TransientDetector.process()` docstring** に causal / no-lookahead
  仕様を明文化。`on` mode の chunked output は best-effort upper bound で
  あり、bit-exact reconstruction ではないことを記載。32 ms lookahead-
  delay 拡張は別 issue で track。
- **`tests/audio/test_transient_detector.py::TestStreamingEquivalence::
  test_on_mode_chunked_is_causal_best_effort`** を追加: telemetry が
  full / chunked で一致することと、`on` mode で flagged frame があれば
  energy が入力より低下することを assert (full vs chunked output の
  bit-exact equality は意図的に検証しない)。
- **default mode を `off` のまま維持** + ドキュメント整合: Issue #295
  / docs / CHANGELOG では旧 v3/v4 で「default observe」と記載していた
  が、PR-B 実装は安全側の `default off` を採用。calibration は
  `--transient-filter observe` で明示的に opt-in する運用を明記。
- 既存挙動への影響: なし (default `off` で detector 構築されないため)。
- Verification: `pytest tests/audio/test_transient_detector.py
  tests/transcription/test_stream.py tests/integration/non_speech_filter/`
  → 86 passed, 8 skipped (env-gated).

#### Layer 1: DSP Transient/Applause Detector (Issue [#295] PR-B)

- **新規 `livecap_cli/audio/transient_detector.py`**: 6 DSP feature
  (`spectral_flatness` / `spectral_centroid_hz` / `zero_crossing_rate` /
  `onset_strength` / `voiced_ratio` / `rms_db`) を AND 結合して
  applause-like フレームを検出する frame-based stateful detector。
  3 mode: `off` (構築されない) / `observe` (telemetry のみ、audio 不変) /
  `on` (applause-flag frame を zero-out)。
- **`StreamTranscriber` 統合**: 新引数
  `transient_detector: Optional[TransientDetector] = None`、`feed_audio`
  の NoiseGate 後 / VAD 前で起動。`reset()` / `close()` テレメトリにも
  対応 (EnergyGate と同じ pattern)。
- **CLI flags** (`transcribe` サブコマンド): `--transient-filter`
  (`off`/`observe`/`on`、default `off`) + 6 threshold flag。
- **Benchmark CLI flags** 同名で揃え、`build_pipeline()` に
  `transient_config` kwarg を追加。`None` を渡せば baseline pipeline は
  bit-identical に保たれる (PR-0 BASELINE_INVARIANTS regression なし)。
- **新規 `benchmarks/non_speech_filter/sweep.py`**: 5 named preset
  (`baseline_off` / `observe_defaults` / `on_conservative` /
  `on_moderate` / `on_aggressive`) を回す threshold sweep harness。
  CSV + Markdown 出力。
- **テスト**:
  - `tests/audio/test_transient_detector.py`: 26 unit test
    (feature 算出 / AND 決定 / streaming 等価性 / mode semantics /
    config validation)。
  - `tests/integration/non_speech_filter/test_transient_detector_integration.py`:
    7 integration test (observe = no-op on metrics, on-mode positive
    preservation, WebRTC burst no-regression)。
- **検証結果 (private real corpus + synthetic、mock engine)**:
  - observe mode は BASELINE_INVARIANTS と完全一致 (silero 0/0/0 %,
    tenvad 25/100/100 %, webrtc 75/100/100 %)。
  - on mode (moderate/aggressive) で **WebRTC × synthetic burst の
    false_trigger 75 % → 62.5 %**。
  - WebRTC × real desk_tap は default 閾値で 50 % のまま (per-clip 観測で
    `centroid_min_hz=2500` が desk_tap の低域成分を弾いていることを確認、
    calibration follow-up で対応)。
- **限界 (docs に明示)**:
  - 既定閾値は **synthetic rapid burst 想定**で、private real corpus の
    個別 clip (desk_tap、scattered 拍手) は未較正。
  - reject default ON 化は PR-B のスコープ外 — calibration sweep の
    結果が出てから別 PR で実施する設計。
- **Out of scope**:
  - PR-C で予定する Layer 2 (VADStateMachine cooldown) との signaling は
    実装しない (検出器は event を emit するが消費側は別 PR)。
  - PR-A 系の confidence filter / prompt reset は対象外。

#### Non-speech filter evaluation harness (Issue [#295] PR-0)

- **新規 `tests/integration/non_speech_filter/`**: Phase 1 多段防御
  (DSP transient detector / VADStateMachine cooldown 拡張 / Confidence filter /
  Prompt reset) の **baseline + regression 検出基盤** を導入。3 VAD backend
  (Silero / TenVAD / WebRTC) × 13 件の synthetic corpus (negative 8 + positive 5、
  うち短発話 2) で現状 pipeline (NoiseGate + VAD + EnergyGate) を計測し、
  `baselines/{backend}.json` に schema v1 で永続化。後続 PR-B/C/A はこの JSON を
  比較基準にする。
- **新規 `benchmarks/non_speech_filter/`**:
  `python -m benchmarks.non_speech_filter` で ad-hoc 評価可能な runner +
  Markdown/JSON レポート。`--engine whispers2t` 等を指定すれば
  `non_empty_hallucination_rate` (engine が非空 text を返した負例の割合) も計測。
  実音源は `LIVECAP_NON_SPEECH_CORPUS_DIR` で manifest+WAV を渡すと自動 load。
- **指標**: `false_asr_trigger_rate` / `speech_recall` /
  **`short_utterance_recall`** (最重要) / `non_empty_hallucination_rate` (opt-in) /
  `added_latency_p50_ms` / `_p95_ms`。
- **新 marker**: `evaluation_harness` (pyproject.toml に登録)。`-m evaluation_harness`
  で opt-in 実行、CI baseline tests のみ拾う。
- **既存コード touch ゼロ**: `livecap_cli/` 配下は無変更。`benchmarks/common/`
  (`DatasetManager` / `BenchmarkEngineManager`) は real-engine 経路でのみ利用。
- **動機**: Issue #295 v2 のレビュー指摘
  「**評価ハーネス先行整備** + pre-engine 優先 + DSP detector default off-by-default
  + 実装前 corpus 整備」を満たすため、Phase 1 PR-B/C/A の前提として独立着地させる。
- **限界 (docs/benchmarks/non-speech-filter.md に明記)**:
  - 合成 speech proxy は Silero VAD (実音声学習) で recall=0 になる構造的限界。
    Silero baseline を意味のある形で測るには実音源 fixture が必要。
  - WebRTC backend は binary 出力 (0.0/1.0) のため、Phase 1 PR-C で導入予定の
    hysteresis は no-op (duration-based cooldown のみ機能)。
- **検証**:
  - `uv run pytest tests/integration/non_speech_filter/ -m evaluation_harness`
    → 6 passed (3 backend × 2 tests), 6 skipped (env-var gated)。
  - `uv run python -m benchmarks.non_speech_filter --mode quick --backend silero,tenvad,webrtc`
    → JSON + Markdown 出力、Silero / TenVAD / WebRTC の baseline 差を可視化。
  - 既存 `tests/integration/vad/` + `tests/audio/` の 74 test に regression なし。

#### **BREAKING** `StreamTranscriber` に engine-input low-energy gate (EnergyGate) を追加 (Issue [#292])

- **Before**: VAD segment は energy 不問で全て `engine.transcribe()` に渡され、低 RMS / 純ノイズ segment で hallucination ("うん"/"ピッ"/"え?"/"どうぞ" 等) が発生。
- **After**: `StreamTranscriber` の 3 callsites (`_transcribe_segment` / `_transcribe_segment_async` / `_transcribe_interim`) で共通 helper `_should_skip_low_energy(audio, kind)` を呼び、per-segment energy が threshold 未満なら `engine.transcribe()` を skip。
- **動機**: `#291` (NoiseGate 単位ミスマッチ) は NoiseGate 有効時の primary fix だが、NoiseGate は opt-in (default off) で大半のユーザーは VAD のみで防御。VAD false-positive segment が engine に渡って hallucination する経路が残っていた (Mega-Gorilla/livecap-gui#331 の root-cause の一つ)。実音源 pre-evaluation で parakeet_ja は **silent audio に対して 100% hallucination** を確認、EnergyGate を経由すると -45 dBFS threshold で 26% 削減 (stress test)。production 条件 (VAD default threshold) では VAD が一次防御として効き、本機能は副次防御として機能する。
- **API 変更点 (`StreamTranscriber.__init__`)**:
  - 新引数: `engine_min_rms_dbfs: float = -45.0` — threshold (dBFS)。`float("-inf")` で完全 opt-out。
  - 新引数: `engine_energy_metric: str = "max_frame_rms"` — 4 metric から選択 (`max_frame_rms` / `whole_rms` / `p95_frame_rms` / `top3_frame_rms`)。default は VAD padding 希釈に耐性の `max_frame_rms`。
  - 新引数: `engine_energy_frame_ms: float = 32.0` — frame-based metric の窓長 (ms)。
- **CLI 変更**:
  - `transcribe` に `--engine-min-rms` / `--engine-energy-metric` / `--engine-energy-frame-ms` の 3 flag を追加。`--engine-min-rms` には custom type を実装し `off` / `disabled` / `none` 文字列を `float("-inf")` に map (argparse の leading-`-` value 制約のため bare `-inf` は不可、`=-inf` か `off` を使う)。
  - `levels` に `--engine-min-rms-margin` flag を追加 (default `+6 dB`)。`suggested_engine_min_rms_dbfs` の margin を user 任意に調整可能。
- **`NoiseAnalysis` 変更**:
  - 新 field: `suggested_engine_min_rms_dbfs: float` (= `noise_rms_p95_db + engine_min_rms_margin_db`)。CLI `levels` で 1 回の calibration から peak-unit / RMS-unit 両方の suggested 値が得られる。
  - `analyze_noise_samples()` に optional `engine_min_rms_margin_db` キーワード引数を追加 (default `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0`)。
- **新公開 API (`livecap_cli.audio`)**:
  - `ENGINE_MIN_RMS_SAFETY_MARGIN_DB = 6.0` (定数)
  - `ENERGY_METRICS = ("max_frame_rms", "whole_rms", "p95_frame_rms", "top3_frame_rms")`
  - `_segment_energy_dbfs(audio, sample_rate, metric, frame_ms) -> float` (helper、user-configurable metric/frame で per-segment energy を測定)
- **Telemetry**: `StreamTranscriber.close()` 時に drop counter の内訳 (`final_sync` / `final_async` / `interim`) を `logger.info` で 1 行サマリ。silent failure 防止。
- **Migration**:
  - 既存挙動を完全に維持したい場合: `StreamTranscriber(engine=..., engine_min_rms_dbfs=float("-inf"))` または CLI `--engine-min-rms off` で opt-out。
  - 通常はデフォルト (`-45.0`) で問題なく動作 (synthetic regression + 実音源プローブで通常会話・小声・ささやきレベル speech は pass を確認)。whisper 録音など特殊用途は閾値を下げる (`--engine-min-rms -50` 等) か opt-out。
- **検証**:
  - `tests/audio/test_analysis.py::TestSegmentEnergyDbfs` (10 cases) — 4 metric ごとの動作 / fallback / 物理的妥当性。
  - `tests/audio/test_energy_gate_regression.py` (新規 7 cases) — synthetic fixture (silent noise / speech-like burst / padded short utterance) で default threshold の drop/pass を assert。VAD padding 希釈に max_frame が耐性を持ち、whole_rms は希釈で false-drop することを documentation。
  - `tests/transcription/test_stream.py::TestEnergyGate` (10 cases) — 3 callsites (sync / async / interim) で `engine.transcribe()` が呼ばれない / 呼ばれることを mock の `call_count` で検証 + opt-out + invalid arg validation + close() log。
  - `tests/core/cli/test_cli.py::TestEnergyGateFlags` (12 cases) — `--engine-min-rms` の 4 parse パターン (numeric / off / disabled / =-inf) + invalid raises + metric choices + frame-ms parse + help text 可視性。
- **限界**:
  - EnergyGate は **silver bullet ではない**。実音源プローブで parakeet_ja は 73 silent windows で 100% hallucinate し、-45 dBFS threshold (max_frame_rms) で削減できるのは 26% のみ (transient を含む noisy silence は max frame で pass する)。完全防御には `--noise-gate` との併用 + VAD threshold 適切化が必要。
  - Engine choice が hallucination 耐性に大きく影響 (Parakeet 100% vs ReazonSpeech は known 低い)。
- **将来 follow-up**:
  - `BaseEngine` 側へのガード追加 (`StreamTranscriber` を経由しない advanced user 向け)
  - top-k metric の k=3 から user-configurable に拡張

#### **BREAKING** `NoiseAnalysis` / `analyze_noise_samples()` を peak-based calibration に置換 (Issue [#291])

- **Before**: `analyze_noise_samples(samples_db, sample_rate_hz)` →
  `suggested_threshold_db = noise_peak (chunk RMS p95) + 10 dB`
- **After**: `analyze_noise_samples(samples_db, peak_samples_db, sample_rate_hz)` →
  `suggested_threshold_db = peak_p95 (per-chunk |x|.max() p95) + 6 dB`
- **動機**: `NoiseGate` (`livecap_cli/audio/noise_gate.py`) の envelope follower
  は per-sample peak を追跡するが、calibration は chunk RMS を計測していた → unit
  mismatch により impulsive noise (キーボード/呼吸/breath bursts) で threshold が
  peak の下に潜り、無音時 hallucination ("あ"/"うん"/"ピッ") を引き起こしていた
  (Mega-Gorilla/livecap-gui#331 root-cause)。White noise の crest factor ≈ 11 dB
  が偶然 `+10` で吸収されていたが、impulsive noise では crest factor がより大きく
  破綻する。
- **API 変更点**:
  - `NoiseAnalysis` 新 field: `peak_p95_db: float` (per-chunk `|x|.max()` の 95%ile)
  - `NoiseAnalysis` 改名: `noise_peak_db` → `noise_rms_p95_db` (unit を field 名に明示)
  - `NoiseAnalysis` 削除: `safe_zone_min_db` (新 `suggested_threshold_db` と 1 dB 差で意味崩壊)
  - `analyze_noise_samples()` 新 required 引数: `peak_samples_db` (Optional=None の
    legacy default は無し: pre-1.0 backward-compat policy で旧バグ温存の flag は不可)
  - 新 module-level 定数: `livecap_cli.audio.PEAK_SAFETY_MARGIN_DB = 6.0`
  - `danger_zone` は据え置き (RMS-unit diagnostic として docstring で明記)
- **Migration**:
  ```python
  # 旧
  rms_db_list = [20*log10(rms(chunk)) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list)

  # 新
  rms_db_list  = [20*log10(rms(chunk))    for chunk in chunks]
  peak_db_list = [20*log10(|chunk|.max()) for chunk in chunks]
  a = analyze_noise_samples(rms_db_list, peak_db_list)
  # a.noise_peak_db    -> a.noise_rms_p95_db
  # a.safe_zone_min_db -> (削除; suggested_threshold_db を直接使用)
  # a.peak_p95_db      -> 新 field; threshold の基準
  ```
  CLI `levels` は内部で per-chunk peak を収集するように移行済みのため、
  外部から CLI を呼ぶ場合の変更は不要 (JSON schema のみ変更)。
- **検証**: `tests/audio/test_noise_gate_calibration.py` (新規) で synthetic
  impulsive noise を旧 / 新 threshold それぞれで NoiseGate に通し、旧で gate
  が開く / 新で閉じ続けることを assert する end-to-end 回帰テストを追加。
- **GUI ペア PR**: livecap-gui 側は `core/noise_statistics.py` を削除し本 API
  に委譲する PR を別 issue (Mega-Gorilla/livecap-gui#335 の対) で受ける。
- **将来 follow-up**: NoiseGate の envelope follower を calibration 入力に対して
  simulate し envelope の 95%ile を取れば margin を 1-2 dB に縮められる ([#283]
  と組で別 issue 化予定)。

#### **BREAKING** `NoiseGate` デフォルト `release_ms` 変更 (Issue [#283] PR C)

- **Before** (PR #279 / PR #281 / PR #282): `release_ms=30`
- **After**: `release_ms=100`
- **動機**: PR #282 で導入された hard-mute と短い release の組み合わせが、aggressive な閾値 (-25/-17 dB) で whisper 系エンジンの fragmentation ハルシネーション (「んんん...」loop) を引き起こす。A/B 実測で `release_ms=100` により完全解消を確認 (316→102, 299→96 chars)
- **Migration**: 旧挙動を明示的に再現するには `release_ms=30` を直接渡す:
  ```python
  NoiseGate(release_ms=30)  # pre-PR-C default
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate --noise-gate-release 30 ...
  ```
- **検証結果**: `docs/benchmarks/noise-gate-ab.md` に更新後のテーブル掲載

#### **BREAKING** `NoiseGate` 既定挙動変更 (Issue [#280] PR B)

- **Before** (PR [#279] / PR [#281]): 単一閾値 + `-60 dB` soft-mute
- **After** (PR B): 自動ヒステリシス (`threshold_db - 6 dB`) + hard-mute (出力ゼロ)
- **動機**: PR #281 の A/B 検証で、PR #281 までの挙動が whisper 系エンジンで flicker によるハルシネーション暴走 ("どうもどうも..." 等) を引き起こすことが実証された
- **Migration**: 過去挙動を明示的に再現するには以下を指定:
  ```python
  NoiseGate(
      threshold_db=-35,
      close_threshold_db=-35,  # single-threshold (ヒステリシス無効)
      noise_floor_db=-60,      # soft-mute
  )
  ```
  CLI の場合:
  ```bash
  livecap-cli transcribe --noise-gate \
      --noise-gate-threshold -35 \
      --noise-gate-close-threshold -35 \
      --noise-gate-floor -60 \
      ...
  ```

新規オプション (既存呼び出しは無変更で動作、挙動のみ変化):

- `NoiseGate` / `transcribe` CLI に `close_threshold_db` / `--noise-gate-close-threshold` を追加 (ヒステリシス制御)
- `NoiseGate` / `transcribe` CLI に `noise_floor_db` / `--noise-gate-floor` を追加 (ゲート閉鎖時の減衰量制御)
- 初期化ログが resolved 値 (open/close/noise_floor) を出力するように改善 (ポリシー準拠)

既知の follow-up:

- `release_ms=30` は PR B の新しい gate 挙動 (hard-mute による clean silence) に対して短すぎるため、攻撃的な閾値で fragmentation hallucination が発生することがあります。`--noise-gate-release 100` または `200` で回避可能。デフォルト値の変更は別 issue で対応予定。

#### Breaking Changes

| Before | After |
|--------|-------|
| Package: `livecap-core` | Package: `livecap-cli` |
| Module: `livecap_core` | Module: `livecap_cli` |
| CLI: `livecap-core --info` | CLI: `livecap-cli info` |
| CLI: `livecap-core --as-json` | CLI: `livecap-cli info --as-json` |

#### API Changes

- `TranscriptionEventDict` → `TranscriptionResult` dataclass
- Engine creation unified to `EngineFactory.create_engine()`
- VAD configuration via `VADConfig` instead of dict
- `detect_device()` returns `str` instead of `Tuple` ([#175])

### Deprecated


- `TranscriptionEventDict` (use `TranscriptionResult`)
- `languages.py` module (use `langcodes` for BCP-47) ([#173])

### Removed


- Old flag-based CLI interface (`--info`, `--ensure-ffmpeg`, `--as-json`)
- `livecap-core` entry point
- `livecap_core` module name
- `Languages.get_engines_for_language()` (use `EngineMetadata`) ([#171])

#### 旧 download helper の削除と ASCII 保証境界への移行 (Issue [#375] PR 3、epic [#380])

- **Removed**: **`unicode_safe_download_directory()`**。`%TEMP%` を `cache_root/downloads/<uuid>` へ移設するだけで、**その `cache_root` はユーザー名を含み得るため ASCII 保証が無い** (棚卸し §5.1 で実測)。`unicode_safe` を名乗る名前自体が「これを使えば ASCII 安全」という誤読を招く。shim は残さない (pre-1.0)
  - **Before**: `with unicode_safe_download_directory() as temp_dir:`
  - **After**: `with ascii_safe_temp_environment(boundary="engine.<name>.<op>", purpose="download") as temp_dir:`
  - **Migration**: `boundary` は**必須キーワード引数**である (失敗メッセージの 1 番目に出る診断契約)。旧 helper が包んでいた 5 箇所のうち、**`%TEMP%` を消費する 3 箇所** — `engine.parakeet.from_pretrained` / `engine.canary.from_pretrained` / `engine.qwen3asr.from_pretrained` — を新 API へ移し、**残る 2 箇所 (ReazonSpeech) は包み直さず単純削除**した (下記)
  - **意味が変わる点**: **ASCII 保証された root を用意できない環境では `AsciiStagingUnavailableError` になる**。従来は非 ASCII の `cache_root` へ黙って移設して「動いていた」が、それこそが epic [#380] の狙う silent failure である。対処法 (`LIVECAP_CORE_ASCII_STAGING_DIR`) は例外メッセージが名指しする
- **Removed**: `livecap_cli.utils` からの **`TempEnvironmentConflictError` の再 export**。`__all__` に載っていたが**消費者が 0 件**だった (旧 helper のテストのみ)。公開元は `livecap_cli.paths` に一本化する
- **Removed**: `temp_environment()` の **`base=None` 分岐**。上記削除で呼び出しがゼロになった。既定値を残すと「ASCII 保証の無い場所へ黙って移設する」経路が復活するので、`base` を**必須キーワード引数**にした (`unique` 引数を PR 2 で除去したのと同じ理由 — 旧挙動を保つためだけの分岐を残さない)
- **ReazonSpeech の 2 経路は wrapper を付け直していない** (`_download_model` の int8 / float32)。`download_file()` は `cache_root/downloads` へ直接書き、`temporary_directory()` は `dir=` を、`snapshot_download()` は `cache_dir=` を明示するので **`%TEMP%` を消費しない**。棚卸しでも当該行は ②wide-path が実測で確定している。**実測でも 713 MB のダウンロード中に移設先へ落ちたファイルは 0 件**だった (staging entry に `.livecap-entry` しか残らない)。**②wide-path が確定している経路を ③staging へ格上げすると、ASCII staging root を確保できない環境で本来動くダウンロードを新たに失敗させる** — 旧 helper がそこに居たことは包み直す理由にならない (pre-1.0 方針)
- **Docs**: 公開 API 名の不一致を解消した。`livecap_cli/resources/__init__.py` の package docstring が「ホスト向けの入口は 2 つ」のままで `docs/reference/api.md` の 3 つと矛盾しており、`docs/reference/feature-inventory.md` は**削除済みの `reset_resource_managers()`** を import 例・実行例として案内していた (**そのまま写すと `ImportError`**)。`docs/contributor/adding-an-engine.md` の AP-6 / §10.3 も、消えた helper を前提にした説明から `ascii_safe_temp_environment()` / `ascii_safe_workspace()` の使い分けへ書き換えた
- **Tests**: #386 のデータ消失回帰テストを `tests/core/utils/` から `tests/core/paths/test_temp_env_and_workspace.py::TestDataLossRegressions` へ**移設**した (helper が消えても保証は消さない)。**本物の別スレッドと子プロセス**で「移設先に落ちたファイルが退出後も生き残る」ことを見る — 既存の「退出時に消さない」テストは victim を手で置くので、*巻き込むからこそ消せない* という #386 の核心を示していない。棚卸し表からは `utils.unicode_safe_download_directory` 行と `utils.download_dir_data_loss` probe を除去し、代わりに **`engine.{parakeet,canary}.from_pretrained` の 2 行を追加**した — 実行時ログの `boundary=` を突合できる行が無く、**現在の staging 利用が表に現れなかった**ため (③staging 9 → **10 行**: 本 PR で対応済み 2 / #379 が 3 / [#413] が 5)。あわせて `BoundarySpec.staging_api` / `staging_purpose` を追加し、**境界一覧の SSOT を registry に一本化**した — `livecap_cli` を AST で走査した**実使用**と `staging_api` を持つ行を**双方向で完全一致**させる (`test_registry.py::test_every_staging_call_is_registered`)。一方向だと**registry に無いファイルへ `ascii_safe_*` 呼び出しを足しても検査対象にならず緑のまま**になる。`purpose` もテスト側に持たせない — #379 が既定の `"runtime"` 等を使ったときにハードコードが誤って落ちるため。以後 #379 / #413 が新しい境界を包むときは、**registry へ行を足さない限り CI が落ちる**。`utterance_wav` 5 行の `followup_issue` も #375 → [#413] へ付け替えた (#375 は本 PR で close するため)。**新 API を非 ASCII ハーネスで実測してはいない** — `runner.py` が `LIVECAP_CORE_ASCII_STAGING_DIR` を注入しないため、子プロセスが実 `%ProgramData%` へ書いてハーネスの隔離が壊れる。削除した probe は docstring 自身が「非 ASCII とは独立した欠陥なのでどちらでも同じ結果になる」と書いており、非 ASCII 軸の情報は元から持っていなかった

#### `SharedEngineManager` orphan module 削除 (Issue [#326])

[Issue #321 PR #3](https://github.com/Mega-Gorilla/livecap-cli/pull/325) の
API contract cleanup 中に発見した orphan code (`livecap_cli/engines/shared_engine_manager.py`、
**467 行**) を完全削除。pre-1.0 cleanup。

**削除対象** (3 symbols すべて zero caller、`__all__` 非 export、production / tests
からの参照ゼロを grep で確認):

- `ProgressCallback` Protocol
- `TranscriptionRequest` dataclass (`__lt__` 比較含む)
- `SharedEngineManager` class (threading + queue + 進捗 callback)

**Migration**: production / tests から未参照のため影響なし。仮に第三者
plugin が import していた場合は git history (`git log -- livecap_cli/engines/shared_engine_manager.py`)
から復元可能。

**reviewer feedback で追加 scope** (本 PR で実施):

- `livecap_cli/transcription/stream.py` の `TranscriptionEngine` Protocol
  docstring 2 箇所 (line 118 / 153) から `SharedEngineManager._process_request`
  の挙動説明を削除、`apply_filter` 単一 consumer 記述に整理
- `AGENTS.md:5` の repo guidance を更新、共有 tooling 説明を
  `shared_engine_manager.py` → `model_memory_cache.py` / `library_preloader.py` /
  `nemo_utils.py` (actually active な shared utility) に置換

**Verification** (本 PR merge 後):

```pwsh
git grep -n "SharedEngineManager\|shared_engine_manager" -- `
  livecap_cli tests AGENTS.md docs/reference docs/guides
# → 0 件 (CHANGELOG.md と docs/planning/archive/* の歴史的言及は許容)

uv run python -c "from livecap_cli.engines import EngineFactory, BaseEngine; print('OK')"
# → OK
```

### Fixed


- GitHub Actions workflows updated for module rename ([#201])
- Integration test path filters updated
- Async translation deadlock in concurrent scenarios ([#189])
- Translation timeout handling improvements ([#187])
- OPUS-MT context disabled by default for stability ([#191])

#### PyTorch CUDA Jiterator の kernel cache が ACP 外 path で全 CUDA 演算を壊す (Issue [#422]、epic [#380])

**非 ASCII ユーザー名の Windows 環境で、CUDA の Jiterator 経路に入る演算がすべて `UnicodeDecodeError` で失敗していた。** モデルは無関係で `torch` だけで再現する (`torch.fft.rfft(x).abs()`)。境界は kernel cache の置き場所 (`PYTORCH_KERNEL_CACHE_PATH` → 既定 `%TEMP%\torch\kernels`) であり、`%TEMP%` を ASCII にしても cache 先を非 ASCII にすれば同じ失敗が出る。**例外はパスを一切名指ししない**ので、epic [#380] の言う「診断上 fail_silent」に該当する。**`cjk_kana` (`ユーザー`) では再現しない** — cp932 の内側なので日本語 Windows では通ってしまい、素朴な確認では見逃す。

- **Added**: `livecap_cli/runtime/pytorch.py` — `configure_pytorch_runtime()`。**冪等・スレッド安全・`torch` を import しない** (環境変数を決めるだけなので CPU-only 環境と import コストを変えない)。決定は純関数 `_decide(environ, platform)` に閉じており、決定表をプロセス env を触らずに網羅テストできる
- **Added**: 呼び出し位置は **`BaseEngine.__init__` / `BaseTranslator.__init__` / `SileroVAD._initialize` / `cmd_transcribe`**。`EngineFactory` に置くだけでは **engine クラスを直接生成する library 利用者を守れない**。`load_model()` も使えない — `BaseEngine` 側は parakeet / reazonspeech が override しており、`BaseTranslator` 側は基底が no-op でローカル translator 2 つが override している。**`import livecap_cli` では自動実行しない** (ホストの `configure_resources()` を横取りしないため)
- **Added**: 新しい torch consumer の追加漏れを検出する audit test と、具象 engine / translator が `super().__init__()` を呼ぶことの静的検査。1 つでも抜けると**その経路だけが非 ASCII 環境で壊れる**
- **Added**: 棚卸し表へ `framework.pytorch.cuda_jiterator_kernel_cache` 行と `framework.pytorch.jiterator_cache` probe。**モデル不要で CUDA だけを要求する `gpu` tier** を新設した — `real_model` / `heavy` に混ぜると実モデルや NeMo が見つからず**黙って skip** し、「CUDA があるのに測っていない」状態が緑で通る。`cjk_kana` と `outside_acp` の両方を要求し、`%TEMP%` 以外の root は ASCII へ固定する
- **Added**: **raw / mitigated の 2 トラック** ([#379] で確立した構成)。`tests/integration/runtime/test_pytorch_kernel_cache.py` が**上流の性質**を fresh subprocess で固定し (ACP 外で壊れる / `cjk_kana` では壊れない / CPU は無関係 / **cache が populate されない**)、`test_whispers2t_nonascii_temp.py` が **production 経路** (`EngineFactory` → `load_model()` → `transcribe()`) の成功を見る。**両方あって初めて「欠陥は実在し、我々の経路では起きない」と言える。** 前者は**上流が直ったら落ちて再評価を促す**設計である
  - `tests/nonascii` の `engine.whispers2t.utterance_wav` 行は `%TEMP%` を ASCII へ固定したままにする — あれが測るのは `cache_root` に置かれる**発話 wav** であって `%TEMP%` ではなく、両方を非 ASCII にすると失敗の帰属ができない ([#413] で実際に誤帰属しかけた)。その穴を mitigated track が埋める
- **Changed**: **既定で `USE_PYTORCH_KERNEL_CACHE=0` を設定する。**
  - **Before**: 何も設定せず、PyTorch が `%TEMP%\torch\kernels` を使う。非 ASCII なユーザー名だと CUDA 演算が `UnicodeDecodeError` で落ちる
  - **After**: 明示指定が何も無いときだけ無効化する。**代償が無いことは実測で確認済み** — PyTorch 2.9.1 の Windows 書き込み経路は `<name>_tmp_<pid>` から最終名への rename を行わず (`std::ofstream` を閉じる前に `std::rename()` を呼ぶ)、ルックアップは最終名で行われるため **cache が populate されない**。実機の `%TEMP%\torch\kernels` には 75 ファイル / 最終名 0 / 実カーネル 2 種が積み上がっていた。つまり従来もコンパイル代 (~80 ms / カーネル / プロセス) を毎回払っており、見返りはゼロでファイルだけが増えていた
  - **Migration**: 有効化したい場合は `USE_PYTORCH_KERNEL_CACHE=1` または `PYTORCH_KERNEL_CACHE_PATH` を明示する (**明示指定は尊重する** — 外部で pre-populate した cache は実際にヒットする)。非 Windows は **no-op** で従来どおり
- **Changed**: **`USE_PYTORCH_KERNEL_CACHE` は `0` / `1` 以外を fail loud にする。**
  - **Before**: PyTorch の解釈をそのまま使う。`false` / `no` / 空文字はすべて**有効**として扱われる (実測)
  - **After**: Windows でのみ、`0` / `1` 以外は `PyTorchRuntimeError` を送出し、「PyTorch はこれを**有効**として扱う」と明示する。**`USE_PYTORCH_KERNEL_CACHE=false` と書いた利用者は無効化したつもりで有効化していた** — 意図と実際が食い違うのに兆候がゼロなのは epic [#380] が排除している形そのもの
  - **Migration**: 無効化は `0`、有効化は `1`。非 Windows では検証しない (境界が存在しないため)
- **Changed**: **明示された非 ASCII / 利用不能な cache path は fail loud。** 黙って上書きすると「運用者が指定した場所を使わない」ことになるため。メッセージには**境界名・変数名・path** を必ず含める
  - **空文字も fail loud。** `Path("")` は `Path(".")` なので素直に検証すると cwd を probe して「使える」と答えるが、実測では PyTorch はこれを空のディレクトリ名として扱い**キャッシュを黙って一切行わない** (非 ASCII な `%TEMP%` でも落ちない = 経路に入っていない)。**設定が何もしていない**ことを伝える
  - **相対 path は絶対 path へ正規化して適用する。** PyTorch が cache 先を解決するのは最初の Jiterator 実行時なので、初期化からそこまでに cwd が動くと**検証した場所と実際に使う場所がずれる**
- **Changed**: **`USE_PYTORCH_KERNEL_CACHE=1` で明示 path が無い場合も、解決した既定の置き場所を `PYTORCH_KERNEL_CACHE_PATH` へ pin する。**
  - **Before**: `%TEMP%\torch\kernels` を検証するだけで、変数は設定しない
  - **After**: 検証した絶対 path を明示的に設定し、`expected_env` にも載せる。**検証するだけでは保証にならない** — PyTorch が cache 先を解決するのは最初の Jiterator 実行時なので、それまでに `TEMP` / `HOME` が変われば**検証していない場所が使われる**。しかも解決の材料である `TEMP` / `HOME` は drift 検査の対象外なので気付けない。実測では、確定後に `TEMP` を ACP 外へ変えると pin 無しでは `UnicodeDecodeError`、pin ありでは成功し cache は pin 先へ書かれた
  - **Migration**: なし (有効化を選んだ利用者にとって置き場所は変わらない)。書き足したことは warning に出す
- **Note**: `USE_PYTORCH_KERNEL_CACHE=1` で明示 path が無いときに検証する場所は、**上流の解決順を実測で写した** — `%TEMP%\torch\kernels` → `%HOME%\.cache\torch\kernels`。**`TMP` / `TMPDIR` / `USERPROFILE` は PyTorch が参照しない**ので見ない (`TEMP` 未設定 + `TMP` が ASCII + `HOME` が非 ASCII、という環境で `TMP` を検証すると**PyTorch が使わない path を「安全」と答える**)。空文字の `TEMP` は未設定と同じ扱いになる
- **Note**: 再呼び出し時は**環境変数の drift を検出して fail loud** にする。黙って再適用しないのは、PyTorch がキャッシュ先を**最初の Jiterator 実行時に一度だけ**解決し (CUDA 初期化時ではない — 実測)、**確定済みかを読む公開 API が無い**ため。再適用が効いた保証が無い以上、「直したつもり」のログを残すより誰が何を壊したかを見せる方がよい
- **Note**: `ascii_safe_temp_environment()` は**使わない**。PyTorch がキャッシュ先を関数内 static として保持するので、スコープを抜けて `%TEMP%` を戻すと**握っている path と実体の寿命が一致しなくなる** ([#386] と同型)。永続 ASCII cache root を確保する案は、上流が rename を直すまで作らない ([#377] と同じ判断)。なお `USE_PYTORCH_KERNEL_CACHE=1` の pin は、この同型の破綻を**利用者が有効化を選んだ経路でも**防ぐ — 本 repo の `ascii_safe_temp_environment()` の内側で最初の Jiterator が走っても、PyTorch が握るのは**スコープに依存しない検証済みの path** である

#### Qwen3-ASR の consumer を実測し、一時 wav 5 行すべてを ②wide-path で確定 (Issue [#413] PR C、epic [#380])

**「NeMo と依存が競合して同居できないかもしれない」という前提が実測で否定された。**

- **Changed**: `engine.qwen3asr.utterance_wav` を `candidate_method` / `verified_method` とも **②wide-path** へ確定した。これで**一時 wav の 5 consumer すべてが実測で確定**し、**1 つも staging を追加しないまま** [#413] の主題が閉じた
  - **Before**: `covers_boundary=False` の producer-only probe (`tempfile.named_temporary_wav`) を指し、`unmeasured_reason` は「`qwen_asr` が未導入。NeMo と同居できるかが不明」
  - **After**: `cjk_kana` / `outside_acp` の両方で ASCII control と転写が一致。**同居できないという想定は誤りだった** — `uv sync --extra engines-qwen3asr` は **25 パッケージの純粋な追加**で削除もダウングレードも無く (`uv.lock` は universal lock なので解決済み)、`nemo 2.3.0` / `torch 2.9.1` / `transformers 4.57.6` と同居した runtime で `qwen_asr` が import できた。したがって [#413] が想定した**隔離環境も証拠の集約基盤も要らなかった**
- **Added**: `tests/nonascii/probes/utterance_wav.py` に qwen3asr の consumer probe。**言語を渡さないのが要点である** — 一時 wav を書くのは `_transcribe_via_wrapper_fallback()` だけで、そこへ入るのは `_asr_language is None` (auto-detect) のときに限られる。**言語を指定すると `_transcribe_with_scores()` へ行き境界を迂回する**ので、他の 4 engine とは逆に固定しない。迂回した場合は `_WavRecorder` が「一時 wav が variant root 配下に無い」で落とす (変異で確認済み: `{"language": "en"}` を入れると `error_harness`)
- **Added**: `qwen3asr_snapshot_dir()` — **marker だけでは重みの存在を保証しない**。models root にあるのは `model=Qwen/Qwen3-ASR-0.6B` と書かれた 38 バイトのテキストで、実体は **`huggingface_hub` が実際に使う hub cache** (`huggingface_hub.constants.HF_HUB_CACHE`) にある。marker の存在だけで「使える」と答えると **real_model tier の「ネットワークを使わない」契約を破ってダウンロードが走る**。判定を probe 側へ置くのは `sherpa.from_transducer.real` と同じ理由 (テスト側にファイル名を書くと二重管理になる)
- **Added**: qwen3asr probe の worker を **`HF_HUB_CACHE` = 実効 hub cache / `HF_HUB_OFFLINE=1` で起動する** (`test_probes._real_model_env()`)。**場所を当てにいくのをやめ、ネットワークへ出たら落ちるようにした**のが要点である。`huggingface_hub` は**どちらの値も import 時に確定する**ので env は worker の起動前に決めなければならず、probe の中で `os.environ` を書き換えても間に合わない (`_isolation_env` と同じ制約)。効いたことは probe が `huggingface_hub.constants` の **`HF_HUB_CACHE` と `HF_HUB_OFFLINE` の両方**で確かめて片方でも欠けたら fail loud させる — **cache path だけでは足りない**: 親 env から継承した path が期待値と一致していると path 検査は通るのに offline は効かない (実測: `cache_matches=True` / `constant_offline=False`)。`ModelManager.huggingface_cache()` は実行時に `HF_HOME` を書き換えるが `huggingface_hub` は import 時に cache path を確定するため効かない (production 側の食い違いは [#428] が追跡する)
- **Removed**: producer-only の `tempfile.named_temporary_wav` probe。5 consumer すべてが本物の probe を持ったので役目を終えた (producer 境界は `soundfile.write.path` / `soundfile.read.path` が測っている)。[#413] の受け入れ条件「`tempfile.named_temporary_wav` probe の帰属を決める」の解である
- **Fixed**: `integration-tests.yml` の `paths` に **`tests/nonascii/**` を追加**した。**非 ASCII の real-model / gpu ゲートは本 workflow の中にあるのに、`tests/nonascii/` を変更しても起動しなかった** — PR B ([#426]) で実際に Integration Tests が走らず発覚した。ゲートを持つ workflow が、そのゲートの対象を変更しても起動しないのは穴である
- **Changed**: GPU job の sync に `--extra engines-qwen3asr` を、warm step に `warm('qwen3asr', 'cuda', 'en')` を追加した。**warm の目的は HF hub cache を埋めること**である — qwen3asr の重みは models root に置かれず marker だけが残るので、ここでロードしておかないと probe が skip する (`HF_HUB_OFFLINE=1` を課しているのでダウンロードには落ちない)
- **Added**: `benchmark_results/nonascii/2026-09-01/results.json` — clean tree (`363fc69`) から **cheap / real_model / heavy / gpu を 1 セッション**で生成した証拠 (51 passed, 2 skipped / 116 レコード / 36 probe)。**PR B の `2026-08-31/results.json` は生成時のまま残す** — `<date>` は測定日という規約に従い、probe を変えたら測り直して新しい日付へ出す (`test_verified_rows_match_committed_evidence` は最新 1 件しか読まないので、古い方は履歴である)
- **Note**: skip した 2 件 (`whispers2t.load_model` / `qwen3asr.from_pretrained`) は `_REAL_MODEL_SOURCES` に source 定義が無い [#387] 追跡行で、どちらも `verified_method=None` なのでゲートには影響しない

#### 発話ごとの一時 wav 4 consumer を ②wide-path で確定 — **当初方針を実測が覆した** (Issue [#413] PR B、epic [#380])

**「5 consumer を ASCII staging へ移す」という当初計画 ([#375] PR 4 由来) を実装しなかった。実測が不要だと示したためである。**

- **Changed**: `engine.{parakeet,canary,whispers2t,voxtral}.utterance_wav` の 4 行を `candidate_method` / `verified_method` とも **②wide-path** へ確定した。
  - **Before**: `candidate_method=③staging` / `verified_method=None`。rationale は「正解は `ascii_safe_workspace()` で最初から ASCII 空間に ASCII 名で作ること」と書いていた
  - **After**: **4 engine とも `cjk_kana` / `outside_acp` の両方で ASCII control と転写が一致**した。[#378] §6.10「② で足りる境界に ③ を持ち込まない」に該当するため、**staging を追加してはならない**行として記録する。効果ゼロのコピーと後片付けを抱え込まずに済む
  - **Migration**: なし (production コードは 1 行も変えていない)
- **Added**: `benchmark_results/nonascii/2026-08-31/results.json` — clean tree (`024a86b`) から **cheap / real_model / heavy / gpu を 1 セッションで**生成した証拠 (118 レコード / 36 probe / 42 passed)。`test_verified_rows_match_committed_evidence` は `benchmark_results/nonascii/*/results.json` の**最新 1 件しか読まない**ので、tier を分けて実行すると**既に verified な 29 行が一斉に「証拠なし」になる**
- **Changed**: 証拠 JSON の `tiers_enabled` を**宣言ではなく実績**から書くようにした (`tests/nonascii/conftest.py`)。
  - **Before**: `["cheap"] + (["real_model"] if LIVECAP_NONASCII_REAL_MODELS else [])`
  - **After**: teardown で `sorted({r.tier for r in results})`。**heavy は `importorskip("nemo")`、gpu は CUDA の有無でしか gate されず**この env と無関係に走るため、従来は「走っていない」と主張したまま記録だけ入る状態になり得た
- **Fixed**: `tests/nonascii/README.md` の証拠生成コマンド。旧例は `-m nonascii_paths` を持っており**全 tier を収集していた** (実測: cheap 25 / real_model 8 / heavy 6 / gpu 1 node) が、**`LIVECAP_NONASCII_REAL_MODELS=1` が無かった**。real_model tier は `pytest.skip` で `_execute` の**前に**抜けるので**レコードが 1 件も作られず**、一方 **heavy / gpu はこの env に依存しない**ため走ってしまう。結果として「heavy と gpu はあるが real_model が丸ごと欠けた JSON」ができ、上記の「最新 1 件しか読まない」設計と合わさって real_model の verified 行が「実測レコードが無い」で落ちる。新コマンドの `and not network` は将来 network probe が増えたときの混入を防ぐためで、現時点で除外される node は無い
- **Tests**: `TestSlowResultFinalizationOrder` に**陰性対照**を 1 件足した。既存 4 件は順序と優先度を見ており、**回帰そのものを捕まえる経路にテストが無かった** — 期待 verdict の検査を外す変異で、新規 1 件だけが落ちることを確認済み
- **Note**: **`③staging` を主張する NeMo の 3 行は再実測後も `fail_silent` のまま**である。それらの probe は `nemo_asr.models.ASRModel.restore_from()` を**直接**呼び raw 境界を測るので、[#379] の production 側の緩和とは独立している。**probe が raw を測るか production 経路を測るかで、整合する `verified_method` が決まる** — この読み方が成り立たない唯一の行が [#422] の Jiterator であり、[#425] へ移管した
- **Note**: `framework.pytorch.cuda_jiterator_kernel_cache` は実測済み (両 variant とも pass) だが **`verified_method=None` を維持**し、`followup_issue` を [#413] から [#425] へ付け替えた。`Method` が「境界の能力」(①②) と「production の緩和」(③④) を混在させており、#422 の**複合戦略** (既定で境界を回避 + 明示 opt-in 時のみ fail-fast) を表現できないため。④fail-fast にすると「主張は fail-fast だが実測は全て pass」で弾かれ、②wide-path は上流が narrow のままなので嘘になる
- **Note**: `engine.qwen3asr.utterance_wav` は**触っていない**。probe が producer-only (`covers_boundary=False`) で、`test_measurement_caveat_rows_are_not_verified` が `verified_method` の設定を**機械的に禁じている**。隔離環境での実測は #413 PR C が行う

#### 発話ごとの一時 wav の consumer を実モデルで測る probe を追加 (Issue [#413] PR A、epic [#380])

`tests/nonascii/registry.py` の `engine.*.utterance_wav` 5 行は、**consumer 側を一度も測っていなかった** — 参照していた `tempfile.named_temporary_wav` は producer (`sf.write` と読み戻し) しか覆わず、本当の境界である「その path をネイティブ ASR に渡す側」には届いていなかった。

- **Added**: `tests/nonascii/probes/utterance_wav.py` — parakeet / canary (heavy tier) と whispers2t / voxtral (real_model tier) の 4 consumer を**実モデルで**測る。**probe_id は engine ごとに分ける** (`_REAL_MODEL_SOURCES` が probe_id 単位で source を引くため)。engine 自身の一時ファイル生成先はハーネスが既に variant root へ向けている (`TEMP` と `LIVECAP_CORE_CACHE_DIR`) ので、**production の `transcribe()` をそのまま呼ぶ**
- **Added**: `BoundarySpec.required_variants`。**`cjk_kana` だけでは足りない** — `ユーザー` は cp932 の内側なので、consumer が narrow path でも**日本語 Windows なら通ってしまう**。ACP の外側 (`outside_acp` = `한국어Ω`) まで要求する。**足りなければ skip ではなく fail** する
- **Added**: `BoundarySpec.ascii_pinned_roots`。**「この境界が測りたい 1 つ」以外の root を ASCII へ固定する**ための切り分け。worker は models / cache / resources / `%TEMP%` / `HF_HOME` を**すべて** variant root へ向けるので、そのままだと**複数の境界を同時に非 ASCII にする**ことになり、失敗しても原因を特定できない (**#422 で実際に誤帰属しかけた**)。既存の module-level `_HEAVY_ASCII_TEMP_BOUNDARIES` を吸収した。**env は worker 起動前に決める** — `tempfile.gettempdir()` や huggingface_hub は初回参照で値をキャッシュするため、probe の中で書き換えても間に合わない
- **Added**: probe が**存在確認した source と実際にロードしたモデルの identity が一致すること**を検査する。engine kwargs を省くと `EngineFactory` が metadata の既定値をマージするため**宣言と別のモデルが読まれ得る** — 実際 whispers2t は `whispers2t_base` の存在を確認しながら既定の `large-v3` を読んでいた。**緑が persistent runner の残留状態に依存し**、fresh runner ではダウンロード (`real_model` tier は**ネットワークを使わない**契約) か失敗になる状態だった
- **Added**: slow tier の判定順序を `skip -> harness -> control 安定性 -> expected verdict` に固定した (`_finalize_slow_results`)。**逆順だと非決定性で `fail_silent` の assertion がその場で止まり、安定性検査へ到達しない** — しかも証拠には `fail_silent` が残り「非決定性は `error_harness` とする」契約と食い違う。不一致時は**記録される verdict も `error_harness` へ書き換える**。順序は合成 `ProbeResult` の単体テストで固定した (**実モデル不要**)
- **Added**: control 観測の安定性検査。control と trial は別 worker プロセスでモデルも別ロードなので、推論が非決定的なら**path と無関係な差を fail_silent と誤判定する**。両 variant を回すと control が 2 回走るので**追加のモデルロード無しで**前提を検査できる (実測: 4 engine とも別プロセス 2 回で fingerprint も confidence も完全一致)
- **Fixed**: `test_real_model_boundary` に **probe skip の伝播が無かった**。`_assert_expected_verdict` は skipped を早期 return するので、**probe が動かなくても PASSED** になっていた。heavy tier には [#379] で入れた対策が real_model tier には無く、CI がこの PASSED をゲートに使う以上「ゲートは緑だが対象経路を通っていない」状態だった
- **Fixed**: **証拠の照合が `boundary_id` だけだった。** 同じ境界を別 probe で測り直すと、**古い probe の pass が新しい主張の証拠として通る** — 実際 `engine.*.utterance_wav` は producer only の証拠を持っており、**新しい実測を一切せずに `verified_method=WIDE_PATH` を名乗れた** (registry を書き換えて実証済み)。照合を `registry.evidence_rows_for()` に一本化し、`probe_id` / `tier` / 要求 variant まで見るようにした
- **Fixed**: **同じ穴が `report.py` にもあった** (`rows = [r for r in results if r["boundary_id"] == ...]`)。検査だけ直すと**人間が読む棚卸し表が古い証拠を新 probe の実測として表示し続ける**。検査と表が同じ規則を使うよう、照合は 1 箇所に置いた
- **CI**: 既存の非 ASCII real-model step へ 4 行の `PASSED` 要求を追加した。両 variant は node の内側で回るので、node の PASSED が両 variant の完走も保証する
- **`verified_method` は設定していない。** 証拠 JSON の生成と SSOT 更新は PR B で **clean tree から**行う — probe を書きながら証拠も作ると「どの版で測ったのか」が曖昧になる
- **Discovered**: 切り分けの過程で **[#422]** を発見した。**PyTorch の CUDA Jiterator kernel cache** が ACP 外の path だと CUDA 上の複素数演算が `UnicodeDecodeError` で落ちる — `%TEMP%` はその**既定の置き場所にすぎない** (`PYTORCH_KERNEL_CACHE_PATH` を非 ASCII にすれば `%TEMP%` が ASCII でも落ちる)。WhisperS2T は前処理が `torch.fft.rfft(...).abs()` を通るため**最初に踏んだ consumer**で、**utterance_wav とは別の境界**である
- **Removed**: probe 用音声ローダの重複。`native_models._load_probe_speech()` と本 PR で足した同等関数を `artifacts.load_probe_speech(stem)` へ 1 本化した (言語別の資産を選べるよう stem を引数にした)

#### realtime mode で `--translate` が黙って無視される問題を修正 (Issue [#403])

**`livecap-cli transcribe --realtime --mic 0 --translate google` が、エラーも警告も出さずに翻訳せず動いていた。** 翻訳を求めた実行が、翻訳せずに「成功」していた。

- **Before**: `_transcribe_realtime()` は translation を一切参照していなかった (`translat` の出現 **0 件**)。`TranslatorFactory.create_translator()` の呼び出しは `_transcribe_file()` 内の 1 箇所だけ。file mode の silent no-op は [#363] で解消済みだったが、**realtime 側に同等の仕組みが無かった**
- **After**: realtime で `--translate` を指定すると、**ASR モデルのロード前に** stderr へ理由を出して **exit 1** する
- **Migration**: realtime で翻訳が必要な場合は **file mode か livecap-gui** を使う。`--translate` を指定していなければ realtime の挙動は変わらない
- **実装ではなく拒否にした理由**: realtime に翻訳を配線すると、**翻訳の待ち時間が音声読み取りループそのものをブロックする**。`_translate_text()` の `future.result(timeout=5s)` の間 `transcribe_sync()` は `audio_source` から読まず、`MicrophoneSource._queue` は **maxsize 無し・drop 無し**なので**遅れは戻らず単調に増える**。[#402] が翻訳 executor を ASR から分離したのは「翻訳が ASR 推論をブロックしない」ことであって、**音声入力ループがブロックされないことではない**
- **warning ではなく exit 1 にした理由**: [#363] が warning で扱ったのは「無視されても品質がわずかに劣化するだけの補助オプション」である。翻訳は**その実行の主目的**で、無視されると**求めた出力が得られない**。さらに realtime は字幕が流れ続けるため、**起動時の warning は即座にスクロールして消える** — silent no-op を warning へ格下げしただけになる
- **Changed**: `--translate` / `--target-lang` の help に file mode 専用である旨と、`--target-lang` が `--translate` 指定時のみ意味を持つ旨を明記した。**`--target-lang` 単独指定の runtime 検出は入れていない** ([#382] の scope)
- **realtime 翻訳の実装は非スコープ**。着手するなら音声キャプチャと翻訳待機の分離 / キュー上限と drop 方針 / 字幕遅延の上限測定が要る
- **Tests**: exit 1 と理由の出力 / **engine を生成しないこと** (`_load_engine` を呼ばれたら fail に差し替えて確認) / file mode の `--translate` が影響を受けないこと / `--translate` 未指定の realtime が従来どおりであること

#### realtime 経路で `engine.cleanup()` が呼ばれない問題を修正 (Issue [#407])

**`_transcribe_realtime()` は正常終了・例外・Ctrl+C のいずれでも engine を片付けていなかった。**

- **Before**: `_load_engine()` は `load_model()` の失敗時だけ cleanup し、コメントは「caller の finally」が拾う前提だった。**realtime 側にその finally が無かった**。`with StreamTranscriber(...)` は transcriber を閉じるだけで、**注入された engine には触れない** ([#402] D9「生成した者が所有する」) — 生成したのは CLI である
- **After**: `_transcribe_file()` と同じく `finally` で片付ける。**`KeyboardInterrupt` は `except Exception` に捕まらないが `finally` は通る**ので、ループ内・ループ外どちらの Ctrl+C でも片付く
- **`_load_engine()` は `except Exception` → `except BaseException` にした (レビュー指摘)。** **外側の `finally` だけでは「モデルロード中の Ctrl+C」を救えない** — `KeyboardInterrupt` は `BaseException` 派生で `except Exception` を素通りし、caller 側は `engine = _load_engine(args)` の**代入が完了しないまま**中断されるため、caller の `finally` から見た `engine` は `None` になる。**取得途中のリソースは取得した側が始末する**。握り潰さず必ず再送出するので `KeyboardInterrupt` / `SystemExit` の終了意味論は変わらない。**realtime / file 双方が同じ関数を使うので両経路が直る**
- **Migration**: 不要。挙動は「片付くようになる」だけである
- **影響**: GPU engine では **VRAM が解放されないまま**関数を抜けていた。1 回転写して終了する現在の CLI では実害が限定的だが、**契約違反であり realtime 経路に所有物を足すたびに漏れが増える**
- **`_transcribe_file()` の `for closer in (...)` ループは真似していない。** あちらは所有物が 3 つあり順序が必須だからその形をしている。realtime は 1 つなので、1 要素のループは理由の無い模倣になる。**順序契約 (`StreamTranscriber.close()` → `translator.cleanup()` → `engine.cleanup()`) はコメントで残した** — 将来 realtime へ translator を足すときに engine より前へ置けるように
- **Tests**: 正常終了 / 転写中の例外 / **ループ内の Ctrl+C** / **マイク起動中の Ctrl+C** / **モデルロード中の Ctrl+C** / マイク起動失敗 / `cleanup()` 自体が投げても終了コードが変わらないこと。`_load_engine()` 側も `Exception` / `KeyboardInterrupt` の両方で cleanup + 再送出することと、**cleanup の失敗が本来の例外を隠さない**ことを固定した
- **Tests (CI で実行されるようにした)**: 既存の realtime e2e テストは `monkeypatch.setattr("livecap_cli.MicrophoneSource", ...)` が既存値確認で `__getattr__` を呼び PortAudio を import するため、**hosted Linux runner では skip されていた**。新規テストは `monkeypatch.setitem(livecap_cli.__dict__, ...)` で `__getattr__` を回避し、**PortAudio 無しでも実行される**

#### Voxtral が `--language` 未指定だと必ず失敗する問題を修正 (Issue [#418])

**`livecap-cli transcribe <wav> --engine voxtral` が、言語を指定しないと `TypeError: object of type 'NoneType' has no len()` で必ず落ちていた。** `voxtral` の `cli_default_language` は `auto` なので、**既定の呼び出しがそのまま壊れていた**。

- **Before**: auto を `None` に解決し、`apply_transcription_request(language=None, ...)` と**素の値**で渡していた。上流 (`transformers 4.57.6`) の validator は `str` か list しか想定しておらず、`isinstance(language, str)` の分岐にも入らないまま `len(language)` へ落ちる
- **After**: **audio 1 件につき 1 要素の list** (`[None]`) で渡す。`_processor_languages()` に閉じ込めた
- **Migration**: **不要。** CLI 引数も既定言語も変わらない。`--language` 未指定 / `auto` が**動くようになる**だけで、明示指定の挙動は変わらない
- **`auto` は廃止していない。** 当初は「既定を `en` にして `supports_language_auto=False`」を検討したが、**上流で auto は生きている**ことを実測で確認したため撤回した。実 processor が組み立てるプロンプトは `[None]` が `lang:` トークンを**含まず**、`["en"]` は `lang:en` を含む — 既定言語へ黙って落ちるのではなく、本当に自動判定になる。`cli_default_language="auto"` / `supports_language_auto=True` / `_resolve_language("auto") -> None` はいずれも維持している
- **ずれていたのは値ではなく形だった。** [#365] が定めた「auto = `None`」という**値**は上流 (`TranscriptionRequest.from_openai()`) の契約どおり正しい。混同していたのは mistral-common の `TranscriptionRequest` 契約と、実際に呼んでいる Transformers の `VoxtralProcessor` 公開 API の契約である
- **Fixed (なぜ CI が緑だったか)**: `test_voxtral_language.py` の上流契約テストが **`MagicMock` に素の `None` が渡ること**を期待しており、**実物の入力検証を一度も通っていなかった** — 障害経路そのものを仕様として固定していた。加えて smoke は全ケースが言語を明示するため、**既定値の呼び出しはどこでも実行されていなかった**。[#379] / [#409] と同じ「ゲートは緑だが対象経路を通っていない」形である
- **Tests**: 上流契約テストを `[None]` / `["en"]` へ更新し、`_processor_languages()` の写像 (`None -> [None]` / `en -> ["en"]` / `fr -> ["fr"]`) を固定した。**mock では auto が本当に auto かを確かめられない**ので、`tests/integration/engines/test_voxtral_language_contract.py` を新設し、**実 processor が組み立てるプロンプトに `lang:` があるか無いか**を見る (**token 数は pin しない** — tokenizer 更新で動くため)。上流が素の `None` を受け入れるようになったら落ちるテストも置いた (auto を再検討する trigger になる)
- **Tests**: engine smoke に **`voxtral_gpu_default_language`** を追加した。`EngineSmokeCase.use_engine_default_language` で **`language` を engine へ渡さない**ケースを表現する
- **CI**: self-hosted step で上記の `PASSED` を明示的に要求する。`LIVECAP_REQUIRE_ENGINE_SMOKE` は repo variable 依存で、未設定だと **skip が黙って緑になる**

#### ReazonSpeech の cache identity が不足し、root 衝突・設定混同・モデル更新後の陳腐化が起きる問題を修正 (Issue [#409])

**`ModelMemoryCache` のキーが `use_int8` とディレクトリの basename しか含んでいなかった。**

- **Before**: `cache_key = f"reazonspeech_{use_int8}_{model_path.name}"`。① **異なる models root の同名ディレクトリが衝突する**、② `tokens.txt` / `encoder` / `decoder` / `joiner` を差し替えても**古い recognizer が返る**、③ `num_threads` / `decoding_method` を変えても同じキーになる (**どちらも `from_transducer()` に実際に渡している**)
- **After**: **`reazonspeech:v2:<sha256(identity)>`**。identity は **正規化済み model root / `tokens.txt` の SHA-256 / encoder・decoder・joiner の (path, `st_size`, `st_mtime_ns`) / `use_int8` / `num_threads` / `decoding_method` / `sherpa-onnx` と `sherpa-onnx-core` の版**から作る
- **Migration**: **legacy な v1 キーは読みも書きもしない。** `ModelMemoryCache` はプロセス内メモリなので、再起動時に持ち越されるものは無い
- **root 正規化は `Path.resolve()` + `os.path.normcase()`。** 単なる `str(Path)` では Windows の大文字小文字・相対 path・symlink を同一視できない
- **ONNX 本体の SHA-256 は取らない。** lookup のたびに数 GB を読むことになり、メモリ cache の利点を失う。`tokens.txt` は小さいので内容ハッシュを使う
- **`sherpa-onnx-core` の版も含める。** native 処理には core も関係し、`pyproject.toml` は両者を同一版へ固定している。版は `importlib.metadata.version()` で取る — `sherpa_onnx.__version__` は **wrapper 側の版しか示さない**。**版の一致を検証するのは別責務**なので本 PR では扱わない
- **identity は cache lookup より前に確定させる。** モデルファイルが欠けていれば**そこで落とし、キャッシュ済みの recognizer も返さない** — ここで fallback すると「identity を取れないときは簡易キーを使う」経路を作り込むことになる
- **path は `resolve()` してから組み立てる。** 相対 root のまま constructor へ渡すと、identity が記録する path (解決済み) と**実際に開く path がずれ**、cwd がプロセス内で変われば同じキーが別のファイルを指し得る
- **lookup の前に、必要ファイルを read mode で 1 byte だけ開く。** `is_file()` も `stat()` も metadata しか触らないので、**内容の読み取りだけが拒否されている状態を見逃す** (Windows の ACL、権限を落としたネットワーク共有など)。ONNX は identity へ stat しか入れないため、確認しないと **cold path は `from_transducer()` で落ちるのに warm path は cache hit を返す** — 同じ環境なのにプロセス内の順番で結果が変わる。**全内容の hash は要らない**
- **Added**: `ModelIdentityChangedError`。**構築中にモデルファイルが変わったら保存しない**。そのまま保存すると古い identity のキーへ新しい内容の recognizer が入る
- **Changed**: ファイル名リストの出所を **`reazonspeech_cache.required_files()` の 1 箇所**にした。以前は `_verify_model_integrity` / `_download_model` の 2 分岐 / `_load_model_from_path` の**計 4 箇所**に複製されており、**identity が hash するファイルと constructor が読むファイルがずれ得た**
- **`ModelMemoryCache` 本体は変更していない。** cache の実装は他 engine も共有しており、本 PR が直すのは「何を key にするか」であって「どう保持するか」ではない
- **「壊れた recognizer を保存しない」は本 PR のスコープ外**である — post-load health check と保存ゲートは [#392] が持つ。当初 #409 はこれを受け入れ条件に含んでいたが、**判定そのものを非スコープにしていたため #409 単独では達成できなかった**ので責務を分離した
- **CI**: self-hosted runner の warmup が **ReazonSpeech の int8 モデルを温めていなかった**。`use_int8` は user-facing なパラメータなのに、GPU smoke は float32 しか通しておらず**int8 経路は CI で一度も実行されていなかった** (#409 のゲートが skip として検出)。warmup へ追加した
- **CI**: 実モデルの cache hit 確認を **self-hosted runner の step として明示的に走らせ、PASSED を要求**する。`engine_smoke` + `slow` なので通常の smoke step では収集されず、**どこでも走らないテスト**になっていた
- **Removed**: 未使用の import — `reazonspeech_engine` の `os` (本 PR で `os.path.join` を使わなくなった)、`parakeet_engine` の `Dict` / `Tuple`、`canary_engine` / `whispers2t_engine` / `base_engine` の `Tuple`、`audio/transient_detector` の `field`。**`resources/errors.py` の `Sequence` / `Tuple` は残す** — 文字列注釈 (`"Sequence[Tuple[str, str]]"`) で参照されており、消すと型解決が壊れる
- **Removed**: 実モデルテストの `pytest.importorskip("sherpa_onnx")`。`sherpa-onnx` は**コア依存**なので skip できる状況は「壊れている」ときだけで、guard は breakage を隠すことにしかならない
- **Tests**: identity の変異 (root 違い / `tokens.txt` 内容 / ONNX の stat / `num_threads` / `decoding_method` / 両 native 版 / v1 キーの sentinel / cache hit 時に `from_transducer()` を再実行しない / 構築中の変更 / ファイル欠損時の fail loud / key の決定性) / **相対 root でも constructor へ絶対 path が渡ること** / **ONNX が読めないときに cache hit を返さないこと**を **mock で網羅**し、実モデルは int8 / float32 の cache hit 確認に絞った。**修正前に 17 件中 15 件落ちる**ことを `origin/main` の worktree で実測済み (通る 2 件は既存挙動の回帰ガード)

#### session reaper の liveness テストが Windows CI で稀に失敗する問題を修正 (Issue [#406])

**`tests/nonascii/test_harness_selftest.py::TestReaperLiveness` が Windows で flaky だった** (#409 の CI で観測)。

- **Before**: `subprocess.wait()` が返った直後に生存判定していた。**Windows では process の終了と file handle の解放が同時ではない**ため、終了済みの session が「まだ使用中」に見え、reaper が空を返すことがある
- **After**: handle が解放されるまで**上限付きで待ってから**判定する。**判定そのものは緩めていない** — 生存判定が壊れて常に「使用中」を返すなら、待ち切って同じ assert で落ちる
- 本件は #409 のブランチで踏んだため同 PR で直した。**修正対象は #409 と無関係**である

#### 非 ASCII `%TEMP%` で Parakeet / Canary のローカルモデル復元が黙って失敗する問題を修正 (Issue [#379]、epic [#380])

**Windows のユーザー名や `%TEMP%` が非 ASCII だと、`.nemo` からの復元が原因と無関係な例外にすり替わっていた。**

```text
Can't instantiate abstract class ASRModel with abstract methods
setup_training_data, setup_validation_data
```

- **Before**: `restore_from()` が**素の `%TEMP%`** のまま呼ばれる。NeMo は内部で `.nemo` を `%TEMP%` へ自前展開するので、展開先が非 ASCII だと SentencePiece が読めない。NeMo は具象クラス生成中の例外を捕捉して基底クラスへ fallback するため、**最終例外の `__cause__` を辿っても元例外に到達できない**。`nemo_logger` は `propagate=False` + 独自 stream handler なので、windowed build では一次エラーが app log にも届かなかった
- **After**: `ascii_safe_temp_environment(boundary="engine.{parakeet,canary}.nemo_restore_from", purpose="nemo-restore")` の内側で復元する。**`.nemo` は元パスから直接読む** — staging も copy もしない (`.nemo` 自体は wide path で通ることが [#378] の片側 A/B で確定している)。**初回ダウンロード経路 (`from_pretrained`) は [#375] PR 3 で対策済み**で、本 PR が直すのは**ローカル `.nemo` の復元経路**である (モデルが on-disk に既にある 2 回目以降が該当)
- **Added**: `nemo_utils.restore_nemo_model()` — Parakeet / Canary で重複していた「logger 抑制 → `restore_from()` → 復元」を共通化した。**`%TEMP%` の移設だけは engine 側に残す** — `boundary` を引数で受けて helper 内で開くと**動的値になり、棚卸し registry との AST 突き合わせ (`test_every_staging_call_is_registered`) が成立しない**。境界を決めているのは helper ではなく engine である
- **Added**: **NeMo 一次エラーの app log への転送**。`nemo_logger` へ一時 handler を付け、ERROR record を boundary と `.nemo` パス付きで転送する。**`propagate` が `False` のときだけ**付ける — `True` なら root が既に受け取っており、転送すると**二重出力**になる。パスは `ascii()` で包む (日本語 Windows では stderr が cp932 + strict になり、素のパスはログ自体を `UnicodeEncodeError` で落とす)
- **元例外を `raise ... from exc` で置換しない。** 置換すると「抽象クラスの二次例外にすり替わる」という本 issue の症状を別の形で作り直すことになる。診断はログ側で足し、例外は bare `raise` でそのまま通す
- **専用のロックは持たない。** logger 操作は `ascii_safe_temp_environment()` の内側で行う — 同 API が `_TEMP_ENV_LOCK` をスコープ全期間保持するので直列化を継承できる。ロックを 2 つ持つと deadlock の余地を作るだけである
- **Removed**: 未使用の `from nemo.utils import logging as nemo_logging` (4 箇所、参照ゼロ)
- **Tests**: **raw track と mitigated track を分けた**。既存 heavy probe (`engine.nemo.untar_temp` / `engine.nemo.restore_path_only`) は NeMo を**直接**呼ぶ**基準データ**なので対策済み helper で上書きしない。production 経路が壊れないことは `tests/integration/engines/test_nemo_nonascii_temp.py` (**on-disk warm / in-memory cold**) が見る。加えて **probe だけでは engine 本体が wrapper を呼び忘れても検出できない**ので、`tests/core/engines/test_nemo_restore.py` が NeMo 抜きで配線を固定する (temp context の内側で `restore_from()` が呼ばれること / engine 別 boundary が registry と一致すること / 例外時の環境・logger 復元 / 元例外の非置換 / 二重出力しないこと / 非 ASCII パスでログが落ちないこと)
- **Fixed (レビュー指摘)**: **`all` extra にも同じ上限を入れた。** extra は継承しないので、`engines-nemo` にだけ書いても `pip install livecap-cli[all]` は NeMo 2.5+ / lightning 2.6+ を選べてしまい、同じ import failure を再発できる。**repo の `uv.lock` は全 extra をまとめて解決するのでこの漏れを隠す** — 公開 package metadata を直接見る `tests/core/test_packaging_constraints.py` を追加し、狭い extra の上限が meta extra で外れていないことを検査する (差分は **`narrow - wide`** の向きで取る — 逆向きだと「狭い側が同じ配布物に複数の上限を持ち、広い側がその一部だけを持つ」ケースを見逃す。向き自体を固定する unit test も置いた)
- **Fixed (レビュー指摘)**: NeMo の一次エラーが **app log に 2 回**出ていた。relay が即時に1 record 出したうえで、失敗サマリが `relay.first_error` を再掲していた。サマリからは本文を外し、「上で出した」ことだけ示す (relay が何も掴めなかった場合はその事実を書く)
- **Fixed (レビュー指摘)**: cp932 の回帰テストが `ascii()` の除去を**検出できていなかった**。`"ユーザー"` は cp932 で普通に encode できるため。cp932 の外側の文字を混ぜ、**素のパスなら実際に `UnicodeEncodeError` になる**ことを前提として固定した (変異テストで確認)
- **Fixed (レビュー指摘)**: 棚卸しの `engine.parakeet.nemo_restore_from` が、`NEMO_AVAILABLE=False` のキャッシュを非 ASCII `%TEMP%` の症状として書いていた。`check_nemo_availability()` は `restore_from` より**前**の import 成功時点で `True` をキャッシュするので本経路では触られない。False になるのは import 自体が失敗したとき(実例: 上記 lightning 2.6) であり、**別事象**として分離した
- **Fixed (CI で発覚)**: **`lightning` に上限 `<2.6` を入れた。** lightning 2.6.0 が `NeptuneLogger` を削除した一方、NeMo 2.3.0 は無条件に import するため、**lock を見ない `uv pip install -e .[engines-nemo]`** を使う self-hosted GPU runner では `lightning 2.6.5` が入り `import nemo.collections.asr` が落ちていた。その結果 **`engine-smoke-gpu` は parakeet / parakeet_ja / canary を全部 skip しながら緑**で、NeMo heavy probe も素通りしていた (warmup も例外を握りつぶす作りだった)。`uv.lock` の解決は 2.4.0 のまま変わらない
- **Fixed (同上)**: `test_heavy_boundary` が **probe の skip を PASSED として報告していた**。`_assert_expected_verdict` は skipped を早期 return するので、**probe が動かなくてもテストは緑**になる。CI がこの PASSED をゲートに使っているため「ゲートは緑だが対象経路を通っていない」状態そのものだった。probe が skip したら test も skip するようにした
- **CI**: 非 ASCII real-model step が NeMo 行の `PASSED` を要求するようにした (従来は [#377] の 1 行だけで、NeMo 行が skip / 未収集でも緑だった)。mitigated track 用の step も追加した
- **Docs**: 棚卸し表の `engine.{parakeet,canary}.nemo_restore_from` に `staging_api` / `staging_purpose` を設定。`engine.nemo.untar_temp` / `engine.nemo.restore_path_only` は機構が `nemo_utils.py` へ移ったので callsite を更新。`engine.reazonspeech.sherpa_narrow_path_signature` の `followup_issue` を **closed な [#377] → [#387]** へ付け替えた (計測ギャップ自体は未解消で、`followup_issue` が空でなければ通る検査では孤児化を検出できない)

#### 非 ASCII パスの models root で ReazonSpeech の全 transcribe が失敗する問題を修正 (Issue [#377]、epic [#380])

**Windows のユーザー名が非 ASCII (`C:\Users\ユーザー\...`) の環境で、ReazonSpeech engine がモデルロードに成功した後、全ての transcribe が `IndexError: invalid unordered_map<K, T> key` で失敗していた。** ロード時にはエラーが一切出ないため、フィールド報告では 1 セッション中 **421 回**同一例外で成功した文字起こしは **0 件**、`stream.py` が握って継続するため**プロセスは "running" のまま無出力で回り続けていた**。

- **Before**: `sherpa-onnx 1.12.39` の `SymbolTable` が `tokens.txt` を **narrow path の `std::ifstream`** で開く。Windows では UTF-8 バイト列が ANSI/CP932 として解釈されるため open に失敗し、**空のまま例外なく Init される**。ONNX 本体 (encoder/decoder/joiner) は onnxruntime が wide path を使うため正常にロードされるので**ロード時には気づけず**、デコード時に token id → symbol の lookup が `std::unordered_map::at()` で失敗して初めて表面化する。`debug=True` でも vocab_size は ONNX メタデータ由来で表示されるため、tokens が読めていないことを示すログが出ない
- **After**: **`sherpa-onnx` / `sherpa-onnx-core` を 1.13.6 へ揃えて bump**。上流 [PR #3255](https://github.com/k2-fsa/sherpa-onnx/pull/3255) で `SymbolTable` が `OpenInputFile()` を使い、Windows では `ToWideString()` 経由で開くようになった
- **こちら側の staging は実装していない。** 当初は `tokens.txt` のみを ASCII-safe な場所へ staging する案だったが、**上流が C++ 層で直したものを Python 層で迂回する理由がない**。staging では `tokens.txt` しか救えないのに対し、上流修正は同じ `OpenInputFile()` を通る他の経路にも及ぶ。加えて staging したファイルには寿命・所有権・cleanup・lease の責務が付いて回る
- **実測**: 1.12.39 / 1.13.6 の A/B を **tokens のみ非 ASCII** と**モデルディレクトリ全体が非 ASCII** の両条件で実施 (int8 実モデル)。前者は変数を切り分けるため ONNX を ASCII 固定にしたもの、後者はフィールド報告と同じ条件。**1.12.39 は両方で `IndexError`、1.13.6 は両方で正常転写**
- **confidence 経路の回帰も確認**: `avg_logprob` の供給元 `OfflineRecognitionResult.ys_log_probs` は「1.12.39 で expose されるようになった」**Python 側の result schema** であり依存更新で変わり得る。**schema が消えても転写テキストは正常に出る**ため、テキスト比較では検出できない (その場合 confidence filter は ReazonSpeech に対して pass-through へ degrade する)。両版で `avg_logprob = -0.16629084673794833` / `ys_log_probs_n = 22` の**ビット一致**を確認
- **回帰ゲートを CI へ置いた**: 非 ASCII の real-model probe は `LIVECAP_NONASCII_REAL_MODELS=1` と `slow` マーカーの**両方**を要求するため、通常 CI (`pytest tests`) では skip される。**判定を observation から regression へ変えるだけでは将来の依存更新を防げない**ので、実モデルが常駐する self-hosted Windows の `engine-smoke-gpu` job へステップを追加した
- **Tests**: `tests/integration/engines/test_reazonspeech_confidence_smoke.py` を新設し、実モデルで `avg_logprob` が `float` であることと clean sample が filter 閾値 `-0.40` を上回ることを pin する (**厳密値は固定しない** — 量子化とハードウェアで動くため、守りたいのは schema の生存と閾値との相対関係の 2 点)。既存の `test_token_confidence_populated` は NeMo 系の `token_confidence_mean` しか見ておらず、**ReazonSpeech の `avg_logprob` はどの実モデルテストにも pin されていなかった**
- **棚卸し表を再生成**: `tests/nonascii/registry.py` の sherpa 3 行を **③staging → ②wide-path** へ更新し、`benchmark_results/nonascii/2026-08-25/results.json` を新しい証拠として追加。`docs/research/nonascii-path-boundary-inventory-2026-08.md` の §0 / §3 は**自動生成**なので再生成した (`fail_silent` 7 → 6)。hotwords (#361) は**上流実装では同じ `OpenInputFile()` を通るが呼び出し箇所が無く runtime 未確認**のため、source-level の見立てとして記録し runtime 確認は #361 へ委ねる
- **Migration**: 既存ユーザーへの影響は**改善のみ**。非 ASCII なユーザー名の環境で ReazonSpeech が使えるようになる。ASCII 環境では挙動が変わらない (転写テキスト・`avg_logprob` とも実測で一致)

**本件で観測された cache の問題は 2 つに分離した。** どちらも**本件で観測されたが sherpa-onnx のバージョンに依存しない独立した bug** である。**cache identity** (キーが basename しか含まず、異なる models root の同名ディレクトリが衝突する / `tokens.txt` 更新後も古い recognizer が返る) は **[#409]**、**壊れた recognizer が無条件に strong cache へ入る**ことを止める post-load health check と保存ゲートは **[#392]** が持つ。`ModelMemoryCache` はプロセス内メモリなので、依存更新後の新プロセスへ 1.12.39 時代の壊れた recognizer が残ることはない。

#### 翻訳の失敗が黙って原文になっていた問題を修正 (Issue [#402])

上記 (Google 翻訳の復旧) と同じ issue の後半。**翻訳エンジンが何であれ、失敗が原文として出る構造そのもの**を直した。これを直さないと、次に上流が変わったとき同じ「日本語→日本語」報告が再発する。

- **Before**: 翻訳が失敗しても `logger.warning` を出すだけで `translated_text=None` を返し、表示側はそれを「翻訳なし」として原文を出していた。**ユーザには何も起きていないように見える** — 実際「モデルを変えても再起動しても直らない」という報告になった (原因は Google 側にあり、こちらは何も変わっていなかった)。swallow は `stream.py` の 3 箇所に散っていた
- **After**: **`TranslationStatusEvent`** を新設し、`set_callbacks(on_translation_status=...)` で受け取れるようにした。**segment ごとには発火しない** — `healthy→failed` と `failed→healthy` のときだけ 1 回ずつで、失敗が続く間は沈黙する。復旧も通知するので「いつまで壊れているか」が分かる
- **個々の字幕が原文のままである理由が分かる**: `TranscriptionResult.translation_state` を追加 (`not_requested` / `translated` / `failed` / `skipped_busy` / `empty`)。原文が出る状態は 1 つではなく、**障害と輻輳時の正常な方針を区別できないと今回の不具合と見分けがつかない**。独立イベントにすると `(source_id, start_time, end_time)` での突き合わせが要るため、**結果そのものの属性**にした
- **失敗理由を失わない内部型**: 3 箇所を個別に直すのではなく `_TranslationOutcome` に統一し、sync / async 双方が同じ funnel (`_settle_translation`) を通る。`_do_translate_direct` は **worker スレッドの中で動く**ため callback を呼ばず、outcome を返すだけにした
- **翻訳が文字起こしを止めない**: 翻訳を ASR とは別の executor へ分離した。以前は `max_workers=1` の executor を共用しており、**居座った翻訳が ASR 自体をブロック**していた
- **輻輳しても backlog を積まない**: 翻訳の in-flight は常に 1 件で、前が終わっていなければ後続 segment は `skipped_busy` として飛ばす。timeout した future は誰も読まないので、**古い翻訳が後から字幕に混ざることはない**。順番を守って遅れて全部出すより落とす方が字幕としては良い
- **callback の例外がパイプラインを壊さない**: 通知の失敗で文字起こしまで止まるのは本末転倒。捕捉して警告し、転写は継続する
- **`close()` は翻訳が translator を使い終わるまで待つ**: translator は呼び出し側が所有しており、`close()` の直後に `cleanup()` される。待たずに返すと**借りている `requests.Session` を使っている最中に閉じられる**。`cancel_futures=True` は実行中の future を止めないため、in-flight を明示的に待つ。**上限は設けない** — 打ち切ると、まさに待つ理由だったケースで借用中の Session を閉じさせることになる。`ThreadPoolExecutor` の worker は non-daemon で interpreter 終了時に join されるため、打ち切ってもハングから逃げられるわけでもない (1 試行を打ち切るのは translator 自身の timeout の役目)。長引いたら 1 度だけ警告する。**デストラクタからは待たない** — GC 中のブロックは危険なため。同じ契約を file pipeline にも適用し、CLI の後始末順を `pipeline.close()` → `translator.cleanup()` へ直した。file pipeline は **completeness を優先して queue する** — realtime と違い、走っている翻訳がすぐ終われば後続は自分の予算内に成功できるため。ただし timeout した queued future を `cancel()` できた場合は in-flight の参照を**実行中の previous へ戻す** — cancel された future は `done()` を返すので、そのままだと `close()` が実行中の翻訳を drain せずに返ってしまう
- **`reset()` が翻訳状態も初期化する**: 持ち越すと前セッションの `failed` のせいで次の障害が通知されず、逆に最初の成功が前セッションに対する `recovered` として出る。**in-flight は捨てない** (捨てると走っている worker と新しい翻訳が同じ translator を並行利用する) 代わりに世代番号で分離し、reset を跨いで完了した翻訳が新セッションの文脈へ書き戻さないようにした
- **`recoverable` は `error_type` から導出する**: 独立フィールドだった頃は `error_type="fatal"` かつ `recoverable=True` のような矛盾が constructor から作れた。導出にすれば矛盾が構築不能になる。`failed` は `message` も必須 (理由の分からない通知では受け手が説明できない)
- **Changed**: **`LIVECAP_TRANSLATION_TIMEOUT` の既定を `10.0` → `5.0` 秒に変更**
  - **Before**: 1 segment の翻訳を最大 10 秒待っていた
  - **After**: 5 秒。リアルタイム字幕では**遅れて届いた翻訳は今話している内容と重なるだけで価値が無い**ため 10 秒は明確に長すぎる。一方で実測 (Session 再利用時の中央値 155-191ms、観測した最悪 1331ms) に対し 5 秒は 4 倍近い余裕があり、回線の遅い環境や重いローカルモデルでも正常な翻訳を切らない
  - **Migration**: 従来どおり 10 秒待ちたい場合は `LIVECAP_TRANSLATION_TIMEOUT=10` を明示する。回線が遅い環境では上げる。超過した segment は原文のまま出て `translation_state="failed"` になる
  - あわせて PR 1 で追加した `LIVECAP_TRANSLATION_REALTIME_DEADLINE` を**削除**した (未リリース)。リアルタイムはリトライしないため実効的な上限は「待つ時間」そのもので、**同じ関心事に 2 つの knob があると片方だけ設定して効かない事故になる**
- **診断**: init 時に resolved な待ち時間と translator の見積 (`estimated_attempt_seconds`) をログする。見積が待ち時間を超える設定では毎回 timeout してしまうため、食い違いを警告する。`close()` で失敗数と skip 数を出す (障害と方針を分けて集計)
- **Added**: `TranslationStatusEvent` (`livecap_cli` から re-export)、`TranslationStatus` / `TranslationErrorType` (`livecap_cli.transcription` から export — 型 alias を submodule に留めるのは `StaticSettledReason` と同じ扱い)、`TranscriptionResult.translation_state`、`set_callbacks(on_translation_status=...)`
- **Removed**: `with_retry` デコレータ。リトライを呼び出し側へ移した結果 production から使われなくなり、`RetryPolicy` と同じことを別の形でするコードが 1 つの module に 2 つ残っていた
- **Removed**: `REALTIME_RETRY_POLICY` / `resolve_realtime_deadline()` と関連定数。**リアルタイムは retry しない**ので `max_attempts=1` の policy は direct call と等価で、実際の待機上限は `StreamTranscriber` が持っている。残しておくと `LIVECAP_TRANSLATION_TIMEOUT` を 2 箇所で parse することになり、不正値で**警告が二重に出ていた**
- **Tests**: 新規 60 件。**通知が 1 回だけ発火し連打しない** / **復旧が通知される** / sync・async 双方が同じ funnel を通る / timeout も失敗として通知される / **callback が例外を投げても転写が継続する** / **発話がイベントに載らない** / `translation_state` の 5 状態 / **並行翻訳が起きない (`peak=1`)** / **翻訳が詰まっていても ASR executor が空いている** / `close()` が両方の executor を畳む / `TranslationStatusEvent` の不正状態を `__post_init__` が弾く

#### Google 翻訳が User-Agent 起因で失敗し、原文がそのまま出ていた問題を修正 (Issue [#402])

ユーザ報告「日本語→英語が日本語→日本語になった。モデルを変えても再起動しても直らない」への対応。原因は当リポジトリ外にあり、**Google が `python-requests/2.x` の User-Agent を絞り、HTTP 200 のまま本文に "Error 500" ページを返す**ようになったこと。実測では 10 回中 5 回失敗していた (ブラウザ UA では 10/10 成功)。

- **Before**: `deep-translator` 経由で `translate.google.com/m` をスクレイピング。同ライブラリは `requests.get()` を**ヘッダ無しで**呼ぶため UA を変更できず、`headers` も `session` も渡す口が無い。最新 1.11.4 でも該当コードは同一で、最終リリースは 2023-06-28。**アップグレードでは直らない**
- **After**: Google 経路の HTTP 呼び出しのみを自前 adapter に置き換え、**ブラウザ UA・timeout・`requests.Session`・transport 注入**を渡せるようにした。エンドポイントと解析対象は従来と同じ。実測で **20 連続 ok=20/20、中央値 155ms**
- **connection を再利用**: 従来は字幕 1 本ごとに TLS ハンドシェイクが走っていた。Session 再利用で **403ms → 191ms (53% 改善)**。リアルタイム字幕では体感品質そのもの
- **リトライを adapter から呼び出し側へ移動**: adapter は**自分がリアルタイムかファイル処理か判断できない**ため予算を分けられなかった。**分類は adapter、方針は呼び出し側**に分離し、adapter は HTTP 1 試行のみ + 型による分類 (`TranslationNetworkError` = 再試行の価値あり / `TranslationError` = 恒久的)。リアルタイムは **fail fast** (遅れて出す方が字幕としては邪魔)、ファイル処理は `FILE_RETRY_POLICY` (3 試行 / 10 秒)。**1 試行あたりの所要時間は translator 自身が見積もる** (`estimated_attempt_seconds`) — 同じ policy が任意の `BaseTranslator` に適用され、ローカルモデルは見積もれないため。deadline は **soft** で、「次の試行を始めてよいか」の判断 (admission control) に使う。**上限の保証ではない** — HTTP client の read timeout はバイト間の待ち時間であって総 wall-clock ではなく、実行中の 1 試行を外から止める手段も無いため
- **リアルタイム deadline は設定可能** (`LIVECAP_TRANSLATION_TIMEOUT`)。実測の最悪が 1331ms だったことから逆算。回線・地域差で一律失敗させないため固定値にしていない。不正値は警告のうえ既定へフォールバック
- **HTTP 200 に埋め込まれたエラーページを検出する**: 従来はステータスしか見ておらず、200 を成功とみなして解析に進み、要素が無いことによる例外になっていた。判定は**成功要素が取れなかった場合にのみ**行う (翻訳結果に "Error 500" が含まれ得るため)
- **翻訳対象テキストがログ・例外へ漏れないようにした**: 翻訳対象は GET query の `q=` に入るため、`requests` 由来の例外文字列には**発話内容が percent-encode された URL ごと**含まれる。`deep-translator` は `TranslationNotFound(text)` と発話そのものを例外にしており、**失敗のたびにユーザのログへ発話が書き込まれていた**。`from None` で cause chain を切り、診断情報は `provider` / `reason` / `status_code` の構造化フィールドで持ち越す。`from error` では呼び出し側が `exc_info=True` にした瞬間に `__cause__` 経由で漏れる。あわせて `file_pipeline.py` が timeout 時に `text[:50]` を直接ログしていた箇所も削除
- **Google では文脈 (context) を使わない**: 改行連結した文脈は Google では**行単位に訳され**、VAD で分割された 1 文が壊れる (`'昨日は
雨が
降りました'` → `'Yesterday
rain
I got off'`)。さらに `context[-0:]` が `context[:]` = 全履歴になる潜在バグがあり、単に 0 にすると悪化した。adapter が context を**無視する**ことで構造的に解消。`opus_mt` が [#190] で 0 にしたのと同じ理由
- **`beautifulsoup4` を使わず標準ライブラリの `html.parser` で解析**: bs4 は `deep-translator` の推移的依存でしかなく、同ライブラリを外すと存在が保証されない。新たな依存を足さずに済ませた
- **URL 長を送信前に検証**: 実測で ~16.3KB を超えると HTTP 400。**文字数ではなく percent-encode 後のバイト長**で測る (同じ 1500 文字でも ASCII 1.5KB / 日本語 13.5KB / 絵文字 18KB)
- **User-Agent はリクエスト単位で付与する**: Session の headers に設定すると、**注入された Session では設定されず #402 の障害が再発する**。こちらが所有していないオブジェクトを恒久的に変更しない意味でも正しい
- **Google の HTTP timeout を `(connect 1.5s, read 2.5s)` にした** — 実測の中央値 155-191ms、観測した最悪 1331ms に対し倍近い余裕がある。1 試行の最悪 4.0 秒が申告値になり、ファイル処理の 10 秒予算にリトライが収まる
- **ファイル処理の翻訳は pipeline が単一 worker を所有する** — 従来は呼び出しごとに executor を作っており、timeout した翻訳が走ったまま次の segment が**同じ translator を並行利用**していた (同一 `requests.Session` の並行利用)。単一 worker により後続はキューへ回り、`close()` で回収される
- **Session の所有権を明確化**: adapter が自分で生成した Session のみ `cleanup()` が close する。注入された transport は注入元が所有する。**translator インスタンスを複数の `StreamTranscriber` 間で共有しない** (`requests.Session` の並行利用は保証されない)
- **Migration**: `deep-translator` 依存を削除した。`translation` extra は**空になったが名前は維持**している (CI 5 箇所と install ドキュメントが `--extra translation` を使うため)。Google 翻訳は core の `requests` と標準ライブラリだけで動く。`--translate google` の使い方に変更は無い
- **既知の限界**: これはスクレイピングであり、**Google 側の変更で再び壊る**。過去にも結果要素の class が `t0` → `result-container` へ変わっている。壊れたときの調査手順を `docs/troubleshooting/translation.md` に用意した。自前化で変わるのは壊れる頻度ではなく、**直るまでの時間** (上流待ち = 無限 → 数時間)
- **Tests**: 新規 87 件。**UA が実際に送信されること** (根本原因そのもの) / 埋め込みエラーページ・空結果・レイアウト変更・permanent 4xx の分類 / **adapter が 1 回しか HTTP を投げないこと** / 明示 context が送信されないこと / **機密文字列がログ・例外・`exc_info=True` のいずれにも現れないこと** (6 失敗形 × 3 観点) / URL 長超過を ASCII・日本語・絵文字それぞれで送信前に弾くこと / Session 所有権 / `RetryPolicy` の deadline が試行回数より優先されること / `@pytest.mark.network` による実エンドポイント疎通 (訳文は変わり得るので「非空・原文と異なる・ASCII 英字を含む」の緩い検証)

#### runtime の FFmpeg 自動ダウンロードが全プラットフォームで 404 していた問題を修正 (Issue [#398])

**ユーザー環境で走る経路**の修復。`_resolve_binary()` の解決順 (`LIVECAP_FFMPEG_BIN` → managed cache → 同梱 `ffmpeg-bin` → system PATH) が全て外れたとき、つまり **FFmpeg を持っていない新規ユーザーの初回起動**がこの経路に乗る。原因は 1 つではなく 6 つあった。

- **Before**: `ffmpeg-windows-64.zip` のような**実在しない資産名**を `releases/latest/download/` から取得しようとして**全プラットフォームで 404**。ffbinaries の資産名は `<tool>-<version>-<platform>.zip` 形式でバージョンが名前の一部であり、platform token も `win-64` / `linux-64` / `macos-64` である (`windows-64` / `osx-64` ではない)。アーキテクチャ判定は `"64" in platform.machine()` だったため **aarch64 に x86-64 ビルド**、`armv7l` に **x86 32bit ビルド**を渡す。ffmpeg と ffprobe は別アーカイブなのに 1 本しか取得せず、`_place_binaries()` は見つからないバイナリを `continue` で黙って飛ばしていた。checksum 検証もリトライも無し
- **After**: 固定した **ffbinaries v6.1** から **ffmpeg / ffprobe の 2 アーカイブ**を取得し、archive と展開後 binary の **SHA-256 を両方検証**する。固定値は `livecap_cli/resources/ffmpeg_manifest.json` (CI の setup action と**共有する単一の正本**)。リトライは **timeout / DNS・transport error / 408 / 429 / 5xx のみ**で指数バックオフ、permanent 4xx とチェックサム不一致は fail loud。失敗時の `FFmpegNotFoundError` は **URL・試行回数・最後のエラー・`LIVECAP_FFMPEG_BIN` による回避策**を含む
- **managed cache と host 管理を分離**: managed cache (`<cache_root>/ffmpeg`) は自動ダウンロードが置いた領域なので固定 SHA-256 と照合し、一致しなければ**対で再取得**する。`LIVECAP_FFMPEG_BIN` / 同梱 `ffmpeg-bin` / PATH のバイナリは**検証も置換もしない** — ユーザーが選んだものを勝手に差し替えない
- **pair は不可分**: ffmpeg だけ正常で ffprobe が欠けた managed cache は、`ensure_executable()` が ffmpeg を見つけた時点で戻るため**従来は永久に修復されなかった**。両方が固定値と一致したときだけ managed cache を候補にすることで、片方の破損が対全体の再取得を発火させる
- **インストールは準原子的**: 一時 workspace で 2 本ともダウンロード・SHA-256 検証・**実行確認**してから `os.replace()` で配置する。**staging 中の失敗は managed cache を一切変更しない**。配置自体は rename 2 回なので完全な原子操作ではないが、**stamp を最後に書く**ため、その間で失敗すると次回の検証が必ず不一致を検出して対ごと再取得する。同期・非同期の `ensure` は同じ lock で直列化する (#386 と同じ所有権問題)
- **壊れた managed cache は下位ソースへ fall through せず修復する**: managed cache は PATH より優先されるため、破損時に黙って PATH へ落ちると**ファイルが壊れたという理由でアプリが実行する FFmpeg が変わる**。`absent` (何も無い) と `invalid` (あるが一致しない) を区別し、invalid のときだけ修復する。修復に失敗した場合の扱いは**失敗の種類で分ける** — 上流へ到達できなかったとき (`FFmpegUpstreamUnavailable`) だけ fall through して可用性を優先し、**checksum 不一致 / permanent 4xx / archive 不正 / 実行不能 / ローカル書き込み失敗は fail loud** する。これらは「インストールしようとしたものがおかしい」という主張であり、代わりに別のビルドを黙って使うのは本 issue が排除した silent degradation そのものだから。`LIVECAP_FFMPEG_BIN` はもともと managed cache より優先なので対象外
- **対応プラットフォーム**: `win-64` / `linux-64` / `macos-64` (Intel) のみ。**macOS arm64・Linux ARM・32bit は明示的なエラー**で `brew install ffmpeg` 等の導入方法を案内する。とくに macOS arm64 で Intel ビルドを黙って Rosetta 2 実行することはしない
- **Migration**: **既存ユーザーへの影響は無い。** この経路は最初のコミット (`5824265`, 2025-11-05) から一度も成功したことがなく、release tag も存在しないため、managed cache は誰の環境でも空である。`<cache_root>/ffmpeg` へ手動でバイナリを置いていた場合のみ、固定版に置き換わる — それを避けるには `LIVECAP_FFMPEG_BIN` でそのディレクトリを指定する
- **性能**: 起動ごとに 268 MB を再ハッシュしないよう、`<cache_root>/ffmpeg/.livecap-ffmpeg.json` に `(sha256, size, mtime_ns)` を記録する。一致する間はハッシュを省略する (実測 190 ms → 0.7 ms)。これは**陳腐化と破損の検出**であり、cache ディレクトリへ書ける相手に対する防御ではない
- **ライセンス**: 取得するバイナリは **GPLv3** (`--enable-gpl --enable-version3`、`--enable-nonfree` 無し)。第三者からユーザーの環境へ取得され別プロセスとして起動されるもので、当プロジェクト (AGPL-3.0) は再配布もリンクもしない。ffbinaries-prebuilt は SPDX 未設定かつ 2023-12-28 以降更新が無く、**v6.1 は「CI と同じだから」ではなく上流の最新であるため**の選択。バージョンを上げる際は供給元の再評価とセットで CHANGELOG に記載すること
- **実装**: `livecap_cli/resources/ffmpeg_pins.py` (manifest + platform 表引き) と `livecap_cli/resources/downloader.py` (分類表・バックオフ・リトライ) を新設。`ModelManager.download_file()` は**変更しない** — 全モデル取得の挙動を巻き込むため、共通化は別途
- **Tests**: 新規 88 件 (うち 1 件は `network` マーカーで既定除外)。platform 写像 parametrize (`x86_64` / `AMD64` / `aarch64` / `arm64` / `armv7l` / `i686` × 3 OS → token または明示エラー) / manifest 整合 (資産名にバージョン・`-windows-`/`-osx-` 不在・ffprobe 名を ffmpeg から導出しない) / 分類表と partial 削除 / **pair 契約** (ffprobe 欠損・ffmpeg 改竄で対ごと再取得) / **stamp** (一致時にハッシュしない) / **原子性** (2 本目の失敗で cache 不変) / **並行 ensure が 1 回だけ install する** / `@pytest.mark.network` で固定 6 URL に HEAD

#### `unicode_safe_download_directory()` が並行処理の一時ファイルを削除していた問題を修正 (Issue [#386])

このヘルパは**プロセス全体**の `TEMP` / `TMP` / `TMPDIR` / `tempfile.tempdir` を `cache_root/downloads` へ向ける。したがってスコープが開いている間は、**プロセス内のあらゆるスレッドの `NamedTemporaryFile()` がそこへ落ちる**。それにもかかわらずスコープ退出時にその**共有**ディレクトリを `shutil.rmtree` していたため、**別処理が使用中の一時ファイルまで削除していた** — `dir=` を指定しない parakeet / canary / qwen3asr の発話ごとの wav が該当する。非 ASCII とは独立した実在の data-loss bug で、Issue [#378] の検証ハーネスで実測されたもの。

- **Before**: スコープ退出時に `cache_root/downloads` を丸ごと `shutil.rmtree` (`_cleanup_directory`)。ロック・refcount・ネスト深度カウンタのいずれも無く、並行/ネストしたスコープが互いの環境スナップショットを壊し得た
- **After**: **最外周スコープごと**に `cache_root/downloads/<uuid>` を作り、**退出時に再帰削除しない**。module level の `threading.RLock` をスコープの全期間保持して直列化し、深度カウンタにより **0→1 でのみ**環境を変更、**1→0 でのみ**復元する。ネストしたスコープは**外側のディレクトリを再利用し同じパスを返す** (環境は外側を指したままなので、別パスを返すと呼び出し側に嘘をつくことになる)。purpose が異なるネストは新設の `TempEnvironmentConflictError` で失敗する
- **Migration**: 呼び出し側の変更は不要 (シグネチャと yield 値の意味は互換)。ただし挙動が 3 点変わる — ①**スコープを抜けてもディレクトリが残る** (回収は Issue [#375] PR 2 の lease / reaper が担当。「別 pid かつ N 時間経過」は生存判定にならないため、安全に回収できる仕組みが揃うまで意図的にリークを受け入れる)、②**別スレッドのダウンロードスコープは直列化される** (プロセス全体の状態を書き換える以上、並行実行は一貫させられない)、③ASCII 保証は**依然として無い** (`cache_root` はユーザー名を含む)。無関係な一時ファイルの**置き場所がずれる問題も未解消**で、消えなくなるだけである。①〜③の恒久対応は Issue [#375] PR 2 / PR 3 が担当する
- **Tests**: `tests/core/utils/test_temp_environment.py` 新規 10 件 (別スレッド / 子プロセスのファイル残存、直列化、ネストのパス一致、例外時の復元、再帰削除しないこと、purpose 衝突)。`tests/nonascii/` のプローブ期待値を `victim_survived_scope_exit=True` へ反転し、**再発したら落ちる**向きで固定

#### Voxtral が `language="auto"` 文字列を mistral-common へ渡していた契約違反を修正 (Issue [#365])

`VoxtralEngine` は `__init__` の生値 (default `"auto"` を含む) をそのまま `processor.apply_transcription_request(language=...)` へ渡していたが、mistral-common の `TranscriptionRequest.language` は **`LanguageAlpha2 | None`** (自動検出 = `None`、具体言語 = ISO 639-1) であり `"auto"` は契約外だった。

- qwen3asr `_resolve_language` と同型の `_asr_language` を導入: `auto`/`None`/空 → `None`、具体言語 → `to_iso639_1` で ISO 639-1 正規化
- 推論時は `language=self._asr_language` を渡す (`self.language` は生値のままログ用に保持)
- **Tests**: `tests/core/engines/test_voxtral_language.py` 新規 — mock processor で上流引数をモデルロードなしに固定 (auto → `None` / `en` → `"en"` / constructor 経由の wiring)

#### CLI `transcribe <file>` が構築時 TypeError で全滅していた問題を修正 (Issue [#363])

`livecap-cli transcribe <file>` (CLI ファイル文字起こし経路) は Phase 6B (ee4ffdc) の初回統合時から `FileTranscriptionPipeline` の**実在しない API** (`FileTranscriptionPipeline(engine=)` / `pipeline.transcribe()` / `result.to_srt()` / `result.segments`) を前提に実装されており、pipeline 構築時点で `TypeError` となり一切機能しなかった (CLI file 成功経路のテストが無く未検出)。`--translate` 経路も実在しない `translator.initialize()` を呼んでいた (実 API は `load_model()`)。

- **`livecap_cli/cli.py:_transcribe_file` を実 pipeline 契約へ全面書き換え**:
  - engine から `segment_transcriber` closure を構築 (`engine.transcribe(audio, sr).text` — `TranscriptionResult` は #314 以降 tuple unpack 不可) → `pipeline.process_file(path, segment_transcriber=…, write_subtitles=False, …)`
  - translator lifecycle 修正: `load_model()` 呼び出し + 言語ペア (`--language` / `--target-lang`) を `create_translator()` へ渡す (OPUS-MT が constructor default `ja→en` に固定される問題も同時解消) + finally で `cleanup()`
  - `result.success=False` (Issue [#362] の全滅検出) を exit 1 + stderr `error` 表示に反映
  - 入力ファイルの存在検証をモデルロード前に移動
  - engine / translator / pipeline の cleanup を finally で保証
- **`FileTranscriptionPipeline.close()` の `getattr` 防御**: 構築時 TypeError で `__init__` 本体が未実行のままインスタンスが GC されると `__del__` → `close()` が未初期化 `_temp_root` を参照して二次 `AttributeError` を出していた
- **`transcribe --help` が日本語 Windows console (cp932) で crash する既存バグを修正**: `--confidence-filter` help に cp932 非対応の em-dash (U+2014) が混入しており `UnicodeEncodeError` で help 表示自体が失敗していた (テストは StringIO 捕捉のため CI で未検出)。help 文字列を ASCII 安全化し、cp932 encode の regression テストで固定
- **help 表記**: realtime 専用オプション 19 件の argparse help に `[realtime only]` を明記、`-o` help に「省略時 stdout」を明記
- **refactor**: engine 生成 boilerplate (device 解決 + kwargs routing + `create_engine` + `load_model`) を `_load_engine()` に一元化し realtime / file 両経路で共有 (経路間 contract drift — #363 の原因パターン — の再発防止)
- **Tests**: `tests/core/cli/test_transcribe_file.py` 新規 13 件 — mock engine が実 `TranscriptionResult` を返し **pipeline は mock せず** temp WAV から実 `process_file()` を通す CLI E2E (engine/torch/FFmpeg/network 不要)。今回のような API drift を CI で検出可能にする

**関連**: [Issue #363](https://github.com/Mega-Gorilla/livecap-cli/issues/363) / [#362] (pipeline 側全滅検出 — 本修正の土台) / [#365] (`--language` routing、別 issue) / [#366] (file mode filter parity、別 issue)

#### Voxtral cache が `LIVECAP_ENGINE_STRONG_CACHE` に従わない問題を修正 (Issue [#198])

Voxtral engine は `(model, processor)` **tuple** を cache していたが、 tuple は `weakref.ref()` を作れないため `ModelMemoryCache.set()` の weak-cache path で `TypeError` になり、 **`LIVECAP_ENGINE_STRONG_CACHE` の設定に関わらず常に強参照** で cache されていた (= env var が Voxtral に対して no-op、 一度 load すると同一プロセス内で VRAM を永続保持)。

- **`livecap_cli/engines/voxtral_engine.py`**: `(model, processor)` tuple を **`VoxtralModelContainer` dataclass** (weakref 可能) に置換
  - `_load_model_from_path()`: `VoxtralModelContainer(model, processor)` を cache に保存、 env var 未 opt-in 時は `allow_promotion=False` も渡す
  - `_configure_model()`: container から model / processor を分離しつつ、 container への strong ref を `self._model_container` で保持 (weak-cache された container が engine 生存中に GC されないように)。 旧 tuple コード由来の未到達 `else` fallback (「単一モデルが直接渡された場合」の compat path、 `_load_model_from_path` は必ず container を返すため dead) を削除し、 contract-trust の bare unpacking に整理 (Issue [#321] 方針)。 未使用の `Tuple` import も削除
  - `cleanup()`: `self._model_container = None` で解放 → 最後の engine が消えれば container も GC され VRAM 解放
- **`livecap_cli/engines/model_memory_cache.py`**: `set()` に `allow_promotion: bool = True` param 追加 (codex-review)
  - `get()` の weak-hit auto-promotion (`_access_count > 3` で `_promote_to_strong_ref`) は、 `allow_promotion=False` の key では走らない
  - **これがないと env var 未設定でも hot-access (4 回目〜) で weak→strong 昇格し VRAM を永続保持してしまう** — Issue #198 の目的 (env var で制御) と衝突するため fix
- **挙動 (env var 別)**:
  - `LIVECAP_ENGINE_STRONG_CACHE=1/true/yes`: 強参照 cache (VRAM 保持、 cross-instance 高速再利用) — 従来と同じ
  - **未設定 (default)**: **weak-cache + auto-promotion 無効** — engine 生存中は再利用、 hot-access でも昇格せず、 最後の参照が消えれば GC で VRAM 解放 (**修正前は env var 無視で常に強参照 or hot-access で昇格していた**)
- **Tests**: 新規 `tests/core/engines/test_voxtral_cache.py` 9 test — tuple が weakref 不可 (root cause) / container が weakref 可能 / weak-cache が holder drop 後に GC / strong-cache が生存 / engine holder が weak-cache を生かす / **default auto-promotion (regression guard)** / **`allow_promotion=False` で >3 access でも weak 維持** / `_no_promote_keys` bookkeeping。 実 Voxtral model は load せず参照挙動のみ検証 (GPU 不要)。
- Option A (weakref-able container + auto-promotion opt-out) 採用 — env var 未設定でも weak-cache として engine 生存中の再利用を維持しつつ、 hot-access 昇格も含め VRAM 永続保持を回避。

#### Layer 3 noise rotation bias fix — 646 pool 全体を uniform stride で span (Issue [#338] Phase 2b)

[Phase 2 report](docs/research/calibration-japan-engines-phase2-2026-07.md) §5.4 / §5.7 で判明した Layer 3 noise diversity bug の恒久修正。 `benchmarks/confidence_calibration/gen_mixed_noisy_speech.py` の `noise_pool[i % len(noise_pool)]` rotation は、 `select_noise_pool` の path sort と組み合わさって **noise pool の先頭 N=n_samples entries のみ選択** する意図しない挙動になっていた。 ESC-50 default 15 categories (alphabetically: breathing → car_horn → ...) では typical n_samples=50 で **breathing (30 files × 5 SNR = 150 sample) + car_horn (20 files × 5 SNR = 100 sample) の 2 subtypes 250 sample だけ** が Layer 3 に含まれ、 remaining 13 ESC-50 categories + MUSAN 全 pool 未使用だった。 dev-only、 production runtime (`livecap_cli/`) には影響なし。

**Before / After distribution** (n_samples=50、 646 pool = 450 ESC-50 + 196 MUSAN):

| Subtype coverage | Before (`i % len`) | After (`_uniform_stride_indices`) |
|---|---|---|
| ESC-50 categories touched | **2 / 15** (breathing、 car_horn のみ) | **15 / 15** (全 category) |
| MUSAN subtypes touched | **0 / 2** (未評価) | **2 / 2** (free-sound、 sound-bible) |
| breathing sample 比率 | **60%** (150/250) | **~7%** (~1/15) |
| car_horn sample 比率 | **40%** (100/250) | **~7%** (~1/15) |
| Layer 3 total variety | 2 subtypes | 17 subtypes (15 ESC-50 + 2 MUSAN) |

**主要変更** (`benchmarks/confidence_calibration/gen_mixed_noisy_speech.py`):

- 新規 `_uniform_stride_indices(pool_size, n_samples) -> list[int]` helper 追加
- `augment()` 内の rotation loop を stride-based に変更:
  ```python
  # Before (biased to first N sorted entries)
  noise_entry = noise_pool[i % len(noise_pool)]
  # After (uniform stride via np.linspace)
  noise_indices = _uniform_stride_indices(len(noise_pool), len(speech_samples))
  noise_entry = noise_pool[noise_indices[i]]
  ```
- Module docstring "Design (Plan D3)" を revised
- 挙動: `n_samples ≤ pool_size` 時は `np.linspace(0, pool_size-1, n_samples)` round で均等分布、 `n_samples > pool_size` 時は各 index を floor/ceil 回数使用 (grouped、 max diff = 1)

**Tests**: 12 新 test 追加 (`TestUniformStrideIndices` × 10 [初回 7 + codex-review 2nd round で pool=3/n=8, pool=4/n=7 の regression + property test の 3 追加] + `TestNoiseSubtypeDiversity` × 2)、 既存 `test_noise_rotation` を invariant-based に update (rename も含む、 rotation pattern を pin しない設計)、 `_fake_corpus_multi_subtype` helper 追加。 全 63 test pass in `test_gen_mixed_noisy_speech.py` + 419 pass in calibration suite、 退行ゼロ。

**Migration (既存 Layer 3 corpus を持つ user)**:

```bash
# Layer 3 entries を削除して再生成 (Layer 1/2 は保持)
uv run python -m benchmarks.confidence_calibration.gen_mixed_noisy_speech \
    --samples 50 --snr-db-list="-5,0,5,10,20" --force --speech-language ja
# --force は source_dataset=layer3_mix の既存 entries を消して再生成

# ja_noisy_speech/ 内の旧 wav が残る場合は手動削除推奨:
rm -rf "$LIVECAP_CALIBRATION_CORPUS_DIR/ja_noisy_speech"/*.wav
```

**フォローアップ (別 PR で対応予定)**:

- Phase 2 report §5.7 addendum: user GPU 環境で corpus 再生成 + Phase 5 sweep 再実行 (RTX 4090 で ~20 min) 後、 Before/After 実測比較を report に追記
- 必要に応じて Issue #334 PR-4 default 閾値 (`avg_logprob_thresholds["qwen3-asr"] = -0.42` 等) の再算定 (`qwen3-asr: -0.42` の根拠が §2.2 で breathing + car_horn only を base としていたため、 diverse subtype 下での SNR 10 dB borderline 6% が悪化 / 改善する可能性を Layer 4 replay と合わせて判断)

**関連**: Parent Issue [#338](https://github.com/Mega-Gorilla/livecap-cli/issues/338)、 上流依存 PR #347 (Issue #334 PR-4、 merged) + PR #348 (calibration corpus persistent dir、 merged)

### Documentation

#### 新規 ASR engine 実装 contributor guide 追加 (Issue [#334] PR-6)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit で
発見した「engine 追加時の docstring stale 化」「signal scale 誤認」「silent
fail-open」、および「量子化 calibration 観点の明文化」を構造的に行うため、
新規 ASR engine 追加 contributor 向けの **single source of truth doc** を
新設。本 audit の findings (F2 / F5 / F6 / F8) を anti-pattern として codify
(F8 は既存 ReazonSpeech では PR-A.5.1 で int8 / float32 両方 verify 済、本
codify は新 engine への一般則として位置付け)。

- **`docs/contributor/adding-an-engine.md` 新規**: 9 section (Quickstart 10-step
  checklist / Engine 契約 / 登録 flow / Confidence signal extraction / Threshold
  calibration / 既存 7 engine の reference table / Anti-patterns AP-1 ~ AP-5 /
  Testing 慣用 pattern / CHANGELOG・docs update checklist) を 1 doc で完結
  (~444 行)。
- **`livecap_cli/engines/base_engine.py` `BaseEngine` class docstring 拡張**:
  必須 attribute (`engine_name` / `device`) / Abstract method 4 個 / Hook method
  6 個 / Optional contract (`engine_confidence` populate) を明文化、
  `docs/contributor/adding-an-engine.md` への link。
- **`CLAUDE.md` / `AGENTS.md` cross-reference**: engine adapter section に
  「新規 engine 追加時は `docs/contributor/adding-an-engine.md` 参照」を 1 行
  ずつ追加。
- **Codified anti-patterns** (Issue #334 audit 由来):
  - **AP-1** (F2): 「engine_confidence は常に全 None」 docstring → 後で populate
    追加時に stale 化、新規 consumer が誤読
  - **AP-2** (F2): `token_confidence_mean` threshold を直感で 0.5 等に変更
    → engine 別 scale (Parakeet ja 0.0504 / en 0.2452 / Canary 0.0724) を
    知らないと全 speech false reject regression
  - **AP-3** (F8、一般則): 新 engine 追加時に量子化 (int8 / float32) を smoke
    verify せず threshold 採用 → 量子化で signal 分布が変わる可能性。
    既存 ReazonSpeech は PR-A.5.1 で両量子化 verified (margin +0.13 / +0.10)、
    本 codify は新 engine への一般則。
  - **AP-4** (F6): auto-detect / fail-open path を user 通知なしで残す
    → 「filter on にしたのに reject 0 件」silent failure。
    `StreamTranscriber._maybe_warn_qwen3_auto_detect_fail_open` (PR #336) が
    参考実装
  - **AP-5** (本 doc 自身に対する meta-rule): 新 engine 追加時に本 doc の
    reference table を update しない → doc が stale 化

#### Engine confidence signal semantics clarified (Issue [#334] Findings 1 / 2 / 5)

Issue [#334](https://github.com/Mega-Gorilla/livecap-cli/issues/334) audit
で発見した既存 docstring と実装の乖離 + signal semantics の誤認 risk を
docstring/comment レベルで解消。code 挙動の変更なし、low-risk な
documentation cleanup。

- **`EngineConfidence` の各 field 説明を `Attributes:` section に拡充**
  (`livecap_cli/engines/base_engine.py:22-44`):
  - 各 field の **scale / populate engine / filter 取扱**を明記
  - `token_confidence_mean` の **低 scale (Parakeet ja ≈ 0.0504、
    Parakeet en ≈ 0.2452、Canary en ≈ 0.0724、典型 NeMo confidence 0.85+
    ではない)** を明示 (Issue #334 Finding 2)
  - 「ReazonSpeech / qwen3asr は未対応で全 None」という冒頭の stale 記述を
    削除 (PR-A.5.1 / PR-A.5.2 で対応済)
- **`ReazonSpeechEngine.transcribe()` docstring を PR-A.5.1 反映**
  (`livecap_cli/engines/reazonspeech_engine.py:443-454`):
  - 以前は「`engine_confidence` は **常に全 None**、filter fail-open」と
    読めたが、現在は `avg_logprob` populate 済 (sherpa-onnx 1.12.39+ の
    `ys_log_probs` mean、engine-specific threshold `-0.2`)
- **`FilterConfig.no_speech_threshold` の公式 Whisper 0.6 との差を明記**
  (`livecap_cli/transcription/confidence_filter.py:86-101`):
  - livecap-cli は ``0.5`` (公式より ``0.1`` strict)、PR-A.0 data-calibrated
  - Speech margin / non-speech margin の数値も明記 (Issue #334 Finding 1)
- **`FilterConfig.token_conf_threshold` の docstring に engine 別 scale 追加**
  (`livecap_cli/transcription/confidence_filter.py:102-120`):
  - 「threshold を高い値に変更すると全 speech が false reject される深刻
    regression」を明示 (Issue #334 Finding 2)
- **`FilterConfig.compression_ratio_threshold` の「未使用予約 field」を実態
  に書き換え** (`livecap_cli/transcription/confidence_filter.py:121-128`):
  - extract logic は実装済だが、**現 CTranslate2 backend (WhisperS2T base)
    では `compression_ratio` は常に `None`** (`whispers2t_engine.py:31-33`
    smoke verify 済)
  - forward-compatibility 用、enable には populate verify + calibration の
    2 段階が必要 (Issue #334 Finding 5)

### Security


- No security issues in this release


---

## Migration Guide

### From `livecap-core` to `livecap-cli`

#### 1. Update package installation

```bash
# Before
pip install livecap-core[engines-torch]

# After
pip install livecap-cli[engines-torch]

# Or use the recommended bundle:
pip install livecap-cli[recommended]
```

#### 2. Update imports

```python
# Before
from livecap_core import StreamTranscriber, EngineFactory
from livecap_core.vad import VADProcessor, VADConfig

# After
from livecap_cli import StreamTranscriber, EngineFactory
from livecap_cli.vad import VADProcessor, VADConfig
```

#### 3. Update CLI commands

```bash
# Before
livecap-core --info
livecap-core --as-json

# After
livecap-cli info
livecap-cli info --as-json

# New commands
livecap-cli devices
livecap-cli engines
livecap-cli translators
livecap-cli transcribe input.mp4 -o output.srt
livecap-cli transcribe --realtime --mic 0
```

#### 4. Update result handling (if using old dict API)

```python
# Before (TranscriptionEventDict)
result = {"text": "...", "start": 0.0, "end": 1.0}
print(result["text"])

# After (TranscriptionResult dataclass)
# result is now a dataclass with attributes
print(result.text)
print(result.start_time)
print(result.end_time)
print(result.to_srt_entry(index=1))
```

---

## Issue References

- Epic: [#64] - livecap-cli リファクタリング
- Phase 1: [#69] - リアルタイム文字起こし実装
- Phase 2: [#70] - API 統一と Config 簡素化
- Phase 3: [#71] - パッケージ構造整理
- Phase 4: [#72] - 翻訳機能実装
- Phase 5: [#73] - エンジン最適化
- Phase 6: [#74] - 依存関係整理・CLI・パッケージ名変更
- Docs: [#75] - ドキュメント更新

---

[Unreleased]: https://github.com/Mega-Gorilla/livecap-cli/compare/main...HEAD

[#64]: https://github.com/Mega-Gorilla/livecap-cli/issues/64
[#65]: https://github.com/Mega-Gorilla/livecap-cli/issues/65
[#66]: https://github.com/Mega-Gorilla/livecap-cli/issues/66
[#67]: https://github.com/Mega-Gorilla/livecap-cli/issues/67
[#68]: https://github.com/Mega-Gorilla/livecap-cli/issues/68
[#69]: https://github.com/Mega-Gorilla/livecap-cli/issues/69
[#70]: https://github.com/Mega-Gorilla/livecap-cli/issues/70
[#71]: https://github.com/Mega-Gorilla/livecap-cli/issues/71
[#72]: https://github.com/Mega-Gorilla/livecap-cli/issues/72
[#73]: https://github.com/Mega-Gorilla/livecap-cli/issues/73
[#74]: https://github.com/Mega-Gorilla/livecap-cli/issues/74
[#75]: https://github.com/Mega-Gorilla/livecap-cli/issues/75
[#171]: https://github.com/Mega-Gorilla/livecap-cli/pull/171
[#173]: https://github.com/Mega-Gorilla/livecap-cli/pull/173
[#175]: https://github.com/Mega-Gorilla/livecap-cli/pull/175
[#180]: https://github.com/Mega-Gorilla/livecap-cli/pull/180
[#181]: https://github.com/Mega-Gorilla/livecap-cli/pull/181
[#182]: https://github.com/Mega-Gorilla/livecap-cli/pull/182
[#184]: https://github.com/Mega-Gorilla/livecap-cli/pull/184
[#186]: https://github.com/Mega-Gorilla/livecap-cli/pull/186
[#187]: https://github.com/Mega-Gorilla/livecap-cli/pull/187
[#189]: https://github.com/Mega-Gorilla/livecap-cli/pull/189
[#191]: https://github.com/Mega-Gorilla/livecap-cli/pull/191
[#194]: https://github.com/Mega-Gorilla/livecap-cli/pull/194
[#196]: https://github.com/Mega-Gorilla/livecap-cli/pull/196
[#197]: https://github.com/Mega-Gorilla/livecap-cli/pull/197
[#201]: https://github.com/Mega-Gorilla/livecap-cli/pull/201
[#278]: https://github.com/Mega-Gorilla/livecap-cli/issues/278
[#279]: https://github.com/Mega-Gorilla/livecap-cli/pull/279
[#280]: https://github.com/Mega-Gorilla/livecap-cli/issues/280
[#281]: https://github.com/Mega-Gorilla/livecap-cli/pull/281
[#283]: https://github.com/Mega-Gorilla/livecap-cli/issues/283
[#291]: https://github.com/Mega-Gorilla/livecap-cli/issues/291
[#292]: https://github.com/Mega-Gorilla/livecap-cli/issues/292
[#230]: https://github.com/Mega-Gorilla/livecap-cli/issues/230
[#295]: https://github.com/Mega-Gorilla/livecap-cli/issues/295
[#314]: https://github.com/Mega-Gorilla/livecap-cli/issues/314
[#362]: https://github.com/Mega-Gorilla/livecap-cli/issues/362
[#363]: https://github.com/Mega-Gorilla/livecap-cli/issues/363
[#365]: https://github.com/Mega-Gorilla/livecap-cli/issues/365
[#366]: https://github.com/Mega-Gorilla/livecap-cli/issues/366
[#375]: https://github.com/Mega-Gorilla/livecap-cli/issues/375
[#377]: https://github.com/Mega-Gorilla/livecap-cli/issues/377
[#378]: https://github.com/Mega-Gorilla/livecap-cli/issues/378
[#379]: https://github.com/Mega-Gorilla/livecap-cli/issues/379
[#380]: https://github.com/Mega-Gorilla/livecap-cli/issues/380
[#387]: https://github.com/Mega-Gorilla/livecap-cli/issues/387
[#386]: https://github.com/Mega-Gorilla/livecap-cli/issues/386
[#395]: https://github.com/Mega-Gorilla/livecap-cli/issues/395
[#398]: https://github.com/Mega-Gorilla/livecap-cli/issues/398
[#190]: https://github.com/Mega-Gorilla/livecap-cli/issues/190
[#402]: https://github.com/Mega-Gorilla/livecap-cli/issues/402
[#392]: https://github.com/Mega-Gorilla/livecap-cli/issues/392
[#406]: https://github.com/Mega-Gorilla/livecap-cli/issues/406
[#403]: https://github.com/Mega-Gorilla/livecap-cli/issues/403
[#407]: https://github.com/Mega-Gorilla/livecap-cli/issues/407
[#382]: https://github.com/Mega-Gorilla/livecap-cli/issues/382
[#413]: https://github.com/Mega-Gorilla/livecap-cli/issues/413
[#422]: https://github.com/Mega-Gorilla/livecap-cli/issues/422
[#425]: https://github.com/Mega-Gorilla/livecap-cli/issues/425
[#426]: https://github.com/Mega-Gorilla/livecap-cli/pull/426
[#428]: https://github.com/Mega-Gorilla/livecap-cli/issues/428
[#429]: https://github.com/Mega-Gorilla/livecap-cli/pull/429
[#430]: https://github.com/Mega-Gorilla/livecap-cli/issues/430
[#409]: https://github.com/Mega-Gorilla/livecap-cli/issues/409
[#418]: https://github.com/Mega-Gorilla/livecap-cli/issues/418
