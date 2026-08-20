# 非 ASCII パス境界の棚卸し (2026-08 実測)

Issue [#378](https://github.com/Mega-Gorilla/livecap-cli/issues/378) — Epic [#380](https://github.com/Mega-Gorilla/livecap-cli/issues/380) の Phase 0。

livecap_cli が **ネイティブ / 第三者ライブラリへ filesystem パスを渡している箇所**を
網羅的に列挙し、非 ASCII パスでの挙動を実測し、境界ごとに採るべき方式を確定する。

> **§0 と §3 は自動生成される。手で書き換えないこと。**
>
> ```bash
> LIVECAP_NONASCII_REAL_MODELS=1 uv run pytest tests/nonascii -m "nonascii_paths" \
>     --nonascii-report=benchmark_results/nonascii/<date>/results.json
> uv run python -m tests.nonascii.report --json benchmark_results/nonascii/<date>/results.json \
>     --inject docs/research/nonascii-path-boundary-inventory-2026-08.md
> ```

---

## なぜ調査から始めたのか

非 ASCII パス起因の不具合が、実ユーザー報告から 2 件見つかっていた。

| 境界 | 症状 | 失敗の可視性 |
|---|---|---|
| sentencepiece (NeMo `restore_from`) | モデル復元失敗 → 抽象クラスの二次例外にすり替わる | 元例外が消える |
| sherpa-onnx `tokens.txt` | ロードは成功、デコードが全件 `IndexError` | **兆候ゼロ** |

いずれも「偶然ユーザーが報告してくれたから見つかった」2 箇所であり、網羅性の根拠がない。
個別に潰しても同じクラスの不具合が別の境界で再発するため、まず露出を確定させる必要があった。

**推測では切り分けられない。** 同一プロセス内で、ライブラリごとに wide path 対応状況が
バラバラであることが実測で確認できる (§3 参照) — `soundfile` は `sf_wchar_open` を使い、
`onnxruntime` は非 ASCII でも通る一方、`sherpa-onnx` は narrow path で黙って壊れる。
コードを読むだけでは決まらず、**実測が要る**。

---

<!-- BEGIN:METADATA -->
## 0. 測定メタデータ

| 項目 | 値 |
|---|---|
| OS / arch | Windows-10-10.0.26200-SP0 / AMD64 |
| Python | 3.11.13 |
| 実測ホスト | 開発機 (Windows / 日本語ロケール) |
| ANSI code page (ACP) | 932 |
| OEM code page | 932 |
| filesystem encoding | utf-8 |
| locale preferred encoding | cp932 |
| Python UTF-8 mode | False |
| LongPathsEnabled | True |
| 8.3 生成の状態 | ボリュームを開けませんでした。 / エラー 5: アクセスが拒否されました。 |
| Windows ユーザー名は ASCII か | True |
| システム %TEMP% は ASCII か | True |
| プローブ root のボリューム | C:\ |
| 実モデルの実体化方式 | mixed |
| 対応した variant | control, cjk_kana, outside_acp, space_paren, nfd |
| 非対応の variant | なし |
| NFD 正規化の保存 | True |
| 有効な tier | cheap, real_model |
| git commit | 1a9492134a09cb049603d701466e49831d038599 |
| run_id | 2026-08-20T05-46-26Z |
| 最終検証日 | 2026-08-20 |

パッケージ版数:

| パッケージ | 版 |
|---|---|
| appdirs | 1.4.4 |
| ffmpeg-python | 0.2.0 |
| huggingface-hub | 0.36.0 |
| librosa | 0.11.0 |
| nemo-toolkit | (not installed) |
| numpy | 1.26.4 |
| onnxruntime | 1.23.2 |
| qwen-asr | (not installed) |
| safetensors | 0.6.2 |
| sentencepiece | (not installed) |
| sherpa-onnx | 1.12.39 |
| soundfile | 0.13.1 |
| tokenizers | 0.22.1 |
| torch | 2.9.1+cu128 |
| transformers | 4.57.6 |
| whisper-s2t | 1.3.1 |
<!-- END:METADATA -->

---

## 1. 分類方式の定義

境界ごとに、**この優先順位**で方式を決める。
`ascii_safe_path()` (= ③) は **第 3 の fallback であり共通解ではない**。

| # | 方式 | 採用条件 | 採用に必要な証拠 |
|---|---|---|---|
| ① | bytes / serialized-proto / buffer API | パスではなくデータを渡せる API がある | 該当 API の存在をライブラリ版数付きで確認 |
| ② | wide path 対応済み (現状維持) | 実測で非 ASCII が通る | control と観測的に等価であること (`pass`) |
| ③ | ASCII staging (`ascii_safe_path()`) | ①②が使えない narrow path 境界 | ①が無いことの確認 + 実測 NG |
| ④ | fail-fast | ③も成立しない (staging root が確保できない等)、あるいは staging では直らない failure family | — |
| — | 非該当 | そもそもパス境界でない (ndarray 授受など) | なぜパス境界でないかの説明 |

**方式①②が使える境界に ③ を適用してはならない。** 実装 PR は、③ を追加する際に
「①②が使えないことの証拠 (ライブラリ版数を含む)」を呼び出し箇所のコメントに残すこと。

### 完了条件 (機械化済み)

`tests/nonascii/test_registry.py` が以下を CI で強制する:

- `test_no_unclassified_rows` — **未分類ゼロ**
- `test_no_unassigned_silent_failure_rows` — 黙って壊れると実測された行に ② (現状維持) を割り当てない
- `test_callsites_exist` — 表がコードとずれていない (行番号ではなく symbol で追跡)
- `test_every_row_has_evidence` — 全行が証拠種別を持ち、runtime 行は実在の probe を指す
- `test_unmeasured_rows_state_why` — 未実測の行は理由を明記している
- `test_staging_rows_have_granularity` — ③ の行は粒度 (file / dir / %TEMP%) が決まっている

---

## 2. 判定語彙

すべてのプローブは同じ操作を 2 回走らせる — ASCII の **control** root で 1 回、
非 ASCII **variant** で 1 回 — そして verdict は**その比較**から導出する
(**differential 方式**)。固定の期待値を持たないのでモデル / ライブラリ更新に耐え、
何より **`fail_silent` を機械的に検出できる唯一の方法**である。

| verdict | 意味 |
|---|---|
| `pass` | control と観測的に等価 |
| `fail_loud` | 失敗するが**問題のパスを名指しする** (診断可能)。ネイティブ `abort()` による非ゼロ終了・timeout もここ |
| `fail_silent` | **利用者に何が起きたか分からない失敗**。epic #380 の中核関心事 |
| `skipped` | 依存未導入 / FS が variant を拒否 / tier gate off。理由を必ず記録 |
| `error_harness` | **control が失敗した** = プローブのバグ。バグの証拠として数えない |

`fail_silent` の判定根拠 (`silent_criteria_hit`):

1. `no_exception_output_differs_from_control` — 例外なしで観測が control と異なる
2. `deferred_failure_at_later_stage` — control が成功した地点より**後段**で落ちた (遅延失敗)
3. `mangled_exception:...` — 真因が既知の汎用メッセージにすり替わっている
4. `exit_zero_but_no_result` / `exception_does_not_name_path`

### variant は「日本語を混ぜたもの」ではない

各 variant は**別々の失敗機構**を切り分ける。ある行が `cjk_kana` を通るのに
`outside_acp` で落ちる場合と、`space_paren` で落ちる場合とでは、必要な修正が根本的に異なる。

| id | segment | 切り分ける機構 |
|---|---|---|
| `control` | `ascii_control` | 差分の基準 |
| `cjk_kana` | `ユーザー` | 実世界ケース (JP ユーザー名)。cp932 の**内側** / cp1252 の**外側** → 「UTF-8 で書いたものを Win32 narrow API が ACP として解釈」を切り分ける |
| `outside_acp` | `한국어Ω` | cp932 と cp1252 の**両方の外側** → JP 開発機でも en-US CI でも「ACP で表現不能」モードを強制し、両ホストの結果を比較可能にする |
| `space_paren` | `テスト フォルダ (1)` | 空白 + 括弧 = **別の failure family** (argv quoting であってエンコーディングではない)。**ASCII staging では直らない**バグを捕まえる |
| `nfd` | NFD 分解形 (`か` + U+3099) | 正規化を仮定するライブラリがファイルを見失う。契約が NFC 入力を仮定してはいけない根拠 |
| `emoji_astral` | astral 面 (U+1F3B5) | UTF-16 サロゲートペア。BMP(UCS-2) 前提を突く (既定 off) |
| `long_mixed` | 260 文字近く | `MAX_PATH` との相互作用 (別軸・別 issue、既定 off) |

FS が variant を受理しない場合 (macOS APFS の NFC/NFD 正規化など) は `skipped` + 理由として
記録される。**「非 ASCII が通った」と「非 ASCII を試していない」を混同させないため。**

---

<!-- BEGIN:SUMMARY -->
## 集計

- 棚卸し行数: **44**、未分類: **0**
- 実測レコード数: **127**
- 方式の内訳: ②wide-path 27 行 / ③staging 14 行 / ④fail-fast 1 行 / 非該当 2 行
- 判定の内訳: ⚠️ fail_loud 2 / 🔴 **fail_silent** 5 / ✅ pass 119 / ⏭ skipped 1
<!-- END:SUMMARY -->

---

<!-- BEGIN:TABLE -->
## 3. 棚卸し表

### 3.1 エンジンモデルロード

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/engines/reazonspeech_engine.py:393` | モデルディレクトリ (basedir) に tokens.txt / encoder / decoder / joiner を os.path.join | sherpa-onnx (native, narrow path) | 非対応 | 🔴 **fail_silent**: cjk_kana | **黙る**。ロードは成功し decode が全件 IndexError。さらに壊れた recognizer が ModelMemoryCache.set(..., strong=True) でプロセス寿命の間キャッシュされる。 (判定根拠: deferred_failure_at_later_stage) | **③staging** | dir | #377 |
| `livecap_cli/engines/reazonspeech_engine.py:401` | hotwords ファイル (#361 で追加予定。現時点では未実装) | sherpa-onnx (native, narrow path) | 非対応 | — 未実測 (#361 未実装のため呼び出し箇所がまだ存在しない) | 未実装。#361 実装時に本行を runtime 実測へ格上げすること。 | **③staging** | file | #361 |
| `livecap_cli/engines/parakeet_engine.py:259` | .nemo ファイルの絶対パス (str(model_path)) | NeMo (tar 展開) → sentencepiece (native, narrow path) | 非対応 | — 未実測 (sentencepiece / nemo-toolkit が engines-nemo extra 側で未導入。`uv sync --extra engines-nemo` が必要 (GB 級)。.nemo 自体はローカルに存在する。) | **黙る / すり替わる**。元例外が抽象クラスの二次例外に置換される。加えて nemo_utils.check_nemo_availability() が NEMO_AVAILABLE=False をプロセス全体に キャッシュし、呼び出し側は汎用 ImportError('NeMo is not installed') を raise する。 | **③staging** | file | #379 |
| `livecap_cli/engines/canary_engine.py:275` | .nemo ファイルの絶対パス (str(model_path)) | NeMo (tar 展開) → sentencepiece (native, narrow path) | 非対応 | — 未実測 (sentencepiece / nemo-toolkit 未導入 (engines-nemo extra)) | **黙る / すり替わる** (parakeet と同一)。 | **③staging** | file | #379 |
| `livecap_cli/engines/parakeet_engine.py:260` | NeMo が内部で選ぶ %TEMP% 展開先 (我々からは名前が見えない) | NeMo internal untar → sentencepiece (narrow path) | 非対応 | — 未実測 (nemo-toolkit 未導入。NeMo 内部の展開先は外から観測できないため間接測定になる。) | **黙る**。展開先が非 ASCII だと sentencepiece が読めず二次例外にすり替わる。 | **③staging** | %TEMP% | #379 |
| `livecap_cli/engines/voxtral_engine.py:313` | ローカルモデルディレクトリ (str(model_path)) | transformers → safetensors / torch.load | 対応の見込み | ✅ pass: cjk_kana | — | **②wide-path** | dir | — |
| `livecap_cli/engines/voxtral_engine.py:323` | ローカルモデルディレクトリ (str(model_path)) | transformers → tokenizer / config (mistral-common tekken) | 要実測 (tokenizers は Rust native) | ⏭ skipped: cjk_kana | (skip 理由: processor の optional 依存が未導入 (`uv sync --extra engines-voxtral` が必要):  MistralCommonTokenizer requires the mistral-common library but it was not found in your environment. You can install it with pip: `pip install mistral-common`. Please note that you may need to restart your runtime after installation. ) | **②wide-path** | dir | — |
| `livecap_cli/engines/whispers2t_engine.py:315` | HF repo id (パスではない) + 既定 HF cache ディレクトリ | whisper_s2t → huggingface_hub → CTranslate2 (native) + tokenizers | 要実測 (CTranslate2 は native) | — 未実測 (既定 HF cache 配下のモデルを非 ASCII HF_HOME へ再配置する実装が未了。CTranslate2 は native なので narrow path の可能性があり、real_model tier の別 PR で実測すること。) | — | **②wide-path** | dir | — |
| `livecap_cli/engines/qwen3asr_engine.py:390` | HF repo id + HF_HOME (unicode_safe_download_directory + huggingface_cache 内) | qwen_asr → transformers → HF snapshot + safetensors + tokenizer | 要実測 | — 未実測 (qwen_asr パッケージ未導入 (engines-qwen3asr extra)。HF snapshot はローカルにある。) | — | **②wide-path** | dir | — |
| `livecap_cli/engines/reazonspeech_engine.py:394` | 不正な ONNX + tokens.txt を ASCII / 非 ASCII に置き、エラー署名を比較 | sherpa-onnx (native, narrow path) | 非対応 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | **この行の pass は「sherpa-onnx が安全」を意味しない。** 不正な ONNX は tokens.txt より先に検証されるため、本プローブが到達できるのは ONNX 層までで (ASCII / 非 ASCII のどちらも同じ parse 失敗署名になった)、既知 NG の本体である tokens.txt の SymbolTable 誤読には届かない。そちらは real_model tier で fail_silent を再現している。 | **③staging** | dir | #377 |
| `livecap_cli/engines/reazonspeech_engine.py:395` | encoder / decoder / joiner の .onnx パス (sherpa-onnx 内部で ORT へ渡る) | onnxruntime (native) | 対応 (実測済み) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/engines/voxtral_engine.py:315` | 重みファイルのパス (transformers 内部で torch.load へ渡る) | torch (native) | 対応の見込み。方式①も可 (IO[bytes] を受ける) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/engines/voxtral_engine.py:317` | safetensors 重みファイルのパス | safetensors (Rust native) | 対応の見込み。方式①も可 (load(data: bytes) がある) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/engines/whispers2t_engine.py:317` | tokenizer.json のパス (whispers2t / transformers が共有する層) | tokenizers (Rust native) | 要実測 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/engines/base_engine.py:305` | ダウンロード済みモデルファイル (open(model_path, 'rb')) | CPython builtin open | 対応 (CPython は *W API) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | **黙る**。except Exception: return False で呼び出し側がファイルを削除し ValueError('ダウンロードしたモデルが破損') を raise するため、真因 (権限・エンコーディング等) が消える。 | **②wide-path** | file | — |

### 3.2 ランタイム temp wav

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/engines/parakeet_engine.py:491` | 発話ごとの一時 wav (dir= 指定なし → 素の %TEMP%) | soundfile (書き込み) → ネイティブ ASR (読み込み) | 書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/engines/canary_engine.py:442` | 発話ごとの一時 wav (dir= 指定なし → 素の %TEMP%) | soundfile (書き込み) → ネイティブ ASR (読み込み) | 書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/engines/qwen3asr_engine.py:499` | 発話ごとの一時 wav (dir= 指定なし → 素の %TEMP% (auto-detect 経路のみ)) | soundfile (書き込み) → ネイティブ ASR (読み込み) | 書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/engines/whispers2t_engine.py:441` | 発話ごとの一時 wav (dir=self._tmp_dir → cache_root/whispers2t (唯一 %TEMP% を避けている)) | soundfile (書き込み) → ネイティブ ASR (読み込み) | 書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/engines/voxtral_engine.py:513` | 発話ごとの一時 wav (get_temp_dir() → cache_root/runtime) | soundfile (書き込み) → ネイティブ ASR (読み込み) | 書き込みは対応 (sf_wchar_open) / 読み込み側は engine 依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/engines/voxtral_engine.py:515` | 発話 wav の書き込み先パス | soundfile / libsndfile | 対応 (soundfile.py が sf_wchar_open を使う) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |

### 3.3 ダウンロード / アーカイブ展開

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/resources/model_manager.py:130` | cache_root/downloads 配下のダウンロード先 | CPython urllib | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/resources/model_manager.py:193` | HF_HOME 環境変数経由で huggingface_hub に渡る cache ディレクトリ | huggingface_hub / transformers | 対応の見込み (pure Python) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | — |
| `livecap_cli/engines/reazonspeech_engine.py:334` | cache_dir=str(hf_cache) | huggingface_hub | 対応の見込み (pure Python) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | — |
| `livecap_cli/engines/reazonspeech_engine.py:296` | アーカイブパス + 展開先ディレクトリ (+ メンバ名) | CPython tarfile | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | — |
| `livecap_cli/resources/ffmpeg_manager.py:211` | アーカイブパス + 展開先ディレクトリ (+ メンバ名) | CPython zipfile | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | — |
| `livecap_cli/utils/__init__.py:122` | TEMP / TMP / TMPDIR / tempfile.tempdir を cache_root/downloads へ移設 | プロセス全体 (os.environ + tempfile.tempdir) | **移設先自体が ASCII 保証でない** | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | **黙ってデータを消す**。download スコープが開いている間、プロセス内のあらゆる NamedTemporaryFile が downloads/ に飛ばされ、スコープ退出時の共有 rmtree で削除される (発話 wav を含む)。 | **③staging** | %TEMP% | #375 |
| `livecap_cli/utils/__init__.py:103` | TEMP を cache_root/runtime へ移設 (**デッドコード**) | プロセス全体 (os.environ + tempfile.tempdir) | **移設先自体が ASCII 保証でない** | 🔴 **fail_silent**: cjk_kana, nfd, outside_acp, space_paren | デッドコードのため実害は無いが、ASCII 安全策と誤解される危険がある。 (判定根拠: no_exception_output_differs_from_control) | **③staging** | %TEMP% | #375 |

### 3.4 音声 I/O・ffmpeg

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/audio_sources/file.py:72` | ユーザー指定の入力音声パス (Path オブジェクトをそのまま渡す) | soundfile / libsndfile | 対応の見込み (soundfile.py が sf_wchar_open を使う) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/transcription/file_pipeline.py:241` | pipeline の作業ディレクトリ (**cache_root ではなくシステム %TEMP%**) | CPython tempfile → 後段の ffmpeg / soundfile | CPython 側は対応。後段の消費者に依存 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **③staging** | dir | #375 |
| `livecap_cli/transcription/file_pipeline.py:547` | ユーザー指定の入力ファイルパス | ffmpeg-python → subprocess argv (シェル文字列ではない) | 要実測 (CreateProcessW 経由の list-argv) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/transcription/file_pipeline.py:546` | **ユーザーのファイル名 stem から組み立てた** temp wav の出力先 | ffmpeg-python → subprocess argv | 要実測 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/transcription/file_pipeline.py:560` | ffmpeg 実行ファイルのパス | subprocess (CreateProcessW) | 要実測 | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/transcription/file_pipeline.py:528` | 解決済み ffmpeg / ffprobe パスをプロセス env に流す | pydub / moviepy 系の第三者コンシューマ | 対応 (env は str) | — 未実測 (実際の消費者は pydub / moviepy 系の第三者ライブラリであり、本リポジトリからは観測できない。source-check で ② と判定する。) | — | **②wide-path** | - | — |
| `livecap_cli/transcription/file_pipeline.py:570` | 音声ファイルパス (librosa の内部リーダ経路) | librosa → soundfile / audioread | 対応の見込み。方式①も可 (BinaryIO を受ける) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |

### 3.5 出力・CLI・リソース解決

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/transcription/srt.py:66` | SRT 出力先パス | CPython open(..., encoding='utf-8') | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | file | — |
| `livecap_cli/cli.py:1116` | input_file (positional) と -o/--output。いずれも素の str | argparse → Path() | 対応 (str→Path は無損失) | — 未実測 (argparse は CPython のみを経由し情報を失わない。ここは ③ の境界へパスが流入する入口であって、それ自体が壊れる箇所ではないため runtime 実測の対象外とする。) | — | **②wide-path** | file | — |
| `livecap_cli/cli.py:951` | 非 ASCII パスを stderr へ出力する | コンソール / リダイレクト先のエンコーダ | n/a (エンコーディングの話であってパスの話ではない) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | 落ちない (エスケープされる)。 | **②wide-path** | - | — |
| `livecap_cli/cli.py:1002` | SRT 本文 (認識結果テキスト) と パス文字列を stdout へ出力する | コンソール / リダイレクト先のエンコーダ | n/a (エンコーディングの話) | ⚠️ fail_loud: nfd, outside_acp / ✅ pass: cjk_kana, space_paren | **落ちる**。ただし真因と無関係な UnicodeEncodeError として現れる。 (エラーが問題のパスを名指しする) | **④fail-fast** | - | 別 issue (本調査で新規発見) |
| `livecap_cli/resources/model_manager.py:35` | models_root / cache_root (env var または appdirs 既定) | CPython pathlib → 後段の全境界 | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | #375 |
| `livecap_cli/resources/resource_locator.py:14` | LIVECAP_RESOURCE_ROOT からの同梱リソース解決 | CPython pathlib / importlib.resources | 対応 (CPython) | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **②wide-path** | dir | — |
| `livecap_cli/resources/resource_locator.py:18` | **インストール先ディレクトリ**から導出される探索 root | CPython pathlib / importlib.resources | 対応 (CPython) だが後段の消費者に依存 | — 未実測 (非 ASCII パス配下への第二 install tree が必要 (site-packages を丸ごと複製する)。本 issue のコストに見合わないため未実測。#375 着手時に判断する。) | — | **②wide-path** | dir | — |

### 3.6 非該当

| 呼び出し元 | 渡すパス | 受け側ライブラリ | wide path 対応 | 非 ASCII 実測 | 失敗の可視性 | 採用方式 | 粒度 | 追跡 |
|---|---|---|---|---|---|---|---|---|
| `livecap_cli/engines/parakeet_engine.py:462` | なし (ndarray in / ndarray out) | librosa | n/a | — 未実測 (未実測) | — | **非該当** | - | — |
| `tests/nonascii/paths.py:171` | 非 ASCII ディレクトリの 8.3 短縮名を照会する | kernel32.GetShortPathNameW | n/a | ✅ pass: cjk_kana, nfd, outside_acp, space_paren | — | **非該当** | - | — |

<!-- END:TABLE -->

---

## 4. 未実測の一覧と理由

**「試していない」と「試したら通った」を混同させないため、未実測は必ず理由付きで残す。**

### 4.1 NeMo / sentencepiece (`#379`)

- 対象行: `engine.parakeet.nemo_restore_from` / `engine.canary.nemo_restore_from` / `engine.nemo.untar_temp`
- 理由: `sentencepiece` と `nemo-toolkit` が `engines-nemo` extra 側にあり未導入。
  `uv sync --extra engines-nemo` は nemo-toolkit + lightning + datasets + lhotse で GB 級。
- **`.nemo` ファイル自体はローカルに存在する** (`<models_root>/nvidia--parakeet-tdt-0.6b-v2.nemo` 等)。
  足りないのは toolkit だけなので、下記コマンドで即座に実測できる:

  ```bash
  uv sync --extra engines-nemo
  LIVECAP_NONASCII_REAL_MODELS=1 uv run pytest tests/nonascii -m "nonascii_paths and slow" -q \
      --nonascii-report=benchmark_results/nonascii/<date>/results.json
  ```

- **分類は source-check で既に確定している**ので「未分類ゼロ」は満たされている:
  - `restore_from` の `.nemo` パス → **③ (file 粒度)**
  - NeMo 内部の untar 先 → **③ (`%TEMP%` 移設)**
  - sentencepiece のモデルロード → **原理上は①** (`LoadFromSerializedProto(bytes)` が存在する) だが、
    `restore_from` は自前で untar 先を決めるため **NeMo API 越しには到達不能** → 実質③

### 4.2 Qwen3-ASR

- 理由: `qwen_asr` パッケージが `engines-qwen3asr` extra 側にあり未導入 (HF snapshot はローカルにある)。
- **重要な source-check 結論**: Qwen3ASR は**唯一 `unicode_safe_download_directory()` で包まれた
  engine** だが、同ヘルパは `%TEMP%` を `cache_root` へ移すだけで、その `cache_root` は
  appdirs 既定では**ユーザー名を含む**。したがって**包んでも ASCII 安全にはならない**
  (§5 参照、実測で裏付け済み)。

### 4.3 whispers2t (CTranslate2)

- 理由: 既定 HF cache 配下のモデルを非 ASCII `HF_HOME` へ再配置する実装が未了。
- CTranslate2 は native なので narrow path の可能性があり、real_model tier の別 PR で実測すること。

### 4.4 Voxtral の AutoProcessor

- 理由: optional 依存 `mistral-common` が未導入で skip された。
  `uv sync --extra engines-voxtral` を入れた環境で再測定すること。
- なお `AutoConfig` 側 (`engine.voxtral.from_pretrained`) は実測済みで **pass**。

### 4.5 非 ASCII なインストール先 (`resource_locator.py` の `__file__` 由来 root)

- 理由: 非 ASCII パス配下への**第二 install tree** が必要 (site-packages を丸ごと複製する) で、
  本 issue のコストに見合わない。#375 着手時に判断する。

### 4.6 sherpa-onnx hotwords (`#361`)

- 理由: #361 が未実装のため呼び出し箇所がまだ存在しない。
- **分類は確定済み**: `OfflineRecognizerConfig` に `hotwords_file` はあるが
  **`hotwords_buf` は無い** (sherpa-onnx 1.12.39 で実測) → 方式①不可 → **③ (file 粒度)**。
  #361 実装時に本表の行を runtime 実測へ格上げすること。

### 4.7 8.3 生成状態のボリューム設定

`fsutil 8dot3name query` は管理者権限を要求し、本測定では
「アクセスが拒否されました」となった (§0 参照)。ただし 8.3 を採用しない判断は
この照会に依存しない — §6 の却下理由 (1)(3) が実測で単独に成立している。

---

## 5. 失敗の可視性 — 横断的な欠陥

**記録のみ。修正は各実装 issue の担当。** これらがあるため、非 ASCII が原因であっても
利用者にもログにも真因が届かない。

| 箇所 | 何をするか | 帰結 |
|---|---|---|
| `livecap_cli/engines/nemo_utils.py` `check_nemo_availability` | `except (ImportError, AttributeError)` で `NEMO_AVAILABLE=False` を**プロセス全体にキャッシュ** | 呼び出し側は汎用 `ImportError("NeMo is not installed")` を raise。真因はログにしか残らない |
| `livecap_cli/engines/engine_factory.py` `_get_engine_class` | `raise ValueError("Failed to load engine class…")` | 型もメッセージも差し替わる (cause だけ chain) |
| `livecap_cli/engines/__init__.py` | **裸の `except ImportError: pass`** | engine モジュールの import 失敗が完全に消える |
| `livecap_cli/engines/base_engine.py` `_verify_model_integrity` | `except Exception: return False` | 呼び出し側がファイルを削除し `ValueError("ダウンロードしたモデルが破損")` を raise。権限・エンコーディング等の真因が消える |
| `livecap_cli/engines/model_memory_cache.py` | 健全性チェック無し | ReazonSpeech は**壊れた recognizer を `strong=True` でキャッシュ**しプロセス寿命の間返し続ける (`reazonspeech_engine.py` の `ModelMemoryCache.set`) |

### 5.1 `unicode_safe_*` は ASCII 安全ヘルパではない (実測で確定)

`livecap_cli/utils/__init__.py` の 2 つのヘルパは `%TEMP%` / `TMP` / `TMPDIR` /
`tempfile.tempdir` を `ModelManager.cache_root` 配下へ移設するだけで、その `cache_root` は
appdirs 既定では `%LOCALAPPDATA%\PineLab\LiveCap\Cache` = **ユーザー名を含む**。

**実測**: `utils.unicode_safe_temp_directory` 行は全 variant (`cjk_kana` / `outside_acp` /
`space_paren` / `nfd`) で **`fail_silent`** — ヘルパを通過した後の temp path が
**非 ASCII のまま**で、しかも例外もログも出ない。「unicode-safe」という名前が
ASCII 安全性を約束していると誤読される危険がある。

加えて以下の欠陥がある:

1. **ネスト / 交錯で restore が壊れる** — ロック無し・refcount 無し・ネスト深度カウンタ無し。
   内側の `saved` スナップショットが外側の**既に上書き済み**の値を掴み、外側の restore が
   それを恒久的に書き戻す。
2. **共有ディレクトリの `rmtree`** — `unicode_safe_download_directory` の `_cleanup_directory` が
   **共有**の `cache_root/downloads` を `shutil.rmtree` する。
3. **影響範囲がプロセス全体** — download スコープが開いている間、プロセス内の*あらゆる*
   `NamedTemporaryFile` が `downloads/` に飛ばされ、(2) で削除される。
   **実測で確認済み** (`utils.download_dir_data_loss` プローブ:
   `victim_was_redirected_into_downloads=True` / `victim_survived_scope_exit=False`) —
   仮説ではなく**実在するデータ消失経路**であり、発話ごとの一時 wav がこれに該当する。
   これは非 ASCII とは独立した欠陥なので differential 判定では `pass` になる。
   `tests/nonascii/test_probes.py::test_download_directory_data_loss_is_recorded` が
   観測値に対して直接 assert しており、#375 が直したら反転させること。
4. `unicode_safe_temp_directory` は**デッドコード** — 4 engine が import しているが**呼び出しはゼロ**。

### 5.2 stdout と stderr で挙動が違う (本調査で新規発見)

Windows で stdio が**パイプ**の場合の実測 (Python 3.11 / ACP=932):

| ストリーム | encoding | errors | 非 ASCII を書いたとき |
|---|---|---|---|
| `sys.stdout` | cp932 | `surrogateescape` | **`UnicodeEncodeError` で落ちる** |
| `sys.stderr` | cp932 | `backslashreplace` | エスケープされるだけで落ちない |

したがって:

- `cli.py` の `Transcribing: {...}` (**stderr**) は**安全**。当初 ④fail-fast と見込んでいたが、
  実測により **②** へ変更した。
- `cli.py` の `sys.stdout.write(build_srt(...))` (**stdout**) は**落ちる**
  (実測: `outside_acp` と `nfd` で `fail_loud`)。SRT 本文には任意言語の認識結果が乗るので
  **実害がある**。対処は ASCII staging ではなく**出力ストリームの明示的な UTF-8 化**。
  → **別 issue を起票すること。**

本ハーネスの `report.py` 自身が最初この欠陥を踏んだ (⚠ / 🔴 を stdout に書いて
`UnicodeEncodeError`)。修正は `sys.stdout.reconfigure(encoding="utf-8")` の 1 行で、
`cli.py` にも同じ対処が使える。

### 5.3 ログ自身が爆発しないようにする

engine 側には `logger.info(f"… {model_path}")` 形式のパスログが複数ある。stderr 経由なら
`backslashreplace` で救われるが、`logging.FileHandler` 等でファイルへ出す構成に変えると
落ち得る。#375 は `safe_path_repr(p) = ascii(os.fspath(p))` を提供し、
`livecap_cli/paths/` の全メッセージ・全ログで使うこと。

---

## 6. `ascii_safe_path()` 契約

実装は #375 (モジュール本体) / #379 (NeMo) / #377 (sherpa-onnx) が担当する。
本 issue の成果物は**設計の確定**まで。

### 6.1 使ってはいけない場合

- **方式①**: `*_buf` / `*_bytes` / serialized-proto / file-object overload がある API。
- **方式②**: CPython 経由のみで到達するもの (`open` / `pathlib` / `shutil` / `tarfile` / `json`)。
  §3 の実測どおり `tarfile.extractall` / `zipfile.extractall` / `urlretrieve` /
  `huggingface_hub` はすべて `pass` — **触らないこと**。
- `soundfile` の**書き込みはバグではない** (`sf_wchar_open` を使う。§3 の `lib.soundfile.write` 行が
  全 variant で `pass`)。バグは書いた path を**ネイティブ ASR に渡す側**にある。

### 6.2 API

**新パッケージ `livecap_cli/paths/`** に置く (`utils/__init__.py` には入れない —
`docs/architecture/core-api-spec.md` が `__all__` を安定 API と宣言しており、staging cache /
reaper / lock 機構を寄せ集めモジュールに足すのは筋が悪い)。
構成: `roots.py` / `staging.py` / `workspace.py` / `reaper.py` / `errors.py`。

```python
@contextmanager
def ascii_safe_path(source, *, boundary: str,             # boundary は必須 (エラーメッセージ契約)
                    kind: Literal["auto","file","dir"] = "auto",
                    include=None, mechanisms=None,
                    retention=Retention.PERSISTENT, verify=Verify.CHEAP,
                    allow_nonascii_children=False, recursive=False,
                    force=None, progress=None) -> Iterator[AsciiSafePath]: ...

def stage_ascii_path(...) -> AsciiSafePath: ...   # 非スコープ形。上記はこれ + try/finally
```

- **同期のみ**。非同期が要るなら `ModelManager.download_file_async` と同じく
  `asyncio.to_thread` で包む (2.5 GB のコピー経路を二重に持たない)。
- `AsciiSafePath` は frozen dataclass。`__fspath__` を実装し
  (`os.path.join(handle, "tokens.txt")` が通る)、`.staged` / `.mechanism is IDENTITY` で
  恒等 fast-path を判別できる。
- **API は 1 本** (`kind="auto"`)。file / dir の機構非対称は内部の per-kind ladder で吸収する。
- 入力正規化は `abspath` + `normpath`。**`Path.resolve()` は使わない**
  (junction / symlink / `subst` を辿って ASCII を非 ASCII に戻し得る)。

### 6.3 機構 ladder

各段は `OSError` / `NotImplementedError` / `AttributeError` で次段へ降格し、理由を蓄積する。

| kind | ladder |
|---|---|
| file | `HARDLINK` → `SYMLINK` → `COPY` |
| dir | **`HARDLINK_FARM`** → `JUNCTION` → `SYMLINK` → `COPY` |

- `HARDLINK`: 同一ボリューム必須。事前に `st_dev` 比較 (Windows でも CPython は `st_dev` を
  埋めるので判別可能 — 本測定機で `C:` と `D:` が別値であることを確認済み)。0 バイト・O(1)。
- **`HARDLINK_FARM`**: ディレクトリ内の各子に `os.link` を張った**実ディレクトリ**を作る。
  ReazonSpeech の形 (必要な子は 4 ファイルのみ) にちょうど合い、epic 指定の hardlink 優先順を
  守り、reparse point ではないので §6.4 の realpath 危険を回避し、`include=` で
  int8 / float32 両方を持つモデルディレクトリから 4 ファイルだけを stage できる。
- `JUNCTION`: `_winapi.CreateJunction` (本測定機の 3.11 / 3.13 双方で存在確認済み。private なので
  `hasattr` ガード)。**管理者不要**なので dir では symlink より上位。
  **`mklink /J` へのシェルアウトは禁止** (cmd.exe 起動 + 非 ASCII 引数のクォートは、
  まさに直そうとしているバグ)。
- `SYMLINK`: Windows では開発者モードが要る (`winerror 1314`) ので may-fail 段。
- `COPY`: 事前に `disk_usage().free >= size * 1.05` を確認し、失敗理由に両方の数値を出す。

### 6.4 realpath 危険と `mechanisms=` の存在理由

`GetFinalPathNameByHandle` / `std::filesystem::canonical` を内部で呼ぶライブラリは、
junction / symlink 越しに**元の非 ASCII パスを復元して**バグを再発させ得る。
**`COPY` だけが realpath の ASCII を保証する。**

既定 ladder は速度優先でこの危険を受け入れる。canonicalize すると観測された境界は
`mechanisms=(Mechanism.COPY,)` を**観測記録付きで** pin すること。

**#379 / #377 向けの見分け方**: staging は成功 (§6.7 の不変条件も成立) しているのに
ライブラリが依然 encode / `IndexError` / `FileNotFoundError` を出す → `COPY` に固定して
直れば、その境界は canonicalize している。

### 6.5 staging root の選定

環境変数 `LIVECAP_CORE_ASCII_STAGING_DIR` (`ModelManager` の `LIVECAP_CORE_*` 命名に一致)。
付随: `_MAX_BYTES` (既定 8 GiB、COPY エントリのみ) / `_TTL_HOURS` (既定 336 = 14 日) /
`_FORCE` (恒等 fast-path を無効化し Linux CI で ladder を通す)。

**明示指定が述語を満たさない場合は即 `AsciiStagingUnavailableError`** —
運用者の明示指示を黙って無視するのは epic が禁じる silent degradation そのもの。

候補 ladder (先勝ち)。**hardlink 段を生かすため「ソースと同一ボリューム」候補を最上位に昇格**:

| # | 候補 | ASCII 保証 |
|---|---|---|
| 0 | `$LIVECAP_CORE_ASCII_STAGING_DIR` | 運用者責任 (不正なら fail loud) |
| 1 | `<ソースのボリューム>\LiveCapStaging` | ○ (ドライブレター + ASCII リテラル) |
| 2 | `%ProgramData%\LiveCap\staging\<8hex of sha256(username)>` | ○ (**ユーザー名そのものは絶対に使わない**) |
| 3 | `%SystemDrive%\LiveCap\staging` | ○ |
| 4 | `%PUBLIC%\LiveCap\staging` | ○ |
| 5 | `cache_root/"ascii-staging"` | **×** (述語を通った場合のみ) |
| 6 | `gettempdir()/"livecap-ascii"` | **×** (同上) |

述語: `isascii()` → `len <= 120` (MAX_PATH 余裕。これにより `\\?\` を一切使わずに済む) →
`mkdir` → **書き込みプローブ** (Windows の ACL 検査は当てにならない) → `st_dev` 記録。
全候補が落ちたら `AsciiStagingUnavailableError` = **方式④**。

**葉の名前も ASCII でなければならない**: ファイルは `<digest>/<ascii_stem><suffix>`
(**suffix は逐語保存** — NeMo は `.nemo` を要求)。ディレクトリの子は**バイト単位で保存**する
(消費側が既知の名前を base に join するため)。したがって**子名に非 ASCII があれば
リネームせず `NonAsciiChildError`**。ReazonSpeech の 4 ファイル名は ASCII 済みで、
壊れているのは親ディレクトリだと実測で確認している。

### 6.6 生存期間

> **G-LIFETIME**: `with` を抜けても、そのプロセスが生きている限り `handle.path` は
> 無効化されない。削除は (a) 明示 `purge_ascii_staging()`、(b) 起動時 reaper、
> (c) 予算 eviction のみで、(b)(c) はプロセス内 refcount またはプロセス間 lease が
> 生きているエントリに触れない。

これが「sherpa-onnx が `with` を抜けても ONNX を mmap し続ける」への直接の答え。
staging 領域は**スクラッチではなくキャッシュ**である。

retention は `PERSISTENT` (既定) / `PROCESS` / `SCOPED` / `SCOPED_EAGER`。
**PR 作者向けルール: 境界が返すオブジェクトがファイルを保持する (ORT session / mmap /
遅延読み archive) なら `PERSISTENT` か `PROCESS`。`SCOPED_EAGER` は禁止。**

- sherpa-onnx: recognizer が `strong=True` でキャッシュされる → `PERSISTENT`
- NeMo `restore_from`: `.nemo` は呼び出し中に読み切って閉じるので `SCOPED` が*正しい*が、
  起動ごとの 2.5 GB 再コピーを避けるため `PERSISTENT` を*推奨*

**content addressing**:
`sha256(v1 ∥ normcase(abspath(source)) ∥ kind ∥ sorted[(name,size,mtime_ns)])[:16]`。
**機構は意図的に除外** (COPY で作ったエントリを、後に HARDLINK が使えるようになっても再利用する)。
`(size, mtime_ns)` で再ダウンロードを検出する。

**atomic publish**: `.incoming/<pid>-<uuid4>/` に作る → `os.replace` で `<digest>/` へ →
`FileExistsError` / `WinError 183` は**他プロセスがレースに勝った成功パス**
(相手の `.meta.json` を検証して再利用) → 例外時は `finally` で incoming を rmtree。
**部分コピーが公開名で到達可能になることは決してない** — これが新種の silent corruption を防ぐ。
`.meta.json` は**最後に**書き、その存在が「公開済み」フラグとなる。

**孤児回収** (`reaper.py`、root 初回使用時に 1 回の `scandir`): 6h 超の `.incoming/*`、
`.meta.json` 無しの 6h 超エントリ、TTL 超過かつ未使用のエントリ、COPY 総量が予算超過なら
LRU eviction。**PID 生存判定は使わない** (PID 再利用で不健全)。reaper は best-effort。

### 6.7 並行ロード契約

`threading.RLock` (ネストと `progress` コールバックからの再入でデッドロックしないため
`Lock` ではない) + digest ごとの `_Entry`。**大量 I/O は global lock の外**で行う
(2.5 GB コピー中に全 engine を直列化しない): global lock 下で entry を find-or-create して
per-digest lock を取り、global lock を離してから作業する。

**保証**:

- **G1** スレッド安全
- **G2** 再入・ネスト可 (内側の退出が外側の lease を無効化しない)
- **G3** プロセス間冪等 — content addressing + atomic publish で N プロセスが 1 エントリに収束。
  最悪でも二重作業で、壊れた結果には決してならない → **プロセス間ロックは正しさには不要**
- **G4** reaper は使用中エントリを消さない
- **G5** 失敗の原子性 — 失敗した staging は公開物も leak した refcount も残さない

**明示的な非保証** (モジュール docstring に必ず書く): fork 安全でない (子は
`reset_ascii_staging_state()` を呼ぶこと) / staging 中のソース外部変更は保護しない /
無関係な境界を直列化しない (グローバルなモデルロードロックではない) / 消費側ライブラリの
スレッド安全性については何も言わない / ブロッキング (イベントループスレッドから呼ばない) /
`Verify.CHEAP` は内容破損を検出しない。

**cleanup vs in-use レース (プロセス間)**: acquire 時に `<digest>/.inuse/<pid>-<uuid>.lock` を
**開いたまま保持**する。Windows では保持ハンドルが削除を阻むので reaper の `os.remove` が
`PermissionError` → 使用中と判定してエントリごとスキップ (OS に仕事をさせる)。
POSIX では追加で `flock(LOCK_EX|LOCK_NB)` を試し `BlockingIOError` で判定する。

**真にプロセスグローバルな部分 = `%TEMP%`。** 置換の契約 (§5.1 の欠陥に対応):
モジュール level `RLock` + **深度カウンタ** + **深度 0 でのみ**スナップショット取得、
**0→1 でのみ**環境変更・**1→0 でのみ**復元。異なるターゲットでのネストは
`TempEnvironmentConflictError`、同一ターゲットなら深度を上げるだけ。各スコープは
`mkdtemp(dir=base)` で**固有サブディレクトリ**を得て、cleanup はそれだけを消す
(**共有 rmtree は削除**)。

既約な非保証も明記すること: `os.environ` / `tempfile.tempdir` はプロセスグローバルなので、
移設ウィンドウ中に読む並行コードは移設後の値を見る。ロックは変更を*一貫*させられるが
*スレッドスコープ*にはできない → **ウィンドウは最小に**
(#379 では `_load_model_from_path` 全体ではなく `restore_from` 呼び出しだけを包む)。

### 6.8 fail-loud

例外階層 (`livecap_cli/translation/exceptions.py` の形に倣う):
`AsciiPathError(RuntimeError)` ← `AsciiStagingUnavailableError` /
`AsciiStagingFailedError` (`attempts: tuple[(Mechanism, str), ...]` を持つ) /
`AsciiStagingVerificationError` / `NonAsciiChildError` / `TempEnvironmentConflictError`。
**`OSError` ではなく `RuntimeError` 派生** (呼び出し側が `except OSError` で握り潰すため)。
安定 `code: str` を i18n フックにし、メッセージ本文は英語。

メッセージ契約 (この順で必須): **境界名** (だから `boundary` は必須キーワード引数) →
**問題のパス** (`safe_path_repr()` 経由) → **何を試して各々なぜ失敗したか** (`errno` / `winerror` 付き)
→ **env var を名指しした実行可能な対処**。

**`logger.warning` を出して元のパスを返すのは禁止。`strict=False` も
`LIVECAP_..._STRICT=0` も作らない。**

成功時の事後条件 (`assert` ではなく明示 `if not …: raise` — `-O` で消えるため):

- **P1** `str(handle.path).isascii()` (`IDENTITY` を含む全機構で必ず検査)
- **P2** 存在し kind が一致する
- **P3 バイト同一**: hardlink 系は `st_ino` + `st_dev` 一致、junction / symlink は
  `os.path.samefile`、COPY は全メンバのサイズ一致 (`Verify.STRONG` では SHA-256 を**再計算**。
  保存された digest は現在のバイトを何も証明しない)
- **P4 silent fallback 禁止**: `handle.staged is True or is_ascii_safe(handle.source)`。
  staging が必要だったのに `IDENTITY` なら **hard bug** → `AsciiStagingVerificationError`
- **P5** dir の全子名が ASCII

### 6.9 silent corruption との接続 (#379 / #377 が守るべき 3 点)

1. **staging は境界呼び出しの前に行う** — 失敗すればオブジェクト構築前 =
   `ModelMemoryCache.set` 前に raise する。
2. **`AsciiPathError` を飲まない** — ReazonSpeech の裸 `except Exception` の手前に
   `except AsciiPathError: raise` を置くか catch を狭める。
3. **cache key は staging 後のパスではなく論理ソースから計算する** —
   現行は `model_path.name` を使っており、staging 後の digest ディレクトリになると
   2 枠しかない strong cache に機構依存の重複エントリが出る。

### 6.10 `%TEMP%` 移設は兄弟 API

```python
@contextmanager
def ascii_safe_temp_environment(*, boundary, purpose="runtime") -> Iterator[Path]: ...
    # TEMP/TMP/TMPDIR/tempfile.tempdir を ASCII 保証ディレクトリへ。
    # refcount + RLock + 深度カウンタ。毎回「新しい固有サブディレクトリ」を yield し、
    # それだけを消す。

@contextmanager
def ascii_safe_workspace(*, boundary, purpose="runtime") -> Iterator[Path]: ...
    # 「我々が作るファイル」用の ASCII 保証の空ディレクトリ。env を触らないので
    # 自明にスレッド安全・ネスト可。
```

発話ごとの一時 wav の正解は **`ascii_safe_workspace`** — 非 ASCII `%TEMP%` に作ってから
staging するのではなく、**最初から ASCII 空間に ASCII 名で作る**。
**ここで `ascii_safe_temp_environment` を使ってはいけない** (発話ごとにプロセスグローバル
状態を書き換えるのは現行バグの縮小再生産)。

#379 の合成は**両方必要** (`.nemo` のパスと NeMo 内部の展開先は別の副境界):

```python
with ascii_safe_path(model_path, boundary="parakeet.nemo.restore_from", kind="file") as safe:
    with ascii_safe_temp_environment(boundary="parakeet.nemo.restore_from.untar"):
        model = nemo_asr.models.ASRModel.restore_from(restore_path=str(safe.path), ...)
```

### 6.11 旧ヘルパの処遇

- **`unicode_safe_temp_directory` → deprecate してから削除 (修理しない)**。
  デッドコード (import 4 箇所・呼び出し 0) であり、そもそも ASCII 保証がない (§5.1、実測で確定)。
  「修理」とは `ascii_safe_temp_environment` に書き直すことなので、ロック実装を 2 つ
  保守する意味がない。`utils.__all__` にあり `core-api-spec.md` が 1 マイナー以上の
  deprecation window を約束しているので、#375 で (1) 死んだ import 4 本を即削除、
  (2) 名前は `DeprecationWarning` 付きの薄い shim として残す、(3) 次マイナーで削除。
- **`unicode_safe_download_directory` → 転送で修理し、名前とシグネチャは維持**
  (生きた呼び出しが 5 箇所あり、維持すれば #375 はそれらに触らずに済む)。
  **docstring と PR に明記すべき意味変更が 3 点**:
  1. yield されるのは共有ディレクトリではなく**呼び出しごとの固有サブディレクトリ**
  2. cleanup はそのサブディレクトリだけ (**共有 `rmtree` は削除** = §5.1 のデータ消失バグ)
  3. ASCII root が見つからない環境では **`AsciiStagingUnavailableError` を raise する**
     (従来「動いていた」挙動こそが epic の狙う silent failure。対処は env var でメッセージに出す)

### 6.12 却下した代替案

#### 8.3 短縮名 (`GetShortPathName`) — 却下 (独立した 3 つの理由)

1. **実測で不発**。`ユーザー` は 8.3 に収まるので別名が生成されない。
   §3 の `win32.short_path_name` 行の観測では、`GetShortPathNameW` は
   `short_name_returned=True` / `short_name_differs_from_input=True` を返すが
   **`short_name_is_ascii=False`** — 長いセグメント (`livecap-nonascii-…` など) だけが
   短縮され、`ユーザー` はそのまま残るため、**結果は依然として非 ASCII**。
2. 現代の Windows では非システムボリュームで 8.3 生成が既定無効であり、無効中に作られた
   ファイルには恒久的に別名が無い。
3. **黙って失敗する**。別名が無いとき `GetShortPathNameW` はエラーも signal も出さず
   長い名前を返す。epic が消そうとしている silent degradation の上に修正を建てることになる。

#### `\\?\` 拡張長プレフィックス — 却下 (別の問題を解く)

これは ***W* API の機能**である。ライブラリが `CreateFileA` / `fopen` / `std::ifstream` に
narrow path を渡す時点で、ANSI→UTF-16 変換は A-shim / CRT 内で**カーネルがプレフィックスを
見る前に**起きており、ACP (cp932) で表現できない文字は既に best-fit または `?` に潰れている。
情報が失われた後の文字列に前置しても何も変わらない。むしろパスを文字列操作するライブラリを
**壊す**。残る用途は「使わなくて済むように staging root を 120 文字以下に保つ」ことだけで、
`is_ascii_safe()` は `\\?\` 付き入力を `ValueError` で拒否する。

#### コードページ変更 — 却下

- `chcp 65001` は**コンソール**のコードページだけで、ネイティブライブラリ内の `*A` ファイル API が
  使う ACP には無関係。
- `PYTHONUTF8=1` も無効 — 本測定機の `getfilesystemencoding()` は既に utf-8 で、
  **CPython は既に `*W` を使っている** (§3 の CPython 経由行がすべて `pass` であることが裏付け)。
  壊れているのは第三者ネイティブコードの中で、UTF-8 モードはそこに届かない。
- ACP=65001 のベータ設定は技術的には本物の修正だが、マシン全体・管理者権限・再起動が要り、
  Microsoft 自身が他アプリを壊すと警告している。
- アプリマニフェストの `<activeCodePage>UTF-8` (Win10 1903+) は**将来の狭い機会** —
  本リポジトリは frozen build を作るので PyInstaller 製 exe には載せ得るが、任意の
  `python.exe` 下でも動く以上 staging は必須のまま。別の探索 issue にするが、
  **staging 作業を遅らせたり薄めたりしてはならない**。

#### 「ASCII パスに再インストールしてもらう」 — 解決策としては却下、回避策としては維持

1. **壊れているパスを取り違えている** — 多くの場合インストール先ではなくモデルキャッシュ
   (`appdirs.user_cache_dir("LiveCap","PineLab")` = **ユーザー名由来**) であり、
   再インストールでは動かない。
2. **ユーザーには直せない** — Windows プロファイルディレクトリのリネームは危険・非サポートの
   アカウント移行。日本語 / 中国語 / キリル文字の名前のユーザーに Windows アカウントを
   作り直せとは言えない。
3. **面が足りない** — モデルとキャッシュを ASCII にしても `%TEMP%` はユーザープロファイル
   配下のままなので、発話 wav と NeMo 内部展開は壊れ続ける。

**推奨手動回避策としては維持し、エラーメッセージに載せる**:
`LIVECAP_CORE_MODELS_DIR` + `LIVECAP_CORE_CACHE_DIR`。

### 6.13 #375 着手前に決める論点

1. 既定 retention `PERSISTENT` の予算既定値 (COPY のみ 8 GiB)
2. `unicode_safe_temp_directory` の deprecate→削除 (推奨) vs `__all__` 安定性を押し切って即削除
3. `unicode_safe_download_directory` が ASCII root 無し環境で raise するようになる件の是認
   (epic の要求からは是認が筋)
4. #377 が ReazonSpeech に **post-load ヘルスチェック** (1 トークン decode) を
   `ModelMemoryCache.set(..., strong=True)` の前に足すか。staging で既知の破損経路は
   到達不能になるが、次の 1 件への多層防御。別 issue 候補
5. **リソース root の setter API 不在** — `resources/__init__.py` が引数無しで `ModelManager()` を
   作るので、ホストアプリは env 変更 + `reset_resource_managers()` しか手が無い。
   「`LIVECAP_CORE_MODELS_DIR` を ASCII に向ける」が最も安い実世界の対処である以上、
   `configure_resource_managers(...)` は **#375 のスコープ**とする

---

## 7. 検証ハーネス

実装とその設計判断は [`tests/nonascii/README.md`](../../tests/nonascii/README.md) を参照。
要点だけ:

- **全プローブは子プロセスで走る** — ネイティブ `abort()` への耐性、および
  「測定対象そのものがプロセス全体の env 書き換えである」ため。
  ハーネスは `unicode_safe_*` を呼ばず、親の `os.environ` も触らない
  (`test_parent_process_state_is_untouched` が固定)。
- **differential 方式** — control との比較で判定するので golden 値を持たない。
- **仕込み欠陥による自己検証** — `test_harness_selftest.py` が、握り潰し / 遅延失敗 /
  メッセージすり替え / ネイティブ abort / timeout の 5 パターンを正しく分類できることを
  CI 時点で assert する。**`selftest.silent_truncation` を `fail_silent` と分類できない
  ハーネスは証拠として使えない。**
- **実コードに対する positive control** — ReazonSpeech の既知 NG
  (ロード成功 → decode で `IndexError`) を実モデルで再現し、`fail_silent` と分類できることを
  確認済み。**#377 が必要とする回帰ゲートそのもの**である。

### CI 常設化の判断

**cheap tier のみ既定スイートに載せる** (`core-tests.yml` / `core-tests-windows.yml` に
相乗りし、workflow 編集はゼロ)。新規マーカーは `nonascii_paths` の 1 個だけで、
重い tier は既存の `slow` / `network` ゲートを再利用する。

`nonascii_paths` を `addopts` の deny-list に**入れない**理由: cheap tier は速く決定的で、
まさに #375 / #379 / #377 が必要とする回帰ゲートだから。opt-in にすると走らなくなり形骸化する。

caveat:

- narrow / wide path は Win32 の概念なので効くのは Windows job。ただし Linux job も残す価値がある —
  将来の `ascii_safe_path()` が **POSIX では no-op** であること、プローブ自体が可搬であることを
  固定できる。
- **CI runner の ACP は cp1252、本測定機は cp932** → 各ホストが検出できる失敗の部分集合が異なる。
  どちらも単独では権威ではないので、**どの測定がどのホスト由来か**を §0 に必ず記録する。
  `outside_acp` variant は両方の ACP の外側にあり、この差を吸収するために設計されている。
- 両 workflow に `paths-ignore: ['docs/**','*.md']` があるため、doc だけの再生成では再実行されない。
  鮮度保証は CI スケジュールではなく `test_registry.py` が担う。
- 実モデル / NeMo tier は **CI に載せない** (hosted runner にモデルが無い。self-hosted Windows
  runner は `integration-tests.yml` の管轄)。

---

## 8. 再現手順

```bash
# cheap tier (既定スイートに含まれる)
uv run pytest tests/nonascii -m nonascii_paths -q

# real_model tier を含めて実測し、証拠 JSON を書き出す
LIVECAP_NONASCII_REAL_MODELS=1 uv run pytest tests/nonascii -m "nonascii_paths" -q \
    --nonascii-report=benchmark_results/nonascii/<date>/results.json

# 本 doc の §0 / §3 を再生成する
uv run python -m tests.nonascii.report --json benchmark_results/nonascii/<date>/results.json \
    --inject docs/research/nonascii-path-boundary-inventory-2026-08.md
```

生データ: [`benchmark_results/nonascii/2026-08-20/results.json`](../../benchmark_results/nonascii/2026-08-20/results.json)

---

## 9. 変更履歴

| 日付 | commit | 環境 | 内容 |
|---|---|---|---|
| 2026-08-20 | 初版 | Windows 11 26200 / AMD64 / Python 3.11.13 / ACP=932 | 44 行を棚卸し。cheap tier 全項目 + real_model tier (ReazonSpeech / Voxtral) を実測。NeMo / Qwen3ASR / whispers2t は未実測 (理由は §4)。**新規発見**: stdout と stderr のエラーハンドラ差 (§5.2)、`unicode_safe_temp_directory` が ASCII 保証でないことの実証 (§5.1) |

---

## 関連

- Epic [#380](https://github.com/Mega-Gorilla/livecap-cli/issues/380) — 非 ASCII パス耐性
- [#375](https://github.com/Mega-Gorilla/livecap-cli/issues/375) — 供給側 API (`ascii_safe_path()` の実装置き場)
- [#379](https://github.com/Mega-Gorilla/livecap-cli/issues/379) — NeMo / SentencePiece への適用
- [#377](https://github.com/Mega-Gorilla/livecap-cli/issues/377) — sherpa-onnx への適用
- [#361](https://github.com/Mega-Gorilla/livecap-cli/issues/361) — sherpa-onnx hotwords (同じ narrow path を踏む)
- [新規 ASR engine 実装ガイド](../contributor/adding-an-engine.md) — §10 にパス境界チェックリスト
