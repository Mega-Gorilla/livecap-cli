## 結論

**一時 wav の 5 consumer すべてが実測で確定し、1 つも staging を追加しないまま [#413] の主題が閉じました。**

当初 ([#375] PR 4) の「5 consumer を `ascii_safe_workspace()` へ移す」計画は、実測が **5/5 で ②wide-path** を示したため実装しません。

## 前提が実測で覆りました

[#413] は「NeMo と依存が競合して同居できないかもしれない。**分かってから**予防的 staging の採否を判断する」と書いていました。**競合しません。**

```
uv sync ... --extra engines-qwen3asr --dry-run
-> Would install 25 packages   (削除もダウングレードも無し)
```

`uv.lock` は universal lock なので解決は既に済んでおり、torch / transformers / NeMo は一切動きません。さらに実際に導入して `import nemo, torch, transformers, qwen_asr` が**同居する runtime で通る**ことも確認しました (dry-run が保証するのは導入可能性までなので、ここは別に確かめる必要がありました)。

したがって [#413] が想定した**隔離環境も証拠の集約基盤も要りませんでした** — 他の全 tier と同じ 1 セッションで測れます。これは [#413] の「集約基盤は作らない」決定 (2026-08-29) と整合します。

## 実装で越えた 3 つの障害

### 1. 境界は auto-detect 経路にしかない

一時 wav を書くのは `_transcribe_via_wrapper_fallback()` だけで、そこへ入るのは `_asr_language is None` のときに限られます。**言語を指定すると `_transcribe_with_scores()` へ行き、境界を迂回します。**

そのため probe は**他の 4 engine とは逆に、言語を渡しません**。

```
変異: _Case の kwargs に {"language": "en"} を入れる
  -> 「一時 wav が variant root 配下に書かれなかった」 -> error_harness
```

**この probe が本当に境界を通っている証拠**です。迂回したまま緑にはなりません。

### 2. 重みが models root に無く、**場所を推測すると契約を破る**

models root にあるのは `model=Qwen/Qwen3-ASR-0.6B` と書かれた **38 バイトの marker** だけで、実体 (1.8 GB) は `huggingface_hub` が **import 時に確定した hub cache** (`huggingface_hub.constants.HF_HUB_CACHE`) にあります。

当初は `manager.huggingface_cache()` が指す `<cache_root>/huggingface` を見ていましたが、**そこは使われません**。`ModelManager.huggingface_cache()` は実行時に `HF_HOME` を書き換えるものの、`huggingface_hub` の cache 定数は import 時に確定するためです (production 側の食い違いは [#428] が追跡)。

```
CI:   管理 cache が空 -> precondition が skip -> ゲートが赤
ローカル: 既定 cache から **1.8 GB をダウンロードしていた**
          -> real_model tier の「ネットワークを使わない」契約に違反
```

**場所を当てにいくのをやめました。** worker を `HF_HUB_CACHE` = 実効 hub cache / **`HF_HUB_OFFLINE=1`** で起動します — ネットワークへ出たら落ちます。

**env は worker の起動前に決めます** (`test_probes._real_model_env()`)。`huggingface_hub` は**どちらの値も import 時に確定する**ので、probe の中で `os.environ` を書き換えても間に合いません — これは `_isolation_env()` が以前から明文化している制約です。

効いたことは probe が `huggingface_hub.constants` の **`HF_HUB_CACHE` と `HF_HUB_OFFLINE` の両方**で確かめます。**cache path だけでは足りません** — 親 env から継承した path が期待値と一致すると、path 検査は通るのに offline は効きません。

```
[先行 import あり]     cache_matches=True  constant_offline=False  env_offline='1'
[起動前に env で渡す]  cache_matches=True  constant_offline=True
```

一時 wav は `dir=` 指定なしの `NamedTemporaryFile` で**素の `%TEMP%`** に書かれるので、**測定対象は変わりません**。

```
変異: 実効 HF cache からモデルを退避 -> skip   (guard が正しい場所を見ている)
変異: 管理 cache の複製を撤去         -> 依然 pass (あれは無関係だった)
```

### 3. source 判定が dir を要求する

`_real_model_is_usable()` は `path.is_dir()` を要求しますが marker はファイルです。判定を probe 側の `qwen3asr_snapshot_dir()` へ委譲しました (`sherpa.from_transducer.real` と同じ前例 — テスト側にファイル名を書くと二重管理になる)。

**marker だけでは重みの存在を保証しません。** marker の存在だけで「使える」と答えると、real_model tier の「ネットワークを使わない」契約を破ってダウンロードが走ります。

```
変異: source を存在しない marker にする -> skip (CI の PASSED 要求が落ちる)
変異: snapshot だけ無くす               -> skip (marker はあるのに使えないと判定)
```

## 実測

clean tree (`363fc69`) から **cheap / real_model / heavy / gpu を 1 セッション**で生成しました (`benchmark_results/nonascii/2026-09-01/results.json`)。

```
51 passed, 2 skipped in 746.89s
116 レコード / 36 probe  (git_commit=363fc69 / git_dirty=false)
engine.qwen3asr.utterance_wav   cjk_kana: pass   outside_acp: pass
```

棚卸し表: 実測で確定 33 → **34 行** / ②wide-path 29 → **30 行** / ③staging 6 → **5 行**。

証拠 JSON は **既存 34 行すべてを含む**ことを独立に検証しています (14 項目 OK)。PR B の 118 レコードとの差も説明できます — `tempfile.named_temporary_wav` の 4 レコードが消え、`asr.utterance_wav.qwen3asr` の 2 レコードが入りました。

**PR B の `2026-08-31/results.json` は生成時のまま残しています。** `<date>` は測定日という規約に従い、probe を変えたら測り直して新しい日付へ出します (照合は最新 1 件だけを読むので、古い方は履歴です)。

## `tempfile.named_temporary_wav` を削除しました

5 consumer すべてが本物の probe を持ったので、producer-only の代役 probe は役目を終えました。producer 境界は `soundfile.write.path` / `soundfile.read.path` が測っています。

[#413] の受け入れ条件「**`tempfile.named_temporary_wav` probe の帰属を決める**」の解です。

## CI の穴を 1 つ塞ぎました

`integration-tests.yml` の `paths` に **`tests/nonascii/**` を追加**しました。

**非 ASCII の real-model / gpu ゲートは本 workflow の中にあるのに、`tests/nonascii/` を変更しても起動しませんでした。** PR B ([#426]) で実際に Integration Tests が走らず発覚しています。ゲートを持つ workflow が、そのゲートの対象を変更しても起動しないのは穴です。

あわせて GPU job に `--extra engines-qwen3asr` と `warm('qwen3asr', ...)` を追加しました。**warm の目的は HF hub cache を埋めること**です — 埋めないと probe が実測時にダウンロードへ落ちます。

## 検証

```
tests/nonascii/test_registry.py            440 passed
tests/nonascii                             544 passed
証拠 re-measure (全 tier)                  51 passed, 2 skipped in 746.89s
全体 (tests/integration/sed 除く)         2894 passed, 54 skipped
```

`tests/integration/sed/test_inference_smoke.py` は全体実行時のみ落ちる**既存問題** (torchvision の二重登録) で本 PR と無関係です。

## 残った 2 件 (本 PR のスコープ外)

`whispers2t.load_model` / `qwen3asr.from_pretrained` は `_REAL_MODEL_SOURCES` に source 定義が無く skip します。どちらも `verified_method=None` の [#387] 追跡行なのでゲートには影響しません。

## #413 のクローズ

本 PR のマージで [#413] を close できます。残る非 ASCII の未確定行は [#387] (4 行) と [#425] (Jiterator 1 行) が追跡します。

🤖 Generated with [Claude Code](https://claude.com/claude-code)

