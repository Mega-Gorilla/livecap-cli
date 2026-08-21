# 非 ASCII パス境界の検証ハーネス (Issue #378 / epic #380)

ネイティブ / 第三者ライブラリへ **filesystem パスを渡している箇所**が、
非 ASCII パスでどう振る舞うかを実測するハーネス。

棚卸し表 (docs) はこのハーネスの出力から**自動生成**される。表を手で書き換えないこと。

- 棚卸し表: [`docs/research/nonascii-path-boundary-inventory-2026-08.md`](../../docs/research/nonascii-path-boundary-inventory-2026-08.md)
- 実測証拠: [`benchmark_results/nonascii/`](../../benchmark_results/nonascii/)

## なぜこの形なのか

### 1. すべてのプローブは**子プロセス**で走る

- sherpa-onnx / NeMo のネイティブコードは `abort()` し得る。親で走らせると全証拠が消える。
- **測定対象そのものがプロセス全体の env 書き換え**である (`TEMP` / `TMP` / `TMPDIR` /
  `tempfile.tempdir` / `HF_HOME`)。親の `os.environ` を触ると、まさに調査対象である
  `livecap_cli/utils/__init__.py` の「ロック無し・refcount 無し」欠陥をハーネス内で
  再現してしまう。**ハーネスは `unicode_safe_*` を呼ばず、`subprocess.run(env=...)` で渡す。**
- 汚染された global state は回復不能 — `nemo_utils` の `NEMO_AVAILABLE` キャッシュ、
  `ModelMemoryCache` の strong ref、ONNX mmap のファイルロック。
- stdio エンコーディング (cp932 パイプ) を制御できるのは子プロセスだけ。

`test_harness_selftest.py::test_parent_process_state_is_untouched` がこの不変条件を固定する。

### 2. **differential 方式** — control との比較で判定する

すべてのプローブは同じ操作を 2 回走らせる: ASCII の control root で 1 回、
非 ASCII variant で 1 回。verdict は**その比較**から導出する。

golden 値を持たないのでモデル / ライブラリ更新に耐え、何より
**`fail_silent` を機械的に検出できる唯一の方法**である。

> **プローブ実装の規約**: 観測 dict に**パスそのものを含めてはならない**。
> control と variant の観測を等値比較するため、パスを含めると常に差分が出て
> 全行が `fail_silent` になる。「読めたバイト数」「テキスト内容」「shape」など、
> パスに依存しない事実だけを返すこと。

### 3. verdict の語彙

| verdict | 意味 |
|---|---|
| `pass` | control と観測的に等価 |
| `fail_loud` | 失敗するが**問題のパスを名指しする** (診断可能)。ネイティブ `abort()` による非ゼロ終了・timeout もここ |
| `fail_silent` | **利用者に何が起きたか分からない失敗**。epic #380 の中核関心事 |
| `skipped` | 依存未導入 / FS が variant を拒否 / tier gate off。理由を必ず記録 |
| `error_harness` | **control が失敗した** = プローブのバグ。バグの証拠として数えない |

`fail_silent` の判定根拠 (`silent_criteria_hit`):

1. `no_exception_output_differs_from_control` — 例外なしで観測が control と違う
2. `deferred_failure_at_later_stage` — control が成功した地点より後段で落ちた
3. `mangled_exception:...` — 真因が既知の汎用メッセージにすり替わっている
4. `exit_zero_but_no_result` / `exception_does_not_name_path`

## variant 語彙

各 variant は「日本語を混ぜたもの」ではなく、**別々の失敗機構**を切り分ける。
ある行が `cjk_kana` を通るのに `outside_acp` で落ちる場合と、`space_paren` で
落ちる場合とでは、必要な修正が根本的に異なる。

| id | segment | 切り分ける機構 |
|---|---|---|
| `control` | `ascii_control` | 差分の基準 |
| `cjk_kana` | `ユーザー` | 実世界ケース。cp932 の**内側** / cp1252 の**外側** |
| `outside_acp` | `한국어Ω` | cp932 と cp1252 の**両方の外側**。JP 開発機でも en-US CI でも「ACP で表現不能」を強制 |
| `space_paren` | `test folder (1)` | argv quoting の family。**ASCII staging では直らない**バグを捕まえる。**ASCII-only** — 非 ASCII を混ぜると quoting と encoding を切り分けられない |
| `nfd` | NFD 分解形 | 正規化を仮定するライブラリの検出 |
| `emoji_astral` | astral 面 | UTF-16 サロゲートペア (既定 off) |
| `long_mixed` | 長大パス | `MAX_PATH` との相互作用 (既定 off) |

FS が variant を受理しない場合 (macOS APFS の NFD 正規化など) は
`skipped` + 理由として記録される。「非 ASCII が通った」と「非 ASCII を試していない」が
混同されないようにするため。

## 実行方法

```bash
# cheap tier — 合成アーティファクトのみ。モデルもネットワークも不要 (既定スイートに含まれる)
uv run pytest tests/nonascii -m nonascii_paths -q

# ハーネス自身の検証 (仕込み欠陥を正しく分類できるか)
uv run pytest tests/nonascii/test_harness_selftest.py -q

# 受け入れ条件 (未分類ゼロ / silent-failure ゼロ / callsite 生存)
uv run pytest tests/nonascii/test_registry.py -q

# real_model tier — ローカルの実モデルを使う (ネットワーク不要)
LIVECAP_NONASCII_REAL_MODELS=1 uv run pytest tests/nonascii -m "nonascii_paths and slow" -q

# heavy tier — NeMo / sentencepiece (実測済み。既存パッケージのバージョンは動かない)
uv sync --extra engines-nemo
LIVECAP_NONASCII_REAL_MODELS=1 uv run pytest tests/nonascii -m "nonascii_paths and slow" -q

# 証拠 JSON を書き出す
uv run pytest tests/nonascii -m nonascii_paths \
    --nonascii-report=benchmark_results/nonascii/<date>/results.json

# 棚卸し表を再生成する (docs の §0 / §3 に貼る)
uv run python -m tests.nonascii.report --json benchmark_results/nonascii/<date>/results.json
```

### 環境変数

| 変数 | 用途 |
|---|---|
| `LIVECAP_NONASCII_ROOT` | プローブ用 base root を明示する (**ASCII 必須**) |
| `LIVECAP_NONASCII_REAL_MODELS` | `1` で real_model tier を有効化 |

base root は **ASCII かつ書き込み可能な候補を探索**する (`roots.py`):
「モデルと同一ボリューム → `repo/.tmp` → `%ProgramData%` → `%SystemDrive%` →
`%PUBLIC%` → システム `%TEMP%`」。システム `%TEMP%` へ無条件に落とすと、
**まさに検証したい環境** (Windows ユーザー名が非 ASCII) でハーネスが動かない。
モデルと同一ボリュームを優先するのは `os.link` (ハードリンク、管理者不要・
追加バイトゼロ) を効かせるため — 8.8 GB のモデルでもミリ秒で実体化できる。
不可なら `shutil.copy2` に降格し、どちらを使ったかを run メタデータに記録する。
`LIVECAP_NONASCII_ROOT` の明示指定が使えない場合は**黙って fallback せず raise** する。

**探索で得られるのは「共有される親」で、実際の base root はその下の
`run-<pid>-<uuid>` (run 固有)**。固定 root を共有すると並行 run が同じ probe パスを
読み書きし、片方の teardown がもう片方の実行中データを消してしまう —
これは本ハーネスが調査対象としている「共有ディレクトリを rmtree する」欠陥と
同じ構造なので、繰り返さないようにしている。後始末は session root だけを消す。

異常終了の残骸は best-effort 回収するが、削除するのは以下を**すべて**満たすものだけ:

1. 厳密な session 名形式 — glob の `run-*` だけでは `run-backup` も拾う
2. **所有権マーカー** (`.livecap-nonascii-session.json`) — `LIVECAP_NONASCII_ROOT` に
   利用者が任意の既存ディレクトリを指定できる以上、我々の生成物であることを確認する
3. **使用中ロックを掴める** = 所有プロセスが終了済み。経過時間だけで判定すると、
   heavy / real_model tier や低速環境で閾値を超えて**実行中**の session を消してしまう
4. マーカーの `created_at` が閾値より古い (保守的な追加条件)

回収の目的は**ディスクの衛生**である。session root は UUID で分離されているので、
古い残骸が新しい run に混入することはない。「あれば嬉しい」程度の位置づけなので、
少しでも危ないなら消さない方に倒している。
`LIVECAP_NONASCII_KEEP=1` で後始末を丸ごとスキップできる。

**root を確保できない状態、および非 ASCII variant が 1 つも受理されない状態は
skip ではなく失敗**になる。cheap tier は既定スイートに載っているので、
「green = 実際に測った」でなければ意味がないため。

## tier とゲート

```
cheap       : @pytest.mark.nonascii_paths                                  → 既定スイート
real_model  : + @pytest.mark.slow + LIVECAP_NONASCII_REAL_MODELS=1 + ローカルモデル存在
heavy/nemo  : + @pytest.mark.slow + nemo-toolkit (engines-nemo extra)
network     : + @pytest.mark.network
```

heavy tier では `.nemo` のパスと NeMo 内部の `%TEMP%` 展開先という **2 つの副境界**を
切り分けて測る。`.nemo` のパスだけを変えたい行では、`env_extra` で `%TEMP%` を ASCII 側へ
固定する — 両方を同時に非 ASCII にすると主因が分からなくなるため。

新規マーカーは `nonascii_paths` の 1 個だけ。重い tier は既存の `slow` / `network`
ゲートを再利用する。`nonascii_paths` は `addopts` の deny-list に**入れない** —
cheap tier は速く決定的で、まさに #375 / #379 / #377 が必要とする回帰ゲートだから
(opt-in にすると走らなくなり形骸化する)。

## 実装 PR (#375 / #379 / #377) からの使い方

`tests/` は import 可能なパッケージなので、runner を直接呼べる:

```python
from tests.nonascii import run_probe

result = run_probe(
    "sherpa.from_transducer.real",
    variant_id="cjk_kana",
    base_root=tmp_path,
    boundary_id="engine.reazonspeech.sherpa_from_transducer",
    payload={"model_source": str(model_dir)},
)
assert result.verdict == "pass"   # 修正後はこうなるはず
```

境界を直したら **`registry.py` の `expected_verdict` / `verified_method` を更新し、棚卸し表を再生成する**。
`test_probes.py` がその期待値を回帰ゲートとして assert している。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `registry.py` | **棚卸し表の source of truth**。1 BoundarySpec = 表の 1 行。`candidate_method` (決定) と `verified_method` (実測で確定) を分ける |
| `roots.py` | ASCII 保証された base root の探索。システム `%TEMP%` が非 ASCII でもハーネスが動くようにする |
| `record.py` | 結果レコード / 測定メタデータ / JSON I/O |
| `paths.py` | variant 語彙、FS 受理判定、8.3 照会 |
| `runner.py` | 子 env 構築、timeout、crash 捕捉、**verdict 導出** |
| `worker.py` | 子プロセス側エントリ (`python -m tests.nonascii.worker`) |
| `artifacts.py` | 合成アーティファクト生成、実モデルの hardlink 実体化 |
| `report.py` | registry + results → markdown |
| `probes/` | 境界ごとのプローブ実装 |
| `test_registry.py` | **受け入れ条件の機械化** (未分類ゼロ / callsite 生存 / 証拠の有無) |
| `test_probes.py` | registry × variant の実行 + 期待値の回帰ゲート |
| `test_harness_selftest.py` | **ハーネス自身の検証** (仕込み欠陥を正しく分類できるか) |

## 行番号を registry に書かない理由

行番号は数日で腐る。`callsite_symbol` (ファイル内に存在する文字列) を保持し、
レンダリング時に検索して行番号を解決する。`test_registry.py::test_callsites_exist`
が全 callsite の生存を検査するので、#375 / #379 / #377 が実装でコードを動かしても
表が黙って腐らない。
