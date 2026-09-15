# 翻訳のトラブルシューティング

## Google 翻訳が動かない / 原文がそのまま出る

**Google 翻訳は非公式エンドポイントを叩いており、Google 側の変更で壊れる。** これは既知の設計上のトレードオフで、Issue [#402](https://github.com/Mega-Gorilla/livecap-cli/issues/402) の D6 に方針として記録してある。恒久的な保証は公式 Cloud Translation API (API キー必須) にしか無い。

過去に壊れた例:

| 時期 | 原因 | 症状 |
|---|---|---|
| **2026-09** ([#442](https://github.com/Mega-Gorilla/livecap-cli/issues/442)) | `translate.google.com/m` が **abuse 検知に振り分けられ、302 → `www.google.com/sorry/` → 429 + reCAPTCHA**。同じ頃 `translate_a/single` でも **`client=gtx` が遮断** (429 + "automated queries") | `TranslationError(reason="bot_challenge")`。**再送しても直らない**。`translate.googleapis.com/translate_a/single` (JSON) へ切り替え、client 識別子を `at` にした (下記「client 識別子」) |
| 2026-08 ([#402](https://github.com/Mega-Gorilla/livecap-cli/issues/402)) | User-Agent が絞られ、**HTTP 200 のまま本文が "Error 500" ページ**になった | 原文がそのまま出る |
| (それ以前) | 結果要素の class が `t0` → `result-container` へ変わった | 原文がそのまま出る |

### まず切り分ける

**ローカルの翻訳エンジンへ切り替えて再現するか確認する。**

```bash
uv run livecap-cli transcribe input.mp4 -o out.srt --translate opus_mt --target-lang en
```

`opus_mt` (ja↔en) はローカル実行なので Google の状態に依存しない。これで翻訳できるなら、原因は Google 経路にある。

### 例外の `reason` で判断する

`livecap_cli/translation/impl/google.py` は失敗を `reason` で分類する。**まず `reason` を見る** — 対処が変わる。

| `reason` | 意味 | 対処 |
|---|---|---|
| **`bot_challenge`** | Google の reCAPTCHA 判定 (sorry ページへの redirect、または 429 + captcha 本文) | **再送しない。** 時間を置くか `opus_mt` / `riva_instruct` へ。プログラムからは突破できない |
| `http_status` (429 / 5xx) | 一時的な失敗 | 呼び出し側の retry policy が再送する |
| `http_status` (4xx) | 恒久的な失敗 | endpoint / パラメータを疑う (下記 2) |
| `layout_changed` | 200 だが JSON の形が想定と違う | 契約が変わった (下記 3) |
| `embedded_error_page` | 200 だが本文が HTML エラーページ | 一時的。再送される |
| `unsupported_language_pair` | 同一言語、または**実在しないコード** (`xx` / `jp`) | 言語コードを直す。gtx は無効なコードでも 200 で原文を返すので**送信前に弾いている** |

### 調査手順

対象は `livecap_cli/translation/impl/google.py`。定数はすべてファイル冒頭に集約してある。

#### 1. bot 判定されていないか確認する

**これが最初。** 2026-09 の障害はここだった。

```bash
curl -s -o /dev/null -w 'status=%{http_code} final=%{url_effective}\n' -L \
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36" \
  "https://translate.googleapis.com/translate_a/single?client=dict-chrome-ex&sl=ja&tl=en&dt=t&dj=1&q=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF"
```

- `final=` が **`www.google.com/sorry/...`** → **bot 判定**。ヘッダを足しても越えられない (#442 で実測)。時間を置くか、別 translator を使う
- `status=429` で本文が **`<title>Sorry...</title>` / 「automated queries」** → 同上 (gtx は redirect せずこの形で返す)
- `status=200` → 次へ

> **`/m` (旧経路) は使わない。** `https://translate.google.com/m?...` を叩くと 2026-09 以降は sorry へ飛ばされる。

**curl で 200 なのに adapter は 429、という状態があり得る。** 2026-09-15 の実測では、同じ URL・同じ UA・同じヘッダ・同じ HTTP/1.1 でも `curl` (Schannel TLS) は 200、Python の `requests` (OpenSSL) は 429 Sorry だった — Google は **TLS クライアント指紋**でも判定している。adapter と同じ経路で確かめるには Python から叩くこと:

```python
import requests
from livecap_cli.translation.impl.google import BROWSER_UA, DEFAULT_CLIENT, ENDPOINT, FIXED_PARAMS
r = requests.get(ENDPOINT, params={"client": DEFAULT_CLIENT, **FIXED_PARAMS, "sl": "ja", "tl": "en", "q": "こんにちは"},
                 headers={"User-Agent": BROWSER_UA}, timeout=20)
print(r.status_code, r.url, r.text[:120])
```

> **TLS 指紋を偽装して越える (browser impersonation ライブラリ等) ことはしない。** それは bot 検知の回避であり、この adapter の設計方針の外にある。この状態のときは `bot_challenge` として落ち、`opus_mt` / `riva_instruct` を案内する。

#### client 識別子 — `gtx` の遮断 (2026-09-14 頃〜)

`translate_a/single` の `client=` は Google 側の判定に使われる。**`gtx` は 2026-09-14 頃から遮断された** — 複数の無関係な ISP から同じ curl で 429 + "Sorry... automated queries" が再現している ([eeeXun/gtt#43](https://github.com/eeeXun/gtt/issues/43)、[noctalia-dev/official-plugins#64](https://github.com/noctalia-dev/official-plugins/issues/64))。手元 (2026-09-15) でも:

```
                                 2026-09-15 (#442)   2026-09-16 (#451、同一 IP)
requests  client=gtx            429 Sorry            429 Sorry
requests  client=at             200 JSON             429 Sorry  (同日中に 429 へ)
requests  client=dict-chrome-ex 200 JSON             200 JSON   (12/12)
```

adapter の既定は **`client=dict-chrome-ex`** (`DEFAULT_CLIENT` 定数。#442 で採った `at` は翌日この IP から 429 になった、#451)。環境変数 **`LIVECAP_GOOGLE_TRANSLATE_CLIENT`** で上書きできる (例: `at` が通る環境で戻す、次に既定が塞がれたとき再デプロイ無しで切り替える)。fallback 連鎖はしない — bot 判定された endpoint へ別 client で再送する形になるため。**`at` / `dict-chrome-ex` は Google 自身のアプリ / 拡張の識別子**であり、gtx の遮断が第三者利用を切る意図なら、これはその意図を迂回する形になる。次に `at` が塞がれる可能性は残り、そのときは `bot_challenge` で落ちる。**識別子を `gtx` へ戻さないこと** (テスト `test_client_is_not_gtx` が守る)。恒久的な保証は公式 Cloud Translation API ([#445](https://github.com/Mega-Gorilla/livecap-cli/issues/445)) にしか無い。

#### 2. endpoint とパラメータを確認する

上の curl で `status=200` なのに翻訳されない場合。

```bash
curl -s \
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36" \
  "https://translate.googleapis.com/translate_a/single?client=dict-chrome-ex&sl=ja&tl=en&dt=t&dj=1&q=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF"
```

期待する形:

```json
{"sentences":[{"trans":"Hello","orig":"こんにちは","backend":10}],"src":"ja","spell":{}}
```

- **200 以外** → `ENDPOINT` / `FIXED_PARAMS` が変わった可能性
- **JSON だが `sentences[*].trans` が無い** → 次へ

#### 3. JSON の契約を確認する

adapter は `sentences` (list) の各要素の `trans` (str) を**連結**する。複数文は要素が分かれ、改行は `trans` の中に保持される。**1 つでも `{"trans": str}` を満たさない要素があれば `layout_changed` で落とす** — 読み飛ばして途中までの翻訳を成功として返すと、字幕では parser error より危険な silent degradation になる。

```python
import requests
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
r = requests.get("https://translate.googleapis.com/translate_a/single",
                 params={"client": "at", "dt": "t", "dj": "1", "sl": "ja", "tl": "en", "q": "こんにちは。今日はいい天気ですね。"},
                 headers={"User-Agent": UA}, timeout=20)
print(r.status_code, r.url)
print(r.json())
```

`sentences` のキー名や入れ子が変わっていれば `_extract_translation()` を更新する。**`dt=t` だけの配列形式には戻さない** — 位置依存で、要素が増減すると黙って壊れる。

#### 4. User-Agent を A/B する

**必ず交互に実行する。** 連続実行だと時間帯による回復と区別がつかない。

```python
import requests, time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
PARAMS = {"client": "at", "dt": "t", "dj": "1", "sl": "ja", "tl": "en", "q": "こんにちは"}

def hit(headers):
    r = requests.get("https://translate.googleapis.com/translate_a/single",
                     params=PARAMS, headers=headers, timeout=20)
    return r.status_code == 200 and "sentences" in r.text

default = browser = 0
for _ in range(10):
    default += hit({})
    time.sleep(0.4)
    browser += hit({"User-Agent": UA})
    time.sleep(0.4)
print(f"default UA: {default}/10   browser UA: {browser}/10")
```

差が出るなら **UA の問題**。`BROWSER_UA` を実在する新しいブラウザのものへ更新する。

#### 5. テストで固定する

修正したら、**その失敗形を再現する単体テストを必ず追加する**。#402 の根本原因が長く気付かれなかったのは、この経路に実 HTTP のテストが無かったためである。

```bash
uv run pytest tests/core/translation -q
uv run pytest tests/core/translation -q -m network   # 実エンドポイントへの疎通
```

`-m network` は既定で除外されている (`pyproject.toml` の `addopts`)。

### 直せない・時間が無い場合

**`opus_mt` を案内する。** ローカル実行で ja↔en に対応し、Google の状態に影響されない。GPU があれば `riva_instruct` も使える。

## ログに翻訳対象のテキストが出ていないか

**出ていたら bug として報告してほしい。** 翻訳対象は GET query の `q=` に入るため、通信ライブラリの例外文字列には発話内容が percent-encode された URL ごと含まれる。**bot 判定時の sorry ページ URL も `continue=` に query 全体を含む**ので、adapter は URL を例外メッセージに入れない。

`livecap_cli/translation/impl/google.py` は例外を必ず `from None` で chain を切り、診断情報は `provider` / `reason` / `status_code` の構造化フィールドだけを持つ (#402 D8)。`from error` に戻すと、呼び出し側が `exc_info=True` でログを出した瞬間に発話が漏れる。

回帰テストは `tests/core/translation/test_google_translator.py::TestNoSpeechLeak` にある。

## 翻訳が途中から出なくなる / 遅れて出る

リアルタイム経路は **fail fast** で、失敗した発話の翻訳は諦めて次へ進む (#402 D10)。遅れて出す方が字幕としては邪魔になるため。

**どの状態なのかは `TranscriptionResult.translation_state` で分かる** — `failed` (障害) / `skipped_busy` (輻輳時の方針) / `empty` / `not_requested` / `translated`。障害なら `on_translation_status` にも通知が飛ぶ。

待ち時間は既定 5.0 秒で、環境変数で調整できる:

```bash
LIVECAP_TRANSLATION_TIMEOUT=10 uv run livecap-cli ...
```

回線が遅い環境や、プロキシ経由で一律失敗する場合に上げる。不正な値 (0 以下・数値以外) は警告のうえ既定へフォールバックする。

**knob はこれ 1 つ。** リアルタイムはリトライしない (`max_attempts=1`) ので、実効的な上限は「待つ時間」そのものになる。リトライ予算用に別の変数を持つと、片方だけ設定して効かない事故になる。

超過した segment は原文のまま出て `translation_state="failed"` になり、`on_translation_status` で 1 回通知される。前の翻訳が終わるまで、後続の segment は `skipped_busy` として翻訳を飛ばす — 数秒前の発話に対する字幕が今の音声に重なるのを防ぐため。

## 関連

- [#402](https://github.com/Mega-Gorilla/livecap-cli/issues/402) — Google 翻訳の修復 (設計判断 D1〜D10)
- [#442](https://github.com/Mega-Gorilla/livecap-cli/issues/442) — `/m` の reCAPTCHA 化と gtx への切り替え、bot 判定の non-retryable 化
- `livecap_cli/translation/impl/google.py` — adapter 本体。定数は冒頭に集約
- `livecap_cli/translation/retry.py` — リトライ方針 (呼び出し側が選ぶ)
