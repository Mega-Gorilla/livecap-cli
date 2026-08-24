# 翻訳のトラブルシューティング

## Google 翻訳が動かない / 原文がそのまま出る

**Google 翻訳はウェブ版のスクレイピングであり、Google 側の変更で壊れる。** これは既知の設計上のトレードオフで、Issue [#402](https://github.com/Mega-Gorilla/livecap-cli/issues/402) の D6 に方針として記録してある。

過去に壊れた例:

| 時期 | 原因 |
|---|---|
| 2026-08 | User-Agent が絞られ、**HTTP 200 のまま本文が "Error 500" ページ**になった |
| (それ以前) | 結果要素の class が `t0` → `result-container` へ変わった |

### まず切り分ける

**ローカルの翻訳エンジンへ切り替えて再現するか確認する。**

```bash
uv run livecap-cli transcribe input.mp4 -o out.srt --translate opus_mt --target-lang en
```

`opus_mt` (ja↔en) はローカル実行なので Google の状態に依存しない。これで翻訳できるなら、原因は Google 経路にある。

### 調査手順

対象は `livecap_cli/translation/impl/google.py`。定数はすべてファイル冒頭に集約してある。

#### 1. User-Agent を A/B する

**必ず交互に実行する。** 連続実行だと時間帯による回復と区別がつかない。

```python
import requests, time
from bs4 import BeautifulSoup  # 調査用。製品コードは stdlib を使う

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

def hit(headers):
    r = requests.get("https://translate.google.com/m",
                     params={"sl": "ja", "tl": "en", "q": "こんにちは"},
                     headers=headers, timeout=20)
    return 'class="result-container"' in r.text

default = browser = 0
for _ in range(10):
    default += hit({})
    time.sleep(0.4)
    browser += hit({"User-Agent": UA})
    time.sleep(0.4)
print(f"default UA: {default}/10   browser UA: {browser}/10")
```

差が出るなら **UA の問題**。`BROWSER_UA` を実在する新しいブラウザのものへ更新する。

#### 2. エンドポイントを確認する

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36" \
  "https://translate.google.com/m?sl=ja&tl=en&q=%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%81%AF"
```

- **200 以外** → `ENDPOINT` が移動した可能性
- **200 だが翻訳が出ない** → 次へ

#### 3. 結果要素の class を確認する

```python
import re, requests
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
r = requests.get("https://translate.google.com/m",
                 params={"sl": "ja", "tl": "en", "q": "こんにちは"},
                 headers={"User-Agent": UA}, timeout=20)
body = r.text[r.text.find("<body"):]
print(re.sub(r"\s+", " ", body)[:1500])
```

`<div class="...">Hello</div>` の class 名が `RESULT_CLASS` と違っていれば、そこを更新する。

**`Error 500 (Server Error)` が本文に含まれていないかも確認する** — HTTP 200 でもエラーページが返ることがある。その場合 `_ERROR_PAGE_MARKERS` に文言を追加する。

#### 4. テストで固定する

修正したら、**その失敗形を再現する単体テストを必ず追加する**。#402 の根本原因が長く気付かれなかったのは、この経路に実 HTTP のテストが無かったためである。

```bash
uv run pytest tests/core/translation -q
uv run pytest tests/core/translation -q -m network   # 実エンドポイントへの疎通
```

`-m network` は既定で除外されている (`pyproject.toml` の `addopts`)。

### 直せない・時間が無い場合

**`opus_mt` を案内する。** ローカル実行で ja↔en に対応し、Google の状態に影響されない。GPU があれば `riva_instruct` も使える。

## ログに翻訳対象のテキストが出ていないか

**出ていたら bug として報告してほしい。** 翻訳対象は GET query の `q=` に入るため、通信ライブラリの例外文字列には発話内容が percent-encode された URL ごと含まれる。

`livecap_cli/translation/impl/google.py` は例外を必ず `from None` で chain を切り、診断情報は `provider` / `reason` / `status_code` の構造化フィールドだけを持つ (#402 D8)。`from error` に戻すと、呼び出し側が `exc_info=True` でログを出した瞬間に発話が漏れる。

回帰テストは `tests/core/translation/test_google_translator.py::TestNoSpeechLeak` にある。

## 翻訳が途中から出なくなる / 遅れて出る

リアルタイム経路は **fail fast** で、失敗した発話の翻訳は諦めて次へ進む (#402 D10)。遅れて出す方が字幕としては邪魔になるため。

**どの状態なのかは `TranscriptionResult.translation_state` で分かる** — `failed` (障害) / `skipped_busy` (輻輳時の方針) / `empty` / `not_requested` / `translated`。障害なら `on_translation_status` にも通知が飛ぶ。

待ち時間は既定 2.0 秒で、環境変数で調整できる:

```bash
LIVECAP_TRANSLATION_TIMEOUT=5 uv run livecap-cli ...
```

回線が遅い環境や、プロキシ経由で一律失敗する場合に上げる。不正な値 (0 以下・数値以外) は警告のうえ既定へフォールバックする。

**knob はこれ 1 つ。** リアルタイムはリトライしない (`max_attempts=1`) ので、実効的な上限は「待つ時間」そのものになる。リトライ予算用に別の変数を持つと、片方だけ設定して効かない事故になる。

超過した segment は原文のまま出て `translation_state="failed"` になり、`on_translation_status` で 1 回通知される。前の翻訳が終わるまで、後続の segment は `skipped_busy` として翻訳を飛ばす — 数秒前の発話に対する字幕が今の音声に重なるのを防ぐため。

## 関連

- [#402](https://github.com/Mega-Gorilla/livecap-cli/issues/402) — Google 翻訳の修復 (設計判断 D1〜D10)
- `livecap_cli/translation/impl/google.py` — adapter 本体。定数は冒頭に集約
- `livecap_cli/translation/retry.py` — リトライ方針 (呼び出し側が選ぶ)
