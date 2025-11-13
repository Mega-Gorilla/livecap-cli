# 実装ロードマップ

**作成日**: 2025-10-24
**タイムライン**: 4-6週間（最短）〜 3ヶ月（推奨）
**前提**: 初期投資ゼロ、週数時間の作業時間

---

## 📋 目次

1. [全体スケジュール](#全体スケジュール)
2. [Phase 1: 準備と確認](#phase-1-準備と確認)
3. [Phase 2: ドキュメント整備](#phase-2-ドキュメント整備)
4. [Phase 3: GitHubリポジトリ準備](#phase-3-githubリポジトリ準備)
5. [Phase 4: 公開実施](#phase-4-公開実施)
6. [Phase 5: フォローアップ](#phase-5-フォローアップ)

---

## 全体スケジュール

### クリティカルパス

```
Week 1-2: Steam確認 ← 最重要！これなしに進めない
    ↓
Week 2-3: ライセンス文書・README作成
    ↓
Week 3-4: GitHubリポジトリ準備
    ↓
Week 4: コミュニティ告知（Steam + Discord）
    ↓
Week 5: GitHub公開
    ↓
Week 6〜: フォローアップ・コミュニティ育成
```

### タイムライン別プラン

#### 最速プラン（4-5週間）

| Week | Phase | 作業時間/週 |
|------|-------|-----------|
| 1 | Steam確認 + Discord告知 | 2時間 |
| 2 | ライセンス文書作成 | 4時間 |
| 3 | GitHub準備 + README | 5時間 |
| 4 | コミュニティ告知 | 2時間 |
| 5 | 公開実施 | 3時間 |

**メリット**: 早期公開
**デメリット**: 準備不足のリスク

#### 推奨プラン（6-8週間）

| Week | Phase | 作業時間/週 |
|------|-------|-----------|
| 1-2 | Steam確認 + コミュニティ意見収集 | 3時間 |
| 3-4 | ライセンス・CLA・ドキュメント | 4時間 |
| 5-6 | GitHub準備・テスト | 4時間 |
| 7 | コミュニティ告知・調整 | 2時間 |
| 8 | 公開実施 | 3時間 |

**メリット**: 丁寧な準備、リスク最小化
**デメリット**: 時間がかかる

#### 慎重プラン（3ヶ月）

| Month | Phase | 特徴 |
|-------|-------|------|
| 1ヶ月目 | 調査・確認 | Steam確認、Discord意見収集、弁護士相談（オプション） |
| 2ヶ月目 | ドキュメント整備 | 完璧なREADME、多言語対応、ビルド手順 |
| 3ヶ月目 | 公開準備・実施 | ソフトローンチ、段階的公開 |

**メリット**: 最もリスクが低い
**デメリット**: モメンタムを失う可能性

---

## Phase 1: 準備と確認

**期間**: Week 1-2
**目標**: Steam規約確認とコミュニティの反応把握

### タスクリスト

#### 1.1 Steam事前調査 ✅ **完了**

- [x] **Steam公式ドキュメント調査**
  - [Distributing Open Source Applications on Steam](https://partner.steamgames.com/doc/sdk/uploading/distributing_opensource)
  - GPL互換性: Steamworks SDK未使用なら問題なし
  - 詳細: [06-steam-research-findings.md](06-steam-research-findings.md)

- [x] **成功事例の調査**
  - Tales of Maj'Eyal（GPLv3エンジン + Steam販売）
  - HyperRogue（GPL v2 + Steam販売）
  - 両者とも GitHub無料 + Steam有料で成功

- [x] **Steamworks SDK使用状況確認**
  - コードベース全体を検索
  - **結果**: Steamworks SDKは一切使用していない ✅
  - **影響**: GPL互換性問題は完全に存在しない

**所要時間**: 完了（2-3時間）

**📊 調査結果サマリー**:
- ✅ LiveCapのOSS化（GPL v3）+ Steam販売は完全に可能
- ✅ Steamworks SDK未使用 → GPL互換性問題なし
- ✅ 成功事例2件以上確認済み
- ⚠️ Steamサポートへの問い合わせは任意（推奨だが必須ではない）

---

#### 1.2 Steam規約確認（オプション・推奨）

**注**: 上記調査により、Steamサポート問い合わせは任意となりました。
成功事例があり、技術的問題もないため、問い合わせなしで進めることも可能です。

- [ ] Steamサポートへ問い合わせ送信（**任意**）
  - 最適化されたテンプレート: [06-steam-research-findings.md](06-steam-research-findings.md#推奨アクション)
  - 送信先: Steam Partner Support
  - 期待回答時間: 3-5営業日

- [ ] Steam回答の記録（問い合わせした場合）
  - 承認/条件付き承認/不承認
  - 条件がある場合の対応策検討

**所要時間**: 30分（問い合わせ作成、任意） + 回答待ち（3-5営業日）

**推奨判断**:
- **問い合わせする**: より慎重、公式確認あり、時間がかかる
- **問い合わせしない**: 成功事例と調査結果で十分、早く進められる

---

#### 1.3 コミュニティ意見収集

- [ ] Discord内部アンケート（以下のテンプレートを使用）
  ```markdown
  @everyone

  # LiveCapをオープンソース化しようと思ってます

  やあ、Hakaseです。

  突然ですが、**LiveCapをオープンソース化しようか考えてます**。GPL v3ライセンスで、GitHubに全コード公開する計画です。

  でも、実行する前に**あなたの意見を聞きたい**。

  「え、待って。俺Steam版買ったんだけど？」って思った方、ちょっと待ってください。後で説明します。

  ---

  ## なんでオープンソースにするの？

  正直に言います。

  **最近、開発がめっちゃ遅い。** 一人でやってると限界があるんですよね。

  で、考えました：
  - もっと色んな人に開発に参加してもらいたい
  - バグ修正も新機能も、コミュニティ全体でやれたら最高じゃない？
  - コード公開すれば「このソフト、変なことしてない？」って不安もなくなる
  - 万が一私が開発やめても、プロジェクトは生き続ける

  ---

  ## で、あなたにとって何が良いの？

  - **コードが丸見え**: 変なことしてないって証明できる
  - **自分好みに改造OK**: Pythonわかる人は好きにいじってください
  - **バグ修正が超速**: コミュニティ全体で対応するから早い
  - **新機能リクエスト**: 欲しい機能、自分で実装してもいいし、誰かが作ってくれるかも
  - **永久サポート**: 私が消えても、誰かが引き継いでくれる（たぶん）

  私にとっても良いことあります：
  - みんなの力でもっと良いソフトになる
  - 翻訳とか、ドキュメントとか、テストとか、手伝ってくれる人が増える
  - オープンソースプロジェクトって言えるとカッコいい（これ本音）

  ---

  ## 正直、デメリットもある

  良いことばっかりじゃないので、正直に言います。

  **パクられる可能性**
  誰かが「LiveCap改」とか作って配布するかも。でもGPL v3のおかげで、そのパクリ版もオープンソースにしないといけないルールです。

  **サポート地獄になるかも**
  ユーザー増える → 質問増える → 私死ぬ。でもコミュニティサポートで乗り切ります。助けてください。

  **GitHubのコミュニティ運用、初めてです**
  正直言うと、GitHubでオープンソースプロジェクトの運用、やったことない。Issue管理とか、PR レビューとか、うまくできるか不安。

  でも、**大丈夫。いままでLiveCapを育ててきたんだから。**

  **ビルドがめんどい**
  GitHub版は自分でビルドしないといけない。Python環境とか、正直めんどくさい。だからSteam版があるんです（後述）。

  **企業がタダ乗りするかも**
  どっかの会社が勝手に製品化する可能性。これは私が泣きます。GPL v3で一応防げるけど、商用ライセンスも用意して対応します。

  ---

  ## で、Steam版買った人はどうなるの？

  **大丈夫。Steam版は継続販売します。**

  実は、すでにオープンソース化してSteamで売ってるソフト、結構あります。
  - **Tales of Maj'Eyal**: オープンソース（GPLv3）+ Steam販売で成功してる
  - **HyperRogue**: 同じくオープンソース（GPL v2）+ Steam販売

  つまり、**オープンソース化してもSteam販売は普通に続けられる**ってこと。先例あり。

  ### GitHub版 vs Steam版

  | 項目 | GitHub版 | Steam版 |
  |------|----------|---------|
  | **値段** | タダ | ¥1,980 |
  | **インストール** | 自分でビルド（地獄） | ポチッと完了 |
  | **アップデート** | 自分でやる | 勝手に更新される |
  | **サポート** | コミュニティ頼み | 優先対応します |
  | **技術要件** | Python環境とか色々 | 何もいらない |
  | **開発支援** | できない | できる |

  ### すでに買ってくれた方へ

  **マジでありがとうございます。**

  あなた方のおかげでここまで開発できました。感謝してます。

  お礼として：
  - Discord内で「**Early Supporter**」ロール付与します（カッコいい）
  - 優先サポート継続（当然）
  - 開発方針、あなたの意見を優先的に聞きます

  ---

  ## あなたの意見を聞かせてください

  リアクションで教えてください：

  - 👍 **いいね** - やろう
  - 🤔 **う〜ん** - もうちょい説明して
  - 👎 **ダメ** - やめとけ
  - 💬 **もっと話したい** - スレッドでどうぞ

  ### 特に知りたいこと

  1. Steam版買った人、不安ある？正直に言ってください
  2. 開発に協力したい？バグ修正とか、機能追加とか、ドキュメントとか
  3. 欲しい機能ある？今のうちに言って
  4. 他に心配なことある？

  **〜11/1まで待ってます**
  ```

- [ ] Discord内部アンケート【英語版】（以下のテンプレートを使用）
  ```markdown
  @everyone

  # Thinking about making LiveCap open source

  Hey, it's Hakase.

  So, I'm **thinking about making LiveCap open source**. Planning to release all the code on GitHub under GPL v3 license.

  But before I do that, **I want to hear what you think**.

  "Wait, I bought the Steam version though?" — Hold on, I'll explain.

  ---

  ## Why go open source?

  Let me be honest.

  **Development has been really slow lately.** There's only so much one person can do.

  So I thought:
  - I want more people to join development
  - Bug fixes and new features could be done by the whole community — wouldn't that be awesome?
  - Open code means no more "is this software doing something sketchy?" worries
  - If I ever stop developing, the project can live on

  ---

  ## What's in it for you?

  - **Code is transparent**: Proof nothing sketchy is going on
  - **Customize freely**: If you know Python, go wild
  - **Bug fixes are faster**: The whole community tackles them
  - **Feature requests**: Implement it yourself, or maybe someone else will
  - **Forever support**: If I disappear, someone will pick it up (probably)

  What's in it for me:
  - With everyone's help, the software gets better
  - More people to help with translations, docs, testing
  - Being able to say "open source project" sounds cool (yeah, this is real)

  ---

  ## Honestly, there are downsides

  Not everything is great, so let me be real.

  **Could get copied**
  Someone might make "LiveCap Plus" and distribute it. But thanks to GPL v3, those copies have to be open source too.

  **Support hell incoming**
  More users → more questions → I die. But we'll handle it with community support. Please help.

  **First time running a GitHub community**
  Honestly, I've never managed an open source project on GitHub. Issue management, PR reviews — I'm worried I'll mess up.

  But **it's fine. I've built LiveCap this far, haven't I?**

  **Building is a pain**
  GitHub version requires building yourself. Python environment and all that — honestly annoying. That's why the Steam version exists (more below).

  **Companies might freeload**
  Some company might turn it into a product without permission. That would make me cry. GPL v3 should prevent this, but I'll also offer commercial licensing.

  ---

  ## What about people who bought the Steam version?

  **Don't worry. Steam version continues.**

  Actually, there are already open source software sold on Steam.
  - **Tales of Maj'Eyal**: Open source (GPLv3) + Steam sales, successful
  - **HyperRogue**: Same, open source (GPL v2) + Steam sales

  So, **going open source doesn't stop Steam sales**. There's precedent.

  ### GitHub version vs Steam version

  | Item | GitHub version | Steam version |
  |------|----------------|---------------|
  | **Price** | Free | ¥1,980 |
  | **Install** | Build yourself (hell) | One click, done |
  | **Updates** | Do it yourself | Automatic |
  | **Support** | Community-based | Priority support |
  | **Tech requirements** | Python environment etc. | Nothing needed |
  | **Dev support** | Can't | Can |

  ### To those who already bought it

  **Thank you so much.**

  I've been able to develop this far thanks to you. I appreciate it.

  As thanks:
  - Discord "**Early Supporter**" role (looks cool)
  - Priority support continues (of course)
  - Your opinions on development direction get priority

  ---

  ## Let me know what you think

  React to let me know:

  - 👍 **Yes** - Let's do it
  - 🤔 **Hmm** - Need more explanation
  - 👎 **No** - Don't do it
  - 💬 **Want to talk more** - Use thread

  ### What I especially want to know

  1. Steam buyers, any concerns? Be honest
  2. Want to help with development? Bug fixes, features, docs, etc.
  3. Any features you want? Speak up now
  4. Any other worries?

  **Waiting until Nov 1st**
  ```

- [ ] 既存ユーザーの反応分析
  - 賛成/反対の比率
  - 懸念事項の抽出
  - 返金要求の可能性評価
  - Early Supporterロールへの反応

**所要時間**: 1時間（アンケート作成・投稿） + 1週間（回答収集・分析）

**📊 分析ポイント**:
- リアクション数の集計（賛成/保留/反対）
- コメントでの具体的な懸念事項
- Steam購入者の反応（特に重要）
- 機能要望の収集

---

#### 1.4 リポジトリ監査

- [ ] 機密情報の確認
  - APIキー、トークンの削除
  - 個人情報の削除
  - テスト用のハードコード値の削除

- [ ] ファイル整理
  - 不要なファイルの削除
  - .gitignoreの見直し

- [ ] コミット履歴の確認
  - 機密情報が過去のコミットにないか
  - 必要なら履歴のクリーンアップ（git filter-branch）

**所要時間**: 2-3時間

---

## Phase 2: ドキュメント整備

**期間**: Week 2-4
**目標**: 必要なライセンス文書とドキュメントの作成

### タスクリスト

#### 2.1 ライセンスファイル作成

- [ ] `LICENSE` ← GPLv3全文
  - テンプレート: [GPLv3公式テキスト](https://www.gnu.org/licenses/gpl-3.0.txt)
  - トップに著作権表示追加:
    ```
    LiveCap - Real-time Speech-to-Text Tool
    Copyright (C) 2025 Pine Lab
    ```

- [ ] `LICENSE-COMMERCIAL.md` ← 商用ライセンス
  - テンプレート: [05-templates.md](05-templates.md#商用ライセンス)
  - 連絡先メールアドレス記載

- [ ] `CLA.md` ← Contributor License Agreement
  - テンプレート: [05-templates.md](05-templates.md#cla)

**所要時間**: 1-2時間

#### 2.2 README更新

- [ ] デュアルライセンス説明追加
  - テンプレート: [05-templates.md](05-templates.md#readme)
  - 日本語版・英語版の両方

- [ ] ビルド手順の明文化
  - 初心者でもビルドできるレベルの詳細
  - 各OS（Windows/Linux/Mac）ごとの手順

- [ ] ライセンス選択ガイドへのリンク
  - `docs/LICENSE_SELECTION_GUIDE.md`を作成

**所要時間**: 3-4時間

#### 2.3 貢献ガイドライン

- [ ] `CONTRIBUTING.md` 作成
  ```markdown
  # Contributing to LiveCap

  ## License Agreement
  By contributing, you agree to the [CLA](CLA.md).

  ## How to Contribute
  1. Fork the repository
  2. Create a feature branch
  3. Make your changes
  4. Submit a Pull Request
  5. Sign the CLA (required)

  ## Code Style
  [既存のコーディング規約]

  ## Testing
  [テスト方法]
  ```

- [ ] Issue/PRテンプレート作成
  - `.github/ISSUE_TEMPLATE/bug_report.md`
  - `.github/ISSUE_TEMPLATE/feature_request.md`
  - `.github/PULL_REQUEST_TEMPLATE.md`

**所要時間**: 2時間

#### 2.4 ライセンス選択ガイド

- [ ] `docs/LICENSE_SELECTION_GUIDE.md` 作成
  - 個人利用 vs 商用利用の判断基準
  - Q&A形式
  - 日本語・英語版

**所要時間**: 1-2時間

---

## Phase 3: GitHubリポジトリ準備

**期間**: Week 4-5
**目標**: GitHub公開の準備完了

### タスクリスト

#### 3.1 リポジトリ設定

- [ ] GitHubリポジトリ作成（プライベート → 後でパブリック化）
  - リポジトリ名: `LiveCap` または `live-cap`
  - 説明: "Real-time multilingual speech-to-text with OBS/VRChat integration"
  - トピック: `speech-recognition`, `subtitles`, `obs`, `vrchat`, `asr`

- [ ] ライセンスバッジ追加
  - README.mdに `[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)`

- [ ] 基本的なリポジトリ設定
  - Issues有効化
  - Discussions有効化（コミュニティ用）
  - Wikiは無効（混乱回避）

**所要時間**: 30分

#### 3.2 CLA自動化（オプション・推奨）

- [ ] CLA Assistant導入
  - https://cla-assistant.io/ にサインアップ
  - GitHubリポジトリと連携
  - `CLA.md`のURLを設定
  - テストPRで動作確認

**所要時間**: 30分（初回設定のみ）

#### 3.3 ドキュメントの最終確認

- [ ] すべてのリンクが正しいか確認
- [ ] 画像が表示されるか確認
- [ ] ビルド手順が本当に動作するかテスト
  - クリーン環境（新しいVM等）で実行
  - 手順の抜け・誤りをチェック

**所要時間**: 2-3時間

#### 3.4 コミュニティ機能準備

- [ ] GitHub Discussions設定
  - カテゴリ作成: General, Q&A, Ideas, Show and Tell
  - 初期投稿作成（自己紹介・ルール）

- [ ] GitHub Projects（オプション）
  - ロードマップの可視化
  - コミュニティに開発優先順位を共有

**所要時間**: 1時間

---

## Phase 4: 公開実施

**期間**: Week 5-6
**目標**: GitHub公開とコミュニティへのアナウンス

### タスクリスト

#### 4.1 Steam対応（Steam確認結果に応じて）

- [ ] Steam説明ページ更新
  - テンプレート: [03-steam-integration.md](03-steam-integration.md#option-1の場合)
  - 日本語版・英語版

- [ ] Steam コミュニティ投稿作成
  - アナウンスメント形式
  - GitHubリンク含む

**所要時間**: 1-2時間

#### 4.2 コミュニティアナウンス

- [ ] Discord正式アナウンス
  - テンプレート: [05-templates.md](05-templates.md#discordアナウンス)
  - @everyone メンション（慎重に）

- [ ] Steam アップデート告知
  - 「重要なお知らせ」として投稿

**所要時間**: 30分

#### 4.3 GitHub公開

- [ ] リポジトリをパブリック化
  - Settings → Change visibility → Public
  - 確認メッセージに同意

- [ ] 初期ReleaseCreate
  - Tag: v2.1.0.1（現在のバージョン）
  - タイトル: "Initial Open Source Release"
  - 説明:
    ```markdown
    # LiveCap v2.1.0.1 - Initial Open Source Release

    LiveCap is now open source under GPL v3!

    ## What's LiveCap?
    Real-time speech-to-text tool with multiple ASR engines
    supporting 12+ languages, OBS/VRChat integration.

    ## Getting Started
    See [README.md](README.md) for build instructions.

    ## License
    Dual-licensed: GPL v3 (free) or Commercial License.
    See [LICENSE](LICENSE) and [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).
    ```

**所要時間**: 30分

#### 4.4 外部告知（オプション）

- [ ] Reddit投稿
  - r/opensource
  - r/selfhosted
  - r/VRchat

- [ ] Hacker News（控えめに）
  - Show HN形式
  - コミュニティの反応を見てから

**所要時間**: 1時間（各プラットフォーム）

---

## Phase 5: フォローアップ

**期間**: Week 6〜（継続的）
**目標**: コミュニティ育成と継続的改善

### タスクリスト

#### 5.1 初期対応（Week 6-8）

- [ ] GitHub Issuesの確認・対応
  - 1日1回チェック
  - 質問には48時間以内に回答

- [ ] 初期Pull Requestのレビュー
  - CLA署名確認
  - コードレビュー
  - 感謝のメッセージ

- [ ] Discordでの質問対応
  - `#oss-support`チャンネル作成
  - コミュニティメンバーも回答参加を促す

**所要時間**: 週2-3時間

#### 5.2 コミュニティ育成（Week 9〜）

- [ ] コントリビューターの感謝
  - READMEに`## Contributors`セクション追加
  - 初期貢献者には特別なDiscordロール

- [ ] Good First Issue ラベル
  - 初心者向けIssueにラベル付け
  - コミュニティ参加を促進

- [ ] 定期的な進捗報告
  - 月1回のDiscord開発報告
  - GitHub Discussionsで議論

**所要時間**: 週1-2時間

#### 5.3 商用ライセンス問い合わせ対応

- [ ] 問い合わせメール対応
  - テンプレート: [02-dual-license-strategy.md](02-dual-license-strategy.md#問い合わせ対応フロー)
  - 48時間以内に返信

- [ ] 問い合わせ記録
  - スプレッドシートで管理
  - 価格決定の参考データ

**所要時間**: 問い合わせ1件あたり30分〜1時間

---

## チェックリスト（公開前）

公開ボタンを押す前に、以下をすべて確認：

### 法的・ライセンス

- [ ] LICENSE（GPLv3）配置済み
- [ ] LICENSE-COMMERCIAL.md 配置済み
- [ ] CLA.md 配置済み
- [ ] THIRD_PARTY_LICENSES.md 最新
- [ ] Steam規約確認完了（承認取得）

### ドキュメント

- [ ] README.md（日本語・英語）完成
- [ ] CONTRIBUTING.md 完成
- [ ] ビルド手順が実際に動作することを確認
- [ ] LICENSE_SELECTION_GUIDE.md 完成

### リポジトリ

- [ ] 機密情報削除確認
- [ ] .gitignore 適切に設定
- [ ] Issue/PRテンプレート配置
- [ ] GitHub Discussions有効化

### コミュニティ

- [ ] Discord告知文作成
- [ ] Steam説明ページ更新文作成
- [ ] FAQ準備

### 技術的

- [ ] ビルドが通ることを確認
- [ ] CI/CD（GitHub Actions）設定（オプション）
- [ ] Releaseタグ準備

---

## リスク管理

### 想定されるリスクと対策

| リスク | 確率 | 影響 | 対策 |
|-------|-----|------|------|
| Steam規約違反 | 中 | 高 | 事前確認必須 |
| 大量Issue/PRで対応不能 | 中 | 中 | Good First Issueで分散 |
| 商用ライセンス問い合わせ殺到 | 低 | 中 | テンプレート準備 |
| セキュリティ脆弱性指摘 | 低 | 高 | 迅速対応・パッチリリース |
| コミュニティトラブル | 低 | 中 | Code of Conduct制定 |

---

## 次のステップ

### ✅ Phase 1（部分完了）

Steam調査とSteamworks SDK確認が完了しました。残りのタスク：

1. **Discordでコミュニティアンケート** ← 次の最優先タスク
   - テンプレート: [Phase 1.3](#13-コミュニティ意見収集)
   - 所要時間: 30分（投稿） + 数日（回答収集）

2. **リポジトリ監査**
   - 機密情報のチェック
   - 詳細: [Phase 1.4](#14-リポジトリ監査)
   - 所要時間: 2-3時間

3. **Steam問い合わせ（任意）**
   - より慎重に進めたい場合のみ
   - テンプレート: [06-steam-research-findings.md](06-steam-research-findings.md#推奨アクション)

### 📋 重要な判断ポイント

**Steamサポート問い合わせをどうするか？**

| 選択肢 | メリット | デメリット | 推奨度 |
|--------|---------|-----------|--------|
| **送信する** | 公式確認、最も安全 | 3-5営業日待ち | ⭐⭐⭐ |
| **送信しない** | 早く進められる、成功事例あり | 公式確認なし | ⭐⭐⭐⭐ |

**調査結果から**: どちらでも問題なし。積極的に進めるなら送信不要。

### この文書の使い方

- [x] Phase 1.1: Steam調査 ✅
- [ ] Phase 1.2: Steam問い合わせ（任意）
- [ ] Phase 1.3: Discordアンケート ← **次はこれ**
- [ ] Phase 1.4: リポジトリ監査
- [ ] Phase 2以降のタスク実行
- [ ] 完了したら✅をつける
- [ ] 問題があればこの文書を更新

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成 |
| 2025-10-24 | 1.1 | Steam調査完了を反映、優先順位更新 |
