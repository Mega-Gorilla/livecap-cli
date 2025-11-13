# Steam版との整合性

**作成日**: 2025-10-24
**重要度**: 最高（既存ビジネスへの影響）
**必須確認事項**: Steam価格平等性ポリシーの確認が必要

---

## 📋 目次

1. [課題の概要](#課題の概要)
2. [Steam規約の確認ポイント](#steam規約の確認ポイント)
3. [対応オプション](#対応オプション)
4. [既存購入者への対応](#既存購入者への対応)
5. [実装推奨案](#実装推奨案)

---

## 課題の概要

### 現状

- LiveCapは現在Steam上で有料販売中（~¥1,980想定）
- 購入者: 350人
- 売上実績あり

### OSS化による問題

**GitHubで無料公開** → Steamで有料販売は矛盾？

Steam価格平等性ポリシー（Price Parity Policy）:
> 他のPCプラットフォームでSteam版より大幅に安く販売してはいけない

**疑問**:
- GitHubで無料公開 = Steam規約違反？
- 既存購入者から返金要求？
- 新規購入者の減少？

---

## Steam規約の確認ポイント

### 必須: Steamサポートへの問い合わせ

**問い合わせ文例**（英語）:

```
Subject: Open Source Release and Price Parity Policy Inquiry

Hello Steam Partner Support,

I am the developer of [LiveCap App Name] (App ID: XXXXXX), currently
sold on Steam for [¥1,980].

I am planning to release the source code under GPL v3 license on GitHub
as an open-source project. Users would be able to build the software
from source for free.

However, the Steam version would continue to be sold with the following
differentiators:

1. Pre-built installer (no compilation needed)
2. Official technical support via Steam Community
3. Automatic updates through Steam
4. Supporting ongoing development
5. Optional: Steam Workshop integration

My questions:

1. Would this violate Steam's price parity policy?
2. Do I need to add specific disclosures on the Steam store page?
3. Are there any other compliance requirements?

The source code would be freely available but require technical knowledge
to build. Most users would prefer the convenience of the Steam version.

Thank you for your guidance.

Best regards,
[Your Name]
[Your Studio Name]
App ID: XXXXXX
```

### 確認すべき規約セクション

1. **Steamworks Documentation > Distribution**
   - "Price Parity Policy"のセクション

2. **Steam Distribution Agreement**
   - 契約書の価格関連条項

3. **Steam Partner Forum**
   - 他の開発者のOSS化事例

---

## 対応オプション

3つの主要オプションを検討：

### Option 1: プレミアムエディション方式（推奨）

**モデル**:
```
GitHub版（無料・GPL v3）
├─ ソースコードから自分でビルド
├─ コミュニティサポートのみ
├─ 手動アップデート
└─ 技術知識必要

Steam版（有料継続・¥1,980）
├─ ワンクリックインストール
├─ 公式サポート（Discord優先対応）
├─ 自動アップデート
├─ Steam実績・クラウド保存
├─ 開発支援の意味合い
└─ 技術知識不要
```

#### Steam説明ページの文言

**日本語版**:
```markdown
## オープンソースプロジェクト

LiveCapはGPL v3ライセンスのオープンソースソフトウェアです。

GitHub: [リンク]

### Steam版の価値

このSteam版には以下が含まれます：

✅ すぐに使えるインストーラー（ビルド不要）
✅ 自動アップデート
✅ 公式技術サポート（Discord優先対応）
✅ Steam実績・クラウド保存
✅ 継続的な開発への支援

GitHubから無料でソースコードを入手してビルドすることも可能ですが、
Steam版の購入は開発継続のサポートになります。

### 商用利用について

個人利用はGitHub版で無料ですが、企業の商用利用には別途
商用ライセンスが必要です。詳細はGitHubページをご覧ください。
```

**英語版**:
```markdown
## Open Source Project

LiveCap is open source software licensed under GPL v3.

GitHub: [link]

### What Steam Version Offers

Your purchase includes:

✅ Ready-to-use installer (no build required)
✅ Automatic updates via Steam
✅ Official technical support (priority Discord access)
✅ Steam achievements & cloud saves
✅ Supporting continued development

You can build from source for free on GitHub, but purchasing on Steam
helps sustain ongoing development.

### Commercial Use

Free for personal use via GitHub. Commercial use by organizations
requires a separate commercial license. See GitHub for details.
```

#### Steam規約適合性

✅ **価格差別ではない理由**:
- GitHub版: ビルド必要（技術的障壁）
- Steam版: 便利さ + サポート + 開発支援

類似事例: Blender, Godot Engine（寄付モデル）

#### ✅ メリット

1. Steam規約違反リスク最小
2. 既存購入者への説明が容易
3. 新規購入者も納得しやすい
4. 収益維持（一部）

#### ⚠️ デメリット

1. Steamサポート確認が必須
2. 新規購入者は減少見込み
3. サポート負荷は継続

---

### Option 2: Steam版を無料化（最も安全）

**モデル**:
```
Steam版（無料化）
├─ 基本ソフトは無料ダウンロード
├─ GitHub版と同等機能
└─ Steam配信の利便性

DLC: 開発者サポートパック（¥500-1,000）
├─ 機能追加なし（純粋な寄付）
├─ 特典: Discordロール、名前クレジット
└─ 「開発を応援」的位置付け
```

#### 既存購入者への対応

1. **DLC無料付与**
   - 既存購入者全員にDLC無料配布
   - Steamの機能で自動付与可能

2. **または返金対応**
   - 希望者には返金
   - Steam返金ポリシー（2週間以内、2時間未満プレイ）外でも個別対応

3. **お詫びアナウンス**
```markdown
## 重要なお知らせ

LiveCapをオープンソース化することを決定いたしました。

既存購入者の皆様には以下の対応をいたします：

1. 「開発者サポートパック」DLCを無料付与
2. ご希望の方には返金対応

オープンソース化により、より多くの方に使っていただけること、
そしてコミュニティの力で更に良いソフトウェアになることを
期待しています。

これまでのご支援、誠にありがとうございました。
```

#### ✅ メリット

1. Steam規約違反リスクゼロ
2. OSSコミュニティに好印象
3. ユーザー数の大幅増加

#### ❌ デメリット

1. 収益はほぼゼロ（寄付のみ）
2. 既存購入者への説明・対応コスト
3. 返金対応の手間（数十件想定）

---

### Option 3: Steam版を独立製品として維持

**モデル**:
```
GitHub版（GPL・無料）
├─ 最新の開発版
├─ 不安定な可能性
├─ 自分でビルド
└─ コミュニティサポート

Steam版（有料・別物として）
├─ 安定版のみ
├─ 独自の商用ライセンス付き
├─ 公式サポート
└─ 追加機能（Steam連携等）
```

#### ⚠️ リスク

**Steam規約違反の可能性が高い**:
- 「同じソフトを他で無料配布」と見なされるリスク
- 差別化が不十分と判断される可能性

#### このオプションを選ぶ条件

- Steamサポートから明確な承認を得た場合のみ
- GitHub版とSteam版の機能差が明確
- Steam独自機能（Workshop, 実績等）を大幅追加

---

## 既存購入者への対応

### 原則

1. **誠実な説明**
   - OSS化の理由を正直に伝える
   - コミュニティ貢献と開発加速が目的

2. **感謝の表明**
   - 初期支援への感謝
   - 既存ユーザーの重要性を強調

3. **納得できる対応**
   - 返金 or 特典付与 or 説明のみ

### 対応オプション別の説明

#### Option 1の場合（プレミアム化）

**Discord / Steamアナウンス**:
```markdown
## LiveCapオープンソース化のお知らせ

いつもLiveCapをご利用いただきありがとうございます。

LiveCapをGPL v3ライセンスでオープンソース化することを決定しました。

### なぜオープンソース化？

- コミュニティの力で更に良いソフトウェアに
- より多くの方に使っていただきたい
- 透明性と信頼性の向上

### Steam版はどうなる？

Steam版は引き続き有料で提供します。以下の価値を提供：

✅ ワンクリックインストール
✅ 自動アップデート
✅ 公式サポート
✅ 開発支援

GitHub版は無料ですが、自分でビルドする必要があります。

### 既存購入者の皆様へ

皆様の早期からのご支援に心より感謝申し上げます。
引き続き最高のサポートを提供してまいります。

特典として、Discordで「Early Supporter」ロールを付与いたします。

ご質問があればお気軽にお問い合わせください。
```

#### Option 2の場合（無料化）

**Discord / Steamアナウンス**:
```markdown
## LiveCap無料化とオープンソース化のお知らせ

LiveCapをオープンソース化し、Steam版も無料にすることを決定しました。

### 既存購入者の皆様へ

早期からのご支援に心より感謝申し上げます。
以下の対応をさせていただきます：

1. 「開発者サポートパック」DLCを無料付与
   - 特別なDiscordロール
   - クレジットへのお名前掲載

2. ご希望の方には返金対応
   - Steamサポートチケット経由でご連絡ください

皆様のおかげでここまで開発を続けることができました。
本当にありがとうございました。

オープンソース化により、更に多くの方に使っていただけることを
楽しみにしています。
```

### 想定される質問への回答

**Q: なぜ今更無料にするのか？**
A: より多くの人に使ってもらい、コミュニティの力で開発を加速させたいからです。

**Q: 返金してもらえるのか？**
A: [Option 2の場合] はい、ご希望の方には対応します。
A: [Option 1の場合] Steam版は引き続き価値を提供しますので、返金は予定していません。ただし、ご事情がある場合は個別にご相談ください。

**Q: Steam版を買った意味がなくなるのでは？**
A: [Option 1] Steam版は便利さとサポート付きです。GitHub版はビルドが必要です。
A: [Option 2] 皆様の支援のおかげでここまで開発できました。特別DLCで感謝を表します。

**Q: 開発は継続されるのか？**
A: はい、むしろコミュニティの貢献で加速します。

---

## 実装推奨案

### 推奨: Option 1（プレミアムエディション方式）

**理由**:
1. ✅ Steam規約違反リスクが低い
2. ✅ 既存購入者への説明が容易
3. ✅ 一定の収益維持（開発継続に必要）
4. ✅ 実装が最もシンプル

### 実装ステップ

#### Phase 1: Steam確認（Week 1-2）

1. **Steamサポートに問い合わせ**
   - 上記の文例を使用
   - 回答を待つ（通常3-5営業日）

2. **回答に基づいて最終決定**
   - 承認 → Option 1実行
   - 条件付き承認 → 条件に合わせて調整
   - 不承認 → Option 2検討

#### Phase 2: コミュニティ告知（Week 3）

1. **Discord先行告知**
   - OSS化の意向を伝える
   - フィードバックを収集
   - 既存ユーザーの反応を確認

2. **Steam アップデート告知**
   - 「近日重要なお知らせ」として予告
   - 期待感を醸成

#### Phase 3: 公開実施（Week 4-5）

1. **GitHubリポジトリ公開**
2. **Steam説明ページ更新**
   - Option 1の文言を追加
3. **Discord / Steam コミュニティ投稿**
   - 正式アナウンス

#### Phase 4: フォローアップ（Week 6〜）

1. **質問対応**
2. **必要に応じて説明文の調整**
3. **コミュニティのフィードバック収集**

---

## リスク管理

### リスク1: Steam規約違反

**確率**: 中（確認必須）
**影響**: 高（Steamから削除される可能性）
**対策**:
- ✅ 事前にSteamサポート確認（必須）
- ✅ 承認なしに実行しない

### リスク2: 既存購入者からの批判

**確率**: 低-中
**影響**: 中（レビュー評価低下）
**対策**:
- ✅ 誠実な説明
- ✅ 感謝の表明
- ✅ 特典付与（Discord role等）
- ✅ 個別の懸念に丁寧に対応

### リスク3: 新規購入者の減少

**確率**: 高
**影響**: 中（収益減）
**対策**:
- ✅ Steam版の価値を明確化
- ✅ サポート強化
- ⚠️ 収益減は受け入れ（目的が収益ではないため）

### リスク4: 返金要求の殺到

**確率**: 低（Option 1の場合）
**影響**: 低-中
**対策**:
- ✅ 事前に説明を徹底
- ✅ 返金ポリシーを明確化
- ✅ 予算確保（最大50件 × ¥1,980 = 約¥100,000）

---

## 次のステップ

1. ✅ このSteam対応戦略を確認
2. **最重要**: Steamサポートに問い合わせ送信
3. → [04-implementation-roadmap.md](04-implementation-roadmap.md) で全体スケジュール確認
4. → [05-templates.md](05-templates.md) で具体的な文例を確認

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成 |
