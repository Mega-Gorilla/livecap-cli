# デュアルライセンス戦略の詳細

**作成日**: 2025-10-24
**対象**: LiveCap OSS化におけるデュアルライセンスの運用方法
**戦略**: シンプルデュアルライセンス（GPL v3 + 商用ライセンス）

> ⚠️ **2025-11-13 時点の注記**  
> 本ドキュメントは将来的にデュアルライセンスを導入する場合の参考資料です。
> 現在の LiveCap Core は AGPL-3.0 のみで配布しており、最新の手順は
> [`docs/dev-docs/licensing/license-checklist.md`](../licensing/license-checklist.md) を参照してください。

---

## 📋 目次

1. [デュアルライセンスとは](#デュアルライセンスとは)
2. [LiveCapの実装モデル](#livecapの実装モデル)
3. [ライセンス選択ガイド](#ライセンス選択ガイド)
4. [商用ライセンスの運用](#商用ライセンスの運用)
5. [CLA（Contributor License Agreement）](#cla-contributor-license-agreement)
6. [よくある質問](#よくある質問)

---

## デュアルライセンスとは

### 基本原理

同じソフトウェアを**2つの異なるライセンスで提供**する戦略です。

```
         LiveCap（Pine Labが著作権保持）
              /                    \
    GPL v3ライセンス版          商用ライセンス版
    （無料・OSS）              （有料・独自条件）
         ↓                           ↓
   個人・OSS開発者               企業・クローズド製品
   改変版も公開必須             非公開利用OK
```

### なぜ可能か

- **あなたが100%の著作権を保持**しているから
- 自分のコードは自分が好きなライセンスで提供できる
- GPL版を使う人 → GPLルールに従う
- 商用ライセンスを買う人 → あなたの定める条件に従う

### 成功事例

#### Qt Framework
```
GPL版（無料）
  ↓ オープンソースアプリ開発に利用

商用ライセンス（$5,000+/年）
  ↓ Adobe、Autodeskなどが採用
```

#### MySQL
```
GPL版（無料）
  ↓ Webアプリ、個人利用

商用ライセンス（Oracle販売）
  ↓ 商用製品に組み込み
```

---

## LiveCapの実装モデル

### シンプルデュアルライセンス方式

LiveCapの状況（副業開発、初期投資ゼロ、コミュニティ重視）に最適化した**軽量モデル**：

```
LiveCap
├─ GPL v3版（無料・オープンソース）
│  ├─ GitHub完全公開
│  ├─ 個人・教育・OSS利用は完全無料
│  ├─ ソースからビルド
│  ├─ コミュニティサポート
│  ├─ 改変・再配布自由（GPL v3準拠なら）
│  └─ 商用利用は要問い合わせ
│
└─ 商用ライセンス（有料・個別対応）
   ├─ 企業の商用利用向け
   ├─ メールで問い合わせ（自動販売なし）
   ├─ ケースバイケースで柔軟対応
   ├─ 料金: $100〜$500（規模に応じて）
   ├─ ソースコード非公開OK
   ├─ 独自ブランディングOK
   └─ メールサポート含む
```

### なぜこのモデルか

#### ✅ メリット

**1. 初期投資ゼロ**
- 自動販売システム不要
- 決済プラットフォーム不要
- 弁護士レビュー不要（シンプル文書使用）

**2. 管理負荷が最小**
- 商用問い合わせは月1-2件程度と予想
- 手動対応で十分
- 時間的制約に適合

**3. 柔軟な対応**
- 企業規模に応じた価格設定
- 特殊ケースに個別対応
- 長期的な関係構築

**4. ただ乗り防止**
- 企業の無断商用利用を抑止
- GPL義務（ソース公開）が心理的障壁
- 商用ライセンスへの誘導

#### ❌ デメリット・制約

**1. スケーラビリティ**
- 大量の商用利用者には不向き
- 手動対応の限界（月10件超えたら要見直し）

**2. 問い合わせ対応**
- メール対応の手間（週1時間程度）
- 英語対応必要

**3. 自動化なし**
- 即時購入不可（問い合わせ必須）
- 一部の企業には不便

---

## ライセンス選択ガイド

### あなたに商用ライセンスが必要か？

#### ✅ GPL v3で問題ない場合（無料）

以下のすべてに該当する場合：

- [ ] 個人として使用
- [ ] 教育・研究目的
- [ ] オープンソースプロジェクト（GPL互換）
- [ ] 改変版をGPL v3で公開できる
- [ ] 商用サービスではない

**例:**
- 個人配信者がOBSで字幕表示
- 大学の研究プロジェクト
- オープンソースの配信ツール開発
- 個人VRChatイベントでの使用

#### 💼 商用ライセンスが必要な場合（有料）

以下のいずれかに該当する場合：

- [ ] 企業・法人として使用
- [ ] 商用サービスの一部として利用
- [ ] クローズドソースの製品に組み込み
- [ ] LiveCapを改変して非公開にしたい
- [ ] 独自ブランディングで再配布

**例:**
- 企業の社内会議で使用
- 配信プラットフォームへの統合
- VRChat イベント会社での使用
- 教育サービスへの組み込み
- 企業のカスタマーサポートツール

#### ❓ 不明な場合

以下の質問で判断：

| 質問 | YES → 商用必要 | NO → GPL可 |
|------|---------------|-----------|
| 会社・法人として使いますか？ | 商用 | GPL |
| 顧客に課金するサービスで使いますか？ | 商用 | GPL |
| 改変版を公開したくないですか？ | 商用 | GPL |
| 社内ツールとして使いますか？ | 商用 | GPL |
| 営利目的ですか？ | 商用 | GPL |

**それでも不明なら**: メールで問い合わせ（無料相談）

### GPL v3の具体的な義務

GPL v3版を使う場合の義務：

1. **ソースコード公開**
   - LiveCapを改変したらソースコードを公開
   - 同じGPL v3で配布

2. **ライセンス継承**
   - LiveCapを組み込んだソフトもGPL v3になる
   - クローズドソースにできない

3. **著作権表示**
   - オリジナルの著作権表示を保持
   - 変更点を明記

4. **免責事項**
   - 無保証であることを明記

**重要**: 個人利用のみで配布しない場合、ソースコード公開義務はありません。

---

## 商用ライセンスの運用

### 価格設定（目安）

| 利用ケース | 推奨価格 | 期間 | サポート |
|-----------|---------|------|---------|
| **スタートアップ** | $100-$200 | 1年 | メール |
| **中小企業** | $300-$500 | 1年 | メール |
| **大企業** | $500-$1,000 | 1年 | メール+優先対応 |
| **OEM/統合** | $2,000-$5,000 | 買い切り | カスタム |

**柔軟に対応**:
- 非営利団体: 割引検討
- 長期契約: 複数年割引
- 教育機関: ケースバイケース

### 問い合わせ対応フロー

#### ステップ1: 初回問い合わせ受信

**テンプレート返信**:
```markdown
Subject: Re: LiveCap Commercial License Inquiry

Hello,

Thank you for your interest in LiveCap commercial licensing.

To provide you with an appropriate quote, could you please provide:

1. Company name and size
2. Intended use case (internal tool, product integration, etc.)
3. Number of users/installations
4. Any customization requirements

Our commercial licensing typically ranges from $100-$500/year depending
on the scope.

Best regards,
[Your Name]
Pine Lab
```

#### ステップ2: 見積もり提示

**考慮要素**:
- 企業規模（従業員数）
- 利用用途（社内 vs 顧客向け）
- ユーザー数
- 売上規模（推定）
- カスタマイズ要否

**見積もり例**:
```
LiveCap Commercial License - Quote

Customer: [Company Name]
Use Case: Internal meeting transcription
Users: ~50 employees

License Fee: $300/year
Includes:
- Commercial use rights
- No source code disclosure obligation
- Email support (48h response time)
- Updates for 1 year

Payment: PayPal / Bank Transfer
Renewal: Optional (discounted rate)
```

#### ステップ3: 契約・支払い

**簡易プロセス**:
1. 見積もり合意
2. 商用ライセンス文書送付（後述のテンプレート）
3. 支払い確認（PayPal / 銀行振込）
4. ライセンスキー発行（オプション）

**ライセンスキー**（オプション）:
- 必須ではない（信頼ベース）
- 技術的制限なし（DRM不要）
- 記録用途のみ

#### ステップ4: サポート提供

**メールサポート範囲**:
- ✅ インストール・設定支援
- ✅ バグ報告の優先対応
- ✅ 機能要望の検討
- ❌ カスタム開発（別途相談）
- ❌ 24時間対応（48時間以内返信）

### 支払い方法

**初期（手動）**:
- PayPal: 手数料3.6% + $0.30
- 銀行振込: 海外送金手数料に注意
- Wise (TransferWise): 低手数料

**将来（自動化）**:
- Gumroad: 手数料10%、ライセンスキー自動発行
- Stripe: 手数料2.9% + $0.30、API統合必要
- Paddle: 手数料5%、税務処理込み

---

## CLA (Contributor License Agreement)

### なぜCLAが必要か

デュアルライセンスを維持するには、**すべてのコードの著作権をPine Labが保持**する必要があります。

**問題例**:
```
開発者Aが機能Xを貢献（GPL v3で）
  ↓
Pine Labは機能Xを商用版に含められない！
（開発者Aの許可なしに商用ライセンスで配布不可）
```

### 解決策：CLA

貢献者に「著作権をPine Labに譲渡」または「Pine Labに任意のライセンスで利用する権利を付与」してもらいます。

### シンプルCLA（推奨）

```markdown
# LiveCap Contributor License Agreement (CLA)

By contributing to LiveCap, you agree that:

1. **Grant of Rights**
   You grant Pine Lab (Hakase) a perpetual, worldwide, non-exclusive,
   royalty-free license to:
   - Use your contribution under any license (including commercial)
   - Modify and redistribute your contribution
   - Include it in both GPL and commercial versions of LiveCap

2. **Ownership**
   You retain copyright ownership of your contribution.

3. **Original Work**
   You confirm that your contribution is your original work and you
   have the right to grant this license.

4. **No Warranty**
   Your contribution is provided "as is" without warranty.

This allows LiveCap to maintain dual licensing while respecting
your authorship.

---

To agree: Add a comment to your Pull Request:
"I have read and agree to the Contributor License Agreement."

Or sign electronically at: [CLA Assistant link]
```

### CLA導入方法

#### オプション1: 手動（初期）

1. `CLA.md`をリポジトリルートに配置
2. `CONTRIBUTING.md`でCLAに言及
3. PRレビュー時に確認
4. 貢献者リストで記録

#### オプション2: 自動（推奨）

**CLA Assistant** (https://cla-assistant.io/)
- 無料サービス
- GitHub連携
- PR作成時に自動でCLA同意を求める
- 電子署名記録

**設定手順**:
1. CLA Assistantにサインアップ
2. GitHubリポジトリを連携
3. `CLA.md`のURLを指定
4. 自動化完了

---

## よくある質問

### GPL v3について

**Q: 個人で使うだけなら改変版を公開しなくてもいい？**
A: はい。配布しない限りソースコード公開義務はありません。

**Q: 会社で使うと必ず商用ライセンスが必要？**
A: 社内利用のみで配布しない場合も、企業として使うなら商用ライセンス推奨です。GPLの義務は発生しませんが、ビジネス利用の公平性のため。

**Q: GPLだとOBSで配信に使えない？**
A: 使えます。OBS配信は「配布」ではないのでGPL義務は発生しません。

### 商用ライセンスについて

**Q: 商用ライセンスを買うとソースコードももらえる？**
A: はい、同じソースコードです。ただしソースコード公開義務がなくなります。

**Q: 商用ライセンスは永続？**
A: 基本は1年間。更新は任意（割引あり）。更新しなくても購入時のバージョンは引き続き使用可能。

**Q: 複数プロジェクトで使える？**
A: 見積もり時に相談。基本的には1企業1ライセンスで複数プロジェクト可。

### 技術的な質問

**Q: ライセンスキーやDRMはある？**
A: いいえ。信頼ベースです。技術的制限はありません。

**Q: Steam版と違いは？**
A: 機能は同じ。Steam版は公式サポート付き、GitHub版はコミュニティサポート。

---

## 収益シミュレーション

あくまで参考値（収益は主目的ではない）：

### 保守的予測（1年目）

```
GPL版ユーザー: 5,000人（無料）
商用ライセンス購入:
  - スタートアップ 5社 × $150 = $750
  - 中小企業 2社 × $400 = $800

年間合計: 約 $1,550（約23万円）
```

### 現実的予測（2-3年目）

```
GPL版ユーザー: 20,000人
商用ライセンス購入:
  - スタートアップ 20社 × $150 = $3,000
  - 中小企業 10社 × $400 = $4,000
  - 大企業 3社 × $800 = $2,400

年間合計: 約 $9,400（約140万円）
```

**重要**: この収益は**副次的**。主目的はコミュニティ貢献とただ乗り防止。

---

## 次のステップ

1. ✅ このデュアルライセンス戦略を確認
2. → [03-steam-integration.md](03-steam-integration.md) でSteam対応の詳細を確認
3. → [04-implementation-roadmap.md](04-implementation-roadmap.md) で実装計画を確認
4. → [05-templates.md](05-templates.md) で各種テンプレートを入手

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成 |
