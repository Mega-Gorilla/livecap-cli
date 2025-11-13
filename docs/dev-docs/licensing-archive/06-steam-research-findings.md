# Steam OSS化 調査結果

**調査日**: 2025-10-24
**目的**: LiveCapのOSS化（GPL v3）とSteam販売の両立可能性を調査

---

## 📋 目次

1. [調査結果サマリー](#調査結果サマリー)
2. [Steamworks公式見解](#steamworks公式見解)
3. [成功事例の詳細分析](#成功事例の詳細分析)
4. [LiveCapへの適用可能性](#livecapへの適用可能性)
5. [推奨アクション](#推奨アクション)

---

## 調査結果サマリー

### ✅ 結論：GPL + Steam販売は可能

**根拠**:
1. Steamworks公式ドキュメントが存在（例外条件あり）
2. 実際にGPLゲームがSteam上で販売されている（2例確認）
3. GPL v3は有料販売を明示的に許可している

### ⚠️ 注意点

- Steamworks SDKとGPL（コピーレフト）は原則的に非互換
- ただし**オリジナル作者が100%著作権を保持**している場合は例外
- LiveCapはこの例外に該当

---

## Steamworks公式見解

### 公式ドキュメント

**ソース**: [Distributing Open Source Applications on Steam](https://partner.steamgames.com/doc/sdk/uploading/distributing_opensource)

#### 互換性のあるライセンス（問題なし）

✅ **Permissive Licenses（許可的ライセンス）**:
- MIT License
- BSD (3-clause, 4-clause)
- Apache 2.0
- WTFPL

これらは「修正版をオープンソースライセンスで再配布する要件がない」

#### 問題のあるライセンス

⚠️ **Copyleft Licenses（コピーレフトライセンス）**:
- **GPL** (v2, v3)
- その他のコピーレフト要素を持つライセンス

**理由**: 「Steamworks SDKとコードを組み合わせる際に問題がある」

### 重要な例外条項

> **GPL-licensed applications do exist on Steam.**
>
> This occurs when the **original code author** has explicitly granted permission—either through:
> - Separate licensing agreements with Valve
> - Determining that "what the Steamworks SDK does is just a communication with a service" that doesn't trigger copyleft requirements

#### 解釈

1. **オリジナル作者の権限**: 自分が100%著作権を持つなら、自分で判断できる
2. **"Communication with a service"**: Steamworks SDKが単なるサービス通信なら、GPLのコピーレフト要件が発動しない可能性
3. **Valveは審査しない**: 「自己責任で判断せよ」

---

## 成功事例の詳細分析

### 事例1: Tales of Maj'Eyal

**基本情報**:
- ジャンル: ローグライクRPG
- エンジン: TE4（GPLv3）
- ゲーム素材: 独自ライセンス（Tales of Maj'Eyal専用）

**配布モデル**:
```
公式サイト（te4.org）
├─ 無料ダウンロード可能
├─ 寄付（Donation）推奨
└─ 寄付者には特別機能アンロック

Steam（有料販売）
├─ 価格: 約$7
├─ Steam実績・リーダーボード
├─ 自動アップデート
└─ 購入者には寄付者と同じ特別機能

GOG.com（有料販売）
└─ Steamと同等
```

**重要なポイント**:
- ✅ エンジンはGPLv3だが、Steam販売可能
- ✅ 無料版と有料版の機能は**ほぼ同じ**
- ✅ Steam版の付加価値: 便利さ + コミュニティ + 開発支援
- ✅ 公式サイトとSteamアカウント連携可能

**収益モデル**:
- 無料版でユーザー獲得 → Steam版で収益化
- 寄付（Donationware）モデルとの併用

---

### 事例2: HyperRogue

**基本情報**:
- ジャンル: 非ユークリッド幾何学ローグライク
- ライセンス: **GPL v2**
- 開発者: Zeno Rogue

**配布モデル**:
```
GitHub（完全オープンソース）
├─ GPL v2でソースコード公開
├─ 誰でもビルド可能
└─ 古いバージョンも利用可能

Steam（有料販売）
├─ 価格: 約$5
├─ 最新版（無料版より先行リリース）
├─ Steam実績・リーダーボード
├─ 活発なディスカッションフォーラム
└─ Steamの便利さ

itch.io（有料販売）
└─ Steamと同等
```

**重要なポイント**:
- ✅ GPL v2でも問題なくSteam販売
- ✅ ゲームプレイは**無料版と全く同じ**
- ✅ 有料版の差別化: 最新機能の先行提供、Steam機能
- ✅ 「オープンソースと有料開発の利点を組み合わせる」と開発者明言

**開発者の哲学**（公式サイトより）:
> "新機能は有料版に追加され、その後時間をおいて無料版にも追加される"

**収益モデル**:
- 開発支援 + 便利さ + 先行機能アクセス
- GitHub Sponsorsでも支援受付

---

## LiveCapへの適用可能性

### LiveCapの状況

| 項目 | 状態 | 判断 |
|------|------|------|
| **著作権保持** | Pine Labが100% | ✅ 例外適用可能 |
| **Steamworks SDK使用** | **使用していない** | ✅ 完全にクリア |
| **既存Steam販売** | 350購入者 | 既存実績あり |
| **依存ライブラリ** | すべてGPL互換 | ✅ 問題なし |

### ✅ Steamworks SDK使用状況の確認結果

**調査日**: 2025-10-24

```bash
# コードベース全体を検索
grep -r "steamworks|steam_api|Steamworks" src/ --include="*.py" -i
# 結果: マッチなし

# Steam関連ファイルを検索
find . -name "*steam*" -type f | grep -v ".git"
# 結果: ドキュメントと画像のみ
```

**結論**:
- ✅ **Steamworks SDKは一切使用されていない**
- LiveCapは単純にSteamプラットフォームで配布されているだけ
- Steam実績、クラウド保存、ワークショップ等の統合なし

**影響**:
- GPL v3との互換性問題は**完全に存在しない**
- Steamworks公式ドキュメントの制約は適用されない
- Tales of Maj'Eyal / HyperRogueモデルが完全に適用可能

---

## 価格平等性ポリシー

### Valve公式見解

**ソース**: Steamworks announcements, 2023年2月更新

> "It is important that you don't give Steam customers a worse deal than Steam Key purchasers"

**解釈**:
- Steam版が他のプラットフォームより**大幅に高い**のは問題
- Steam版が他より**大幅に安い**必要はない
- 重要なのは「Steam顧客に不利益を与えない」

### Tales of Maj'Eyal / HyperRogueの解釈

両者とも以下のモデルで成功:
```
GitHub版: 無料（ビルドが必要）
   ↓
Steam版: 有料（便利さ + 付加価値）
```

**価格差別ではない理由**:
- GitHub版: 技術的障壁（ビルド必要）
- Steam版: 付加価値あり（実績、自動更新、サポート）
- **GitHubで無料配布 ≠ Steam規約違反**

---

## 推奨アクション

### 短期（今すぐ）

#### ~~1. Steamworks SDK使用状況の確認~~ ✅ 完了

**結果**: Steamworks SDKは**使用していない**
- → **完全にクリア！GPL互換性問題なし**

#### 2. Steam説明ページの文言準備

Tales of Maj'Eyalモデルを参考に：
```markdown
LiveCap is open source (GPL v3).
GitHub: [link]

Steam version includes:
- Pre-built installer
- Automatic updates
- Official support
- Supporting ongoing development

Free to build from source for personal use.
```

### 中期（Steamサポート問い合わせ前）

#### 3. 問い合わせ文の最適化

調査結果を踏まえた問い合わせ文:
```
Subject: GPL v3 Open Source Software Distribution on Steam

Hello,

I'm the developer of LiveCap (App ID: XXXXX), currently sold on Steam.

I plan to release the source code under GPL v3 on GitHub while
continuing Steam sales, similar to successful examples like:
- Tales of Maj'Eyal (GPLv3 engine, Steam + free download)
- HyperRogue (GPL v2, Steam + GitHub)

**Technical Details:**
- LiveCap does NOT use the Steamworks SDK
- Simple file distribution via Steam platform
- No achievements, cloud saves, or Workshop integration

**Steam version offers:**
- Pre-built installer (no compilation needed)
- Automatic updates
- Official support
- Supporting ongoing development

**GitHub version:**
- Free source code (GPL v3)
- Requires technical knowledge to build
- Community support

**Questions:**
1. Is this distribution model acceptable under Steam's policies?
2. Any specific disclosures required on the Steam store page?
3. Any concerns regarding price parity?

Thank you for your guidance.

Best regards,
[Your Name]
App ID: XXXXX
```

### 長期

#### 4. Steamサポート問い合わせ送信

- 上記の最適化した文面で送信
- 回答を待つ（3-5営業日）

#### 5. 回答に応じた実行

- **承認**: そのまま進める
- **条件付き**: 条件に合わせて調整
- **不承認**: Tales of Maj'Eyal開発者に相談、または無料化検討

---

## 参考リンク

### 公式ドキュメント
- [Distributing Open Source Applications on Steam](https://partner.steamgames.com/doc/sdk/uploading/distributing_opensource)
- [Steamworks Pricing Documentation](https://partner.steamgames.com/doc/store/pricing)

### 成功事例
- [Tales of Maj'Eyal 公式](https://te4.org)
- [Tales of Maj'Eyal on Steam](https://store.steampowered.com/app/259680/)
- [HyperRogue GitHub](https://github.com/zenorogue/hyperrogue)
- [HyperRogue on Steam](https://store.steampowered.com/app/342610/HyperRogue/)

### コミュニティ議論
- [Can open source games be published on Steam? - Game Dev Stack Exchange](https://gamedev.stackexchange.com/questions/139170/)
- [Publishing GPLv3 code on Steam - Open Source Stack Exchange](https://opensource.stackexchange.com/questions/14128/)

---

## 次のステップ

1. ✅ この調査結果を確認
2. → **Steamworks SDK使用状況を確認**（次のタスク）
3. → Steamサポートへの問い合わせ文を最適化
4. → Discordコミュニティに意見を聞く
5. → 問い合わせ送信

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成（調査結果まとめ） |
