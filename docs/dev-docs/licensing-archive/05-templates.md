# テンプレート集

**作成日**: 2025-10-24
**目的**: OSS化で使用する各種テンプレートをまとめて提供

このドキュメントには以下のテンプレートが含まれています：

> ⚠️ **2025-11-13 時点の注記**  
> 商用ライセンス関連テンプレートは歴史的資料として残しています。現在は
> AGPL-3.0 のみで配布しており、問い合わせは Discord
> (https://discord.gg/hdSV4hJR8Y) を窓口とします。

---

## 📋 目次

1. [Steam問い合わせテンプレート](#steam問い合わせ)
2. [商用ライセンス文書](#商用ライセンス文書)
3. [CLA（Contributor License Agreement）](#cla)
4. [README更新案](#readme更新案)
5. [Discord/Steamアナウンス](#discordsteamアナウンス)
6. [商用ライセンス問い合わせ対応](#商用ライセンス問い合わせ対応)

---

## Steam問い合わせ

### 英語版（Steamサポート宛）

```
Subject: Open Source Release and Price Parity Policy Inquiry

Hello Steam Partner Support,

I am the developer of LiveCap (App ID: [YOUR_APP_ID]), currently sold
on Steam for approximately ¥1,980.

I am planning to release the source code under GPL v3 license on GitHub
as an open-source project. Users would be able to build the software
from source for free (requires technical knowledge).

However, the Steam version would continue to be sold with the following
differentiators:

1. Pre-built installer (no compilation required)
2. Official technical support via Steam Community and Discord
3. Automatic updates through Steam
4. Steam achievements and cloud saves
5. Supporting ongoing development

My questions:

1. Would this violate Steam's price parity policy?
2. Do I need to add specific disclosures on the Steam store page?
3. Are there any other compliance requirements I should be aware of?

The source code would be freely available on GitHub, but building from
source requires setting up a development environment (Python, PyTorch,
etc.), which most users would find challenging.

The Steam version provides convenience and official support, similar to
how Red Hat Enterprise Linux is sold despite being based on open-source
CentOS.

I would appreciate your guidance on this matter.

Thank you for your time.

Best regards,
[Your Name]
[Your Studio Name]
App ID: [YOUR_APP_ID]
Email: [your@email.com]
```

---

## 商用ライセンス文書

### LICENSE-COMMERCIAL.md

````markdown
# LiveCap Commercial License

**Copyright (c) 2025 Pine Lab**

This is a commercial license agreement for LiveCap.

---

## 1. Grant of Rights

This license grants the Licensee (you or your organization) the following rights:

- **Use**: Run LiveCap for commercial purposes
- **Modify**: Create derivative works without source code disclosure obligations
- **Distribute**: Include LiveCap in your products (subject to restrictions below)
- **Sublicense**: Distribute modified versions under your own terms

## 2. Restrictions

You may NOT:

- Resell LiveCap as a standalone product
- Use the "LiveCap" trademark without permission
- Remove or modify copyright notices
- Compete directly with the official LiveCap product

## 3. License Fee

**Pricing**: $100 - $500 USD depending on use case
- Startups (<10 employees): $100-$200/year
- Small/Medium Business: $300-$500/year
- Enterprise: $500-$1,000/year
- OEM/Integration: $2,000-$5,000 (one-time)

**Payment Methods**: PayPal, Bank Transfer, or Wise

## 4. License Term

- **Duration**: 1 year from payment date
- **Renewal**: Optional (discounted renewal rate available)
- **Perpetual Use**: You may continue using the licensed version after expiration

## 5. Support

Includes:

- Email support (48-hour response time)
- Bug fix priority
- Feature request consideration

Does NOT include:

- Custom development
- 24/7 support
- On-site assistance

## 6. Updates

- Updates released during the license term are included
- After expiration, continued use of previously licensed versions is permitted
- New versions require license renewal

## 7. Warranty Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## 8. Limitation of Liability

IN NO EVENT SHALL PINE LAB BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE.

## 9. Termination

This license terminates automatically if you violate any terms.
Upon termination, you must cease all use and delete all copies.

## 10. Export Control

This software contains technology subject to U.S. export control laws.
You agree to comply with all applicable export regulations.

---

## How to Obtain This License

**Contact**: [your-email@example.com]

**Information Needed**:
1. Company name and size
2. Intended use case
3. Number of users/installations
4. Any customization requirements

We will respond within 48 hours with a quote.

---

**Pine Lab**
Email: [your-email@example.com]
Website: [your-website.com]
````

---

## CLA

### CLA.md

```markdown
# Contributor License Agreement (CLA)

Thank you for your interest in contributing to LiveCap!

To maintain our dual-licensing model (GPL v3 and Commercial License),
we need contributors to grant us certain rights.

---

## Agreement

By contributing to LiveCap, you agree that:

### 1. Grant of Rights

You grant Pine Lab (Hakase) a **perpetual, worldwide, non-exclusive,
royalty-free, irrevocable** license to:

- Use your contribution under any license (including commercial licenses)
- Modify, reproduce, and distribute your contribution
- Sublicense your contribution to third parties
- Include your contribution in both GPL v3 and Commercial versions

### 2. Ownership

- You **retain copyright ownership** of your contribution
- Your contribution will be credited in the project

### 3. Originality

You confirm that:

- Your contribution is your original work
- You have the legal right to grant this license
- Your contribution does not violate any third-party rights

### 4. No Warranty

Your contribution is provided "as is" without any warranty.

---

## Why is this needed?

LiveCap uses dual licensing:
- **GPL v3** for open-source community
- **Commercial License** for businesses

Without this agreement, we cannot include your contributions in the
commercial version, which funds ongoing development.

---

## How to Sign

### Option 1: Comment on Pull Request

Add this comment to your Pull Request:

```
I have read and agree to the Contributor License Agreement (CLA).
```

### Option 2: Electronic Signature (Recommended)

We use [CLA Assistant](https://cla-assistant.io/) for electronic signatures.
When you create a Pull Request, you'll be prompted to sign automatically.

---

## Questions?

If you have questions about this CLA, please open an issue or contact us
at [your-email@example.com].

Thank you for contributing to LiveCap!
```

---

## README更新案

### 日本語版（README.mdに追加）

```markdown
# LiveCap

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-red.svg)](LICENSE-COMMERCIAL.md)
[![GitHub stars](https://img.shields.io/github/stars/Mega-Gorilla/Live_Cap_v3)](https://github.com/Mega-Gorilla/Live_Cap_v3)
[![Platform](https://img.shields.io/badge/platform-Windows-blue.svg)](https://github.com/Mega-Gorilla/Live_Cap_v3)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289da.svg)](https://discord.gg/hdSV4hJR8Y)

**複数の最先端音声認識エンジンを搭載した、リアルタイム音声文字起こしツール**

🎉 **オープンソース化しました！**

**対応OS**: Windows 10/11（Windowsのみ動作確認済み）

---

## 📜 ライセンス

LiveCapはデュアルライセンスで提供されています：

### 個人・オープンソース向け

✅ **無料** - [GPL v3](LICENSE)ライセンス
- 個人利用
- 教育・研究利用
- オープンソースプロジェクト
- 改変版はGPL v3で公開

### 企業・商用利用向け

💼 **商用ライセンス** - [詳細はこちら](LICENSE-COMMERCIAL.md)
- 企業の業務利用
- クローズドソース製品への組み込み
- ソースコード非公開での改変
- 料金: $100〜$500（規模に応じて）

**どちらを選ぶべきか？** → [ライセンス選択ガイド](docs/LICENSE_SELECTION_GUIDE.md)

---

## 💖 開発を支援

LiveCapを気に入っていただけましたか？

- ⭐ このリポジトリにStar
- 🐛 バグ報告・機能提案
- 🔧 Pull Requestの送信
- 💰 [Steam版を購入](YOUR_STEAM_LINK)（公式サポート付き）
- 💼 [商用ライセンスの購入](mailto:your-email@example.com)

---

## 🚀 クイックスタート

[既存の内容]

---

## 🤝 コントリビューション

貢献歓迎！[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

**重要**: Pull Requestを送る前に[CLA](CLA.md)への同意が必要です。

---
```

### 英語版（README.en.mdに追加）

```markdown
# LiveCap

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-red.svg)](LICENSE-COMMERCIAL.md)
[![Platform](https://img.shields.io/badge/platform-Windows-blue.svg)](https://github.com/Mega-Gorilla/Live_Cap_v3)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289da.svg)](https://discord.gg/hdSV4hJR8Y)

**Real-time speech-to-text tool with multiple state-of-the-art ASR engines**

🎉 **Now Open Source!**

**Platform**: Windows 10/11 (Windows only - tested)

---

## 📜 Licensing

LiveCap is dual-licensed:

### For Individuals & Open Source

✅ **Free** under [GPL v3](LICENSE)
- Personal use
- Educational/Research use
- Open source projects
- Modifications must be shared under GPL v3

### For Commercial Use

💼 **Commercial License** - [Learn more](LICENSE-COMMERCIAL.md)
- Business/corporate use
- Proprietary software integration
- Closed-source modifications
- Pricing: $100-$500 depending on scale

**Not sure which license?** → [License Selection Guide](docs/LICENSE_SELECTION_GUIDE.md)

---

## 💖 Support Development

Love LiveCap? Support us:

- ⭐ Star this repository
- 🐛 Report bugs / Request features
- 🔧 Submit Pull Requests
- 💰 [Buy on Steam](YOUR_STEAM_LINK) (includes official support)
- 💼 [Purchase Commercial License](mailto:your-email@example.com)

---
```

---

## Discord/Steamアナウンス

### Discord日本語版

```markdown
@everyone

## 🎉 LiveCapオープンソース化のお知らせ

いつもLiveCapをご利用いただき、ありがとうございます。

この度、LiveCapを **GPL v3ライセンスでオープンソース化** することを決定しました。

### なぜオープンソース化するのか？

最近、開発が停滞気味でした。オープンソース化により：

✅ コミュニティの力で更に良いソフトウェアに
✅ より多くの方に使っていただける
✅ 透明性と信頼性の向上
✅ 多様なユースケースへの対応

### GitHubリポジトリ

🔗 [GitHub: LiveCap](https://github.com/Mega-Gorilla/Live_Cap_v3)

- 完全なソースコード公開
- Issue/Pull Request歓迎
- ビルド手順も完備

### Steam版はどうなる？

Steam版は引き続き販売します。以下の価値を提供：

✅ ワンクリックインストール（ビルド不要）
✅ 自動アップデート
✅ 公式サポート（このDiscordでの優先対応）
✅ Steam実績・クラウド保存
✅ 継続的な開発への支援

GitHub版は無料ですが、Python環境のセットアップやビルドが必要です。

### 既存購入者の皆様へ

早期からのご支援に心より感謝申し上げます。

特典として、Discord内で「Early Supporter」ロールを付与いたします。
引き続き、最高のサポートを提供してまいります。

### 企業の皆様へ

個人利用は無料のGPL版で問題ありませんが、企業の商用利用には
別途商用ライセンスが必要です。

詳細: [LICENSE-COMMERCIAL.md](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/LICENSE-COMMERCIAL.md)
お問い合わせ: your-email@example.com

### 今後の展開

- コミュニティからの機能要望を積極的に取り入れます
- 開発ロードマップをGitHub Projectsで公開します
- 貢献者への感謝を形にします（クレジット掲載等）

ご質問やご意見があれば、お気軽にお寄せください！

---

開発者: Hakase (Pine Lab)
GitHub: https://github.com/Mega-Gorilla/Live_Cap_v3
Discord: https://discord.gg/hdSV4hJR8Y
メール: your-email@example.com
```

### Discord英語版

```markdown
@everyone

## 🎉 LiveCap is Now Open Source!

We're excited to announce that LiveCap is now **open source under GPL v3**!

### Why Open Source?

Development has slowed recently. By going open source, we aim to:

✅ Accelerate development with community contributions
✅ Reach more users worldwide
✅ Increase transparency and trust
✅ Support diverse use cases

### GitHub Repository

🔗 [GitHub: LiveCap](https://github.com/Mega-Gorilla/Live_Cap_v3)

- Full source code available
- Issues and Pull Requests welcome
- Complete build instructions included

### What About Steam Version?

The Steam version continues to be sold, offering:

✅ One-click installer (no build required)
✅ Automatic updates
✅ Official support (priority on this Discord)
✅ Steam achievements & cloud saves
✅ Supporting ongoing development

GitHub version is free but requires building from source.

### For Early Supporters

Thank you for your early support!

You'll receive an "Early Supporter" role on this Discord as a token
of our appreciation.

### For Businesses

Personal use is free under GPL v3, but commercial use by organizations
requires a separate commercial license.

Details: [LICENSE-COMMERCIAL.md](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/LICENSE-COMMERCIAL.md)
Contact: your-email@example.com

### What's Next?

- Feature requests from the community will be prioritized
- Development roadmap will be public on GitHub Projects
- Contributors will be recognized (credits, etc.)

Questions? Feel free to ask!

---

Developer: Hakase (Pine Lab)
GitHub: https://github.com/Mega-Gorilla/Live_Cap_v3
Discord: https://discord.gg/hdSV4hJR8Y
Email: your-email@example.com
```

---

## 商用ライセンス問い合わせ対応

### 初回返信テンプレート（英語）

```
Subject: Re: LiveCap Commercial License Inquiry

Hello [Name],

Thank you for your interest in LiveCap's commercial licensing.

To provide you with an appropriate quote, could you please share:

1. **Company Information**
   - Company name and size (number of employees)
   - Industry/business type

2. **Intended Use**
   - How will LiveCap be used? (internal tool, customer-facing product, etc.)
   - Will it be integrated into another product?

3. **Scale**
   - Number of users/installations
   - Estimated usage volume

4. **Customization Needs**
   - Any specific feature requirements?
   - Technical support level needed?

Our commercial licensing typically ranges from $100-$500/year depending
on the scope. Larger integrations or OEM arrangements are quoted separately.

Looking forward to hearing from you.

Best regards,
[Your Name]
Pine Lab
Email: your-email@example.com
GitHub: https://github.com/Mega-Gorilla/Live_Cap_v3
```

### 見積もり提示テンプレート（英語）

```
Subject: LiveCap Commercial License - Quote for [Company Name]

Hello [Name],

Thank you for providing the details. Based on your requirements:

**Use Case**: [Describe their use case]
**Company Size**: [Number] employees
**Users**: [Number] users/installations

---

## Commercial License Quote

**License Fee**: $[Amount] USD / year

**Includes**:
- Commercial use rights (no GPL obligations)
- Closed-source modifications permitted
- Email support (48-hour response time)
- Bug fix priority
- Updates for 1 year

**Payment**: PayPal, Bank Transfer, or Wise

**License Agreement**: [LICENSE-COMMERCIAL.md](link)

---

## Next Steps

1. Review the license agreement
2. Confirm acceptance
3. Proceed with payment
4. Receive license confirmation (we trust you, no DRM)

**Renewal**: Optional after 1 year (discounted rate: [X]% off)

**Questions?** Feel free to ask.

Best regards,
[Your Name]
Pine Lab
```

### 日本語版（初回返信）

```
件名: Re: LiveCap商用ライセンスについて

[お名前]様

お問い合わせいただき、ありがとうございます。

適切なお見積もりを提供させていただくため、以下の情報を
お教えいただけますでしょうか：

1. **企業情報**
   - 企業名と規模（従業員数）
   - 業種

2. **利用用途**
   - LiveCapをどのように使用されますか？（社内ツール、顧客向け製品など）
   - 他の製品に組み込みますか？

3. **規模**
   - 利用ユーザー数・インストール数
   - 想定利用量

4. **カスタマイズ**
   - 特別な機能要件はありますか？
   - 必要なサポートレベルは？

商用ライセンスは通常$100〜$500/年（規模に応じて）です。
大規模統合やOEM契約は個別にお見積もりいたします。

ご返信お待ちしております。

よろしくお願いいたします。

[あなたの名前]
Pine Lab
メール: your-email@example.com
GitHub: https://github.com/Mega-Gorilla/Live_Cap_v3
```

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-10-24 | 1.0 | 初版作成 |

---

## 使用方法

1. **必要なテンプレートをコピー**
2. **[YOUR_*, your-*]などのプレースホルダーを置換**
3. **内容を状況に合わせて調整**
4. **使用前に最終確認**

---

これですべてのテンプレートが揃いました！
必要に応じてカスタマイズしてご利用ください。
