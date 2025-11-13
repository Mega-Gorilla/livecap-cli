# ライセンス出荷チェックリスト

**最終更新**: 2025-11-13
**対象**: LiveCap Core を AGPL-3.0 ライセンスで配布する際の確認作業

> ℹ️ LiveCap Core は AGPL-3.0 で提供されています。連絡・質問は Discord
> コミュニティ (https://discord.gg/hdSV4hJR8Y) を窓口とします。

---

## 🎯 目的

1. AGPL-3.0 の義務を満たした状態でソースコードと配布物を公開する。
2. サードパーティライセンスと付随ドキュメントを最新に保つ。
3. 利用者からの問い合わせチャネル (Discord) を明示する。

---

## ✅ チェックリスト

### 1. リポジトリと配布物の表記

- [ ] ルートの `LICENSE` が AGPL-3.0 の完全な本文であることを確認。
- [ ] `README.md` の「Licensing」セクションが AGPL-3.0 のみを案内し、
      Discord への問い合わせ先リンクを掲載している。
- [ ] `LICENSE-COMMERCIAL.md` が「商用ライセンス提供なし (not available today)」である旨を明記している。
- [ ] バイナリ/配布物 (PyPI, TestPyPI, Steam など) にも AGPL 表記と
      `LICENSE` が同梱されているかチェック。

### 2. サードパーティライセンス

- [ ] `docs/dev-docs/licensing-archive/01-license-audit.md` の依存リストを最新化。
- [ ] 追加依存がある場合は SPDX 識別子とライセンス本文へのリンクを追記。
- [ ] Steam や商用配布を行わない場合でも、配布物に第三者ライセンスの
      NOTICE ファイルを含める。

### 3. 問い合わせ経路

- [ ] README とドキュメントに Discord 招待リンク
      (https://discord.gg/hdSV4hJR8Y) を掲載。
- [ ] Discord サーバー内に #licensing などの連絡用スレッドを設置。
- [ ] 重要な問い合わせが来た場合の対応フロー
      (triage → issue化 → 対応) をチームで共有。

### 4. リリース前後の作業

- [ ] `docs/dev-docs/licensing-archive/04-implementation-roadmap.md` の公開前チェックリストを更新。
- [ ] PyPI や GitHub Release のリリースノートにライセンス要約を記載。
- [ ] Live_Cap_v3 などサブモジュール利用先にも AGPL である旨を通知。

---

## 📎 参考リンク

- `LICENSE`
- `README.md`
- `docs/dev-docs/licensing-archive/01-license-audit.md`
- `docs/dev-docs/licensing-archive/02-dual-license-strategy.md` (歴史的参照)
- `docs/dev-docs/licensing-archive/05-templates.md` (告知テンプレート。必要に応じて調整)

---

## 🗒 メモ

- 商用ライセンス需要の調査は Discord での問い合わせ数を指標にする。
- 再び有償プランを検討する場合、本ドキュメントに追加項目を設ける。
