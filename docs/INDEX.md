# Documentation Index

> **最終更新:** 2025-12-01

livecap-cli ドキュメントの目次と状態管理。

---

## Planning (計画)

アクティブな計画と完了した計画を管理。

### Active

現在アクティブな計画はありません。

### Archive (完了)

| ドキュメント | 完了日 | Issue | 説明 |
|-------------|--------|-------|------|
| [phase1-implementation-plan.md](planning/archive/phase1-implementation-plan.md) | 2025-11-25 | #69 | リアルタイム文字起こし実装計画 |
| [refactoring-plan.md](planning/archive/refactoring-plan.md) | 2025-11-28 | #69, #86 | LiveCap Core リファクタリング計画 |
| [vad-benchmark-plan.md](planning/archive/vad-benchmark-plan.md) | 2025-11-28 | #86 | VAD + ASR ベンチマーク実装計画 |
| [vad-optimization-plan.md](planning/archive/vad-optimization-plan.md) | 2025-11-29 | #126 | VAD パラメータ最適化計画 |
| [language-based-vad-optimization.md](planning/archive/language-based-vad-optimization.md) | 2025-12-01 | #139 | 言語別VAD最適化計画 |

---

## Architecture (設計)

システム設計とAPI仕様。

| ドキュメント | 説明 |
|-------------|------|
| [core-api-spec.md](architecture/core-api-spec.md) | Core API 仕様 |

---

## Reference (参考資料)

技術調査と比較分析。

| ドキュメント | 説明 |
|-------------|------|
| [vad/comparison.md](reference/vad/comparison.md) | VAD バックエンド比較分析 + ベンチマーク結果 |
| [feature-inventory.md](reference/feature-inventory.md) | 機能一覧 |
| [livecap-gui-realtime-analysis.md](reference/livecap-gui-realtime-analysis.md) | livecap-gui リアルタイム処理分析 |

---

## Guides (ガイド)

使い方ガイド。

### 基本ガイド

| ドキュメント | 説明 |
|-------------|------|
| [realtime-transcription.md](guides/realtime-transcription.md) | リアルタイム文字起こしガイド |

### ベンチマークガイド

| ドキュメント | 説明 |
|-------------|------|
| [benchmark/asr-benchmark.md](guides/benchmark/asr-benchmark.md) | ASR ベンチマークガイド |
| [benchmark/vad-benchmark.md](guides/benchmark/vad-benchmark.md) | VAD ベンチマークガイド |
| [benchmark/vad-optimization.md](guides/benchmark/vad-optimization.md) | VAD パラメータ最適化ガイド |

---

## Testing (テスト)

テスト関連ドキュメント。

| ドキュメント | 説明 |
|-------------|------|
| [README.md](testing/README.md) | テスト概要 |

---

## 関連 Issue

| Issue | 状態 | 説明 |
|-------|------|------|
| #69 | ✅ Closed | Phase 1: リアルタイム文字起こし実装 |
| #86 | ✅ Closed | VAD + ASR ベンチマーク実装 |
| #126 | ✅ Closed | Phase D: VAD パラメータ最適化 |
| #127 | ✅ Closed | ドキュメント整理・アーカイブ化 |
| #139 | ✅ Closed | 言語別VAD最適化の実装 |
| #154 | 🚧 Open | ドキュメント構成のリファクタリング |
