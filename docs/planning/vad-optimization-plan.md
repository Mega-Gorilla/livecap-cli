# Phase D: VAD パラメータ最適化 実装計画

> **Status**: ACTIVE
> **作成日:** 2025-11-28
> **関連 Issue:** #126
> **前提:** Phase C 完了（VAD Benchmark 実装済み）

---

## 1. 概要

### 1.1 目的

VAD バックエンドのパラメータを **Bayesian Optimization (Optuna)** を用いて言語別に最適化し、ASR の精度（CER/WER）を改善する。

### 1.2 背景

Issue #86 の VAD Benchmark (standard mode) で以下の結果を得た：

| 言語 | Best VAD | 精度 | 備考 |
|------|----------|------|------|
| JA | javad_balanced | 7.9% CER | デフォルトパラメータ |
| EN | javad_balanced | 3.2% WER | デフォルトパラメータ |

**仮説**: 各 VAD のパラメータを言語別に調整することで、さらなる精度向上が期待できる。

### 1.3 成功基準

| 指標 | 現状 | 目標 |
|------|------|------|
| 日本語 CER | 7.9% | **5% 以下** |
| 英語 WER | 3.2% | **2.5% 以下** |

---

## 2. 技術設計

### 2.1 アーキテクチャ

```
benchmarks/
└── optimization/                    # 新規モジュール
    ├── __init__.py                  # 公開 API
    ├── param_spaces.py              # パラメータ探索空間定義
    ├── objective.py                 # 目的関数（CER/WER 最小化）
    ├── vad_optimizer.py             # Optuna ベースの最適化器
    ├── presets.py                   # 最適化結果の保存/読込
    └── __main__.py                  # CLI エントリポイント

tests/benchmark_tests/optimization/  # テスト
    ├── __init__.py
    ├── test_param_spaces.py
    └── test_objective.py
```

### 2.2 最適化対象パラメータ

#### Silero VAD (5 パラメータ)

| パラメータ | 型 | 探索範囲 | ステップ |
|-----------|-----|----------|----------|
| `threshold` | float | 0.2 - 0.8 | - |
| `neg_threshold` | float | 0.1 - 0.5 | - |
| `min_speech_ms` | int | 100 - 500 | 50 |
| `min_silence_ms` | int | 30 - 300 | 10 |
| `speech_pad_ms` | int | 30 - 200 | 10 |

#### TenVAD (6 パラメータ)

| パラメータ | 型 | 探索範囲 | ステップ |
|-----------|-----|----------|----------|
| `hop_size` | categorical | [160, 256] | - |
| `threshold` | float | 0.2 - 0.8 | - |
| `neg_threshold` | float | 0.1 - 0.5 | - |
| `min_speech_ms` | int | 100 - 500 | 50 |
| `min_silence_ms` | int | 30 - 300 | 10 |
| `speech_pad_ms` | int | 30 - 200 | 10 |

#### WebRTC VAD (5 パラメータ)

| パラメータ | 型 | 探索範囲 | ステップ |
|-----------|-----|----------|----------|
| `mode` | categorical | [0, 1, 2, 3] | - |
| `frame_duration_ms` | categorical | [10, 20, 30] | - |
| `min_speech_ms` | int | 100 - 500 | 50 |
| `min_silence_ms` | int | 30 - 300 | 10 |
| `speech_pad_ms` | int | 30 - 200 | 10 |

#### JaVAD (1 パラメータ)

| パラメータ | 型 | 探索範囲 |
|-----------|-----|----------|
| `model` | categorical | [tiny, balanced, precise] |

> **Note**: JaVAD は VADConfig 非対応のため、プリセット選択のみ。
> Grid Search で十分なため、Bayesian 最適化の優先度は低い。

### 2.3 目的関数設計

```python
def objective(trial: optuna.Trial) -> float:
    """
    1 trial = 1 パラメータセットの評価

    Returns:
        float: CER (JA) または WER (EN) - 最小化対象
    """
    # 1. パラメータ取得
    params = suggest_params(trial, vad_type=self.vad_type)

    # 2. VAD 作成（カスタムパラメータ適用）
    vad = create_vad_with_params(self.vad_type, params)

    # 3. ミニベンチマーク実行
    #    - ASR エンジンは事前ロード済み（trial 間で共有）
    #    - Quick mode 相当のデータセット使用
    results = []
    for audio_file in self.dataset:
        segments = vad.process_audio(audio_file.audio, audio_file.sample_rate)
        transcript = self._transcribe_segments(segments, audio_file)

        if self.language == "ja":
            score = calculate_cer(audio_file.transcript, transcript, lang="ja")
        else:
            score = calculate_wer(audio_file.transcript, transcript, lang="en")
        results.append(score)

    # 4. 平均スコア返却
    return statistics.mean(results)
```

### 2.4 実行時間見積もり

| 項目 | 時間 |
|------|------|
| 1 trial (30 ファイル処理 = quick モード) | ~45 秒 |
| 50 trials | ~38 分 |
| 1 VAD × 2 言語 | ~76 分 |
| 4 VAD × 2 言語 | **~300 分 (5 時間)** |

> **Note**: quick モード = 30 ファイル/言語（最適化に最適なサイズ）

### 2.5 GPU メモリ管理

```python
class VADOptimizer:
    def __init__(self, ...):
        # ASR エンジンは1回だけロード
        self.engine = self._load_engine(engine_id)
        self.engine.load_model()

    def _objective(self, trial):
        # VAD は毎回再作成（軽量）
        vad = create_vad_with_params(...)

        # ASR エンジンは共有（GPU メモリ節約）
        for audio_file in self.dataset:
            transcript = self.engine.transcribe(...)

        # VAD のみ解放
        del vad
```

---

## 3. 実装フェーズ

### Phase D-1: コアフレームワーク構築

**目標**: 最適化の基盤モジュールを作成

#### D-1a: パラメータ空間定義

```python
# benchmarks/optimization/param_spaces.py

from typing import Any
import optuna

PARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "silero": {
        "threshold": {"type": "float", "low": 0.2, "high": 0.8},
        "neg_threshold": {"type": "float", "low": 0.1, "high": 0.5},
        "min_speech_ms": {"type": "int", "low": 100, "high": 500, "step": 50},
        "min_silence_ms": {"type": "int", "low": 30, "high": 300, "step": 10},
        "speech_pad_ms": {"type": "int", "low": 30, "high": 200, "step": 10},
    },
    "tenvad": {
        "hop_size": {"type": "categorical", "choices": [160, 256]},
        "threshold": {"type": "float", "low": 0.2, "high": 0.8},
        # ... 他のパラメータ
    },
    # ... 他の VAD
}

def suggest_params(trial: optuna.Trial, vad_type: str) -> dict[str, Any]:
    """Trial からパラメータを提案"""
    space = PARAM_SPACES[vad_type]
    params = {}

    for name, config in space.items():
        if config["type"] == "float":
            params[name] = trial.suggest_float(name, config["low"], config["high"])
        elif config["type"] == "int":
            params[name] = trial.suggest_int(
                name, config["low"], config["high"], step=config.get("step", 1)
            )
        elif config["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, config["choices"])

    return params
```

#### D-1b: 目的関数実装

```python
# benchmarks/optimization/objective.py

from benchmarks.common import calculate_cer, calculate_wer

class VADObjective:
    """VAD 最適化の目的関数"""

    def __init__(
        self,
        vad_type: str,
        language: str,
        engine: TranscriptionEngine,
        dataset: list[AudioFile],
    ):
        self.vad_type = vad_type
        self.language = language
        self.engine = engine
        self.dataset = dataset

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(trial, self.vad_type)
        vad = create_vad_with_params(self.vad_type, params)

        scores = []
        for audio_file in self.dataset:
            # VAD 処理
            segments = vad.process_audio(audio_file.audio, audio_file.sample_rate)

            # ASR 処理
            transcript = self._transcribe_segments(segments, audio_file)

            # スコア計算
            if self.language == "ja":
                score = calculate_cer(audio_file.transcript, transcript, lang="ja")
            else:
                score = calculate_wer(audio_file.transcript, transcript, lang="en")
            scores.append(score)

        return statistics.mean(scores)
```

#### D-1c: 最適化器実装

```python
# benchmarks/optimization/vad_optimizer.py

import optuna
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """最適化結果"""
    vad_type: str
    language: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    study: optuna.Study

class VADOptimizer:
    """VAD パラメータ最適化器"""

    def __init__(
        self,
        vad_type: str,
        language: str,
        engine_id: str,
        device: str = "cuda",
    ):
        self.vad_type = vad_type
        self.language = language
        self.engine = self._load_engine(engine_id, device)
        self.dataset = self._load_dataset(language)

    def optimize(
        self,
        n_trials: int = 50,
        seed: int = 42,
        storage: str | None = None,
    ) -> OptimizationResult:
        """最適化を実行"""

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            storage=storage,
            study_name=f"{self.vad_type}_{self.language}",
            load_if_exists=True,
        )

        objective = VADObjective(
            vad_type=self.vad_type,
            language=self.language,
            engine=self.engine,
            dataset=self.dataset,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return OptimizationResult(
            vad_type=self.vad_type,
            language=self.language,
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=n_trials,
            study=study,
        )
```

#### D-1d: テスト

```python
# tests/benchmark_tests/optimization/test_param_spaces.py

def test_suggest_silero_params():
    """Silero パラメータ提案のテスト"""
    study = optuna.create_study()
    trial = study.ask()

    params = suggest_params(trial, "silero")

    assert "threshold" in params
    assert 0.2 <= params["threshold"] <= 0.8
    assert "min_speech_ms" in params
    assert params["min_speech_ms"] % 50 == 0
```

### Phase D-2: CLI 実装

**目標**: コマンドラインから最適化を実行可能に

#### D-2a: CLI エントリポイント

```python
# benchmarks/optimization/__main__.py

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="VAD Parameter Optimization")
    parser.add_argument("--vad", required=True, choices=["silero", "tenvad", "webrtc"])
    parser.add_argument("--language", required=True, choices=["ja", "en"])
    parser.add_argument("--engine", required=True, help="ASR engine ID")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--storage", help="Optuna storage URL (e.g., sqlite:///study.db)")

    args = parser.parse_args()

    optimizer = VADOptimizer(
        vad_type=args.vad,
        language=args.language,
        engine_id=args.engine,
    )

    result = optimizer.optimize(
        n_trials=args.n_trials,
        seed=args.seed,
        storage=args.storage,
    )

    # 結果出力
    print(f"\n=== Optimization Complete ===")
    print(f"VAD: {result.vad_type}")
    print(f"Language: {result.language}")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Best Params: {result.best_params}")

    if args.output:
        save_result(result, args.output)

if __name__ == "__main__":
    main()
```

#### D-2b: 使用例

```bash
# Silero × 日本語 の最適化
python -m benchmarks.optimization \
  --vad silero \
  --language ja \
  --engine parakeet_ja \
  --n-trials 50 \
  --output results/silero_ja.json \
  --storage sqlite:///optimization.db

# TenVAD × 英語 の最適化
python -m benchmarks.optimization \
  --vad tenvad \
  --language en \
  --engine parakeet \
  --n-trials 50 \
  --output results/tenvad_en.json
```

### Phase D-3: 結果の統合

**目標**: 最適化結果を livecap_core で利用可能に

#### D-3a: プリセット管理

```python
# benchmarks/optimization/presets.py

import json
from pathlib import Path

PRESETS_FILE = Path(__file__).parent.parent.parent / "config" / "vad_optimized_presets.json"

def save_preset(vad_type: str, language: str, params: dict) -> None:
    """最適化結果をプリセットとして保存"""
    presets = load_all_presets()

    if vad_type not in presets:
        presets[vad_type] = {}
    presets[vad_type][language] = params

    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)

def load_preset(vad_type: str, language: str) -> dict | None:
    """プリセットを読み込み"""
    presets = load_all_presets()
    return presets.get(vad_type, {}).get(language)
```

#### D-3b: プリセットファイル形式

```json
// config/vad_optimized_presets.json
{
  "silero": {
    "ja": {
      "threshold": 0.38,
      "neg_threshold": 0.23,
      "min_speech_ms": 200,
      "min_silence_ms": 62,
      "speech_pad_ms": 85
    },
    "en": {
      "threshold": 0.52,
      "neg_threshold": 0.37,
      "min_speech_ms": 150,
      "min_silence_ms": 95,
      "speech_pad_ms": 70
    }
  },
  "tenvad": {
    "ja": {
      "hop_size": 160,
      "threshold": 0.42,
      // ...
    }
  }
}
```

#### D-3c: Factory 統合

```python
# benchmarks/vad/factory.py への追加

def create_vad(
    vad_id: str,
    language: str | None = None,
    use_optimized: bool = False,
) -> VADBenchmarkBackend:
    """
    VAD バックエンドを作成

    Args:
        vad_id: VAD 識別子
        language: 言語コード（最適化プリセット使用時に必要）
        use_optimized: True の場合、最適化済みプリセットを使用
    """
    if use_optimized and language:
        preset = load_preset(vad_id, language)
        if preset:
            return _create_vad_with_params(vad_id, preset)

    return _create_vad_default(vad_id)
```

### Phase D-4: 検証

**目標**: 最適化の効果を検証

#### D-4a: Standard モードで再ベンチマーク

```bash
# 最適化パラメータでベンチマーク実行
python -m benchmarks.vad \
  --mode standard \
  --use-optimized \
  --languages ja en
```

#### D-4b: 比較レポート作成

| VAD | 言語 | Before | After | 改善率 |
|-----|------|--------|-------|--------|
| Silero | JA | 8.5% CER | ? | ? |
| Silero | EN | 4.9% WER | ? | ? |
| TenVAD | JA | 8.2% CER | ? | ? |
| TenVAD | EN | 6.2% WER | ? | ? |

---

## 4. 依存関係

### pyproject.toml への追加

```toml
[project.optional-dependencies]
optimization = [
    "optuna>=3.0",
]
```

---

## 5. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| **過学習** | 検証データで性能低下 | Quick で最適化 → Standard で検証 |
| **局所最適** | 真の最適解を逃す | n_trials 増加、複数 seed 実行 |
| **GPU メモリ不足** | 最適化中断 | Engine 共有、適切な解放 |
| **実行時間超過** | CI タイムアウト | 手動トリガー、分割実行 |

---

## 6. タスクリスト

### Phase D-1: コアフレームワーク
- [ ] `benchmarks/optimization/__init__.py` 作成
- [ ] `benchmarks/optimization/param_spaces.py` 実装
- [ ] `benchmarks/optimization/objective.py` 実装
- [ ] `benchmarks/optimization/vad_optimizer.py` 実装
- [ ] `pyproject.toml` に `optimization` extra 追加
- [ ] 単体テスト作成

### Phase D-2: CLI 実装
- [ ] `benchmarks/optimization/__main__.py` 実装
- [ ] Silero × JA で end-to-end テスト
- [ ] 他の VAD × 言語に拡張
- [ ] (Optional) GitHub Actions workflow 作成

### Phase D-3: 結果統合
- [ ] `benchmarks/optimization/presets.py` 実装
- [ ] `config/vad_optimized_presets.json` フォーマット設計
- [ ] `benchmarks/vad/factory.py` に `use_optimized` 追加

### Phase D-4: 検証
- [ ] 全 VAD × 言語の最適化実行
- [ ] Standard モードで検証ベンチマーク
- [ ] 比較レポート作成
- [ ] Issue #126 クローズ

---

## 7. 参考資料

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- Issue #86: VAD + ASR ベンチマーク実装
- Issue #126: VAD パラメータ最適化

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
