"""EnergyGate ablation の simulate ロジック単体 test (Issue #357)。

engine を必要としない pure logic (``summarize`` / ``CONFIG_DROP`` /
``GuardRecord.norm_label``) のみを検証する。torch 非依存。
"""
from __future__ import annotations

import pytest

from benchmarks.confidence_calibration.energygate_ablation import (
    CONFIG_DROP,
    GuardRecord,
    summarize,
)


def rec(label, *, energy_drop=False, empty_text=False, conf_reject=False,
        energy_dbfs=-20.0, text="x", path="p", error=False, error_reason=None):
    return GuardRecord(
        path=path, label=label, energy_dbfs=energy_dbfs, energy_drop=energy_drop,
        empty_text=empty_text, conf_reject=conf_reject, text=text,
        error=error, error_reason=error_reason,
    )


class TestConfigDrop:
    def test_baseline_drops_only_empty_text(self):
        assert CONFIG_DROP["baseline"](rec("speech", empty_text=True)) is True
        assert CONFIG_DROP["baseline"](rec("speech", energy_drop=True)) is False
        assert CONFIG_DROP["baseline"](rec("speech", conf_reject=True)) is False

    def test_energy_config_adds_energy_drop(self):
        assert CONFIG_DROP["energy"](rec("non_speech", energy_drop=True)) is True
        assert CONFIG_DROP["energy"](rec("non_speech", conf_reject=True)) is False

    def test_confidence_config_adds_conf_reject(self):
        assert CONFIG_DROP["confidence"](rec("non_speech", conf_reject=True)) is True
        assert CONFIG_DROP["confidence"](rec("non_speech", energy_drop=True)) is False

    def test_both_is_union(self):
        assert CONFIG_DROP["both"](rec("non_speech", energy_drop=True)) is True
        assert CONFIG_DROP["both"](rec("non_speech", conf_reject=True)) is True
        assert CONFIG_DROP["both"](rec("non_speech", empty_text=True)) is True
        assert CONFIG_DROP["both"](rec("non_speech")) is False


class TestNormLabel:
    def test_noisy_speech_is_speech(self):
        assert rec("noisy_speech").norm_label == "speech"
        assert rec("speech").norm_label == "speech"
        assert rec("non_speech").norm_label == "non_speech"


class TestSummarize:
    def test_sample_counts_normalize_noisy(self):
        recs = [rec("speech"), rec("noisy_speech"), rec("non_speech")]
        s = summarize(recs)
        assert s["sample_count"]["speech"] == 2
        assert s["sample_count"]["non_speech"] == 1
        assert s["sample_count"]["evaluated"] == 3
        assert s["sample_count"]["errored_excluded"] == 0
        assert s["error_count"] == 0

    def test_errored_records_excluded_from_confusion(self):
        # engine error は confusion 集計に混ぜず error_count で別途報告 (codex-review #1)
        recs = [
            rec("non_speech", conf_reject=True),
            rec("non_speech", error=True, error_reason="CUDA OOM"),
            rec("speech"),
        ]
        s = summarize(recs)
        assert s["error_count"] == 1
        assert s["sample_count"]["evaluated"] == 2  # error 除外後
        assert s["sample_count"]["non_speech"] == 1  # errored non_speech は除外
        conf = s["configs"]["confidence"]
        assert conf["non_speech_total"] == 1  # error sample は分母に含まない
        assert conf["non_speech_suppressed"] == 1
        assert s["error_reasons"][0]["reason"] == "CUDA OOM"

    def test_energy_unique_marginal(self):
        # non_speech: energy が落とすが empty/conf は捕捉しない → unique 付加価値
        recs = [
            rec("non_speech", energy_drop=True),  # unique
            rec("non_speech", energy_drop=True, conf_reject=True),  # overlap
            rec("non_speech", conf_reject=True),  # confidence のみ
        ]
        s = summarize(recs)
        m = s["energy_gate_marginal"]
        assert m["non_speech_energy_unique"] == 1
        assert m["non_speech_energy_overlap"] == 1
        assert m["non_speech_energy_total_drop"] == 2

    def test_speech_harm_counted(self):
        # speech を energy だけが落とす = net-new harm
        recs = [
            rec("speech", energy_drop=True),  # net harm
            rec("speech", energy_drop=True, conf_reject=True),  # confidence も落とすので net でない
            rec("speech"),
        ]
        s = summarize(recs)
        m = s["energy_gate_marginal"]
        assert m["speech_energy_unique_harm"] == 1
        assert m["speech_energy_total_drop"] == 2

    def test_suppression_and_frr_rates(self):
        recs = [
            rec("non_speech", conf_reject=True),
            rec("non_speech"),  # missed by all
            rec("speech"),
            rec("speech", conf_reject=True),  # false drop by confidence
        ]
        s = summarize(recs)
        conf = s["configs"]["confidence"]
        assert conf["non_speech_suppressed"] == 1
        assert conf["non_speech_total"] == 2
        assert conf["suppression_rate"] == 0.5
        assert conf["speech_dropped"] == 1
        assert conf["false_drop_rate"] == 0.5

    def test_silence_hallucination_detected(self):
        # 無音 (energy_drop) な non_speech で engine 非空・conf pass = EnergyGate-only 救済
        recs = [
            rec("non_speech", energy_drop=True, empty_text=False, conf_reject=False,
                text="幻聴テキスト", energy_dbfs=-200.0),
            rec("non_speech", energy_drop=True, empty_text=True),  # 空text guard が捕捉
        ]
        s = summarize(recs)
        h = s["silence_hallucination"]
        assert h["silent_non_speech_total"] == 2
        assert h["engine_nonempty"] == 1
        assert h["engine_nonempty_conf_pass"] == 1
        assert len(h["examples"]) == 1
        assert h["examples"][0]["text"] == "幻聴テキスト"

    def test_both_config_never_worse_suppression_than_parts(self):
        recs = [
            rec("non_speech", energy_drop=True),
            rec("non_speech", conf_reject=True),
            rec("non_speech", empty_text=True),
            rec("speech"),
        ]
        s = summarize(recs)["configs"]
        assert s["both"]["non_speech_suppressed"] >= s["energy"]["non_speech_suppressed"]
        assert s["both"]["non_speech_suppressed"] >= s["confidence"]["non_speech_suppressed"]
        assert s["both"]["non_speech_suppressed"] == 3

    def test_empty_records_safe(self):
        s = summarize([])
        assert s["sample_count"]["evaluated"] == 0
        assert s["error_count"] == 0
        assert s["configs"]["both"]["suppression_rate"] == 0.0
