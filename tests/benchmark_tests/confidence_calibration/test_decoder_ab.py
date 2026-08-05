"""decoder_ab の集計ロジック単体 test (Issue #373)。

engine を必要としない pure logic (``build_decoding_cfg`` / ``percentile`` /
``summarize_latency`` / ``filter_confusion`` / ``normalize_text`` /
``count_script_sandwich`` / ``detect_truncation`` / ``pairwise_quality``)
のみを検証する。torch 非依存。
"""
from __future__ import annotations

import pytest

from benchmarks.confidence_calibration.decoder_ab import (
    ClipMeasurement,
    build_decoding_cfg,
    count_script_sandwich,
    coverage_warnings,
    detect_truncation,
    filter_confusion,
    normalize_text,
    pairwise_quality,
    parse_decoder_order,
    percentile,
    run_stats,
    signal_coverage,
    signal_distribution,
    summarize_latency,
)


def clip(path="p", label="speech", *, text="こんにちは", signal=0.5, latency=0.1,
         duration=1.0, rejected=False, error=False):
    return ClipMeasurement(
        path=path, label=label, duration_sec=duration, text=text,
        signal_value=signal, latency_sec=latency, rejected=rejected, error=error,
    )


class TestBuildDecodingCfg:
    def test_ctc_maps_to_ctc_decoder_type(self):
        cfg, decoder_type = build_decoding_cfg("ctc")
        assert decoder_type == "ctc"
        assert cfg["strategy"] == "greedy_batch"
        assert cfg["confidence_cfg"]["preserve_token_confidence"] is True

    def test_tdt_maps_to_rnnt_decoder_type(self):
        cfg, decoder_type = build_decoding_cfg("tdt")
        assert decoder_type == "rnnt"
        # TDT でも filter signal を維持する比較なので confidence_cfg は同一
        assert cfg["confidence_cfg"] == build_decoding_cfg("ctc")[0]["confidence_cfg"]

    def test_unknown_decoder_raises(self):
        with pytest.raises(ValueError):
            build_decoding_cfg("beam")


class TestParseDecoderOrder:
    def test_default_pair(self):
        assert parse_decoder_order("ctc,tdt") == ["ctc", "tdt"]

    def test_reversed_order_preserved(self):
        assert parse_decoder_order("tdt,ctc") == ["tdt", "ctc"]

    def test_single_decoder_allowed(self):
        assert parse_decoder_order("tdt") == ["tdt"]

    def test_unknown_decoder_raises(self):
        with pytest.raises(ValueError):
            parse_decoder_order("ctc,beam")

    def test_duplicate_raises(self):
        with pytest.raises(ValueError):
            parse_decoder_order("ctc,ctc")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_decoder_order(" , ")


class TestRunStats:
    def test_counts_errors(self):
        out = run_stats([clip(), clip(error=True), clip()])
        assert out == {
            "total": 3,
            "success": 2,
            "errors": 1,
            "error_rate": pytest.approx(1 / 3),
        }

    def test_empty(self):
        assert run_stats([])["error_rate"] is None


class TestSignalCoverage:
    def test_missing_signal_counted(self):
        ms = [
            clip(signal=0.5),
            clip(signal=None),  # fail-open で pass するが coverage には欠損として出る
            clip(signal=0.1, error=True),  # error は除外
            clip(label="non_speech", signal=0.0),
        ]
        out = signal_coverage(ms)
        assert out["speech"]["success"] == 2
        assert out["speech"]["signal_non_null"] == 1
        assert out["speech"]["signal_missing"] == 1
        assert out["speech"]["coverage_rate"] == pytest.approx(0.5)
        assert out["non_speech"]["coverage_rate"] == pytest.approx(1.0)

    def test_empty_label(self):
        assert signal_coverage([clip()])["non_speech"] == {"success": 0}


class TestCoverageWarnings:
    def test_below_threshold_warns_per_condition_label(self):
        cov = {
            "tdt": {
                "speech": {"success": 10, "signal_non_null": 5,
                           "signal_missing": 5, "coverage_rate": 0.5,
                           "is_available_true": 10},
                "non_speech": {"success": 0},
            },
            "ctc": {
                "speech": {"success": 10, "signal_non_null": 10,
                           "signal_missing": 0, "coverage_rate": 1.0,
                           "is_available_true": 10},
            },
        }
        warnings = coverage_warnings(cov, 0.95)
        assert len(warnings) == 1
        assert "[tdt] speech" in warnings[0]

    def test_all_covered_returns_empty(self):
        cov = {"ctc": {"speech": {"success": 1, "signal_non_null": 1,
                                  "signal_missing": 0, "coverage_rate": 1.0,
                                  "is_available_true": 1}}}
        assert coverage_warnings(cov, 0.95) == []


class TestPercentile:
    def test_single_value(self):
        assert percentile([3.0], 95) == 3.0

    def test_median_interpolation(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)


class TestSummarizeLatency:
    def test_excludes_errors_and_computes_rtf(self):
        ms = [
            clip(latency=0.1, duration=1.0),
            clip(latency=0.3, duration=1.0),
            clip(latency=99.0, error=True),
        ]
        out = summarize_latency(ms)
        assert out["n"] == 2
        assert out["latency_sec"]["p50"] == pytest.approx(0.2)
        assert out["rtf"]["p50"] == pytest.approx(0.2)

    def test_empty_returns_n_zero(self):
        assert summarize_latency([clip(error=True)]) == {"n": 0}


class TestSignalDistribution:
    def test_split_by_norm_label(self):
        ms = [
            clip(label="speech", signal=0.5),
            clip(label="noisy_speech", signal=0.3),  # norm → speech
            clip(label="non_speech", signal=0.0),
            clip(label="non_speech", signal=None),  # None は除外
        ]
        out = signal_distribution(ms)
        assert out["speech"]["n"] == 2
        assert out["non_speech"]["n"] == 1
        assert out["non_speech"]["max"] == 0.0


class TestFilterConfusion:
    def test_speech_false_reject_rate(self):
        ms = [clip(rejected=True), clip(rejected=False), clip(label="noisy_speech")]
        out = filter_confusion(ms)
        assert out["speech"]["n"] == 3
        assert out["speech"]["rejected"] == 1
        assert out["speech"]["false_reject_rate"] == pytest.approx(1 / 3)

    def test_non_speech_leak_counts_only_unrejected_non_empty(self):
        ms = [
            clip(label="non_speech", text="どうぞ", rejected=False),  # leak
            clip(label="non_speech", text="どうぞ", rejected=True),  # filter 捕捉
            clip(label="non_speech", text="", rejected=False),  # 空text guard 相当
        ]
        out = filter_confusion(ms)
        assert out["non_speech"]["n"] == 3
        assert out["non_speech"]["non_empty_text"] == 2
        assert out["non_speech"]["leak"] == 1
        assert out["non_speech"]["leak_rate"] == pytest.approx(1 / 3)

    def test_errors_excluded_and_counted(self):
        out = filter_confusion([clip(error=True)])
        assert "speech" not in out
        assert out["errors_excluded"] == 1


class TestNormalizeText:
    def test_nfkc_and_punct_strip(self):
        assert normalize_text("ｖｔｕｂｅｒ、です。") == "vtuberです"
        assert normalize_text(" a b ") == "ab"
        assert normalize_text(None) == ""


class TestCountScriptSandwich:
    def test_detects_mixed_script_artifact(self):
        ms = [clip(text="ぼくにヒつじの絵を描いて。")]
        out = count_script_sandwich(ms)
        assert out["clips"] == 1

    def test_clean_kana_not_detected(self):
        assert count_script_sandwich([clip(text="ひつじのえをかいて。")])["clips"] == 0

    def test_legit_boundary_is_known_false_positive(self):
        # 「エンジンのトラブル」は単語境界をまたぐ正当な表記だが、保守的
        # heuristic は hit する (両 decoder 同条件のため相対比較には無害)。
        # この挙動を pin して、regex 変更時に意図を再確認させる。
        assert count_script_sandwich([clip(text="エンジンのトラブルで。")])["clips"] == 1

    def test_non_speech_excluded(self):
        assert count_script_sandwich([clip(label="non_speech", text="ヒつじ")])["clips"] == 0


class TestDetectTruncation:
    def test_a_truncated(self):
        assert detect_truncation("そこで僕は次の手を出。", "そこで僕は次の手を出した。") == "a"

    def test_b_truncated(self):
        assert detect_truncation("あいうえお", "あいう") == "b"

    def test_equal_or_divergent_is_none(self):
        assert detect_truncation("おなじ。", "おなじ") is None
        assert detect_truncation("ちがう", "べつもの") is None

    def test_min_gap_respected(self):
        # 1 文字差は切り捨てとみなさない (末尾助詞ゆれと区別するため)
        assert detect_truncation("あいうえ", "あいうえお") is None


class TestPairwiseQuality:
    def test_agreement_and_truncation_asymmetry(self):
        ctc = [
            clip(path="a", text="エンジンを止め。"),
            clip(path="b", text="おなじです。"),
            clip(path="n", label="non_speech", text="x"),  # speech 以外は除外
        ]
        tdt = [
            clip(path="a", text="エンジンを止めないと。"),
            clip(path="b", text="おなじです"),
            clip(path="n", label="non_speech", text="y"),
        ]
        out = pairwise_quality(ctc, tdt)
        assert out["speech_pairs"] == 2
        assert out["text_agreement"] == 1  # b は正規化後一致
        assert out["truncation_ctc"] == 1
        assert out["truncation_tdt"] == 0

    def test_error_rows_excluded(self):
        out = pairwise_quality([clip(path="a", error=True)], [clip(path="a")])
        assert out["speech_pairs"] == 0
        assert out["text_agreement_rate"] is None
