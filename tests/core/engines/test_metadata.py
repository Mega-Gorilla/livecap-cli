"""EngineMetadata.recommend() と recommend 用 metadata の単体テスト (Issue #286)。

torch 非依存 (langcodes のみ)。GPU/VRAM は recommend() の引数で制御するため
patch 不要。
"""
from __future__ import annotations

import pytest

from livecap_cli.engines.metadata import (
    EngineInfo,
    EngineMetadata,
    EngineRecommendation,
    LanguageQuality,
    ReasonCode,
)


def _ids(recs):
    return [r.engine_id for r in recs]


class TestNewMetadataFields:
    def test_new_fields_default(self):
        """新 field は default 付きで、最小構築でも安全 (後方互換)。"""
        info = EngineInfo(
            id="stub",
            display_name="Stub",
            description="stub",
            supported_languages=["ja"],
        )
        assert info.quality_tier == {}
        assert info.vram_required_mb is None
        assert info.cpu_supported is True
        assert info.cpu_recommended is False
        assert info.gpu_recommended is True
        assert info.realtime_on_cpu is False

    def test_registered_engines_have_quality_tier(self):
        """実 registry の specialist が想定の tier を持つ。"""
        assert EngineMetadata.get("reazonspeech").quality_tier["ja"].tier == "best"
        assert EngineMetadata.get("parakeet").quality_tier["en"].tier == "best"
        assert EngineMetadata.get("canary").quality_tier["de"].tier == "best"
        # v1 draft の「qwen3asr 全 best」撤回 → good
        assert EngineMetadata.get("qwen3asr").quality_tier["zh"].tier == "good"
        # whispers2t は全 fallback
        assert EngineMetadata.get("whispers2t").quality_tier["ja"].tier == "fallback"

    def test_vram_required_measured_values(self):
        """VRAM 正本は EngineInfo。実測 seed 値を確認。"""
        assert EngineMetadata.get("parakeet").vram_required_mb == 2417
        assert EngineMetadata.get("canary").vram_required_mb == 6830
        assert EngineMetadata.get("voxtral").vram_required_mb == 8923
        # 未計測/size 依存は None (filter で除外されない)
        assert EngineMetadata.get("reazonspeech").vram_required_mb is None
        assert EngineMetadata.get("whispers2t").vram_required_mb is None


class TestRecommend:
    def test_ja_gpu_best_first(self):
        """ja + GPU8GB → best tier が上位、whispers2t(fallback) が最下位。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        top2 = {r.engine_id for r in recs if r.rank <= 2}
        assert top2 == {"parakeet_ja", "reazonspeech"}  # 順序は非固定 (共に best)
        assert all(r.quality == "best" for r in recs if r.rank <= 2)
        assert recs[-1].engine_id == "whispers2t"
        assert recs[-1].quality == "fallback"

    def test_ja_cpu_prefers_light_engine(self):
        """ja + CPU → realtime_on_cpu な reazonspeech が parakeet_ja より上位。"""
        recs = EngineMetadata.recommend("ja", gpu_available=False)
        rank = {r.engine_id: r.rank for r in recs}
        assert rank["reazonspeech"] < rank["parakeet_ja"]
        rz = next(r for r in recs if r.engine_id == "reazonspeech")
        assert ReasonCode.REALTIME_ON_CPU in rz.reason_codes

    def test_en_gpu_parakeet_first(self):
        """en + GPU → parakeet(best) rank1、canary/voxtral は good。"""
        recs = EngineMetadata.recommend("en", gpu_available=True, vram_gb=8.0)
        assert recs[0].engine_id == "parakeet"
        assert recs[0].quality == "best"
        qualities = {r.engine_id: r.quality for r in recs}
        assert qualities["canary"] == "good"
        assert qualities["voxtral"] == "good"

    def test_vram_filter_sinks_exceeding(self):
        """ja + GPU2GB → parakeet_ja(2500MB) は EXCEEDS_VRAM で reazonspeech より下位。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=2.0)
        rank = {r.engine_id: r.rank for r in recs}
        assert rank["reazonspeech"] < rank["parakeet_ja"]
        pj = next(r for r in recs if r.engine_id == "parakeet_ja")
        assert ReasonCode.EXCEEDS_VRAM in pj.reason_codes
        assert pj.scores["hardware_fit_score"] == 0.0
        # 除外はされない (list に残る)
        assert "parakeet_ja" in _ids(recs)

    def test_zh_qwen3asr_top(self):
        """zh (専用 engine 不在) → qwen3asr(good) rank1、whispers2t(fallback)。"""
        recs = EngineMetadata.recommend("zh", gpu_available=True, vram_gb=8.0)
        assert recs[0].engine_id == "qwen3asr"
        assert recs[0].quality == "good"
        assert "whispers2t" in _ids(recs)

    def test_whispers2t_params_size_by_hardware(self):
        """whispers2t は hardware に応じた params["model_size"] を返す。"""
        def whisper_size(gpu, vram):
            recs = EngineMetadata.recommend("ja", gpu_available=gpu, vram_gb=vram)
            w = next(r for r in recs if r.engine_id == "whispers2t")
            return w.params["model_size"]

        assert whisper_size(True, 8.0) == "large-v3"
        assert whisper_size(True, 4.0) == "small"
        assert whisper_size(True, 1.0) == "base"
        assert whisper_size(False, None) == "base"

    def test_non_whisper_params_empty(self):
        """whispers2t 以外の rec.params は空 dict。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        for r in recs:
            if r.engine_id != "whispers2t":
                assert r.params == {}

    def test_rank_is_contiguous_and_sorted(self):
        """rank は 1..N の連番・昇順。"""
        recs = EngineMetadata.recommend("en", gpu_available=True, vram_gb=8.0)
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))

    def test_reason_codes_present(self):
        """各 rec の reason_codes は非空。ja 特化 engine に LANGUAGE_SPECIALIZED。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        for r in recs:
            assert r.reason_codes
            assert all(isinstance(c, ReasonCode) for c in r.reason_codes)
        rz = next(r for r in recs if r.engine_id == "reazonspeech")
        assert ReasonCode.LANGUAGE_SPECIALIZED in rz.reason_codes

    def test_multilingual_engine_not_marked_specialized(self):
        """canary (EN/DE/FR/ES) は多言語 engine → MULTILINGUAL (best tier でも特化扱いしない)。"""
        recs = EngineMetadata.recommend("de", gpu_available=True, vram_gb=8.0)
        canary = next(r for r in recs if r.engine_id == "canary")
        assert canary.quality == "best"  # de では best tier
        assert ReasonCode.MULTILINGUAL in canary.reason_codes
        assert ReasonCode.LANGUAGE_SPECIALIZED not in canary.reason_codes

    def test_single_language_engine_marked_specialized(self):
        """単一言語 engine (reazonspeech=ja) は LANGUAGE_SPECIALIZED。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        rz = next(r for r in recs if r.engine_id == "reazonspeech")
        assert ReasonCode.LANGUAGE_SPECIALIZED in rz.reason_codes
        assert ReasonCode.MULTILINGUAL not in rz.reason_codes

    def test_only_candidate_code(self):
        """対応 engine が単一の言語では ONLY_CANDIDATE が付く。"""
        # whispers2t のみが対応する言語を registry から探す
        solo_lang = None
        for lang in EngineMetadata.get("whispers2t").supported_languages:
            if EngineMetadata.get_engines_for_language(lang) == ["whispers2t"]:
                solo_lang = lang
                break
        assert solo_lang is not None, "単一対応言語が見つからない (registry 変化?)"
        recs = EngineMetadata.recommend(solo_lang)
        assert len(recs) == 1
        assert ReasonCode.ONLY_CANDIDATE in recs[0].reason_codes

    def test_bcp47_normalized(self):
        """BCP-47 (zh-CN) は ISO639-1 (zh) と同じ結果。"""
        a = _ids(EngineMetadata.recommend("zh-CN", gpu_available=True, vram_gb=8.0))
        b = _ids(EngineMetadata.recommend("zh", gpu_available=True, vram_gb=8.0))
        assert a == b

    def test_subset_of_get_engines_for_language(self):
        """recommend の engine_id 群 ⊆ get_engines_for_language (後方互換の健全性)。"""
        for lang in ("ja", "en", "zh", "de"):
            rec_ids = set(_ids(EngineMetadata.recommend(lang)))
            legacy = set(EngineMetadata.get_engines_for_language(lang))
            assert rec_ids == legacy

    def test_params_launch_shape(self):
        """rec.params は create_engine(**params) に展開できる dict。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        for r in recs:
            assert isinstance(r.params, dict)
            assert all(isinstance(k, str) for k in r.params)
        w = next(r for r in recs if r.engine_id == "whispers2t")
        assert set(w.params) == {"model_size"}

    def test_scores_keys(self):
        """scores は4分解キーを持つ (単一合計でなく個別露出)。"""
        recs = EngineMetadata.recommend("ja", gpu_available=True, vram_gb=8.0)
        for r in recs:
            assert set(r.scores) == {
                "quality_score",
                "hardware_fit_score",
                "latency_score",
                "streaming_score",
            }

    def test_returns_engine_recommendation_type(self):
        recs = EngineMetadata.recommend("ja")
        assert recs
        assert all(isinstance(r, EngineRecommendation) for r in recs)


class TestBackwardCompat:
    def test_existing_apis_unchanged(self):
        """既存 API は破壊されない。"""
        assert EngineMetadata.get("reazonspeech").display_name == "ReazonSpeech K2 v2"
        assert "whispers2t" in EngineMetadata.get_all()
        assert EngineMetadata.get_engines_for_language("ja")
        assert EngineMetadata.to_iso639_1("zh-CN") == "zh"
