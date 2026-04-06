"""ResultCoalescer のユニットテスト。"""

from __future__ import annotations

import pytest

from livecap_cli.transcription.result import TranscriptionResult
from livecap_cli.transcription.result_coalescer import ResultCoalescer


def _make_result(
    text: str,
    start: float = 0.0,
    end: float = 1.0,
    language: str = "",
    confidence: float = 0.9,
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        start_time=start,
        end_time=end,
        language=language,
        confidence=confidence,
    )


# === _is_short() テスト ===


class TestIsShort:
    """_is_short() の多段ヒューリスティクス判定。"""

    def test_japanese_short(self):
        c = ResultCoalescer()
        assert c._is_short("はい") is True  # 2文字 ≤ 4

    def test_japanese_not_short(self):
        c = ResultCoalescer()
        assert c._is_short("そうですね") is False  # 5文字 > 4

    def test_japanese_punctuation(self):
        c = ResultCoalescer()
        assert c._is_short("はい。") is False  # 句読点終端

    def test_english_short(self):
        c = ResultCoalescer()
        assert c._is_short("yes") is True  # 3文字 ≤ 4

    def test_english_not_short_multiword(self):
        c = ResultCoalescer()
        assert c._is_short("I agree") is False  # 2語 > 1

    def test_english_ok(self):
        c = ResultCoalescer()
        assert c._is_short("OK") is True  # 2文字 ≤ 4

    def test_chinese_short(self):
        c = ResultCoalescer()
        assert c._is_short("好的") is True  # 2文字 ≤ 4

    def test_korean_short(self):
        c = ResultCoalescer()
        assert c._is_short("네") is True  # 1文字 ≤ 4

    def test_korean_not_short(self):
        c = ResultCoalescer()
        assert c._is_short("알겠습니다") is False  # 5文字 > 4

    def test_empty_string(self):
        c = ResultCoalescer()
        assert c._is_short("") is False

    def test_whitespace_only(self):
        c = ResultCoalescer()
        assert c._is_short("   ") is False

    def test_punctuation_exclamation(self):
        c = ResultCoalescer()
        assert c._is_short("OK!") is False

    def test_punctuation_question(self):
        c = ResultCoalescer()
        assert c._is_short("え？") is False

    def test_fullwidth_punctuation(self):
        c = ResultCoalescer()
        assert c._is_short("はい！") is False


# === _join_text() テスト ===


class TestJoinText:
    """言語ベースのテキスト結合。"""

    def test_english_space(self):
        c = ResultCoalescer()
        assert c._join_text("yes", "I agree", "en") == "yes I agree"

    def test_japanese_no_space(self):
        c = ResultCoalescer()
        assert c._join_text("はい", "今日は", "ja") == "はい今日は"

    def test_korean_space(self):
        c = ResultCoalescer()
        assert c._join_text("네", "알겠습니다", "ko") == "네 알겠습니다"

    def test_chinese_no_space(self):
        c = ResultCoalescer()
        assert c._join_text("好的", "今天天气不错", "zh") == "好的今天天气不错"

    def test_french_space(self):
        c = ResultCoalescer()
        assert c._join_text("oui", "je comprends", "fr") == "oui je comprends"

    def test_empty_language_no_space(self):
        c = ResultCoalescer()
        assert c._join_text("はい", "今日は", "") == "はい今日は"

    def test_none_language_no_space(self):
        c = ResultCoalescer()
        assert c._join_text("はい", "今日は", None) == "はい今日は"

    def test_language_prefix_match(self):
        """en-US のように長いコードでも先頭 2 文字で判定。"""
        c = ResultCoalescer()
        assert c._join_text("yes", "OK", "en-US") == "yes OK"


# === _merge() テスト ===


class TestMerge:
    """2 つの TranscriptionResult の結合。"""

    def test_merge_basic(self):
        c = ResultCoalescer()
        a = _make_result("はい", start=1.0, end=1.5, language="ja", confidence=0.9)
        b = _make_result("今日は", start=1.8, end=2.5, language="ja", confidence=0.8)
        merged = c._merge(a, b)

        assert merged.text == "はい今日は"
        assert merged.start_time == 1.0
        assert merged.end_time == 2.5
        assert merged.confidence == 0.8  # min(0.9, 0.8)
        assert merged.language == "ja"
        assert merged.translated_text is None
        assert merged.target_language is None

    def test_merge_english_with_space(self):
        c = ResultCoalescer()
        a = _make_result("yes", start=0.0, end=0.5, language="en")
        b = _make_result("I agree", start=0.8, end=1.5, language="en")
        merged = c._merge(a, b)

        assert merged.text == "yes I agree"

    def test_merge_preserves_source_id(self):
        c = ResultCoalescer()
        a = TranscriptionResult(
            text="はい", start_time=0.0, end_time=0.5, source_id="mic-1"
        )
        b = _make_result("今日は", start=0.8, end=1.5)
        merged = c._merge(a, b)

        assert merged.source_id == "mic-1"

    def test_merge_language_fallback(self):
        """a.language が空なら b.language を使用。"""
        c = ResultCoalescer()
        a = _make_result("yes", language="")
        b = _make_result("OK", language="en")
        merged = c._merge(a, b)

        assert merged.language == "en"


# === push() テスト ===


class TestPush:
    """push() のケースバイケーステスト。"""

    def test_short_result_is_pending(self):
        """短い結果は保留される。"""
        c = ResultCoalescer()
        r = _make_result("はい", start=1.0, end=1.5, language="ja")
        outputs = c.push(r, now=1.5)

        assert outputs == []
        assert c._pending is not None

    def test_long_result_passes_through(self):
        """十分な長さの結果は即確定。"""
        c = ResultCoalescer()
        r = _make_result("今日は天気がいいですね", start=1.0, end=3.0, language="ja")
        outputs = c.push(r, now=3.0)

        assert len(outputs) == 1
        assert outputs[0].text == "今日は天気がいいですね"
        assert c._pending is None

    def test_short_then_long_within_window(self):
        """短文 + 窓内後続 → マージ。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5, language="ja"), now=1.5)

        outputs = c.push(
            _make_result("今日は天気がいいですね", start=2.0, end=4.0, language="ja"),
            now=4.0,
        )

        assert len(outputs) == 1
        assert outputs[0].text == "はい今日は天気がいいですね"
        assert outputs[0].start_time == 1.0
        assert outputs[0].end_time == 4.0

    def test_short_then_long_outside_window(self):
        """短文 + 窓外後続 → flush + 新判定。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5, language="ja"), now=1.5)

        outputs = c.push(
            _make_result("別の話題ですが", start=5.0, end=7.0, language="ja"),
            now=7.0,
        )

        assert len(outputs) == 2
        assert outputs[0].text == "はい"  # flushed pending
        assert outputs[1].text == "別の話題ですが"  # new (not short)

    def test_consecutive_short_results_merge(self):
        """連続短文 → 再保留。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5, language="ja"), now=1.5)

        outputs = c.push(
            _make_result("うん", start=2.0, end=2.3, language="ja"), now=2.3
        )

        # "はいうん" = 4文字 ≤ 4 → 再保留
        assert outputs == []
        assert c._pending is not None
        assert c._pending.text == "はいうん"

    def test_short_outside_window_then_new_short(self):
        """短文 + 窓外の短文 → flush + 新保留。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5, language="ja"), now=1.5)

        outputs = c.push(
            _make_result("うん", start=5.0, end=5.3, language="ja"), now=5.3
        )

        assert len(outputs) == 1
        assert outputs[0].text == "はい"  # flushed
        assert c._pending is not None
        assert c._pending.text == "うん"  # new pending


# === flush() テスト ===


class TestFlush:
    """flush() のタイムアウトと強制 flush。"""

    def test_flush_no_pending(self):
        c = ResultCoalescer()
        assert c.flush(10.0) is None

    def test_flush_within_window(self):
        """窓内ではフラッシュされない。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5), now=1.5)

        assert c.flush(2.0) is None  # 2.0 < 1.5 + 3.0

    def test_flush_timeout(self):
        """タイムアウトでフラッシュ。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5), now=1.5)

        result = c.flush(4.5)  # 4.5 >= 1.5 + 3.0
        assert result is not None
        assert result.text == "はい"
        assert c._pending is None

    def test_flush_force(self):
        """force=True で即時フラッシュ。"""
        c = ResultCoalescer()
        c.push(_make_result("はい", start=1.0, end=1.5), now=1.5)

        result = c.flush(1.5, force=True)
        assert result is not None
        assert result.text == "はい"


# === reset() テスト ===


class TestReset:
    def test_reset_clears_pending(self):
        c = ResultCoalescer()
        c.push(_make_result("はい"), now=0.0)
        assert c._pending is not None

        c.reset()
        assert c._pending is None


# === マージ無効化テスト ===


class TestDisableMerge:
    """max_chars_single_token=0, max_words=0 でマージ無効化。"""

    def test_all_results_pass_through(self):
        c = ResultCoalescer(max_chars_single_token=0, max_words=0)

        out1 = c.push(_make_result("はい", language="ja"), now=0.0)
        out2 = c.push(_make_result("yes", language="en"), now=1.0)

        assert len(out1) == 1
        assert out1[0].text == "はい"
        assert len(out2) == 1
        assert out2[0].text == "yes"
        assert c._pending is None


# === language 未設定時のスペースヒューリスティクス ===


class TestSpaceHeuristic:
    """translator なし（language=""）でもスペース区切り言語が正しく結合される。"""

    def test_english_no_language_with_space_in_text(self):
        """b にスペースがあれば language="" でもスペース挿入。"""
        c = ResultCoalescer()
        assert c._join_text("yes", "I agree", "") == "yes I agree"

    def test_english_no_language_single_words(self):
        """両方単一語でスペースなし → language="" では直接結合。"""
        c = ResultCoalescer()
        # "yes" + "OK" → language 不明で両方スペースなし → 直接結合
        assert c._join_text("yes", "OK", "") == "yesOK"

    def test_japanese_no_language_no_space(self):
        """日本語は language="" でもスペースなしで正しく結合。"""
        c = ResultCoalescer()
        assert c._join_text("はい", "今日は", "") == "はい今日は"

    def test_merge_english_no_language(self):
        """push() 経由で translator なし英語の結合。"""
        c = ResultCoalescer()
        # "yes" is short (3 chars), "I agree with you" is not short
        c.push(_make_result("yes", start=1.0, end=1.5, language=""), now=1.5)
        outputs = c.push(
            _make_result("I agree with you", start=2.0, end=4.0, language=""),
            now=4.0,
        )
        assert len(outputs) == 1
        # "I agree with you" にスペースがあるのでスペース挿入
        assert outputs[0].text == "yes I agree with you"
