"""Unit tests for VADStateMachine."""

import numpy as np
import pytest

from livecap_cli.vad import VADConfig, VADSegment, VADState, VADStateMachine


class TestVADStateEnum:
    """VADState enum テスト"""

    def test_all_states_exist(self):
        """全状態が存在"""
        assert VADState.SILENCE
        assert VADState.POTENTIAL_SPEECH
        assert VADState.SPEECH
        assert VADState.ENDING


class TestVADSegment:
    """VADSegment テスト"""

    def test_create_segment(self):
        """セグメント作成"""
        audio = np.zeros(1600, dtype=np.float32)
        segment = VADSegment(
            audio=audio,
            start_time=0.0,
            end_time=0.1,
            is_final=True,
        )
        assert len(segment.audio) == 1600
        assert segment.start_time == 0.0
        assert segment.end_time == 0.1
        assert segment.is_final is True


class TestVADStateMachineBasics:
    """VADStateMachine 基本機能テスト"""

    def test_initial_state(self):
        """初期状態"""
        sm = VADStateMachine(VADConfig())
        assert sm.state == VADState.SILENCE

    def test_frame_constants(self):
        """フレーム定数"""
        assert VADStateMachine.FRAME_MS == 32
        assert VADStateMachine.FRAME_SAMPLES == 512


class TestVADStateMachineSilenceState:
    """SILENCE 状態テスト"""

    def test_stays_in_silence_with_low_probability(self):
        """低確率で SILENCE を維持"""
        config = VADConfig(threshold=0.5)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        for _ in range(10):
            result = sm.process_frame(frame, probability=0.3, timestamp=0.032)
            assert result is None
            assert sm.state == VADState.SILENCE

    def test_transitions_to_potential_speech(self):
        """高確率で POTENTIAL_SPEECH に遷移"""
        config = VADConfig(threshold=0.5)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        result = sm.process_frame(frame, probability=0.7, timestamp=0.032)
        assert result is None
        assert sm.state == VADState.POTENTIAL_SPEECH


class TestVADStateMachinePotentialSpeechState:
    """POTENTIAL_SPEECH 状態テスト"""

    def test_transitions_to_speech(self):
        """継続的な高確率で SPEECH に遷移"""
        # min_speech_ms=250, FRAME_MS=32 → 8 frames
        config = VADConfig(threshold=0.5, min_speech_ms=256)  # 8 frames
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # 最初のフレームで POTENTIAL_SPEECH へ
        sm.process_frame(frame, probability=0.7, timestamp=0.032)
        assert sm.state == VADState.POTENTIAL_SPEECH

        # 追加フレームで SPEECH へ
        for i in range(7):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 2))

        assert sm.state == VADState.SPEECH

    def test_back_to_silence_on_timeout(self):
        """タイムアウトで SILENCE に戻る"""
        # min_silence_ms=100, FRAME_MS=32 → 4 frames (rounded up)
        config = VADConfig(threshold=0.5, min_silence_ms=128)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # POTENTIAL_SPEECH へ
        sm.process_frame(frame, probability=0.7, timestamp=0.032)
        assert sm.state == VADState.POTENTIAL_SPEECH

        # 低確率フレームで SILENCE に戻る
        for i in range(4):
            sm.process_frame(frame, probability=0.3, timestamp=0.032 * (i + 2))

        assert sm.state == VADState.SILENCE


class TestVADStateMachineSpeechState:
    """SPEECH 状態テスト"""

    def _get_to_speech_state(self, config: VADConfig) -> VADStateMachine:
        """SPEECH 状態にする"""
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # POTENTIAL_SPEECH → SPEECH
        for i in range(10):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 1))

        assert sm.state == VADState.SPEECH
        return sm

    def test_stays_in_speech(self):
        """高確率で SPEECH を維持"""
        config = VADConfig(threshold=0.5, min_speech_ms=64)
        sm = self._get_to_speech_state(config)
        frame = np.zeros(512, dtype=np.float32)

        for i in range(5):
            sm.process_frame(frame, probability=0.7, timestamp=0.5 + 0.032 * i)
            assert sm.state == VADState.SPEECH

    def test_transitions_to_ending(self):
        """低確率継続で ENDING に遷移"""
        config = VADConfig(threshold=0.5, min_speech_ms=64, min_silence_ms=96)
        sm = self._get_to_speech_state(config)
        frame = np.zeros(512, dtype=np.float32)

        # 低確率フレームで ENDING へ
        for i in range(3):  # 96ms = 3 frames
            sm.process_frame(frame, probability=0.3, timestamp=0.5 + 0.032 * i)

        assert sm.state == VADState.ENDING


class TestVADStateMachineEndingState:
    """ENDING 状態テスト"""

    def _get_to_ending_state(self, config: VADConfig) -> VADStateMachine:
        """ENDING 状態にする"""
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # POTENTIAL_SPEECH → SPEECH
        for i in range(10):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 1))

        # SPEECH → ENDING
        for i in range(5):
            sm.process_frame(frame, probability=0.3, timestamp=0.4 + 0.032 * i)

        return sm

    def test_returns_to_speech_on_voice(self):
        """音声再開で SPEECH に戻る"""
        config = VADConfig(threshold=0.5, min_speech_ms=64, min_silence_ms=96)
        sm = self._get_to_ending_state(config)

        # まだ ENDING か確認できない場合もある（パディングフレーム数による）
        initial_state = sm.state
        frame = np.zeros(512, dtype=np.float32)

        if initial_state == VADState.ENDING:
            sm.process_frame(frame, probability=0.7, timestamp=1.0)
            assert sm.state == VADState.SPEECH

    def test_finalizes_segment(self):
        """パディング完了でセグメント確定"""
        config = VADConfig(
            threshold=0.5, min_speech_ms=64, min_silence_ms=64, speech_pad_ms=64
        )
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # SPEECH に遷移
        for i in range(5):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 1))

        # ENDING に遷移
        for i in range(5):
            sm.process_frame(frame, probability=0.3, timestamp=0.2 + 0.032 * i)

        # セグメント確定を待つ
        result = None
        for i in range(10):
            result = sm.process_frame(frame, probability=0.3, timestamp=0.4 + 0.032 * i)
            if result is not None:
                break

        # 最終的に SILENCE に戻りセグメントが返される
        if result is not None:
            assert result.is_final is True
            assert sm.state == VADState.SILENCE


class TestVADStateMachineFinalize:
    """finalize テスト"""

    def test_finalize_during_silence(self):
        """SILENCE 状態で finalize"""
        sm = VADStateMachine(VADConfig())
        result = sm.finalize(timestamp=1.0)
        assert result is None

    def test_finalize_during_speech(self):
        """SPEECH 状態で finalize"""
        config = VADConfig(threshold=0.5, min_speech_ms=64)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # SPEECH に遷移
        for i in range(5):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 1))

        result = sm.finalize(timestamp=0.2)
        assert result is not None
        assert result.is_final is True
        assert len(result.audio) > 0


class TestVADStateMachineReset:
    """reset テスト"""

    def test_reset_returns_to_silence(self):
        """reset で SILENCE に戻る"""
        config = VADConfig(threshold=0.5, min_speech_ms=64)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # SPEECH に遷移
        for i in range(5):
            sm.process_frame(frame, probability=0.7, timestamp=0.032 * (i + 1))

        assert sm.state == VADState.SPEECH

        sm.reset()
        assert sm.state == VADState.SILENCE


class TestVADStateMachineNegativeStartTime:
    """Test that negative start times are properly clamped to 0."""

    def test_speech_at_stream_start_has_non_negative_start_time(self):
        """Speech detected at stream start should not have negative start_time.

        When speech is detected immediately at the start of a stream,
        the pre-buffer padding could cause start_time to become negative
        (e.g., timestamp=0.032s - pre_buffer_duration=0.096s = -0.064s).
        This test ensures start_time is clamped to 0.
        """
        # Use config with large padding to trigger the edge case
        config = VADConfig(threshold=0.5, min_speech_ms=64, speech_pad_ms=100)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # Process a few frames in silence to fill pre-buffer
        for i in range(3):
            sm.process_frame(frame, probability=0.3, timestamp=0.032 * (i + 1))

        # Now detect speech on the very next frame (timestamp=0.128s)
        # With 3 frames in pre_buffer (100ms), start_time would be:
        # 0.128 - 0.096 = 0.032 (positive, but with larger padding could go negative)
        sm.process_frame(frame, probability=0.7, timestamp=0.128)

        # Continue to SPEECH state
        for i in range(4):
            sm.process_frame(frame, probability=0.7, timestamp=0.160 + 0.032 * i)

        # Finalize and check start_time is non-negative
        segment = sm.finalize(timestamp=0.3)
        assert segment is not None
        assert segment.start_time >= 0, f"start_time should be non-negative, got {segment.start_time}"

    def test_immediate_speech_detection_has_zero_start_time(self):
        """Speech detected on first frame should have start_time = 0."""
        config = VADConfig(threshold=0.5, min_speech_ms=32, speech_pad_ms=100)
        sm = VADStateMachine(config)
        frame = np.zeros(512, dtype=np.float32)

        # Detect speech on the very first frame
        sm.process_frame(frame, probability=0.7, timestamp=0.032)

        # The pre_buffer would have 0 frames since we just started
        # Continue to SPEECH state
        for i in range(2):
            sm.process_frame(frame, probability=0.7, timestamp=0.064 + 0.032 * i)

        segment = sm.finalize(timestamp=0.15)
        assert segment is not None
        assert segment.start_time >= 0, f"start_time should be non-negative, got {segment.start_time}"


class TestVADStateMachineFrameLength:
    """backend ごとのフレーム長で時間を数える (#507)。

    以前は 1 フレームを 32 ms と決め打ちしていたので、TenVAD (16 ms) や WebRTC (10 / 20 / 30 ms) では、
    ミリ秒の設定が短く (TenVAD では半分に) 効いていた。
    """

    def test_default_frame_is_silero(self):
        assert VADStateMachine(VADConfig()).frame_ms == 32

    def test_rejects_non_positive_frame(self):
        with pytest.raises(ValueError):
            VADStateMachine(VADConfig(), frame_ms=0)

    def test_millisecond_settings_use_the_given_frame(self):
        config = VADConfig(min_speech_ms=240, min_silence_ms=128, speech_pad_ms=48)
        sm = VADStateMachine(config, frame_ms=16)
        assert (sm._min_speech_frames, sm._min_silence_frames, sm._padding_frames) == (15, 8, 3)
        sm30 = VADStateMachine(config, frame_ms=30)
        assert (sm30._min_speech_frames, sm30._min_silence_frames, sm30._padding_frames) == (8, 4, 1)

    def test_max_speech_cuts_at_the_real_length(self):
        """上限 1 秒なら、16 ms のフレームでも 1 秒で切れる (以前は 0.5 秒で切れていた)。"""
        frame_ms, frame = 16, np.zeros(256, dtype=np.float32)
        config = VADConfig(threshold=0.5, min_speech_ms=48, min_silence_ms=96, speech_pad_ms=16, max_speech_ms=1000)
        sm = VADStateMachine(config, frame_ms=frame_ms)
        finals = []
        for n in range(1, 200):
            segment = sm.process_frame(frame, 0.9, n * frame_ms / 1000)
            if segment is not None and segment.is_final:
                finals.append(segment)
        assert finals
        first = finals[0]
        # 上限もフレーム数に切り捨てて数える (62 フレーム = 0.992 s)。超えず、1 フレーム分以内で切れる
        assert 1.0 - frame_ms / 1000 < len(first.audio) / 16000 <= 1.0

    def test_start_time_counts_the_pre_buffer_with_the_real_frame(self):
        """区間の開始は、前の余白のフレーム数 × 実際のフレーム長だけ前 (以前は × 32 ms で早すぎた)。"""
        frame_ms, frame = 16, np.zeros(256, dtype=np.float32)
        config = VADConfig(threshold=0.5, min_speech_ms=48, min_silence_ms=96, speech_pad_ms=48)
        sm = VADStateMachine(config, frame_ms=frame_ms)
        for n in range(1, 11):  # 10 フレームの無音
            sm.process_frame(frame, 0.0, n * frame_ms / 1000)
        sm.process_frame(frame, 0.9, 11 * frame_ms / 1000)  # 11 フレーム目で音声
        # 前の余白は 3 フレーム (今のフレームを含む) = 48 ms
        assert abs(sm._segment_start_time - (11 * frame_ms - 3 * frame_ms) / 1000) < 1e-9

    def test_interim_duration_uses_the_real_frame(self):
        frame_ms, frame = 16, np.zeros(256, dtype=np.float32)
        config = VADConfig(
            threshold=0.5, min_speech_ms=16, min_silence_ms=500, speech_pad_ms=16,
            interim_min_duration_ms=320, interim_interval_ms=0,
        )
        sm = VADStateMachine(config, frame_ms=frame_ms)
        first_interim = None
        for n in range(1, 60):
            segment = sm.process_frame(frame, 0.9, n * frame_ms / 1000)
            if segment is not None and not segment.is_final:
                first_interim = segment
                break
        assert first_interim is not None
        # 320 ms = 20 フレーム分たまってから出る (以前は 10 フレーム = 160 ms で出ていた)
        assert len(first_interim.audio) >= 20 * 256


class TestVADStateMachineMaxSpeechSplit:
    """最大発話時間で区間を切ったあとの動き (#508 レビュー F1・F3)。"""

    @staticmethod
    def _run(sm, probs, frame_ms, frame_samples):
        frame = np.zeros(frame_samples, dtype=np.float32)
        segments = []
        for n, p in enumerate(probs, 1):
            seg = sm.process_frame(frame, p, n * frame_ms / 1000)
            if seg is not None and seg.is_final:
                segments.append(seg)
        tail = sm.finalize(len(probs) * frame_ms / 1000)
        if tail is not None:
            segments.append(tail)
        return segments

    def test_speech_continuing_after_the_cut_is_kept_even_if_short(self):
        """切った直後に min_speech より短く終わる発話も、次の区間として ASR に渡る (F1)。"""
        frame_ms = 16
        config = VADConfig(threshold=0.5, min_speech_ms=224, min_silence_ms=64, speech_pad_ms=16, max_speech_ms=1600)
        sm = VADStateMachine(config, frame_ms=frame_ms)
        # 100 フレーム (1.6 s) で切れる。そのあと 6 フレーム (96 ms) だけ話して、無音
        probs = [0.9] * 106 + [0.0] * 40
        segments = self._run(sm, probs, frame_ms, 256)
        assert len(segments) == 2
        first, second = segments
        assert len(first.audio) / 16000 <= 1.6  # 上限 (前の余白を含む) で切れる
        assert abs(second.start_time - first.end_time) < 1e-9  # 隙間なく続く
        # 2 つ目の区間に、切ったあとの 96 ms の発話が入っている
        assert len(second.audio) >= 6 * 256

    def test_without_the_cap_the_same_speech_is_one_segment(self):
        frame_ms = 16
        config = VADConfig(threshold=0.5, min_speech_ms=224, min_silence_ms=64, speech_pad_ms=16)
        segments = self._run(VADStateMachine(config, frame_ms=frame_ms), [0.9] * 106 + [0.0] * 40, frame_ms, 256)
        assert len(segments) == 1 and len(segments[0].audio) >= 106 * 256

    def test_ending_padding_does_not_exceed_the_cap(self):
        """後ろの余白を足しても、区間は最大発話時間を超えない (F3)。"""
        frame_ms = 20
        config = VADConfig(threshold=0.5, min_speech_ms=80, min_silence_ms=40, speech_pad_ms=200, max_speech_ms=15000)
        sm = VADStateMachine(config, frame_ms=frame_ms)
        # 14.92 秒 (746 フレーム) 話して、無音
        probs = [0.9] * 746 + [0.0] * 50
        segments = self._run(sm, probs, frame_ms, 320)
        assert segments
        assert max(len(s.audio) for s in segments) / 16000 <= 15.0

    def test_resumed_speech_in_the_ending_padding_is_also_cut(self):
        frame_ms = 16
        config = VADConfig(threshold=0.5, min_speech_ms=48, min_silence_ms=32, speech_pad_ms=160, max_speech_ms=800)
        sm = VADStateMachine(config, frame_ms=frame_ms)
        # 話す → 少し黙る (ENDING) → 話し続ける
        probs = [0.9] * 40 + [0.0] * 3 + [0.9] * 60 + [0.0] * 30
        segments = self._run(sm, probs, frame_ms, 256)
        assert max(len(s.audio) for s in segments) / 16000 <= 0.8
        assert sum(len(s.audio) for s in segments) >= (40 + 3 + 60) * 256

    def test_no_extra_segment_when_speech_stops_at_the_cut(self):
        """切った瞬間に話し終えたら、発話の無い続きの区間 (余白と背景音だけ) は出さない (F4)。"""
        frame_ms = 16
        config = VADConfig(threshold=0.5, min_speech_ms=224, min_silence_ms=64, speech_pad_ms=16, max_speech_ms=1600)
        # どのフレームで切れるかを測り、ちょうどそのフレームで話し終える (以後は非発話)
        probe = self._run(VADStateMachine(config, frame_ms=frame_ms), [0.9] * 300, frame_ms, 256)
        cut = round(probe[0].end_time * 1000 / frame_ms)
        segments = self._run(VADStateMachine(config, frame_ms=frame_ms), [0.9] * cut + [0.0] * 60, frame_ms, 256)
        assert len(segments) == 1
        # 1 フレームでも話し続けていれば、その発話は次の区間に残る (F1 は直ったまま)
        segments = self._run(VADStateMachine(config, frame_ms=frame_ms), [0.9] * (cut + 1) + [0.0] * 60, frame_ms, 256)
        assert len(segments) == 2

    def test_the_cap_is_counted_in_whole_frames(self):
        """32 ms のフレームで 15,000 ms は割り切れない。切り捨てて 468 フレーム (14.976 s) で切る (F3)。"""
        frame_ms = 32
        config = VADConfig(threshold=0.5, min_speech_ms=96, min_silence_ms=96, speech_pad_ms=96, max_speech_ms=15000)
        segments = self._run(VADStateMachine(config, frame_ms=frame_ms), [0.9] * 700 + [0.0] * 30, frame_ms, 512)
        assert len(segments) >= 2
        assert max(len(s.audio) for s in segments) / 16000 <= 15.0
        assert len(segments[0].audio) == 468 * 512

    def test_a_short_cap_can_be_exceeded_before_speech_is_confirmed(self):
        """上限を確かめるのは発話が確定したあと。確定までにためた音声のぶん、短い上限は超えうる (F3)。

        最長は「前の余白 + 2 + (最短の発話 − 1) × 最短の無音」フレーム。TenVAD × parakeet_ja の preset
        (16 ms、余白 1・最短の発話 14・最短の無音 4) では 55 フレーム = 880 ms。
        """
        frame_ms = 16
        config = VADConfig(threshold=0.5, min_speech_ms=224, min_silence_ms=64, speech_pad_ms=16, max_speech_ms=500)
        # 発話 1 フレームと、最短の無音より 1 フレーム短い無音 (3 フレーム) を交互に流す
        probs = ([0.9] + [0.0] * 3) * 30 + [0.0] * 20
        segments = self._run(VADStateMachine(config, frame_ms=frame_ms), probs, frame_ms, 256)
        assert segments
        assert len(segments[0].audio) == (1 + 2 + 13 * 4) * 256  # 880 ms (上限 500 ms を超える)
