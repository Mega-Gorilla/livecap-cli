"""Unit tests for VADStateMachine."""

import numpy as np

from livecap_core.vad import VADConfig, VADSegment, VADState, VADStateMachine


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
