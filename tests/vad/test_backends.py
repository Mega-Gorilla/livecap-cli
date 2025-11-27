"""Unit tests for VAD backends."""

from __future__ import annotations

import pytest

from livecap_core.vad.backends import VADBackend


class TestVADBackendProtocol:
    """VADBackend Protocol テスト"""

    def test_protocol_has_required_methods(self):
        """Protocol が必要なメソッドを定義"""
        # Protocol の属性を確認
        assert hasattr(VADBackend, "process")
        assert hasattr(VADBackend, "reset")
        assert hasattr(VADBackend, "frame_size")
        assert hasattr(VADBackend, "name")


class TestSileroVAD:
    """SileroVAD テスト"""

    @pytest.fixture
    def silero_vad(self):
        """SileroVAD インスタンス"""
        try:
            from livecap_core.vad.backends import SileroVAD

            return SileroVAD(onnx=True)
        except ImportError:
            pytest.skip("silero-vad not installed")

    def test_frame_size(self, silero_vad):
        """フレームサイズ"""
        assert silero_vad.frame_size == 512

    def test_name(self, silero_vad):
        """バックエンド名"""
        assert silero_vad.name == "silero"


class TestWebRTCVAD:
    """WebRTCVAD テスト"""

    @pytest.fixture
    def webrtc_vad(self):
        """WebRTCVAD インスタンス"""
        try:
            from livecap_core.vad.backends import WebRTCVAD

            return WebRTCVAD(mode=3, frame_duration_ms=20)
        except ImportError:
            pytest.skip("webrtcvad not installed")

    def test_frame_size(self, webrtc_vad):
        """フレームサイズ（20ms @ 16kHz = 320 samples）"""
        assert webrtc_vad.frame_size == 320

    def test_name(self, webrtc_vad):
        """バックエンド名"""
        assert webrtc_vad.name == "webrtc_mode3"

    def test_invalid_mode_raises(self):
        """無効なモードでエラー"""
        try:
            from livecap_core.vad.backends import WebRTCVAD
        except ImportError:
            pytest.skip("webrtcvad not installed")

        with pytest.raises(ValueError, match="mode must be 0-3"):
            WebRTCVAD(mode=5)

    def test_invalid_frame_duration_raises(self):
        """無効なフレーム長でエラー"""
        try:
            from livecap_core.vad.backends import WebRTCVAD
        except ImportError:
            pytest.skip("webrtcvad not installed")

        with pytest.raises(ValueError, match="frame_duration_ms must be one of"):
            WebRTCVAD(frame_duration_ms=25)


class TestTenVAD:
    """TenVAD テスト"""

    @pytest.fixture
    def tenvad(self):
        """TenVAD インスタンス"""
        try:
            from livecap_core.vad.backends import TenVAD

            # 警告を抑制してインスタンス作成
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return TenVAD(hop_size=256)
        except ImportError:
            pytest.skip("ten-vad not installed")

    def test_frame_size(self, tenvad):
        """フレームサイズ"""
        assert tenvad.frame_size == 256

    def test_name(self, tenvad):
        """バックエンド名"""
        assert tenvad.name == "tenvad"

    def test_license_warning(self, tenvad):
        """ライセンス警告が表示される（fixture 経由でテスト）"""
        # tenvad fixture で警告が抑制されるため、別途インスタンス作成でテスト
        # fixture が成功した = ten-vad がインストール済み
        import warnings

        from livecap_core.vad.backends import TenVAD

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TenVAD()
            assert len(w) == 1
            assert "limited license" in str(w[0].message).lower()

    def test_invalid_hop_size_raises(self):
        """無効な hop_size でエラー"""
        try:
            from livecap_core.vad.backends import TenVAD
        except ImportError:
            pytest.skip("ten-vad not installed")

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match="hop_size must be one of"):
                TenVAD(hop_size=200)
