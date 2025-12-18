"""Unit tests for MicrophoneSource.

These tests use mocks since actual hardware is not available in CI.
Skipped via conftest.py if sounddevice/PortAudio is not available.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from livecap_cli.audio_sources import MicrophoneSource


class TestMicrophoneSourceBasics:
    """MicrophoneSource 基本機能テスト"""

    def test_create_microphone_source(self):
        """MicrophoneSource 作成"""
        source = MicrophoneSource()
        assert source.device is None
        assert source.sample_rate == 16000
        assert source.chunk_ms == 100
        assert source.is_active is False

    def test_create_with_device_index(self):
        """デバイスインデックス指定で作成"""
        source = MicrophoneSource(device=0)
        assert source.device == 0

    def test_create_with_device_name(self):
        """デバイス名指定で作成"""
        source = MicrophoneSource(device="Test Device")
        assert source.device == "Test Device"

    def test_create_with_custom_params(self):
        """カスタムパラメータで作成"""
        source = MicrophoneSource(
            device=1,
            sample_rate=22050,
            chunk_ms=50,
        )
        assert source.device == 1
        assert source.sample_rate == 22050
        assert source.chunk_ms == 50


class TestMicrophoneSourceListDevices:
    """MicrophoneSource list_devices テスト"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices(self, mock_sd):
        """list_devices() がデバイス一覧を返す"""
        mock_sd.query_devices.return_value = [
            {
                "name": "Input Device 1",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "Output Device",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
            {
                "name": "Input Device 2",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
            },
        ]
        mock_sd.default.device = (0, 1)  # (input, output)

        devices = MicrophoneSource.list_devices()

        # 入力デバイスのみ返す
        assert len(devices) == 2
        assert devices[0].name == "Input Device 1"
        assert devices[0].channels == 2
        assert devices[0].is_default is True
        assert devices[1].name == "Input Device 2"
        assert devices[1].channels == 1
        assert devices[1].is_default is False

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_list_devices_empty(self, mock_sd):
        """入力デバイスがない場合"""
        mock_sd.query_devices.return_value = [
            {
                "name": "Output Only",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]
        mock_sd.default.device = (None, 0)

        devices = MicrophoneSource.list_devices()
        assert devices == []


class TestMicrophoneSourceStartStop:
    """MicrophoneSource start/stop テスト（モック使用）"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_start_creates_stream(self, mock_sd):
        """start() がストリームを作成"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()

        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()
        assert source.is_active is True

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_stop_closes_stream(self, mock_sd):
        """stop() がストリームを閉じる"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()
        source.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert source.is_active is False

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_double_start_is_safe(self, mock_sd):
        """二重 start() は安全"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()
        source.start()  # Should not create second stream

        # InputStream は 1 回だけ呼ばれる
        assert mock_sd.InputStream.call_count == 1

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_double_stop_is_safe(self, mock_sd):
        """二重 stop() は安全"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()
        source.stop()
        source.stop()  # Should not raise

        assert source.is_active is False


class TestMicrophoneSourceRead:
    """MicrophoneSource read テスト（モック使用）"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_read_returns_from_queue(self, mock_sd):
        """read() がキューからデータを返す"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()

        # キューに直接データを追加
        test_audio = np.zeros(1600, dtype=np.float32)
        source._queue.put(test_audio)

        chunk = source.read(timeout=1.0)
        assert chunk is not None
        assert len(chunk) == 1600

        source.stop()

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_read_returns_none_on_timeout(self, mock_sd):
        """タイムアウトで None"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        source = MicrophoneSource()
        source.start()

        # キューは空
        chunk = source.read(timeout=0.01)
        assert chunk is None

        source.stop()

    def test_read_returns_none_when_not_active(self):
        """非アクティブ時は None"""
        source = MicrophoneSource()
        chunk = source.read()
        assert chunk is None


class TestMicrophoneSourceContextManager:
    """MicrophoneSource コンテキストマネージャ テスト"""

    @patch("livecap_cli.audio_sources.microphone.sd")
    def test_context_manager(self, mock_sd):
        """コンテキストマネージャとして使用"""
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {"name": "Test Device"}

        with MicrophoneSource() as source:
            assert source.is_active is True

        assert source.is_active is False
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()
