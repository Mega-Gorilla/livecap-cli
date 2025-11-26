"""Unit tests for FileSource."""

from pathlib import Path

import numpy as np
import pytest

from livecap_core.audio_sources import FileSource, DeviceInfo


# テスト用音声ファイルパス
TEST_AUDIO_DIR = Path(__file__).parent.parent / "assets" / "audio"
TEST_WAV_JA = TEST_AUDIO_DIR / "ja" / "jsut_basic5000_0001.wav"
TEST_WAV_EN = TEST_AUDIO_DIR / "en" / "librispeech_1089-134686-0001.wav"


class TestFileSourceBasics:
    """FileSource 基本機能テスト"""

    def test_create_file_source(self):
        """FileSource 作成"""
        source = FileSource(TEST_WAV_JA)
        assert source.file_path == TEST_WAV_JA
        assert source.sample_rate == 16000
        assert source.chunk_ms == 100
        assert source.realtime is False
        assert source.is_active is False

    def test_create_with_custom_params(self):
        """カスタムパラメータで作成"""
        source = FileSource(
            TEST_WAV_JA,
            sample_rate=22050,
            chunk_ms=50,
            realtime=True,
        )
        assert source.sample_rate == 22050
        assert source.chunk_ms == 50
        assert source.realtime is True

    def test_chunk_size_calculation(self):
        """チャンクサイズ計算"""
        # 16kHz, 100ms -> 1600 samples
        source = FileSource(TEST_WAV_JA, sample_rate=16000, chunk_ms=100)
        assert source.chunk_size == 1600

        # 22050Hz, 50ms -> 1102.5 -> 1102 samples (int conversion)
        source = FileSource(TEST_WAV_JA, sample_rate=22050, chunk_ms=50)
        assert source.chunk_size == 1102

    def test_file_not_found(self):
        """存在しないファイル"""
        source = FileSource("nonexistent.wav")
        with pytest.raises(FileNotFoundError):
            source.start()


class TestFileSourceStartStop:
    """FileSource start/stop テスト"""

    def test_start_loads_audio(self):
        """start() でファイル読み込み"""
        source = FileSource(TEST_WAV_JA)
        source.start()
        try:
            assert source.is_active is True
            assert source.duration > 0
        finally:
            source.stop()

    def test_stop_clears_state(self):
        """stop() で状態クリア"""
        source = FileSource(TEST_WAV_JA)
        source.start()
        source.stop()
        assert source.is_active is False
        assert source.duration == 0.0

    def test_context_manager(self):
        """コンテキストマネージャとして使用"""
        with FileSource(TEST_WAV_JA) as source:
            assert source.is_active is True
        assert source.is_active is False

    def test_double_start_is_safe(self):
        """二重 start() は安全"""
        source = FileSource(TEST_WAV_JA)
        source.start()
        source.start()  # Should not raise
        try:
            assert source.is_active is True
        finally:
            source.stop()

    def test_double_stop_is_safe(self):
        """二重 stop() は安全"""
        source = FileSource(TEST_WAV_JA)
        source.start()
        source.stop()
        source.stop()  # Should not raise
        assert source.is_active is False


class TestFileSourceRead:
    """FileSource read テスト"""

    def test_read_returns_chunks(self):
        """read() がチャンクを返す"""
        with FileSource(TEST_WAV_JA) as source:
            chunk = source.read()
            assert chunk is not None
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
            assert len(chunk) == source.chunk_size

    def test_read_all_chunks(self):
        """全チャンクを読み取り"""
        chunks = []
        with FileSource(TEST_WAV_JA) as source:
            while True:
                chunk = source.read()
                if chunk is None:
                    break
                chunks.append(chunk)

        assert len(chunks) > 0
        # 全チャンクが正しいサイズ
        for chunk in chunks:
            assert len(chunk) == 1600  # 16kHz * 100ms

    def test_read_returns_none_at_end(self):
        """ファイル終端で None"""
        with FileSource(TEST_WAV_JA) as source:
            # すべて読み取る
            while source.read() is not None:
                pass
            # 終端後は None
            assert source.read() is None
            assert source.is_active is False


class TestFileSourceIteration:
    """FileSource イテレーション テスト"""

    def test_sync_iteration(self):
        """同期イテレーション"""
        chunks = list(FileSource(TEST_WAV_JA))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_async_iteration(self):
        """非同期イテレーション"""
        import asyncio

        async def run():
            chunks = []
            async for chunk in FileSource(TEST_WAV_JA):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) > 0


class TestFileSourceProperties:
    """FileSource プロパティ テスト"""

    def test_duration_property(self):
        """duration プロパティ"""
        with FileSource(TEST_WAV_JA) as source:
            # 約3秒のファイル
            assert 1.0 < source.duration < 10.0

    def test_position_seconds(self):
        """position_seconds プロパティ"""
        with FileSource(TEST_WAV_JA) as source:
            assert source.position_seconds == 0.0
            source.read()
            # 100ms 進む
            assert abs(source.position_seconds - 0.1) < 0.01

    def test_remaining_seconds(self):
        """remaining_seconds プロパティ"""
        with FileSource(TEST_WAV_JA) as source:
            total = source.duration
            assert source.remaining_seconds == total
            source.read()
            assert abs(source.remaining_seconds - (total - 0.1)) < 0.01


class TestFileSourceReset:
    """FileSource reset テスト"""

    def test_reset_returns_to_start(self):
        """reset() で先頭に戻る"""
        with FileSource(TEST_WAV_JA) as source:
            # 少し読む
            first_chunk = source.read()
            source.read()
            source.read()

            # リセット
            source.reset()

            assert source.position_seconds == 0.0
            reset_chunk = source.read()

            # 同じデータが返る
            np.testing.assert_array_equal(first_chunk, reset_chunk)


class TestFileSourceResampling:
    """FileSource リサンプリング テスト"""

    def test_resample_to_different_rate(self):
        """異なるサンプリングレートへリサンプリング"""
        # テストファイルは 16kHz
        # 8kHz にリサンプリング
        with FileSource(TEST_WAV_JA, sample_rate=8000) as source:
            chunk = source.read()
            # 8kHz * 100ms = 800 samples
            assert len(chunk) == 800

    def test_resample_preserves_duration(self):
        """リサンプリングで再生時間が保持される"""
        with FileSource(TEST_WAV_JA, sample_rate=16000) as src16:
            duration16 = src16.duration

        with FileSource(TEST_WAV_JA, sample_rate=8000) as src8:
            duration8 = src8.duration

        # 誤差 1% 以内
        assert abs(duration16 - duration8) / duration16 < 0.01


class TestFileSourceDeviceInfo:
    """FileSource list_devices テスト"""

    def test_list_devices_returns_empty(self):
        """list_devices() は空リスト"""
        devices = FileSource.list_devices()
        assert devices == []
        assert isinstance(devices, list)


class TestDeviceInfoDataclass:
    """DeviceInfo データクラス テスト"""

    def test_create_device_info(self):
        """DeviceInfo 作成"""
        info = DeviceInfo(
            index=0,
            name="Test Device",
            channels=2,
            sample_rate=48000,
            is_default=True,
        )
        assert info.index == 0
        assert info.name == "Test Device"
        assert info.channels == 2
        assert info.sample_rate == 48000
        assert info.is_default is True

    def test_device_info_is_frozen(self):
        """DeviceInfo は frozen"""
        info = DeviceInfo(
            index=0,
            name="Test",
            channels=1,
            sample_rate=16000,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            info.index = 1

    def test_device_info_default_is_default(self):
        """is_default のデフォルト値"""
        info = DeviceInfo(
            index=0,
            name="Test",
            channels=1,
            sample_rate=16000,
        )
        assert info.is_default is False
