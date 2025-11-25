"""Audio source module for livecap_core.

Provides abstract base class and concrete implementations for audio input.
"""

from typing import TYPE_CHECKING

from .base import AudioSource, DeviceInfo
from .file import FileSource

# MicrophoneSource は遅延インポート
# sounddevice が PortAudio を必要とし、CI環境等では利用できないため
if TYPE_CHECKING:
    from .microphone import MicrophoneSource

__all__ = [
    "AudioSource",
    "DeviceInfo",
    "FileSource",
    "MicrophoneSource",
]


def __getattr__(name: str):
    """遅延インポート for MicrophoneSource."""
    if name == "MicrophoneSource":
        from .microphone import MicrophoneSource

        return MicrophoneSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
