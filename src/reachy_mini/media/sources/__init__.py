"""Media source implementations.

This package provides various media source implementations that conform to
MediaSourceProtocol for use with publishers.
"""

from reachy_mini.media.sources.ipc_source import IPCAudioSource, IPCVideoSource
from reachy_mini.media.sources.local_capture_source import LocalAudioCaptureSource
from reachy_mini.media.sources.sinkable_source import SinkableSource

__all__ = [
    "IPCAudioSource",
    "IPCVideoSource",
    "LocalAudioCaptureSource",
    "SinkableSource",
]