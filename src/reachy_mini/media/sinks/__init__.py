"""Media sink implementations.

This package provides various media sink implementations that conform to
MediaSinkProtocol for use with publishers.
"""

from reachy_mini.media.sinks.zeromq_sink import (
    JPEGEncodedZMQSink,
    ZeroMQAudioSink,
    ZeroMQClientSink,
    ZeroMQServerSink,
)

__all__ = [
    "JPEGEncodedZMQSink",
    "ZeroMQAudioSink",
    "ZeroMQClientSink",
    "ZeroMQServerSink",
]