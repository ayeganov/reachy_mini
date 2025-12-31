"""Media sink implementations.

This package provides various media sink implementations that conform to
MediaSinkProtocol for use with publishers.
"""

from reachy_mini.media.sinks.zeromq_sink import (
    H264PassthroughZMQSink,
    JPEGEncodedZMQSink,
    ZeroMQAudioSink,
    ZeroMQClientSink,
    ZeroMQServerSink,
)

__all__ = [
    "H264PassthroughZMQSink",
    "JPEGEncodedZMQSink",
    "ZeroMQAudioSink",
    "ZeroMQClientSink",
    "ZeroMQServerSink",
]