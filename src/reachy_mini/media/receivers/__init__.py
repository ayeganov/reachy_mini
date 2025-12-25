"""Media receivers module.

This module contains receiver implementations that consume media from
various sources and expose a unified MediaSource interface.
"""

from reachy_mini.media.receivers.base import MediaSource
from reachy_mini.media.receivers.local_ipc_receiver import LocalIPCReceiver
from reachy_mini.media.receivers.zeromq_receiver import ZeroMQReceiver

__all__ = [
    "MediaSource",
    "LocalIPCReceiver",
    "ZeroMQReceiver",
]
