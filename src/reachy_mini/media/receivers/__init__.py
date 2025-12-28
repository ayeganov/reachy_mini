"""Media receivers module.

This module contains receiver implementations that consume media from
various sources and expose a unified MediaClient interface.
"""

from reachy_mini.media.receivers.base import MediaClient
from reachy_mini.media.receivers.local_ipc_receiver import LocalIPCReceiver
from reachy_mini.media.receivers.zeromq_client import ZeroMQClient

__all__ = [
    "MediaClient",
    "LocalIPCReceiver",
    "ZeroMQClient",
]
