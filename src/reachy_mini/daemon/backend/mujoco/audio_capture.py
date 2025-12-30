"""Null audio capture for backends without audio support.

This module provides a stub AudioCaptureProtocol implementation for
backends like MuJoCo that don't support audio simulation.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zmq import Socket


class NullAudioCapture:
    """No-op audio capture implementation.

    Used for backends that don't support audio (e.g., MuJoCo simulation).
    Implements AudioCaptureProtocol but does nothing.
    """

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize NullAudioCapture.

        Args:
            log_level: Logging level string.

        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)
        self._running = False
        self._sample_rate = 16000
        self._channels = 2

    @property
    def sample_rate(self) -> int:
        """Return default sample rate (not used)."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Return default channels (not used)."""
        return self._channels

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    def open(self) -> bool:
        """Open audio capture (no-op).

        Returns:
            True (always succeeds).

        """
        self._logger.info("NullAudioCapture opened (audio not simulated)")
        return True

    def start(self) -> None:
        """Start audio capture (no-op)."""
        self._running = True
        self._logger.debug("NullAudioCapture started")

    def stop(self) -> None:
        """Stop audio capture (no-op)."""
        self._running = False
        self._logger.debug("NullAudioCapture stopped")

    def close(self) -> None:
        """Close audio capture (no-op)."""
        self.stop()
        self._logger.debug("NullAudioCapture closed")

    def set_zmq_socket(self, socket: "Socket") -> None:
        """Set ZMQ socket (ignored - no audio to publish).

        Args:
            socket: ZMQ PUB socket (ignored).

        """
        pass
