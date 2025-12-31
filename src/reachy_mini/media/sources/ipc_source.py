"""IPC-based media sources for daemon-side publishing.

These sources read from ZeroMQ IPC sockets published by MediaCapture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import zmq

from reachy_mini.media.capture import AudioMetadata, EncodedVideoMetadata, VideoMetadata
from reachy_mini.media.media_constants import AUDIO_IPC_ENDPOINT, VIDEO_IPC_ENDPOINT
from reachy_mini.media.publishers.base import AudioData, MediaChunk, VideoData

if TYPE_CHECKING:
    from zmq import Context, Socket


class IPCVideoSource:
    """Reads video frames from IPC socket published by MediaCapture."""

    def __init__(
        self,
        endpoint: str = VIDEO_IPC_ENDPOINT,
        hwm: int = 2,
        log_level: str = "INFO",
    ) -> None:
        """Initialize IPC video source.

        Args:
            endpoint: ZMQ IPC endpoint to subscribe to.
            hwm: High water mark for the socket.
            log_level: Logging level string.

        """
        self._endpoint = endpoint
        self._hwm = hwm
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._context: Optional[Context] = None
        self._socket: Optional[Socket] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if source is open and ready."""
        return self._is_open

    def open(self) -> bool:
        """Initialize the IPC subscriber socket."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self._endpoint)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.set_hwm(self._hwm)
            self._is_open = True
            self._logger.info("Video IPC source connected to %s", self._endpoint)
            return True
        except Exception as e:
            self._logger.error("Failed to open video IPC source: %s", e)
            self.close()
            return False

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        if self._socket is None:
            return False
        return self._socket.poll(timeout=timeout_ms) != 0

    def read(self) -> Optional[MediaChunk[VideoData, VideoMetadata]]:
        """Read next video frame from IPC."""
        if self._socket is None:
            return None

        try:
            parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(parts) != 2:
                self._logger.warning("Invalid video message: %d parts", len(parts))
                return None

            metadata = VideoMetadata.from_json(parts[0].decode("utf-8"))
            frame_bytes = parts[1]

            frame = np.frombuffer(frame_bytes, dtype=np.dtype(metadata.dtype))
            frame = frame.reshape((metadata.height, metadata.width, metadata.channels))

            return MediaChunk(data=frame, metadata=metadata)

        except zmq.Again:
            return None
        except Exception as e:
            self._logger.error("Error reading video from IPC: %s", e)
            return None

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._logger.info("Video IPC source closed")


class IPCH264VideoSource:
    """Reads H.264 encoded video frames from IPC socket.

    Unlike IPCVideoSource which reads raw frames, this source reads
    pre-encoded H.264 data from Picamera2H264Capture.
    """

    def __init__(
        self,
        endpoint: str = VIDEO_IPC_ENDPOINT,
        hwm: int = 2,
        log_level: str = "INFO",
    ) -> None:
        """Initialize IPC H.264 video source.

        Args:
            endpoint: ZMQ IPC endpoint to subscribe to.
            hwm: High water mark for the socket.
            log_level: Logging level string.

        """
        self._endpoint = endpoint
        self._hwm = hwm
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._context: Optional[Context] = None
        self._socket: Optional[Socket] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if source is open and ready."""
        return self._is_open

    def open(self) -> bool:
        """Initialize the IPC subscriber socket."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self._endpoint)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.set_hwm(self._hwm)
            self._is_open = True
            self._logger.info("H.264 video IPC source connected to %s", self._endpoint)
            return True
        except Exception as e:
            self._logger.error("Failed to open H.264 video IPC source: %s", e)
            self.close()
            return False

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        if self._socket is None:
            return False
        return self._socket.poll(timeout=timeout_ms) != 0

    def read(self) -> Optional[MediaChunk[bytes, EncodedVideoMetadata]]:
        """Read next encoded H.264 frame from IPC.

        Returns:
            MediaChunk containing raw H.264 bytes and metadata, or None.

        """
        if self._socket is None:
            return None

        try:
            parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(parts) != 2:
                self._logger.warning("Invalid H.264 message: %d parts", len(parts))
                return None

            metadata = EncodedVideoMetadata.from_json(parts[0].decode("utf-8"))
            frame_bytes = bytes(parts[1])

            return MediaChunk(data=frame_bytes, metadata=metadata)

        except zmq.Again:
            return None
        except Exception as e:
            self._logger.error("Error reading H.264 from IPC: %s", e)
            return None

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._logger.info("H.264 video IPC source closed")


class IPCAudioSource:
    """Reads audio chunks from IPC socket published by MediaCapture."""

    def __init__(
        self,
        endpoint: str = AUDIO_IPC_ENDPOINT,
        hwm: int = 10,
        log_level: str = "INFO",
    ) -> None:
        """Initialize IPC audio source.

        Args:
            endpoint: ZMQ IPC endpoint to subscribe to.
            hwm: High water mark for the socket.
            log_level: Logging level string.

        """
        self._endpoint = endpoint
        self._hwm = hwm
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._context: Optional[Context] = None
        self._socket: Optional[Socket] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if source is open and ready."""
        return self._is_open

    def open(self) -> bool:
        """Initialize the IPC subscriber socket."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self._endpoint)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.set_hwm(self._hwm)
            self._is_open = True
            self._logger.info("Audio IPC source connected to %s", self._endpoint)
            return True
        except Exception as e:
            self._logger.error("Failed to open audio IPC source: %s", e)
            self.close()
            return False

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        if self._socket is None:
            return False
        return self._socket.poll(timeout=timeout_ms) != 0

    def read(self) -> Optional[MediaChunk[AudioData, AudioMetadata]]:
        """Read next audio chunk from IPC."""
        if self._socket is None:
            return None

        try:
            parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(parts) != 2:
                self._logger.warning("Invalid audio message: %d parts", len(parts))
                return None

            metadata = AudioMetadata.from_json(parts[0].decode("utf-8"))
            audio_bytes = parts[1]

            audio = np.frombuffer(audio_bytes, dtype=np.dtype(metadata.dtype))
            if metadata.channels > 1:
                audio = audio.reshape((metadata.samples, metadata.channels))

            return MediaChunk(data=audio, metadata=metadata)

        except zmq.Again:
            return None
        except Exception as e:
            self._logger.error("Error reading audio from IPC: %s", e)
            return None

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._logger.info("Audio IPC source closed")
