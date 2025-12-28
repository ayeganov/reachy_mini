"""ZeroMQ-based media sinks for network publishing.

These sinks handle the transport layer for publishing media over ZeroMQ TCP.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import cv2
import zmq
import numpy as np
import numpy.typing as npt

from reachy_mini.media.capture import EncodedVideoMetadata, VideoMetadata, AudioMetadata
from reachy_mini.media.publishers.base import MediaSinkProtocol, VideoData, AudioData

if TYPE_CHECKING:
    from zmq import Context, Socket


class BaseZeroMQServerSink:
    """Base ZMQ PUB socket that binds (for daemon/server side)."""

    def __init__(
        self,
        port: int,
        hwm: int = 2,
        log_level: str = "INFO",
    ) -> None:
        """Initialize server sink.

        Args:
            port: TCP port to bind to.
            hwm: High water mark for the socket.
            log_level: Logging level string.

        """
        self._port = port
        self._hwm = hwm
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._context: Optional[Context] = None
        self._socket: Optional[Socket] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if sink is open and ready."""
        return self._is_open

    def open(self) -> bool:
        """Initialize and bind the PUB socket."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.set_hwm(self._hwm)
            self._socket.bind(f"tcp://*:{self._port}")
            self._is_open = True
            self._logger.info("ZeroMQ server sink bound to port %d", self._port)
            return True
        except Exception as e:
            self._logger.error("Failed to bind server sink: %s", e)
            self.close()
            return False

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._logger.info("ZeroMQ server sink closed")

    def _send_multipart(self, parts: list[bytes]) -> None:
        """Helper to send multipart data."""
        if self._socket is None:
            return
        try:
            self._socket.send_multipart(parts, copy=False)
        except Exception as e:
            self._logger.error("Failed to send data: %s", e)


ZeroMQServerSink = BaseZeroMQServerSink



class JPEGEncodedZMQSink(
    BaseZeroMQServerSink, MediaSinkProtocol[VideoMetadata, VideoData]
):
    """Sink that encodes video frames to JPEG before sending over ZMQ."""

    def __init__(
        self,
        port: int,
        hwm: int = 2,
        jpeg_quality: int = 85,
        log_level: str = "INFO",
    ) -> None:
        super().__init__(port, hwm, log_level)
        self._jpeg_quality = jpeg_quality

    def send(self, topic: bytes, metadata: VideoMetadata, data: VideoData) -> None:
        """Encode and send video data.

        Args:
            topic: ZMQ topic.
            metadata: VideoMetadata object.
            data: Video frame (numpy array).
        """
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success, encoded = cv2.imencode(".jpg", data, encode_params)
            if not success:
                self._logger.warning("Failed to encode frame to JPEG")
                return

            encoded_meta = EncodedVideoMetadata(
                ts=metadata.ts,
                width=metadata.width,
                height=metadata.height,
                encoding="jpeg",
                quality=self._jpeg_quality,
            )

            self._send_multipart(
                [
                    topic,
                    encoded_meta.to_json().encode("utf-8"),
                    encoded.tobytes(),
                ]
            )
        except Exception as e:
            self._logger.error("Error encoding/sending video: %s", e)


class ZeroMQAudioSink(
    BaseZeroMQServerSink, MediaSinkProtocol[AudioMetadata, AudioData]
):
    """Sink that serializes audio data before sending over ZMQ."""

    def send(self, topic: bytes, metadata: AudioMetadata, data: AudioData) -> None:
        """Serialize and send audio data.

        Args:
            topic: ZMQ topic.
            metadata: AudioMetadata object.
            data: Audio chunk (numpy array).
        """
        match metadata:
            case AudioMetadata():
                pass
            case _:
                self._logger.error(
                    "Invalid metadata type for audio sink: %s", type(metadata)
                )
                return

        try:
            # Re-serialize metadata to ensure it's clean for the network
            meta_json = metadata.to_json().encode("utf-8")
            data_bytes = data.tobytes()

            self._send_multipart([topic, meta_json, data_bytes])
        except Exception as e:
            self._logger.error("Error sending audio: %s", e)


class ZeroMQClientSink(
    MediaSinkProtocol[
        Union[VideoMetadata, AudioMetadata, EncodedVideoMetadata, bytes],
        Union[VideoData, AudioData, bytes],
    ]
):
    """ZMQ PUB socket that connects (for client side -> daemon)."""

    def __init__(
        self,
        host: str,
        port: int,
        hwm: int = 10,
        log_level: str = "INFO",
    ) -> None:
        """Initialize client sink.

        Args:
            host: Remote host address to connect to.
            port: TCP port to connect to.
            hwm: High water mark for the socket.
            log_level: Logging level string.

        """
        self._host = host
        self._port = port
        self._hwm = hwm
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._context: Optional[Context] = None
        self._socket: Optional[Socket] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if sink is open and ready."""
        return self._is_open

    def open(self) -> bool:
        """Initialize and connect the PUB socket."""
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.set_hwm(self._hwm)
            self._socket.connect(f"tcp://{self._host}:{self._port}")
            self._is_open = True
            self._logger.info(
                "ZeroMQ client sink connected to %s:%d", self._host, self._port
            )
            return True
        except Exception as e:
            self._logger.error("Failed to connect client sink: %s", e)
            self.close()
            return False

    def send(
        self,
        topic: bytes,
        metadata: Union[VideoMetadata, AudioMetadata, EncodedVideoMetadata, bytes],
        data: Union[VideoData, AudioData, bytes],
    ) -> None:
        """Send encoded data with topic prefix.

        Supports both raw bytes (if pre-serialized) and metadata objects.
        """
        if self._socket is None:
            return

        try:
            match metadata:
                case bytes():
                    meta_bytes = metadata
                case VideoMetadata() | AudioMetadata() | EncodedVideoMetadata():
                    meta_bytes = metadata.to_json().encode("utf-8")
                case _:
                    self._logger.error("Unknown metadata format: %s", type(metadata))
                    return

            match data:
                case bytes():
                    data_bytes = data
                case np.ndarray():
                    data_bytes = data.tobytes()
                case _:
                    self._logger.error("Unknown data format: %s", type(data))
                    return

            self._socket.send_multipart([topic, meta_bytes, data_bytes], copy=False)
        except Exception as e:
            self._logger.error("Failed to send data: %s", e)

    def send_multipart(self, parts: list[bytes]) -> None:
        """Send a raw multipart message."""
        if self._socket is None:
            return
        try:
            self._socket.send_multipart(parts, copy=False)
        except Exception as e:
            self._logger.error("Failed to send multipart data: %s", e)

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._logger.info("ZeroMQ client sink closed")
