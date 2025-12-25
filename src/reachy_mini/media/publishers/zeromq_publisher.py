"""ZeroMQ network publisher for media streaming.

Subscribes to the IPC bus and republishes encoded media over TCP
for remote clients to consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import numpy.typing as npt
import zmq

from reachy_mini.media.capture import (
    AudioMetadata,
    VideoMetadata,
)
from reachy_mini.media.publishers.base import PublisherBase, PublisherConfig

if TYPE_CHECKING:
    from zmq import Context, Socket

VIDEO_TCP_PORT = 5555
AUDIO_TCP_PORT = 5556
VIDEO_TOPIC = b"reachy_video"
AUDIO_TOPIC = b"reachy_audio"


@dataclass
class ZeroMQPublisherConfig(PublisherConfig):
    """Configuration for ZeroMQ network publisher.

    Attributes:
        video_tcp_port: TCP port for video publishing.
        audio_tcp_port: TCP port for audio publishing.
        video_topic: ZMQ topic for video messages.
        audio_topic: ZMQ topic for audio messages.
        jpeg_quality: JPEG encoding quality (0-100).
        bind_address: Network address to bind to.

    """

    video_tcp_port: int = VIDEO_TCP_PORT
    audio_tcp_port: int = AUDIO_TCP_PORT
    video_topic: bytes = VIDEO_TOPIC
    audio_topic: bytes = AUDIO_TOPIC
    jpeg_quality: int = 85
    bind_address: str = "*"


@dataclass
class EncodedVideoMetadata:
    """Metadata for JPEG-encoded video frames.

    Attributes:
        ts: Timestamp in seconds (monotonic clock).
        width: Frame width in pixels.
        height: Frame height in pixels.
        encoding: Encoding format (e.g., 'jpeg').
        quality: Encoding quality.

    """

    ts: float
    width: int
    height: int
    encoding: str = "jpeg"
    quality: int = 85

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps({
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "quality": self.quality,
        })

    @classmethod
    def from_json(cls, data: str) -> EncodedVideoMetadata:
        """Deserialize metadata from JSON string.

        Args:
            data: JSON string containing video metadata.

        Returns:
            EncodedVideoMetadata instance.

        """
        parsed = json.loads(data)
        return cls(
            ts=parsed["ts"],
            width=parsed["width"],
            height=parsed["height"],
            encoding=parsed.get("encoding", "jpeg"),
            quality=parsed.get("quality", 85),
        )


@dataclass
class NetworkAudioMetadata:
    """Metadata for audio samples over network.

    Attributes:
        ts: Timestamp in seconds (monotonic clock).
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        samples: Number of samples in the chunk.
        dtype: Numpy dtype string.

    """

    ts: float
    sample_rate: int
    channels: int
    samples: int
    dtype: str = "float32"

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps({
            "ts": self.ts,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "dtype": self.dtype,
        })

    @classmethod
    def from_json(cls, data: str) -> NetworkAudioMetadata:
        """Deserialize metadata from JSON string.

        Args:
            data: JSON string containing audio metadata.

        Returns:
            NetworkAudioMetadata instance.

        """
        parsed = json.loads(data)
        return cls(
            ts=parsed["ts"],
            sample_rate=parsed["sample_rate"],
            channels=parsed["channels"],
            samples=parsed["samples"],
            dtype=parsed.get("dtype", "float32"),
        )


class ZeroMQPublisher(PublisherBase):
    """ZeroMQ TCP publisher for encoded media.

    Subscribes to raw media from the IPC bus, encodes video to JPEG,
    and publishes over TCP for remote clients.

    Example:
        config = ZeroMQPublisherConfig(
            video_tcp_port=5555,
            audio_tcp_port=5556,
            jpeg_quality=85,
        )
        publisher = ZeroMQPublisher(config)
        publisher.start()
        # ... video is encoded and published to tcp://*:5555
        # ... audio is published to tcp://*:5556
        publisher.stop()

    """

    def __init__(
        self,
        config: Optional[ZeroMQPublisherConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize ZeroMQ publisher.

        Args:
            config: Publisher configuration.
            log_level: Logging level string.

        """
        self._zmq_config = config or ZeroMQPublisherConfig()
        super().__init__(config=self._zmq_config, log_level=log_level)

        self._video_pub_socket: Optional[Socket] = None
        self._audio_pub_socket: Optional[Socket] = None
        self._output_context: Optional[Context] = None

    def _init_output(self) -> None:
        """Initialize TCP publisher sockets."""
        self._output_context = zmq.Context()

        if self._zmq_config.video_enabled:
            self._video_pub_socket = self._output_context.socket(zmq.PUB)
            self._video_pub_socket.set_hwm(2)
            video_addr = (
                f"tcp://{self._zmq_config.bind_address}:"
                f"{self._zmq_config.video_tcp_port}"
            )
            self._video_pub_socket.bind(video_addr)
            self._logger.info("Video TCP publisher bound to %s", video_addr)

        if self._zmq_config.audio_enabled:
            self._audio_pub_socket = self._output_context.socket(zmq.PUB)
            self._audio_pub_socket.set_hwm(10)
            audio_addr = (
                f"tcp://{self._zmq_config.bind_address}:"
                f"{self._zmq_config.audio_tcp_port}"
            )
            self._audio_pub_socket.bind(audio_addr)
            self._logger.info("Audio TCP publisher bound to %s", audio_addr)

    def _cleanup_output(self) -> None:
        """Clean up TCP publisher sockets."""
        if self._video_pub_socket is not None:
            self._video_pub_socket.close()
            self._video_pub_socket = None

        if self._audio_pub_socket is not None:
            self._audio_pub_socket.close()
            self._audio_pub_socket = None

        if self._output_context is not None:
            self._output_context.term()
            self._output_context = None

    def _process_video_frame(
        self,
        frame: npt.NDArray[np.uint8],
        metadata: VideoMetadata,
    ) -> None:
        """Encode frame to JPEG and publish over TCP.

        Args:
            frame: Video frame as numpy array (H, W, C).
            metadata: Frame metadata.

        """
        if self._video_pub_socket is None:
            return

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._zmq_config.jpeg_quality]
            success, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                self._logger.warning("Failed to encode frame to JPEG")
                return

            encoded_metadata = EncodedVideoMetadata(
                ts=metadata.ts,
                width=metadata.width,
                height=metadata.height,
                encoding="jpeg",
                quality=self._zmq_config.jpeg_quality,
            )

            self._video_pub_socket.send_multipart([
                self._zmq_config.video_topic,
                encoded_metadata.to_json().encode("utf-8"),
                encoded.tobytes(),
            ])

        except Exception as e:
            self._logger.error("Failed to publish video frame: %s", e)

    def _process_audio_chunk(
        self,
        audio: npt.NDArray[np.float32],
        metadata: AudioMetadata,
    ) -> None:
        """Publish audio chunk over TCP.

        Args:
            audio: Audio samples as numpy array.
            metadata: Audio metadata.

        """
        if self._audio_pub_socket is None:
            return

        try:
            network_metadata = NetworkAudioMetadata(
                ts=metadata.ts,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                samples=metadata.samples,
                dtype=str(audio.dtype),
            )

            self._audio_pub_socket.send_multipart([
                self._zmq_config.audio_topic,
                network_metadata.to_json().encode("utf-8"),
                audio.tobytes(),
            ])

        except Exception as e:
            self._logger.error("Failed to publish audio chunk: %s", e)
