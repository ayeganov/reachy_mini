"""Base classes and protocols for media publishers.

Publishers subscribe to the IPC bus and republish media over network protocols.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import zmq

from reachy_mini.media.capture import (
    AUDIO_IPC_ENDPOINT,
    VIDEO_IPC_ENDPOINT,
    AudioMetadata,
    VideoMetadata,
)

if TYPE_CHECKING:
    from zmq import Context, Socket


@runtime_checkable
class MediaPublisherProtocol(Protocol):
    """Protocol defining the interface for media publishers.

    Publishers consume media from the IPC bus and republish over
    specific network protocols (ZMQ TCP, WebRTC, WebSocket).
    """

    @property
    def is_running(self) -> bool:
        """Check if publisher is running."""
        ...

    def start(self) -> bool:
        """Start the publisher.

        Returns:
            True if started successfully, False otherwise.

        """
        ...

    def stop(self) -> None:
        """Stop the publisher and release resources."""
        ...


@dataclass
class PublisherConfig:
    """Configuration for media publishers.

    Attributes:
        video_ipc_endpoint: ZMQ IPC endpoint to subscribe for video.
        audio_ipc_endpoint: ZMQ IPC endpoint to subscribe for audio.
        video_enabled: Whether to publish video.
        audio_enabled: Whether to publish audio.
        log_level: Logging level string.

    """

    video_ipc_endpoint: str = VIDEO_IPC_ENDPOINT
    audio_ipc_endpoint: str = AUDIO_IPC_ENDPOINT
    video_enabled: bool = True
    audio_enabled: bool = True
    log_level: str = "INFO"


class PublisherBase:
    """Base implementation for media publishers.

    Handles IPC subscription boilerplate. Subclasses implement
    protocol-specific publishing logic.
    """

    def __init__(
        self,
        config: Optional[PublisherConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize publisher base.

        Args:
            config: Publisher configuration.
            log_level: Logging level string.

        """
        self._config = config or PublisherConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._zmq_context: Optional[Context] = None
        self._video_sub_socket: Optional[Socket] = None
        self._audio_sub_socket: Optional[Socket] = None

        self._running = False
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None

        self._video_frame_count: int = 0
        self._audio_chunk_count: int = 0

    @property
    def is_running(self) -> bool:
        """Check if publisher is running."""
        return self._running

    def start(self) -> bool:
        """Start the publisher.

        Returns:
            True if started successfully, False otherwise.

        """
        if self._running:
            self._logger.warning("Publisher already running")
            return True

        try:
            self._init_ipc_subscribers()
            self._init_output()

            self._running = True

            if self._config.video_enabled:
                self._video_thread = threading.Thread(
                    target=self._video_loop,
                    name=f"{self.__class__.__name__}_video",
                    daemon=True,
                )
                self._video_thread.start()

            if self._config.audio_enabled:
                self._audio_thread = threading.Thread(
                    target=self._audio_loop,
                    name=f"{self.__class__.__name__}_audio",
                    daemon=True,
                )
                self._audio_thread.start()

            self._logger.info("Publisher started")
            return True

        except Exception as e:
            self._logger.error("Failed to start publisher: %s", e)
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the publisher and release resources."""
        self._running = False

        if self._video_thread is not None:
            self._video_thread.join(timeout=2.0)
            self._video_thread = None

        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None

        self._cleanup_ipc()
        self._cleanup_output()
        self._logger.info("Publisher stopped")

    def _init_ipc_subscribers(self) -> None:
        """Initialize ZMQ IPC subscriber sockets."""
        self._zmq_context = zmq.Context()

        if self._config.video_enabled:
            self._video_sub_socket = self._zmq_context.socket(zmq.SUB)
            self._video_sub_socket.connect(self._config.video_ipc_endpoint)
            self._video_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._video_sub_socket.set_hwm(2)
            self._logger.info(
                "Video IPC subscribed to %s", self._config.video_ipc_endpoint
            )

        if self._config.audio_enabled:
            self._audio_sub_socket = self._zmq_context.socket(zmq.SUB)
            self._audio_sub_socket.connect(self._config.audio_ipc_endpoint)
            self._audio_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._audio_sub_socket.set_hwm(10)
            self._logger.info(
                "Audio IPC subscribed to %s", self._config.audio_ipc_endpoint
            )

    def _cleanup_ipc(self) -> None:
        """Clean up IPC subscriber sockets."""
        if self._video_sub_socket is not None:
            self._video_sub_socket.close()
            self._video_sub_socket = None

        if self._audio_sub_socket is not None:
            self._audio_sub_socket.close()
            self._audio_sub_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

    def _init_output(self) -> None:
        """Initialize output transport. Override in subclasses."""
        pass

    def _cleanup_output(self) -> None:
        """Clean up output transport. Override in subclasses."""
        pass

    def _video_loop(self) -> None:
        """Video processing loop. Receives from IPC and publishes."""
        while self._running:
            if self._video_sub_socket is None:
                break

            try:
                if self._video_sub_socket.poll(timeout=100) == 0:
                    continue

                parts = self._video_sub_socket.recv_multipart()
                if len(parts) != 2:
                    self._logger.warning("Invalid video message: %d parts", len(parts))
                    continue

                metadata = VideoMetadata.from_json(parts[0].decode("utf-8"))
                frame_bytes = parts[1]

                frame = np.frombuffer(frame_bytes, dtype=np.dtype(metadata.dtype))
                frame = frame.reshape(
                    (metadata.height, metadata.width, metadata.channels)
                )

                self._process_video_frame(frame, metadata)
                self._video_frame_count += 1

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in video loop: %s", e)
            except Exception as e:
                self._logger.error("Error in video loop: %s", e)

    def _audio_loop(self) -> None:
        """Audio processing loop. Receives from IPC and publishes."""
        while self._running:
            if self._audio_sub_socket is None:
                break

            try:
                if self._audio_sub_socket.poll(timeout=100) == 0:
                    continue

                parts = self._audio_sub_socket.recv_multipart()
                if len(parts) != 2:
                    self._logger.warning("Invalid audio message: %d parts", len(parts))
                    continue

                metadata = AudioMetadata.from_json(parts[0].decode("utf-8"))
                audio_bytes = parts[1]

                audio = np.frombuffer(audio_bytes, dtype=np.dtype(metadata.dtype))
                if metadata.channels > 1:
                    audio = audio.reshape((metadata.samples, metadata.channels))

                self._process_audio_chunk(audio, metadata)
                self._audio_chunk_count += 1

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in audio loop: %s", e)
            except Exception as e:
                self._logger.error("Error in audio loop: %s", e)

    def _process_video_frame(
        self,
        frame: npt.NDArray[np.uint8],
        metadata: VideoMetadata,
    ) -> None:
        """Process and publish a video frame. Override in subclasses.

        Args:
            frame: Video frame as numpy array (H, W, C).
            metadata: Frame metadata.

        """
        pass

    def _process_audio_chunk(
        self,
        audio: npt.NDArray[np.float32],
        metadata: AudioMetadata,
    ) -> None:
        """Process and publish an audio chunk. Override in subclasses.

        Args:
            audio: Audio samples as numpy array.
            metadata: Audio metadata.

        """
        pass
