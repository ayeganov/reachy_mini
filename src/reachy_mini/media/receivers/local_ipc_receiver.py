"""Local IPC receiver for on-robot media consumption.

Subscribes directly to the IPC bus for low-latency media access
when running on the same machine as the MediaCapture producer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
import zmq

from reachy_mini.media.capture import AudioMetadata, VideoMetadata
from reachy_mini.media.media_constants import AUDIO_IPC_ENDPOINT, VIDEO_IPC_ENDPOINT

if TYPE_CHECKING:
    from zmq import Context, Socket

CONNECTION_TIMEOUT_SEC = 2.0


@dataclass
class LocalIPCReceiverConfig:
    """Configuration for Local IPC receiver.

    Attributes:
        video_ipc_endpoint: ZMQ IPC endpoint for video.
        audio_ipc_endpoint: ZMQ IPC endpoint for audio.
        video_enabled: Whether to receive video.
        audio_enabled: Whether to receive audio.
        receive_timeout_ms: Socket receive timeout in milliseconds.
        log_level: Logging level string.

    """

    video_ipc_endpoint: str = VIDEO_IPC_ENDPOINT
    audio_ipc_endpoint: str = AUDIO_IPC_ENDPOINT
    video_enabled: bool = True
    audio_enabled: bool = True
    receive_timeout_ms: int = 100
    log_level: str = "INFO"


class LocalIPCReceiver:
    """Local IPC receiver for on-robot media consumption.

    Subscribes directly to the IPC bus, providing low-latency access
    to raw media frames when running on the same machine as the
    MediaCapture producer.

    Example:
        receiver = LocalIPCReceiver()
        receiver.start()

        frame = receiver.get_frame()
        if frame is not None:
            # Process BGR frame (raw, no JPEG encoding)
            pass

        receiver.close()

    """

    def __init__(
        self,
        config: Optional[LocalIPCReceiverConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize Local IPC receiver.

        Args:
            config: Receiver configuration.
            log_level: Logging level string.

        """
        self._config = config or LocalIPCReceiverConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._zmq_context: Optional[Context] = None
        self._video_socket: Optional[Socket] = None
        self._audio_socket: Optional[Socket] = None

        self._running = False
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None

        self._video_lock = threading.Lock()
        self._audio_lock = threading.Lock()

        self._latest_frame: Optional[npt.NDArray[np.uint8]] = None
        self._latest_frame_metadata: Optional[dict[str, object]] = None
        self._latest_audio: Optional[npt.NDArray[np.float32]] = None
        self._latest_audio_metadata: Optional[dict[str, object]] = None

        self._video_resolution: Optional[tuple[int, int]] = None
        self._audio_sample_rate: Optional[int] = None

        self._last_video_receive_time: float = 0.0
        self._last_audio_receive_time: float = 0.0

        # Camera intrinsics from video stream
        self._K: Optional[npt.NDArray[np.float64]] = None
        self._D: Optional[npt.NDArray[np.float64]] = None

    @property
    def is_connected(self) -> bool:
        """Check if receiver is actively receiving data from producer.

        Returns True if data has been received within the connection timeout.
        """
        if not self._running:
            return False

        now = time.monotonic()
        video_active = (
            not self._config.video_enabled
            or (now - self._last_video_receive_time) < CONNECTION_TIMEOUT_SEC
        )
        audio_active = (
            not self._config.audio_enabled
            or (now - self._last_audio_receive_time) < CONNECTION_TIMEOUT_SEC
        )
        return video_active or audio_active

    @property
    def video_resolution(self) -> Optional[tuple[int, int]]:
        """Get current video resolution (width, height)."""
        return self._video_resolution

    @property
    def audio_sample_rate(self) -> Optional[int]:
        """Get current audio sample rate in Hz."""
        return self._audio_sample_rate

    @property
    def K(self) -> Optional[npt.NDArray[np.float64]]:
        """Get camera intrinsic matrix for current resolution."""
        return self._K

    @property
    def D(self) -> Optional[npt.NDArray[np.float64]]:
        """Get camera distortion coefficients."""
        return self._D

    def start(self) -> bool:
        """Start receiving media from IPC bus.

        Returns:
            True if started successfully, False otherwise.

        """
        if self._running:
            self._logger.warning("Receiver already running")
            return True

        try:
            self._zmq_context = zmq.Context()

            if self._config.video_enabled:
                self._video_socket = self._zmq_context.socket(zmq.SUB)
                self._video_socket.connect(self._config.video_ipc_endpoint)
                self._video_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self._video_socket.set_hwm(2)
                self._logger.info(
                    "Video IPC connected to %s", self._config.video_ipc_endpoint
                )

            if self._config.audio_enabled:
                self._audio_socket = self._zmq_context.socket(zmq.SUB)
                self._audio_socket.connect(self._config.audio_ipc_endpoint)
                self._audio_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self._audio_socket.set_hwm(10)
                self._logger.info(
                    "Audio IPC connected to %s", self._config.audio_ipc_endpoint
                )

            self._running = True

            if self._config.video_enabled:
                self._video_thread = threading.Thread(
                    target=self._video_receive_loop,
                    name="LocalIPCReceiver_video",
                    daemon=True,
                )
                self._video_thread.start()

            if self._config.audio_enabled:
                self._audio_thread = threading.Thread(
                    target=self._audio_receive_loop,
                    name="LocalIPCReceiver_audio",
                    daemon=True,
                )
                self._audio_thread.start()

            self._logger.info("Local IPC receiver started")
            return True

        except Exception as e:
            self._logger.error("Failed to start receiver: %s", e)
            self.close()
            return False

    def close(self) -> None:
        """Stop receiving and release resources."""
        self._running = False

        if self._video_thread is not None:
            self._video_thread.join(timeout=2.0)
            self._video_thread = None

        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None

        if self._video_socket is not None:
            self._video_socket.close()
            self._video_socket = None

        if self._audio_socket is not None:
            self._audio_socket.close()
            self._audio_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        self._logger.info("Local IPC receiver closed")

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the latest available video frame.

        Returns:
            BGR numpy array (H, W, 3) or None if no frame available.

        """
        with self._video_lock:
            return self._latest_frame

    def get_frame_with_metadata(
        self,
    ) -> Optional[tuple[npt.NDArray[np.uint8], dict[str, object]]]:
        """Return the latest frame with its metadata.

        Returns:
            Tuple of (frame, metadata_dict) or None if no frame available.

        """
        with self._video_lock:
            if self._latest_frame is None or self._latest_frame_metadata is None:
                return None
            return self._latest_frame, self._latest_frame_metadata

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Return the latest audio chunk.

        Returns:
            Audio samples as numpy array or None if no audio available.

        """
        with self._audio_lock:
            audio = self._latest_audio
            self._latest_audio = None
            return audio

    def get_audio_sample_with_metadata(
        self,
    ) -> Optional[tuple[npt.NDArray[np.float32], dict[str, object]]]:
        """Return the latest audio chunk with its metadata.

        Returns:
            Tuple of (audio, metadata_dict) or None if no audio available.

        """
        with self._audio_lock:
            if self._latest_audio is None or self._latest_audio_metadata is None:
                return None
            audio = self._latest_audio
            metadata = self._latest_audio_metadata
            self._latest_audio = None
            self._latest_audio_metadata = None
            return audio, metadata

    def _video_receive_loop(self) -> None:
        """Video receiving loop."""
        while self._running:
            if self._video_socket is None:
                break

            try:
                if self._video_socket.poll(timeout=self._config.receive_timeout_ms) == 0:
                    continue

                parts = self._video_socket.recv_multipart()
                if len(parts) != 2:
                    self._logger.warning("Invalid video message: %d parts", len(parts))
                    continue

                metadata = VideoMetadata.from_json(parts[0].decode("utf-8"))
                frame_bytes = parts[1]

                frame = np.frombuffer(frame_bytes, dtype=np.dtype(metadata.dtype))
                frame = frame.reshape(
                    (metadata.height, metadata.width, metadata.channels)
                )

                with self._video_lock:
                    self._latest_frame = frame
                    self._latest_frame_metadata = {
                        "ts": metadata.ts,
                        "width": metadata.width,
                        "height": metadata.height,
                        "format": metadata.format.value,
                    }
                    self._video_resolution = (metadata.width, metadata.height)
                    # Update camera intrinsics from metadata
                    if metadata.K is not None:
                        self._K = np.array(metadata.K, dtype=np.float64)
                    if metadata.D is not None:
                        self._D = np.array(metadata.D, dtype=np.float64)

                self._last_video_receive_time = time.monotonic()

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in video receive: %s", e)
            except Exception as e:
                self._logger.error("Error in video receive loop: %s", e)

    def _audio_receive_loop(self) -> None:
        """Audio receiving loop."""
        while self._running:
            if self._audio_socket is None:
                break

            try:
                if self._audio_socket.poll(timeout=self._config.receive_timeout_ms) == 0:
                    continue

                parts = self._audio_socket.recv_multipart()
                if len(parts) != 2:
                    self._logger.warning("Invalid audio message: %d parts", len(parts))
                    continue

                metadata = AudioMetadata.from_json(parts[0].decode("utf-8"))
                audio_bytes = parts[1]

                audio = np.frombuffer(audio_bytes, dtype=np.dtype(metadata.dtype))
                if metadata.channels > 1:
                    audio = audio.reshape((metadata.samples, metadata.channels))

                with self._audio_lock:
                    self._latest_audio = audio
                    self._latest_audio_metadata = {
                        "ts": metadata.ts,
                        "sample_rate": metadata.sample_rate,
                        "channels": metadata.channels,
                        "samples": metadata.samples,
                        "doa_rad": metadata.doa_rad,
                        "doa_is_speech": metadata.doa_is_speech,
                    }
                    self._audio_sample_rate = metadata.sample_rate

                self._last_audio_receive_time = time.monotonic()

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in audio receive: %s", e)
            except Exception as e:
                self._logger.error("Error in audio receive loop: %s", e)
