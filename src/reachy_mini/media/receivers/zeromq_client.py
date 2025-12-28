"""ZeroMQ TCP receiver for remote media consumption.

Connects to a ZeroMQ TCP publisher and provides decoded media
through the MediaSource interface.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import numpy.typing as npt
import zmq

from reachy_mini.media.capture import AudioMetadata, EncodedVideoMetadata
from reachy_mini.media.media_constants import (
    AUDIO_OUTPUT_TCP_PORT,
    AUDIO_OUTPUT_TOPIC,
    AUDIO_TCP_PORT,
    AUDIO_TOPIC,
    PLAY_SOUND_TOPIC,
    VIDEO_TCP_PORT,
    VIDEO_TOPIC,
)
from reachy_mini.media.publishers import GenericMediaPublisher
from reachy_mini.media.sinks import ZeroMQClientSink
from reachy_mini.media.sources import SinkableSource

if TYPE_CHECKING:
    from zmq import Context, Socket

CONNECTION_TIMEOUT_SEC = 2.0


@dataclass
class ZeroMQClientConfig:
    """Configuration for ZeroMQ client.

    Attributes:
        host: Remote host address.
        video_port: TCP port for video subscription.
        audio_port: TCP port for audio subscription.
        video_topic: ZMQ topic for video messages.
        audio_topic: ZMQ topic for audio messages.
        video_enabled: Whether to receive video.
        audio_enabled: Whether to receive audio.
        receive_timeout_ms: Socket receive timeout in milliseconds.
        log_level: Logging level string.

    """

    host: str = "localhost"
    video_port: int = VIDEO_TCP_PORT
    audio_port: int = AUDIO_TCP_PORT
    video_topic: bytes = VIDEO_TOPIC
    audio_topic: bytes = AUDIO_TOPIC
    video_enabled: bool = True
    audio_enabled: bool = True
    receive_timeout_ms: int = 100
    log_level: str = "INFO"


class ZeroMQClient:
    """ZeroMQ TCP client for remote media.

    Connects to a remote ZeroMQ publisher, receives JPEG-encoded video
    and raw audio, decodes them, and provides access through the
    MediaClient interface.

    Example:
        config = ZeroMQReceiverConfig(host="192.168.1.100")
        client = ZeroMQClient(config)
        client.start()

        frame = client.get_frame()
        if frame is not None:
            # Process BGR frame
            pass

        client.close()

    """

    def __init__(
        self,
        config: Optional[ZeroMQClientConfig] = None,
        host: Optional[str] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize ZeroMQ client.

        Args:
            config: Client configuration.
            host: Remote host address (overrides config if provided).
            log_level: Logging level string.

        """
        self._config = config or ZeroMQClientConfig()
        if host is not None:
            self._config.host = host

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
        self._first_video_received: bool = False
        self._first_audio_received: bool = False

        # Camera intrinsics from video stream
        self._K: Optional[npt.NDArray[np.float64]] = None
        self._D: Optional[npt.NDArray[np.float64]] = None

        # For sending audio to the robot
        self._audio_out_source = SinkableSource()
        self._audio_out_sink = ZeroMQClientSink(
            host=self._config.host, port=AUDIO_OUTPUT_TCP_PORT, log_level=log_level
        )
        self._audio_out_publisher = GenericMediaPublisher(
            source=self._audio_out_source,
            sink=self._audio_out_sink,
            topic=AUDIO_OUTPUT_TOPIC,
            name="AudioOutPublisher",
            log_level=log_level,
        )

    @property
    def is_connected(self) -> bool:
        """Check if receiver is actively receiving data from publisher.

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

    def start(self, wait_timeout: float = 0.0) -> bool:
        """Start receiving media.

        Args:
            wait_timeout: If > 0, wait up to this many seconds for
                the first data to arrive before returning.

        Returns:
            True if started successfully (and connected if wait_timeout > 0).

        """
        if self._running:
            self._logger.warning("Receiver already running")
            return True

        try:
            self._zmq_context = zmq.Context()

            if self._config.video_enabled:
                self._video_socket = self._zmq_context.socket(zmq.SUB)
                video_addr = f"tcp://{self._config.host}:{self._config.video_port}"
                self._logger.debug("Connecting video socket to %s...", video_addr)
                self._video_socket.connect(video_addr)
                self._video_socket.setsockopt(zmq.SUBSCRIBE, self._config.video_topic)
                self._video_socket.set_hwm(2)
                self._logger.info("Video socket connected to %s", video_addr)

            if self._config.audio_enabled:
                self._audio_socket = self._zmq_context.socket(zmq.SUB)
                audio_addr = f"tcp://{self._config.host}:{self._config.audio_port}"
                self._logger.debug("Connecting audio socket to %s...", audio_addr)
                self._audio_socket.connect(audio_addr)
                self._audio_socket.setsockopt(zmq.SUBSCRIBE, self._config.audio_topic)
                self._audio_socket.set_hwm(10)
                self._logger.info("Audio socket connected to %s", audio_addr)

            self._running = True

            if self._config.video_enabled:
                self._video_thread = threading.Thread(
                    target=self._video_receive_loop,
                    name="ZeroMQClient_video",
                    daemon=True,
                )
                self._video_thread.start()
                self._logger.debug("Video receive thread started")

            if self._config.audio_enabled:
                self._audio_thread = threading.Thread(
                    target=self._audio_receive_loop,
                    name="ZeroMQClient_audio",
                    daemon=True,
                )
                self._audio_thread.start()
                self._logger.debug("Audio receive thread started")

            self._audio_out_publisher.start()
            self._logger.info("ZeroMQ client started")

            if wait_timeout > 0:
                self._logger.debug("Waiting up to %ss for first data...", wait_timeout)
                if not self.wait_for_connection(wait_timeout):
                    self._logger.warning(
                        "No data received within %ss timeout", wait_timeout
                    )
                    return False
                self._logger.info("Connection established - receiving data")

            return True

        except Exception as e:
            self._logger.error("Failed to start receiver: %s", e)
            self.close()
            return False

    def wait_for_connection(self, timeout: float) -> bool:
        """Wait for connection to be established (first data received).

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if connected within timeout, False otherwise.

        """
        if not self._running:
            return False

        poll_interval = 0.05
        elapsed = 0.0

        while elapsed < timeout:
            if self.is_connected:
                return True
            time.sleep(poll_interval)
            elapsed += poll_interval

        return False

    def push_audio_sample(self, data: npt.NDArray[np.float32], sample_rate: int):
        """Push a chunk of audio data to be sent to the robot.

        Args:
            data (npt.NDArray[np.float32]): Numpy array of audio data.
            sample_rate (int): The sample rate of the audio data.
        """
        if data.ndim > 2 or data.shape[0] == 0:
            self._logger.warning("Invalid audio data shape.")
            return

        num_samples = data.shape[0]
        num_channels = data.shape[1] if data.ndim == 2 else 1

        metadata = AudioMetadata(
            ts=time.time(),
            sample_rate=sample_rate,
            channels=num_channels,
            samples=num_samples,
            dtype=str(data.dtype),
        )
        # The SinkableSource's send method ignores the topic, so it can be empty.
        self._audio_out_source.send(topic=b"", metadata=metadata, data=data)

    def play_sound(self, asset_name: str) -> None:
        """Send a command to the robot to play a pre-shipped audio asset.

        Args:
            asset_name (str): The filename of the asset on the robot (e.g. "go_sleep.wav").
        """
        self._logger.info("Requesting robot to play asset: %s", asset_name)
        self._audio_out_sink.send_multipart(
            [PLAY_SOUND_TOPIC, asset_name.encode("utf-8")]
        )

    def stream_sound(self, sound_file: str) -> None:
        """Read a local audio file and stream its raw data to the robot.

        This method reads the entire file into memory and pushes it to the
        send queue for streaming.

        Args:
            sound_file (str): Path to the local audio file (e.g., WAV).
        """
        try:
            import soundfile as sf
        except ImportError:
            self._logger.error(
                "soundfile library is required to stream sounds from file. "
                "Please install it (`pip install soundfile`)."
            )
            return

        try:
            audio_data, sample_rate = sf.read(sound_file, dtype="float32")
            self.push_audio_sample(audio_data, sample_rate)
            self._logger.info("Queued '%s' for streaming to robot.", sound_file)

        except Exception as e:
            self._logger.error(f"Failed to stream sound file {sound_file}: {e}")

    def close(self) -> None:
        """Stop receiving and release resources."""
        self._running = False
        self._first_video_received = False
        self._first_audio_received = False

        self._audio_out_publisher.stop()

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

        self._logger.info("ZeroMQ client closed")

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
                if (
                    self._video_socket.poll(timeout=self._config.receive_timeout_ms)
                    == 0
                ):
                    continue

                parts = self._video_socket.recv_multipart()
                if len(parts) != 3:
                    self._logger.warning("Invalid video message: %d parts", len(parts))
                    continue

                metadata = EncodedVideoMetadata.from_json(parts[1].decode("utf-8"))
                jpeg_bytes = parts[2]

                frame = self._decode_jpeg(jpeg_bytes)
                receive_time = time.monotonic()
                if frame is not None:
                    with self._video_lock:
                        self._latest_frame = frame
                        self._latest_frame_metadata = {
                            "ts": metadata.ts,
                            "receive_ts": receive_time,
                            "width": metadata.width,
                            "height": metadata.height,
                            "encoding": metadata.encoding,
                        }
                        self._video_resolution = (metadata.width, metadata.height)
                        # Update camera intrinsics from metadata
                        if metadata.K is not None:
                            self._K = np.array(metadata.K, dtype=np.float64)
                        if metadata.D is not None:
                            self._D = np.array(metadata.D, dtype=np.float64)

                    if not self._first_video_received:
                        self._first_video_received = True
                        self._logger.debug(
                            "First video frame received: %dx%d",
                            metadata.width,
                            metadata.height,
                        )

                self._last_video_receive_time = receive_time

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
                if (
                    self._audio_socket.poll(timeout=self._config.receive_timeout_ms)
                    == 0
                ):
                    continue

                parts = self._audio_socket.recv_multipart()
                if len(parts) != 3:
                    self._logger.warning("Invalid audio message: %d parts", len(parts))
                    continue

                metadata = AudioMetadata.from_json(parts[1].decode("utf-8"))
                audio_bytes = parts[2]

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
                    }
                    self._audio_sample_rate = metadata.sample_rate

                if not self._first_audio_received:
                    self._first_audio_received = True
                    self._logger.debug(
                        "First audio sample received: %dHz, %dch",
                        metadata.sample_rate,
                        metadata.channels,
                    )

                self._last_audio_receive_time = time.monotonic()

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in audio receive: %s", e)
            except Exception as e:
                self._logger.error("Error in audio receive loop: %s", e)

    def _decode_jpeg(self, jpeg_bytes: bytes) -> Optional[npt.NDArray[np.uint8]]:
        """Decode JPEG bytes to numpy array.

        Args:
            jpeg_bytes: JPEG encoded image data.

        Returns:
            BGR numpy array or None on error.

        """
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            self._logger.error("Failed to decode JPEG: %s", e)
            return None
