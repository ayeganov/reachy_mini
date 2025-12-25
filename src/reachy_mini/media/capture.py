"""Media capture module implementing the Producer side of the IPC architecture.

This module provides classes for capturing video and audio from hardware and
broadcasting raw frames to ZeroMQ IPC sockets for consumption by downstream
publishers.

Architecture:
    MediaCapture (orchestrator) owns:
    - VideoCaptureProtocol implementations (Picamera2Capture, OpenCVCapture)
    - AudioCapture (SoundDevice implementation)
    - ZMQ IPC PUB sockets for video and audio

IPC Message Protocol:
    Frame 1 (Metadata): JSON string containing:
        - ts: timestamp (float)
        - width, height: frame dimensions
        - channels: number of channels
        - format: pixel format (bgr, rgb, yuv)
        - dtype: numpy dtype string
    Frame 2 (Payload): Raw bytes (numpy array tobytes())
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import zmq

if TYPE_CHECKING:
    from zmq import Context, Socket

VIDEO_IPC_ENDPOINT = "ipc:///tmp/reachy_video"
AUDIO_IPC_ENDPOINT = "ipc:///tmp/reachy_audio"


class VideoFormat(str, Enum):
    """Video pixel format enumeration."""

    BGR = "bgr"
    RGB = "rgb"
    YUV = "yuv"
    GRAY = "gray"


@dataclass
class VideoMetadata:
    """Metadata for video frames transmitted over IPC.

    Attributes:
        ts: Timestamp in seconds (monotonic clock).
        width: Frame width in pixels.
        height: Frame height in pixels.
        channels: Number of color channels.
        format: Pixel format (bgr, rgb, yuv, gray).
        dtype: Numpy dtype string (e.g., 'uint8').

    """

    ts: float
    width: int
    height: int
    channels: int
    format: VideoFormat
    dtype: str = "uint8"

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps({
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format.value,
            "dtype": self.dtype,
        })

    @classmethod
    def from_json(cls, data: str) -> VideoMetadata:
        """Deserialize metadata from JSON string.

        Args:
            data: JSON string containing video metadata.

        Returns:
            VideoMetadata instance.

        """
        parsed = json.loads(data)
        return cls(
            ts=parsed["ts"],
            width=parsed["width"],
            height=parsed["height"],
            channels=parsed["channels"],
            format=VideoFormat(parsed["format"]),
            dtype=parsed.get("dtype", "uint8"),
        )


@dataclass
class AudioMetadata:
    """Metadata for audio samples transmitted over IPC.

    Attributes:
        ts: Timestamp in seconds (monotonic clock).
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.
        samples: Number of samples in the chunk.
        dtype: Numpy dtype string (e.g., 'float32').

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
    def from_json(cls, data: str) -> AudioMetadata:
        """Deserialize metadata from JSON string.

        Args:
            data: JSON string containing audio metadata.

        Returns:
            AudioMetadata instance.

        """
        parsed = json.loads(data)
        return cls(
            ts=parsed["ts"],
            sample_rate=parsed["sample_rate"],
            channels=parsed["channels"],
            samples=parsed["samples"],
            dtype=parsed.get("dtype", "float32"),
        )


@dataclass
class CaptureConfig:
    """Configuration for MediaCapture.

    Attributes:
        video_enabled: Whether to capture video.
        audio_enabled: Whether to capture audio.
        video_ipc_endpoint: ZMQ IPC endpoint for video.
        audio_ipc_endpoint: ZMQ IPC endpoint for audio.
        video_width: Target video width (may be overridden by hardware).
        video_height: Target video height (may be overridden by hardware).
        video_fps: Target video framerate.
        audio_sample_rate: Audio sample rate in Hz.
        audio_channels: Number of audio channels.
        log_level: Logging level string.

    """

    video_enabled: bool = True
    audio_enabled: bool = True
    video_ipc_endpoint: str = VIDEO_IPC_ENDPOINT
    audio_ipc_endpoint: str = AUDIO_IPC_ENDPOINT
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 30
    audio_sample_rate: int = 16000
    audio_channels: int = 2
    log_level: str = "INFO"


@runtime_checkable
class VideoCaptureProtocol(Protocol):
    """Protocol defining the interface for video capture implementations.

    Implementations must provide methods for opening, reading frames,
    and closing the capture device. This uses structural typing for
    loose coupling and testability.
    """

    @property
    def width(self) -> int:
        """Get current frame width."""
        ...

    @property
    def height(self) -> int:
        """Get current frame height."""
        ...

    @property
    def fps(self) -> int:
        """Get target frames per second."""
        ...

    @property
    def channels(self) -> int:
        """Get number of color channels."""
        ...

    @property
    def format(self) -> VideoFormat:
        """Get pixel format."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        ...

    @property
    def measured_fps(self) -> float:
        """Get measured frames per second."""
        ...

    def open(self) -> bool:
        """Open the video capture device.

        Returns:
            True if successful, False otherwise.

        """
        ...

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a single frame from the capture device.

        Returns:
            Frame as numpy array (H, W, C) or None on error.

        """
        ...

    def close(self) -> None:
        """Close the video capture device and release resources."""
        ...

    def set_zmq_socket(self, socket: Socket) -> None:
        """Set the ZMQ socket for IPC publishing.

        Args:
            socket: ZMQ PUB socket.

        """
        ...

    def start(self) -> None:
        """Start the capture thread."""
        ...

    def stop(self) -> None:
        """Stop the capture thread."""
        ...


@runtime_checkable
class AudioCaptureProtocol(Protocol):
    """Protocol defining the interface for audio capture implementations."""

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        ...

    @property
    def channels(self) -> int:
        """Get number of audio channels."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        ...

    def open(self) -> bool:
        """Open audio capture device.

        Returns:
            True if successful, False otherwise.

        """
        ...

    def start(self) -> None:
        """Start audio capture stream."""
        ...

    def stop(self) -> None:
        """Stop audio capture stream."""
        ...

    def close(self) -> None:
        """Close audio capture and release resources."""
        ...

    def set_zmq_socket(self, socket: Socket) -> None:
        """Set the ZMQ socket for IPC publishing.

        Args:
            socket: ZMQ PUB socket.

        """
        ...


class VideoCaptureBase:
    """Base implementation for video capture with threading and IPC publishing.

    Provides common functionality for all video capture implementations.
    Subclasses must implement open(), read_frame(), and close().
    """

    def __init__(
        self,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize video capture base.

        Args:
            zmq_socket: ZMQ PUB socket for broadcasting frames.
            log_level: Logging level string.

        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._zmq_socket = zmq_socket
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._width: int = 0
        self._height: int = 0
        self._fps: int = 30
        self._channels: int = 3
        self._format: VideoFormat = VideoFormat.BGR

        self._frame_count: int = 0
        self._last_fps_time: float = 0.0
        self._measured_fps: float = 0.0

    @property
    def width(self) -> int:
        """Get current frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get current frame height."""
        return self._height

    @property
    def fps(self) -> int:
        """Get target frames per second."""
        return self._fps

    @property
    def channels(self) -> int:
        """Get number of color channels."""
        return self._channels

    @property
    def format(self) -> VideoFormat:
        """Get pixel format."""
        return self._format

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @property
    def measured_fps(self) -> float:
        """Get measured frames per second."""
        return self._measured_fps

    def open(self) -> bool:
        """Open the video capture device.

        Returns:
            True if successful, False otherwise.

        """
        raise NotImplementedError("Subclasses must implement open()")

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a single frame from the capture device.

        Returns:
            Frame as numpy array (H, W, C) or None on error.

        """
        raise NotImplementedError("Subclasses must implement read_frame()")

    def close(self) -> None:
        """Close the video capture device and release resources."""
        raise NotImplementedError("Subclasses must implement close()")

    def set_zmq_socket(self, socket: Socket) -> None:
        """Set the ZMQ socket for IPC publishing.

        Args:
            socket: ZMQ PUB socket.

        """
        self._zmq_socket = socket

    def start(self) -> None:
        """Start the capture thread."""
        if self._running:
            self._logger.warning("Capture already running")
            return

        self._running = True
        self._frame_count = 0
        self._last_fps_time = time.monotonic()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.__class__.__name__}_capture",
            daemon=True,
        )
        self._capture_thread.start()
        self._logger.info("Video capture started")

    def stop(self) -> None:
        """Stop the capture thread."""
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        self._logger.info("Video capture stopped")

    def _capture_loop(self) -> None:
        """Run the main capture loop in a separate thread."""
        target_frame_time = 1.0 / self._fps if self._fps > 0 else 0.033

        while self._running:
            loop_start = time.monotonic()

            frame = self.read_frame()
            if frame is not None:
                self._publish_frame(frame)
                self._update_fps_stats()

            elapsed = time.monotonic() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Publish frame to ZMQ IPC socket.

        Args:
            frame: Frame as numpy array (H, W, C).

        """
        if self._zmq_socket is None:
            return

        metadata = VideoMetadata(
            ts=time.monotonic(),
            width=frame.shape[1],
            height=frame.shape[0],
            channels=frame.shape[2] if frame.ndim == 3 else 1,
            format=self._format,
            dtype=str(frame.dtype),
        )

        try:
            self._zmq_socket.send_multipart(
                [metadata.to_json().encode("utf-8"), frame.tobytes()],
                copy=False,
            )
        except Exception as e:
            self._logger.error("Failed to publish frame: %s", e)

    def _update_fps_stats(self) -> None:
        """Update FPS measurement statistics."""
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time

        if elapsed >= 1.0:
            self._measured_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now


class OpenCVCapture(VideoCaptureBase):
    """Video capture implementation using OpenCV.

    Provides a fallback capture mechanism when Picamera2 is not available.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize OpenCV capture.

        Args:
            device_id: Video device index or path.
            width: Target frame width.
            height: Target frame height.
            fps: Target frames per second.
            zmq_socket: ZMQ PUB socket for broadcasting.
            log_level: Logging level string.

        """
        super().__init__(zmq_socket=zmq_socket, log_level=log_level)
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Any = None

    def open(self) -> bool:
        """Open OpenCV video capture device.

        Returns:
            True if successful, False otherwise.

        """
        try:
            import cv2

            self._cap = cv2.VideoCapture(self._device_id)
            if not self._cap.isOpened():
                self._logger.error("Failed to open device %s", self._device_id)
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))

            self._width = actual_width
            self._height = actual_height
            self._fps = actual_fps

            self._logger.info(
                "OpenCV capture opened: %dx%d@%dfps",
                self._width, self._height, self._fps,
            )
            return True

        except ImportError:
            self._logger.error("OpenCV (cv2) not available")
            return False
        except Exception as e:
            self._logger.error("Failed to open OpenCV capture: %s", e)
            return False

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from OpenCV capture.

        Returns:
            Frame as BGR numpy array or None on error.

        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        return frame

    def close(self) -> None:
        """Close OpenCV capture and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._logger.info("OpenCV capture closed")


class Picamera2Capture(VideoCaptureBase):
    """Video capture implementation using Picamera2.

    Primary capture method for Raspberry Pi camera module.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize Picamera2 capture.

        Args:
            width: Target frame width.
            height: Target frame height.
            fps: Target frames per second.
            zmq_socket: ZMQ PUB socket for broadcasting.
            log_level: Logging level string.

        """
        super().__init__(zmq_socket=zmq_socket, log_level=log_level)
        self._width = width
        self._height = height
        self._fps = fps
        self._picam2: Any = None
        self._format = VideoFormat.BGR

    def open(self) -> bool:
        """Open Picamera2 capture device.

        Returns:
            True if successful, False otherwise.

        """
        try:
            from picamera2 import Picamera2

            self._picam2 = Picamera2()

            config = self._picam2.create_video_configuration(
                main={
                    "size": (self._width, self._height),
                    "format": "BGR888",
                },
                controls={
                    "FrameRate": self._fps,
                },
            )
            self._picam2.configure(config)
            self._picam2.start()

            self._logger.info(
                "Picamera2 capture opened: %dx%d@%dfps",
                self._width, self._height, self._fps,
            )
            return True

        except ImportError:
            self._logger.error("Picamera2 not available")
            return False
        except Exception as e:
            self._logger.error("Failed to open Picamera2 capture: %s", e)
            return False

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from Picamera2.

        Returns:
            Frame as BGR numpy array or None on error.

        """
        if self._picam2 is None:
            return None

        try:
            frame = self._picam2.capture_array("main")
            return frame
        except Exception as e:
            self._logger.error("Failed to capture frame: %s", e)
            return None

    def close(self) -> None:
        """Close Picamera2 and release resources."""
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception as e:
                self._logger.warning("Error closing Picamera2: %s", e)
            finally:
                self._picam2 = None
                self._logger.info("Picamera2 capture closed")


class AudioCapture:
    """Audio capture implementation using SoundDevice.

    Captures audio from the microphone and publishes to ZMQ IPC socket.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 2,
        chunk_size: int = 1024,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            chunk_size: Samples per audio chunk.
            zmq_socket: ZMQ PUB socket for broadcasting.
            log_level: Logging level string.

        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._zmq_socket = zmq_socket

        self._stream: Any = None
        self._device_id: Optional[int] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Get number of audio channels."""
        return self._channels

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    def set_zmq_socket(self, socket: Socket) -> None:
        """Set the ZMQ socket for IPC publishing.

        Args:
            socket: ZMQ PUB socket.

        """
        self._zmq_socket = socket

    def open(self) -> bool:
        """Open audio capture device.

        Returns:
            True if successful, False otherwise.

        """
        try:
            import sounddevice as sd

            self._device_id = self._find_respeaker_device()

            if self._device_id is not None:
                device_info = sd.query_devices(self._device_id)
                self._sample_rate = int(device_info["default_samplerate"])
                self._channels = min(device_info["max_input_channels"], 4)
                self._logger.info(
                    "Using ReSpeaker device %s: %sHz, %sch",
                    self._device_id, self._sample_rate, self._channels,
                )
            else:
                self._logger.warning("ReSpeaker not found, using default device")

            return True

        except ImportError:
            self._logger.error("SoundDevice not available")
            return False
        except Exception as e:
            self._logger.error("Failed to initialize audio capture: %s", e)
            return False

    def start(self) -> None:
        """Start audio capture stream."""
        if self._running:
            self._logger.warning("Audio capture already running")
            return

        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self._device_id,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self._logger.info("Audio capture started")

        except Exception as e:
            self._logger.error("Failed to start audio capture: %s", e)

    def stop(self) -> None:
        """Stop audio capture stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._logger.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None
                self._logger.info("Audio capture stopped")

    def close(self) -> None:
        """Close audio capture and release resources."""
        self.stop()

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """SoundDevice callback for audio data.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Stream time information.
            status: Stream status flags.

        """
        if status:
            self._logger.debug("Audio callback status: %s", status)

        if self._zmq_socket is None:
            return

        metadata = AudioMetadata(
            ts=time.monotonic(),
            sample_rate=self._sample_rate,
            channels=indata.shape[1] if indata.ndim > 1 else 1,
            samples=frames,
            dtype=str(indata.dtype),
        )

        try:
            self._zmq_socket.send_multipart(
                [metadata.to_json().encode("utf-8"), indata.tobytes()],
                copy=False,
            )
        except Exception as e:
            self._logger.error("Failed to publish audio: %s", e)

    def _find_respeaker_device(self) -> Optional[int]:
        """Find ReSpeaker audio device.

        Returns:
            Device index or None if not found.

        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            respeaker_names = ["Reachy Mini Audio", "respeaker", "ReSpeaker"]

            for idx, device in enumerate(devices):
                for name in respeaker_names:
                    if (
                        name.lower() in device["name"].lower()
                        and device["max_input_channels"] > 0
                    ):
                        return idx

            return None

        except Exception:
            return None


@dataclass
class MediaCaptureStats:
    """Statistics for media capture performance monitoring.

    Attributes:
        video_fps: Measured video frames per second.
        audio_chunks_per_sec: Audio chunks published per second.
        video_frames_total: Total video frames captured.
        audio_samples_total: Total audio samples captured.

    """

    video_fps: float = 0.0
    audio_chunks_per_sec: float = 0.0
    video_frames_total: int = 0
    audio_samples_total: int = 0


class MediaCapture:
    """Main media capture orchestrator.

    Owns video and audio capture devices and manages ZMQ IPC sockets
    for broadcasting raw media to downstream publishers.

    Example:
        config = CaptureConfig(video_enabled=True, audio_enabled=True)
        capture = MediaCapture(config)
        capture.start()
        # ... media is now being broadcast to IPC sockets
        capture.stop()

    """

    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        video_capture: Optional[VideoCaptureProtocol] = None,
        audio_capture: Optional[AudioCaptureProtocol] = None,
    ) -> None:
        """Initialize media capture.

        Args:
            config: Capture configuration. Uses defaults if None.
            video_capture: Custom video capture implementation.
            audio_capture: Custom audio capture implementation.

        """
        self._config = config or CaptureConfig()
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(self._config.log_level)

        self._zmq_context: Optional[Context] = None
        self._video_socket: Optional[Socket] = None
        self._audio_socket: Optional[Socket] = None

        self._video_capture = video_capture
        self._audio_capture = audio_capture

        self._running = False
        self._stats = MediaCaptureStats()

    @property
    def is_running(self) -> bool:
        """Check if capture is active."""
        return self._running

    @property
    def stats(self) -> MediaCaptureStats:
        """Get capture statistics."""
        if self._video_capture is not None:
            self._stats.video_fps = self._video_capture.measured_fps
        return self._stats

    @property
    def video_capture(self) -> Optional[VideoCaptureProtocol]:
        """Get video capture instance."""
        return self._video_capture

    @property
    def audio_capture(self) -> Optional[AudioCaptureProtocol]:
        """Get audio capture instance."""
        return self._audio_capture

    def start(self) -> bool:
        """Start media capture and IPC broadcasting.

        Returns:
            True if started successfully, False otherwise.

        """
        if self._running:
            self._logger.warning("MediaCapture already running")
            return True

        try:
            self._init_zmq()
            self._init_video()
            self._init_audio()

            if self._video_capture is not None and self._config.video_enabled:
                self._video_capture.start()

            if self._audio_capture is not None and self._config.audio_enabled:
                self._audio_capture.start()

            self._running = True
            self._logger.info("MediaCapture started")
            return True

        except Exception as e:
            self._logger.error("Failed to start MediaCapture: %s", e)
            self.stop()
            return False

    def stop(self) -> None:
        """Stop media capture and close resources."""
        self._running = False

        if self._video_capture is not None:
            self._video_capture.stop()
            self._video_capture.close()

        if self._audio_capture is not None:
            self._audio_capture.stop()

        self._cleanup_zmq()
        self._logger.info("MediaCapture stopped")

    def _init_zmq(self) -> None:
        """Initialize ZMQ context and sockets."""
        self._zmq_context = zmq.Context()

        if self._config.video_enabled:
            self._video_socket = self._zmq_context.socket(zmq.PUB)
            self._video_socket.set_hwm(2)
            self._video_socket.bind(self._config.video_ipc_endpoint)
            self._logger.info("Video IPC bound to %s", self._config.video_ipc_endpoint)

        if self._config.audio_enabled:
            self._audio_socket = self._zmq_context.socket(zmq.PUB)
            self._audio_socket.set_hwm(10)
            self._audio_socket.bind(self._config.audio_ipc_endpoint)
            self._logger.info("Audio IPC bound to %s", self._config.audio_ipc_endpoint)

    def _init_video(self) -> None:
        """Initialize video capture if enabled."""
        if not self._config.video_enabled:
            return

        if self._video_capture is None:
            self._video_capture = self._create_video_capture()

        if self._video_capture is not None:
            if self._video_socket is not None:
                self._video_capture.set_zmq_socket(self._video_socket)
            if not self._video_capture.open():
                self._logger.error("Failed to open video capture")
                self._video_capture = None

    def _init_audio(self) -> None:
        """Initialize audio capture if enabled."""
        if not self._config.audio_enabled:
            return

        if self._audio_capture is None:
            self._audio_capture = AudioCapture(
                sample_rate=self._config.audio_sample_rate,
                channels=self._config.audio_channels,
                log_level=self._config.log_level,
            )

        if self._audio_capture is not None:
            if self._audio_socket is not None:
                self._audio_capture.set_zmq_socket(self._audio_socket)
            if not self._audio_capture.open():
                self._logger.error("Failed to open audio capture")
                self._audio_capture = None

    def _create_video_capture(self) -> Optional[VideoCaptureProtocol]:
        """Create appropriate video capture based on platform.

        Returns:
            VideoCaptureProtocol instance or None if creation fails.

        """
        try:
            picam = Picamera2Capture(
                width=self._config.video_width,
                height=self._config.video_height,
                fps=self._config.video_fps,
                log_level=self._config.log_level,
            )
            self._logger.info("Using Picamera2 capture")
            return picam
        except Exception:
            self._logger.info("Picamera2 not available, trying OpenCV")

        try:
            opencv_cap = OpenCVCapture(
                width=self._config.video_width,
                height=self._config.video_height,
                fps=self._config.video_fps,
                log_level=self._config.log_level,
            )
            self._logger.info("Using OpenCV capture")
            return opencv_cap
        except Exception as e:
            self._logger.error("Failed to create video capture: %s", e)
            return None

    def _cleanup_zmq(self) -> None:
        """Clean up ZMQ sockets and context."""
        if self._video_socket is not None:
            self._video_socket.close()
            self._video_socket = None

        if self._audio_socket is not None:
            self._audio_socket.close()
            self._audio_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None
