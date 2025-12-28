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
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy.signal
import sounddevice as sd
import soundfile as sf
import zmq

from reachy_mini.media.audio_control_utils import ReSpeaker, init_respeaker_usb
from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import CameraResolution, CameraSpecs
from reachy_mini.media.camera_opencv import OpenCVCamera
from reachy_mini.media.media_constants import (
    AUDIO_IPC_ENDPOINT,
    AUDIO_OUTPUT_TCP_PORT,
    AUDIO_OUTPUT_TOPIC,
    PLAY_SOUND_TOPIC,
    VIDEO_IPC_ENDPOINT,
)
from reachy_mini.utils.constants import ASSETS_ROOT_PATH

if TYPE_CHECKING:
    from zmq import Context, Socket


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
        K: Camera intrinsic matrix (3x3) for current resolution.
        D: Camera distortion coefficients (5 values).

    """

    ts: float
    width: int
    height: int
    channels: int
    format: VideoFormat
    dtype: str = "uint8"
    K: Optional[list[list[float]]] = None
    D: Optional[list[float]] = None

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        data = {
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format.value,
            "dtype": self.dtype,
        }
        if self.K is not None:
            data["K"] = self.K
        if self.D is not None:
            data["D"] = self.D
        return json.dumps(data)

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
            K=parsed.get("K"),
            D=parsed.get("D"),
        )


@dataclass
class EncodedVideoMetadata:
    """Metadata for encoded video frames (e.g. JPEG).

    Attributes:
        ts: Timestamp in seconds (monotonic clock).
        width: Frame width in pixels.
        height: Frame height in pixels.
        encoding: Encoding format (e.g., 'jpeg').
        quality: Encoding quality.
        K: Camera intrinsic matrix (3x3) for current resolution.
        D: Camera distortion coefficients (5 values).

    """

    ts: float
    width: int
    height: int
    encoding: str = "jpeg"
    quality: int = 85
    K: Optional[list[list[float]]] = None
    D: Optional[list[float]] = None

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        data = {
            "ts": self.ts,
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
            "quality": self.quality,
        }
        if self.K is not None:
            data["K"] = self.K
        if self.D is not None:
            data["D"] = self.D
        return json.dumps(data)

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
            K=parsed.get("K"),
            D=parsed.get("D"),
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
        doa_rad: Optional direction of arrival in radians.
        doa_is_speech: Optional speech detection flag.

    """

    ts: float
    sample_rate: int
    channels: int
    samples: int
    dtype: str = "float32"
    doa_rad: Optional[float] = None
    doa_is_speech: Optional[bool] = None

    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        data = {
            "ts": self.ts,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "dtype": self.dtype,
        }
        if self.doa_rad is not None:
            data["doa_rad"] = self.doa_rad
        if self.doa_is_speech is not None:
            data["doa_is_speech"] = self.doa_is_speech
        return json.dumps(data)

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
            doa_rad=parsed.get("doa_rad"),
            doa_is_speech=parsed.get("doa_is_speech"),
        )


@dataclass
class CaptureConfig:
    """Configuration for MediaCapture.

    Attributes:
        video_enabled: Whether to capture video.
        audio_enabled: Whether to capture audio.
        video_ipc_endpoint: ZMQ IPC endpoint for video.
        audio_ipc_endpoint: ZMQ IPC endpoint for audio.
        video_resolution: Camera resolution to use.
        audio_sample_rate: Audio sample rate in Hz.
        audio_channels: Number of audio channels.
        log_level: Logging level string.

    """

    video_enabled: bool = True
    audio_enabled: bool = True
    video_ipc_endpoint: str = VIDEO_IPC_ENDPOINT
    audio_ipc_endpoint: str = AUDIO_IPC_ENDPOINT
    video_resolution: CameraResolution = CameraResolution.R1920x1080at60fps
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

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """Get camera intrinsic matrix for current resolution."""
        ...

    @property
    def D(self) -> npt.NDArray[np.float64]:
        """Get camera distortion coefficients."""
        ...

    @property
    def camera_specs(self) -> CameraSpecs:
        """Get camera specifications."""
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

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution.

        Args:
            resolution: The CameraResolution to set.

        """
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

        # Camera intrinsics
        self._camera_specs: Optional[CameraSpecs] = None
        self._resized_K: Optional[npt.NDArray[np.float64]] = None

    @property
    def camera(self) -> CameraBase:
        """Get underlying camera supplying frame data."""
        raise NotImplementedError("Subclasses must implement this property")

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

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution.

        Args:
            resolution: The CameraResolution to set.

        """
        raise NotImplementedError("Subclasses must implement set_resolution()")

    def _capture_loop(self) -> None:
        """Run the main capture loop in a separate thread."""
        while self._running:
            loop_start = time.monotonic()

            frame = self.read_frame()
            if frame is not None:
                self._publish_frame(frame)
                self._update_fps_stats()

            # Recalculate target frame time each iteration to pick up resolution changes
            target_frame_time = 1.0 / self._fps if self._fps > 0 else 0.033
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

        K = self.camera.K.tolist() if self.camera.K is not None else None
        D = self.camera.D.tolist() if self.camera.D is not None else None
        metadata = VideoMetadata(
            ts=time.monotonic(),
            width=frame.shape[1],
            height=frame.shape[0],
            channels=frame.shape[2] if frame.ndim == 3 else 1,
            format=self._format,
            dtype=str(frame.dtype),
            K=K,
            D=D,
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

    Delegates to OpenCVCamera for camera operations and intrinsics.
    """

    def __init__(
        self,
        resolution: CameraResolution = CameraResolution.R1280x720at30fps,
        udp_camera: Optional[str] = None,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize OpenCV capture.

        Args:
            resolution: Camera resolution to use.
            udp_camera: Optional UDP stream URL (for Mujoco simulation).
            zmq_socket: ZMQ PUB socket for broadcasting.
            log_level: Logging level string.

        """
        super().__init__(zmq_socket=zmq_socket, log_level=log_level)
        self._resolution = resolution
        self._udp_camera = udp_camera
        self._format = VideoFormat.RGB
        self._camera = OpenCVCamera(log_level=log_level)

    @property
    def camera(self) -> CameraBase:
        """Underlying camera implementation."""
        return self._camera

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """Get camera intrinsic matrix for current resolution."""
        return self._camera.K

    @property
    def D(self) -> npt.NDArray[np.float64]:
        """Get camera distortion coefficients."""
        return self._camera.D

    @property
    def camera_specs(self) -> CameraSpecs:
        """Get camera specifications."""
        return self._camera.camera_specs

    def open(self) -> bool:
        """Open OpenCV video capture device."""
        try:
            self._camera.open(udp_camera=self._udp_camera)
            self._camera.set_resolution(self._resolution)

            self._width, self._height = self._camera.resolution
            self._fps = self._camera.framerate

            self._logger.info(
                "OpenCV capture opened: %dx%d@%dfps",
                self._width,
                self._height,
                self._fps,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to open OpenCV capture: %s", e)
            return False

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from OpenCV capture."""
        try:
            return self._camera.read()
        except Exception as e:
            self._logger.error("Failed to capture frame: %s", e)
            return None

    def close(self) -> None:
        """Close OpenCV capture and release resources."""
        try:
            self._camera.close()
            self._logger.info("OpenCV capture closed")
        except Exception as e:
            self._logger.warning("Error closing OpenCV: %s", e)

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        self._camera.set_resolution(resolution)
        self._width, self._height = self._camera.resolution
        self._fps = resolution.value[2]
        self._logger.info(
            "OpenCV resolution changed to: %dx%d@%dfps",
            self._width,
            self._height,
            self._fps,
        )


class Picamera2Capture(VideoCaptureBase):
    """Video capture implementation using Picamera2.

    Delegates to Picamera2Camera for camera operations and intrinsics.
    """

    def __init__(
        self,
        resolution: CameraResolution = CameraResolution.R1920x1080at60fps,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize Picamera2 capture.

        Args:
            resolution: Camera resolution to use.
            zmq_socket: ZMQ PUB socket for broadcasting.
            log_level: Logging level string.

        """
        super().__init__(zmq_socket=zmq_socket, log_level=log_level)
        self._resolution = resolution
        self._format = VideoFormat.RGB
        from reachy_mini.media.camera_picamera2 import Picamera2Camera

        self._camera = Picamera2Camera(log_level=log_level)

    @property
    def camera(self) -> CameraBase:
        """Underlying camera implementation."""
        return self._camera

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """Get camera intrinsic matrix for current resolution."""
        return self._camera.K

    @property
    def D(self) -> npt.NDArray[np.float64]:
        """Get camera distortion coefficients."""
        return self._camera.D

    @property
    def camera_specs(self) -> CameraSpecs:
        """Get camera specifications."""
        return self._camera.camera_specs

    def open(self) -> bool:
        """Open Picamera2 capture device."""
        try:
            self._camera.open()
            self._camera.set_resolution(self._resolution)

            self._width, self._height = self._camera.resolution
            self._fps = self._camera.framerate

            self._logger.info(
                "Picamera2 capture opened: %dx%d@%dfps",
                self._width,
                self._height,
                self._fps,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to open Picamera2 capture: %s", e)
            return False

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from Picamera2."""
        try:
            return self._camera.read()
        except Exception as e:
            self._logger.error("Failed to capture frame: %s", e)
            return None

    def close(self) -> None:
        """Close Picamera2 and release resources."""
        try:
            self._camera.close()
            self._logger.info("Picamera2 capture closed")
        except Exception as e:
            self._logger.warning("Error closing Picamera2: %s", e)

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        self._camera.set_resolution(resolution)
        self._width, self._height = self._camera.resolution
        self._fps = resolution.value[2]
        self._logger.info(
            "Picamera2 resolution changed to: %dx%d@%dfps",
            self._width,
            self._height,
            self._fps,
        )


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
        self._publish_thread: Optional[threading.Thread] = None
        self._audio_queue: Queue[tuple[npt.NDArray[np.float32], int]] = Queue(
            maxsize=100
        )
        self._respeaker: Optional[ReSpeaker] = init_respeaker_usb()

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
            self._device_id = self._find_respeaker_device()

            if self._device_id is not None:
                device_info = sd.query_devices(self._device_id)
                self._sample_rate = int(device_info["default_samplerate"])
                self._channels = min(device_info["max_input_channels"], 4)
                self._logger.info(
                    "Using ReSpeaker device %s: %sHz, %sch",
                    self._device_id,
                    self._sample_rate,
                    self._channels,
                )
            else:
                self._logger.warning("ReSpeaker not found, using default device")

            return True

        except Exception as e:
            self._logger.error("Failed to initialize audio capture: %s", e)
            return False

    def start(self) -> None:
        """Start audio capture stream."""
        if self._running:
            self._logger.warning("Audio capture already running")
            return

        try:
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
            self._publish_thread = threading.Thread(
                target=self._publish_loop, name="AudioCapture_publish", daemon=True
            )
            self._publish_thread.start()
            self._logger.info("Audio capture started")

        except Exception as e:
            self._logger.error("Failed to start audio capture: %s", e)

    def stop(self) -> None:
        """Stop audio capture stream."""
        self._running = False

        if self._publish_thread is not None:
            self._publish_thread.join(timeout=2.0)
            self._publish_thread = None

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
        if self._respeaker:
            self._respeaker.close()

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """SoundDevice callback for audio data.

        This should be as fast as possible to avoid blocking the audio driver.
        """
        if status:
            self._logger.debug("Audio callback status: %s", status)

        try:
            self._audio_queue.put_nowait((indata.copy(), frames))
        except Full:
            self._logger.warning("Audio capture queue is full, dropping frame.")

    def _publish_loop(self) -> None:
        """Pull audio from the queue and publish over ZMQ."""
        while self._running:
            try:
                indata, frames = self._audio_queue.get(timeout=0.1)
            except Empty:
                continue

            if self._zmq_socket is None:
                continue

            doa_rad, doa_is_speech = None, None
            if self._respeaker:
                try:
                    result = self._respeaker.read("DOA_VALUE_RADIANS")
                    if result:
                        doa_rad, doa_is_speech = float(result[0]), bool(result[1])
                except Exception as e:
                    self._logger.warning("Could not read DoA from ReSpeaker: %s", e)

            metadata = AudioMetadata(
                ts=time.monotonic(),
                sample_rate=self._sample_rate,
                channels=indata.shape[1] if indata.ndim > 1 else 1,
                samples=frames,
                dtype=str(indata.dtype),
                doa_rad=doa_rad,
                doa_is_speech=doa_is_speech,
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

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the video capture resolution.

        Args:
            resolution: The CameraResolution to set.

        """
        if self._video_capture is not None:
            self._video_capture.set_resolution(resolution)
            self._logger.info("MediaCapture resolution changed to %s", resolution.name)
        else:
            self._logger.warning("Cannot set resolution: no video capture available")

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
                resolution=self._config.video_resolution,
                log_level=self._config.log_level,
            )
            self._logger.info("Using Picamera2 capture")
            return picam
        except Exception:
            self._logger.info("Picamera2 not available, trying OpenCV")

        try:
            opencv_cap = OpenCVCapture(
                resolution=self._config.video_resolution,
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


@dataclass
class AudioOutputConfig:
    """Configuration for AudioOutput.

    Attributes:
        tcp_port: TCP port to bind for receiving audio.
        bind_address: Network address to bind to.
        sample_rate: Audio sample rate for playback.
        channels: Number of audio channels for playback.
        buffer_size_ms: Target buffer size in milliseconds for low-latency playback.
        log_level: Logging level string.

    """

    tcp_port: int = AUDIO_OUTPUT_TCP_PORT
    bind_address: str = "*"
    sample_rate: int = 48000
    channels: int = 2
    buffer_size_ms: int = 50
    log_level: str = "INFO"


class AudioOutput:
    """Receives audio from ZeroMQ and plays through speaker.

    Listens on a TCP port for audio samples from remote clients and plays
    them through the local speaker using SoundDevice. Also handles play_sound
    commands for playing pre-recorded sound files.

    Example:
        config = AudioOutputConfig(tcp_port=5557)
        output = AudioOutput(config)
        output.start()
        # ... audio is received and played
        output.stop()

    """

    def __init__(
        self,
        config: Optional[AudioOutputConfig] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize audio output.

        Args:
            config: Audio output configuration.
            log_level: Logging level string.

        """
        self._config = config or AudioOutputConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._zmq_context: Optional[Context] = None
        self._audio_socket: Optional[Socket] = None
        self._command_socket: Optional[Socket] = None

        self._running = False
        self._receive_thread: Optional[threading.Thread] = None

        self._output_stream: Optional[Any] = None
        self._output_buffer: list[npt.NDArray[np.float32]] = []
        self._output_lock = threading.Lock()

        self._output_device_id: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Check if audio output is active."""
        return self._running

    def start(self) -> bool:
        """Start receiving and playing audio.

        Returns:
            True if started successfully, False otherwise.

        """
        if self._running:
            self._logger.warning("AudioOutput already running")
            return True

        try:
            self._init_sounddevice()
            self._init_zmq()
            self._start_output_stream()

            self._running = True
            self._receive_thread = threading.Thread(
                target=self._receive_loop,
                name="AudioOutput_receive",
                daemon=True,
            )
            self._receive_thread.start()

            self._logger.info("AudioOutput started on port %d", self._config.tcp_port)
            return True

        except Exception as e:
            self._logger.error("Failed to start AudioOutput: %s", e)
            self.stop()
            return False

    def stop(self) -> None:
        """Stop receiving and playing audio."""
        self._running = False

        if self._receive_thread is not None:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None

        self._stop_output_stream()
        self._cleanup_zmq()

        self._logger.info("AudioOutput stopped")

    def _init_sounddevice(self) -> None:
        """Initialize SoundDevice and find output device."""
        devices = sd.query_devices()
        respeaker_names = ["Reachy Mini Audio", "respeaker", "ReSpeaker"]

        for idx, device in enumerate(devices):
            for name in respeaker_names:
                if (
                    name.lower() in device["name"].lower()
                    and device["max_output_channels"] > 0
                ):
                    self._output_device_id = idx
                    self._config.sample_rate = int(device["default_samplerate"])
                    self._config.channels = min(device["max_output_channels"], 2)
                    self._logger.info(
                        "Using output device %s: %dHz, %dch",
                        device["name"],
                        self._config.sample_rate,
                        self._config.channels,
                    )
                    return

        self._logger.warning("ReSpeaker not found, using default output device")

    def _init_zmq(self) -> None:
        """Initialize ZMQ context and sockets."""
        self._zmq_context = zmq.Context()

        self._audio_socket = self._zmq_context.socket(zmq.SUB)
        self._audio_socket.setsockopt(zmq.SUBSCRIBE, AUDIO_OUTPUT_TOPIC)
        self._audio_socket.setsockopt(zmq.SUBSCRIBE, PLAY_SOUND_TOPIC)
        self._audio_socket.set_hwm(10)
        addr = f"tcp://{self._config.bind_address}:{self._config.tcp_port}"
        self._audio_socket.bind(addr)
        self._logger.info("AudioOutput ZMQ bound to %s", addr)

    def _start_output_stream(self) -> None:
        """Start the SoundDevice output stream."""
        try:
            self._output_stream = sd.OutputStream(
                device=self._output_device_id,
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="float32",
                callback=self._output_callback,
                blocksize=int(
                    self._config.sample_rate * self._config.buffer_size_ms / 1000
                ),
            )
            self._output_stream.start()
            self._logger.info("Audio output stream started")

        except Exception as e:
            self._logger.error("Failed to start output stream: %s", e)
            raise

    def _stop_output_stream(self) -> None:
        """Stop the SoundDevice output stream."""
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception as e:
                self._logger.warning("Error stopping output stream: %s", e)
            finally:
                self._output_stream = None

        with self._output_lock:
            self._output_buffer.clear()

    def _cleanup_zmq(self) -> None:
        """Clean up ZMQ sockets and context."""
        if self._audio_socket is not None:
            self._audio_socket.close()
            self._audio_socket = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

    def _receive_loop(self) -> None:
        """Receive audio from ZMQ in a loop."""
        while self._running:
            if self._audio_socket is None:
                break

            try:
                if self._audio_socket.poll(timeout=100) == 0:
                    continue

                parts = self._audio_socket.recv_multipart()
                if len(parts) < 2:
                    continue

                topic = parts[0]

                if topic == AUDIO_OUTPUT_TOPIC:
                    self._handle_audio_data(parts)
                elif topic == PLAY_SOUND_TOPIC:
                    self._handle_play_sound(parts)

            except zmq.ZMQError as e:
                if self._running:
                    self._logger.error("ZMQ error in receive loop: %s", e)
            except Exception as e:
                self._logger.error("Error in audio receive loop: %s", e)

    def _process_and_queue_audio(
        self, data: npt.NDArray[np.float32], input_samplerate: int
    ) -> None:
        """Resample, remap channels, and queue audio data for playback."""
        # Resample if necessary
        if input_samplerate != self._config.sample_rate:
            num_samples = int(len(data) * self._config.sample_rate / input_samplerate)
            data = scipy.signal.resample(data, num_samples)

        # Ensure correct channel mapping
        if data.ndim == 1 and self._config.channels > 1:
            data = np.column_stack([data] * self._config.channels)
        elif data.ndim == 2 and data.shape[1] != self._config.channels:
            if data.shape[1] > self._config.channels:
                data = data[:, : self._config.channels]
            else:
                # Duplicate first channel to fill the rest
                data = np.column_stack([data[:, 0]] * self._config.channels)

        with self._output_lock:
            self._output_buffer.append(data.astype(np.float32))

    def _handle_audio_data(self, parts: list[bytes]) -> None:
        """Handle incoming audio data from ZMQ."""
        if len(parts) != 3:
            self._logger.warning("Invalid audio message: %d parts", len(parts))
            return

        try:
            metadata = AudioMetadata.from_json(parts[1].decode("utf-8"))
            audio_bytes = parts[2]

            audio = np.frombuffer(audio_bytes, dtype=np.dtype(metadata.dtype))
            if metadata.channels > 1:
                audio = audio.reshape((-1, metadata.channels))

            self._process_and_queue_audio(audio, metadata.sample_rate)

        except Exception as e:
            self._logger.error("Failed to handle audio data: %s", e)

    def _handle_play_sound(self, parts: list[bytes]) -> None:
        """Handle play_sound command from ZMQ."""
        if len(parts) != 2:
            self._logger.warning("Invalid play_sound message: %d parts", len(parts))
            return

        try:
            sound_file = parts[1].decode("utf-8")
            self._play_sound_file(sound_file)
        except Exception as e:
            self._logger.error("Failed to handle play_sound: %s", e)

    def _play_sound_file(self, sound_file: str) -> None:
        """Play a sound file from local assets."""
        try:
            if not os.path.exists(sound_file):
                file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
                if not os.path.exists(file_path):
                    self._logger.error("Sound file not found: %s", sound_file)
                    return
            else:
                file_path = sound_file

            data, samplerate_in = sf.read(file_path, dtype="float32")
            self._process_and_queue_audio(data, samplerate_in)
            self._logger.info("Playing sound: %s", sound_file)

        except Exception as e:
            self._logger.error("Failed to play sound %s: %s", sound_file, e)

    def _output_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """SoundDevice callback for audio output.

        Args:
            outdata: Output buffer to fill.
            frames: Number of frames requested.
            time_info: Stream time information.
            status: Stream status flags.

        """
        if status:
            self._logger.debug("Output callback status: %s", status)

        with self._output_lock:
            filled = 0
            while filled < frames and self._output_buffer:
                chunk = self._output_buffer[0]
                needed = frames - filled
                available = len(chunk)
                take = min(needed, available)

                outdata[filled : filled + take] = chunk[:take]
                filled += take

                if take < available:
                    self._output_buffer[0] = chunk[take:]
                else:
                    self._output_buffer.pop(0)

            if filled < frames:
                outdata[filled:] = 0
