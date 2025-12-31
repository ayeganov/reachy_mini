"""MuJoCo video capture implementation for the MediaCapture pipeline.

This module provides a VideoCaptureProtocol implementation that renders
frames from MuJoCo simulation and publishes them to ZMQ IPC.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import mujoco
import numpy as np
import numpy.typing as npt
from zmq import Socket

from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    MujocoCameraSpecs,
)
from reachy_mini.media.capture import VideoFormat, VideoMetadata

CAMERA_REACHY = "eye_camera"
CAMERA_STUDIO_CLOSE = "studio_close"
CAMERA_SIZES = {CAMERA_REACHY: (1280, 720), CAMERA_STUDIO_CLOSE: (640, 640)}


class MujocoCamera(CameraBase):
    """Virtual camera implementation that wraps MuJoCo renderer.

    Provides camera intrinsics and specs for MuJoCo-rendered frames.

    Note: The renderer is created lazily on first read() call to ensure
    it's created on the same thread that will use it (MuJoCo's OpenGL
    context is not thread-safe).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = CAMERA_REACHY,
        log_level: str = "INFO",
    ) -> None:
        """Initialize MuJoCo camera.

        Args:
            model: MuJoCo model instance.
            data: MuJoCo data instance.
            camera_name: Name of camera in MuJoCo model.
            log_level: Logging level.

        """
        super().__init__(log_level=log_level)
        self._model = model
        self._data = data
        self._camera_name = camera_name
        self._renderer: Optional[mujoco.Renderer] = None
        self._camera_id: int = -1
        self._opened = False

    def open(self) -> None:
        """Validate camera exists and set up specs.

        Note: The actual renderer is created lazily on first read()
        to ensure it's on the correct thread.
        """
        self._camera_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name
        )
        if self._camera_id < 0:
            raise RuntimeError(
                f"Camera '{self._camera_name}' not found in MuJoCo model"
            )

        self.camera_specs = MujocoCameraSpecs()
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K.copy()
        self._opened = True

        size = CAMERA_SIZES.get(self._camera_name, (1280, 720))
        self.logger.info(
            "MuJoCo camera '%s' configured: %dx%d (renderer created on first read)",
            self._camera_name,
            size[0],
            size[1],
        )

    def _ensure_renderer(self) -> None:
        """Create renderer if not already created (must be called from render thread)."""
        if self._renderer is None:
            size = CAMERA_SIZES.get(self._camera_name, (1280, 720))
            self._renderer = mujoco.Renderer(
                self._model,
                height=size[1],
                width=size[0],
            )
            self.logger.info(
                "MuJoCo renderer created for camera '%s'", self._camera_name
            )

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Render a frame from MuJoCo.

        Returns:
            BGR frame as numpy array (converted from MuJoCo's RGB), or None on error.

        """
        if not self._opened:
            return None

        try:
            # Create renderer lazily on the capture thread
            self._ensure_renderer()

            self._renderer.update_scene(self._data, self._camera_id)

            # Disable expensive effects for performance
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

            frame = self._renderer.render()

            # MuJoCo renders RGB, convert to BGR for consistency with OpenCV pipeline
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame.astype(np.uint8)

        except Exception as e:
            self.logger.error("MuJoCo render error: %s", e)
            return None

    def close(self) -> None:
        """Release MuJoCo renderer."""
        self._renderer = None
        self._opened = False
        self.logger.info("MuJoCo camera '%s' closed", self._camera_name)


class MujocoVideoCapture:
    """Video capture implementation that renders from MuJoCo simulation.

    Implements VideoCaptureProtocol for use with MediaCapture pipeline.
    Frames are rendered from MuJoCo's virtual camera and published to ZMQ IPC.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = CAMERA_REACHY,
        zmq_socket: Optional[Socket] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize MuJoCo video capture.

        Args:
            model: MuJoCo model instance.
            data: MuJoCo data instance.
            camera_name: Name of camera in MuJoCo model.
            zmq_socket: ZMQ PUB socket for IPC publishing.
            log_level: Logging level.

        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._mujoco_camera = MujocoCamera(
            model=model,
            data=data,
            camera_name=camera_name,
            log_level=log_level,
        )

        self._zmq_socket = zmq_socket
        self._format = VideoFormat.BGR

        # Capture state
        self._width: int = 0
        self._height: int = 0
        self._fps: int = 30
        self._channels: int = 3
        self._running = False

        # Capture thread
        self._capture_thread: Optional[threading.Thread] = None

        # FPS measurement
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

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """Get camera intrinsic matrix."""
        if self._mujoco_camera.K is not None:
            return self._mujoco_camera.K
        return np.eye(3, dtype=np.float64)

    @property
    def D(self) -> npt.NDArray[np.float64]:
        """Get camera distortion coefficients."""
        if self._mujoco_camera.D is not None:
            return self._mujoco_camera.D
        return np.zeros(5, dtype=np.float64)

    @property
    def camera_specs(self) -> CameraSpecs:
        """Get camera specifications."""
        if self._mujoco_camera.camera_specs is not None:
            return self._mujoco_camera.camera_specs
        return MujocoCameraSpecs()

    def set_zmq_socket(self, socket: Socket) -> None:
        """Set the ZMQ socket for IPC publishing.

        Args:
            socket: ZMQ PUB socket.

        """
        self._zmq_socket = socket

    def open(self) -> bool:
        """Open MuJoCo virtual camera.

        Returns:
            True if successful, False otherwise.

        """
        try:
            self._mujoco_camera.open()

            width, height = self._mujoco_camera.resolution
            self._width = width
            self._height = height
            self._fps = self._mujoco_camera.framerate

            self._logger.info(
                "MuJoCo capture opened: %dx%d@%dfps",
                self._width,
                self._height,
                self._fps,
            )
            return True

        except Exception as e:
            self._logger.error("Failed to open MuJoCo capture: %s", e)
            return False

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from MuJoCo renderer."""
        return self._mujoco_camera.read()

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
            name="MujocoVideoCapture",
            daemon=True,
        )
        self._capture_thread.start()
        self._logger.info("MuJoCo capture thread started")

    def stop(self) -> None:
        """Stop the capture thread."""
        self._running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._logger.info("MuJoCo capture thread stopped")

    def close(self) -> None:
        """Close MuJoCo capture."""
        self.stop()
        self._mujoco_camera.close()
        self._logger.info("MuJoCo capture closed")

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set resolution (limited support in MuJoCo).

        Note: MuJoCo camera resolution is fixed at creation time.
        This logs a warning if a different resolution is requested.

        Args:
            resolution: The CameraResolution to set.

        """
        current = self._mujoco_camera._resolution
        if current != resolution:
            self._logger.warning(
                "MuJoCo camera resolution is fixed. Requested %s but using %s",
                resolution.name,
                current.name if current else "default",
            )

    def _capture_loop(self) -> None:
        """Capture loop that reads frames and publishes to ZMQ IPC."""
        target_interval = 1.0 / self._fps if self._fps > 0 else 1.0 / 30.0

        while self._running:
            loop_start = time.monotonic()

            frame = self.read_frame()
            if frame is not None and self._zmq_socket is not None:
                self._publish_frame(frame)

            # Update FPS measurement
            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._measured_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            # Sleep to maintain target FPS
            loop_duration = time.monotonic() - loop_start
            sleep_time = target_interval - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Publish frame to ZMQ IPC socket.

        Args:
            frame: BGR frame to publish.

        """
        if self._zmq_socket is None:
            return

        try:
            K_list = self.K.tolist() if self.K is not None else None
            D_list = self.D.tolist() if self.D is not None else None

            metadata = VideoMetadata(
                ts=time.monotonic(),
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if frame.ndim > 2 else 1,
                format=self._format,
                dtype=str(frame.dtype),
                K=K_list,
                D=D_list,
            )

            self._zmq_socket.send_multipart(
                [metadata.to_json().encode("utf-8"), frame.tobytes()],
                copy=False,
            )

        except Exception as e:
            self._logger.error("Failed to publish frame: %s", e)
