"""Picamera2 camera backend.

This module provides an implementation of the CameraBase class using Picamera2.
"""

from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_constants import (
    CameraResolution,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
)

try:
    from picamera2 import Picamera2
except ImportError as e:
    raise ImportError(
        "The 'picamera2' module is required for Picamera2Camera but could not be imported. "
        "Please install the picamera2 backend: pip install .[picamera2]."
    ) from e

from .camera_base import CameraBase


class Picamera2Camera(CameraBase):
    """Camera implementation using Picamera2."""

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the Picamera2 camera."""
        super().__init__(log_level=log_level)
        self.picam2: Optional[Picamera2] = None
        self._config: Optional[dict] = None
        self._started = False

    def _build_config(self) -> dict:
        if self.picam2 is None:
            raise RuntimeError("Camera is not initialized.")
        if self._resolution is None:
            raise RuntimeError("Camera resolution is not set.")

        frame_duration_us = int(1_000_000 / self.framerate)
        config = self.picam2.create_video_configuration(
            main={
                "size": (self._resolution.value[0], self._resolution.value[1]),
                "format": "RGB888",
            },
            controls={
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
            },
            buffer_count=1,
        )
        config["queue"] = False
        return config

    def set_resolution(self, resolution: CameraResolution) -> None:
        """Set the camera resolution."""
        super().set_resolution(resolution)

        self._resolution = resolution
        if self.picam2 is None:
            return

        was_started = self._started
        if self._started:
            self.picam2.stop()
            self._started = False

        self._config = self._build_config()
        self.picam2.configure(self._config)

        if was_started:
            self.picam2.start()
            self._started = True

    def open(self) -> None:
        """Open the camera using Picamera2."""
        self.picam2 = Picamera2()
        self.camera_specs = cast(CameraSpecs, ReachyMiniLiteCamSpecs)
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        self._config = self._build_config()
        self.picam2.configure(self._config)
        self.picam2.start()
        self._started = True

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the camera.

        Returns:
            The frame as a uint8 numpy array, or None if no frame could be read.

        Raises:
            RuntimeError: If the camera is not opened.

        """
        if self.picam2 is None:
            raise RuntimeError("Camera is not opened.")
        frame = self.picam2.capture_array()
        return cast(npt.NDArray[np.uint8], frame)

    def close(self) -> None:
        """Release the camera resource."""
        if self.picam2 is not None:
            if self._started:
                self.picam2.stop()
                self._started = False
            self.picam2.close()
            self.picam2 = None
