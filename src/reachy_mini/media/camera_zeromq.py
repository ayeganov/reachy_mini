"""ZeroMQ camera implementation for remote media streaming.

Wraps ZeroMQReceiver to provide the CameraBase interface for seamless
integration with MediaManager.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.camera_base import CameraBase
from reachy_mini.media.camera_constants import (
    CameraResolution,
    ReachyMiniWirelessCamSpecs,
)
from reachy_mini.media.receivers import ZeroMQReceiver
from reachy_mini.media.receivers.zeromq_receiver import ZeroMQReceiverConfig


class ZeroMQCamera(CameraBase):
    """Camera implementation using ZeroMQ TCP streaming.

    Connects to a remote ZeroMQ publisher (typically the daemon's
    ZeroMQPublisher) and provides frames through the CameraBase interface.

    Example:
        camera = ZeroMQCamera(host="192.168.1.100")
        camera.open()

        frame = camera.read()
        if frame is not None:
            # Process BGR frame
            pass

        camera.close()

    """

    def __init__(
        self,
        host: str = "localhost",
        video_port: int = 5555,
        log_level: str = "INFO",
    ) -> None:
        """Initialize ZeroMQ camera.

        Args:
            host: Remote host address where ZeroMQPublisher is running.
            video_port: TCP port for video subscription.
            log_level: Logging level string.

        """
        super().__init__(log_level=log_level)
        self._host = host
        self._video_port = video_port
        self._log_level = log_level

        self._receiver: Optional[ZeroMQReceiver] = None
        self._is_open = False

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

    def open(self, udp_camera: Optional[str] = None) -> None:
        """Open the ZeroMQ camera connection.

        Args:
            udp_camera: Ignored for ZeroMQ camera (compatibility with CameraBase).

        """
        if self._is_open:
            self.logger.warning("Camera is already open")
            return

        config = ZeroMQReceiverConfig(
            host=self._host,
            video_port=self._video_port,
            video_enabled=True,
            audio_enabled=False,
            log_level=self._log_level,
        )
        self._receiver = ZeroMQReceiver(config=config, log_level=self._log_level)

        if not self._receiver.start():
            raise RuntimeError(
                f"Failed to connect to ZeroMQ video publisher at {self._host}:{self._video_port}"
            )

        self._is_open = True
        self._resolution = CameraResolution.R1920x1080at30fps

        self.camera_specs = ReachyMiniWirelessCamSpecs()
        self.resized_K = self.camera_specs.K.copy()

        self.logger.info(
            "ZeroMQ camera connected to %s:%s", self._host, self._video_port
        )

    def read(self) -> Optional[npt.NDArray[np.uint8]]:
        """Read a frame from the ZeroMQ stream.

        Returns:
            BGR numpy array (H, W, 3) or None if no frame available.

        """
        if self._receiver is None or not self._is_open:
            return None

        frame = self._receiver.get_frame()
        if frame is not None and self._resolution is not None:
            resolution = self._receiver.video_resolution
            if resolution is not None:
                width, height = resolution
                for res in CameraResolution:
                    if res.value[0] == width and res.value[1] == height:
                        if self._resolution != res:
                            self._resolution = res
                            self._update_intrinsics()
                        break

        return frame

    def _update_intrinsics(self) -> None:
        """Update camera intrinsics based on current resolution."""
        if self.camera_specs is None or self._resolution is None:
            return

        w_ratio = (
            self._resolution.value[0]
            / self.camera_specs.default_resolution.value[0]
        )
        h_ratio = (
            self._resolution.value[1]
            / self.camera_specs.default_resolution.value[1]
        )
        self.resized_K = self.camera_specs.K.copy()
        self.resized_K[0, 0] *= w_ratio
        self.resized_K[1, 1] *= h_ratio
        self.resized_K[0, 2] *= w_ratio
        self.resized_K[1, 2] *= h_ratio

    def close(self) -> None:
        """Close the ZeroMQ camera connection."""
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None

        self._is_open = False
        self.logger.info("ZeroMQ camera closed")

    @property
    def is_connected(self) -> bool:
        """Check if actively receiving frames from publisher."""
        if self._receiver is None:
            return False
        return self._receiver.is_connected
