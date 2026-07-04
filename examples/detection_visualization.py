"""Latest-frame visualization transport for detector examples."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import zmq

DEFAULT_VISUALIZATION_ENDPOINT = "ipc:///tmp/reachy-model-target-viewer"


def encode_annotated_frame(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode one annotated BGR frame for local transport."""
    encoded, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not encoded:
        raise RuntimeError("could not encode visualization frame")
    return buffer.tobytes()


def decode_annotated_frame(message: bytes) -> npt.NDArray[np.uint8]:
    """Decode one transported frame for display."""
    frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("visualization message is not a JPEG frame")
    return cast(npt.NDArray[np.uint8], frame)


class DetectionVisualizationPublisher:
    """Publish annotated frames without waiting for a viewer."""

    def __init__(
        self,
        endpoint: str = DEFAULT_VISUALIZATION_ENDPOINT,
        *,
        context: Any | None = None,
    ) -> None:
        """Bind a latest-frame publisher at the requested endpoint."""
        self._owns_context = context is None
        self._context = context if context is not None else zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self.endpoint = endpoint
        self.published_frames = 0

    def publish(self, frame: npt.NDArray[np.uint8]) -> None:
        """Publish one frame, dropping it rather than blocking control."""
        message = encode_annotated_frame(frame)
        try:
            self._socket.send(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        else:
            self.published_frames += 1

    def close(self) -> None:
        """Release the publisher without waiting for queued frames."""
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()
