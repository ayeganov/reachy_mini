# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy>=2,<3",
#   "opencv-python>=4.10,<4.13",
#   "pyzmq>=25,<28",
# ]
# ///
"""Latest-frame transport and isolated OpenCV viewer for model detections."""

from __future__ import annotations

import argparse
import os
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zmq

DEFAULT_ENDPOINT = "ipc:///tmp/reachy-attention-viewer"


def encode_frame(frame: npt.NDArray[np.uint8]) -> bytes:
    """Encode one annotated BGR frame for local transport."""
    import cv2

    encoded, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not encoded:
        raise RuntimeError("could not encode visualization frame")
    return buffer.tobytes()


def decode_frame(message: bytes) -> npt.NDArray[np.uint8]:
    """Decode one transported JPEG frame."""
    import cv2

    frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("visualization message is not a JPEG frame")
    return cast(npt.NDArray[np.uint8], frame)


class DetectionPublisher:
    """Publish annotated frames without waiting for a viewer."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
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

    def publish(self, frame: npt.NDArray[np.uint8]) -> None:
        """Publish one frame, dropping it rather than blocking control."""
        try:
            self._socket.send(encode_frame(frame), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        """Release the publisher without waiting for queued frames."""
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()


def build_parser() -> argparse.ArgumentParser:
    """Build the viewer command-line interface."""
    parser = argparse.ArgumentParser(description="Display Reachy attention detections")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    return parser


def run(endpoint: str) -> None:
    """Display only the newest annotated frame."""
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.pop("QT_STYLE_OVERRIDE", None)

    import cv2

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.connect(endpoint)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    print(f"Waiting for attention detections at {endpoint}", flush=True)
    try:
        while True:
            if subscriber in dict(poller.poll(50)):
                cv2.imshow(
                    "Reachy attention detections", decode_frame(subscriber.recv())
                )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close(linger=0)
        context.term()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main() -> None:
    """Run the isolated detection viewer."""
    run(build_parser().parse_args().endpoint)


if __name__ == "__main__":
    main()
