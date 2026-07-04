# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy>=2,<3",
#   "opencv-python>=4.10,<5",
#   "pyzmq>=25,<28",
# ]
# ///
"""Display target-filtered model detections in an isolated GUI process."""

from __future__ import annotations

import argparse
import os

# OpenCV's Linux wheel includes XCB but not a native Wayland Qt plugin.
if os.environ.get("WAYLAND_DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
# Desktop Qt themes are not necessarily installed in the isolated environment.
os.environ.pop("QT_STYLE_OVERRIDE", None)

import cv2
import zmq
from detection_visualization import (
    DEFAULT_VISUALIZATION_ENDPOINT,
    decode_annotated_frame,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the viewer command-line interface."""
    parser = argparse.ArgumentParser(
        description="Display annotated frames from model_target_follow.py"
    )
    parser.add_argument("--endpoint", default=DEFAULT_VISUALIZATION_ENDPOINT)
    return parser


def run(endpoint: str) -> None:
    """Display only the newest frame received from the detector process."""
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.connect(endpoint)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    print(f"Waiting for target-filtered detections at {endpoint}", flush=True)
    try:
        while True:
            if subscriber in dict(poller.poll(50)):
                frame = decode_annotated_frame(subscriber.recv())
                cv2.imshow("Reachy model detections", frame)
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
    args = build_parser().parse_args()
    run(args.endpoint)


if __name__ == "__main__":
    main()
