# ruff: noqa: D100,D103

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from detection_visualization import (  # noqa: E402
    DetectionVisualizationPublisher,
    decode_annotated_frame,
    encode_annotated_frame,
)
from model_detection_viewer import build_parser as build_viewer_parser  # noqa: E402


def test_visualization_frame_round_trip() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.rectangle(frame, (5, 6), (30, 25), (0, 0, 255), -1)

    decoded = decode_annotated_frame(encode_annotated_frame(frame))

    assert decoded.shape == frame.shape
    assert decoded.dtype == np.uint8
    assert int(decoded[12, 12, 2]) > 200


def test_decode_rejects_non_jpeg_message() -> None:
    with pytest.raises(ValueError, match="not a JPEG"):
        decode_annotated_frame(b"not an image")


def test_publisher_sends_frame_to_subscriber(tmp_path: Path) -> None:
    endpoint = f"ipc://{tmp_path}/detections.sock"
    context = zmq.Context()
    publisher = DetectionVisualizationPublisher(endpoint, context=context)
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect(endpoint)
    try:
        time.sleep(0.05)
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        for _ in range(3):
            publisher.publish(frame)
            if subscriber.poll(100):
                break

        assert subscriber.poll(1000)
        assert decode_annotated_frame(subscriber.recv()).shape == frame.shape
        assert publisher.published_frames >= 1
    finally:
        subscriber.close(linger=0)
        publisher.close()
        context.term()


def test_viewer_uses_default_local_endpoint() -> None:
    args = build_viewer_parser().parse_args([])

    assert args.endpoint == "ipc:///tmp/reachy-model-target-viewer"
