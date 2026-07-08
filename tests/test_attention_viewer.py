# ruff: noqa: D100,D103

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

from reachy_mini.tools.attention_viewer import (
    DetectionPublisher,
    build_parser,
    decode_frame,
    encode_frame,
)


def test_visualization_frame_round_trip() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.rectangle(frame, (5, 6), (30, 25), (0, 0, 255), -1)

    decoded = decode_frame(encode_frame(frame))

    assert decoded.shape == frame.shape
    assert int(decoded[12, 12, 2]) > 200


def test_decode_rejects_non_jpeg_message() -> None:
    with pytest.raises(ValueError, match="not a JPEG"):
        decode_frame(b"not an image")


def test_publisher_sends_latest_frame_without_blocking(tmp_path: Path) -> None:
    socket_path = tmp_path / "detections.sock"
    endpoint = f"ipc://{socket_path}"
    context = zmq.Context()
    publisher = DetectionPublisher(endpoint, context=context)
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
        assert decode_frame(subscriber.recv()).shape == frame.shape
    finally:
        subscriber.close(linger=0)
        publisher.close()
        context.term()

    assert not socket_path.exists()


def test_viewer_uses_local_default_endpoint() -> None:
    assert (
        build_parser().parse_args([]).endpoint == "ipc:///tmp/reachy-attention-viewer"
    )


def test_viewer_script_declares_isolated_gui_opencv() -> None:
    source = (
        Path(__file__).parents[1] / "src/reachy_mini/tools/attention_viewer.py"
    ).read_text()

    assert source.startswith("# /// script\n")
    assert '"opencv-python>=4.10,<4.13"' in source
    assert "opencv-python-headless" not in source
