# ruff: noqa: D100,D103

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from yellow_box_follow import (  # noqa: E402
    APPROVED_TRACKING_CONFIG,
    YellowBoxDetection,
    YellowBoxDetectorConfig,
    _detection_payload,
    detect_yellow_box,
)


def _yellow_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


def _detection(u: float, v: float) -> YellowBoxDetection:
    return YellowBoxDetection(
        u=u,
        v=v,
        bounding_box=(0, 0, 1, 1),
        area_ratio=0.1,
    )


def test_detector_selects_largest_passing_yellow_component() -> None:
    frame = _yellow_frame()
    cv2.rectangle(frame, (40, 80), (120, 120), (0, 255, 255), -1)
    cv2.rectangle(frame, (320, 140), (600, 240), (0, 255, 255), -1)

    detection, mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is not None
    assert detection.u == pytest.approx(460.0, abs=1.0)
    assert detection.v == pytest.approx(190.0, abs=1.0)
    assert mask[210, 460] == 255


def test_detector_accepts_non_rectangular_yellow_component() -> None:
    frame = _yellow_frame()
    cv2.circle(frame, (320, 180), 45, (0, 255, 255), -1)

    detection, _mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is not None
    assert (detection.u, detection.v) == pytest.approx((320.0, 180.0), abs=1.0)


def test_detector_rejects_small_and_low_saturation_regions() -> None:
    frame = _yellow_frame()
    cv2.rectangle(frame, (10, 10), (20, 20), (0, 255, 255), -1)
    cv2.rectangle(frame, (200, 100), (400, 300), (160, 210, 210), -1)

    detection, _mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is None


def test_detection_payload_contains_only_current_centroid() -> None:
    payload = _detection_payload(
        _detection(100.0, 80.0),
        width=640,
        height=360,
        frame_id=7,
    )

    assert payload == {
        "u": 100.0,
        "v": 80.0,
        "width": 640,
        "height": 360,
        "confidence": 1.0,
        "frame_id": 7,
    }
    assert "timestamp" not in payload


def test_tracking_config_drops_stale_centroids_quickly() -> None:
    assert APPROVED_TRACKING_CONFIG["max_detection_age"] == 0.35


@pytest.mark.parametrize(
    "config",
    [
        lambda: YellowBoxDetectorConfig(hue_low=50, hue_high=40),
        lambda: YellowBoxDetectorConfig(saturation_low=256),
        lambda: YellowBoxDetectorConfig(min_area_ratio=0.5, max_area_ratio=0.4),
        lambda: YellowBoxDetectorConfig(morphology_size=4),
    ],
)
def test_invalid_detector_configuration_is_rejected(config) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        config()
