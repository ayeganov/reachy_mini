# ruff: noqa: D100,D103

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from yellow_box_follow import (  # noqa: E402
    APPROVED_TRACKING_CONFIG,
    DEFAULT_CORRECTION_DEGREES,
    YellowBoxDetection,
    YellowBoxDetectorConfig,
    _target_payload,
    detect_yellow_box,
    target_from_current_gaze,
)

from reachy_mini.daemon.tracking.look_at_reference import (  # noqa: E402
    LookAtReference,
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


def _state(*, pitch: float = 0.0, yaw: float = 0.0) -> dict[str, object]:
    return {
        "head_pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": pitch,
            "yaw": yaw,
        }
    }


def _direction(target: LookAtReference) -> np.ndarray:
    vector = np.array([target.x, target.y, target.z])
    return vector / np.linalg.norm(vector)


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


def test_target_is_one_correction_from_current_gaze() -> None:
    correction = math.radians(DEFAULT_CORRECTION_DEGREES)
    target = target_from_current_gaze(
        _detection(0.0, 180.0),
        width=640,
        height=360,
        state=_state(yaw=0.4),
        origin=(0.0, 0.0, 0.0),
        correction_radians=correction,
    )

    assert math.atan2(target.y, target.x) == pytest.approx(0.4 + correction)


def test_centered_target_tracks_current_gaze_instead_of_returning_neutral() -> None:
    target = target_from_current_gaze(
        _detection(320.0, 180.0),
        width=640,
        height=360,
        state=_state(pitch=-0.2, yaw=-0.5),
        origin=(0.0, 0.0, 0.0),
        correction_radians=math.radians(DEFAULT_CORRECTION_DEGREES),
    )

    direction = _direction(target)
    assert math.atan2(direction[1], direction[0]) == pytest.approx(-0.5)
    assert math.asin(direction[2]) == pytest.approx(0.2)


def test_corner_correction_is_bounded() -> None:
    correction = math.radians(DEFAULT_CORRECTION_DEGREES)
    target = target_from_current_gaze(
        _detection(640.0, 360.0),
        width=640,
        height=360,
        state=_state(),
        origin=(0.0, 0.0, 0.0),
        correction_radians=correction,
    )

    angular_distance = math.acos(float(np.clip(_direction(target)[0], -1.0, 1.0)))
    assert angular_distance <= correction + 1e-12
    assert target.y < 0.0
    assert target.z < 0.0


def test_target_payload_uses_daemon_receipt_time() -> None:
    payload = _target_payload(LookAtReference(x=0.5, y=-0.1, z=0.2), frame_id=7)

    assert payload == {
        "x": 0.5,
        "y": -0.1,
        "z": 0.2,
        "confidence": 1.0,
        "frame_id": 7,
    }
    assert "timestamp" not in payload


def test_tracking_config_tolerates_measured_transport_stall() -> None:
    assert APPROVED_TRACKING_CONFIG["max_detection_age"] == 2.0


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
