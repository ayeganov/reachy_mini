# ruff: noqa: D100,D103

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from yellow_box_follow import (  # noqa: E402
    APPROVED_TRACKING_CONFIG,
    FixedRateLookAtSender,
    YellowBoxDetection,
    YellowBoxDetectorConfig,
    YellowBoxFollower,
    _target_payload,
    detect_yellow_box,
)

from reachy_mini.daemon.tracking.look_at_reference import (  # noqa: E402
    ImageErrorReferenceConfig,
    LookAtReference,
    LookAtSphere,
    SphericalLookAtReferenceController,
)


def _yellow_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


def test_detector_selects_largest_passing_yellow_component() -> None:
    frame = _yellow_frame()
    cv2.rectangle(frame, (40, 80), (120, 120), (0, 255, 255), -1)
    cv2.rectangle(frame, (320, 140), (600, 240), (0, 255, 255), -1)

    detection, mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is not None
    assert detection.u == pytest.approx(460.0, abs=1.0)
    assert detection.v == pytest.approx(190.0, abs=1.0)
    assert detection.bounding_box == (320, 140, 281, 101)
    assert mask[210, 460] == 255


def test_detector_accepts_non_rectangular_yellow_component() -> None:
    frame = _yellow_frame()
    cv2.circle(frame, (320, 180), 45, (0, 255, 255), -1)

    detection, _mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is not None
    assert detection.u == pytest.approx(320.0, abs=1.0)
    assert detection.v == pytest.approx(180.0, abs=1.0)


def test_detector_rejects_small_and_low_saturation_regions() -> None:
    frame = _yellow_frame()
    cv2.rectangle(frame, (10, 10), (20, 20), (0, 255, 255), -1)
    cv2.rectangle(frame, (200, 100), (400, 300), (160, 210, 210), -1)

    detection, _mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is None


def test_detection_normalized_error_uses_expected_image_signs() -> None:
    detection = YellowBoxDetection(
        u=480.0,
        v=90.0,
        bounding_box=(0, 0, 1, 1),
        area_ratio=0.1,
    )

    assert detection.normalized_error(640, 360) == pytest.approx((0.5, -0.5))


def test_follower_updates_immediately_then_freezes_on_loss() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
        ImageErrorReferenceConfig(max_angular_speed=2.0),
    )
    follower = YellowBoxFollower(controller)
    detection = YellowBoxDetection(
        u=480.0,
        v=180.0,
        bounding_box=(400, 100, 160, 160),
        area_ratio=0.1,
    )

    observed = follower.observe(detection, width=640, height=360, dt=0.02)
    target = observed.update.target
    lost = follower.observe(None, width=640, height=360, dt=0.02)

    assert observed.update.reason == "tracking"
    assert target.y < 0.0
    assert lost.update.reason == "target_lost"
    assert lost.update.target == target


def test_follower_accepts_a_far_centroid_without_identity_gating() -> None:
    controller = SphericalLookAtReferenceController(LookAtSphere())
    follower = YellowBoxFollower(controller)
    box = YellowBoxDetection(
        u=480.0,
        v=180.0,
        bounding_box=(400, 140, 160, 60),
        area_ratio=0.1,
    )
    other_yellow_object = YellowBoxDetection(
        u=80.0,
        v=180.0,
        bounding_box=(20, 140, 120, 60),
        area_ratio=0.1,
    )
    first = follower.observe(box, width=640, height=360, dt=0.02)

    accepted = follower.observe(
        other_yellow_object,
        width=640,
        height=360,
        dt=0.02,
    )

    assert accepted.update.reason == "tracking"
    assert accepted.update.target.y > first.update.target.y


def test_tracking_config_tolerates_measured_transport_stall() -> None:
    assert APPROVED_TRACKING_CONFIG["max_detection_age"] == 2.0


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


def test_sender_refreshes_latest_target_without_blocking_caller(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []

    def fake_request(
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        calls.append(
            {
                "method": method,
                "base_url": base_url,
                "path": path,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return {}

    monkeypatch.setattr("yellow_box_follow._request_json", fake_request)
    sender = FixedRateLookAtSender(
        base_url="http://robot/api",
        initial_target=LookAtReference(x=0.5, y=0.0, z=0.0),
        request_timeout=0.1,
        period=0.01,
    )

    sender.start()
    sender.set_target(LookAtReference(x=0.4, y=0.2, z=0.1))
    time.sleep(0.035)
    sender.stop()

    assert sender.sent_count >= 2
    assert sender.error_count == 0
    assert calls[-1]["payload"] == {
        "x": 0.4,
        "y": 0.2,
        "z": 0.1,
        "confidence": 1.0,
        "frame_id": sender.sent_count - 1,
    }


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
