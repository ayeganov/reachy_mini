# ruff: noqa: D100,D103

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from yellow_box_follow import (  # noqa: E402
    YellowBoxDetection,
    YellowBoxDetectorConfig,
    YellowBoxFollower,
    detect_yellow_box,
)

from reachy_mini.daemon.tracking.look_at_reference import (  # noqa: E402
    ImageErrorReferenceConfig,
    LookAtSphere,
    SphericalLookAtReferenceController,
)


def _yellow_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


def test_detector_selects_largest_passing_yellow_rectangle() -> None:
    frame = _yellow_frame()
    cv2.rectangle(frame, (40, 80), (120, 120), (0, 255, 255), -1)
    cv2.rectangle(frame, (320, 140), (600, 240), (0, 255, 255), -1)

    detection, mask = detect_yellow_box(frame, YellowBoxDetectorConfig())

    assert detection is not None
    assert detection.u == pytest.approx(460.0, abs=1.0)
    assert detection.v == pytest.approx(190.0, abs=1.0)
    assert detection.bounding_box == (320, 140, 281, 101)
    assert detection.rectangularity > 0.95
    assert mask[210, 460] == 255


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
        rectangularity=1.0,
    )

    assert detection.normalized_error(640, 360) == pytest.approx((0.5, -0.5))


def test_follower_requires_acquisition_then_freezes_on_loss() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
        ImageErrorReferenceConfig(max_angular_speed=2.0),
    )
    follower = YellowBoxFollower(controller, acquisition_frames=3)
    detection = YellowBoxDetection(
        u=480.0,
        v=180.0,
        bounding_box=(400, 100, 160, 160),
        area_ratio=0.1,
        rectangularity=1.0,
    )

    first = follower.observe(detection, width=640, height=360, dt=0.02)
    second = follower.observe(detection, width=640, height=360, dt=0.02)
    acquired = follower.observe(detection, width=640, height=360, dt=0.02)
    target = acquired.update.target
    lost = follower.observe(None, width=640, height=360, dt=0.02)

    assert first.update.reason == "acquiring" and not first.submit_target
    assert second.update.reason == "acquiring" and not second.submit_target
    assert acquired.update.reason == "tracking" and acquired.submit_target
    assert target.y < 0.0
    assert lost.update.reason == "target_lost" and lost.submit_target
    assert lost.update.target == target


def test_follower_rejects_a_far_candidate_after_acquisition() -> None:
    controller = SphericalLookAtReferenceController(LookAtSphere())
    follower = YellowBoxFollower(controller, acquisition_frames=1)
    box = YellowBoxDetection(
        u=480.0,
        v=180.0,
        bounding_box=(400, 140, 160, 60),
        area_ratio=0.1,
        rectangularity=1.0,
    )
    other_yellow_object = YellowBoxDetection(
        u=80.0,
        v=180.0,
        bounding_box=(20, 140, 120, 60),
        area_ratio=0.1,
        rectangularity=1.0,
    )
    acquired = follower.observe(box, width=640, height=360, dt=0.02)

    rejected = follower.observe(
        other_yellow_object,
        width=640,
        height=360,
        dt=0.02,
    )

    assert rejected.update.reason == "candidate_jump"
    assert rejected.submit_target
    assert rejected.update.target == acquired.update.target


@pytest.mark.parametrize(
    "config",
    [
        lambda: YellowBoxDetectorConfig(hue_low=50, hue_high=40),
        lambda: YellowBoxDetectorConfig(saturation_low=256),
        lambda: YellowBoxDetectorConfig(min_area_ratio=0.5, max_area_ratio=0.4),
        lambda: YellowBoxDetectorConfig(min_rectangularity=1.1),
        lambda: YellowBoxDetectorConfig(min_aspect_ratio=3.0, max_aspect_ratio=2.0),
        lambda: YellowBoxDetectorConfig(morphology_size=4),
    ],
)
def test_invalid_detector_configuration_is_rejected(config) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        config()
