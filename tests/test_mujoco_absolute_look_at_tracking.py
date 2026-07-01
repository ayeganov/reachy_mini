# ruff: noqa: D100,D103

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("mujoco")

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from mujoco_red_target_tracking import (  # noqa: E402
    CONTROL_DT,
    SENSOR_TICKS,
    MujocoRedTargetHarness,
    run_orbit_sweep,
    run_scenario,
)


def _drive_until_centered(
    harness: MujocoRedTargetHarness,
    maximum_ticks: int = 250,
) -> tuple[float | None, float]:
    centered_at = None
    centered_ticks = 0
    for tick in range(maximum_ticks):
        if tick % SENSOR_TICKS == 0:
            detection = harness.observe(CONTROL_DT * SENSOR_TICKS)
            assert detection is not None
        else:
            detection = harness.last_detection
            assert detection is not None
        record = harness.step()
        assert record["target_type"] == "look_at"
        assert record["reason"] == "commanded"
        assert record["limit_hits"] == []
        assert not any(
            hit.get("source") == "profile_position"
            for hit in record["profile_limit_hits"]
        )
        if abs(detection.error_x) <= 0.03 and abs(detection.error_y) <= 0.03:
            centered_ticks += 1
            if centered_at is None:
                centered_at = tick * CONTROL_DT
        else:
            centered_ticks = 0
        if centered_ticks * CONTROL_DT >= 0.5:
            break
    return centered_at, centered_ticks * CONTROL_DT


def test_red_marker_is_detected_from_rendered_eye_camera_pixels() -> None:
    harness = MujocoRedTargetHarness()
    try:
        detection = harness.observe(CONTROL_DT * SENSOR_TICKS)
        assert detection is not None
        assert detection.pixel_count > 100
        assert abs(detection.error_x) <= 0.01
        assert abs(detection.error_y) <= 0.01
    finally:
        harness.close()


def test_harness_step_reads_only_latest_telemetry_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = MujocoRedTargetHarness()
    try:
        monkeypatch.setattr(
            harness.servo.telemetry,
            "query",
            lambda *args, **kwargs: pytest.fail("full telemetry query used per tick"),
        )
        harness.observe(CONTROL_DT * SENSOR_TICKS)
        record = harness.step()

        assert record["reason"] == "commanded"
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("name", "marker_azimuth", "marker_height"),
    [
        ("center", 0.0, 0.0),
        ("left", math.atan2(0.2, 0.5), 0.0),
        ("right", -math.atan2(0.2, 0.5), 0.0),
        ("top", 0.0, 0.2),
        ("bottom", 0.0, -0.2),
        ("top_left", math.atan2(0.1414, 0.5), 0.1414),
        ("top_right", -math.atan2(0.1414, 0.5), 0.1414),
        ("bottom_left", math.atan2(0.1414, 0.5), -0.1414),
        ("bottom_right", -math.atan2(0.1414, 0.5), -0.1414),
    ],
)
def test_rendered_marker_grid_centers_through_existing_look_at_path(
    name: str,
    marker_azimuth: float,
    marker_height: float,
) -> None:
    result = run_scenario(name, marker_azimuth, marker_height)

    assert result.centered_at_s is not None, result
    assert result.centered_at_s <= 2.5, result
    assert result.held_center_s >= 0.5, result
    assert result.reasons == ("commanded",), result
    assert result.ik_failures == 0, result
    assert result.guard_hits == 0, result
    assert result.profile_position_hits == 0, result
    assert result.maximum_direction_norm_error <= 1e-12, result
    assert result.maximum_abs_elevation <= math.atan2(0.2, 0.5) + 1e-12, result


def test_marker_loss_freezes_absolute_target_without_drift() -> None:
    harness = MujocoRedTargetHarness()
    harness.set_marker_orbit(-0.2, 0.0)
    try:
        for tick in range(30):
            if tick % SENSOR_TICKS == 0:
                assert harness.observe(CONTROL_DT * SENSOR_TICKS) is not None
            harness.step()
        before = harness.reference.target

        harness.marker_visible = False
        for tick in range(30):
            if tick % SENSOR_TICKS == 0:
                assert harness.observe(CONTROL_DT * SENSOR_TICKS) is None
            harness.step()

        assert harness.reference.target == before
        assert harness.last_reference_update.reason == "no_observation"
        assert harness.guard_hits == 0
        assert harness.profile_position_hits == 0
    finally:
        harness.close()


def test_abrupt_horizontal_reversal_centers_without_reset_or_stuck_state() -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.set_marker_orbit(math.atan2(0.2, 0.5), 0.0)
        first_centered, first_hold = _drive_until_centered(harness)
        first_target = harness.reference.target

        harness.set_marker_orbit(-math.atan2(0.2, 0.5), 0.0)
        second_centered, second_hold = _drive_until_centered(harness)

        assert first_centered is not None and first_centered <= 2.5
        assert first_hold >= 0.5
        assert second_centered is not None and second_centered <= 3.5
        assert second_hold >= 0.5
        assert harness.reference.target.y < first_target.y
        assert harness.guard_hits == 0
        assert harness.profile_position_hits == 0
        assert harness.maximum_direction_norm_error <= 1e-12
    finally:
        harness.close()


def test_marker_can_be_placed_at_every_azimuth_without_projection_error() -> None:
    harness = MujocoRedTargetHarness()
    try:
        for azimuth in (
            -2.0 * math.pi,
            -math.pi,
            -math.pi / 2.0,
            0.0,
            math.pi / 2.0,
            math.pi,
            2.0 * math.pi,
        ):
            harness.set_marker_orbit(azimuth, 0.1)
            offset = harness.marker_position - harness.marker_orbit_center
            assert math.hypot(float(offset[0]), float(offset[1])) == pytest.approx(
                harness.marker_orbit_radius
            )
            assert float(offset[2]) == pytest.approx(0.1)
            assert float(offset[0]) == pytest.approx(
                harness.marker_orbit_radius * math.cos(azimuth)
            )
            assert float(offset[1]) == pytest.approx(
                harness.marker_orbit_radius * math.sin(azimuth)
            )
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("u", "v", "expected_azimuth"),
    [
        (640.0, 320.0, 0.0),
        (320.0, 0.0, math.pi / 2.0),
        (0.0, 320.0, math.pi),
        (320.0, 640.0, -math.pi / 2.0),
    ],
)
def test_orbit_pad_maps_directly_to_robot_frame_azimuth(
    u: float,
    v: float,
    expected_azimuth: float,
) -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.move_marker_from_orbit_pad(u, v, 640)

        assert harness.marker_azimuth == pytest.approx(expected_azimuth)
    finally:
        harness.close()


@pytest.mark.parametrize("end_degrees", [200.0, -200.0])
def test_slow_visual_orbit_engages_body_yaw_without_losing_marker(
    end_degrees: float,
) -> None:
    result = run_orbit_sweep(end_degrees)

    assert result.reached_degrees == end_degrees, result
    assert all(step.marker_visible for step in result.steps), result
    assert all(step.centered_at_s is not None for step in result.steps), result
    assert all(step.held_center_s >= 0.5 for step in result.steps), result
    assert abs(result.steps[-1].body_yaw_degrees) > 125.0, result
    assert result.ik_failures == 0, result
    assert result.guard_hits == 0, result
    assert result.profile_position_hits == 0, result


def test_oracle_and_vision_modes_both_submit_look_at_targets() -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.observe(CONTROL_DT * SENSOR_TICKS)
        vision_record = harness.step()
        harness.mode = "oracle"
        oracle_record = harness.step()

        assert vision_record["target_type"] == "look_at"
        assert oracle_record["target_type"] == "look_at"
        assert vision_record["input_target"]["kind"] == "look_at"
        assert oracle_record["input_target"]["kind"] == "look_at"
    finally:
        harness.close()


def test_oracle_compensates_camera_to_head_vertical_parallax() -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.mode = "oracle"
        oracle = harness._metric_target()

        assert harness.marker_position[2] - oracle[2] == pytest.approx(0.0525)
        assert oracle[2] == pytest.approx(
            harness.look_at_sphere.origin_z,
            abs=1e-6,
        )
    finally:
        harness.close()
