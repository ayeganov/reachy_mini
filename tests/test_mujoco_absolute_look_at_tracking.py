# ruff: noqa: D100,D103

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
mujoco = pytest.importorskip("mujoco")

from examples.mujoco_red_target_tracking import (  # noqa: E402
    CONTROL_DT,
    SCENARIOS,
    SENSOR_TICKS,
    MujocoRedTargetHarness,
    run_orbit_sweep,
    run_scenario,
)
from reachy_mini.daemon.tracking import (  # noqa: E402
    visual_servo as visual_servo_module,
)
from reachy_mini.daemon.tracking.visual_servo import VisualServoConfig  # noqa: E402


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
        assert record["target_type"] == "detection"
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


def test_mujoco_harness_uses_centralized_motion_defaults() -> None:
    config = VisualServoConfig()

    assert (
        config.joint_safety_margin,
        config.max_joint_velocity,
        config.max_joint_acceleration,
        config.max_joint_jerk,
        config.look_at_profile_response_hz,
    ) == (0.1745329252, 0.6, 2.4, 16.0, 2.0)


@pytest.mark.parametrize(
    ("name", "marker_azimuth", "marker_height"),
    [(name, *coordinates) for name, coordinates in SCENARIOS.items()],
)
def test_rendered_marker_grid_centers_through_production_detection_path(
    name: str,
    marker_azimuth: float,
    marker_height: float,
) -> None:
    result = run_scenario(name, marker_azimuth, marker_height)

    assert result.centered_at_s is not None, result
    assert result.centered_at_s <= 2.5, result
    assert result.held_center_s >= 0.5, result
    assert result.target_types == ("detection",), result
    assert result.reasons == ("commanded",), result
    assert result.ik_failures == 0, result
    assert result.guard_hits == 0, result
    assert result.profile_position_hits == 0, result
    assert result.maximum_direction_norm_error <= 1e-12, result
    assert result.maximum_abs_elevation <= math.atan2(0.2, 0.5) + 1e-12, result


def test_marker_loss_expires_detection_and_stops_without_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    harness = MujocoRedTargetHarness()
    harness.set_marker_orbit(-0.2, 0.0)
    try:
        for tick in range(30):
            if tick % SENSOR_TICKS == 0:
                assert harness.observe(CONTROL_DT * SENSOR_TICKS) is not None
            harness.step()
            now += CONTROL_DT
        before = harness.last_target

        harness.marker_visible = False
        now += harness.servo.config.max_detection_age
        reasons = set()
        for tick in range(30):
            if tick % SENSOR_TICKS == 0:
                assert harness.observe(CONTROL_DT * SENSOR_TICKS) is None
            reasons.add(str(harness.step()["reason"]))
            now += CONTROL_DT

        assert harness.last_target == before
        assert reasons <= {"stopping_no_target", "holding_no_target"}
        assert harness.guard_hits == 0
        assert harness.profile_position_hits == 0
    finally:
        harness.close()


def test_mujoco_target_gap_preserves_command_motion_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    harness = MujocoRedTargetHarness()
    harness.set_marker_orbit(math.atan2(0.2, 0.5), 0.0)
    try:
        for tick in range(12):
            if tick % SENSOR_TICKS == 0:
                assert harness.observe(CONTROL_DT * SENSOR_TICKS) is not None
            harness.step()
            now += CONTROL_DT

        now += harness.servo.config.max_detection_age
        gap_reasons = []
        for _ in range(6):
            assert harness.servo.step(dt=CONTROL_DT)
            record = harness.servo.telemetry.latest()
            assert record is not None
            gap_reasons.append(record["reason"])
            assert harness.backend.target_head_joint_positions is not None
            harness.backend.data.ctrl[:7] = harness.backend.target_head_joint_positions
            for _ in range(round(CONTROL_DT / harness.backend.model.opt.timestep)):
                mujoco.mj_step(harness.backend.model, harness.backend.data)
            harness._refresh_backend_state()
            now += CONTROL_DT

        assert harness.observe(CONTROL_DT * SENSOR_TICKS) is not None
        resumed = harness.step()

        assert set(gap_reasons) <= {"stopping_no_target", "holding_no_target"}
        assert resumed["reason"] == "commanded"
        assert resumed["limit_hits"] == []
    finally:
        harness.close()


def test_abrupt_horizontal_reversal_centers_without_reset_or_stuck_state() -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.set_marker_orbit(math.atan2(0.2, 0.5), 0.0)
        first_centered, first_hold = _drive_until_centered(harness)
        first_target = harness.last_target

        harness.set_marker_orbit(-math.atan2(0.2, 0.5), 0.0)
        second_centered, second_hold = _drive_until_centered(harness)

        assert first_centered is not None and first_centered <= 2.5
        assert first_hold >= 0.5
        assert second_centered is not None and second_centered <= 3.5
        assert second_hold >= 0.5
        assert harness.last_target[1] < first_target[1]
        assert harness.guard_hits == 0
        assert harness.profile_position_hits == 0
        assert harness.maximum_direction_norm_error <= 1e-12
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("first_height", "second_height"),
    [(0.2, -0.2), (-0.2, 0.2)],
)
def test_abrupt_vertical_reversal_centers_on_same_controller(
    first_height: float,
    second_height: float,
) -> None:
    harness = MujocoRedTargetHarness()
    try:
        harness.set_marker_orbit(0.0, first_height)
        first_centered, first_hold = _drive_until_centered(harness)
        first_target = harness.last_target

        # A direct +0.2 m to -0.2 m jump leaves the fixed target outside the
        # rendered eye frame. Reverse as far as the camera can still observe,
        # then continue to the requested extreme on the same controller.
        harness.set_marker_orbit(0.0, second_height * 0.75)
        reversal_centered, reversal_hold = _drive_until_centered(harness)
        harness.set_marker_orbit(0.0, second_height)
        extreme_centered, extreme_hold = _drive_until_centered(harness)

        assert first_centered is not None and first_centered <= 2.5
        assert first_hold >= 0.5
        assert reversal_centered is not None and reversal_centered <= 3.5
        assert reversal_hold >= 0.5
        assert extreme_centered is not None and extreme_centered <= 2.5
        assert extreme_hold >= 0.5
        assert (harness.last_target[2] - first_target[2]) * (
            second_height - first_height
        ) > 0.0
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


def test_harness_submits_rendered_centroid_as_detection() -> None:
    harness = MujocoRedTargetHarness()
    try:
        detection = harness.observe(CONTROL_DT * SENSOR_TICKS)
        assert detection is not None

        record = harness.step()

        assert record["target_type"] == "detection"
        assert record["input_target"] == {
            "kind": "detection",
            "u": detection.u,
            "v": detection.v,
            "timestamp": record["input_target"]["timestamp"],
            "confidence": 1.0,
            "frame_id": 0,
            "width": harness.width,
            "height": harness.height,
        }
    finally:
        harness.close()
