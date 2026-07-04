# ruff: noqa: D100,D103
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.tracking import visual_servo as visual_servo_module
from reachy_mini.daemon.tracking.telemetry import (
    VisualServoTelemetryBuffer,
    dump_jsonl,
    finite_json_value,
    load_jsonl,
    summarize_records,
)
from reachy_mini.daemon.tracking.visual_servo import (
    DetectionBuffer,
    JointCommandLimiter,
    JointCommandSafetyGuard,
    LookAtJointCommandProfile,
    LookAtTargetFilter,
    PixelTargetFilter,
    TrackingDetection,
    TrackingLookAtTarget,
    VisualServoConfig,
    VisualServoController,
)


def _acceptance_profile_config() -> VisualServoConfig:
    return VisualServoConfig(
        joint_safety_margin=0.1745329252,
        max_joint_velocity=0.60,
        max_joint_acceleration=1.60,
        max_joint_jerk=8.0,
        look_at_profile_response_hz=1.0,
    )


def test_look_at_joint_profile_initializes_from_current_and_profiles_body_yaw() -> None:
    profile = LookAtJointCommandProfile(config=_acceptance_profile_config())
    current = np.array([0.1, *([0.2] * 6)])
    desired = np.array([0.7, *([0.3] * 6)])

    command, _hits = profile.update_with_telemetry(desired, current, dt=0.02)

    assert current[0] < command[0] < desired[0]
    assert np.all(command[1:] >= current[1:])
    assert np.all(command[1:] < desired[1:])
    assert np.max(command[1:] - current[1:]) == pytest.approx(8.0 * 0.02**3)


def test_look_at_joint_profile_converges_to_fixed_target_within_two_seconds() -> None:
    profile = LookAtJointCommandProfile(config=_acceptance_profile_config())
    current = np.zeros(7)
    desired = np.array([0.0, *([0.3] * 6)])
    positions = []

    for _ in range(100):
        command, _hits = profile.update_with_telemetry(desired, current, dt=0.02)
        positions.append(float(command[1]))

    assert abs(positions[-1] - 0.3) <= 0.001
    assert all(
        after >= before - 1e-12 for before, after in zip(positions, positions[1:])
    )
    assert max(positions) <= 0.3 + 1e-12
    np.testing.assert_allclose(profile._velocity, 0.0)
    np.testing.assert_allclose(profile._acceleration, 0.0)


def test_look_at_joint_profile_jerk_limits_normal_acceleration_changes() -> None:
    config = VisualServoConfig(
        joint_safety_margin=0.0,
        max_joint_velocity=100.0,
        max_joint_acceleration=100.0,
        max_joint_jerk=2.0,
    )
    profile = LookAtJointCommandProfile(config=config)
    desired = np.array([0.0, *([0.3] * 6)])

    profile.update_with_telemetry(desired, np.zeros(7), dt=0.02)
    first = profile._acceleration.copy()
    profile.update_with_telemetry(desired, np.zeros(7), dt=0.02)

    assert np.max(np.abs(profile._acceleration - first)) <= 2.0 * 0.02


def test_look_at_joint_profile_has_no_unbounded_target_crossing_reset() -> None:
    config = _acceptance_profile_config()
    profile = LookAtJointCommandProfile(config=config)
    desired = np.array([0.0, *([0.3] * 6)])
    previous_position = np.zeros(6)
    previous_velocity = np.zeros(6)
    previous_acceleration = np.zeros(6)

    for dt in [0.0114, 0.0286, 0.017, 0.023, 0.02] * 25:
        command, hits = profile.update_with_telemetry(desired, np.zeros(7), dt=dt)
        velocity = (command[1:] - previous_position) / dt
        acceleration = (velocity - previous_velocity) / dt
        jerk = (acceleration - previous_acceleration) / dt

        assert np.max(np.abs(velocity)) <= config.max_joint_velocity + 1e-9
        assert np.max(np.abs(acceleration)) <= config.max_joint_acceleration + 1e-9
        assert np.max(np.abs(jerk)) <= config.max_joint_jerk + 1e-7
        assert not any(hit.get("reason") == "target_crossing_reset" for hit in hits)
        assert np.max(command[1:]) <= 0.3 + 1e-9

        previous_position = command[1:].copy()
        previous_velocity = velocity
        previous_acceleration = acceleration

    np.testing.assert_allclose(command[1:], 0.3, atol=1e-9)
    np.testing.assert_allclose(profile._velocity, 0.0, atol=1e-9)
    np.testing.assert_allclose(profile._acceleration, 0.0, atol=1e-9)


@pytest.mark.parametrize("target", [-0.3, 0.3])
def test_look_at_joint_profile_stops_bounded_motion_in_both_directions(
    target: float,
) -> None:
    config = _acceptance_profile_config()
    profile = LookAtJointCommandProfile(config=config)
    guard = JointCommandSafetyGuard(config=config)
    current = np.zeros(7)
    desired = np.full(7, target)

    for _ in range(15):
        command, _hits = profile.update_with_telemetry(desired, current, 0.02)
        guard_hits, velocity, acceleration = guard.check(command, current, 0.02)
        assert guard_hits == []
        guard.commit(command, velocity, acceleration)
        current = command

    for _ in range(100):
        command, _hits = profile.stop_with_telemetry(current, 0.02)
        guard_hits, velocity, acceleration = guard.check(command, current, 0.02)
        assert guard_hits == []
        guard.commit(command, velocity, acceleration)
        current = command

    assert profile.stationary
    held, _hits = profile.stop_with_telemetry(current, 0.02)
    np.testing.assert_allclose(held, current, atol=1e-12)


def test_look_at_joint_profile_preserves_boundary_on_abrupt_reversals() -> None:
    config = _acceptance_profile_config()
    profile = LookAtJointCommandProfile(config=config)
    guard = JointCommandSafetyGuard(config=config)
    lower = profile.limits[3, 0] + config.joint_safety_margin
    upper = profile.limits[3, 1] - config.joint_safety_margin
    random = np.random.default_rng(19)
    current = np.zeros(7)
    desired = np.zeros(7)
    hold_ticks = 0

    for _ in range(1_100):
        if hold_ticks == 0:
            desired[3] = (lower - 1.0, upper + 1.0)[int(random.integers(2))]
            hold_ticks = int(random.integers(1, 31))
        hold_ticks -= 1
        dt = float(random.uniform(0.017, 0.026))
        command, profile_hits = profile.update_with_telemetry(desired, current, dt)
        guard_hits, next_velocity, next_acceleration = guard.check(command, current, dt)

        assert lower <= command[3] <= upper
        assert not any(hit.get("source") == "profile_position" for hit in profile_hits)
        assert guard_hits == []
        guard.commit(command, next_velocity, next_acceleration)
        current = command


def test_look_at_joint_profile_reports_velocity_acceleration_and_jerk_clamps() -> None:
    config = VisualServoConfig(
        joint_safety_margin=0.0,
        max_joint_velocity=0.0001,
        max_joint_acceleration=0.01,
        max_joint_jerk=1.0,
    )
    profile = LookAtJointCommandProfile(config=config)
    desired = np.zeros(7)
    desired[1] = 0.5

    _command, hits = profile.update_with_telemetry(desired, np.zeros(7), dt=0.02)

    by_kind = {hit["kind"]: hit for hit in hits if hit["joint_index"] == 1}
    assert {"velocity", "acceleration", "jerk"} <= set(by_kind)
    assert by_kind["velocity"]["source"] == "profile_velocity"
    assert by_kind["acceleration"]["source"] == "profile_acceleration"
    assert by_kind["jerk"]["source"] == "profile_jerk"
    assert all(
        abs(float(hit["value"])) > float(hit["limit"]) for hit in by_kind.values()
    )


def test_joint_command_safety_guard_ignores_only_numerical_limit_noise() -> None:
    config = _acceptance_profile_config()
    guard = JointCommandSafetyGuard(config=config)
    dt = 0.02

    def hits_for_jerk(jerk: float) -> list[dict[str, float | int | str]]:
        command = np.full(7, jerk * dt**3)
        hits, _velocity, _acceleration = guard.check(command, np.zeros(7), dt)
        return hits

    assert not any(hit["kind"] == "jerk" for hit in hits_for_jerk(8.00000245))
    assert any(hit["kind"] == "jerk" for hit in hits_for_jerk(8.00002))


def test_joint_command_safety_guard_accepts_inward_recovery_band_progress() -> None:
    config = _acceptance_profile_config()
    guard = JointCommandSafetyGuard(config=config)
    current = np.zeros(7)
    current[6] = -1.227184630308513
    command = current.copy()
    command[6] += config.max_joint_jerk * 0.02**3

    hits, _velocity, _acceleration, recovery = guard.check_with_telemetry(
        command, current, 0.02
    )

    assert hits == []
    assert recovery == [
        {
            "joint_index": 6,
            "kind": "lower",
            "hard_limit": pytest.approx(guard.limits[6, 0]),
            "soft_limit": pytest.approx(
                guard.limits[6, 0] + config.joint_safety_margin
            ),
            "reference": pytest.approx(current[6]),
            "command": pytest.approx(command[6]),
            "violation_before": pytest.approx(
                guard.limits[6, 0] + config.joint_safety_margin - current[6]
            ),
            "violation_after": pytest.approx(
                guard.limits[6, 0] + config.joint_safety_margin - command[6]
            ),
        }
    ]


def test_joint_command_safety_guard_rejects_invalid_recovery_band_motion() -> None:
    config = VisualServoConfig(
        joint_safety_margin=0.1745329252,
        max_joint_velocity=100.0,
        max_joint_acceleration=100.0,
        max_joint_jerk=100.0,
    )
    current = np.zeros(7)
    current[6] = -1.227184630308513
    lower_hard = JointCommandSafetyGuard(config=config).limits[6, 0]
    lower_soft = lower_hard + config.joint_safety_margin

    cases = (
        (current, np.array([*current[:6], current[6] - 0.0001]), "recovery_outward"),
        (
            np.zeros(7),
            np.array([*([0.0] * 6), lower_soft - 0.0001]),
            "lower_position",
        ),
        (
            current,
            np.array([*current[:6], lower_hard - 0.0001]),
            "lower_hard_position",
        ),
    )

    for reference, command, expected_kind in cases:
        guard = JointCommandSafetyGuard(config=config)
        hits, _velocity, _acceleration, _recovery = guard.check_with_telemetry(
            command, reference, 0.02
        )
        assert expected_kind in {hit["kind"] for hit in hits}


def test_look_at_joint_profile_reset_clears_motion_state() -> None:
    profile = LookAtJointCommandProfile(config=_acceptance_profile_config())
    profile.update_with_telemetry(np.array([0.0, *([0.3] * 6)]), np.zeros(7), dt=0.02)

    profile.reset()

    assert profile._position is None
    assert profile._body_yaw_position is None
    assert profile._body_yaw_velocity == 0.0
    assert profile._body_yaw_acceleration == 0.0
    np.testing.assert_allclose(profile._velocity, 0.0)
    np.testing.assert_allclose(profile._acceleration, 0.0)


def test_look_at_joint_profile_rejects_invalid_inputs_without_state_mutation() -> None:
    profile = LookAtJointCommandProfile(config=_acceptance_profile_config())
    desired = np.array([0.0, *([0.3] * 6)])
    profile.update_with_telemetry(desired, np.zeros(7), dt=0.02)
    state = (
        profile._position.copy(),  # type: ignore[union-attr]
        profile._velocity.copy(),
        profile._acceleration.copy(),
        profile._body_yaw_position,
        profile._body_yaw_velocity,
        profile._body_yaw_acceleration,
    )
    invalid_calls = [
        (desired, np.zeros(7), 0.0),
        (desired, np.zeros(7), -0.02),
        (desired, np.zeros(7), float("nan")),
        (desired, np.zeros(7), float("inf")),
        (np.zeros(6), np.zeros(7), 0.02),
        (desired, np.zeros(6), 0.02),
        (np.array([0.0, np.nan, *([0.0] * 5)]), np.zeros(7), 0.02),
        (desired, np.array([0.0, np.inf, *([0.0] * 5)]), 0.02),
    ]

    for invalid_desired, invalid_current, invalid_dt in invalid_calls:
        with pytest.raises(ValueError):
            profile.update_with_telemetry(invalid_desired, invalid_current, invalid_dt)
        np.testing.assert_array_equal(profile._position, state[0])
        np.testing.assert_array_equal(profile._velocity, state[1])
        np.testing.assert_array_equal(profile._acceleration, state[2])
        assert profile._body_yaw_position == state[3]
        assert profile._body_yaw_velocity == state[4]
        assert profile._body_yaw_acceleration == state[5]

    for response_hz in (0.0, -1.0, 5.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="look_at_profile_response_hz"):
            LookAtJointCommandProfile(
                config=VisualServoConfig(look_at_profile_response_hz=response_hz)
            )


class _MotionTestKinematics:
    automatic_body_yaw = False

    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.joints = np.full(7, 0.2) if joints is None else joints

    def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
        self.automatic_body_yaw = automatic_body_yaw

    def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
        joints = self.joints.copy()
        joints[0] = body_yaw
        return joints


class _MotionTestBackend:
    is_move_running = False

    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.head_kinematics = _MotionTestKinematics(joints)
        self.current = np.zeros(7)
        self.pose = np.eye(4)
        self.commands: list[np.ndarray] = []
        self.joint_reads = 0
        self.pose_reads = 0
        self.fail_writes = 0

    def get_present_head_joint_positions(self) -> np.ndarray:
        self.joint_reads += 1
        return self.current.copy()

    def get_present_head_pose(self) -> np.ndarray:
        self.pose_reads += 1
        return self.pose.copy()

    def set_target_head_joint_positions(self, command: np.ndarray) -> None:
        if self.fail_writes:
            self.fail_writes -= 1
            raise RuntimeError("write failed")
        self.commands.append(command.copy())


def test_detection_target_is_stateless_and_clamps_absolute_elevation() -> None:
    backend = _MotionTestBackend()
    vertical_fov = 0.2
    elevation_limit = 0.4
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(
            smoothing_alpha=1.0,
            image_horizontal_fov=0.4,
            image_vertical_fov=vertical_fov,
            image_error_elevation_limit=elevation_limit,
        ),
    )
    started = time.time()
    elevations = []
    for frame_id in range(20):
        controller.submit(
            TrackingDetection(
                u=640.0,
                v=720.0,
                timestamp=started + frame_id * 0.02,
                frame_id=frame_id,
            )
        )
        assert controller.step(dt=0.02)
        target = controller.telemetry.query()["records"][-1]["smoothed_target"]
        elevations.append(np.arctan2(target["z"], target["x"]))

    assert elevations == pytest.approx([-vertical_fov / 2.0] * 20)

    controller.submit(
        TrackingDetection(
            u=1280.0,
            v=360.0,
            timestamp=started + 0.4,
            frame_id=20,
        )
    )
    assert controller.step(dt=0.02)
    target = controller.telemetry.query()["records"][-1]["smoothed_target"]
    assert np.arctan2(target["y"], target["x"]) == pytest.approx(-0.2)

    backend.pose[:3, :3] = R.from_euler("y", 0.35).as_matrix()
    controller.submit(
        TrackingDetection(
            u=640.0,
            v=0.0,
            timestamp=started + 0.42,
            frame_id=21,
        )
    )
    assert controller.step(dt=0.02)
    target = controller.telemetry.query()["records"][-1]["smoothed_target"]
    assert np.arctan2(target["z"], target["x"]) == pytest.approx(-0.25)

    controller.submit(
        TrackingDetection(
            u=640.0,
            v=720.0,
            timestamp=started + 0.44,
            frame_id=22,
        )
    )
    assert controller.step(dt=0.02)
    target = controller.telemetry.query()["records"][-1]["smoothed_target"]
    assert np.arctan2(target["z"], target["x"]) == pytest.approx(-elevation_limit)


def test_detection_target_supports_asymmetric_elevation_limits() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(
            smoothing_alpha=1.0,
            image_vertical_fov=1.0,
            image_error_elevation_limit=0.4,
            image_error_upward_elevation_limit=0.3,
            image_error_downward_elevation_limit=0.2,
        ),
    )
    pose = np.eye(4)

    upward = controller._look_at_target_from_detection(
        TrackingDetection(u=640.0, v=0.0, frame_id=1),
        pose,
        pose,
    )
    downward = controller._look_at_target_from_detection(
        TrackingDetection(u=640.0, v=720.0, frame_id=2),
        pose,
        pose,
    )

    assert np.arctan2(upward.z, upward.x) == pytest.approx(0.3)
    assert np.arctan2(downward.z, downward.x) == pytest.approx(-0.2)


def test_visual_servo_records_profiled_look_at_command() -> None:
    desired = np.array([0.1, *([0.3] * 6)])
    backend = _MotionTestBackend(desired)
    backend.current = np.array([0.1, *([0.2] * 6)])
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02)

    record = controller.telemetry.query()["records"][0]
    for field in ("ik_joints", "profiled_command", "final_command"):
        vector = record[field]
        assert isinstance(vector, list) and len(vector) == 7
        assert all(isinstance(value, float) and np.isfinite(value) for value in vector)
    assert record["profiled_command"][0] == record["ik_joints"][0]
    assert record["final_command"] == record["profiled_command"]
    assert record["limit_hits"] == []


def test_visual_servo_rejects_unsafe_profiled_body_yaw_without_writing() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]

    def unsafe_profile(
        desired: np.ndarray, current: np.ndarray, dt: float
    ) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
        return np.array([0.2, *([0.2] * 6)]), []

    controller.look_at_profile.update_with_telemetry = unsafe_profile  # type: ignore[method-assign]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert not controller.step(dt=0.02)

    assert backend.commands == []
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "safety_rejected"
    assert record["final_command"] is None
    assert {hit["kind"] for hit in record["limit_hits"]} >= {
        "velocity",
        "acceleration",
        "jerk",
    }
    assert controller.status()["motion_state"] == "fault"
    retained_records = len(controller.telemetry.query()["records"])
    assert not controller.step(dt=0.02)
    assert len(controller.telemetry.query()["records"]) == retained_records
    assert not controller.step(dt=0.041)
    assert controller.status()["motion_state"] == "fault"
    assert controller.status()["motion_fault"] == "safety_rejected"


def test_visual_servo_profile_fields_are_empty_without_profile_update() -> None:
    records = []

    idle_controller = VisualServoController(
        backend=_MotionTestBackend()  # type: ignore[arg-type]
    )
    assert not idle_controller.step(dt=0.02)
    records.extend(idle_controller.telemetry.query()["records"])

    busy_backend = _MotionTestBackend()
    busy_backend.is_move_running = True
    busy_controller = VisualServoController(backend=busy_backend)  # type: ignore[arg-type]
    busy_controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert not busy_controller.step(dt=0.02)
    records.extend(busy_controller.telemetry.query()["records"])

    failed_backend = _MotionTestBackend(np.full(7, np.nan))
    failed_controller = VisualServoController(
        backend=failed_backend  # type: ignore[arg-type]
    )
    failed_controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert not failed_controller.step(dt=0.02)
    records.extend(failed_controller.telemetry.query()["records"])

    error_controller = VisualServoController(
        backend=_MotionTestBackend()  # type: ignore[arg-type]
    )

    def fail_step(
        dt: float | None = None, *, use_command_elapsed: bool = False
    ) -> bool:
        error_controller._stop_event.set()
        raise RuntimeError("boom")

    error_controller.step = fail_step  # type: ignore[method-assign]
    error_controller._run_loop()
    records.extend(error_controller.telemetry.query()["records"])

    assert {record["reason"] for record in records} == {
        "no_fresh_detection",
        "move_running",
        "ik_failed",
        "step_error",
    }
    assert all(record["profiled_command"] is None for record in records)
    assert all(record["profile_limit_hits"] == [] for record in records)


def test_visual_servo_look_at_ik_failure_stops_motion_and_recovers() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    command_count = len(backend.commands)
    backend.current = np.full(7, 0.1)
    backend.head_kinematics.joints = np.full(7, np.nan)

    assert controller.step(dt=0.02)

    assert len(backend.commands) == command_count + 1
    assert controller.look_at_profile._position is not None
    assert controller.look_at_guard._last_command is not None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None
    record = controller.telemetry.query()["records"][-1]
    assert record["ik_failed"] is True
    assert record["reason"] in {"stopping_ik_failed", "holding_ik_failed"}

    backend.current[0] += 0.00153398
    backend.head_kinematics.joints = backend.current.copy()
    assert controller.step(dt=0.02)
    recovered = controller.telemetry.query()["records"][-1]
    assert recovered["reason"] == "commanded"
    assert recovered["limit_hits"] == []


def test_visual_servo_no_target_gap_preserves_look_at_motion_state() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(max_detection_age=0.01),
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    reads = (backend.joint_reads, backend.pose_reads)
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0, timestamp=0.0))

    assert controller.step(dt=0.02)

    assert controller.look_at_profile._position is not None
    assert controller.look_at_guard._last_command is not None
    assert controller._last_command_path is None
    assert backend.joint_reads == reads[0] + 1
    assert backend.pose_reads == reads[1]
    assert controller.telemetry.query()["records"][-1]["reason"] in {
        "stopping_no_target",
        "holding_no_target",
    }


def test_visual_servo_recovers_after_hardware_boundary_dropout() -> None:
    config = _acceptance_profile_config()
    config.max_detection_age = 0.01
    desired = np.zeros(7)
    desired[6] = -0.8
    backend = _MotionTestBackend(desired)
    controller = VisualServoController(
        backend=backend,
        config=config,  # type: ignore[arg-type]
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    commands_before_gap = len(backend.commands)

    backend.current[6] = -1.227184630308513
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0, timestamp=0.0))
    assert controller.step(dt=0.02)
    assert len(backend.commands) == commands_before_gap + 1

    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    for _ in range(20):
        assert controller.step(dt=0.02)
        backend.current = backend.commands[-1].copy()

    records = controller.telemetry.query(limit=100)["records"]
    assert not any(record["reason"] == "safety_rejected" for record in records)
    for command in backend.commands:
        assert np.all(command >= controller.look_at_guard.limits[:, 0])
        assert np.all(command <= controller.look_at_guard.limits[:, 1])


def test_visual_servo_commits_monotonic_soft_boundary_recovery() -> None:
    config = _acceptance_profile_config()
    desired = np.zeros(7)
    desired[6] = -0.8
    backend = _MotionTestBackend(desired)
    backend.current[6] = -1.227184630308513
    controller = VisualServoController(
        backend=backend,
        config=config,  # type: ignore[arg-type]
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    violations = []

    for _ in range(30):
        assert controller.step(dt=0.02)
        record = controller.telemetry.query()["records"][-1]
        if record["recovery"]:
            recovery = record["recovery"][0]
            assert recovery["violation_after"] < recovery["violation_before"]
            violations.append(recovery["violation_after"])
        backend.current = backend.commands[-1].copy()

    assert violations
    assert violations == sorted(violations, reverse=True)
    assert controller.telemetry.query()["records"][-1]["reason"] == "commanded"


def test_visual_servo_detection_reuses_look_at_profile_after_look_at() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(max_detection_age=0.01),
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    assert controller.limiter._last_command is None
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0, timestamp=0.0))
    controller.submit(TrackingDetection(u=10.0, v=10.0))

    assert controller.step(dt=0.02)
    assert controller.look_at_profile._position is not None
    assert controller.look_at_guard._last_command is not None
    assert controller.limiter._last_command is None
    assert controller._last_command_path == "detection"


def test_joint_command_limiter_rejects_invalid_inputs_without_state_mutation() -> None:
    limiter = JointCommandLimiter(limits=np.array([[-1.0, 1.0]]))
    invalid_calls = [
        (np.zeros(1), np.zeros(1), 0.0),
        (np.zeros(1), np.zeros(1), -0.1),
        (np.zeros(1), np.zeros(1), float("nan")),
        (np.zeros(1), np.zeros(1), float("inf")),
        (np.zeros(2), np.zeros(1), 0.02),
        (np.zeros(1), np.zeros(2), 0.02),
        (np.array([np.nan]), np.zeros(1), 0.02),
        (np.zeros(1), np.array([np.inf]), 0.02),
    ]

    for desired, current, dt in invalid_calls:
        with pytest.raises(ValueError):
            limiter.limit_with_telemetry(desired, current, dt)
        assert limiter._last_command is None
        np.testing.assert_allclose(limiter._velocity, 0.0)
        np.testing.assert_allclose(limiter._acceleration, 0.0)

    command = limiter.limit(np.array([0.5]), np.array([0.25]), dt=0.02)
    fresh_command = JointCommandLimiter(limits=np.array([[-1.0, 1.0]])).limit(
        np.array([0.5]), np.array([0.25]), dt=0.02
    )
    np.testing.assert_array_equal(command, fresh_command)


def test_visual_servo_rejects_invalid_dt_before_backend_reads() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    for dt in (float("nan"), float("inf"), 0.0, -0.02):
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            controller.step(dt=dt)

    assert backend.joint_reads == 0
    assert backend.pose_reads == 0
    assert backend.commands == []


def test_visual_servo_resets_without_io_after_a_control_stall() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.look_at_profile._position = np.ones(6)
    controller.look_at_guard._last_command = np.ones(7)
    controller.limiter._last_command = np.ones(7)
    controller._last_command_path = "look_at"

    assert not controller.step(dt=0.041)

    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None
    assert backend.joint_reads == 0
    assert backend.pose_reads == 0
    assert backend.commands == []
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "control_stall"
    assert record["dt"] == 0.041


def test_visual_servo_control_stall_preserves_committed_command_anchor() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    committed = backend.commands[-1].copy()
    command_count = len(backend.commands)

    assert not controller.step(dt=0.041)

    np.testing.assert_array_equal(controller.look_at_profile._position, committed[1:])
    np.testing.assert_array_equal(controller.look_at_guard._last_command, committed)
    assert controller.look_at_profile.stationary
    assert len(backend.commands) == command_count
    assert controller.status()["motion_state"] == "holding_no_target"


def test_visual_servo_no_target_command_stall_holds_committed_anchor() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    committed = backend.commands[-1].copy()
    controller._last_command_time = time.monotonic() - 1.0
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0, timestamp=0.0))

    assert not controller.step(dt=0.02, use_command_elapsed=True)

    record = controller.telemetry.query()["records"][-1]
    assert record["reason"] == "control_stall"
    np.testing.assert_array_equal(controller.look_at_guard._last_command, committed)
    assert controller.look_at_profile.stationary
    assert controller.status()["motion_state"] == "holding_no_target"


def test_visual_servo_ik_failure_command_stall_holds_committed_anchor() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    committed = backend.commands[-1].copy()
    command_count = len(backend.commands)
    controller._last_command_time = time.monotonic() - 1.0
    backend.head_kinematics.joints = np.full(7, np.nan)

    assert not controller.step(dt=0.02, use_command_elapsed=True)

    record = controller.telemetry.query()["records"][-1]
    assert record["reason"] == "control_stall"
    assert len(backend.commands) == command_count
    np.testing.assert_array_equal(controller.look_at_guard._last_command, committed)
    assert controller.look_at_profile.stationary


def test_visual_servo_look_at_commands_obey_elapsed_time_limits_under_jitter() -> None:
    config = _acceptance_profile_config()
    desired = np.array([0.0, *([0.3] * 6)])
    backend = _MotionTestBackend(desired)
    controller = VisualServoController(
        backend=backend,
        config=config,  # type: ignore[arg-type]
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    dts = [0.0114, 0.0286, 0.017, 0.023, 0.02] * 25
    previous_command = backend.current.copy()
    previous_velocity = np.zeros(7)
    previous_acceleration = np.zeros(7)

    for dt in dts:
        assert controller.step(dt=dt)
        command = backend.commands[-1]
        velocity = (command - previous_command) / dt
        acceleration = (velocity - previous_velocity) / dt
        jerk = (acceleration - previous_acceleration) / dt
        assert np.max(np.abs(velocity)) <= config.max_joint_velocity + 1e-9
        assert np.max(np.abs(acceleration)) <= config.max_joint_acceleration + 1e-9
        assert np.max(np.abs(jerk)) <= config.max_joint_jerk + 1e-7
        previous_command = command
        previous_velocity = velocity
        previous_acceleration = acceleration
        backend.current = command.copy()

    records = controller.telemetry.query()["records"]
    assert [record["dt"] for record in records] == dts
    assert all(record["reason"] == "commanded" for record in records)
    assert all(
        record["final_command"] == record["profiled_command"] for record in records
    )
    assert all(record["limit_hits"] == [] for record in records)


def test_visual_servo_profiles_body_yaw_encoder_steps() -> None:
    config = _acceptance_profile_config()
    backend = _MotionTestBackend(np.array([0.0, *([0.2] * 6)]))
    backend.current[0] = 0.003067961575771161
    controller = VisualServoController(
        backend=backend,
        config=config,  # type: ignore[arg-type]
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02)
    first = backend.commands[-1].copy()
    backend.current = first.copy()
    backend.current[0] = 0.004601942363656963
    assert controller.step(dt=0.02)
    second = backend.commands[-1]

    assert first[0] < second[0] < backend.current[0]
    assert (second[0] - first[0]) / 0.02 <= config.max_joint_velocity
    records = controller.telemetry.query()["records"]
    assert all(record["reason"] == "commanded" for record in records)
    assert all(record["limit_hits"] == [] for record in records)
    assert all(
        record["final_command"] == record["profiled_command"] for record in records
    )


def test_visual_servo_runtime_dt_matches_command_timestamps() -> None:
    backend = _MotionTestBackend(np.array([0.0, *([0.3] * 6)]))
    controller = VisualServoController(
        backend=backend,
        config=_acceptance_profile_config(),  # type: ignore[arg-type]
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02, use_command_elapsed=True)
    backend.current = backend.commands[-1].copy()
    assert controller.step(dt=0.02, use_command_elapsed=True)

    first, second = controller.telemetry.query()["records"]
    assert first["dt"] == 0.02
    assert second["dt"] == pytest.approx(
        second["monotonic_timestamp"] - first["monotonic_timestamp"]
    )
    assert second["dt"] > 0.0


def test_visual_servo_backend_write_failure_resets_state_and_recovers() -> None:
    desired = np.array([0.2, *([0.6] * 6)])
    backend = _MotionTestBackend(desired)
    backend.fail_writes = 1
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    with pytest.raises(RuntimeError, match="write failed"):
        controller.step(dt=0.02)
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None

    backend.current = np.full(7, 0.4)
    assert controller.step(dt=0.02)

    assert len(backend.commands) == 1
    np.testing.assert_allclose(backend.commands[0][1:], 0.4, atol=0.001)
    assert controller._last_command_path == "look_at"


def test_visual_servo_detection_backend_write_failure_resets_profile_and_recovers() -> (
    None
):
    config = VisualServoConfig()
    desired = np.full(7, 0.2)
    backend = _MotionTestBackend(desired)
    backend.fail_writes = 1
    controller = VisualServoController(
        backend=backend,
        config=config,  # type: ignore[arg-type]
    )
    controller.submit(TrackingDetection(u=10.0, v=10.0))

    with pytest.raises(RuntimeError, match="write failed"):
        controller.step(dt=0.02)
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None

    backend.current = np.full(7, 0.4)
    desired[0] = backend.current[0]
    assert controller.step(dt=0.02)

    assert controller.look_at_profile._position is not None
    assert controller.look_at_guard._last_command is not None
    assert controller._last_command_path == "detection"


def test_visual_servo_non_finite_current_joints_fail_closed_and_recover() -> None:
    backend = _MotionTestBackend(np.full(7, 0.5))
    backend.current[1] = np.nan
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    with pytest.raises(ValueError, match="current must be a finite 7-element vector"):
        controller.step(dt=0.02)
    assert backend.pose_reads == 0
    assert backend.commands == []
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None

    backend.current = np.full(7, 0.3)
    assert controller.step(dt=0.02)
    np.testing.assert_allclose(backend.commands[0][1:], 0.3, atol=0.001)


def test_visual_servo_start_and_stop_reset_motion_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ControlledThread:
        def __init__(self, **_kwargs: object) -> None:
            self.alive = False

        def start(self) -> None:
            self.alive = True

        def join(self, timeout: float | None = None) -> None:
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.look_at_profile._position = np.ones(6)
    controller.look_at_guard._last_command = np.ones(7)
    controller.limiter._last_command = np.ones(7)
    controller._last_command_path = "look_at"
    monkeypatch.setattr(visual_servo_module.threading, "Thread", ControlledThread)

    controller.start()
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None

    controller.look_at_profile._position = np.ones(6)
    controller.look_at_guard._last_command = np.ones(7)
    controller.limiter._last_command = np.ones(7)
    controller._last_command_path = "look_at"
    controller.stop()
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller.limiter._last_command is None
    assert controller._last_command_path is None
    assert backend.joint_reads == 0
    assert backend.pose_reads == 0
    assert backend.commands == []

    class StuckThread:
        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    timed_out = VisualServoController(backend=backend)  # type: ignore[arg-type]
    timed_out._thread = StuckThread()  # type: ignore[assignment]
    timed_out.look_at_profile._position = np.ones(6)
    timed_out.look_at_guard._last_command = np.ones(7)
    timed_out.limiter._last_command = np.ones(7)
    timed_out._last_command_path = "look_at"

    timed_out.stop()

    np.testing.assert_array_equal(timed_out.look_at_profile._position, np.ones(6))
    np.testing.assert_array_equal(timed_out.look_at_guard._last_command, np.ones(7))
    np.testing.assert_array_equal(timed_out.limiter._last_command, np.ones(7))
    assert timed_out._last_command_path == "look_at"


def test_detection_buffer_keeps_only_latest_detection() -> None:
    buffer = DetectionBuffer()

    first = TrackingDetection(u=100.0, v=200.0, timestamp=1.0, frame_id=1)
    second = TrackingDetection(u=300.0, v=400.0, timestamp=2.0, frame_id=2)

    buffer.submit(first)
    buffer.submit(second)

    assert buffer.latest() == second


def test_detection_buffer_rejects_stale_and_low_confidence_detection() -> None:
    now = time.time()
    config = VisualServoConfig(max_detection_age=0.1, min_confidence=0.5)
    buffer = DetectionBuffer()

    buffer.submit(
        TrackingDetection(
            u=320.0,
            v=240.0,
            timestamp=now - 1.0,
            confidence=1.0,
        )
    )
    buffer.submit(
        TrackingDetection(
            u=320.0,
            v=240.0,
            timestamp=now,
            confidence=0.1,
        )
    )

    assert buffer.fresh(config=config, now=now) is None


def test_joint_command_limiter_clamps_to_safe_limits() -> None:
    limits = np.array([[-0.5, 0.5], [-1.0, 1.0]], dtype=np.float64)
    config = VisualServoConfig(
        joint_safety_margin=0.1,
        max_joint_velocity=1e9,
        max_joint_acceleration=1e9,
        max_joint_jerk=1e9,
    )
    limiter = JointCommandLimiter(limits=limits, config=config)

    command = limiter.limit(
        desired=np.array([10.0, -10.0]),
        current=np.array([0.0, 0.0]),
        dt=0.02,
    )

    np.testing.assert_allclose(command, np.array([0.4, -0.9]))


def test_joint_command_limiter_rejects_invalid_limit_shape() -> None:
    with np.testing.assert_raises(ValueError):
        JointCommandLimiter(limits=np.array([1.0, 2.0], dtype=np.float64))


def test_joint_command_limiter_limits_velocity_acceleration_and_jerk() -> None:
    limits = np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=np.float64)
    config = VisualServoConfig(
        joint_safety_margin=0.0,
        max_joint_velocity=1.0,
        max_joint_acceleration=2.0,
        max_joint_jerk=10.0,
    )
    limiter = JointCommandLimiter(limits=limits, config=config)

    first = limiter.limit(
        desired=np.array([10.0, -10.0]),
        current=np.array([0.0, 0.0]),
        dt=0.02,
    )
    second = limiter.limit(
        desired=np.array([10.0, -10.0]),
        current=first,
        dt=0.02,
    )

    np.testing.assert_allclose(first, np.array([0.00008, -0.00008]))
    np.testing.assert_allclose(second, np.array([0.00032, -0.00032]))


def test_joint_command_limiter_does_not_reset_when_joint_state_lags_command() -> None:
    limits = np.array([[-10.0, 10.0]], dtype=np.float64)
    config = VisualServoConfig(
        joint_safety_margin=0.0,
        max_joint_velocity=1.0,
        max_joint_acceleration=2.0,
        max_joint_jerk=10.0,
    )
    limiter = JointCommandLimiter(limits=limits, config=config)

    current = np.array([0.0])
    command = current
    for _ in range(30):
        command = limiter.limit(
            desired=np.array([1.0]),
            current=current,
            dt=0.02,
        )
        current = current + 0.25 * (command - current)

    assert command[0] > 0.01


def test_joint_command_limiter_settles_on_constant_target() -> None:
    limits = np.array([[-10.0, 10.0]], dtype=np.float64)
    config = VisualServoConfig(
        joint_safety_margin=0.0,
        max_joint_velocity=0.30,
        max_joint_acceleration=0.80,
        max_joint_jerk=4.0,
    )
    limiter = JointCommandLimiter(limits=limits, config=config)

    current = np.array([0.0])
    commands = []
    for _ in range(600):
        command = limiter.limit(
            desired=np.array([0.5]),
            current=current,
            dt=0.02,
        )
        current = command.copy()
        commands.append(float(command[0]))

    settled_commands = commands[-100:]
    assert max(settled_commands) - min(settled_commands) < 0.01
    assert abs(commands[-1] - 0.5) < 0.01


def test_pixel_filter_eases_first_detection_from_image_center() -> None:
    target_filter = PixelTargetFilter(alpha=0.5)

    filtered = target_filter.update(
        TrackingDetection(u=0.0, v=0.0, width=100, height=50)
    )

    np.testing.assert_allclose(filtered, np.array([25.0, 12.5]))


def test_look_at_filter_eases_successive_metric_targets() -> None:
    target_filter = LookAtTargetFilter(alpha=0.25)

    first = target_filter.update(
        TrackingLookAtTarget(
            x=0.5,
            y=0.0,
            z=0.0,
            timestamp=10.0,
            confidence=0.8,
            frame_id=1,
        )
    )
    second = target_filter.update(
        TrackingLookAtTarget(
            x=0.5,
            y=0.3,
            z=0.2,
            timestamp=11.0,
            confidence=0.7,
            frame_id=2,
        )
    )

    assert first == TrackingLookAtTarget(
        x=0.5,
        y=0.0,
        z=0.0,
        timestamp=10.0,
        confidence=0.8,
        frame_id=1,
    )
    assert second == TrackingLookAtTarget(
        x=0.5,
        y=0.075,
        z=0.05,
        timestamp=11.0,
        confidence=0.7,
        frame_id=2,
    )


def test_visual_servo_detection_ik_failure_recovers_on_new_detection() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.calls = 0

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.calls += 1
            if self.calls == 1:
                return np.full(7, np.nan)
            return np.full(7, 0.2)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.command: np.ndarray | None = None

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit(TrackingDetection(u=0.0, v=0.0, width=100, height=50))

    assert not controller.step(dt=0.02)
    controller.submit(TrackingDetection(u=0.0, v=0.0, width=100, height=50))

    assert controller.step(dt=0.02)
    assert backend.head_kinematics.calls == 2
    assert backend.command is not None


def test_visual_servo_commands_from_3d_look_at_target() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.last_pose: np.ndarray | None = None

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.last_pose = pose
            return np.array([body_yaw, *([0.2] * 6)])

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.command: np.ndarray | None = None

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    commanded = controller.step(dt=0.02)

    assert commanded
    assert controller.status()["accepted_look_at_targets"] == 1
    assert controller.status()["last_target_type"] == "look_at"
    assert backend.command is not None
    assert backend.head_kinematics.last_pose is not None


def test_visual_servo_records_smoothed_look_at_target() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.forward_axes: list[np.ndarray] = []

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.forward_axes.append(pose[:3, 0].copy())
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(
            smoothing_alpha=0.5,
            max_joint_velocity=1e9,
            max_joint_acceleration=1e9,
            max_joint_jerk=1e9,
        ),
    )
    controller.submit_look_at(
        TrackingLookAtTarget(
            x=0.5,
            y=0.0,
            z=0.0,
            timestamp=time.time(),
            frame_id=1,
        )
    )
    assert controller.step(dt=0.02)
    controller.submit_look_at(
        TrackingLookAtTarget(
            x=0.5,
            y=0.2,
            z=0.2,
            timestamp=time.time(),
            frame_id=2,
        )
    )
    assert controller.step(dt=0.02)

    records = controller.telemetry.query()["records"]
    assert records[0]["input_target"]["y"] == 0.0  # type: ignore[index]
    assert records[0]["smoothed_target"]["y"] == 0.0  # type: ignore[index]
    assert records[1]["input_target"]["y"] == 0.2  # type: ignore[index]
    assert records[1]["smoothed_target"] == {
        "kind": "look_at",
        "x": 0.5,
        "y": 0.1,
        "z": 0.1,
        "timestamp": records[1]["input_target"]["timestamp"],  # type: ignore[index]
        "confidence": 1.0,
        "frame_id": 2,
    }
    expected_forward = np.array([0.5, 0.1, 0.1], dtype=np.float64)
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(
        backend.head_kinematics.forward_axes[1], expected_forward
    )


def test_visual_servo_3d_look_at_uses_world_up_for_predictable_target_plane() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.last_pose: np.ndarray | None = None

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.last_pose = pose
            return np.array([body_yaw, *([0.2] * 6)])

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.command: np.ndarray | None = None

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            pose = np.eye(4)
            pose[:3, :3] = R.from_euler("x", 0.7).as_matrix()
            return pose

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    commanded = controller.step(dt=0.02)

    assert commanded
    assert backend.head_kinematics.last_pose is not None
    np.testing.assert_allclose(
        backend.head_kinematics.last_pose[:3, 2],
        np.array([0.0, 0.0, 1.0]),
        atol=1e-12,
    )


def test_visual_servo_3d_look_at_keeps_fixed_reference_origin() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.pose_origins: list[np.ndarray] = []

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.pose_origins.append(pose[:3, 3].copy())
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.pose_index = 0

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            pose = np.eye(4)
            if self.pose_index > 0:
                pose[:3, 3] = np.array([0.05, -0.02, 0.03])
            return pose

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.pose_index += 1

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02)
    assert controller.step(dt=0.02)
    assert len(backend.head_kinematics.pose_origins) == 2
    np.testing.assert_allclose(
        backend.head_kinematics.pose_origins[1],
        backend.head_kinematics.pose_origins[0],
        atol=1e-12,
    )


def test_visual_servo_detection_refreshes_reference_across_target_gap() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.poses: list[np.ndarray] = []

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.poses.append(pose.copy())
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.pose_index = 0

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            pose = np.eye(4)
            if self.pose_index > 0:
                pose[:3, 3] = np.array([0.05, -0.02, 0.03])
            return pose

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.pose_index += 1

    backend = FakeBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(smoothing_alpha=1.0, max_detection_age=0.01),
    )
    controller.submit(TrackingDetection(u=640.0, v=360.0, frame_id=0))
    assert controller.step(dt=0.02)
    controller.submit(TrackingDetection(u=640.0, v=360.0, frame_id=1, timestamp=0.0))
    assert controller.step(dt=0.02)
    controller.submit(TrackingDetection(u=640.0, v=360.0, frame_id=1))
    assert controller.step(dt=0.02)

    assert len(backend.head_kinematics.poses) == 2
    np.testing.assert_allclose(
        backend.head_kinematics.poses[1][:3, 3],
        np.array([0.05, -0.02, 0.03]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        backend.head_kinematics.poses[1][:3, 0],
        np.array([1.0, 0.0, 0.0]),
        atol=1e-12,
    )


def test_visual_servo_restores_previous_automatic_body_yaw_on_stop() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.automatic_body_yaw = False
            self.states: list[bool] = []

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw
            self.states.append(automatic_body_yaw)

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]

    controller.start()
    controller.stop()

    assert backend.head_kinematics.states[0] is True
    assert backend.head_kinematics.states[-1] is False
    assert backend.head_kinematics.automatic_body_yaw is False


def test_visual_servo_stop_reports_running_when_worker_does_not_exit() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.automatic_body_yaw = False
            self.states: list[bool] = []

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw
            self.states.append(automatic_body_yaw)

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    class StuckThread:
        def join(self, timeout: float | None = None) -> None:
            self.timeout = timeout

        def is_alive(self) -> bool:
            return True

    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.start()
    controller._thread = StuckThread()  # type: ignore[assignment]

    controller.stop()

    status = controller.status()
    assert status["running"]
    assert status["last_reason"] == "stop_timeout"
    assert status["error"] == "Visual servo thread did not stop within 2.0s."
    assert backend.head_kinematics.automatic_body_yaw is True


def test_visual_servo_uses_backend_motion_guard_for_joint_write() -> None:
    class FakeKinematics:
        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.array([body_yaw, *([0.2] * 6)])

    class GuardedBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.guard_depth = 0
            self.command: np.ndarray | None = None

        def _try_start_move(self) -> bool:
            self.guard_depth += 1
            return True

        def _end_move(self) -> None:
            self.guard_depth -= 1

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            assert self.guard_depth == 1
            self.command = command

    backend = GuardedBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02)
    assert backend.command is not None
    assert backend.guard_depth == 0


def test_visual_servo_skips_command_when_backend_motion_guard_is_busy() -> None:
    class FakeKinematics:
        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.array([body_yaw, *([0.2] * 6)])

    class BusyBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def _try_start_move(self) -> bool:
            return False

        def _end_move(self) -> None:
            raise AssertionError("busy guard must not be released")

        def get_present_head_joint_positions(self) -> np.ndarray:
            raise AssertionError("servo must not read joints without motion ownership")

        def get_present_head_pose(self) -> np.ndarray:
            raise AssertionError("servo must not read pose without motion ownership")

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            raise AssertionError(
                "servo must not command joints without motion ownership"
            )

    backend = BusyBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert not controller.step(dt=0.02)
    assert controller.status()["last_reason"] == "move_running"


def test_visual_servo_does_not_acquire_motion_guard_without_fresh_target() -> None:
    class FakeKinematics:
        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.full(7, 0.2)

    class GuardedBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.guard_attempts = 0

        def _try_start_move(self) -> bool:
            self.guard_attempts += 1
            return True

        def _end_move(self) -> None:
            raise AssertionError("idle servo must not acquire the motion guard")

        def get_present_head_joint_positions(self) -> np.ndarray:
            raise AssertionError("idle servo must not read joints")

        def get_present_head_pose(self) -> np.ndarray:
            raise AssertionError("idle servo must not read pose")

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            raise AssertionError("idle servo must not command joints")

    backend = GuardedBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]

    assert not controller.step(dt=0.02)
    assert backend.guard_attempts == 0
    assert controller.status()["last_reason"] == "no_fresh_detection"


def test_tracking_telemetry_buffer_drops_old_records() -> None:
    buffer = VisualServoTelemetryBuffer(capacity=2)

    buffer.append({"timestamp": 10.0, "sequence": 0, "reason": "a"})
    buffer.append({"timestamp": 11.0, "sequence": 1, "reason": "b"})
    buffer.append({"timestamp": 12.0, "sequence": 2, "reason": "c"})

    result = buffer.query()

    assert result["dropped_records"] == 1
    assert result["oldest_sequence"] == 1
    assert result["newest_sequence"] == 2
    assert result["oldest_timestamp"] == 11.0
    assert result["newest_timestamp"] == 12.0
    assert [record["reason"] for record in result["records"]] == ["b", "c"]


def test_latest_telemetry_returns_only_an_isolated_newest_record() -> None:
    buffer = VisualServoTelemetryBuffer(capacity=3000)
    for sequence in range(3000):
        buffer.append({"sequence": sequence, "nested": {"value": sequence}})

    latest = buffer.latest()

    assert latest == {"sequence": 2999, "nested": {"value": 2999}}
    assert latest is not None
    latest["sequence"] = -1
    assert buffer.latest() == {"sequence": 2999, "nested": {"value": 2999}}


def test_tracking_telemetry_query_filters_by_time_sequence_and_limit() -> None:
    buffer = VisualServoTelemetryBuffer(capacity=10)
    for sequence in range(5):
        buffer.append(
            {
                "timestamp": 100.0 + sequence,
                "sequence": sequence,
                "reason": str(sequence),
            }
        )

    result = buffer.query(
        from_timestamp=101.0,
        to_timestamp=104.0,
        from_sequence=2,
        to_sequence=4,
        limit=2,
    )

    assert [record["sequence"] for record in result["records"]] == [2, 3]
    assert result["returned"] == 2
    assert result["limit"] == 2
    assert result["oldest_sequence"] == 0
    assert result["newest_sequence"] == 4


def test_finite_json_value_replaces_non_finite_numpy_values() -> None:
    value = finite_json_value(np.array([0.1, np.nan, np.inf, -np.inf]))

    assert value == [0.1, None, None, None]


def test_tracking_telemetry_jsonl_writes_strict_json_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"

    dump_jsonl(
        [
            {
                "timestamp": 1.0,
                "sequence": 1,
                "reason": "non_finite",
                "value": float("nan"),
                "nested": {"value": np.inf},
            }
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert load_jsonl(path) == [
        {
            "timestamp": 1.0,
            "sequence": 1,
            "reason": "non_finite",
            "value": None,
            "nested": {"value": None},
        }
    ]


def test_tracking_telemetry_load_rejects_non_object_and_non_finite_jsonl(
    tmp_path: Path,
) -> None:
    array_path = tmp_path / "array.jsonl"
    array_path.write_text("[1,2,3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSONL row must be an object"):
        load_jsonl(array_path)

    non_finite_path = tmp_path / "non_finite.jsonl"
    non_finite_path.write_text('{"value":NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_jsonl(non_finite_path)


def test_tracking_telemetry_summary_skips_empty_and_mismatched_commands() -> None:
    summary = summarize_records(
        [
            {"timestamp": 1.0, "sequence": 1, "final_command": []},
            {"timestamp": 2.0, "sequence": 2, "final_command": [1.0, 2.0]},
            {"timestamp": 3.0, "sequence": 3, "final_command": [2.0]},
        ]
    )

    assert summary["command_smoothness"] == {
        "max_velocity": None,
        "max_acceleration": None,
        "max_jerk": None,
    }


def test_tracking_telemetry_summary_counts_every_backend_command() -> None:
    records = [
        {
            "timestamp": 1.0,
            "sequence": 1,
            "reason": "recovering",
            "final_command": [0.0] * 7,
        },
        {
            "timestamp": 2.0,
            "sequence": 2,
            "reason": "holding_no_target",
            "final_command": [0.0] * 7,
        },
        {
            "timestamp": 3.0,
            "sequence": 3,
            "reason": "safety_rejected",
            "final_command": None,
        },
    ]

    summary = summarize_records(records)

    assert summary["command_count"] == 2
    assert summary["reason_counts"] == {
        "recovering": 1,
        "holding_no_target": 1,
        "safety_rejected": 1,
    }


def test_tracking_telemetry_summary_includes_profile_and_final_smoothness() -> None:
    records = [
        {
            "timestamp": float(index),
            "profiled_command": profiled,
            "final_command": final,
        }
        for index, (profiled, final) in enumerate(
            (
                ([0.0, 0.0], [0.0, 0.0]),
                ([1.0, 0.0], [0.5, 0.0]),
                ([1.0, 1.0], [0.5, 0.5]),
                ([2.0, 1.0], [1.0, 0.5]),
            ),
            start=1,
        )
    ]

    summary = summarize_records(records)

    keys = {"max_velocity", "max_acceleration", "max_jerk"}
    assert set(summary["profiled_command_smoothness"]) == keys
    assert set(summary["final_command_smoothness"]) == keys
    assert all(
        isinstance(value, float)
        for value in summary["profiled_command_smoothness"].values()
    )
    assert all(
        isinstance(value, float)
        for value in summary["final_command_smoothness"].values()
    )
    assert summary["command_smoothness"] == summary["final_command_smoothness"]


def test_tracking_telemetry_smoothness_prefers_monotonic_timestamps() -> None:
    records = [
        {
            "timestamp": wall_time,
            "monotonic_timestamp": monotonic_time,
            "final_command": [command],
        }
        for wall_time, monotonic_time, command in (
            (1.0, 10.0, 0.0),
            (1.1, 11.0, 1.0),
            (2.1, 12.0, 2.0),
        )
    ]

    summary = summarize_records(records)

    assert summary["final_command_smoothness"] == {
        "max_velocity": 1.0,
        "max_acceleration": 0.0,
        "max_jerk": None,
    }


def test_tracking_telemetry_query_rejects_non_integer_bounds() -> None:
    buffer = VisualServoTelemetryBuffer()

    with pytest.raises(
        ValueError,
        match="timestamp filters must be finite real numbers",
    ):
        buffer.query(from_timestamp=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="from_sequence must be an integer"):
        buffer.query(from_sequence=0.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="limit must be an integer"):
        buffer.query(limit=True)  # type: ignore[arg-type]


def test_joint_command_limiter_reports_limit_hits_without_changing_command() -> None:
    limits = np.array([[-0.5, 0.5]], dtype=np.float64)

    def find_hit(
        hits: list[dict[str, float | int | str]],
        kind: str,
    ) -> dict[str, float | int | str]:
        matches = [hit for hit in hits if hit["kind"] == kind]
        assert len(matches) == 1
        return matches[0]

    def run_case(config: VisualServoConfig) -> list[dict[str, float | int | str]]:
        plain_limiter = JointCommandLimiter(limits=limits, config=config)
        telemetry_limiter = JointCommandLimiter(limits=limits, config=config)
        command = plain_limiter.limit(
            desired=np.array([10.0]),
            current=np.array([0.0]),
            dt=0.02,
        )
        command_with_telemetry, telemetry = telemetry_limiter.limit_with_telemetry(
            desired=np.array([10.0]),
            current=np.array([0.0]),
            dt=0.02,
        )

        np.testing.assert_allclose(command_with_telemetry, command)
        assert telemetry["clamped_desired"] == [0.4]
        return telemetry["limit_hits"]

    position_hits = run_case(
        VisualServoConfig(
            joint_safety_margin=0.1,
            max_joint_velocity=1e9,
            max_joint_acceleration=1e9,
            max_joint_jerk=1e9,
        )
    )
    velocity_hits = run_case(
        VisualServoConfig(
            joint_safety_margin=0.1,
            max_joint_velocity=0.01,
            max_joint_acceleration=1e9,
            max_joint_jerk=1e9,
        )
    )
    acceleration_hits = run_case(
        VisualServoConfig(
            joint_safety_margin=0.1,
            max_joint_velocity=1e9,
            max_joint_acceleration=0.02,
            max_joint_jerk=1e9,
        )
    )
    jerk_hits = run_case(
        VisualServoConfig(
            joint_safety_margin=0.1,
            max_joint_velocity=1e9,
            max_joint_acceleration=1e9,
            max_joint_jerk=0.03,
        )
    )

    position_hit = find_hit(position_hits, "upper_position")
    velocity_hit = find_hit(velocity_hits, "velocity")
    acceleration_hit = find_hit(acceleration_hits, "acceleration")
    jerk_hit = find_hit(jerk_hits, "jerk")

    assert float(position_hit["value"]) > float(position_hit["limit"])
    assert float(velocity_hit["value"]) > float(velocity_hit["limit"])
    assert float(acceleration_hit["value"]) > float(acceleration_hit["limit"])
    assert float(jerk_hit["value"]) > float(jerk_hit["limit"])


def test_visual_servo_records_no_fresh_detection_without_backend_reads() -> None:
    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = object()

        def get_present_head_joint_positions(self) -> np.ndarray:
            raise AssertionError("no fresh target must not read joints")

        def get_present_head_pose(self) -> np.ndarray:
            raise AssertionError("no fresh target must not read pose")

    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]

    assert not controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "no_fresh_detection"
    assert record["target_type"] == "none"
    assert record["current_joints"] is None
    assert record["current_pose"] is None
    assert record["actual_joints"] is None
    assert record["final_command"] is None


def test_visual_servo_records_busy_motion_guard_without_backend_reads() -> None:
    class FakeKinematics:
        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.zeros(7)

    class BusyBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def _try_start_move(self) -> bool:
            return False

        def _end_move(self) -> None:
            raise AssertionError("busy guard must not be released")

        def get_present_head_joint_positions(self) -> np.ndarray:
            raise AssertionError("busy guard must not read joints")

        def get_present_head_pose(self) -> np.ndarray:
            raise AssertionError("busy guard must not read pose")

    controller = VisualServoController(backend=BusyBackend())  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert not controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "move_running"
    assert record["target_type"] == "look_at"
    assert record["current_joints"] is None
    assert record["current_pose"] is None


def test_visual_servo_records_commanded_look_at_tick() -> None:
    class FakeKinematics:
        automatic_body_yaw = True

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.array([body_yaw, *([0.2] * 6)])

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.error = None

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "commanded"
    assert record["target_type"] == "look_at"
    assert record["input_target"]["kind"] == "look_at"  # type: ignore[index]
    assert record["current_joints"] == [0.0] * 7
    assert record["actual_joints"] == [0.0] * 7
    assert record["actual_joints_source"] == "present_read_before_command"
    assert record["final_command"] is not None
    assert record["ik_joints"] == [0.0, *([0.2] * 6)]
    assert record["ik_target"] is not None
    assert record["body_yaw"]["current"] == 0.0  # type: ignore[index]
    assert record["body_yaw"]["ik_input"] == 0.0  # type: ignore[index]
    assert record["latency"]["target_age"] is not None  # type: ignore[index]


def test_visual_servo_records_non_finite_ik_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTime:
        def __init__(self) -> None:
            self.monotonic_values = iter([1.0, 1.1, 1.7])

        def monotonic(self) -> float:
            return next(self.monotonic_values)

        def time(self) -> float:
            return 100.0

    class FakeKinematics:
        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.array([0.0, np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0])

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.command_called = False

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command_called = True
            raise AssertionError("IK failure must not command")

    monkeypatch.setattr(visual_servo_module, "time", FakeTime())
    backend = FakeBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    assert not controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "ik_failed"
    assert record["ik_failed"] is True
    assert record["ik_joints"] == [0.0, None, None, 0.0, 0.0, 0.0, 0.0]
    assert record["final_command"] is None
    assert record["latency"]["processing_duration"] == pytest.approx(0.7)  # type: ignore[index]
    assert backend.command_called is False


def test_visual_servo_records_detection_as_generated_look_at_target() -> None:
    class FakeKinematics:
        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.full(7, 0.2)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]
    controller.submit(TrackingDetection(u=0.0, v=0.0, width=100, height=50))

    assert controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][0]
    assert record["input_target"]["kind"] == "detection"
    assert record["smoothed_target"]["kind"] == "look_at"
    assert record["profiled_command"] is not None
    assert record["projected_target"] is None


def test_visual_servo_run_loop_records_step_error() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.head_kinematics = object()

    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]

    def fail_step(
        dt: float | None = None, *, use_command_elapsed: bool = False
    ) -> bool:
        controller._stop_event.set()
        raise RuntimeError("boom")

    controller.step = fail_step  # type: ignore[method-assign]
    controller._run_loop()

    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "step_error"
    assert record["error"] == "boom"


def test_visual_servo_run_loop_uses_elapsed_time_without_catch_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.head_kinematics = object()

    class FakeStopEvent:
        def __init__(self) -> None:
            self.stopped = False
            self.waits: list[float] = []

        def is_set(self) -> bool:
            return self.stopped

        def set(self) -> None:
            self.stopped = True

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            clock[0] += timeout
            return self.stopped

    clock = [10.0]
    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]
    stop_event = FakeStopEvent()
    controller._stop_event = stop_event  # type: ignore[assignment]
    dts: list[float] = []
    processing_times = iter([0.005, 0.03, 0.0])

    def monotonic() -> float:
        return clock[0]

    def step(dt: float | None = None, *, use_command_elapsed: bool = False) -> bool:
        assert dt is not None
        assert use_command_elapsed
        dts.append(dt)
        clock[0] += next(processing_times)
        if len(dts) == 3:
            stop_event.set()
        return False

    monkeypatch.setattr(visual_servo_module.time, "monotonic", monotonic)
    controller.step = step  # type: ignore[method-assign]

    controller._run_loop()

    assert dts == pytest.approx([0.02, 0.02, 0.03])
    assert stop_event.waits == pytest.approx([0.015, 0.0, 0.02])
