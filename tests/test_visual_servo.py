# ruff: noqa: D100,D103
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from reachy_mini.daemon.tracking import visual_servo as visual_servo_module
from reachy_mini.daemon.tracking.config import VisualServoConfig
from reachy_mini.daemon.tracking.telemetry import (
    VisualServoTelemetryBuffer,
    dump_jsonl,
    finite_json_value,
    load_jsonl,
    summarize_records,
)
from reachy_mini.daemon.tracking.visual_servo import (
    LatestTargetBuffer,
    TrackingDetection,
    TrackingLookAtTarget,
    VisualServoController,
)


def _motion_config() -> VisualServoConfig:
    return VisualServoConfig(
        joint_safety_margin=0.1745329252,
        max_joint_velocity=0.60,
        max_joint_acceleration=1.60,
        max_joint_jerk=8.0,
        look_at_profile_response_hz=1.0,
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
            image_horizontal_fov=0.4,
            image_vertical_fov=vertical_fov,
            image_error_upward_elevation_limit=elevation_limit,
            image_error_downward_elevation_limit=elevation_limit,
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
        target = controller.telemetry.query()["records"][-1]["look_at_target"]
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
    target = controller.telemetry.query()["records"][-1]["look_at_target"]
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
    target = controller.telemetry.query()["records"][-1]["look_at_target"]
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
    target = controller.telemetry.query()["records"][-1]["look_at_target"]
    assert np.arctan2(target["z"], target["x"]) == pytest.approx(-elevation_limit)


def test_detection_target_supports_asymmetric_elevation_limits() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(
            image_vertical_fov=1.0,
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


def test_visual_servo_rejects_measured_hard_limit_without_writing() -> None:
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    assert len(backend.commands) == 1

    backend.current[6] = controller.look_at_guard.limits[6, 0] - 0.001

    assert not controller.step(dt=0.02)
    assert len(backend.commands) == 1
    record = controller.telemetry.query()["records"][-1]
    assert record["reason"] == "safety_rejected"
    assert record["limit_hits"] == [
        {
            "joint_index": 6,
            "kind": "current_lower_hard_position",
            "value": pytest.approx(backend.current[6]),
            "limit": pytest.approx(controller.look_at_guard.limits[6, 0]),
        }
    ]
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


def test_visual_servo_no_target_gap_preserves_look_at_motion_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(max_detection_age=0.01),
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    reads = (backend.joint_reads, backend.pose_reads)
    now = 10.02

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


def test_visual_servo_recovers_after_hardware_boundary_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    config = _motion_config()
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
    now = 10.02
    assert controller.step(dt=0.02)
    assert len(backend.commands) == commands_before_gap + 1

    now = 10.03
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
    config = _motion_config()
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


def test_visual_servo_detection_reuses_look_at_profile_after_look_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    backend = _MotionTestBackend()
    controller = VisualServoController(
        backend=backend,  # type: ignore[arg-type]
        config=VisualServoConfig(max_detection_age=0.01),
    )
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    now = 10.02
    controller.submit(TrackingDetection(u=10.0, v=10.0))

    assert controller.step(dt=0.02)
    assert controller.look_at_profile._position is not None
    assert controller.look_at_guard._last_command is not None
    assert controller._last_command_path == "detection"


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
    controller.look_at_profile._position = np.ones(7)
    controller.look_at_guard._last_command = np.ones(7)
    controller._last_command_path = "look_at"

    assert not controller.step(dt=0.041)

    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
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

    np.testing.assert_array_equal(controller.look_at_profile._position, committed)
    np.testing.assert_array_equal(controller.look_at_guard._last_command, committed)
    assert controller.look_at_profile.stationary
    assert len(backend.commands) == command_count
    assert controller.status()["motion_state"] == "holding_no_target"


def test_visual_servo_no_target_command_stall_holds_committed_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    backend = _MotionTestBackend()
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))
    assert controller.step(dt=0.02)
    committed = backend.commands[-1].copy()
    controller._last_command_time = 9.0
    now = 11.0

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
    config = _motion_config()
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
    config = _motion_config()
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
        config=_motion_config(),  # type: ignore[arg-type]
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
    controller.look_at_profile._position = np.ones(7)
    controller.look_at_guard._last_command = np.ones(7)
    controller._last_command_path = "look_at"
    monkeypatch.setattr(visual_servo_module.threading, "Thread", ControlledThread)

    controller.start()
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
    assert controller._last_command_path is None

    controller.look_at_profile._position = np.ones(7)
    controller.look_at_guard._last_command = np.ones(7)
    controller._last_command_path = "look_at"
    controller.stop()
    assert controller.look_at_profile._position is None
    assert controller.look_at_guard._last_command is None
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
    timed_out.look_at_profile._position = np.ones(7)
    timed_out.look_at_guard._last_command = np.ones(7)
    timed_out._last_command_path = "look_at"

    timed_out.stop()

    np.testing.assert_array_equal(timed_out.look_at_profile._position, np.ones(7))
    np.testing.assert_array_equal(timed_out.look_at_guard._last_command, np.ones(7))
    assert timed_out._last_command_path == "look_at"


def test_latest_target_buffer_keeps_only_latest_detection() -> None:
    buffer = LatestTargetBuffer[TrackingDetection]()

    first = TrackingDetection(u=100.0, v=200.0, timestamp=1.0, frame_id=1)
    second = TrackingDetection(u=300.0, v=400.0, timestamp=2.0, frame_id=2)

    buffer.submit(first)
    buffer.submit(second)

    assert buffer.latest() == second


def test_latest_target_buffer_rejects_stale_and_low_confidence_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = VisualServoConfig(max_detection_age=0.1, min_confidence=0.5)
    buffer = LatestTargetBuffer[TrackingDetection]()
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: 10.0)

    buffer.submit(
        TrackingDetection(
            u=320.0,
            v=240.0,
            confidence=1.0,
        )
    )
    assert buffer.fresh(config=config, now=10.2) is None

    buffer.submit(
        TrackingDetection(
            u=320.0,
            v=240.0,
            confidence=0.1,
        )
    )

    assert buffer.fresh(config=config, now=10.0) is None


@pytest.mark.parametrize(
    ("buffer", "target"),
    [
        (
            LatestTargetBuffer[TrackingDetection](),
            TrackingDetection(u=320.0, v=240.0, timestamp=10**12),
        ),
        (
            LatestTargetBuffer[TrackingLookAtTarget](),
            TrackingLookAtTarget(x=0.5, y=0.0, z=0.0, timestamp=10**12),
        ),
    ],
)
def test_target_expiry_uses_daemon_receipt_time_not_source_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    buffer: LatestTargetBuffer[TrackingDetection]
    | LatestTargetBuffer[TrackingLookAtTarget],
    target: TrackingDetection | TrackingLookAtTarget,
) -> None:
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: 10.0)
    buffer.submit(target)  # type: ignore[arg-type]

    assert (
        buffer.fresh(
            VisualServoConfig(max_detection_age=0.1),
            now=10.2,
        )
        is None
    )


def test_target_status_and_telemetry_age_use_daemon_receipt_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)
    controller = VisualServoController(
        backend=_MotionTestBackend()  # type: ignore[arg-type]
    )
    controller.submit(TrackingDetection(u=640.0, v=360.0, timestamp=10**12))
    now = 10.05

    assert controller.status()["last_detection_age"] == pytest.approx(0.05)
    assert controller.step(dt=0.02)
    record = controller.telemetry.query()["records"][-1]
    assert record["latency"]["target_age"] == pytest.approx(0.05)


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


def test_visual_servo_uses_latest_look_at_target_without_hidden_filtering() -> None:
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
    assert records[0]["look_at_target"]["y"] == 0.0  # type: ignore[index]
    assert records[1]["input_target"]["y"] == 0.2  # type: ignore[index]
    assert records[1]["look_at_target"] == {
        "kind": "look_at",
        "x": 0.5,
        "y": 0.2,
        "z": 0.2,
        "timestamp": records[1]["input_target"]["timestamp"],  # type: ignore[index]
        "confidence": 1.0,
        "frame_id": 2,
    }
    expected_forward = np.array([0.5, 0.2, 0.2], dtype=np.float64)
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


def test_visual_servo_detection_refreshes_reference_across_target_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(visual_servo_module.time, "monotonic", lambda: now)

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
        config=VisualServoConfig(max_detection_age=0.01),
    )
    controller.submit(TrackingDetection(u=640.0, v=360.0, frame_id=0))
    assert controller.step(dt=0.02)
    now = 10.02
    assert controller.step(dt=0.02)
    now = 10.03
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
            self.monotonic_values = iter([1.0, 1.1, 1.1, 1.1, 1.1, 1.8])

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
    assert record["look_at_target"]["kind"] == "look_at"
    assert record["profiled_command"] is not None


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
