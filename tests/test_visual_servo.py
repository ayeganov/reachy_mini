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
    PixelTargetFilter,
    TrackingDetection,
    TrackingLookAtTarget,
    VisualServoConfig,
    VisualServoController,
)


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


def test_visual_servo_reduces_unreachable_pixel_until_ik_is_valid() -> None:
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

    commanded = controller.step(dt=0.02)

    assert commanded
    assert backend.head_kinematics.calls > 1
    assert backend.command is not None


def test_visual_servo_commands_from_3d_look_at_target() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.last_pose: np.ndarray | None = None

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.last_pose = pose
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
    controller.submit_look_at(TrackingLookAtTarget(x=0.5, y=0.0, z=0.0))

    commanded = controller.step(dt=0.02)

    assert commanded
    assert controller.status()["accepted_look_at_targets"] == 1
    assert controller.status()["last_target_type"] == "look_at"
    assert backend.command is not None
    assert backend.head_kinematics.last_pose is not None


def test_visual_servo_3d_look_at_uses_world_up_for_predictable_target_plane() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.last_pose: np.ndarray | None = None

        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.last_pose = pose
            return np.full(7, 0.2)

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
            return np.full(7, 0.2)

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
            return np.full(7, 0.2)

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
            raise AssertionError("servo must not command joints without motion ownership")

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
            return np.full(7, 0.2)

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
    assert record["ik_joints"] == [0.2] * 7
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


def test_visual_servo_records_detection_projection_fallback() -> None:
    class FakeKinematics:
        def __init__(self) -> None:
            self.calls = 0

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            self.calls += 1
            if self.calls == 1:
                return np.full(7, np.nan)
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
    projection = record["projected_target"]
    assert isinstance(projection, dict)
    assert projection["requested_pixel"] == [32.5, 16.25]
    assert projection["projected_pixel"] is not None
    assert projection["scale_from_center"] is not None
    assert projection["ik_attempts"] > 1
    assert projection["ik_failures"] >= 1


def test_visual_servo_run_loop_records_step_error() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.head_kinematics = object()

    controller = VisualServoController(backend=FakeBackend())  # type: ignore[arg-type]

    def fail_step(dt: float | None = None) -> bool:
        controller._stop_event.set()
        raise RuntimeError("boom")

    controller.step = fail_step  # type: ignore[method-assign]
    controller._run_loop()

    record = controller.telemetry.query()["records"][0]
    assert record["reason"] == "step_error"
    assert record["error"] == "boom"
