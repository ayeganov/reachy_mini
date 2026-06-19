import time

import numpy as np
from scipy.spatial.transform import Rotation as R

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
