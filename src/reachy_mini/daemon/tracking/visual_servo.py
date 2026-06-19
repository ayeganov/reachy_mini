"""Robot-side visual servo controller for 2D detections.

This module keeps perception and actuation separated: a remote computer may send
2D detections, but all smoothing, safety projection, and motor target updates
run locally beside the daemon.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from reachy_mini.daemon.backend.abstract import Backend


HEAD_JOINT_LIMITS = np.deg2rad(
    np.array(
        [
            [-160.0, 160.0],  # body yaw
            [-48.0, 80.0],  # stewart_1
            [-80.0, 70.0],  # stewart_2
            [-48.0, 80.0],  # stewart_3
            [-80.0, 48.0],  # stewart_4
            [-70.0, 80.0],  # stewart_5
            [-80.0, 48.0],  # stewart_6
        ],
        dtype=np.float64,
    )
)

T_HEAD_CAM = np.eye(4, dtype=np.float64)
T_HEAD_CAM[:3, 3] = [0.0437, 0.0, 0.0512]
T_HEAD_CAM[:3, :3] = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TrackingDetection:
    """A single 2D detection from an external perception model."""

    u: float
    v: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    frame_id: int | None = None
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class TrackingLookAtTarget:
    """A metric 3D look-at target in the robot/world frame."""

    x: float
    y: float
    z: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    frame_id: int | None = None


@dataclass
class VisualServoConfig:
    """Configuration for robot-side visual servoing."""

    control_frequency: float = 50.0
    min_confidence: float = 0.3
    max_detection_age: float = 0.35
    smoothing_alpha: float = 0.35
    lookahead_distance: float = 0.5
    joint_safety_margin: float = np.deg2rad(5.0)
    max_joint_velocity: float = np.deg2rad(80.0)
    max_joint_acceleration: float = np.deg2rad(300.0)
    max_joint_jerk: float = np.deg2rad(2000.0)
    automatic_body_yaw: bool = True


class DetectionBuffer:
    """Thread-safe latest-only detection buffer."""

    def __init__(self) -> None:
        """Initialize the buffer."""
        self._lock = threading.Lock()
        self._latest: TrackingDetection | None = None
        self.accepted_count = 0

    def submit(self, detection: TrackingDetection) -> None:
        """Replace any pending detection with the newest one."""
        with self._lock:
            self._latest = detection
            self.accepted_count += 1

    def latest(self) -> TrackingDetection | None:
        """Return the most recently submitted detection."""
        with self._lock:
            return self._latest

    def fresh(
        self,
        config: VisualServoConfig,
        now: float | None = None,
    ) -> TrackingDetection | None:
        """Return the latest usable detection, or None if stale/low-confidence."""
        detection = self.latest()
        if detection is None:
            return None

        now = time.time() if now is None else now
        if now - detection.timestamp > config.max_detection_age:
            return None
        if detection.confidence < config.min_confidence:
            return None
        return detection


class LookAtTargetBuffer:
    """Thread-safe latest-only metric look-at target buffer."""

    def __init__(self) -> None:
        """Initialize the buffer."""
        self._lock = threading.Lock()
        self._latest: TrackingLookAtTarget | None = None
        self.accepted_count = 0

    def submit(self, target: TrackingLookAtTarget) -> None:
        """Replace any pending target with the newest one."""
        with self._lock:
            self._latest = target
            self.accepted_count += 1

    def latest(self) -> TrackingLookAtTarget | None:
        """Return the most recently submitted target."""
        with self._lock:
            return self._latest

    def fresh(
        self,
        config: VisualServoConfig,
        now: float | None = None,
    ) -> TrackingLookAtTarget | None:
        """Return the latest usable target, or None if stale/low-confidence."""
        target = self.latest()
        if target is None:
            return None

        now = time.time() if now is None else now
        if now - target.timestamp > config.max_detection_age:
            return None
        if target.confidence < config.min_confidence:
            return None
        return target


class PixelTargetFilter:
    """Exponential smoothing for incoming pixel targets."""

    def __init__(self, alpha: float) -> None:
        """Initialize the filter."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Clear the filter state."""
        self._value = None

    def update(self, detection: TrackingDetection) -> npt.NDArray[np.float64]:
        """Update the filter and return the smoothed pixel position."""
        value = np.array([detection.u, detection.v], dtype=np.float64)
        center = np.array(
            [float(detection.width) / 2.0, float(detection.height) / 2.0],
            dtype=np.float64,
        )
        if self._value is None:
            self._value = self.alpha * value + (1.0 - self.alpha) * center
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value.copy()


class JointCommandLimiter:
    """Clamp and jerk-limit head joint commands before sending them to motors."""

    def __init__(
        self,
        limits: npt.NDArray[np.float64] = HEAD_JOINT_LIMITS,
        config: VisualServoConfig | None = None,
    ) -> None:
        """Initialize the limiter."""
        if len(limits.shape) != 2 or limits.shape[1] != 2:
            raise ValueError("limits must have shape (n, 2)")
        self.limits = limits.astype(np.float64)
        self.config = config or VisualServoConfig()
        self._velocity: npt.NDArray[np.float64] = np.zeros(
            self.limits.shape[0], dtype=np.float64
        )
        self._acceleration: npt.NDArray[np.float64] = np.zeros(
            self.limits.shape[0], dtype=np.float64
        )
        self._last_command: npt.NDArray[np.float64] | None = None

    def reset(self, command: npt.NDArray[np.float64] | None = None) -> None:
        """Reset internal velocity and acceleration state."""
        self._velocity = np.zeros(self.limits.shape[0], dtype=np.float64)
        self._acceleration = np.zeros(self.limits.shape[0], dtype=np.float64)
        self._last_command = command.copy() if command is not None else None

    def clamp(self, command: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Clamp command to configured limits with a safety margin."""
        margin = self.config.joint_safety_margin
        lower = self.limits[:, 0] + margin
        upper = self.limits[:, 1] - margin
        return np.clip(command, lower, upper)

    def limit(
        self,
        desired: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Return a clamped, jerk-limited command."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        desired = self.clamp(desired.astype(np.float64))
        current = current.astype(np.float64)

        if self._last_command is None:
            self._last_command = current.copy()
            self._velocity = np.zeros(self.limits.shape[0], dtype=np.float64)
            self._acceleration = np.zeros(self.limits.shape[0], dtype=np.float64)

        reference = self._last_command
        error = desired - reference
        max_stop_velocity = np.sqrt(
            np.maximum(0.0, 2.0 * self.config.max_joint_acceleration * np.abs(error))
        )
        desired_velocity = np.sign(error) * np.minimum(
            self.config.max_joint_velocity,
            max_stop_velocity,
        )

        desired_acceleration = (desired_velocity - self._velocity) / dt
        acceleration_delta = np.clip(
            desired_acceleration - self._acceleration,
            -self.config.max_joint_jerk * dt,
            self.config.max_joint_jerk * dt,
        )
        acceleration = self._acceleration + acceleration_delta
        acceleration = np.clip(
            acceleration,
            -self.config.max_joint_acceleration,
            self.config.max_joint_acceleration,
        )

        velocity = self._velocity + acceleration * dt
        velocity = np.clip(
            velocity,
            -self.config.max_joint_velocity,
            self.config.max_joint_velocity,
        )

        command = reference + velocity * dt
        crossing_target = np.sign(desired - command) != np.sign(error)
        close_to_target = np.abs(error) < 1e-6
        should_snap = crossing_target | close_to_target
        command = cast(npt.NDArray[np.float64], np.where(should_snap, desired, command))
        velocity = cast(
            npt.NDArray[np.float64],
            np.where(should_snap, 0.0, velocity).astype(np.float64),
        )
        acceleration = cast(
            npt.NDArray[np.float64],
            np.where(should_snap, 0.0, acceleration).astype(np.float64),
        )
        command = self.clamp(command)
        self._last_command = command.copy()
        self._velocity = velocity
        self._acceleration = acceleration
        return command


class VisualServoController:
    """Daemon-local latest-detection visual servo controller."""

    def __init__(
        self,
        backend: "Backend",
        config: VisualServoConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the controller."""
        self.backend = backend
        self.config = config or VisualServoConfig()
        self.buffer = DetectionBuffer()
        self.look_at_buffer = LookAtTargetBuffer()
        self.filter = PixelTargetFilter(self.config.smoothing_alpha)
        self.limiter = JointCommandLimiter(config=self.config)
        self.logger = logger or logging.getLogger(__name__)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_command: npt.NDArray[np.float64] | None = None
        self._last_reason = "not_started"
        self._last_detection_time: float | None = None
        self._last_target_type: str | None = None
        self._command_count = 0
        self._error: str | None = None
        self._look_at_reference_pose: npt.NDArray[np.float64] | None = None
        self._previous_automatic_body_yaw: bool | None = None

    @property
    def running(self) -> bool:
        """Return True if the servo thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def submit(self, detection: TrackingDetection) -> None:
        """Submit a detection from a remote perception model."""
        self.buffer.submit(detection)

    def submit_look_at(self, target: TrackingLookAtTarget) -> None:
        """Submit a metric look-at target from a local control surface."""
        self.look_at_buffer.submit(target)

    def start(self) -> None:
        """Start the local servo loop."""
        if self.running:
            return
        self._stop_event.clear()
        self.filter.reset()
        self.limiter.reset()
        self._look_at_reference_pose = None
        if self.config.automatic_body_yaw and hasattr(
            self.backend.head_kinematics, "set_automatic_body_yaw"
        ):
            previous = getattr(self.backend.head_kinematics, "automatic_body_yaw", None)
            self._previous_automatic_body_yaw = (
                bool(previous) if isinstance(previous, bool) else None
            )
            self.backend.head_kinematics.set_automatic_body_yaw(True)
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reachy-visual-servo",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the local servo loop."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._last_reason = "stop_timeout"
                self._error = "Visual servo thread did not stop within 2.0s."
                return
        self._thread = None
        self._look_at_reference_pose = None
        self._restore_automatic_body_yaw()
        self._last_reason = "stopped"
        self._error = None

    def _restore_automatic_body_yaw(self) -> None:
        """Restore the kinematics yaw mode owned by this controller."""
        if self._previous_automatic_body_yaw is None:
            return
        if hasattr(self.backend.head_kinematics, "set_automatic_body_yaw"):
            self.backend.head_kinematics.set_automatic_body_yaw(
                self._previous_automatic_body_yaw
            )
        self._previous_automatic_body_yaw = None

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable status dictionary."""
        latest = self.buffer.latest()
        latest_look_at = self.look_at_buffer.latest()
        now = time.time()
        return {
            "running": self.running,
            "accepted_detections": self.buffer.accepted_count,
            "accepted_look_at_targets": self.look_at_buffer.accepted_count,
            "command_count": self._command_count,
            "last_reason": self._last_reason,
            "last_target_type": self._last_target_type,
            "last_detection_age": None
            if latest is None
            else max(0.0, now - latest.timestamp),
            "last_look_at_age": None
            if latest_look_at is None
            else max(0.0, now - latest_look_at.timestamp),
            "last_command": None
            if self._last_command is None
            else self._last_command.tolist(),
            "error": self._error,
        }

    def _run_loop(self) -> None:
        period = 1.0 / self.config.control_frequency
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                self.step(period)
            except Exception as exc:
                self._error = str(exc)
                self._last_reason = "error"
                log = logging.getLogger(__name__)
                log.exception("Visual servo step failed")

            next_tick += period
            sleep_time = max(0.0, next_tick - time.monotonic())
            if sleep_time == 0.0:
                next_tick = start
            self._stop_event.wait(sleep_time)

    def step(self, dt: float | None = None) -> bool:
        """Execute one servo step.

        Returns True when a motor target was produced.
        """
        dt = (1.0 / self.config.control_frequency) if dt is None else dt

        look_at = self.look_at_buffer.fresh(self.config)
        detection = None
        if look_at is None:
            self._look_at_reference_pose = None
            detection = self.buffer.fresh(self.config)
            if detection is None:
                self._last_reason = "no_fresh_detection"
                return False

        release_motion_guard = self._try_acquire_motion_guard()
        if release_motion_guard is None:
            self._last_reason = "move_running"
            return False

        try:
            current_joints = np.array(
                self.backend.get_present_head_joint_positions(), dtype=np.float64
            )
            current_pose = np.array(
                self.backend.get_present_head_pose(), dtype=np.float64
            )
            if look_at is not None:
                self._last_detection_time = look_at.timestamp
                self._last_target_type = "look_at"
                if self._look_at_reference_pose is None:
                    self._look_at_reference_pose = current_pose.copy()
                desired_joints = self._ik_from_target_world(
                    target_world=np.array([look_at.x, look_at.y, look_at.z]),
                    current_head_pose=self._look_at_reference_pose,
                    body_yaw=float(current_joints[0]),
                )
            else:
                assert detection is not None
                self._last_detection_time = detection.timestamp
                self._last_target_type = "detection"
                pixel = self.filter.update(detection)
                desired_joints = self._reachable_joints_from_pixel(
                    pixel=pixel,
                    detection=detection,
                    current_head_pose=current_pose,
                    body_yaw=float(current_joints[0]),
                )
            if desired_joints is None:
                self._last_reason = "ik_failed"
                return False

            command = self.limiter.limit(
                desired=np.array(desired_joints, dtype=np.float64),
                current=current_joints,
                dt=dt,
            )
            self.backend.set_target_head_joint_positions(command)
            self._last_command = command.copy()
            self._command_count += 1
            self._last_reason = "commanded"
            return True
        finally:
            release_motion_guard()

    def _try_acquire_motion_guard(self) -> Callable[[], None] | None:
        """Acquire backend motion ownership for one servo step."""
        try_start_move = getattr(self.backend, "_try_start_move", None)
        end_move = getattr(self.backend, "_end_move", None)
        if callable(try_start_move) and callable(end_move):
            if not bool(try_start_move()):
                return None

            def release() -> None:
                end_move()

            return release

        if getattr(self.backend, "is_move_running", False):
            return None

        def noop() -> None:
            return None

        return noop

    def _ik_from_target_world(
        self,
        target_world: npt.NDArray[np.float64],
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        target_pose = self._look_at_pose(
            current_head_pose=current_head_pose,
            target_world=target_world,
            up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        joints = self.backend.head_kinematics.ik(target_pose, body_yaw=body_yaw)
        if joints is None:
            return None
        joints_array = np.array(joints, dtype=np.float64)
        if not self._valid_joints(joints_array):
            return None
        return joints_array

    def _reachable_joints_from_pixel(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        joints = self._ik_from_pixel(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        )
        if self._valid_joints(joints):
            assert joints is not None
            return joints

        center = np.array(
            [float(detection.width) / 2.0, float(detection.height) / 2.0],
            dtype=np.float64,
        )
        center_joints = self._ik_from_pixel(
            pixel=center,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        )
        if not self._valid_joints(center_joints):
            return None

        assert center_joints is not None
        best = center_joints
        low = 0.0
        high = 1.0
        for _ in range(8):
            mid = (low + high) / 2.0
            candidate = center + mid * (pixel - center)
            candidate_joints = self._ik_from_pixel(
                pixel=candidate,
                detection=detection,
                current_head_pose=current_head_pose,
                body_yaw=body_yaw,
            )
            if self._valid_joints(candidate_joints):
                assert candidate_joints is not None
                best = candidate_joints
                low = mid
            else:
                high = mid
        return best

    def _ik_from_pixel(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        target_pose = self._pose_from_pixel(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
        )
        joints = self.backend.head_kinematics.ik(target_pose, body_yaw=body_yaw)
        if joints is None:
            return None
        return np.array(joints, dtype=np.float64)

    @staticmethod
    def _valid_joints(joints: npt.NDArray[np.float64] | None) -> bool:
        return joints is not None and not np.any(np.isnan(joints))

    def _pose_from_pixel(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        ray_cam = self._camera_ray(pixel, detection.width, detection.height)
        T_world_cam = current_head_pose @ T_HEAD_CAM
        ray_world = T_world_cam[:3, :3] @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        camera_origin = T_world_cam[:3, 3]
        target_world = camera_origin + self.config.lookahead_distance * ray_world
        return self._look_at_pose(
            current_head_pose=current_head_pose,
            target_world=target_world,
        )

    @staticmethod
    def _camera_ray(
        pixel: npt.NDArray[np.float64],
        width: int,
        height: int,
    ) -> npt.NDArray[np.float64]:
        fx = float(width)
        fy = float(height)
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        ray = np.array(
            [
                (pixel[0] - cx) / fx,
                (pixel[1] - cy) / fy,
                1.0,
            ],
            dtype=np.float64,
        )
        return ray / np.linalg.norm(ray)

    @staticmethod
    def _look_at_pose(
        current_head_pose: npt.NDArray[np.float64],
        target_world: npt.NDArray[np.float64],
        up_hint: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        origin = current_head_pose[:3, 3]
        forward = target_world - origin
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            return current_head_pose.copy()
        x_axis = forward / forward_norm

        if up_hint is None:
            up_hint = current_head_pose[:3, 2]
        y_axis = np.cross(up_hint, x_axis)
        if np.linalg.norm(y_axis) < 1e-9:
            for fallback_up in (
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ):
                y_axis = np.cross(fallback_up, x_axis)
                if np.linalg.norm(y_axis) >= 1e-9:
                    break
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        pose[:3, 3] = origin

        # Ensure a numerically valid rotation matrix for scipy/IK consumers.
        pose[:3, :3] = R.from_matrix(pose[:3, :3]).as_matrix()
        return pose
