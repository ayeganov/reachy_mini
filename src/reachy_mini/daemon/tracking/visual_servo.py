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

from .telemetry import VisualServoTelemetryBuffer, finite_json_value

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
    image_horizontal_fov: float = np.deg2rad(98.88965079926311)
    image_vertical_fov: float = np.deg2rad(66.67916209122708)
    image_error_elevation_limit: float = np.arctan2(0.2, 0.5)
    image_error_upward_elevation_limit: float | None = None
    image_error_downward_elevation_limit: float | None = None
    joint_safety_margin: float = np.deg2rad(5.0)
    max_joint_velocity: float = np.deg2rad(80.0)
    max_joint_acceleration: float = np.deg2rad(300.0)
    max_joint_jerk: float = np.deg2rad(2000.0)
    look_at_profile_response_hz: float = 1.0
    automatic_body_yaw: bool = True
    telemetry_capacity: int = 3000


@dataclass(frozen=True)
class JointTargetTelemetry:
    """IK and projection metadata for one desired joint target."""

    joints: npt.NDArray[np.float64] | None
    ik_target: npt.NDArray[np.float64] | None
    ik_joints: npt.NDArray[np.float64] | None
    ik_failed: bool
    projected_target: dict[str, Any] | None = None


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


class LookAtTargetFilter:
    """Exponential smoothing for incoming metric look-at targets."""

    def __init__(self, alpha: float) -> None:
        """Initialize the filter."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Clear the filter state."""
        self._value = None

    def update(self, target: TrackingLookAtTarget) -> TrackingLookAtTarget:
        """Update the filter and return the smoothed metric target."""
        value = np.array([target.x, target.y, target.z], dtype=np.float64)
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return TrackingLookAtTarget(
            x=float(self._value[0]),
            y=float(self._value[1]),
            z=float(self._value[2]),
            timestamp=target.timestamp,
            confidence=target.confidence,
            frame_id=target.frame_id,
        )


def _finite_joint_vector(
    value: npt.NDArray[np.float64],
    *,
    length: int,
    name: str,
) -> npt.NDArray[np.float64]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {length}-element vector") from exc
    if vector.shape != (length,) or not bool(np.all(np.isfinite(vector))):
        raise ValueError(f"{name} must be a finite {length}-element vector")
    return vector


class LookAtJointCommandProfile:
    """Generate bounded Stewart-joint motion toward look-at IK targets."""

    def __init__(
        self,
        limits: npt.NDArray[np.float64] = HEAD_JOINT_LIMITS,
        config: VisualServoConfig | None = None,
    ) -> None:
        """Initialize the profile."""
        if limits.shape != (7, 2):
            raise ValueError("limits must have shape (7, 2)")
        self.limits = limits.astype(np.float64)
        self.config = config or VisualServoConfig()
        self._validate_response_frequency()
        self._position: npt.NDArray[np.float64] | None = None
        self._velocity = np.zeros(6, dtype=np.float64)
        self._acceleration = np.zeros(6, dtype=np.float64)
        self._body_yaw_position: float | None = None
        self._body_yaw_velocity = 0.0
        self._body_yaw_acceleration = 0.0

    def _validate_response_frequency(self) -> None:
        response_hz = self.config.look_at_profile_response_hz
        if not np.isfinite(response_hz) or not 0.0 < response_hz <= 5.0:
            raise ValueError("look_at_profile_response_hz must be in (0, 5]")

    def reset(self) -> None:
        """Clear position and motion state without commanding hardware."""
        self._position = None
        self._velocity = np.zeros(6, dtype=np.float64)
        self._acceleration = np.zeros(6, dtype=np.float64)
        self._body_yaw_position = None
        self._body_yaw_velocity = 0.0
        self._body_yaw_acceleration = 0.0

    @staticmethod
    def _advance_motion(
        position: float,
        velocity: float,
        acceleration: float,
        jerk: float,
        duration: float,
    ) -> tuple[float, float, float]:
        return (
            position
            + velocity * duration
            + 0.5 * acceleration * duration**2
            + jerk * duration**3 / 6.0,
            velocity + acceleration * duration + 0.5 * jerk * duration**2,
            acceleration + jerk * duration,
        )

    @classmethod
    def _stopping_distance(
        cls,
        velocity: float,
        acceleration: float,
        max_jerk: float,
        max_acceleration: float,
    ) -> float:
        """Estimate forward distance needed for a jerk-bounded stop."""
        if velocity <= 0.0:
            return 0.0

        peak_acceleration = np.sqrt(
            max(0.0, acceleration**2 / 2.0 + max_jerk * velocity)
        )
        position = 0.0
        if peak_acceleration <= max_acceleration:
            ramp_down = max(0.0, (acceleration + peak_acceleration) / max_jerk)
            position, velocity, acceleration = cls._advance_motion(
                position,
                velocity,
                acceleration,
                -max_jerk,
                ramp_down,
            )
            position, _velocity, _acceleration = cls._advance_motion(
                position,
                velocity,
                acceleration,
                max_jerk,
                peak_acceleration / max_jerk,
            )
            return max(0.0, position)

        ramp_down = max(0.0, (acceleration + max_acceleration) / max_jerk)
        position, velocity, acceleration = cls._advance_motion(
            position,
            velocity,
            acceleration,
            -max_jerk,
            ramp_down,
        )
        hold = max(
            0.0,
            velocity / max_acceleration - max_acceleration / (2.0 * max_jerk),
        )
        position, velocity, acceleration = cls._advance_motion(
            position,
            velocity,
            acceleration,
            0.0,
            hold,
        )
        position, _velocity, _acceleration = cls._advance_motion(
            position,
            velocity,
            acceleration,
            max_jerk,
            max_acceleration / max_jerk,
        )
        return max(0.0, position)

    def update_with_telemetry(
        self,
        desired: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], list[dict[str, float | int | str]]]:
        """Advance the profile and return its command and limit hits."""
        self._validate_response_frequency()
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        desired_vector = _finite_joint_vector(desired, length=7, name="desired")
        current_vector = _finite_joint_vector(current, length=7, name="current")

        stewart_position = (
            current_vector[1:7].copy()
            if self._position is None
            else self._position.copy()
        )
        position = np.concatenate(
            (
                np.array(
                    [
                        current_vector[0]
                        if self._body_yaw_position is None
                        else self._body_yaw_position
                    ]
                ),
                stewart_position,
            )
        )
        velocity = np.concatenate(
            (np.array([self._body_yaw_velocity]), self._velocity.copy())
        )
        acceleration = np.concatenate(
            (np.array([self._body_yaw_acceleration]), self._acceleration.copy())
        )
        lower = self.limits[:, 0] + self.config.joint_safety_margin
        upper = self.limits[:, 1] - self.config.joint_safety_margin
        hits: list[dict[str, float | int | str]] = []

        for index, (value, low, high) in enumerate(zip(desired_vector, lower, upper)):
            if value < low:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "lower_position",
                        "source": "desired",
                        "value": float(value),
                        "limit": float(low),
                    }
                )
            if value > high:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "upper_position",
                        "source": "desired",
                        "value": float(value),
                        "limit": float(high),
                    }
                )

        target = np.clip(desired_vector, lower, upper)
        omega = 2.0 * np.pi * self.config.look_at_profile_response_hz
        next_position = position.copy()
        next_velocity = velocity.copy()
        next_acceleration = acceleration.copy()
        max_velocity = self.config.max_joint_velocity
        max_acceleration = self.config.max_joint_acceleration
        max_jerk = self.config.max_joint_jerk

        for local_index in range(7):
            error = float(target[local_index] - position[local_index])
            direction = 1.0 if error >= 0.0 else -1.0
            distance = abs(error)
            directed_velocity = direction * float(velocity[local_index])
            directed_acceleration = direction * float(acceleration[local_index])

            if (
                distance <= 1e-12
                and abs(directed_velocity) <= 1e-10
                and abs(directed_acceleration) <= 1e-8
            ):
                next_position[local_index] = target[local_index]
                next_velocity[local_index] = 0.0
                next_acceleration[local_index] = 0.0
                continue

            requested_directed_acceleration = (
                omega * omega * distance - omega * directed_velocity
            )
            requested_acceleration = direction * requested_directed_acceleration
            requested_delta = requested_acceleration - acceleration[local_index]
            requested_velocity = velocity[local_index] + requested_acceleration * dt
            joint_index = local_index
            if abs(float(requested_delta / dt)) > max_jerk:
                hits.append(
                    {
                        "joint_index": joint_index,
                        "kind": "jerk",
                        "source": "profile_jerk",
                        "value": float(requested_delta / dt),
                        "limit": float(max_jerk),
                    }
                )
            if abs(float(requested_acceleration)) > max_acceleration:
                hits.append(
                    {
                        "joint_index": joint_index,
                        "kind": "acceleration",
                        "source": "profile_acceleration",
                        "value": float(requested_acceleration),
                        "limit": float(max_acceleration),
                    }
                )
            if abs(float(requested_velocity)) > max_velocity:
                hits.append(
                    {
                        "joint_index": joint_index,
                        "kind": "velocity",
                        "source": "profile_velocity",
                        "value": float(requested_velocity),
                        "limit": float(max_velocity),
                    }
                )

            acceleration_low = max(
                -max_acceleration, directed_acceleration - max_jerk * dt
            )
            acceleration_high = min(
                max_acceleration, directed_acceleration + max_jerk * dt
            )
            candidate = float(
                np.clip(
                    requested_directed_acceleration,
                    acceleration_low,
                    acceleration_high,
                )
            )

            def speed_at_rest(acceleration_value: float) -> float:
                velocity_after_step = directed_velocity + acceleration_value * dt
                return velocity_after_step + max(0.0, acceleration_value) ** 2 / (
                    2.0 * max_jerk
                )

            if speed_at_rest(candidate) > max_velocity:
                speed_low = acceleration_low
                speed_high = candidate
                if speed_at_rest(speed_low) >= max_velocity:
                    candidate = speed_low
                else:
                    for _ in range(30):
                        midpoint = (speed_low + speed_high) / 2.0
                        if speed_at_rest(midpoint) > max_velocity:
                            speed_high = midpoint
                        else:
                            speed_low = midpoint
                    candidate = (speed_low + speed_high) / 2.0

            def constrain_for_stop(
                acceleration_value: float,
                motion_velocity: float,
                motion_acceleration: float,
                available_distance: float,
            ) -> float:
                braking = max(-max_acceleration, motion_acceleration - max_jerk * dt)

                def distance_after_step(value: float) -> float:
                    velocity_after_step = max(0.0, motion_velocity + value * dt)
                    return velocity_after_step * dt + self._stopping_distance(
                        velocity_after_step,
                        value,
                        max_jerk,
                        max_acceleration,
                    )

                if distance_after_step(braking) >= available_distance:
                    return braking
                if distance_after_step(acceleration_value) <= available_distance:
                    return acceleration_value
                safe = braking
                unsafe = acceleration_value
                for _ in range(30):
                    midpoint = (safe + unsafe) / 2.0
                    if distance_after_step(midpoint) > available_distance:
                        unsafe = midpoint
                    else:
                        safe = midpoint
                return (safe + unsafe) / 2.0

            chosen_acceleration = constrain_for_stop(
                candidate,
                directed_velocity,
                directed_acceleration,
                distance,
            )
            chosen_acceleration *= direction
            motion_direction = float(
                np.sign(velocity[local_index] + chosen_acceleration * dt)
            )
            if motion_direction == 0.0:
                motion_direction = float(np.sign(velocity[local_index]))
            if motion_direction != 0.0:
                boundary_distance = (
                    upper[local_index] - position[local_index]
                    if motion_direction > 0.0
                    else position[local_index] - lower[local_index]
                )
                chosen_acceleration = motion_direction * constrain_for_stop(
                    motion_direction * chosen_acceleration,
                    motion_direction * velocity[local_index],
                    motion_direction * acceleration[local_index],
                    max(0.0, boundary_distance),
                )

            next_acceleration[local_index] = chosen_acceleration
            next_velocity[local_index] = (
                velocity[local_index] + next_acceleration[local_index] * dt
            )
            next_position[local_index] = (
                position[local_index] + next_velocity[local_index] * dt
            )

        velocity = next_velocity
        acceleration = next_acceleration

        for index, (value, low, high) in enumerate(zip(next_position, lower, upper)):
            if value < low:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "lower_position",
                        "source": "profile_position",
                        "value": float(value),
                        "limit": float(low),
                    }
                )
            if value > high:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "upper_position",
                        "source": "profile_position",
                        "value": float(value),
                        "limit": float(high),
                    }
                )
        position = next_position

        self._body_yaw_position = float(position[0])
        self._body_yaw_velocity = float(velocity[0])
        self._body_yaw_acceleration = float(acceleration[0])
        self._position = position[1:7].copy()
        self._velocity = velocity[1:7].copy()
        self._acceleration = acceleration[1:7].copy()
        return position, hits


class JointCommandSafetyGuard:
    """Reject unsafe profiled commands without generating another trajectory."""

    def __init__(
        self,
        limits: npt.NDArray[np.float64] = HEAD_JOINT_LIMITS,
        config: VisualServoConfig | None = None,
    ) -> None:
        """Initialize the guard."""
        if limits.shape != (7, 2):
            raise ValueError("limits must have shape (7, 2)")
        self.limits = limits.astype(np.float64)
        self.config = config or VisualServoConfig()
        self._last_command: npt.NDArray[np.float64] | None = None
        self._velocity: npt.NDArray[np.float64] = np.zeros(7, dtype=np.float64)
        self._acceleration: npt.NDArray[np.float64] = np.zeros(7, dtype=np.float64)

    def reset(self, command: npt.NDArray[np.float64] | None = None) -> None:
        """Clear derivative state, optionally seeding the command position."""
        self._last_command = command.copy() if command is not None else None
        self._velocity = np.zeros(7, dtype=np.float64)
        self._acceleration = np.zeros(7, dtype=np.float64)

    def check(
        self,
        command: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[
        list[dict[str, float | int | str]],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return violations and derivative state without changing the command."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        command_vector = _finite_joint_vector(command, length=7, name="command")
        current_vector = _finite_joint_vector(current, length=7, name="current")
        reference = current_vector if self._last_command is None else self._last_command
        velocity = (command_vector - reference) / dt
        acceleration = (velocity - self._velocity) / dt
        jerk = (acceleration - self._acceleration) / dt
        margin = self.config.joint_safety_margin
        lower = self.limits[:, 0] + margin
        upper = self.limits[:, 1] - margin
        position_tolerance = 1e-9
        hits: list[dict[str, float | int | str]] = []

        for index, value in enumerate(command_vector):
            if value < lower[index] - position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "lower_position",
                        "value": float(value),
                        "limit": float(lower[index]),
                    }
                )
            if value > upper[index] + position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "upper_position",
                        "value": float(value),
                        "limit": float(upper[index]),
                    }
                )

        for kind, values, limit in (
            ("velocity", velocity, self.config.max_joint_velocity),
            ("acceleration", acceleration, self.config.max_joint_acceleration),
            ("jerk", jerk, self.config.max_joint_jerk),
        ):
            tolerance = max(1e-9, limit * 1e-6)
            for index, value in enumerate(values):
                if abs(float(value)) > limit + tolerance:
                    hits.append(
                        {
                            "joint_index": index,
                            "kind": kind,
                            "value": float(value),
                            "limit": float(limit),
                        }
                    )
        return hits, velocity, acceleration

    def commit(
        self,
        command: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        acceleration: npt.NDArray[np.float64],
    ) -> None:
        """Retain a command after its backend write succeeds."""
        self._last_command = command.copy()
        self._velocity = velocity.copy()
        self._acceleration = acceleration.copy()


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
        command, _telemetry = self.limit_with_telemetry(desired, current, dt)
        return command

    def limit_with_telemetry(
        self,
        desired: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Return a clamped, jerk-limited command and telemetry metadata."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        desired = _finite_joint_vector(
            desired, length=self.limits.shape[0], name="desired"
        )
        current = _finite_joint_vector(
            current, length=self.limits.shape[0], name="current"
        )

        margin = self.config.joint_safety_margin
        lower = self.limits[:, 0] + margin
        upper = self.limits[:, 1] - margin
        clamped_desired = np.clip(desired, lower, upper)
        hits: list[dict[str, float | int | str]] = []
        for index, (raw, low, high) in enumerate(zip(desired, lower, upper)):
            if raw < low:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "lower_position",
                        "value": float(raw),
                        "limit": float(low),
                    }
                )
            if raw > high:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "upper_position",
                        "value": float(raw),
                        "limit": float(high),
                    }
                )
        if self._last_command is None:
            self._last_command = current.copy()
            self._velocity = np.zeros(self.limits.shape[0], dtype=np.float64)
            self._acceleration = np.zeros(self.limits.shape[0], dtype=np.float64)

        reference = self._last_command
        error = clamped_desired - reference
        max_stop_velocity = np.sqrt(
            np.maximum(0.0, 2.0 * self.config.max_joint_acceleration * np.abs(error))
        )
        requested_velocity = np.sign(error) * max_stop_velocity
        desired_velocity = np.sign(error) * np.minimum(
            self.config.max_joint_velocity,
            max_stop_velocity,
        )
        for index, velocity in enumerate(requested_velocity):
            if abs(float(velocity)) > self.config.max_joint_velocity:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "velocity",
                        "value": float(velocity),
                        "limit": float(self.config.max_joint_velocity),
                    }
                )

        desired_acceleration = (desired_velocity - self._velocity) / dt
        requested_acceleration_delta = desired_acceleration - self._acceleration
        acceleration_delta = np.clip(
            requested_acceleration_delta,
            -self.config.max_joint_jerk * dt,
            self.config.max_joint_jerk * dt,
        )
        for index, delta in enumerate(requested_acceleration_delta):
            if abs(float(delta)) > self.config.max_joint_jerk * dt:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "jerk",
                        "value": float(delta / dt),
                        "limit": float(self.config.max_joint_jerk),
                    }
                )
        requested_acceleration = self._acceleration + acceleration_delta
        for index, accel in enumerate(requested_acceleration):
            if abs(float(accel)) > self.config.max_joint_acceleration:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "acceleration",
                        "value": float(accel),
                        "limit": float(self.config.max_joint_acceleration),
                    }
                )
        acceleration = requested_acceleration
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
        crossing_target = np.sign(clamped_desired - command) != np.sign(error)
        close_to_target = np.abs(error) < 1e-6
        should_snap = crossing_target | close_to_target
        command = cast(
            npt.NDArray[np.float64],
            np.where(should_snap, clamped_desired, command),
        )
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
        return command, {
            "clamped_desired": clamped_desired.tolist(),
            "limit_hits": hits,
        }


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
        self.look_at_filter = LookAtTargetFilter(self.config.smoothing_alpha)
        self.look_at_profile = LookAtJointCommandProfile(config=self.config)
        self.look_at_guard = JointCommandSafetyGuard(config=self.config)
        self.limiter = JointCommandLimiter(config=self.config)
        self.telemetry = VisualServoTelemetryBuffer(
            capacity=self.config.telemetry_capacity
        )
        self._telemetry_sequence = 0
        self.logger = logger or logging.getLogger(__name__)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_command: npt.NDArray[np.float64] | None = None
        self._last_reason = "not_started"
        self._last_detection_time: float | None = None
        self._last_target_type: str | None = None
        self._last_command_path: str | None = None
        self._command_count = 0
        self._error: str | None = None
        self._look_at_reference_pose: npt.NDArray[np.float64] | None = None
        self._detection_reference_pose: npt.NDArray[np.float64] | None = None
        self._processed_detection: TrackingDetection | None = None
        self._detection_target_cache: TrackingLookAtTarget | None = None
        self._previous_automatic_body_yaw: bool | None = None
        self._last_command_time: float | None = None

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
        self.look_at_filter.reset()
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self.limiter.reset()
        self._last_command_path = None
        self._look_at_reference_pose = None
        self._detection_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self._last_command_time = None
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
        self._detection_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self.look_at_filter.reset()
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self.limiter.reset()
        self._last_command_path = None
        self._last_command_time = None
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

    def _new_telemetry_record(
        self,
        *,
        dt: float,
        target_type: str,
        reason: str,
        input_target: dict[str, Any] | None = None,
        start_monotonic: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        monotonic_now = time.monotonic()
        record = {
            "sequence": self._telemetry_sequence,
            "timestamp": now,
            "monotonic_timestamp": monotonic_now,
            "dt": dt,
            "target_type": target_type,
            "input_target": input_target,
            "smoothed_target": None,
            "current_joints": None,
            "current_pose": None,
            "ik_target": None,
            "ik_joints": None,
            "projected_target": None,
            "profiled_command": None,
            "profile_limit_hits": [],
            "final_command": None,
            "actual_joints": None,
            "actual_joints_source": "none",
            "reason": reason,
            "ik_failed": False,
            "limit_hits": [],
            "body_yaw": {
                "current": None,
                "ik_input": None,
                "automatic_enabled": self._automatic_body_yaw_state(),
            },
            "latency": {
                "target_age": None,
                "processing_duration": None
                if start_monotonic is None
                else monotonic_now - start_monotonic,
            },
            "backend": self._backend_telemetry(),
            "backend_extra": {},
            "error": None,
        }
        self._telemetry_sequence += 1
        return record

    def _append_telemetry(self, record: dict[str, Any]) -> None:
        self.telemetry.append(record)

    def _backend_telemetry(self) -> dict[str, Any]:
        ready = getattr(self.backend, "ready", None)
        is_set = getattr(ready, "is_set", None)
        return {
            "ready": bool(is_set()) if callable(is_set) else None,
            "error": getattr(self.backend, "error", None),
            "motor_control_mode": getattr(self.backend, "motor_control_mode", None),
        }

    def _automatic_body_yaw_state(self) -> bool | None:
        value = getattr(self.backend.head_kinematics, "automatic_body_yaw", None)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _detection_target(detection: TrackingDetection) -> dict[str, Any]:
        return {
            "kind": "detection",
            "u": detection.u,
            "v": detection.v,
            "timestamp": detection.timestamp,
            "confidence": detection.confidence,
            "frame_id": detection.frame_id,
            "width": detection.width,
            "height": detection.height,
        }

    @staticmethod
    def _look_at_target(target: TrackingLookAtTarget) -> dict[str, Any]:
        return {
            "kind": "look_at",
            "x": target.x,
            "y": target.y,
            "z": target.z,
            "timestamp": target.timestamp,
            "confidence": target.confidence,
            "frame_id": target.frame_id,
        }

    def _run_loop(self) -> None:
        period = 1.0 / self.config.control_frequency
        previous_start: float | None = None
        while not self._stop_event.is_set():
            start = time.monotonic()
            dt = period if previous_start is None else start - previous_start
            previous_start = start
            try:
                self.step(dt, use_command_elapsed=True)
            except Exception as exc:
                self._error = str(exc)
                self._last_reason = "error"
                record = self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="step_error",
                    start_monotonic=start,
                )
                record["error"] = str(exc)
                self._append_telemetry(record)
                log = logging.getLogger(__name__)
                log.exception("Visual servo step failed")

            sleep_time = max(0.0, period - (time.monotonic() - start))
            self._stop_event.wait(sleep_time)

    def step(
        self,
        dt: float | None = None,
        *,
        use_command_elapsed: bool = False,
    ) -> bool:
        """Execute one servo step.

        Returns True when a motor target was produced.
        """
        dt = (1.0 / self.config.control_frequency) if dt is None else dt
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        start_monotonic = time.monotonic()
        if dt > 2.0 / self.config.control_frequency:
            self._reset_motion_state()
            self._last_reason = "control_stall"
            self._append_telemetry(
                self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="control_stall",
                    start_monotonic=start_monotonic,
                )
            )
            return False

        look_at = self.look_at_buffer.fresh(self.config)
        detection = None if look_at is not None else self.buffer.fresh(self.config)
        if look_at is None and detection is None:
            self._reset_motion_state()
            self._last_reason = "no_fresh_detection"
            self._append_telemetry(
                self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="no_fresh_detection",
                    start_monotonic=start_monotonic,
                )
            )
            return False

        target_type = "none"
        input_target: dict[str, Any] | None = None
        target_timestamp: float | None = None
        if look_at is not None:
            target_type = "look_at"
            input_target = self._look_at_target(look_at)
            target_timestamp = look_at.timestamp
        elif detection is not None:
            target_type = "detection"
            input_target = self._detection_target(detection)
            target_timestamp = detection.timestamp

        release_motion_guard = self._try_acquire_motion_guard()
        if release_motion_guard is None:
            self._last_reason = "move_running"
            self._append_telemetry(
                self._new_telemetry_record(
                    dt=dt,
                    target_type=target_type,
                    reason="move_running",
                    input_target=input_target,
                    start_monotonic=start_monotonic,
                )
            )
            return False

        try:
            try:
                current_joints = _finite_joint_vector(
                    self.backend.get_present_head_joint_positions(),
                    length=7,
                    name="current",
                )
            except ValueError:
                self._reset_motion_state()
                raise
            current_pose = np.array(
                self.backend.get_present_head_pose(), dtype=np.float64
            )
            if target_timestamp is not None:
                self._last_detection_time = target_timestamp
            if target_type != "none":
                self._last_target_type = target_type
            record = self._new_telemetry_record(
                dt=dt,
                target_type=target_type,
                reason="ik_failed",
                input_target=input_target,
                start_monotonic=start_monotonic,
            )
            body_yaw = float(current_joints[0])
            if self._last_command_path != target_type:
                self.look_at_filter.reset()
                self._look_at_reference_pose = current_pose.copy()
            if look_at is not None:
                self._processed_detection = None
                self._detection_target_cache = None
                self._detection_reference_pose = None
                target = look_at
            else:
                assert detection is not None
                if self._detection_reference_pose is None:
                    self._detection_reference_pose = current_pose.copy()
                target = self._look_at_target_from_detection(
                    detection,
                    current_pose,
                    self._detection_reference_pose,
                )
            if self._look_at_reference_pose is None:
                self._look_at_reference_pose = current_pose.copy()
            smoothed_look_at = self.look_at_filter.update(target)
            ik_reference_pose = (
                self._look_at_reference_pose
                if look_at is not None
                else self._detection_reference_pose
            )
            assert ik_reference_pose is not None
            target_result = self._ik_from_target_world_with_telemetry(
                target_world=np.array(
                    [smoothed_look_at.x, smoothed_look_at.y, smoothed_look_at.z]
                ),
                current_head_pose=ik_reference_pose,
                body_yaw=body_yaw,
            )

            record["smoothed_target"] = self._look_at_target(smoothed_look_at)
            record["current_joints"] = finite_json_value(current_joints)
            record["current_pose"] = finite_json_value(current_pose)
            record["ik_target"] = finite_json_value(target_result.ik_target)
            record["ik_joints"] = finite_json_value(target_result.ik_joints)
            record["ik_failed"] = target_result.ik_failed
            record["projected_target"] = finite_json_value(
                target_result.projected_target
            )
            record["body_yaw"]["current"] = body_yaw
            record["body_yaw"]["ik_input"] = body_yaw
            record["latency"]["target_age"] = (
                None
                if target_timestamp is None
                else max(0.0, time.time() - target_timestamp)
            )
            if target_result.joints is None:
                self._reset_motion_state()
                record["reason"] = "ik_failed"
                self._last_reason = "ik_failed"
                record["latency"]["processing_duration"] = (
                    time.monotonic() - start_monotonic
                )
                self._append_telemetry(record)
                return False

            desired_joints = np.array(target_result.joints, dtype=np.float64)
            command_time = time.monotonic()
            command_dt = dt
            if use_command_elapsed and self._last_command_time is not None:
                command_dt = command_time - self._last_command_time
            if command_dt > 2.0 / self.config.control_frequency:
                self._reset_motion_state()
                record["dt"] = command_dt
                record["monotonic_timestamp"] = command_time
                record["reason"] = "control_stall"
                record["latency"]["processing_duration"] = (
                    time.monotonic() - start_monotonic
                )
                self._last_reason = "control_stall"
                self._append_telemetry(record)
                return False
            record["dt"] = command_dt
            record["monotonic_timestamp"] = command_time
            desired_joints, profile_hits = self.look_at_profile.update_with_telemetry(
                desired=desired_joints,
                current=current_joints,
                dt=command_dt,
            )
            record["profiled_command"] = finite_json_value(desired_joints)
            record["profile_limit_hits"] = profile_hits
            guard_hits, guard_velocity, guard_acceleration = self.look_at_guard.check(
                command=desired_joints,
                current=current_joints,
                dt=command_dt,
            )
            record["limit_hits"] = guard_hits
            if guard_hits:
                self._reset_motion_state()
                record["reason"] = "safety_rejected"
                record["latency"]["processing_duration"] = (
                    time.monotonic() - start_monotonic
                )
                self._last_reason = "safety_rejected"
                self._append_telemetry(record)
                return False
            command = desired_joints
            try:
                self.backend.set_target_head_joint_positions(command)
            except Exception:
                self._reset_motion_state()
                raise
            self.look_at_guard.commit(
                command,
                guard_velocity,
                guard_acceleration,
            )
            self._last_command_path = target_type
            self._last_command_time = command_time
            self._last_command = command.copy()
            self._command_count += 1
            self._last_reason = "commanded"
            record["final_command"] = finite_json_value(command)
            record["actual_joints"] = finite_json_value(current_joints)
            record["actual_joints_source"] = "present_read_before_command"
            record["reason"] = "commanded"
            record["latency"]["processing_duration"] = (
                time.monotonic() - start_monotonic
            )
            self._append_telemetry(record)
            return True
        finally:
            release_motion_guard()

    def _reset_motion_state(self) -> None:
        """Reset controller-owned motion state without backend I/O."""
        self.filter.reset()
        self.look_at_filter.reset()
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self.limiter.reset()
        self._look_at_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self._last_command_path = None
        self._last_command_time = None

    def _look_at_target_from_detection(
        self,
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        reference_head_pose: npt.NDArray[np.float64],
    ) -> TrackingLookAtTarget:
        """Convert one image error and the measured gaze into a look-at point."""
        if detection is self._processed_detection:
            assert self._detection_target_cache is not None
            return self._detection_target_cache
        pixel = self.filter.update(detection)
        center = np.array(
            [float(detection.width) / 2.0, float(detection.height) / 2.0],
            dtype=np.float64,
        )
        error = (pixel - center) / center
        if bool(np.all(np.abs(error) <= 0.03)):
            error[:] = 0.0
        ray_camera = np.array(
            [
                error[0] * np.tan(self.config.image_horizontal_fov / 2.0),
                error[1] * np.tan(self.config.image_vertical_fov / 2.0),
                1.0,
            ],
            dtype=np.float64,
        )
        ray_world = (current_head_pose @ T_HEAD_CAM)[:3, :3] @ ray_camera
        azimuth = float(np.arctan2(ray_world[1], ray_world[0]))
        upward_limit = self.config.image_error_upward_elevation_limit
        if upward_limit is None:
            upward_limit = self.config.image_error_elevation_limit
        downward_limit = self.config.image_error_downward_elevation_limit
        if downward_limit is None:
            downward_limit = self.config.image_error_elevation_limit
        elevation = float(
            np.clip(
                np.arctan2(ray_world[2], np.hypot(ray_world[0], ray_world[1])),
                -downward_limit,
                upward_limit,
            )
        )
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        target_world = reference_head_pose[:3, 3] + (
            self.config.lookahead_distance * direction
        )
        target = TrackingLookAtTarget(
            x=float(target_world[0]),
            y=float(target_world[1]),
            z=float(target_world[2]),
            timestamp=detection.timestamp,
            confidence=detection.confidence,
            frame_id=detection.frame_id,
        )
        self._processed_detection = detection
        self._detection_target_cache = target
        return target

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

    def _ik_from_target_world_with_telemetry(
        self,
        target_world: npt.NDArray[np.float64],
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> JointTargetTelemetry:
        target_pose = self._look_at_pose(
            current_head_pose=current_head_pose,
            target_world=target_world,
            up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        joints = self.backend.head_kinematics.ik(target_pose, body_yaw=body_yaw)
        if joints is None:
            return JointTargetTelemetry(
                joints=None,
                ik_target=target_pose,
                ik_joints=None,
                ik_failed=True,
            )
        joints_array = np.array(joints, dtype=np.float64)
        failed = not self._valid_joints(joints_array)
        return JointTargetTelemetry(
            joints=None if failed else joints_array,
            ik_target=target_pose,
            ik_joints=joints_array,
            ik_failed=failed,
        )

    def _ik_from_target_world(
        self,
        target_world: npt.NDArray[np.float64],
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        return self._ik_from_target_world_with_telemetry(
            target_world=target_world,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        ).joints

    def _reachable_joints_from_pixel_with_telemetry(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> JointTargetTelemetry:
        attempts = 0
        failures = 0
        requested_pixel = pixel.copy()
        initial = self._ik_from_pixel_with_telemetry(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        )
        attempts += 1
        failures += int(initial.ik_failed)
        if initial.joints is not None:
            return JointTargetTelemetry(
                joints=initial.joints,
                ik_target=initial.ik_target,
                ik_joints=initial.ik_joints,
                ik_failed=False,
                projected_target={
                    "requested_pixel": requested_pixel.tolist(),
                    "projected_pixel": pixel.tolist(),
                    "scale_from_center": 1.0,
                    "ik_attempts": attempts,
                    "ik_failures": failures,
                },
            )

        center = np.array(
            [float(detection.width) / 2.0, float(detection.height) / 2.0],
            dtype=np.float64,
        )
        center_result = self._ik_from_pixel_with_telemetry(
            pixel=center,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        )
        attempts += 1
        failures += int(center_result.ik_failed)
        if center_result.joints is None:
            return JointTargetTelemetry(
                joints=None,
                ik_target=center_result.ik_target,
                ik_joints=center_result.ik_joints,
                ik_failed=True,
                projected_target={
                    "requested_pixel": requested_pixel.tolist(),
                    "projected_pixel": None,
                    "scale_from_center": None,
                    "ik_attempts": attempts,
                    "ik_failures": failures,
                },
            )

        best = center_result
        best_pixel = center.copy()
        best_scale = 0.0
        low = 0.0
        high = 1.0
        for _ in range(8):
            mid = (low + high) / 2.0
            candidate = center + mid * (pixel - center)
            candidate_result = self._ik_from_pixel_with_telemetry(
                pixel=candidate,
                detection=detection,
                current_head_pose=current_head_pose,
                body_yaw=body_yaw,
            )
            attempts += 1
            failures += int(candidate_result.ik_failed)
            if candidate_result.joints is not None:
                best = candidate_result
                best_pixel = candidate
                best_scale = mid
                low = mid
            else:
                high = mid
        return JointTargetTelemetry(
            joints=best.joints,
            ik_target=best.ik_target,
            ik_joints=best.ik_joints,
            ik_failed=best.ik_failed,
            projected_target={
                "requested_pixel": requested_pixel.tolist(),
                "projected_pixel": best_pixel.tolist(),
                "scale_from_center": best_scale,
                "ik_attempts": attempts,
                "ik_failures": failures,
            },
        )

    def _reachable_joints_from_pixel(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        return self._reachable_joints_from_pixel_with_telemetry(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        ).joints

    def _ik_from_pixel_with_telemetry(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> JointTargetTelemetry:
        target_pose = self._pose_from_pixel(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
        )
        joints = self.backend.head_kinematics.ik(target_pose, body_yaw=body_yaw)
        if joints is None:
            return JointTargetTelemetry(
                joints=None,
                ik_target=target_pose,
                ik_joints=None,
                ik_failed=True,
            )
        joints_array = np.array(joints, dtype=np.float64)
        failed = not self._valid_joints(joints_array)
        return JointTargetTelemetry(
            joints=None if failed else joints_array,
            ik_target=target_pose,
            ik_joints=joints_array,
            ik_failed=failed,
        )

    def _ik_from_pixel(
        self,
        pixel: npt.NDArray[np.float64],
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> npt.NDArray[np.float64] | None:
        return self._ik_from_pixel_with_telemetry(
            pixel=pixel,
            detection=detection,
            current_head_pose=current_head_pose,
            body_yaw=body_yaw,
        ).joints

    @staticmethod
    def _valid_joints(joints: npt.NDArray[np.float64] | None) -> bool:
        return (
            joints is not None
            and joints.shape == (7,)
            and bool(np.all(np.isfinite(joints)))
        )

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
