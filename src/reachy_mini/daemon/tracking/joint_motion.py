"""Bounded joint trajectories and command safety for visual tracking."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import VisualServoConfig

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
    """Generate bounded head-joint motion toward look-at IK targets."""

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
        self._velocity = np.zeros(7, dtype=np.float64)
        self._acceleration = np.zeros(7, dtype=np.float64)

    def _validate_response_frequency(self) -> None:
        response_hz = self.config.look_at_profile_response_hz
        if not np.isfinite(response_hz) or not 0.0 < response_hz <= 5.0:
            raise ValueError("look_at_profile_response_hz must be in (0, 5]")

    def reset(self) -> None:
        """Clear position and motion state without commanding hardware."""
        self._position = None
        self._velocity = np.zeros(7, dtype=np.float64)
        self._acceleration = np.zeros(7, dtype=np.float64)

    @property
    def stationary(self) -> bool:
        """Return whether the retained command trajectory is at rest."""
        return bool(
            np.max(np.abs(self._velocity), initial=0.0) <= 1e-10
            and np.max(np.abs(self._acceleration), initial=0.0) <= 1e-8
        )

    def hold(self, command: npt.NDArray[np.float64]) -> None:
        """Retain one command as stationary trajectory state."""
        command_vector = _finite_joint_vector(command, length=7, name="command")
        self._position = command_vector.copy()
        self._velocity.fill(0.0)
        self._acceleration.fill(0.0)

    def stop_with_telemetry(
        self,
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], list[dict[str, float | int | str]]]:
        """Advance the retained trajectory toward a bounded stationary hold."""
        current_vector = _finite_joint_vector(current, length=7, name="current")
        if self._position is None:
            return current_vector, []
        position = self._position.copy()
        velocity = self._velocity.copy()
        acceleration = self._acceleration.copy()
        target = position.copy()
        for index, (speed, rate) in enumerate(zip(velocity, acceleration)):
            direction = float(np.sign(speed if abs(speed) > 1e-10 else rate))
            if direction == 0.0:
                continue
            distance = self._stopping_distance(
                direction * float(speed),
                direction * float(rate),
                self.config.max_joint_jerk,
                self.config.max_joint_acceleration,
            )
            target[index] += direction * max(distance, 1e-12)
        return self.update_with_telemetry(target, current_vector, dt)

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

        position = (
            current_vector.copy() if self._position is None else self._position.copy()
        )
        velocity = self._velocity.copy()
        acceleration = self._acceleration.copy()
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

        self._position = position.copy()
        self._velocity = velocity.copy()
        self._acceleration = acceleration.copy()
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

    def check_with_telemetry(
        self,
        command: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[
        list[dict[str, float | int | str]],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[dict[str, float | int | str]],
    ]:
        """Return violations, derivative state, and inward recovery progress."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        command_vector = _finite_joint_vector(command, length=7, name="command")
        current_vector = _finite_joint_vector(current, length=7, name="current")
        reference = current_vector if self._last_command is None else self._last_command
        velocity = (command_vector - reference) / dt
        acceleration = (velocity - self._velocity) / dt
        jerk = (acceleration - self._acceleration) / dt
        margin = self.config.joint_safety_margin
        hard_lower = self.limits[:, 0]
        hard_upper = self.limits[:, 1]
        soft_lower = hard_lower + margin
        soft_upper = hard_upper - margin
        position_tolerance = 1e-9
        hits: list[dict[str, float | int | str]] = []
        recovery: list[dict[str, float | int | str]] = []

        reference_violation = np.maximum(soft_lower - reference, 0.0) + np.maximum(
            reference - soft_upper, 0.0
        )
        command_violation = np.maximum(soft_lower - command_vector, 0.0) + np.maximum(
            command_vector - soft_upper, 0.0
        )

        for index, value in enumerate(command_vector):
            if reference[index] < hard_lower[index] - position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "reference_lower_hard_position",
                        "value": float(reference[index]),
                        "limit": float(hard_lower[index]),
                    }
                )
            if reference[index] > hard_upper[index] + position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "reference_upper_hard_position",
                        "value": float(reference[index]),
                        "limit": float(hard_upper[index]),
                    }
                )
            if value < hard_lower[index] - position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "lower_hard_position",
                        "value": float(value),
                        "limit": float(hard_lower[index]),
                    }
                )
                continue
            if value > hard_upper[index] + position_tolerance:
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "upper_hard_position",
                        "value": float(value),
                        "limit": float(hard_upper[index]),
                    }
                )
                continue
            if command_violation[index] <= position_tolerance:
                continue
            if reference_violation[index] <= position_tolerance:
                kind = (
                    "lower_position" if value < soft_lower[index] else "upper_position"
                )
                limit = (
                    soft_lower[index]
                    if value < soft_lower[index]
                    else soft_upper[index]
                )
                hits.append(
                    {
                        "joint_index": index,
                        "kind": kind,
                        "value": float(value),
                        "limit": float(limit),
                    }
                )
                continue
            if (
                command_violation[index]
                > reference_violation[index] + position_tolerance
            ):
                hits.append(
                    {
                        "joint_index": index,
                        "kind": "recovery_outward",
                        "value": float(command_violation[index]),
                        "limit": float(reference_violation[index]),
                    }
                )

        has_position_hit = any(
            "position" in str(hit["kind"]) or hit["kind"] == "recovery_outward"
            for hit in hits
        )
        if (
            bool(np.any(command_violation > position_tolerance))
            and not has_position_hit
        ):
            if float(np.sum(command_violation)) >= (
                float(np.sum(reference_violation)) - position_tolerance
            ):
                hits.append(
                    {
                        "joint_index": -1,
                        "kind": "recovery_no_progress",
                        "value": float(np.sum(command_violation)),
                        "limit": float(np.sum(reference_violation)),
                    }
                )
            else:
                for index in np.flatnonzero(
                    command_violation > position_tolerance
                ).tolist():
                    below = command_vector[index] < soft_lower[index]
                    recovery.append(
                        {
                            "joint_index": int(index),
                            "kind": "lower" if below else "upper",
                            "hard_limit": float(
                                hard_lower[index] if below else hard_upper[index]
                            ),
                            "soft_limit": float(
                                soft_lower[index] if below else soft_upper[index]
                            ),
                            "reference": float(reference[index]),
                            "command": float(command_vector[index]),
                            "violation_before": float(reference_violation[index]),
                            "violation_after": float(command_violation[index]),
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
        return hits, velocity, acceleration, recovery

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
