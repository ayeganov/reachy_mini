# ruff: noqa: D100,D103

from __future__ import annotations

import numpy as np
import pytest

from reachy_mini.daemon.tracking.config import VisualServoConfig
from reachy_mini.daemon.tracking.joint_motion import (
    JointCommandSafetyGuard,
    LookAtJointCommandProfile,
)


def motion_config() -> VisualServoConfig:
    return VisualServoConfig(
        joint_safety_margin=0.1745329252,
        max_joint_velocity=0.60,
        max_joint_acceleration=1.60,
        max_joint_jerk=8.0,
        look_at_profile_response_hz=1.0,
    )


def test_profile_bounds_dynamic_target_derivatives() -> None:
    config = motion_config()
    profile = LookAtJointCommandProfile(config=config)
    current = np.zeros(7)
    commands = []

    for tick in range(200):
        desired = np.full(7, 0.5 if tick < 80 else -0.5)
        command, _hits = profile.update_with_telemetry(desired, current, 0.02)
        commands.append(command.copy())
        current = command

    values = np.asarray(commands)
    velocity = np.diff(values, axis=0) / 0.02
    acceleration = np.diff(velocity, axis=0) / 0.02
    jerk = np.diff(acceleration, axis=0) / 0.02
    assert np.max(np.abs(velocity)) <= config.max_joint_velocity + 1e-8
    assert np.max(np.abs(acceleration)) <= config.max_joint_acceleration + 1e-8
    assert np.max(np.abs(jerk)) <= config.max_joint_jerk + 1e-6


def test_guard_accepts_monotonic_inward_soft_limit_recovery() -> None:
    config = motion_config()
    guard = JointCommandSafetyGuard(config=config)
    soft_upper = guard.limits[:, 1] - config.joint_safety_margin
    current = soft_upper.copy()
    current[6] += 0.005
    guard.reset(current)
    command = current.copy()
    command[6] -= config.max_joint_jerk * 0.02**3

    hits, velocity, acceleration, recovery = guard.check_with_telemetry(
        command,
        current,
        0.02,
    )

    assert hits == []
    assert recovery[0]["joint_index"] == 6
    assert recovery[0]["violation_after"] < recovery[0]["violation_before"]
    guard.commit(command, velocity, acceleration)


def test_look_at_joint_profile_initializes_from_current_and_profiles_body_yaw() -> None:
    profile = LookAtJointCommandProfile(config=motion_config())
    current = np.array([0.1, *([0.2] * 6)])
    desired = np.array([0.7, *([0.3] * 6)])

    command, _hits = profile.update_with_telemetry(desired, current, dt=0.02)

    assert current[0] < command[0] < desired[0]
    assert np.all(command[1:] >= current[1:])
    assert np.all(command[1:] < desired[1:])
    assert np.max(command[1:] - current[1:]) == pytest.approx(8.0 * 0.02**3)


def test_look_at_joint_profile_converges_to_fixed_target_within_two_seconds() -> None:
    profile = LookAtJointCommandProfile(config=motion_config())
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
    config = motion_config()
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
    config = motion_config()
    profile = LookAtJointCommandProfile(config=config)
    guard = JointCommandSafetyGuard(config=config)
    current = np.zeros(7)
    desired = np.full(7, target)

    for _ in range(15):
        command, _hits = profile.update_with_telemetry(desired, current, 0.02)
        guard_hits, velocity, acceleration, _recovery = guard.check_with_telemetry(
            command, current, 0.02
        )
        assert guard_hits == []
        guard.commit(command, velocity, acceleration)
        current = command

    for _ in range(100):
        command, _hits = profile.stop_with_telemetry(current, 0.02)
        guard_hits, velocity, acceleration, _recovery = guard.check_with_telemetry(
            command, current, 0.02
        )
        assert guard_hits == []
        guard.commit(command, velocity, acceleration)
        current = command

    assert profile.stationary
    held, _hits = profile.stop_with_telemetry(current, 0.02)
    np.testing.assert_allclose(held, current, atol=1e-12)


def test_look_at_joint_profile_preserves_boundary_on_abrupt_reversals() -> None:
    config = motion_config()
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
        guard_hits, next_velocity, next_acceleration, _recovery = (
            guard.check_with_telemetry(command, current, dt)
        )

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
    config = motion_config()
    guard = JointCommandSafetyGuard(config=config)
    dt = 0.02

    def hits_for_jerk(jerk: float) -> list[dict[str, float | int | str]]:
        command = np.full(7, jerk * dt**3)
        hits, _velocity, _acceleration, _recovery = guard.check_with_telemetry(
            command, np.zeros(7), dt
        )
        return hits

    assert not any(hit["kind"] == "jerk" for hit in hits_for_jerk(8.00000245))
    assert any(hit["kind"] == "jerk" for hit in hits_for_jerk(8.00002))


def test_joint_command_safety_guard_accepts_inward_recovery_band_progress() -> None:
    config = motion_config()
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
    profile = LookAtJointCommandProfile(config=motion_config())
    profile.update_with_telemetry(np.array([0.0, *([0.3] * 6)]), np.zeros(7), dt=0.02)

    profile.reset()

    assert profile._position is None
    np.testing.assert_allclose(profile._velocity, 0.0)
    np.testing.assert_allclose(profile._acceleration, 0.0)


def test_look_at_joint_profile_rejects_invalid_inputs_without_state_mutation() -> None:
    profile = LookAtJointCommandProfile(config=motion_config())
    desired = np.array([0.0, *([0.3] * 6)])
    profile.update_with_telemetry(desired, np.zeros(7), dt=0.02)
    state = (
        profile._position.copy(),  # type: ignore[union-attr]
        profile._velocity.copy(),
        profile._acceleration.copy(),
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

    for response_hz in (0.0, -1.0, 5.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="look_at_profile_response_hz"):
            LookAtJointCommandProfile(
                config=VisualServoConfig(look_at_profile_response_hz=response_hz)
            )
