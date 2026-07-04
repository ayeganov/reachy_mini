# ruff: noqa: D100,D103

from __future__ import annotations

import numpy as np

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
