# ruff: noqa: D100,D103

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from reachy_mini.kinematics import AnalyticalKinematics


def test_analytical_kinematics() -> None:
    ak = AnalyticalKinematics()
    pose = np.eye(4)
    sol = ak.ik(pose)
    assert sol is not None, "IK solution should be found"
    fk_pose = ak.fk(sol, no_iterations=10)
    assert np.allclose(fk_pose, pose, atol=1e-2), "FK should match the original pose"


def test_analytical_kinematics_with_yaw() -> None:
    ak = AnalyticalKinematics()
    pose = np.eye(4)
    body_yaw = np.pi / 4  # 45 degrees
    sol = ak.ik(pose, body_yaw=body_yaw)
    assert sol is not None, "IK solution should be found with body yaw"
    fk_pose = ak.fk(sol, no_iterations=10)
    assert np.allclose(fk_pose, pose, atol=1e-2), (
        "FK should match the original pose with body yaw"
    )


@pytest.mark.parametrize(
    (
        "before_degrees",
        "after_degrees",
        "current_body_degrees",
        "expected_before_degrees",
        "expected_after_degrees",
    ),
    [
        (179.0, -179.0, 110.0, 114.0, 116.0),
        (-179.0, 179.0, -110.0, -114.0, -116.0),
    ],
)
def test_automatic_body_yaw_retains_reachable_branch_across_rear_wrap(
    before_degrees: float,
    after_degrees: float,
    current_body_degrees: float,
    expected_before_degrees: float,
    expected_after_degrees: float,
) -> None:
    kinematics = AnalyticalKinematics()
    before_pose = np.eye(4)
    before_pose[:3, :3] = R.from_euler("z", math.radians(before_degrees)).as_matrix()
    before = kinematics.ik(
        before_pose,
        body_yaw=math.radians(current_body_degrees),
    )

    after_pose = np.eye(4)
    after_pose[:3, :3] = R.from_euler("z", math.radians(after_degrees)).as_matrix()
    after = kinematics.ik(after_pose, body_yaw=float(before[0]))

    assert math.degrees(float(before[0])) == pytest.approx(expected_before_degrees)
    assert math.degrees(float(after[0])) == pytest.approx(expected_after_degrees)
    assert abs(float(after[0] - before[0])) < math.radians(5.0)
