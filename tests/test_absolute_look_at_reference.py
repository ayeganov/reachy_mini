# ruff: noqa: D100,D103

import math

import pytest

from reachy_mini.daemon.tracking.look_at_reference import (
    AbsoluteLookAtReferenceController,
    ImageErrorReferenceConfig,
    LookAtPlane,
)


def test_reference_persists_and_accumulates_absolute_target_updates() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(center_z=0.02),
        ImageErrorReferenceConfig(
            horizontal_rate=0.2,
            vertical_rate=0.3,
            max_target_speed=1.0,
        ),
    )

    first = controller.update(error_x=0.5, error_y=0.0, dt=0.1)
    second = controller.update(error_x=0.5, error_y=0.0, dt=0.1)

    assert first.target.y == pytest.approx(-0.01)
    assert second.target.y == pytest.approx(-0.02)
    assert second.target.x == 0.5
    assert second.target.z == 0.02


def test_vertical_image_error_moves_absolute_target_with_expected_sign() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(),
        ImageErrorReferenceConfig(vertical_rate=0.5, max_target_speed=1.0),
    )

    below_center = controller.update(error_x=0.0, error_y=0.5, dt=0.1)

    assert below_center.target.z == pytest.approx(-0.025)
    assert below_center.target.y == 0.0


def test_target_velocity_is_vector_limited() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(radius=1.0),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_target_speed=0.1,
            max_update_interval=1.0,
        ),
    )

    update = controller.update(error_x=1.0, error_y=-1.0, dt=1.0)

    assert math.hypot(update.delta_y, update.delta_z) == pytest.approx(0.1)


def test_circular_saturation_has_no_hidden_windup() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(radius=0.02),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_target_speed=1.0,
        ),
    )

    saturated = controller.update(error_x=1.0, error_y=0.0, dt=0.1)
    inward = controller.update(error_x=-1.0, error_y=0.0, dt=0.01)

    assert saturated.saturated
    assert saturated.target.y == pytest.approx(-0.02)
    assert not inward.saturated
    assert inward.target.y == pytest.approx(-0.01)


def test_centered_observations_freeze_the_absolute_reference() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(),
        ImageErrorReferenceConfig(center_frames=3),
    )
    moved = controller.update(error_x=0.2, error_y=0.0, dt=0.02)

    controller.update(error_x=0.01, error_y=-0.01, dt=0.02)
    controller.update(error_x=0.01, error_y=-0.01, dt=0.02)
    centered = controller.update(error_x=0.01, error_y=-0.01, dt=0.02)
    held = controller.update(error_x=0.02, error_y=0.02, dt=0.02)

    assert centered.centered
    assert centered.target == moved.target
    assert held.reason == "centered"
    assert held.target == centered.target
    assert held.delta_y == 0.0
    assert held.delta_z == 0.0


def test_error_outside_exit_resumes_from_the_persistent_target() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(),
        ImageErrorReferenceConfig(center_frames=1, max_target_speed=1.0),
    )
    centered = controller.update(error_x=0.0, error_y=0.0, dt=0.02)
    resumed = controller.update(error_x=-0.1, error_y=0.0, dt=0.02)

    assert centered.centered
    assert not resumed.centered
    assert resumed.target.y > centered.target.y


def test_missing_observation_and_stale_interval_freeze_without_mutation() -> None:
    controller = AbsoluteLookAtReferenceController(LookAtPlane())
    moved = controller.update(error_x=0.2, error_y=0.0, dt=0.02)

    missing = controller.freeze()
    stale = controller.update(error_x=1.0, error_y=1.0, dt=0.2)

    assert missing.target == moved.target
    assert missing.reason == "no_observation"
    assert stale.target == moved.target
    assert stale.reason == "stale_interval"


def test_reset_returns_to_the_absolute_plane_center() -> None:
    controller = AbsoluteLookAtReferenceController(
        LookAtPlane(distance=0.6, center_y=0.03, center_z=-0.01),
    )
    controller.update(error_x=0.2, error_y=0.2, dt=0.02)

    target = controller.reset()

    assert (target.x, target.y, target.z) == (0.6, 0.03, -0.01)


@pytest.mark.parametrize(
    "plane",
    [
        LookAtPlane,
        lambda: LookAtPlane(distance=0.0),
        lambda: LookAtPlane(radius=-1.0),
        lambda: LookAtPlane(center_y=math.nan),
    ],
)
def test_invalid_plane_values_are_rejected(plane) -> None:  # type: ignore[no-untyped-def]
    if plane is LookAtPlane:
        plane()
        return
    with pytest.raises(ValueError):
        plane()


@pytest.mark.parametrize(
    "config",
    [
        lambda: ImageErrorReferenceConfig(horizontal_rate=0.0),
        lambda: ImageErrorReferenceConfig(max_target_speed=math.inf),
        lambda: ImageErrorReferenceConfig(center_enter=-0.1),
        lambda: ImageErrorReferenceConfig(center_enter=0.1, center_exit=0.1),
        lambda: ImageErrorReferenceConfig(center_frames=0),
    ],
)
def test_invalid_configuration_is_rejected(config) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        config()


@pytest.mark.parametrize(
    ("error_x", "error_y", "dt"),
    [
        (math.nan, 0.0, 0.02),
        (0.0, math.inf, 0.02),
        (0.0, 0.0, math.nan),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -0.1),
    ],
)
def test_invalid_updates_are_rejected_without_mutation(
    error_x: float,
    error_y: float,
    dt: float,
) -> None:
    controller = AbsoluteLookAtReferenceController(LookAtPlane())
    before = controller.target

    with pytest.raises(ValueError):
        controller.update(error_x=error_x, error_y=error_y, dt=dt)

    assert controller.target == before
