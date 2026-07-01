# ruff: noqa: D100,D103

import math

import pytest

from reachy_mini.daemon.tracking.look_at_reference import (
    ImageErrorReferenceConfig,
    LookAtSphere,
    SphericalLookAtReferenceController,
)


def test_reference_persists_and_rotates_absolute_direction() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(origin_z=0.02),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_angular_speed=2.0,
        ),
    )

    first = controller.update(error_x=0.5, error_y=0.0, dt=0.1)
    second = controller.update(error_x=0.5, error_y=0.0, dt=0.1)

    assert first.direction == pytest.approx((math.cos(0.05), -math.sin(0.05), 0.0))
    assert second.direction == pytest.approx((math.cos(0.1), -math.sin(0.1), 0.0))
    assert second.target.x == pytest.approx(0.5 * math.cos(0.1))
    assert second.target.y == pytest.approx(-0.5 * math.sin(0.1))
    assert second.target.z == pytest.approx(0.02)


def test_vertical_image_error_rotates_direction_with_expected_sign() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
        ImageErrorReferenceConfig(vertical_rate=1.0, max_angular_speed=2.0),
    )

    below_center = controller.update(error_x=0.0, error_y=0.5, dt=0.1)

    assert below_center.direction == pytest.approx(
        (math.cos(0.05), 0.0, -math.sin(0.05))
    )
    assert below_center.target.z == pytest.approx(-0.5 * math.sin(0.05))


def test_angular_rate_is_vector_limited() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(elevation_limit=1.0),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_angular_speed=0.1,
            max_update_interval=1.0,
        ),
    )

    update = controller.update(error_x=1.0, error_y=-1.0, dt=1.0)

    assert math.hypot(update.delta_azimuth, update.delta_elevation) == pytest.approx(
        0.1
    )
    assert math.dist(update.direction, (0.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_elevation_saturation_has_no_hidden_windup() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(elevation_limit=0.02),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_angular_speed=1.0,
        ),
    )

    saturated = controller.update(error_x=0.0, error_y=-1.0, dt=0.1)
    inward = controller.update(error_x=0.0, error_y=1.0, dt=0.01)

    assert saturated.saturated
    assert math.asin(saturated.direction[2]) == pytest.approx(0.02)
    assert not inward.saturated
    assert math.asin(inward.direction[2]) == pytest.approx(0.01)


def test_complete_horizontal_rotation_is_normalized_and_wrap_free() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
        ImageErrorReferenceConfig(
            horizontal_rate=1.0,
            vertical_rate=1.0,
            max_angular_speed=1.0,
        ),
    )
    previous = controller.direction

    for _ in range(80):
        update = controller.update(error_x=-1.0, error_y=0.0, dt=0.1)
        assert math.dist(update.direction, (0.0, 0.0, 0.0)) == pytest.approx(1.0)
        assert math.dist(update.direction, previous) < 0.101
        previous = update.direction

    assert controller.direction == pytest.approx(
        (math.cos(8.0), math.sin(8.0), 0.0),
        abs=1e-12,
    )


def test_centered_observations_freeze_the_absolute_reference() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
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
    assert held.delta_azimuth == 0.0
    assert held.delta_elevation == 0.0


def test_error_outside_exit_resumes_from_the_persistent_direction() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(),
        ImageErrorReferenceConfig(center_frames=1, max_angular_speed=2.0),
    )
    centered = controller.update(error_x=0.0, error_y=0.0, dt=0.02)
    resumed = controller.update(error_x=-0.1, error_y=0.0, dt=0.02)

    assert centered.centered
    assert not resumed.centered
    assert resumed.target.y > centered.target.y


def test_missing_observation_and_stale_interval_freeze_without_mutation() -> None:
    controller = SphericalLookAtReferenceController(LookAtSphere())
    moved = controller.update(error_x=0.2, error_y=0.0, dt=0.02)

    missing = controller.freeze()
    stale = controller.update(error_x=1.0, error_y=1.0, dt=0.2)

    assert missing.target == moved.target
    assert missing.reason == "no_observation"
    assert stale.target == moved.target
    assert stale.reason == "stale_interval"


def test_reset_returns_to_the_absolute_forward_direction() -> None:
    controller = SphericalLookAtReferenceController(
        LookAtSphere(
            distance=0.6,
            origin_x=0.01,
            origin_y=0.03,
            origin_z=-0.01,
        ),
    )
    controller.update(error_x=0.2, error_y=0.2, dt=0.02)

    target = controller.reset()

    assert (target.x, target.y, target.z) == pytest.approx((0.61, 0.03, -0.01))
    assert controller.direction == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "sphere",
    [
        LookAtSphere,
        lambda: LookAtSphere(distance=0.0),
        lambda: LookAtSphere(elevation_limit=0.0),
        lambda: LookAtSphere(elevation_limit=math.pi / 2.0),
        lambda: LookAtSphere(origin_y=math.nan),
    ],
)
def test_invalid_sphere_values_are_rejected(sphere) -> None:  # type: ignore[no-untyped-def]
    if sphere is LookAtSphere:
        sphere()
        return
    with pytest.raises(ValueError):
        sphere()


@pytest.mark.parametrize(
    "config",
    [
        lambda: ImageErrorReferenceConfig(horizontal_rate=0.0),
        lambda: ImageErrorReferenceConfig(max_angular_speed=math.inf),
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
    controller = SphericalLookAtReferenceController(LookAtSphere())
    before = controller.target

    with pytest.raises(ValueError):
        controller.update(error_x=error_x, error_y=error_y, dt=dt)

    assert controller.target == before
