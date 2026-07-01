"""Persistent absolute look-at directions driven by normalized image error."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LookAtSphere:
    """Fixed robot-frame origin, distance, and vertical gaze range."""

    distance: float = 0.5
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    elevation_limit: float = math.atan2(0.2, 0.5)

    def __post_init__(self) -> None:
        """Validate finite spherical workspace geometry."""
        values = (
            self.distance,
            self.origin_x,
            self.origin_y,
            self.origin_z,
            self.elevation_limit,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("look-at sphere values must be finite")
        if self.distance <= 0.0:
            raise ValueError("look-at distance must be positive")
        if not 0.0 < self.elevation_limit < math.pi / 2.0:
            raise ValueError("elevation_limit must be in (0, pi / 2)")


@dataclass(frozen=True)
class ImageErrorReferenceConfig:
    """Tuning and lifecycle limits for absolute direction updates."""

    horizontal_rate: float = 4.0
    vertical_rate: float = 4.0
    max_angular_speed: float = 1.2
    center_enter: float = 0.03
    center_exit: float = 0.05
    center_frames: int = 3
    max_update_interval: float = 0.1

    def __post_init__(self) -> None:
        """Validate rates, hysteresis, and timing limits."""
        positive = (
            self.horizontal_rate,
            self.vertical_rate,
            self.max_angular_speed,
            self.max_update_interval,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "reference rates and intervals must be finite and positive"
            )
        if not math.isfinite(self.center_enter) or not 0.0 <= self.center_enter < 1.0:
            raise ValueError("center_enter must be finite and in [0, 1)")
        if (
            not math.isfinite(self.center_exit)
            or not self.center_enter < self.center_exit <= 1.0
        ):
            raise ValueError("center_exit must be finite, above enter, and at most 1")
        if (
            isinstance(self.center_frames, bool)
            or not isinstance(self.center_frames, int)
            or self.center_frames < 1
        ):
            raise ValueError("center_frames must be a positive integer")


@dataclass(frozen=True)
class LookAtReference:
    """One absolute metric point on the configured gaze sphere."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ReferenceUpdate:
    """Result and telemetry for one observation or freeze event."""

    target: LookAtReference
    direction: tuple[float, float, float]
    error_x: float | None
    error_y: float | None
    delta_azimuth: float
    delta_elevation: float
    centered: bool
    centered_frame_count: int
    saturated: bool
    updated: bool
    reason: str


class SphericalLookAtReferenceController:
    """Integrate image error into one persistent robot-frame gaze direction."""

    def __init__(
        self,
        sphere: LookAtSphere,
        config: ImageErrorReferenceConfig | None = None,
    ) -> None:
        """Initialize the controller facing robot-frame positive X."""
        self.sphere = sphere
        self.config = config or ImageErrorReferenceConfig()
        self._direction = (1.0, 0.0, 0.0)
        self._centered = False
        self._centered_frame_count = 0

    @property
    def direction(self) -> tuple[float, float, float]:
        """Return the stored unit gaze direction without changing state."""
        return self._direction

    @property
    def target(self) -> LookAtReference:
        """Return the current absolute Cartesian reference."""
        dx, dy, dz = self._direction
        return LookAtReference(
            x=self.sphere.origin_x + self.sphere.distance * dx,
            y=self.sphere.origin_y + self.sphere.distance * dy,
            z=self.sphere.origin_z + self.sphere.distance * dz,
        )

    def reset(self) -> LookAtReference:
        """Reset the reference and hysteresis state to forward."""
        self._direction = (1.0, 0.0, 0.0)
        self._centered = False
        self._centered_frame_count = 0
        return self.target

    def freeze(self, reason: str = "no_observation") -> ReferenceUpdate:
        """Return the unchanged target for an interval with no usable observation."""
        return self._result(
            error_x=None,
            error_y=None,
            delta_azimuth=0.0,
            delta_elevation=0.0,
            saturated=False,
            updated=False,
            reason=reason,
        )

    def update(self, error_x: float, error_y: float, dt: float) -> ReferenceUpdate:
        """Apply one normalized image-error observation to the absolute direction."""
        if not all(math.isfinite(value) for value in (error_x, error_y, dt)):
            raise ValueError("image error and dt must be finite")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if dt > self.config.max_update_interval:
            return self.freeze(reason="stale_interval")

        within_enter = (
            abs(error_x) <= self.config.center_enter
            and abs(error_y) <= self.config.center_enter
        )
        outside_exit = (
            abs(error_x) > self.config.center_exit
            or abs(error_y) > self.config.center_exit
        )
        if self._centered:
            if outside_exit:
                self._centered = False
                self._centered_frame_count = 0
        elif within_enter:
            self._centered_frame_count += 1
            if self._centered_frame_count >= self.config.center_frames:
                self._centered = True
        else:
            self._centered_frame_count = 0

        if self._centered or within_enter:
            return self._result(
                error_x=error_x,
                error_y=error_y,
                delta_azimuth=0.0,
                delta_elevation=0.0,
                saturated=False,
                updated=False,
                reason="centered" if self._centered else "centering",
            )

        azimuth_rate = -self.config.horizontal_rate * error_x
        elevation_rate = -self.config.vertical_rate * error_y
        angular_speed = math.hypot(azimuth_rate, elevation_rate)
        if angular_speed > self.config.max_angular_speed:
            scale = self.config.max_angular_speed / angular_speed
            azimuth_rate *= scale
            elevation_rate *= scale

        delta_azimuth = azimuth_rate * dt
        requested_delta_elevation = elevation_rate * dt
        direction_x, direction_y, direction_z = self._direction

        cos_azimuth = math.cos(delta_azimuth)
        sin_azimuth = math.sin(delta_azimuth)
        rotated_x = cos_azimuth * direction_x - sin_azimuth * direction_y
        rotated_y = sin_azimuth * direction_x + cos_azimuth * direction_y

        current_elevation = math.asin(max(-1.0, min(1.0, direction_z)))
        requested_elevation = current_elevation + requested_delta_elevation
        elevation = max(
            -self.sphere.elevation_limit,
            min(self.sphere.elevation_limit, requested_elevation),
        )
        saturated = elevation != requested_elevation
        delta_elevation = elevation - current_elevation

        horizontal_norm = math.hypot(rotated_x, rotated_y)
        if horizontal_norm <= 1e-12:
            raise RuntimeError("look-at direction lost its horizontal component")
        horizontal_scale = math.cos(elevation) / horizontal_norm
        proposed = (
            rotated_x * horizontal_scale,
            rotated_y * horizontal_scale,
            math.sin(elevation),
        )
        norm = math.dist(proposed, (0.0, 0.0, 0.0))
        self._direction = (
            proposed[0] / norm,
            proposed[1] / norm,
            proposed[2] / norm,
        )
        return self._result(
            error_x=error_x,
            error_y=error_y,
            delta_azimuth=delta_azimuth,
            delta_elevation=delta_elevation,
            saturated=saturated,
            updated=delta_azimuth != 0.0 or delta_elevation != 0.0,
            reason="saturated" if saturated else "tracking",
        )

    def _result(
        self,
        *,
        error_x: float | None,
        error_y: float | None,
        delta_azimuth: float,
        delta_elevation: float,
        saturated: bool,
        updated: bool,
        reason: str,
    ) -> ReferenceUpdate:
        return ReferenceUpdate(
            target=self.target,
            direction=self._direction,
            error_x=error_x,
            error_y=error_y,
            delta_azimuth=delta_azimuth,
            delta_elevation=delta_elevation,
            centered=self._centered,
            centered_frame_count=self._centered_frame_count,
            saturated=saturated,
            updated=updated,
            reason=reason,
        )
