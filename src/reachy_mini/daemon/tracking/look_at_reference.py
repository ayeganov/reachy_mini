"""Persistent absolute look-at references driven by normalized image error."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class LookAtPlane:
    """Circular robot-frame target plane used by metric look-at control."""

    distance: float = 0.5
    center_y: float = 0.0
    center_z: float = 0.0
    radius: float = 0.15

    def __post_init__(self) -> None:
        """Validate finite plane geometry."""
        values = (self.distance, self.center_y, self.center_z, self.radius)
        if not all(isfinite(value) for value in values):
            raise ValueError("look-at plane values must be finite")
        if self.distance <= 0.0:
            raise ValueError("look-at plane distance must be positive")
        if self.radius <= 0.0:
            raise ValueError("look-at plane radius must be positive")


@dataclass(frozen=True)
class ImageErrorReferenceConfig:
    """Tuning and lifecycle limits for absolute-reference updates."""

    horizontal_rate: float = 0.5
    vertical_rate: float = 0.5
    max_target_speed: float = 0.15
    center_enter: float = 0.03
    center_exit: float = 0.05
    center_frames: int = 3
    max_update_interval: float = 0.1

    def __post_init__(self) -> None:
        """Validate rates, hysteresis, and timing limits."""
        positive = (
            self.horizontal_rate,
            self.vertical_rate,
            self.max_target_speed,
            self.max_update_interval,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "reference rates and intervals must be finite and positive"
            )
        if not isfinite(self.center_enter) or not 0.0 <= self.center_enter < 1.0:
            raise ValueError("center_enter must be finite and in [0, 1)")
        if (
            not isfinite(self.center_exit)
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
    """One absolute metric target on the configured plane."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ReferenceUpdate:
    """Result and telemetry for one observation or freeze event."""

    target: LookAtReference
    error_x: float | None
    error_y: float | None
    delta_y: float
    delta_z: float
    centered: bool
    centered_frame_count: int
    saturated: bool
    updated: bool
    reason: str


class AbsoluteLookAtReferenceController:
    """Integrate image error into one persistent robot-frame look-at target."""

    def __init__(
        self,
        plane: LookAtPlane,
        config: ImageErrorReferenceConfig | None = None,
    ) -> None:
        """Initialize the controller at the plane center."""
        self.plane = plane
        self.config = config or ImageErrorReferenceConfig()
        self._target_y = plane.center_y
        self._target_z = plane.center_z
        self._centered = False
        self._centered_frame_count = 0

    @property
    def target(self) -> LookAtReference:
        """Return the current absolute reference without changing state."""
        return LookAtReference(
            x=self.plane.distance,
            y=self._target_y,
            z=self._target_z,
        )

    def reset(self) -> LookAtReference:
        """Reset the reference and hysteresis state to the plane center."""
        self._target_y = self.plane.center_y
        self._target_z = self.plane.center_z
        self._centered = False
        self._centered_frame_count = 0
        return self.target

    def freeze(self, reason: str = "no_observation") -> ReferenceUpdate:
        """Return the unchanged target for an interval with no usable observation."""
        return ReferenceUpdate(
            target=self.target,
            error_x=None,
            error_y=None,
            delta_y=0.0,
            delta_z=0.0,
            centered=self._centered,
            centered_frame_count=self._centered_frame_count,
            saturated=False,
            updated=False,
            reason=reason,
        )

    def update(self, error_x: float, error_y: float, dt: float) -> ReferenceUpdate:
        """Apply one normalized image-error observation to the absolute target."""
        if not all(isfinite(value) for value in (error_x, error_y, dt)):
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
            return ReferenceUpdate(
                target=self.target,
                error_x=error_x,
                error_y=error_y,
                delta_y=0.0,
                delta_z=0.0,
                centered=self._centered,
                centered_frame_count=self._centered_frame_count,
                saturated=False,
                updated=False,
                reason="centered" if self._centered else "centering",
            )

        velocity_y = -self.config.horizontal_rate * error_x
        velocity_z = -self.config.vertical_rate * error_y
        speed = hypot(velocity_y, velocity_z)
        if speed > self.config.max_target_speed:
            scale = self.config.max_target_speed / speed
            velocity_y *= scale
            velocity_z *= scale

        proposed_y = self._target_y + velocity_y * dt
        proposed_z = self._target_z + velocity_z * dt
        offset_y = proposed_y - self.plane.center_y
        offset_z = proposed_z - self.plane.center_z
        distance = hypot(offset_y, offset_z)
        saturated = distance > self.plane.radius
        if saturated:
            scale = self.plane.radius / distance
            proposed_y = self.plane.center_y + offset_y * scale
            proposed_z = self.plane.center_z + offset_z * scale

        delta_y = proposed_y - self._target_y
        delta_z = proposed_z - self._target_z
        self._target_y = proposed_y
        self._target_z = proposed_z
        return ReferenceUpdate(
            target=self.target,
            error_x=error_x,
            error_y=error_y,
            delta_y=delta_y,
            delta_z=delta_z,
            centered=False,
            centered_frame_count=self._centered_frame_count,
            saturated=saturated,
            updated=delta_y != 0.0 or delta_z != 0.0,
            reason="saturated" if saturated else "tracking",
        )
