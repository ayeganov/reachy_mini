"""Validated configuration for daemon-owned visual tracking."""

import math

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class VisualServoConfig(BaseModel):
    """Runtime configuration with the hardware-approved defaults."""

    model_config = ConfigDict(extra="forbid")

    control_frequency: FiniteFloat = Field(default=50.0, gt=0.0)
    min_confidence: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    max_detection_age: FiniteFloat = Field(default=0.35, gt=0.0)
    smoothing_alpha: FiniteFloat = Field(default=1.0, gt=0.0, le=1.0)
    lookahead_distance: FiniteFloat = Field(default=0.5, gt=0.0)
    image_horizontal_fov: FiniteFloat = Field(
        default=math.radians(98.88965079926311), gt=0.0, lt=math.pi
    )
    image_vertical_fov: FiniteFloat = Field(
        default=math.radians(66.67916209122708), gt=0.0, lt=math.pi
    )
    image_error_elevation_limit: FiniteFloat = Field(
        default=math.radians(15.0), gt=0.0, lt=math.pi / 2.0
    )
    image_error_upward_elevation_limit: FiniteFloat = Field(
        default=math.radians(20.0), gt=0.0, lt=math.pi / 2.0
    )
    image_error_downward_elevation_limit: FiniteFloat = Field(
        default=math.radians(15.0), gt=0.0, lt=math.pi / 2.0
    )
    joint_safety_margin: FiniteFloat = Field(default=0.1745329252, ge=0.0)
    max_joint_velocity: FiniteFloat = Field(default=0.60, gt=0.0)
    max_joint_acceleration: FiniteFloat = Field(default=2.40, gt=0.0)
    max_joint_jerk: FiniteFloat = Field(default=16.0, gt=0.0)
    look_at_profile_response_hz: FiniteFloat = Field(default=2.0, gt=0.0, le=5.0)
    automatic_body_yaw: bool = True
    telemetry_capacity: int = Field(default=3000, gt=0, le=5000)
