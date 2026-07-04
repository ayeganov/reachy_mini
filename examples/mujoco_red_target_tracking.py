"""Track a mouse-movable red marker through the approved metric look-at path."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend
from reachy_mini.daemon.tracking.look_at_reference import (
    ImageErrorReferenceConfig,
    LookAtSphere,
    ReferenceUpdate,
    SphericalLookAtReferenceController,
)
from reachy_mini.daemon.tracking.visual_servo import (
    TrackingLookAtTarget,
    VisualServoConfig,
    VisualServoController,
)

CONTROL_DT = 0.02
SENSOR_TICKS = 2
SIMULATION_Z_OFFSET = 0.177
MARKER_RADIUS = 0.025
DEFAULT_ORBIT_RADIUS = 0.5
DEFAULT_VERTICAL_RANGE = 0.2
ORBIT_PAD_SIZE = 640
RED = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
APPROVED_MOTION = {
    "joint_safety_margin": 0.1745329252,
    "max_joint_velocity": 0.6,
    "max_joint_acceleration": 2.4,
    "max_joint_jerk": 16.0,
    "look_at_profile_response_hz": 2.0,
}
SCENARIOS = {
    "center": (0.0, 0.0),
    "left": (math.atan2(0.2, 0.5), 0.0),
    "right": (-math.atan2(0.2, 0.5), 0.0),
    "top": (0.0, 0.2),
    "bottom": (0.0, -0.2),
    "top_left": (math.atan2(0.1414, 0.5), 0.1414),
    "top_right": (-math.atan2(0.1414, 0.5), 0.1414),
    "bottom_left": (math.atan2(0.1414, 0.5), -0.1414),
    "bottom_right": (-math.atan2(0.1414, 0.5), -0.1414),
}


@dataclass(frozen=True)
class MarkerDetection:
    """Centroid and normalized image error for the rendered red marker."""

    u: float
    v: float
    error_x: float
    error_y: float
    pixel_count: int


@dataclass(frozen=True)
class ScenarioResult:
    """Deterministic outcome for one rendered-camera target."""

    name: str
    centered_at_s: float | None
    held_center_s: float
    final_error: tuple[float, float] | None
    control_ticks: int
    rendered_frames: int
    command_count: int
    ik_failures: int
    guard_hits: int
    profile_position_hits: int
    maximum_direction_norm_error: float
    maximum_abs_elevation: float
    final_target: tuple[float, float, float]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OrbitStepResult:
    """Rendered-camera outcome for one incremental marker azimuth."""

    azimuth_degrees: float
    marker_visible: bool
    centered_at_s: float | None
    held_center_s: float
    body_yaw_degrees: float


@dataclass(frozen=True)
class OrbitSweepResult:
    """Deterministic result for one incremental visual orbit sweep."""

    requested_end_degrees: float
    reached_degrees: float
    steps: tuple[OrbitStepResult, ...]
    ik_failures: int
    guard_hits: int
    profile_position_hits: int
    maximum_direction_norm_error: float
    maximum_abs_elevation: float
    reasons: tuple[str, ...]


def detect_red_marker(frame: npt.NDArray[np.uint8]) -> MarkerDetection | None:
    """Return the largest bright-red component centroid from an RGB frame."""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    mask = ((red >= 120) & (red >= 2 * green) & (red >= 2 * blue)).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if component_count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixel_count = int(stats[index, cv2.CC_STAT_AREA])
    if pixel_count < 8:
        return None
    u, v = (float(value) for value in centroids[index])
    height, width = frame.shape[:2]
    return MarkerDetection(
        u=u,
        v=v,
        error_x=(u - width / 2.0) / (width / 2.0),
        error_y=(v - height / 2.0) / (height / 2.0),
        pixel_count=pixel_count,
    )


class MujocoRedTargetHarness:
    """Own deterministic physics, eye rendering, marker input, and look-at control."""

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        orbit_radius: float = DEFAULT_ORBIT_RADIUS,
        vertical_range: float = DEFAULT_VERTICAL_RANGE,
    ) -> None:
        """Initialize a neutral simulated robot and its eye-camera renderer."""
        if not math.isfinite(orbit_radius) or orbit_radius <= 0.0:
            raise ValueError("orbit_radius must be finite and positive")
        if not math.isfinite(vertical_range) or vertical_range <= 0.0:
            raise ValueError("vertical_range must be finite and positive")
        self.width = width
        self.height = height
        self.backend = MujocoBackend(scene="empty", headless=True)
        self.backend.data.qpos[self.backend.joint_qpos_addr] = 0.0
        self.backend.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.backend.model, self.backend.data)
        self._refresh_backend_state()
        self.backend.head_kinematics.set_automatic_body_yaw(True)

        self.camera_id = mujoco.mj_name2id(
            self.backend.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "eye_camera",
        )
        if self.camera_id < 0:
            raise RuntimeError("MuJoCo eye_camera was not found")
        self.renderer = mujoco.Renderer(
            self.backend.model,
            height=height,
            width=width,
        )
        self.studio_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.studio_camera)
        self.studio_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.studio_camera.lookat[:] = [0.0, 0.0, 0.15]
        self.studio_camera.distance = 1.4
        self.studio_camera.azimuth = 160.0
        self.studio_camera.elevation = -20.0
        self.studio_renderer: mujoco.Renderer | None = None

        head_pose = self.backend.get_present_head_pose()
        self.look_at_sphere = LookAtSphere(
            distance=orbit_radius,
            origin_x=float(head_pose[0, 3]),
            origin_y=float(head_pose[1, 3]),
            origin_z=float(head_pose[2, 3]),
            elevation_limit=math.atan2(vertical_range, orbit_radius),
        )
        self.reference = SphericalLookAtReferenceController(
            sphere=self.look_at_sphere,
            config=ImageErrorReferenceConfig(),
        )
        self.servo = VisualServoController(
            backend=self.backend,
            config=VisualServoConfig(
                joint_safety_margin=APPROVED_MOTION["joint_safety_margin"],
                max_joint_velocity=APPROVED_MOTION["max_joint_velocity"],
                max_joint_acceleration=APPROVED_MOTION["max_joint_acceleration"],
                max_joint_jerk=APPROVED_MOTION["max_joint_jerk"],
                look_at_profile_response_hz=APPROVED_MOTION[
                    "look_at_profile_response_hz"
                ],
            ),
        )

        neutral_marker = self._neutral_marker_position()
        self.marker_orbit_radius = orbit_radius
        self.marker_height_limit = vertical_range
        self.marker_orbit_center = np.array(
            [
                self.look_at_sphere.origin_x,
                self.look_at_sphere.origin_y,
                float(neutral_marker[2]),
            ],
            dtype=np.float64,
        )
        self.marker_position = neutral_marker
        self.marker_visible = True
        self.mode: Literal["vision", "oracle"] = "vision"
        self.last_detection: MarkerDetection | None = None
        self.last_frame_rgb: npt.NDArray[np.uint8] | None = None
        self.last_reference_update: ReferenceUpdate = self.reference.freeze("start")
        self.frame_id = 0
        self.rendered_frames = 0
        self.command_count = 0
        self.ik_failures = 0
        self.guard_hits = 0
        self.profile_position_hits = 0
        self.reasons: set[str] = set()
        self.maximum_direction_norm_error = 0.0
        self.maximum_abs_elevation = 0.0

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()
        if self.studio_renderer is not None:
            self.studio_renderer.close()

    def _refresh_backend_state(self) -> None:
        joints = self.backend.get_present_head_joint_positions()
        antennas = self.backend.get_present_antenna_joint_positions()
        self.backend.current_head_joint_positions = joints.copy()
        self.backend.current_antenna_joint_positions = antennas.copy()
        self.backend.update_head_kinematics_model(joints, antennas)
        self.backend.current_head_pose = self.backend.get_mj_present_head_pose()

    def _camera_pose_robot_frame(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        origin = self.backend.data.cam_xpos[self.camera_id].copy()
        origin[2] -= SIMULATION_Z_OFFSET
        rotation = self.backend.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        return origin, rotation

    def _neutral_marker_position(self) -> npt.NDArray[np.float64]:
        origin, rotation = self._camera_pose_robot_frame()
        forward = -rotation[:, 2]
        horizontal_direction = forward[:2]
        direction_squared = float(horizontal_direction @ horizontal_direction)
        if direction_squared <= 1e-12:
            raise RuntimeError("eye camera has no horizontal forward direction")
        center = np.array(
            [self.look_at_sphere.origin_x, self.look_at_sphere.origin_y],
            dtype=np.float64,
        )
        offset = origin[:2] - center
        linear = 2.0 * float(offset @ horizontal_direction)
        constant = float(offset @ offset) - self.look_at_sphere.distance**2
        discriminant = linear * linear - 4.0 * direction_squared * constant
        if discriminant < 0.0:
            raise RuntimeError("neutral camera ray does not reach the marker orbit")
        root = math.sqrt(discriminant)
        distances = (
            (-linear - root) / (2.0 * direction_squared),
            (-linear + root) / (2.0 * direction_squared),
        )
        positive_distances = [distance for distance in distances if distance > 0.0]
        if not positive_distances:
            raise RuntimeError("marker orbit is behind the neutral eye camera")
        return np.asarray(origin + max(positive_distances) * forward, dtype=np.float64)

    def reset(self) -> None:
        """Reset marker and persistent reference without changing motion limits."""
        self.set_marker_orbit(0.0, 0.0)
        self.reference.reset()
        self.last_reference_update = self.reference.freeze("reset")
        self.last_detection = None

    @property
    def marker_azimuth(self) -> float:
        """Return marker azimuth around the fixed robot-frame orbit center."""
        offset = self.marker_position - self.marker_orbit_center
        return math.atan2(float(offset[1]), float(offset[0]))

    @property
    def marker_height(self) -> float:
        """Return marker height relative to the neutral orbit."""
        return float(self.marker_position[2] - self.marker_orbit_center[2])

    def set_marker_orbit(self, azimuth: float, height: float) -> None:
        """Place the marker directly in canonical robot-frame orbit coordinates."""
        if not math.isfinite(azimuth) or not math.isfinite(height):
            raise ValueError("marker orbit coordinates must be finite")
        bounded_height = max(
            -self.marker_height_limit,
            min(self.marker_height_limit, height),
        )
        self.marker_position = self.marker_orbit_center + np.array(
            [
                self.marker_orbit_radius * math.cos(azimuth),
                self.marker_orbit_radius * math.sin(azimuth),
                bounded_height,
            ],
            dtype=np.float64,
        )

    def move_marker_from_orbit_pad(self, u: float, v: float, size: int) -> None:
        """Map one top-down pad position directly to marker azimuth."""
        center = float(size) / 2.0
        delta_x = u - center
        delta_y = center - v
        if math.hypot(delta_x, delta_y) <= 1e-9:
            return
        self.set_marker_orbit(
            math.atan2(delta_y, delta_x),
            self.marker_height,
        )

    def adjust_marker_height(self, delta: float) -> None:
        """Move the marker vertically while retaining its world azimuth."""
        if not math.isfinite(delta):
            raise ValueError("marker height delta must be finite")
        self.set_marker_orbit(self.marker_azimuth, self.marker_height + delta)

    def _add_marker(self, scene: mujoco.MjvScene) -> None:
        """Append the verification marker to one prepared render scene."""
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo render scene has no room for marker")
        if self.marker_visible:
            marker_simulation = self.marker_position.copy()
            marker_simulation[2] += SIMULATION_Z_OFFSET
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([MARKER_RADIUS, 0.0, 0.0]),
                marker_simulation,
                np.eye(3).reshape(-1),
                RED,
            )
            scene.ngeom += 1

    def render(self) -> npt.NDArray[np.uint8]:
        """Render the eye camera with one verification-only red sphere."""
        self.renderer.update_scene(self.backend.data, self.camera_id)
        self._add_marker(self.renderer.scene)
        frame = np.asarray(self.renderer.render(), dtype=np.uint8)
        self.rendered_frames += 1
        return frame

    def render_studio(self) -> npt.NDArray[np.uint8]:
        """Render a third-person operator view with the same marker."""
        if self.studio_renderer is None:
            self.studio_renderer = mujoco.Renderer(
                self.backend.model,
                height=640,
                width=640,
            )
        self.studio_renderer.update_scene(self.backend.data, self.studio_camera)
        self._add_marker(self.studio_renderer.scene)
        return np.asarray(self.studio_renderer.render(), dtype=np.uint8)

    def orbit_pad_frame(self, size: int = ORBIT_PAD_SIZE) -> npt.NDArray[np.uint8]:
        """Render a direct top-down azimuth control without camera unprojection."""
        frame = np.full((size, size, 3), 24, dtype=np.uint8)
        center = size // 2
        radius = size // 2 - 54
        cv2.circle(frame, (center, center), radius, (130, 130, 130), 2)
        cv2.line(
            frame,
            (center - radius, center),
            (center + radius, center),
            (70, 70, 70),
            1,
        )
        cv2.line(
            frame,
            (center, center - radius),
            (center, center + radius),
            (70, 70, 70),
            1,
        )
        cv2.arrowedLine(
            frame,
            (center, center),
            (center + 42, center),
            (255, 255, 255),
            3,
            tipLength=0.3,
        )
        body_yaw = float(self.backend.get_present_head_joint_positions()[0])
        body_end = (
            round(center + 58 * math.cos(body_yaw)),
            round(center - 58 * math.sin(body_yaw)),
        )
        cv2.arrowedLine(
            frame,
            (center, center),
            body_end,
            (255, 180, 60),
            3,
            tipLength=0.25,
        )
        marker = (
            round(center + radius * math.cos(self.marker_azimuth)),
            round(center - radius * math.sin(self.marker_azimuth)),
        )
        cv2.circle(frame, marker, 14, (0, 0, 255), -1)
        cv2.circle(frame, marker, 16, (255, 255, 255), 2)
        labels = (
            ("front", (size - 92, center - 10)),
            ("rear", (8, center - 10)),
            ("left", (center - 20, 28)),
            ("right", (center - 24, size - 16)),
        )
        for label, position in labels:
            cv2.putText(
                frame,
                label,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "drag gradually | a/d: 5deg | w/s: height",
            (12, size - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"body yaw={math.degrees(body_yaw):+.1f}deg (limit +/-160deg)",
            (12, size - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 60),
            1,
            cv2.LINE_AA,
        )
        return frame

    def observe(self, dt: float) -> MarkerDetection | None:
        """Render, detect, and update only the persistent absolute reference."""
        frame = self.render()
        self.last_frame_rgb = frame
        detection = detect_red_marker(frame)
        self.last_detection = detection
        if self.mode == "vision":
            if detection is None:
                self.last_reference_update = self.reference.freeze()
            else:
                self.last_reference_update = self.reference.update(
                    error_x=detection.error_x,
                    error_y=detection.error_y,
                    dt=dt,
                )
        return detection

    def _oracle_target(self) -> tuple[float, float, float]:
        camera_origin, _rotation = self._camera_pose_robot_frame()
        ray = self.marker_position - camera_origin
        norm = float(np.linalg.norm(ray))
        if norm <= 1e-9:
            target = self.reference.target
            return target.x, target.y, target.z
        ray /= norm
        reference_origin = np.array(
            [
                self.look_at_sphere.origin_x,
                self.look_at_sphere.origin_y,
                self.look_at_sphere.origin_z,
            ],
            dtype=np.float64,
        )
        target_world = reference_origin + self.look_at_sphere.distance * ray
        return (
            float(target_world[0]),
            float(target_world[1]),
            float(target_world[2]),
        )

    def _metric_target(self) -> tuple[float, float, float]:
        if self.mode == "oracle":
            return self._oracle_target()
        target = self.reference.target
        return target.x, target.y, target.z

    def step(self) -> dict[str, Any]:
        """Submit one metric target, advance the approved controller, and step physics."""
        target = self._metric_target()
        self.servo.submit_look_at(
            TrackingLookAtTarget(
                x=target[0],
                y=target[1],
                z=target[2],
                timestamp=time.time(),
                frame_id=self.frame_id,
            )
        )
        self.frame_id += 1
        commanded = self.servo.step(dt=CONTROL_DT)
        latest = self.servo.telemetry.latest()
        if latest is None:
            raise RuntimeError("look-at controller produced no telemetry record")
        record: dict[str, Any] = dict(latest)
        reason = str(record["reason"])
        self.reasons.add(reason)
        self.ik_failures += int(bool(record.get("ik_failed")))
        self.guard_hits += len(record.get("limit_hits", []))
        self.profile_position_hits += sum(
            hit.get("source") == "profile_position"
            for hit in record.get("profile_limit_hits", [])
        )
        if commanded:
            self.command_count += 1
            assert self.backend.target_head_joint_positions is not None
            self.backend.data.ctrl[:7] = self.backend.target_head_joint_positions

        for _ in range(round(CONTROL_DT / self.backend.model.opt.timestep)):
            mujoco.mj_step(self.backend.model, self.backend.data)
        self._refresh_backend_state()
        direction = self.reference.direction
        norm_error = abs(math.dist(direction, (0.0, 0.0, 0.0)) - 1.0)
        elevation = abs(math.asin(max(-1.0, min(1.0, float(direction[2])))))
        self.maximum_direction_norm_error = max(
            self.maximum_direction_norm_error,
            norm_error,
        )
        self.maximum_abs_elevation = max(self.maximum_abs_elevation, elevation)
        return record

    def annotated_frame(self) -> npt.NDArray[np.uint8]:
        """Return a BGR eye frame with controller state and input overlays."""
        if self.last_frame_rgb is None:
            self.last_frame_rgb = self.render()
        frame = np.asarray(
            cv2.cvtColor(self.last_frame_rgb, cv2.COLOR_RGB2BGR),
            dtype=np.uint8,
        )
        if self.last_detection is not None:
            center = (round(self.last_detection.u), round(self.last_detection.v))
            cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cx, cy = self.width // 2, self.height // 2
        half_width = round(self.width * self.reference.config.center_enter / 2.0)
        half_height = round(self.height * self.reference.config.center_enter / 2.0)
        cv2.rectangle(
            frame,
            (cx - half_width, cy - half_height),
            (cx + half_width, cy + half_height),
            (255, 255, 255),
            1,
        )
        target = self._metric_target()
        detection_text = (
            "missing"
            if self.last_detection is None
            else f"e=({self.last_detection.error_x:+.3f}, "
            f"{self.last_detection.error_y:+.3f})"
        )
        lines = (
            f"mode={self.mode} {detection_text}",
            f"look_at=({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})",
            f"ball az={math.degrees(self.marker_azimuth):+.1f}deg "
            f"height={self.marker_height:+.2f}m | v/o: mode | r: reset | q: quit",
        )
        for row, text in enumerate(lines, start=1):
            cv2.putText(
                frame,
                text,
                (12, 24 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return frame


def run_scenario(
    name: str,
    marker_azimuth: float,
    marker_height: float,
    *,
    maximum_seconds: float = 5.0,
) -> ScenarioResult:
    """Run one deterministic rendered-pixel vision scenario."""
    harness = MujocoRedTargetHarness()
    harness.set_marker_orbit(marker_azimuth, marker_height)
    centered_at: float | None = None
    centered_ticks = 0
    final_error: tuple[float, float] | None = None
    tick = -1
    try:
        for tick in range(round(maximum_seconds / CONTROL_DT)):
            if tick % SENSOR_TICKS == 0:
                detection = harness.observe(CONTROL_DT * SENSOR_TICKS)
                if detection is not None:
                    final_error = (detection.error_x, detection.error_y)
            record = harness.step()
            if record["reason"] != "commanded":
                break
            if final_error is not None and all(
                abs(value) <= harness.reference.config.center_enter
                for value in final_error
            ):
                centered_ticks += 1
                if centered_at is None:
                    centered_at = tick * CONTROL_DT
            else:
                centered_ticks = 0
            if centered_ticks * CONTROL_DT >= 0.5:
                break

        reference = harness.reference.target
        return ScenarioResult(
            name=name,
            centered_at_s=centered_at,
            held_center_s=centered_ticks * CONTROL_DT,
            final_error=final_error,
            control_ticks=tick + 1,
            rendered_frames=harness.rendered_frames,
            command_count=harness.command_count,
            ik_failures=harness.ik_failures,
            guard_hits=harness.guard_hits,
            profile_position_hits=harness.profile_position_hits,
            maximum_direction_norm_error=harness.maximum_direction_norm_error,
            maximum_abs_elevation=harness.maximum_abs_elevation,
            final_target=(reference.x, reference.y, reference.z),
            reasons=tuple(sorted(harness.reasons)),
        )
    finally:
        harness.close()


def run_orbit_sweep(
    end_degrees: float,
    *,
    step_degrees: float = 10.0,
    maximum_step_seconds: float = 4.0,
) -> OrbitSweepResult:
    """Move the rendered marker incrementally and retain visual/body evidence."""
    if not math.isfinite(end_degrees) or end_degrees == 0.0:
        raise ValueError("end_degrees must be finite and non-zero")
    if not math.isfinite(step_degrees) or step_degrees <= 0.0:
        raise ValueError("step_degrees must be finite and positive")
    if not math.isfinite(maximum_step_seconds) or maximum_step_seconds <= 0.0:
        raise ValueError("maximum_step_seconds must be finite and positive")

    harness = MujocoRedTargetHarness()
    direction = math.copysign(1.0, end_degrees)
    current_degrees = 0.0
    reached_degrees = 0.0
    steps: list[OrbitStepResult] = []
    try:
        while abs(current_degrees) < abs(end_degrees):
            current_degrees = direction * min(
                abs(end_degrees),
                abs(current_degrees) + step_degrees,
            )
            harness.set_marker_orbit(math.radians(current_degrees), 0.0)
            centered_at: float | None = None
            centered_ticks = 0
            marker_visible = True
            for tick in range(round(maximum_step_seconds / CONTROL_DT)):
                if tick % SENSOR_TICKS == 0:
                    detection = harness.observe(CONTROL_DT * SENSOR_TICKS)
                    if detection is None:
                        marker_visible = False
                        break
                else:
                    detection = harness.last_detection
                record = harness.step()
                if record["reason"] != "commanded":
                    break
                assert detection is not None
                if (
                    abs(detection.error_x) <= harness.reference.config.center_enter
                    and abs(detection.error_y) <= harness.reference.config.center_enter
                ):
                    centered_ticks += 1
                    if centered_at is None:
                        centered_at = tick * CONTROL_DT
                else:
                    centered_ticks = 0
                if centered_ticks * CONTROL_DT >= 0.5:
                    break

            centered_hold = centered_ticks * CONTROL_DT
            body_yaw = float(harness.backend.get_present_head_joint_positions()[0])
            steps.append(
                OrbitStepResult(
                    azimuth_degrees=current_degrees,
                    marker_visible=marker_visible,
                    centered_at_s=centered_at,
                    held_center_s=centered_hold,
                    body_yaw_degrees=math.degrees(body_yaw),
                )
            )
            if not marker_visible or centered_at is None or centered_hold < 0.5:
                break
            reached_degrees = current_degrees

        return OrbitSweepResult(
            requested_end_degrees=end_degrees,
            reached_degrees=reached_degrees,
            steps=tuple(steps),
            ik_failures=harness.ik_failures,
            guard_hits=harness.guard_hits,
            profile_position_hits=harness.profile_position_hits,
            maximum_direction_norm_error=harness.maximum_direction_norm_error,
            maximum_abs_elevation=harness.maximum_abs_elevation,
            reasons=tuple(sorted(harness.reasons)),
        )
    finally:
        harness.close()


def run_interactive(
    orbit_radius: float = DEFAULT_ORBIT_RADIUS,
    vertical_range: float = DEFAULT_VERTICAL_RANGE,
) -> None:
    """Open the eye camera and direct robot-centered marker orbit control."""
    harness = MujocoRedTargetHarness(
        width=1280,
        height=720,
        orbit_radius=orbit_radius,
        vertical_range=vertical_range,
    )
    window = "Reachy MuJoCo red-target tracking"
    studio_window = "Reachy MuJoCo third-person view"
    orbit_window = "Reachy MuJoCo ball orbit"
    dragging = False

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        if dragging or event == cv2.EVENT_LBUTTONDOWN:
            harness.move_marker_from_orbit_pad(float(x), float(y), ORBIT_PAD_SIZE)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(orbit_window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(orbit_window, mouse)
    sensor_tick = 0
    next_tick = time.monotonic()
    try:
        while True:
            if sensor_tick % SENSOR_TICKS == 0:
                harness.observe(CONTROL_DT * SENSOR_TICKS)
            harness.step()
            cv2.imshow(window, harness.annotated_frame())
            studio = cv2.cvtColor(harness.render_studio(), cv2.COLOR_RGB2BGR)
            cv2.imshow(studio_window, studio)
            cv2.imshow(orbit_window, harness.orbit_pad_frame())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                harness.reset()
            elif key == ord("v"):
                harness.mode = "vision"
                harness.reference.reset()
            elif key == ord("o"):
                harness.mode = "oracle"
                harness.reference.reset()
            elif key == ord("w"):
                harness.adjust_marker_height(0.02)
            elif key == ord("s"):
                harness.adjust_marker_height(-0.02)
            elif key == ord("a"):
                harness.set_marker_orbit(
                    harness.marker_azimuth + math.radians(5.0),
                    harness.marker_height,
                )
            elif key == ord("d"):
                harness.set_marker_orbit(
                    harness.marker_azimuth - math.radians(5.0),
                    harness.marker_height,
                )
            sensor_tick += 1
            next_tick += CONTROL_DT
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        harness.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Run the interactive tool or one deterministic scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--orbit-sweep", type=float)
    parser.add_argument(
        "--orbit-radius",
        "--radius",
        dest="orbit_radius",
        type=float,
        default=DEFAULT_ORBIT_RADIUS,
    )
    parser.add_argument(
        "--vertical-range",
        type=float,
        default=DEFAULT_VERTICAL_RANGE,
    )
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    selected_runs = sum(
        (
            args.scenario is not None,
            args.suite,
            args.orbit_sweep is not None,
        )
    )
    if selected_runs > 1:
        parser.error("choose only one of --scenario, --suite, or --orbit-sweep")
    if selected_runs == 0:
        run_interactive(
            orbit_radius=args.orbit_radius,
            vertical_range=args.vertical_range,
        )
        return
    if args.orbit_sweep is not None:
        data: Any = asdict(run_orbit_sweep(args.orbit_sweep))
    elif args.suite:
        results = [
            run_scenario(name, marker_azimuth, marker_height)
            for name, (marker_azimuth, marker_height) in SCENARIOS.items()
        ]
        data = {"scenarios": [asdict(result) for result in results]}
    else:
        assert args.scenario is not None
        data = asdict(run_scenario(args.scenario, *SCENARIOS[args.scenario]))
    payload = json.dumps(data, indent=2) + "\n"
    if args.artifact is not None:
        args.artifact.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
