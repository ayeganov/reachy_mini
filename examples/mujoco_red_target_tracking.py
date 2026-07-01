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
    AbsoluteLookAtReferenceController,
    ImageErrorReferenceConfig,
    LookAtPlane,
    ReferenceUpdate,
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
RED = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
APPROVED_MOTION = {
    "smoothing_alpha": 1.0,
    "joint_safety_margin": 0.1745329252,
    "max_joint_velocity": 0.6,
    "max_joint_acceleration": 2.4,
    "max_joint_jerk": 16.0,
    "look_at_profile_response_hz": 2.0,
}
SCENARIOS = {
    "center": (0.0, 0.0),
    "left": (0.2, 0.0),
    "right": (-0.2, 0.0),
    "top": (0.0, 0.2),
    "bottom": (0.0, -0.2),
    "top_left": (0.1414, 0.1414),
    "top_right": (-0.1414, 0.1414),
    "bottom_left": (0.1414, -0.1414),
    "bottom_right": (-0.1414, -0.1414),
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
    maximum_target_radius: float
    final_target: tuple[float, float, float]
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
        target_radius: float = 0.2,
    ) -> None:
        """Initialize a neutral simulated robot and its eye-camera renderer."""
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
        self.studio_camera.lookat[:] = [0.25, 0.0, 0.15]
        self.studio_camera.distance = 1.0
        self.studio_camera.azimuth = 160.0
        self.studio_camera.elevation = -15.0
        self.studio_renderer: mujoco.Renderer | None = None

        head_pose = self.backend.get_present_head_pose()
        self.look_at_plane = LookAtPlane(
            center_z=float(head_pose[2, 3]),
            radius=target_radius,
        )
        self.reference = AbsoluteLookAtReferenceController(
            plane=self.look_at_plane,
            config=ImageErrorReferenceConfig(),
        )
        self.servo = VisualServoController(
            backend=self.backend,
            config=VisualServoConfig(
                smoothing_alpha=APPROVED_MOTION["smoothing_alpha"],
                joint_safety_margin=APPROVED_MOTION["joint_safety_margin"],
                max_joint_velocity=APPROVED_MOTION["max_joint_velocity"],
                max_joint_acceleration=APPROVED_MOTION["max_joint_acceleration"],
                max_joint_jerk=APPROVED_MOTION["max_joint_jerk"],
                look_at_profile_response_hz=APPROVED_MOTION[
                    "look_at_profile_response_hz"
                ],
            ),
        )

        self.marker_plane_center = self._neutral_marker_position()
        self.marker_position = self.marker_plane_center.copy()
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
        self.maximum_target_radius = 0.0

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
        if forward[0] <= 1e-9:
            raise RuntimeError("eye camera does not face the positive target plane")
        distance = (self.look_at_plane.distance - origin[0]) / forward[0]
        return np.asarray(origin + distance * forward, dtype=np.float64)

    def reset(self) -> None:
        """Reset marker and persistent reference without changing motion limits."""
        self.marker_position = self.marker_plane_center.copy()
        self.reference.reset()
        self.last_reference_update = self.reference.freeze("reset")
        self.last_detection = None

    def set_marker_offset(self, y: float, z: float) -> None:
        """Place the marker at a bounded offset from its neutral camera ray."""
        length = math.hypot(y, z)
        if length > self.look_at_plane.radius:
            scale = self.look_at_plane.radius / length
            y *= scale
            z *= scale
        self.marker_position = self.marker_plane_center + np.array([0.0, y, z])

    def move_marker_from_pixel(self, u: float, v: float) -> None:
        """Unproject an eye-frame cursor onto the marker's fixed robot-frame plane."""
        origin, rotation = self._camera_pose_robot_frame()
        fovy = math.radians(float(self.backend.model.cam_fovy[self.camera_id]))
        focal = (self.height / 2.0) / math.tan(fovy / 2.0)
        ray_camera = np.array(
            [
                (u - self.width / 2.0) / focal,
                -(v - self.height / 2.0) / focal,
                -1.0,
            ]
        )
        ray_world = rotation @ ray_camera
        if abs(float(ray_world[0])) <= 1e-9:
            return
        distance = (self.look_at_plane.distance - origin[0]) / ray_world[0]
        if distance <= 0.0:
            return
        point = origin + distance * ray_world
        self.set_marker_offset(
            y=float(point[1] - self.marker_plane_center[1]),
            z=float(point[2] - self.marker_plane_center[2]),
        )

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
        head_origin = self.backend.get_present_head_pose()[:3, 3]
        if ray[0] <= 1e-9:
            target = self.reference.target
            return target.x, target.y, target.z
        distance = (self.look_at_plane.distance - head_origin[0]) / ray[0]
        target_world = head_origin + distance * ray
        offset_y = float(target_world[1] - self.look_at_plane.center_y)
        offset_z = float(target_world[2] - self.look_at_plane.center_z)
        radius = math.hypot(offset_y, offset_z)
        if radius > self.look_at_plane.radius:
            scale = self.look_at_plane.radius / radius
            offset_y *= scale
            offset_z *= scale
        return (
            self.look_at_plane.distance,
            self.look_at_plane.center_y + offset_y,
            self.look_at_plane.center_z + offset_z,
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
        reference = self.reference.target
        radius = math.hypot(
            reference.y - self.look_at_plane.center_y,
            reference.z - self.look_at_plane.center_z,
        )
        self.maximum_target_radius = max(self.maximum_target_radius, radius)
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
            f"radius={self.look_at_plane.radius:.2f}m | drag: move marker | "
            "v: vision | o: oracle | r: reset | q: quit",
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
    marker_y: float,
    marker_z: float,
    *,
    maximum_seconds: float = 5.0,
) -> ScenarioResult:
    """Run one deterministic rendered-pixel vision scenario."""
    harness = MujocoRedTargetHarness()
    harness.set_marker_offset(marker_y, marker_z)
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
            maximum_target_radius=harness.maximum_target_radius,
            final_target=(reference.x, reference.y, reference.z),
            reasons=tuple(sorted(harness.reasons)),
        )
    finally:
        harness.close()


def run_interactive(target_radius: float = 0.2) -> None:
    """Open the rendered eye camera and allow direct marker dragging."""
    harness = MujocoRedTargetHarness(
        width=1280,
        height=720,
        target_radius=target_radius,
    )
    window = "Reachy MuJoCo red-target tracking"
    studio_window = "Reachy MuJoCo third-person view"
    dragging = False

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        if dragging or event == cv2.EVENT_LBUTTONDOWN:
            harness.move_marker_from_pixel(float(x), float(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
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
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    if args.scenario is None and not args.suite:
        run_interactive(target_radius=args.radius)
        return
    if args.suite:
        results = [
            run_scenario(name, marker_y, marker_z)
            for name, (marker_y, marker_z) in SCENARIOS.items()
        ]
        data: Any = {"scenarios": [asdict(result) for result in results]}
    else:
        assert args.scenario is not None
        data = asdict(run_scenario(args.scenario, *SCENARIOS[args.scenario]))
    payload = json.dumps(data, indent=2) + "\n"
    if args.artifact is not None:
        args.artifact.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
