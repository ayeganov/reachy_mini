"""Robot-side visual servo controller for 2D detections.

This module keeps perception and actuation separated: a remote computer may send
2D detections, but target conversion and motor updates run locally beside the daemon.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from .config import VisualServoConfig
from .joint_motion import (
    JointCommandSafetyGuard,
    LookAtJointCommandProfile,
    _finite_joint_vector,
)
from .telemetry import VisualServoTelemetryBuffer, finite_json_value

if TYPE_CHECKING:
    from reachy_mini.daemon.backend.abstract import Backend


R_HEAD_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TrackingDetection:
    """A single 2D detection from an external perception model."""

    u: float
    v: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    frame_id: int | None = None
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class TrackingLookAtTarget:
    """A metric 3D look-at target in the robot/world frame."""

    x: float
    y: float
    z: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    frame_id: int | None = None


@dataclass(frozen=True)
class JointTargetTelemetry:
    """IK metadata for one desired joint target."""

    joints: npt.NDArray[np.float64] | None
    ik_target: npt.NDArray[np.float64] | None
    ik_joints: npt.NDArray[np.float64] | None
    ik_failed: bool


class DetectionBuffer:
    """Thread-safe latest-only detection buffer."""

    def __init__(self) -> None:
        """Initialize the buffer."""
        self._lock = threading.Lock()
        self._latest: TrackingDetection | None = None
        self.accepted_count = 0

    def submit(self, detection: TrackingDetection) -> None:
        """Replace any pending detection with the newest one."""
        with self._lock:
            self._latest = detection
            self.accepted_count += 1

    def latest(self) -> TrackingDetection | None:
        """Return the most recently submitted detection."""
        with self._lock:
            return self._latest

    def fresh(
        self,
        config: VisualServoConfig,
        now: float | None = None,
    ) -> TrackingDetection | None:
        """Return the latest usable detection, or None if stale/low-confidence."""
        detection = self.latest()
        if detection is None:
            return None

        now = time.time() if now is None else now
        if now - detection.timestamp > config.max_detection_age:
            return None
        if detection.confidence < config.min_confidence:
            return None
        return detection


class LookAtTargetBuffer:
    """Thread-safe latest-only metric look-at target buffer."""

    def __init__(self) -> None:
        """Initialize the buffer."""
        self._lock = threading.Lock()
        self._latest: TrackingLookAtTarget | None = None
        self.accepted_count = 0

    def submit(self, target: TrackingLookAtTarget) -> None:
        """Replace any pending target with the newest one."""
        with self._lock:
            self._latest = target
            self.accepted_count += 1

    def latest(self) -> TrackingLookAtTarget | None:
        """Return the most recently submitted target."""
        with self._lock:
            return self._latest

    def fresh(
        self,
        config: VisualServoConfig,
        now: float | None = None,
    ) -> TrackingLookAtTarget | None:
        """Return the latest usable target, or None if stale/low-confidence."""
        target = self.latest()
        if target is None:
            return None

        now = time.time() if now is None else now
        if now - target.timestamp > config.max_detection_age:
            return None
        if target.confidence < config.min_confidence:
            return None
        return target


class VisualServoController:
    """Daemon-local latest-detection visual servo controller."""

    def __init__(
        self,
        backend: "Backend",
        config: VisualServoConfig | None = None,
    ) -> None:
        """Initialize the controller."""
        self.backend = backend
        self.config = config or VisualServoConfig()
        self.buffer = DetectionBuffer()
        self.look_at_buffer = LookAtTargetBuffer()
        self.look_at_profile = LookAtJointCommandProfile(config=self.config)
        self.look_at_guard = JointCommandSafetyGuard(config=self.config)
        self.telemetry = VisualServoTelemetryBuffer(
            capacity=self.config.telemetry_capacity
        )
        self._telemetry_sequence = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_command: npt.NDArray[np.float64] | None = None
        self._last_reason = "not_started"
        self._last_target_type: str | None = None
        self._last_command_path: str | None = None
        self._command_count = 0
        self._error: str | None = None
        self._metric_reference_pose: npt.NDArray[np.float64] | None = None
        self._detection_reference_pose: npt.NDArray[np.float64] | None = None
        self._processed_detection: TrackingDetection | None = None
        self._detection_target_cache: TrackingLookAtTarget | None = None
        self._previous_automatic_body_yaw: bool | None = None
        self._last_command_time: float | None = None
        self._motion_state = "idle"
        self._motion_fault: str | None = None

    @property
    def running(self) -> bool:
        """Return True if the servo thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def submit(self, detection: TrackingDetection) -> None:
        """Submit a detection from a remote perception model."""
        self.buffer.submit(detection)

    def submit_look_at(self, target: TrackingLookAtTarget) -> None:
        """Submit a metric look-at target from a local control surface."""
        self.look_at_buffer.submit(target)

    def start(self) -> None:
        """Start the local servo loop."""
        if self.running:
            return
        self._stop_event.clear()
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self._last_command_path = None
        self._metric_reference_pose = None
        self._detection_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self._last_command_time = None
        self._last_command = None
        self._motion_state = "idle"
        self._motion_fault = None
        if self.config.automatic_body_yaw and hasattr(
            self.backend.head_kinematics, "set_automatic_body_yaw"
        ):
            previous = getattr(self.backend.head_kinematics, "automatic_body_yaw", None)
            self._previous_automatic_body_yaw = (
                bool(previous) if isinstance(previous, bool) else None
            )
            self.backend.head_kinematics.set_automatic_body_yaw(True)
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reachy-visual-servo",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the local servo loop."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._last_reason = "stop_timeout"
                self._error = "Visual servo thread did not stop within 2.0s."
                return
        self._thread = None
        self._metric_reference_pose = None
        self._detection_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self._last_command_path = None
        self._last_command_time = None
        self._last_command = None
        self._motion_state = "idle"
        self._motion_fault = None
        self._restore_automatic_body_yaw()
        self._last_reason = "stopped"
        self._error = None

    def _restore_automatic_body_yaw(self) -> None:
        """Restore the kinematics yaw mode owned by this controller."""
        if self._previous_automatic_body_yaw is None:
            return
        if hasattr(self.backend.head_kinematics, "set_automatic_body_yaw"):
            self.backend.head_kinematics.set_automatic_body_yaw(
                self._previous_automatic_body_yaw
            )
        self._previous_automatic_body_yaw = None

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable status dictionary."""
        latest = self.buffer.latest()
        latest_look_at = self.look_at_buffer.latest()
        now = time.time()
        return {
            "running": self.running,
            "accepted_detections": self.buffer.accepted_count,
            "accepted_look_at_targets": self.look_at_buffer.accepted_count,
            "command_count": self._command_count,
            "last_reason": self._last_reason,
            "last_target_type": self._last_target_type,
            "last_detection_age": None
            if latest is None
            else max(0.0, now - latest.timestamp),
            "last_look_at_age": None
            if latest_look_at is None
            else max(0.0, now - latest_look_at.timestamp),
            "last_command": None
            if self._last_command is None
            else self._last_command.tolist(),
            "motion_state": self._motion_state,
            "motion_fault": self._motion_fault,
            "error": self._error,
        }

    def _new_telemetry_record(
        self,
        *,
        dt: float,
        target_type: str,
        reason: str,
        input_target: dict[str, Any] | None = None,
        start_monotonic: float | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        monotonic_now = time.monotonic()
        record = {
            "sequence": self._telemetry_sequence,
            "timestamp": now,
            "monotonic_timestamp": monotonic_now,
            "dt": dt,
            "target_type": target_type,
            "input_target": input_target,
            "look_at_target": None,
            "current_joints": None,
            "current_pose": None,
            "ik_target": None,
            "ik_joints": None,
            "profiled_command": None,
            "profile_limit_hits": [],
            "recovery": [],
            "final_command": None,
            "reason": reason,
            "motion_state": self._motion_state,
            "ik_failed": False,
            "limit_hits": [],
            "body_yaw": {
                "current": None,
                "ik_input": None,
                "automatic_enabled": self._automatic_body_yaw_state(),
            },
            "latency": {
                "target_age": None,
                "processing_duration": None
                if start_monotonic is None
                else monotonic_now - start_monotonic,
            },
            "backend": self._backend_telemetry(),
            "error": None,
        }
        self._telemetry_sequence += 1
        return record

    def _backend_telemetry(self) -> dict[str, Any]:
        ready = getattr(self.backend, "ready", None)
        is_set = getattr(ready, "is_set", None)
        return {
            "ready": bool(is_set()) if callable(is_set) else None,
            "error": getattr(self.backend, "error", None),
            "motor_control_mode": getattr(self.backend, "motor_control_mode", None),
        }

    def _automatic_body_yaw_state(self) -> bool | None:
        value = getattr(self.backend.head_kinematics, "automatic_body_yaw", None)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _detection_target(detection: TrackingDetection) -> dict[str, Any]:
        return {
            "kind": "detection",
            "u": detection.u,
            "v": detection.v,
            "timestamp": detection.timestamp,
            "confidence": detection.confidence,
            "frame_id": detection.frame_id,
            "width": detection.width,
            "height": detection.height,
        }

    @staticmethod
    def _look_at_target(target: TrackingLookAtTarget) -> dict[str, Any]:
        return {
            "kind": "look_at",
            "x": target.x,
            "y": target.y,
            "z": target.z,
            "timestamp": target.timestamp,
            "confidence": target.confidence,
            "frame_id": target.frame_id,
        }

    def _run_loop(self) -> None:
        period = 1.0 / self.config.control_frequency
        previous_start: float | None = None
        while not self._stop_event.is_set():
            start = time.monotonic()
            dt = period if previous_start is None else start - previous_start
            previous_start = start
            try:
                self.step(dt, use_command_elapsed=True)
            except Exception as exc:
                self._error = str(exc)
                self._last_reason = "error"
                record = self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="step_error",
                    start_monotonic=start,
                )
                record["error"] = str(exc)
                self.telemetry.append(record)
                log = logging.getLogger(__name__)
                log.exception("Visual servo step failed")

            sleep_time = max(0.0, period - (time.monotonic() - start))
            self._stop_event.wait(sleep_time)

    def step(
        self,
        dt: float | None = None,
        *,
        use_command_elapsed: bool = False,
    ) -> bool:
        """Execute one servo step.

        Returns True when a motor target was produced.
        """
        dt = (1.0 / self.config.control_frequency) if dt is None else dt
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        start_monotonic = time.monotonic()
        if self._motion_fault is not None:
            self._last_reason = self._motion_fault
            return False
        if dt > 2.0 / self.config.control_frequency:
            if self._last_command is None:
                self._reset_motion_state()
            else:
                self._hold_committed_motion()
            self._last_reason = "control_stall"
            self.telemetry.append(
                self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="control_stall",
                    start_monotonic=start_monotonic,
                )
            )
            return False

        look_at = self.look_at_buffer.fresh(self.config)
        detection = None if look_at is not None else self.buffer.fresh(self.config)
        if look_at is None and detection is None:
            return self._step_without_target(
                dt,
                start_monotonic=start_monotonic,
                use_command_elapsed=use_command_elapsed,
            )

        target_type = "none"
        input_target: dict[str, Any] | None = None
        target_timestamp: float | None = None
        if look_at is not None:
            target_type = "look_at"
            input_target = self._look_at_target(look_at)
            target_timestamp = look_at.timestamp
        elif detection is not None:
            target_type = "detection"
            input_target = self._detection_target(detection)
            target_timestamp = detection.timestamp

        release_motion_guard = self._try_acquire_motion_guard()
        if release_motion_guard is None:
            self._last_reason = "move_running"
            self.telemetry.append(
                self._new_telemetry_record(
                    dt=dt,
                    target_type=target_type,
                    reason="move_running",
                    input_target=input_target,
                    start_monotonic=start_monotonic,
                )
            )
            return False

        try:
            try:
                current_joints = _finite_joint_vector(
                    self.backend.get_present_head_joint_positions(),
                    length=7,
                    name="current",
                )
            except ValueError:
                self._reset_motion_state()
                raise
            current_pose = np.array(
                self.backend.get_present_head_pose(), dtype=np.float64
            )
            if target_type != "none":
                self._last_target_type = target_type
            record = self._new_telemetry_record(
                dt=dt,
                target_type=target_type,
                reason="ik_failed",
                input_target=input_target,
                start_monotonic=start_monotonic,
            )
            body_yaw = float(current_joints[0])
            if self._last_command_path != target_type:
                self._metric_reference_pose = current_pose.copy()
            if look_at is not None:
                self._processed_detection = None
                self._detection_target_cache = None
                self._detection_reference_pose = None
                target = look_at
            else:
                assert detection is not None
                if self._detection_reference_pose is None:
                    self._detection_reference_pose = current_pose.copy()
                target = self._look_at_target_from_detection(
                    detection,
                    current_pose,
                    self._detection_reference_pose,
                )
            if self._metric_reference_pose is None:
                self._metric_reference_pose = current_pose.copy()
            ik_reference_pose = (
                self._metric_reference_pose
                if look_at is not None
                else self._detection_reference_pose
            )
            assert ik_reference_pose is not None
            target_result = self._ik_from_target_world_with_telemetry(
                target_world=np.array([target.x, target.y, target.z]),
                current_head_pose=ik_reference_pose,
                body_yaw=body_yaw,
            )

            record["look_at_target"] = self._look_at_target(target)
            record["current_joints"] = finite_json_value(current_joints)
            record["current_pose"] = finite_json_value(current_pose)
            record["ik_target"] = finite_json_value(target_result.ik_target)
            record["ik_joints"] = finite_json_value(target_result.ik_joints)
            record["ik_failed"] = target_result.ik_failed
            record["body_yaw"]["current"] = body_yaw
            record["body_yaw"]["ik_input"] = body_yaw
            record["latency"]["target_age"] = (
                None
                if target_timestamp is None
                else max(0.0, time.time() - target_timestamp)
            )
            if target_result.joints is None:
                if self._last_command is None:
                    self._reset_motion_state()
                    record["reason"] = "ik_failed"
                    self._last_reason = "ik_failed"
                    record["motion_state"] = self._motion_state
                    record["latency"]["processing_duration"] = (
                        time.monotonic() - start_monotonic
                    )
                    self.telemetry.append(record)
                    return False
                command_time = time.monotonic()
                command_dt = dt
                if use_command_elapsed and self._last_command_time is not None:
                    command_dt = command_time - self._last_command_time
                if command_dt > 2.0 / self.config.control_frequency:
                    self._hold_committed_motion()
                    record["dt"] = command_dt
                    record["monotonic_timestamp"] = command_time
                    record["reason"] = "control_stall"
                    record["motion_state"] = self._motion_state
                    record["latency"]["processing_duration"] = (
                        time.monotonic() - start_monotonic
                    )
                    self._last_reason = "control_stall"
                    self.telemetry.append(record)
                    return False
                self._reset_target_state()
                command, profile_hits = self.look_at_profile.stop_with_telemetry(
                    current_joints, command_dt
                )
                record["dt"] = command_dt
                record["monotonic_timestamp"] = command_time
                record["profiled_command"] = finite_json_value(command)
                record["profile_limit_hits"] = profile_hits
                stationary = self.look_at_profile.stationary
                return self._write_profiled_command(
                    command=command,
                    current_joints=current_joints,
                    command_dt=command_dt,
                    command_time=command_time,
                    record=record,
                    target_type="none",
                    success_reason=(
                        "holding_ik_failed" if stationary else "stopping_ik_failed"
                    ),
                    motion_state=(
                        "holding_ik_failed" if stationary else "stopping_ik_failed"
                    ),
                    start_monotonic=start_monotonic,
                )

            desired_joints = np.array(target_result.joints, dtype=np.float64)
            command_time = time.monotonic()
            command_dt = dt
            if use_command_elapsed and self._last_command_time is not None:
                command_dt = command_time - self._last_command_time
            if command_dt > 2.0 / self.config.control_frequency:
                if self._last_command is None:
                    self._reset_motion_state()
                else:
                    self._hold_committed_motion()
                record["dt"] = command_dt
                record["monotonic_timestamp"] = command_time
                record["reason"] = "control_stall"
                record["latency"]["processing_duration"] = (
                    time.monotonic() - start_monotonic
                )
                self._last_reason = "control_stall"
                self.telemetry.append(record)
                return False
            record["dt"] = command_dt
            record["monotonic_timestamp"] = command_time
            desired_joints, profile_hits = self.look_at_profile.update_with_telemetry(
                desired=desired_joints,
                current=current_joints,
                dt=command_dt,
            )
            record["profiled_command"] = finite_json_value(desired_joints)
            record["profile_limit_hits"] = profile_hits
            return self._write_profiled_command(
                command=desired_joints,
                current_joints=current_joints,
                command_dt=command_dt,
                command_time=command_time,
                record=record,
                target_type=target_type,
                success_reason="commanded",
                motion_state="tracking",
                start_monotonic=start_monotonic,
            )
        finally:
            release_motion_guard()

    def _step_without_target(
        self,
        dt: float,
        *,
        start_monotonic: float,
        use_command_elapsed: bool,
    ) -> bool:
        """Bring committed command motion to a bounded hold without a target."""
        self._reset_target_state()
        if self._last_command is None:
            self._motion_state = "idle"
            self._last_reason = "no_fresh_detection"
            self.telemetry.append(
                self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="no_fresh_detection",
                    start_monotonic=start_monotonic,
                )
            )
            return False

        release_motion_guard = self._try_acquire_motion_guard()
        if release_motion_guard is None:
            self._last_reason = "move_running"
            self.telemetry.append(
                self._new_telemetry_record(
                    dt=dt,
                    target_type="none",
                    reason="move_running",
                    start_monotonic=start_monotonic,
                )
            )
            return False

        try:
            try:
                current_joints = _finite_joint_vector(
                    self.backend.get_present_head_joint_positions(),
                    length=7,
                    name="current",
                )
            except ValueError:
                self._reset_motion_state()
                raise
            command_time = time.monotonic()
            command_dt = dt
            if use_command_elapsed and self._last_command_time is not None:
                command_dt = command_time - self._last_command_time
            if command_dt > 2.0 / self.config.control_frequency:
                self._hold_committed_motion()
                self._last_reason = "control_stall"
                record = self._new_telemetry_record(
                    dt=command_dt,
                    target_type="none",
                    reason="control_stall",
                    start_monotonic=start_monotonic,
                )
                record["monotonic_timestamp"] = command_time
                record["current_joints"] = finite_json_value(current_joints)
                record["motion_state"] = self._motion_state
                self.telemetry.append(record)
                return False
            record = self._new_telemetry_record(
                dt=command_dt,
                target_type="none",
                reason="stopping_no_target",
                start_monotonic=start_monotonic,
            )
            record["monotonic_timestamp"] = command_time
            record["current_joints"] = finite_json_value(current_joints)
            command, profile_hits = self.look_at_profile.stop_with_telemetry(
                current_joints, command_dt
            )
            record["profiled_command"] = finite_json_value(command)
            record["profile_limit_hits"] = profile_hits
            stationary = self.look_at_profile.stationary
            return self._write_profiled_command(
                command=command,
                current_joints=current_joints,
                command_dt=command_dt,
                command_time=command_time,
                record=record,
                target_type="none",
                success_reason=(
                    "holding_no_target" if stationary else "stopping_no_target"
                ),
                motion_state=(
                    "holding_no_target" if stationary else "stopping_no_target"
                ),
                start_monotonic=start_monotonic,
            )
        finally:
            release_motion_guard()

    def _write_profiled_command(
        self,
        *,
        command: npt.NDArray[np.float64],
        current_joints: npt.NDArray[np.float64],
        command_dt: float,
        command_time: float,
        record: dict[str, Any],
        target_type: str,
        success_reason: str,
        motion_state: str,
        start_monotonic: float,
    ) -> bool:
        """Guard, write, and commit one profile-generated command."""
        guard_hits, guard_velocity, guard_acceleration, recovery = (
            self.look_at_guard.check_with_telemetry(
                command=command,
                current=current_joints,
                dt=command_dt,
            )
        )
        record["limit_hits"] = guard_hits
        record["recovery"] = recovery
        if guard_hits:
            self._motion_state = "fault"
            self._motion_fault = "safety_rejected"
            record["motion_state"] = self._motion_state
            record["reason"] = "safety_rejected"
            record["latency"]["processing_duration"] = (
                time.monotonic() - start_monotonic
            )
            self._last_reason = "safety_rejected"
            self.telemetry.append(record)
            return False
        try:
            self.backend.set_target_head_joint_positions(command)
        except Exception:
            self._reset_motion_state()
            raise
        self.look_at_guard.commit(command, guard_velocity, guard_acceleration)
        self._last_command_path = None if target_type == "none" else target_type
        self._last_command_time = command_time
        self._last_command = command.copy()
        self._command_count += 1
        self._motion_state = "recovering" if recovery else motion_state
        reason = "recovering" if recovery else success_reason
        self._last_reason = reason
        record["final_command"] = finite_json_value(command)
        record["reason"] = reason
        record["motion_state"] = self._motion_state
        record["latency"]["processing_duration"] = time.monotonic() - start_monotonic
        self.telemetry.append(record)
        return True

    def _reset_target_state(self) -> None:
        """Clear stale perception state without erasing committed motion."""
        self._metric_reference_pose = None
        self._detection_reference_pose = None
        self._processed_detection = None
        self._detection_target_cache = None
        self._last_command_path = None

    def _hold_committed_motion(self) -> None:
        """Retain the last backend command as a stationary motion anchor."""
        if self._last_command is None:
            self._reset_motion_state()
            return
        self._reset_target_state()
        self.look_at_profile.hold(self._last_command)
        self.look_at_guard.reset(self._last_command)
        self._last_command_time = None
        self._motion_state = "holding_no_target"
        self._motion_fault = None

    def _reset_motion_state(self) -> None:
        """Reset controller-owned motion state without backend I/O."""
        self._reset_target_state()
        self.look_at_profile.reset()
        self.look_at_guard.reset()
        self._last_command_time = None
        self._last_command = None
        self._motion_state = "idle"
        self._motion_fault = None

    def _look_at_target_from_detection(
        self,
        detection: TrackingDetection,
        current_head_pose: npt.NDArray[np.float64],
        reference_head_pose: npt.NDArray[np.float64],
    ) -> TrackingLookAtTarget:
        """Convert one image error and the measured gaze into a look-at point."""
        if detection is self._processed_detection:
            assert self._detection_target_cache is not None
            return self._detection_target_cache
        pixel = np.array([detection.u, detection.v], dtype=np.float64)
        center = np.array(
            [float(detection.width) / 2.0, float(detection.height) / 2.0],
            dtype=np.float64,
        )
        error = (pixel - center) / center
        if bool(np.all(np.abs(error) <= 0.03)):
            error[:] = 0.0
        ray_camera = np.array(
            [
                error[0] * np.tan(self.config.image_horizontal_fov / 2.0),
                error[1] * np.tan(self.config.image_vertical_fov / 2.0),
                1.0,
            ],
            dtype=np.float64,
        )
        ray_world = current_head_pose[:3, :3] @ R_HEAD_CAM @ ray_camera
        azimuth = float(np.arctan2(ray_world[1], ray_world[0]))
        upward_limit = self.config.image_error_upward_elevation_limit
        downward_limit = self.config.image_error_downward_elevation_limit
        elevation = float(
            np.clip(
                np.arctan2(ray_world[2], np.hypot(ray_world[0], ray_world[1])),
                -downward_limit,
                upward_limit,
            )
        )
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        target_world = reference_head_pose[:3, 3] + direction
        target = TrackingLookAtTarget(
            x=float(target_world[0]),
            y=float(target_world[1]),
            z=float(target_world[2]),
            timestamp=detection.timestamp,
            confidence=detection.confidence,
            frame_id=detection.frame_id,
        )
        self._processed_detection = detection
        self._detection_target_cache = target
        return target

    def _try_acquire_motion_guard(self) -> Callable[[], None] | None:
        """Acquire backend motion ownership for one servo step."""
        try_start_move = getattr(self.backend, "_try_start_move", None)
        end_move = getattr(self.backend, "_end_move", None)
        if callable(try_start_move) and callable(end_move):
            if not bool(try_start_move()):
                return None

            def release() -> None:
                end_move()

            return release

        if getattr(self.backend, "is_move_running", False):
            return None

        def noop() -> None:
            return None

        return noop

    def _ik_from_target_world_with_telemetry(
        self,
        target_world: npt.NDArray[np.float64],
        current_head_pose: npt.NDArray[np.float64],
        body_yaw: float,
    ) -> JointTargetTelemetry:
        target_pose = self._look_at_pose(
            current_head_pose=current_head_pose,
            target_world=target_world,
            up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        joints = self.backend.head_kinematics.ik(target_pose, body_yaw=body_yaw)
        if joints is None:
            return JointTargetTelemetry(
                joints=None,
                ik_target=target_pose,
                ik_joints=None,
                ik_failed=True,
            )
        joints_array = np.array(joints, dtype=np.float64)
        failed = not self._valid_joints(joints_array)
        return JointTargetTelemetry(
            joints=None if failed else joints_array,
            ik_target=target_pose,
            ik_joints=joints_array,
            ik_failed=failed,
        )

    @staticmethod
    def _valid_joints(joints: npt.NDArray[np.float64] | None) -> bool:
        return (
            joints is not None
            and joints.shape == (7,)
            and bool(np.all(np.isfinite(joints)))
        )

    @staticmethod
    def _look_at_pose(
        current_head_pose: npt.NDArray[np.float64],
        target_world: npt.NDArray[np.float64],
        up_hint: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        origin = current_head_pose[:3, 3]
        forward = target_world - origin
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            return current_head_pose.copy()
        x_axis = forward / forward_norm

        if up_hint is None:
            up_hint = current_head_pose[:3, 2]
        y_axis = np.cross(up_hint, x_axis)
        if np.linalg.norm(y_axis) < 1e-9:
            for fallback_up in (
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ):
                y_axis = np.cross(fallback_up, x_axis)
                if np.linalg.norm(y_axis) >= 1e-9:
                    break
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        pose[:3, 3] = origin

        # Ensure a numerically valid rotation matrix for scipy/IK consumers.
        pose[:3, :3] = R.from_matrix(pose[:3, :3]).as_matrix()
        return pose
