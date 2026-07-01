"""Preview or follow one yellow box through the metric look-at API."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import numpy.typing as npt

from reachy_mini.daemon.tracking.look_at_reference import (
    ImageErrorReferenceConfig,
    LookAtReference,
    LookAtSphere,
    ReferenceUpdate,
    SphericalLookAtReferenceController,
)
from reachy_mini.daemon.tracking.telemetry import dump_jsonl
from reachy_mini.media.receivers.zeromq_client import ZeroMQClient

DEFAULT_BASE_URL = "http://reachy-mini.local:8017/api"
APPROVED_TRACKING_CONFIG = {
    "max_detection_age": 2.0,
    "smoothing_alpha": 1.0,
    "joint_safety_margin": 0.1745329252,
    "max_joint_velocity": 0.60,
    "max_joint_acceleration": 2.40,
    "max_joint_jerk": 16.0,
    "look_at_profile_response_hz": 2.0,
}
CONTROL_PERIOD = 0.04


@dataclass(frozen=True)
class YellowBoxDetectorConfig:
    """HSV and area thresholds for the hardware target."""

    hue_low: int = 18
    hue_high: int = 42
    saturation_low: int = 170
    value_low: int = 120
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.5
    morphology_size: int = 5

    def __post_init__(self) -> None:
        """Validate detector thresholds."""
        if not 0 <= self.hue_low <= self.hue_high <= 179:
            raise ValueError("yellow hue range must be within [0, 179]")
        if not 0 <= self.saturation_low <= 255:
            raise ValueError("saturation_low must be within [0, 255]")
        if not 0 <= self.value_low <= 255:
            raise ValueError("value_low must be within [0, 255]")
        if not 0.0 < self.min_area_ratio < self.max_area_ratio <= 1.0:
            raise ValueError("area ratios must satisfy 0 < min < max <= 1")
        if self.morphology_size < 1 or self.morphology_size % 2 == 0:
            raise ValueError("morphology_size must be a positive odd integer")


@dataclass(frozen=True)
class YellowBoxDetection:
    """Selected yellow component in image coordinates."""

    u: float
    v: float
    bounding_box: tuple[int, int, int, int]
    area_ratio: float

    def normalized_error(self, width: int, height: int) -> tuple[float, float]:
        """Return centroid error normalized by image half-width and half-height."""
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (
            (self.u - width / 2.0) / (width / 2.0),
            (self.v - height / 2.0) / (height / 2.0),
        )


@dataclass(frozen=True)
class FollowDecision:
    """Reference-controller result for one detector observation."""

    update: ReferenceUpdate


class YellowBoxFollower:
    """Apply any available centroid directly to the spherical reference."""

    def __init__(
        self,
        controller: SphericalLookAtReferenceController,
    ) -> None:
        """Initialize with no identity or acquisition state."""
        self.controller = controller

    def observe(
        self,
        detection: YellowBoxDetection | None,
        *,
        width: int,
        height: int,
        dt: float,
    ) -> FollowDecision:
        """Update the gaze reference directly from one frame."""
        if detection is None:
            return FollowDecision(
                update=self.controller.freeze("target_lost"),
            )

        error_x, error_y = detection.normalized_error(width, height)
        update = self.controller.update(error_x=error_x, error_y=error_y, dt=dt)
        return FollowDecision(
            update=update,
        )


def detect_yellow_box(
    frame: npt.NDArray[np.uint8],
    config: YellowBoxDetectorConfig,
) -> tuple[YellowBoxDetection | None, npt.NDArray[np.uint8]]:
    """Select the largest yellow contour satisfying color and area thresholds."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must be a non-empty BGR image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([config.hue_low, config.saturation_low, config.value_low]),
        np.array([config.hue_high, 255, 255]),
    )
    kernel = np.ones(
        (config.morphology_size, config.morphology_size),
        dtype=np.uint8,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    typed_mask: npt.NDArray[np.uint8] = np.asarray(mask, dtype=np.uint8)

    frame_area = float(frame.shape[0] * frame.shape[1])
    candidates: list[tuple[float, YellowBoxDetection]] = []
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            continue
        detection = YellowBoxDetection(
            u=float(moments["m10"] / moments["m00"]),
            v=float(moments["m01"] / moments["m00"]),
            bounding_box=(x, y, width, height),
            area_ratio=area_ratio,
        )
        candidates.append((area, detection))
    if not candidates:
        return None, typed_mask
    return max(candidates, key=lambda candidate: candidate[0])[1], typed_mask


def annotate_frame(
    frame: npt.NDArray[np.uint8],
    detection: YellowBoxDetection | None,
    decision: FollowDecision | None,
    follow_enabled: bool,
) -> npt.NDArray[np.uint8]:
    """Draw detector and controller state for the operator gate."""
    annotated = frame.copy()
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    deadband = (round(width * 0.03 / 2.0), round(height * 0.03 / 2.0))
    cv2.rectangle(
        annotated,
        (center[0] - deadband[0], center[1] - deadband[1]),
        (center[0] + deadband[0], center[1] + deadband[1]),
        (255, 255, 255),
        2,
    )
    if detection is not None:
        x, y, box_width, box_height = detection.bounding_box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            3,
        )
        cv2.drawMarker(
            annotated,
            (round(detection.u), round(detection.v)),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
        )
        error_x, error_y = detection.normalized_error(width, height)
        detection_text = (
            f"yellow e=({error_x:+.3f},{error_y:+.3f}) area={detection.area_ratio:.3f}"
        )
    else:
        detection_text = "yellow target missing"
    reason = "preview" if decision is None else decision.update.reason
    lines = (
        f"mode={'FOLLOW' if follow_enabled else 'PREVIEW'} reason={reason}",
        detection_text,
        "q: stop safely",
    )
    for row, text in enumerate(lines, start=1):
        cv2.putText(
            annotated,
            text,
            (12, row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _request_json(
    method: str,
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    if not body:
        return {}
    result: Any = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError(f"{path} did not return an object")
    return result


def _neutral_origin(state: Mapping[str, Any]) -> tuple[float, float, float]:
    pose = state.get("head_pose")
    if not isinstance(pose, Mapping):
        raise ValueError("robot state does not include head_pose")
    translation = (
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
    )
    rotation = tuple(float(pose[key]) for key in ("roll", "pitch", "yaw"))
    body_yaw = float(state["body_yaw"])
    if max(abs(value) for value in translation) > 0.03:
        raise RuntimeError("head translation is not near neutral")
    if max(abs(value) for value in rotation) > 0.10:
        raise RuntimeError("head rotation is not near neutral")
    if abs(body_yaw) > 0.10:
        raise RuntimeError("body yaw is not near neutral")
    return translation


def _return_neutral(base_url: str, timeout: float) -> None:
    _request_json(
        "POST",
        base_url,
        "/move/goto",
        {
            "head_pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "antennas": [0.0, 0.0],
            "body_yaw": 0.0,
            "duration": 2.0,
            "interpolation": "minjerk",
        },
        timeout,
    )
    time.sleep(2.5)


def _target_payload(target: LookAtReference, frame_id: int) -> dict[str, Any]:
    return {
        "x": target.x,
        "y": target.y,
        "z": target.z,
        "confidence": 1.0,
        "frame_id": frame_id,
    }


class FixedRateLookAtSender:
    """Refresh the latest metric target without blocking perception or display."""

    def __init__(
        self,
        *,
        base_url: str,
        initial_target: LookAtReference,
        request_timeout: float,
        period: float = CONTROL_PERIOD,
    ) -> None:
        """Initialize a stopped latest-target sender."""
        if request_timeout <= 0.0:
            raise ValueError("request_timeout must be positive")
        if period <= 0.0:
            raise ValueError("period must be positive")
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.period = period
        self._target = initial_target
        self._target_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent_count = 0
        self.error_count = 0
        self.last_error: str | None = None
        self.latencies: list[float] = []

    def set_target(self, target: LookAtReference) -> None:
        """Replace the target used by the next sender tick."""
        with self._target_lock:
            self._target = target

    def start(self) -> None:
        """Start the sender thread."""
        if self._thread is not None:
            raise RuntimeError("look-at sender is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="yellow_box_look_at_sender",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the sender after any in-flight bounded request."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.request_timeout + 1.0)
            if self._thread.is_alive():
                self.error_count += 1
                self.last_error = "look-at sender did not stop"
            else:
                self._thread = None

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._target_lock:
                target = self._target
            started = time.monotonic()
            try:
                _request_json(
                    "POST",
                    self.base_url,
                    "/tracking/look_at",
                    _target_payload(target, self.sent_count + self.error_count),
                    timeout=self.request_timeout,
                )
                self.sent_count += 1
            except Exception as exc:
                self.error_count += 1
                self.last_error = str(exc)
            self.latencies.append(time.monotonic() - started)
            next_tick += self.period
            now = time.monotonic()
            if next_tick <= now:
                next_tick = now + self.period
            self._stop.wait(max(0.0, next_tick - now))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run preview or the explicitly enabled hardware follow loop."""
    base_url = args.base_url.rstrip("/")
    daemon_status = _request_json(
        "GET", base_url, "/daemon/status", timeout=args.timeout
    )
    motor_status = _request_json(
        "GET", base_url, "/motors/status", timeout=args.timeout
    )
    tracking_before = _request_json(
        "GET", base_url, "/tracking/status", timeout=args.timeout
    )
    state_before = _request_json(
        "GET",
        base_url,
        "/state/full?with_head_pose=true&with_head_joints=true&with_body_yaw=true",
        timeout=args.timeout,
    )
    if (
        daemon_status.get("state") != "running"
        or daemon_status.get("error") is not None
    ):
        raise RuntimeError("robot daemon is not healthy")
    if motor_status.get("mode") != "enabled":
        raise RuntimeError("robot motors are not enabled")
    if bool(tracking_before.get("running")):
        raise RuntimeError("tracking is already running")

    camera_host = args.camera_host or daemon_status.get("wlan_ip")
    if not isinstance(camera_host, str) or not camera_host:
        raise RuntimeError("robot daemon did not report a camera host")
    detector_config = YellowBoxDetectorConfig(
        hue_low=args.hue_low,
        hue_high=args.hue_high,
        saturation_low=args.saturation_low,
        value_low=args.value_low,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        morphology_size=args.morphology_size,
    )

    follower: YellowBoxFollower | None = None
    if args.follow:
        origin = _neutral_origin(state_before)
        follower = YellowBoxFollower(
            SphericalLookAtReferenceController(
                LookAtSphere(
                    origin_x=origin[0],
                    origin_y=origin[1],
                    origin_z=origin[2],
                ),
                ImageErrorReferenceConfig(),
            ),
        )

    client = ZeroMQClient(host=camera_host, log_level="WARNING")
    sender: FixedRateLookAtSender | None = None
    tracking_started = False
    telemetry: dict[str, Any] = {}
    frame_count = 0
    detection_count = 0
    control_tick_count = 0
    maximum_area_ratio = 0.0
    last_detection: YellowBoxDetection | None = None
    last_frame: npt.NDArray[np.uint8] | None = None
    last_annotated: npt.NDArray[np.uint8] | None = None
    last_mask: npt.NDArray[np.uint8] | None = None
    last_frame_timestamp: object = None
    last_control_frame_timestamp: object = None
    decision: FollowDecision | None = None
    started_at = time.monotonic()
    next_control_tick = started_at
    stopped_by = "duration"
    run_error: Exception | None = None
    output_prefix = args.output_prefix or Path(
        f"/tmp/reachy-yellow-box-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    try:
        if not client.start(wait_timeout=args.timeout):
            raise RuntimeError("remote camera stream did not start")
        if args.follow:
            _request_json(
                "POST",
                base_url,
                "/tracking/start",
                APPROVED_TRACKING_CONFIG,
                timeout=args.timeout,
            )
            tracking_started = True
            assert follower is not None
            sender = FixedRateLookAtSender(
                base_url=base_url,
                initial_target=follower.controller.target,
                request_timeout=min(args.timeout, 1.5),
            )
            sender.start()

        while args.duration <= 0.0 or time.monotonic() - started_at < args.duration:
            packet = client.get_frame_with_metadata()
            now = time.monotonic()
            new_frame = False
            if packet is not None:
                frame, metadata = packet
                timestamp = metadata.get("ts")
                if timestamp != last_frame_timestamp:
                    new_frame = True
                    last_frame_timestamp = timestamp
                    last_frame = frame.copy()
                    last_detection, last_mask = detect_yellow_box(
                        last_frame,
                        detector_config,
                    )
                    frame_count += 1
                    if last_detection is not None:
                        detection_count += 1
                        maximum_area_ratio = max(
                            maximum_area_ratio,
                            last_detection.area_ratio,
                        )

            if follower is not None and now >= next_control_tick:
                fresh_detection = (
                    last_detection
                    if last_frame_timestamp != last_control_frame_timestamp
                    else None
                )
                decision = follower.observe(
                    fresh_detection,
                    width=last_frame.shape[1] if last_frame is not None else 1280,
                    height=last_frame.shape[0] if last_frame is not None else 720,
                    dt=CONTROL_PERIOD,
                )
                assert sender is not None
                sender.set_target(decision.update.target)
                last_control_frame_timestamp = last_frame_timestamp
                control_tick_count += 1
                next_control_tick += CONTROL_PERIOD
                if next_control_tick <= now:
                    next_control_tick = now + CONTROL_PERIOD

            if new_frame and last_frame is not None:
                last_annotated = annotate_frame(
                    last_frame,
                    last_detection,
                    decision,
                    args.follow,
                )
            if args.display:
                if new_frame and last_annotated is not None and last_mask is not None:
                    cv2.imshow("Reachy yellow-box follow", last_annotated)
                    cv2.imshow("Reachy yellow mask", last_mask)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    stopped_by = "operator"
                    break
            if not new_frame:
                time.sleep(0.002)
    except KeyboardInterrupt:
        stopped_by = "interrupt"
    except Exception as exc:
        stopped_by = "error"
        run_error = exc
    finally:
        if sender is not None:
            sender.stop()
        client.close()
        if args.display:
            cv2.destroyAllWindows()
        if tracking_started:
            try:
                telemetry = _request_json(
                    "GET",
                    base_url,
                    "/tracking/telemetry?limit=3000",
                    timeout=args.timeout,
                )
            finally:
                _request_json("POST", base_url, "/tracking/stop", timeout=args.timeout)
                if args.return_neutral:
                    _return_neutral(base_url, args.timeout)

    state_after = _request_json(
        "GET",
        base_url,
        "/state/full?with_head_pose=true&with_head_joints=true&with_body_yaw=true",
        timeout=args.timeout,
    )
    records = telemetry.get("records", [])
    if not isinstance(records, list):
        records = []
    if records:
        dump_jsonl(records, output_prefix.with_suffix(".jsonl"))
    if last_annotated is not None:
        cv2.imwrite(str(output_prefix.with_suffix(".png")), last_annotated)
    if last_frame is not None:
        cv2.imwrite(str(Path(f"{output_prefix}-raw.png")), last_frame)
    if last_mask is not None:
        cv2.imwrite(str(Path(f"{output_prefix}-mask.png")), last_mask)
    summary = {
        "mode": "follow" if args.follow else "preview",
        "base_url": base_url,
        "camera_host": camera_host,
        "detector_config": asdict(detector_config),
        "tracking_config": APPROVED_TRACKING_CONFIG if args.follow else None,
        "frame_count": frame_count,
        "detection_count": detection_count,
        "control_tick_count": control_tick_count,
        "detection_fraction": detection_count / frame_count if frame_count else 0.0,
        "maximum_area_ratio": maximum_area_ratio,
        "last_detection": asdict(last_detection)
        if last_detection is not None
        else None,
        "submitted_targets": 0 if sender is None else sender.sent_count,
        "sender_errors": 0 if sender is None else sender.error_count,
        "sender_last_error": None if sender is None else sender.last_error,
        "sender_latency_ms": None
        if sender is None or not sender.latencies
        else {
            "median": float(np.median(sender.latencies) * 1000.0),
            "p95": float(np.percentile(sender.latencies, 95) * 1000.0),
            "maximum": float(max(sender.latencies) * 1000.0),
        },
        "stopped_by": stopped_by,
        "error": None if run_error is None else repr(run_error),
        "state_before": state_before,
        "state_after": state_after,
        "tracking_before": tracking_before,
        "telemetry_records": len(records),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if run_error is not None:
        raise run_error
    return summary


def main() -> None:
    """Parse command-line options and run the yellow-box tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--camera-host")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--return-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hue-low", type=int, default=18)
    parser.add_argument("--hue-high", type=int, default=42)
    parser.add_argument("--saturation-low", type=int, default=170)
    parser.add_argument("--value-low", type=int, default=120)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.5)
    parser.add_argument("--morphology-size", type=int, default=5)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
