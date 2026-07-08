"""Preview or follow a model-selected target through Reachy's visual servo."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import numpy.typing as npt
from websockets.sync.client import ClientConnection, connect

from reachy_mini.attention.detectors import (
    Detector,
    ImageDetection,
    available_models,
    create_detector,
    normalize_target,
    select_detection,
    supported_targets_for_model,
)
from reachy_mini.media.receivers.zeromq_client import (
    ZeroMQClient,
    ZeroMQClientConfig,
)
from reachy_mini.tools._robot_api import RobotApi
from reachy_mini.tools.attention_viewer import DEFAULT_ENDPOINT, DetectionPublisher

DEFAULT_BASE_URL = "http://reachy-mini.local:8017/api"

# Joining an H.264 stream between keyframes can produce transient decoder noise.
av.logging.set_level(av.logging.PANIC)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def detection_payload(
    detection: ImageDetection,
    *,
    width: int,
    height: int,
    frame_id: int,
) -> dict[str, Any]:
    """Build one robot-side current-detection payload."""
    u, v = detection.centroid
    return {
        "u": u,
        "v": v,
        "width": width,
        "height": height,
        "confidence": detection.confidence,
        "frame_id": frame_id,
    }


def annotate_frame(
    frame: npt.NDArray[np.uint8],
    detections: list[ImageDetection],
    selected: ImageDetection | None,
    *,
    model_name: str,
    target: str,
    follow: bool,
) -> npt.NDArray[np.uint8]:
    """Draw target-filtered candidates and the submitted centroid."""
    annotated = frame.copy()
    for detection in detections:
        chosen = detection is selected
        color = (0, 0, 255) if chosen else (0, 200, 0)
        start = (round(detection.x1), round(detection.y1))
        end = (round(detection.x2), round(detection.y2))
        cv2.rectangle(annotated, start, end, color, 3 if chosen else 2)
        cv2.putText(
            annotated,
            f"{detection.label} {detection.confidence:.2f}",
            (start[0], max(18, start[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if selected is not None:
        u, v = selected.centroid
        cv2.drawMarker(
            annotated,
            (round(u), round(v)),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
        )
    cv2.putText(
        annotated,
        f"{'FOLLOW' if follow else 'PREVIEW'} | {model_name} -> {target}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def mode_banner(args: argparse.Namespace) -> str:
    """Describe whether this invocation can move the robot."""
    if args.follow:
        return (
            f"FOLLOW ENABLED: moving the robot to neutral, then commanding it "
            f"to track {args.target!r} with {args.model}. Press Ctrl-C to stop."
        )
    return (
        f"PREVIEW ONLY: detecting {args.target!r} with {args.model}, but sending "
        "ZERO robot targets. Add --follow to enable movement."
    )


def preflight(api: RobotApi, *, follow: bool) -> dict[str, Any]:
    """Require a healthy daemon and, for following, available enabled motors."""
    daemon = api.get("/daemon/status")
    if daemon.get("state") != "running" or daemon.get("error"):
        raise RuntimeError("robot daemon is not healthy")
    if follow:
        if api.get("/motors/status").get("mode") != "enabled":
            raise RuntimeError("robot motors are not enabled")
        if api.get("/tracking/status").get("running"):
            raise RuntimeError("tracking is already running")
    return daemon


def finish_tracking(api: RobotApi, *, return_neutral: bool) -> Exception | None:
    """Stop tracking and optionally return neutral, retaining the first error."""
    error: Exception | None = None
    try:
        api.post("/tracking/stop")
    except Exception as exc:
        error = exc
    if return_neutral:
        try:
            api.return_neutral()
        except Exception as exc:
            error = error or exc
    return error


def run(args: argparse.Namespace, detector: Detector | None = None) -> dict[str, Any]:
    """Run preview or explicitly enabled object following."""
    model_name = str(args.model)
    target = normalize_target(args.target)
    supported_targets = supported_targets_for_model(model_name)
    if not target:
        raise ValueError("target must not be empty")
    if supported_targets is not None and target not in supported_targets:
        supported = ", ".join(sorted(supported_targets))
        raise ValueError(
            f"model {model_name!r} does not support target {target!r}; "
            f"supported targets: {supported}"
        )

    detector = detector or create_detector(
        model_name,
        weights=args.weights,
        optimize=args.optimize,
    )
    api = RobotApi(args.base_url, args.timeout)
    daemon = preflight(api, follow=args.follow)
    camera_host = args.camera_host or daemon.get("wlan_ip")
    if not isinstance(camera_host, str) or not camera_host:
        raise RuntimeError("robot daemon did not report a camera host")
    if args.follow:
        api.return_neutral()

    camera = ZeroMQClient(
        config=ZeroMQClientConfig(host=camera_host, audio_enabled=False),
        log_level="WARNING",
    )
    stream: ClientConnection | None = None
    publisher: DetectionPublisher | None = None
    tracking_start_attempted = False
    frame_count = 0
    matching_frames = 0
    submitted_targets = 0
    last_timestamp: object = None
    stopped_by = "duration"
    run_error: Exception | None = None
    visualization_error: str | None = None
    started_at = 0.0

    try:
        if args.visualize:
            try:
                publisher = DetectionPublisher(args.visualization_endpoint)
                print(
                    f"Publishing target-filtered detections at "
                    f"{args.visualization_endpoint}",
                    flush=True,
                )
            except Exception as exc:
                visualization_error = str(exc)
                print(f"Visualization disabled: {exc}", flush=True)
        if not camera.start(wait_timeout=args.timeout):
            raise RuntimeError("remote camera stream did not start")

        deadline = time.monotonic() + args.timeout
        warmup = camera.get_frame_with_metadata()
        while warmup is None and time.monotonic() < deadline:
            time.sleep(0.002)
            warmup = camera.get_frame_with_metadata()
        if warmup is None:
            raise RuntimeError("remote camera did not provide a warm-up frame")
        warmup_frame, metadata = warmup
        last_timestamp = metadata.get("ts")
        detector.detect(
            warmup_frame,
            target=target,
            min_confidence=args.min_confidence,
        )

        if args.follow:
            tracking_start_attempted = True
            api.post("/tracking/start", {})
            websocket_url = api.base_url.replace("http://", "ws://", 1).replace(
                "https://", "wss://", 1
            )
            stream = connect(
                f"{websocket_url}/tracking/ws/detections",
                open_timeout=args.timeout,
                close_timeout=args.timeout,
            )

        started_at = time.monotonic()
        next_visualization = started_at
        while args.duration <= 0.0 or time.monotonic() - started_at < args.duration:
            packet = camera.get_frame_with_metadata()
            if packet is None or packet[1].get("ts") == last_timestamp:
                time.sleep(0.002)
                continue
            frame, metadata = packet
            last_timestamp = metadata.get("ts")
            frame_count += 1
            detections = detector.detect(
                frame,
                target=target,
                min_confidence=args.min_confidence,
            )
            selected = select_detection(
                detections,
                strategy=args.selection,
                width=frame.shape[1],
                height=frame.shape[0],
            )
            if selected is not None:
                matching_frames += 1
                if stream is not None:
                    stream.send(
                        json.dumps(
                            detection_payload(
                                selected,
                                width=frame.shape[1],
                                height=frame.shape[0],
                                frame_id=frame_count,
                            )
                        )
                    )
                    response = json.loads(stream.recv(timeout=args.timeout))
                    if response.get("status") != "accepted":
                        raise RuntimeError(
                            f"detection stream rejected target: {response}"
                        )
                    submitted_targets += 1
            now = time.monotonic()
            if publisher is not None and now >= next_visualization:
                try:
                    publisher.publish(
                        annotate_frame(
                            frame,
                            detections,
                            selected,
                            model_name=model_name,
                            target=target,
                            follow=args.follow,
                        )
                    )
                    next_visualization = now + 1.0 / args.visualization_fps
                except Exception as exc:
                    visualization_error = visualization_error or str(exc)
                    print(f"Visualization disabled: {exc}", flush=True)
                    try:
                        publisher.close()
                    except Exception as close_exc:
                        visualization_error = visualization_error or str(close_exc)
                    publisher = None
    except KeyboardInterrupt:
        stopped_by = "interrupt"
    except Exception as exc:
        stopped_by = "error"
        run_error = exc
    finally:
        cleanup_error: Exception | None = None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                cleanup_error = exc
        if tracking_start_attempted:
            tracking_error = finish_tracking(
                api,
                return_neutral=args.return_neutral,
            )
            cleanup_error = cleanup_error or tracking_error
        try:
            camera.close()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        if publisher is not None:
            try:
                publisher.close()
            except Exception as exc:
                visualization_error = visualization_error or str(exc)
        run_error = run_error or cleanup_error

    summary = {
        "mode": "follow" if args.follow else "preview",
        "model": model_name,
        "target": target,
        "camera_host": camera_host,
        "frame_count": frame_count,
        "matching_frame_count": matching_frames,
        "submitted_targets": submitted_targets,
        "visualization_error": visualization_error,
        "stopped_by": stopped_by,
        "error": None if run_error is None else repr(run_error),
    }
    if run_error is not None:
        raise run_error
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Preview or follow one model-selected image target",
        epilog=(
            "Install the selected model family with --extra attention-yolo-face, "
            "--extra attention-yoloe, or --extra attention-rfdetr."
        ),
    )
    parser.add_argument("--model", default="yolo-face", choices=available_models())
    parser.add_argument("--target", default="face")
    parser.add_argument(
        "--selection",
        choices=("largest", "confidence", "center"),
        default="largest",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--camera-host")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--visualization-fps", type=float, default=15.0)
    parser.add_argument(
        "--return-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    """Run the selected detector or print registry information."""
    parser = build_parser()
    args = parser.parse_args()
    if args.list_models:
        for name, description in available_models().items():
            print(f"{name}\t{description}")
        return
    if args.list_targets:
        targets = supported_targets_for_model(args.model)
        print(
            "any non-empty text label"
            if targets is None
            else "\n".join(sorted(targets))
        )
        return
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0, 1]")
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if not math.isfinite(args.visualization_fps) or args.visualization_fps <= 0.0:
        parser.error("--visualization-fps must be positive and finite")
    print(mode_banner(args), flush=True)
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
