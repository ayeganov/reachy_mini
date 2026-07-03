# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "omegaconf>=2.3,<3",
#   "reachy-mini",
#   "rfdetr>=1.3,<2",
#   "ultralytics>=8.3.241,<9",
# ]
# [tool.uv.sources]
# reachy-mini = { path = "..", editable = true }
# ///
"""Preview or follow a selected model target through Reachy's visual servo."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import numpy.typing as npt
from model_detectors import (
    Detector,
    ImageDetection,
    available_models,
    create_detector,
    normalize_target,
    resolve_model_weights,
    select_detection,
    supported_targets_for_model,
)
from websockets.sync.client import ClientConnection, connect
from yellow_box_follow import (
    APPROVED_TRACKING_CONFIG,
    STATE_PATH,
    _neutral_origin,
    _request_json,
    _return_neutral,
)

from reachy_mini.daemon.tracking.telemetry import dump_jsonl
from reachy_mini.media.receivers.zeromq_client import (
    ZeroMQClient,
    ZeroMQClientConfig,
)

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
    """Build the approved robot-side 2D detection payload."""
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
    reason: str,
    follow: bool,
) -> npt.NDArray[np.uint8]:
    """Draw current candidates and emphasize the submitted box."""
    annotated = frame.copy()
    for detection in detections:
        is_selected = detection is selected
        color = (0, 0, 255) if is_selected else (0, 200, 0)
        thickness = 3 if is_selected else 2
        start = (int(round(detection.x1)), int(round(detection.y1)))
        end = (int(round(detection.x2)), int(round(detection.y2)))
        cv2.rectangle(annotated, start, end, color, thickness)
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
            (int(round(u)), int(round(v))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
        )
    mode = "FOLLOW" if follow else "PREVIEW"
    cv2.putText(
        annotated,
        f"{mode} | {model_name} -> {target} | {reason}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def latency_summary(samples: list[float]) -> dict[str, float] | None:
    """Summarize monotonic duration samples in milliseconds."""
    if not samples:
        return None
    return {
        "median": float(np.median(samples) * 1000.0),
        "p95": float(np.percentile(samples, 95) * 1000.0),
        "maximum": float(max(samples) * 1000.0),
    }


def opencv_gui_available() -> bool:
    """Return whether the imported OpenCV build includes a GUI backend."""
    return "GUI:                           NONE" not in cv2.getBuildInformation()


def mode_banner(args: argparse.Namespace) -> str:
    """Describe whether this invocation can move the robot."""
    if args.follow:
        return (
            f"FOLLOW ENABLED: commanding the robot to track {args.target!r} "
            f"with {args.model}. Press Ctrl-C to stop."
        )
    return (
        f"PREVIEW ONLY: detecting {args.target!r} with {args.model}, but sending "
        "ZERO robot targets. Add --follow to enable movement."
    )


def _preflight(base_url: str, timeout: float) -> tuple[dict[str, Any], ...]:
    daemon_status = _request_json("GET", base_url, "/daemon/status", timeout=timeout)
    motor_status = _request_json("GET", base_url, "/motors/status", timeout=timeout)
    tracking_status = _request_json(
        "GET", base_url, "/tracking/status", timeout=timeout
    )
    state = _request_json("GET", base_url, STATE_PATH, timeout=timeout)
    if daemon_status.get("state") != "running" or daemon_status.get("error"):
        raise RuntimeError("robot daemon is not healthy")
    if motor_status.get("mode") != "enabled":
        raise RuntimeError("robot motors are not enabled")
    if bool(tracking_status.get("running")):
        raise RuntimeError("tracking is already running")
    return daemon_status, motor_status, tracking_status, state


def _finish_tracking(
    base_url: str,
    *,
    timeout: float,
    return_neutral: bool,
) -> tuple[dict[str, Any], Exception | None]:
    """Retain telemetry while guaranteeing stop and optional neutral return."""
    telemetry: dict[str, Any] = {}
    first_error: Exception | None = None
    try:
        telemetry = _request_json(
            "GET",
            base_url,
            "/tracking/telemetry?limit=3000",
            timeout=timeout,
        )
    except Exception as exc:
        first_error = exc
    finally:
        try:
            _request_json("POST", base_url, "/tracking/stop", timeout=timeout)
        except Exception as exc:
            first_error = first_error or exc
        finally:
            if return_neutral:
                try:
                    _return_neutral(base_url, timeout)
                except Exception as exc:
                    first_error = first_error or exc
    return telemetry, first_error


def run(args: argparse.Namespace, detector: Detector | None = None) -> dict[str, Any]:
    """Run preview or explicitly enabled object following."""
    model_name = str(args.model)
    target = normalize_target(args.target)
    supported_targets = supported_targets_for_model(model_name)
    if target not in supported_targets:
        supported = ", ".join(sorted(supported_targets))
        raise ValueError(
            f"model {model_name!r} does not support target {target!r}; "
            f"supported targets: {supported}"
        )

    resolved_weights = (
        resolve_model_weights(model_name, args.weights)
        if detector is None
        else args.weights
    )
    load_started = time.monotonic()
    detector = detector or create_detector(
        model_name, weights=resolved_weights, optimize=args.optimize
    )
    detector_load_duration = time.monotonic() - load_started
    if args.display and not opencv_gui_available():
        raise RuntimeError(
            "this environment has a headless OpenCV build; rerun with --no-display"
        )

    base_url = args.base_url.rstrip("/")
    daemon_status, _motor_status, tracking_before, state_before = _preflight(
        base_url, args.timeout
    )
    camera_host = args.camera_host or daemon_status.get("wlan_ip")
    if not isinstance(camera_host, str) or not camera_host:
        raise RuntimeError("robot daemon did not report a camera host")
    if args.follow:
        _neutral_origin(state_before)

    client = ZeroMQClient(
        config=ZeroMQClientConfig(host=camera_host, audio_enabled=False),
        log_level="WARNING",
    )
    detection_stream: ClientConnection | None = None
    tracking_started = False
    telemetry: dict[str, Any] = {}
    frame_count = 0
    matching_frame_count = 0
    candidate_count = 0
    submitted_targets = 0
    last_frame_timestamp: object = None
    last_frame: npt.NDArray[np.uint8] | None = None
    last_annotated: npt.NDArray[np.uint8] | None = None
    last_selected: ImageDetection | None = None
    inference_latencies: list[float] = []
    stream_latencies: list[float] = []
    reason = "preview" if not args.follow else "target_lost"
    started_at = 0.0
    warmup_latency: float | None = None
    stopped_by = "duration"
    run_error: Exception | None = None
    output_prefix = args.output_prefix or Path(
        f"/tmp/reachy-attend-{model_name}-{target.replace(' ', '-')}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )

    try:
        if not client.start(wait_timeout=args.timeout):
            raise RuntimeError("remote camera stream did not start")

        warmup_deadline = time.monotonic() + args.timeout
        warmup_packet = client.get_frame_with_metadata()
        while warmup_packet is None and time.monotonic() < warmup_deadline:
            time.sleep(0.002)
            warmup_packet = client.get_frame_with_metadata()
        if warmup_packet is None:
            raise RuntimeError("remote camera did not provide a warm-up frame")
        warmup_frame, warmup_metadata = warmup_packet
        last_frame_timestamp = warmup_metadata.get("ts")
        warmup_started = time.monotonic()
        detector.detect(
            warmup_frame.copy(),
            target=target,
            min_confidence=args.min_confidence,
        )
        warmup_latency = time.monotonic() - warmup_started

        if args.follow:
            _request_json(
                "POST",
                base_url,
                "/tracking/start",
                APPROVED_TRACKING_CONFIG,
                timeout=args.timeout,
            )
            tracking_started = True
            websocket_url = base_url.replace("http://", "ws://", 1).replace(
                "https://", "wss://", 1
            )
            detection_stream = connect(
                f"{websocket_url}/tracking/ws/detections",
                open_timeout=args.timeout,
                close_timeout=args.timeout,
            )

        started_at = time.monotonic()
        while args.duration <= 0.0 or time.monotonic() - started_at < args.duration:
            packet = client.get_frame_with_metadata()
            new_frame = False
            current_detections: list[ImageDetection] = []
            if packet is not None:
                frame, metadata = packet
                timestamp = metadata.get("ts")
                if timestamp != last_frame_timestamp:
                    new_frame = True
                    last_frame_timestamp = timestamp
                    last_frame = frame.copy()
                    frame_count += 1
                    inference_started = time.monotonic()
                    current_detections = detector.detect(
                        last_frame,
                        target=target,
                        min_confidence=args.min_confidence,
                    )
                    inference_latencies.append(time.monotonic() - inference_started)
                    candidate_count += len(current_detections)
                    last_selected = select_detection(
                        current_detections,
                        strategy=args.selection,
                        width=last_frame.shape[1],
                        height=last_frame.shape[0],
                    )
                    if last_selected is None:
                        reason = "target_lost"
                    else:
                        matching_frame_count += 1
                        reason = "detected"
                        if args.follow:
                            assert detection_stream is not None
                            request_started = time.monotonic()
                            detection_stream.send(
                                json.dumps(
                                    detection_payload(
                                        last_selected,
                                        width=last_frame.shape[1],
                                        height=last_frame.shape[0],
                                        frame_id=frame_count,
                                    )
                                )
                            )
                            response = json.loads(
                                detection_stream.recv(timeout=args.timeout)
                            )
                            if response.get("status") != "accepted":
                                raise RuntimeError(
                                    f"detection stream rejected target: {response}"
                                )
                            stream_latencies.append(time.monotonic() - request_started)
                            submitted_targets += 1
                            reason = "commanded"
                    last_annotated = annotate_frame(
                        last_frame,
                        current_detections,
                        last_selected,
                        model_name=model_name,
                        target=target,
                        reason=reason,
                        follow=args.follow,
                    )

            if args.display:
                if new_frame and last_annotated is not None:
                    cv2.imshow("Reachy attention", last_annotated)
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
        cleanup_error: Exception | None = None
        if tracking_started:
            telemetry, cleanup_error = _finish_tracking(
                base_url,
                timeout=args.timeout,
                return_neutral=args.return_neutral,
            )
        if detection_stream is not None:
            try:
                detection_stream.close()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        try:
            client.close()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        if args.display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if run_error is None and cleanup_error is not None:
            run_error = cleanup_error

    state_after = _request_json("GET", base_url, STATE_PATH, timeout=args.timeout)
    records = telemetry.get("records", [])
    if not isinstance(records, list):
        records = []
    if records:
        dump_jsonl(records, output_prefix.with_suffix(".jsonl"))
    if last_annotated is not None:
        cv2.imwrite(str(output_prefix.with_suffix(".png")), last_annotated)
    if last_frame is not None:
        cv2.imwrite(str(Path(f"{output_prefix}-raw.png")), last_frame)
    summary = {
        "mode": "follow" if args.follow else "preview",
        "model": model_name,
        "target": target,
        "selection": args.selection,
        "min_confidence": args.min_confidence,
        "weights": None if resolved_weights is None else str(resolved_weights),
        "optimized": bool(args.optimize),
        "base_url": base_url,
        "camera_host": camera_host,
        "tracking_config": APPROVED_TRACKING_CONFIG if args.follow else None,
        "detector_load_ms": detector_load_duration * 1000.0,
        "warmup_latency_ms": (
            None if warmup_latency is None else warmup_latency * 1000.0
        ),
        "frame_count": frame_count,
        "matching_frame_count": matching_frame_count,
        "matching_fraction": (
            matching_frame_count / frame_count if frame_count else 0.0
        ),
        "candidate_count": candidate_count,
        "submitted_targets": submitted_targets,
        "last_selected": (asdict(last_selected) if last_selected is not None else None),
        "inference_latency_ms": latency_summary(inference_latencies),
        "stream_latency_ms": latency_summary(stream_latencies),
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Preview or follow one model-selected image target"
    )
    parser.add_argument("--model", default="yolo-face", choices=available_models())
    parser.add_argument("--target", default="face")
    parser.add_argument(
        "--selection",
        choices=("largest", "confidence", "center"),
        default="largest",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--weights",
        type=Path,
        help="checkpoint path (defaults to a matching file in ../../reachy_rf_detr)",
    )
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--camera-host")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="send detections to the robot (default: preview only, no movement)",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--return-neutral", action=argparse.BooleanOptionalAction, default=True
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
        for target in sorted(supported_targets_for_model(args.model)):
            print(target)
        return
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be in [0, 1]")
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    print(mode_banner(args), flush=True)
    summary = run(args)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
