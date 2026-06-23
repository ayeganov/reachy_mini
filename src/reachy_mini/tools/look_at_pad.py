"""Interactive and replay tools for metric look-at targets."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from reachy_mini.daemon.tracking.telemetry import dump_jsonl

DEFAULT_BASE_URL = "http://reachy-mini.local:8017/api"
DEFAULT_TRACKING_CONFIG = {
    "smoothing_alpha": 0.20,
    "joint_safety_margin": 0.1745329252,
    "max_joint_velocity": 0.60,
    "max_joint_acceleration": 1.60,
    "max_joint_jerk": 8.0,
}
DEFAULT_REPLAY_PATH = ("center", "top", "right", "bottom", "left", "center")


@dataclass(frozen=True)
class CirclePlane:
    """Screen circle mapped to a metric Y/Z plane at fixed robot-forward X."""

    center_px: tuple[float, float]
    radius_px: float
    radius_m: float
    distance_m: float
    eye_height_m: float
    height_offset_m: float


@dataclass(frozen=True)
class ReplayTarget:
    """One metric look-at target emitted by a replay path."""

    x: float
    y: float
    z: float
    frame_id: int

    def payload(self) -> dict[str, float | int]:
        """Return the JSON payload for the look-at endpoint."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": time.time(),
            "confidence": 1.0,
            "frame_id": self.frame_id,
        }


def screen_to_target(
    px: float,
    py: float,
    plane: CirclePlane,
) -> tuple[float, float, float]:
    """Map screen pixels on the pad circle to a robot-frame look-at target."""
    cx, cy = plane.center_px
    y = ((px - cx) / plane.radius_px) * plane.radius_m
    z_offset = ((cy - py) / plane.radius_px) * plane.radius_m
    z = plane.eye_height_m + plane.height_offset_m + z_offset
    return plane.distance_m, y, z


def clamp_to_circle(px: float, py: float, plane: CirclePlane) -> tuple[float, float]:
    """Clamp a screen point to the pad circle boundary."""
    cx, cy = plane.center_px
    dx = px - cx
    dy = py - cy
    distance = math.hypot(dx, dy)
    if distance <= plane.radius_px or distance <= 1e-9:
        return px, py
    scale = plane.radius_px / distance
    return cx + dx * scale, cy + dy * scale


def named_screen_points(canvas_size: int) -> dict[str, tuple[float, float]]:
    """Return canonical named points on the pad circle."""
    center = (float(canvas_size) / 2.0, float(canvas_size) / 2.0)
    radius = float(canvas_size) * 0.40
    return {
        "center": center,
        "top": (center[0], center[1] - radius),
        "right": (center[0] + radius, center[1]),
        "bottom": (center[0], center[1] + radius),
        "left": (center[0] - radius, center[1]),
    }


def build_replay_targets(
    *,
    path_names: Iterable[str],
    plane: CirclePlane,
    canvas_size: int,
    fps: float,
    segment_duration: float,
    hold_duration: float,
    initial_hold: float,
    final_hold: float,
) -> list[ReplayTarget]:
    """Build timestamp-free look-at targets for a named pad path."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    points = named_screen_points(canvas_size)
    path = tuple(path_names)
    if len(path) < 1:
        raise ValueError("path must include at least one named point")
    unknown = [name for name in path if name not in points]
    if unknown:
        known = ", ".join(sorted(points))
        raise ValueError(f"unknown path point(s): {', '.join(unknown)}; known: {known}")

    frame_id = 0
    targets: list[ReplayTarget] = []

    def append_point(px: float, py: float) -> None:
        nonlocal frame_id
        x, y, z = screen_to_target(px, py, plane)
        targets.append(ReplayTarget(x=x, y=y, z=z, frame_id=frame_id))
        frame_id += 1

    def append_hold(point: tuple[float, float], duration: float) -> None:
        for _ in range(_frame_count(duration, fps)):
            append_point(*point)

    append_hold(points[path[0]], initial_hold)
    for start_name, end_name in zip(path, path[1:]):
        for px, py in _interpolate_points(
            points[start_name],
            points[end_name],
            duration=segment_duration,
            fps=fps,
        ):
            append_point(px, py)
        append_hold(points[end_name], hold_duration)
    append_hold(points[path[-1]], final_hold)
    return targets


def resolve_eye_height(
    base_url: str,
    explicit_eye_height: float | None,
    request_fn: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> float:
    """Return the target-plane eye height in the daemon coordinate frame."""
    if explicit_eye_height is not None:
        return explicit_eye_height
    read_json = request_fn or _request_json
    try:
        state = read_json(
            "GET", base_url.rstrip("/"), "/state/full?with_head_pose=true"
        )
        head_pose = state.get("head_pose", {})
        if isinstance(head_pose, dict):
            z = head_pose.get("z")
            if isinstance(z, int | float):
                return float(z)
    except Exception:
        pass
    return 0.0


def should_stream_target(
    *,
    mouse_down: bool,
    hold_current_target: bool,
    tracking_active: bool,
    last_target: tuple[float, float, float] | None,
) -> bool:
    """Return True when the GUI should keep the robot-side target fresh."""
    if last_target is None or not tracking_active:
        return False
    return mouse_down or hold_current_target


def replay(args: argparse.Namespace) -> dict[str, Any]:
    """Run a named pad replay and write telemetry artifacts."""
    base_url = args.base_url.rstrip("/")
    eye_height = resolve_eye_height(base_url, args.eye_height)
    plane = _plane_from_args(args, eye_height)
    path_names = _parse_path(args.path)
    targets = build_replay_targets(
        path_names=path_names,
        plane=plane,
        canvas_size=args.canvas_size,
        fps=args.fps,
        segment_duration=args.segment_duration,
        hold_duration=args.hold_duration,
        initial_hold=args.initial_hold,
        final_hold=args.final_hold,
    )
    output_prefix = _resolve_output_prefix(args.output_prefix)
    jsonl_path = output_prefix.with_suffix(".jsonl")
    summary_path = output_prefix.with_suffix(".json")

    _post_json(
        base_url, "/tracking/start", DEFAULT_TRACKING_CONFIG, timeout=args.timeout
    )
    start_monotonic = time.monotonic()
    for index, target in enumerate(targets):
        _post_json(
            base_url, "/tracking/look_at", target.payload(), timeout=args.timeout
        )
        next_time = start_monotonic + (index + 1) / args.fps
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
    time.sleep(args.settle)

    after_sweep_state = _request_json(
        "GET",
        base_url,
        "/state/full?with_head_joints=true&with_body_yaw=true&with_antenna_positions=true",
        timeout=args.timeout,
    )
    status_after = _request_json(
        "GET", base_url, "/tracking/status", timeout=args.timeout
    )
    telemetry = _request_json(
        "GET",
        base_url,
        f"/tracking/telemetry?limit={args.telemetry_limit}",
        timeout=args.timeout,
    )
    records = telemetry.get("records")
    if not isinstance(records, list):
        raise ValueError("telemetry response must include a records list")
    object_records = _as_object_records(records)
    dump_jsonl(object_records, jsonl_path)
    record_summary = summarize_replay_records(object_records)
    summary = {
        "base_url": base_url,
        "default_config": DEFAULT_TRACKING_CONFIG,
        "plane": _plane_summary(plane),
        "path": list(path_names),
        "submitted_targets": len(targets),
        "after_sweep_state": after_sweep_state,
        "status_after": status_after,
        "telemetry_meta": {
            key: telemetry.get(key)
            for key in (
                "dropped_records",
                "oldest_sequence",
                "newest_sequence",
                "returned",
                "limit",
            )
        },
        **record_summary,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    stop_status = None
    if not args.leave_running:
        stop_status = _post_json(base_url, "/tracking/stop", timeout=args.timeout)
    return_status = None
    if args.return_neutral:
        return_status = _return_neutral(
            base_url,
            duration=args.return_duration,
            timeout=args.timeout,
        )
    return {
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "submitted_targets": len(targets),
        "record_count": len(object_records),
        "stop_status": stop_status,
        "return_status": return_status,
        **record_summary,
    }


def summarize_replay_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return replay-oriented summary fields from telemetry records."""
    look_at = [record for record in records if record.get("target_type") == "look_at"]
    commanded = [
        record for record in look_at if isinstance(record.get("final_command"), list)
    ]
    command_rows = [
        [float(value) for value in record["final_command"]]
        for record in commanded
        if all(isinstance(value, int | float) for value in record["final_command"])
    ]
    target_y = _input_target_values(look_at, "y")
    target_z = _input_target_values(look_at, "z")
    limit_hits: dict[str, int] = {}
    ik_failures = 0
    reason_counts: dict[str, int] = {}
    for record in records:
        reason = str(record.get("reason"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if bool(record.get("ik_failed")):
            ik_failures += 1
        hits = record.get("limit_hits")
        if isinstance(hits, list):
            for hit in hits:
                key = _limit_hit_key(hit)
                limit_hits[key] = limit_hits.get(key, 0) + 1
    command_spans = _joint_spans(command_rows)
    return {
        "record_count": len(records),
        "look_at_record_count": len(look_at),
        "commanded_count": len(commanded),
        "ik_failures": ik_failures,
        "reason_counts": reason_counts,
        "limit_hits": limit_hits,
        "target_y_span_m": _span(target_y),
        "target_z_span_m": _span(target_z),
        "final_command_spans_rad": command_spans,
        "final_command_spans_deg": [math.degrees(value) for value in command_spans],
        "max_final_command_span_rad": max(command_spans, default=0.0),
        "max_final_command_span_deg": math.degrees(max(command_spans, default=0.0)),
    }


class LookAtPadApp:
    """Small local GUI that streams metric look-at targets while dragging."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Create the pad window and bind its controls."""
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.root = tk.Tk()
        self.root.title("Reachy Look-At Pad")
        self.base_url = args.base_url.rstrip("/")
        self.frame_id = 0
        self.mouse_down = False
        self.tracking_active = False
        self.hold_current_target = not args.stream_only_while_dragging
        self.eye_height_m = resolve_eye_height(self.base_url, args.eye_height)
        self.last_screen_point: tuple[float, float] | None = None
        self.last_target: tuple[float, float, float] | None = None
        self.status_var = tk.StringVar(value="idle")

        self.canvas = tk.Canvas(
            self.root,
            width=args.canvas_size,
            height=args.canvas_size,
            bg="#111318",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.height_var = tk.DoubleVar(value=args.height_offset)
        self.distance_var = tk.DoubleVar(value=args.distance)
        self.radius_var = tk.DoubleVar(value=args.radius_m)

        tk.Label(self.root, text="height offset m").grid(row=1, column=0, sticky="ew")
        tk.Scale(
            self.root,
            variable=self.height_var,
            from_=-0.20,
            to=0.20,
            resolution=0.005,
            orient="horizontal",
            command=self._on_slider,
        ).grid(row=1, column=1, columnspan=3, sticky="ew")

        tk.Label(self.root, text="distance m").grid(row=2, column=0, sticky="ew")
        tk.Scale(
            self.root,
            variable=self.distance_var,
            from_=0.20,
            to=1.00,
            resolution=0.01,
            orient="horizontal",
            command=self._on_slider,
        ).grid(row=2, column=1, columnspan=3, sticky="ew")

        tk.Label(self.root, text="circle radius m").grid(row=3, column=0, sticky="ew")
        tk.Scale(
            self.root,
            variable=self.radius_var,
            from_=0.05,
            to=0.30,
            resolution=0.005,
            orient="horizontal",
            command=self._on_slider,
        ).grid(row=3, column=1, columnspan=3, sticky="ew")

        tk.Button(self.root, text="start tracking", command=self._start_tracking).grid(
            row=4, column=0, sticky="ew"
        )
        tk.Button(self.root, text="stop tracking", command=self._stop_tracking).grid(
            row=4, column=1, sticky="ew"
        )
        tk.Button(self.root, text="recenter", command=self._recenter).grid(
            row=4, column=2, sticky="ew"
        )
        tk.Label(self.root, textvariable=self.status_var, anchor="w").grid(
            row=4, column=3, sticky="ew"
        )

        self.root.bind("<space>", lambda _event: self._start_tracking())
        self.root.bind("<Escape>", lambda _event: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        for column in range(4):
            self.root.columnconfigure(column, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._draw()
        if not args.no_start:
            self._start_tracking()
        self.root.after(int(1000 / args.fps), self._stream_tick)

    def _plane(self) -> CirclePlane:
        center = (
            float(self.args.canvas_size) / 2.0,
            float(self.args.canvas_size) / 2.0,
        )
        return CirclePlane(
            center_px=center,
            radius_px=float(self.args.canvas_size) * 0.40,
            radius_m=float(self.radius_var.get()),
            distance_m=float(self.distance_var.get()),
            eye_height_m=self.eye_height_m,
            height_offset_m=float(self.height_var.get()),
        )

    def _post(
        self, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return _post_json(self.base_url, path, payload, timeout=0.75)

    def _start_tracking(self) -> None:
        try:
            status = self._post("/tracking/start", DEFAULT_TRACKING_CONFIG)
            self.tracking_active = True
            self.status_var.set(f"tracking: {status['last_reason']}")
        except Exception as exc:
            self.status_var.set(f"start failed: {exc}")

    def _stop_tracking(self) -> None:
        try:
            status = self._post("/tracking/stop")
            self.tracking_active = False
            self.status_var.set(f"tracking: {status['last_reason']}")
        except Exception as exc:
            self.status_var.set(f"stop failed: {exc}")

    def _recenter(self) -> None:
        plane = self._plane()
        self._set_screen_point(*plane.center_px)
        self._send_current_target()

    def _on_press(self, event: object) -> None:
        self.mouse_down = True
        self._set_screen_point(event.x, event.y)  # type: ignore[attr-defined]
        self._send_current_target()

    def _on_drag(self, event: object) -> None:
        self._set_screen_point(event.x, event.y)  # type: ignore[attr-defined]

    def _on_release(self, event: object) -> None:
        self._set_screen_point(event.x, event.y)  # type: ignore[attr-defined]
        self._send_current_target()
        self.mouse_down = False

    def _on_slider(self, _value: str) -> None:
        if self.last_screen_point is not None:
            self._set_screen_point(*self.last_screen_point)
            self._send_current_target()
        else:
            self._draw()

    def _set_screen_point(self, px: float, py: float) -> None:
        plane = self._plane()
        point = clamp_to_circle(px, py, plane)
        self.last_screen_point = point
        self.last_target = screen_to_target(point[0], point[1], plane)
        self._draw()

    def _send_current_target(self) -> None:
        if self.last_target is None:
            return
        x, y, z = self.last_target
        try:
            status = self._post(
                "/tracking/look_at",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "timestamp": time.time(),
                    "confidence": 1.0,
                    "frame_id": self.frame_id,
                },
            )
            self.frame_id += 1
            self.status_var.set(
                f"x={x:.2f} y={y:.3f} z={z:.3f} cmd={status['command_count']}"
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            self.status_var.set(f"send failed: {exc}")

    def _stream_tick(self) -> None:
        if should_stream_target(
            mouse_down=self.mouse_down,
            hold_current_target=self.hold_current_target,
            tracking_active=self.tracking_active,
            last_target=self.last_target,
        ):
            self._send_current_target()
        self.root.after(int(1000 / self.args.fps), self._stream_tick)

    def _draw(self) -> None:
        self.canvas.delete("all")
        plane = self._plane()
        cx, cy = plane.center_px
        radius = plane.radius_px
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#e6e8ee",
            width=2,
        )
        self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#777d8b")
        self.canvas.create_line(cx, cy - 12, cx, cy + 12, fill="#777d8b")
        self.canvas.create_text(
            cx,
            cy + radius + 20,
            fill="#aeb6c8",
            text=(
                f"eye level z={plane.eye_height_m + plane.height_offset_m:.3f}m, "
                f"distance x={plane.distance_m:.2f}m"
            ),
        )
        if self.last_screen_point is not None:
            px, py = self.last_screen_point
            self.canvas.create_oval(
                px - 7,
                py - 7,
                px + 7,
                py + 7,
                fill="#ff4d4d",
                outline="",
            )

    def run(self) -> None:
        """Run the Tk event loop."""
        self.root.mainloop()

    def quit(self) -> None:
        """Stop tracking unless requested otherwise, then close the window."""
        if not self.args.leave_running:
            self._stop_tracking()
        self.root.destroy()


def main() -> None:
    """Run the look-at pad command-line interface."""
    parser = argparse.ArgumentParser(
        description="Control or replay metric look-at targets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_gui_parser(subparsers)
    _add_replay_parser(subparsers)
    args = parser.parse_args()
    if args.command == "gui":
        LookAtPadApp(args).run()
    elif args.command == "replay":
        print(json.dumps(replay(args), indent=2, allow_nan=False))


def _add_common_pad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--canvas-size", type=int, default=720)
    parser.add_argument(
        "--eye-height",
        type=float,
        default=None,
        help="Target plane center height. Defaults to daemon head_pose.z.",
    )
    parser.add_argument("--height-offset", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=0.5)
    parser.add_argument("--radius-m", type=float, default=0.15)


def _add_gui_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("gui", help="Open the interactive look-at pad.")
    _add_common_pad_args(parser)
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--leave-running", action="store_true")
    parser.add_argument(
        "--stream-only-while-dragging",
        action="store_true",
        help="Do not keep a clicked target fresh after mouse release.",
    )


def _add_replay_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("replay", help="Replay a named pad path.")
    _add_common_pad_args(parser)
    parser.add_argument(
        "--path",
        default=",".join(DEFAULT_REPLAY_PATH),
        help="Comma-separated pad points: center, top, right, bottom, left.",
    )
    parser.add_argument("--segment-duration", type=float, default=1.4)
    parser.add_argument("--hold-duration", type=float, default=0.5)
    parser.add_argument("--initial-hold", type=float, default=0.8)
    parser.add_argument("--final-hold", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--telemetry-limit", type=int, default=5000)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for .jsonl and .json artifacts.",
    )
    parser.add_argument("--leave-running", action="store_true")
    parser.add_argument("--return-neutral", action="store_true")
    parser.add_argument("--return-duration", type=float, default=1.5)


def _parse_path(value: str) -> tuple[str, ...]:
    path = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not path:
        raise ValueError("path must contain at least one point")
    return path


def _frame_count(duration: float, fps: float) -> int:
    if duration < 0.0:
        raise ValueError("duration cannot be negative")
    return max(0, int(duration * fps))


def _interpolate_points(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    duration: float,
    fps: float,
) -> Iterable[tuple[float, float]]:
    frames = max(1, _frame_count(duration, fps))
    for index in range(frames):
        u = _smoothstep(index / max(1, frames - 1))
        yield (start[0] + (end[0] - start[0]) * u, start[1] + (end[1] - start[1]) * u)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _plane_from_args(args: argparse.Namespace, eye_height: float) -> CirclePlane:
    center = (float(args.canvas_size) / 2.0, float(args.canvas_size) / 2.0)
    return CirclePlane(
        center_px=center,
        radius_px=float(args.canvas_size) * 0.40,
        radius_m=float(args.radius_m),
        distance_m=float(args.distance),
        eye_height_m=eye_height,
        height_offset_m=float(args.height_offset),
    )


def _plane_summary(plane: CirclePlane) -> dict[str, Any]:
    return {
        "center_px": list(plane.center_px),
        "radius_px": plane.radius_px,
        "radius_m": plane.radius_m,
        "distance_m": plane.distance_m,
        "eye_height_m": plane.eye_height_m,
        "height_offset_m": plane.height_offset_m,
    }


def _resolve_output_prefix(prefix: Path | None) -> Path:
    if prefix is not None:
        return prefix
    return Path(f"/tmp/reachy-look-at-pad-replay-{time.strftime('%Y%m%d-%H%M%S')}")


def _request_json(
    method: str,
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    parsed = _request_payload(method, base_url, path, payload, timeout)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def _request_payload(
    method: str,
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 10.0,
) -> Any:
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
    return json.loads(body, parse_constant=_reject_non_finite_json_constant)


def _post_json(
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    return _request_json("POST", base_url, path, payload, timeout=timeout)


def _return_neutral(
    base_url: str,
    *,
    duration: float,
    timeout: float,
) -> dict[str, Any]:
    move = _post_json(
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
            "duration": duration,
            "interpolation": "minjerk",
        },
        timeout=timeout,
    )
    uuid = move.get("uuid")
    deadline = time.monotonic() + max(timeout, duration + 5.0)
    while isinstance(uuid, str) and time.monotonic() < deadline:
        running = _request_payload("GET", base_url, "/move/running", timeout=timeout)
        if not isinstance(running, list):
            break
        if not any(
            item.get("uuid") == uuid for item in running if isinstance(item, dict)
        ):
            break
        time.sleep(0.25)
    return move


def _as_object_records(records: list[Any]) -> list[dict[str, Any]]:
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("telemetry records must be objects")
    return records


def _input_target_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        target = record.get("input_target")
        if isinstance(target, dict):
            value = target.get(field)
            if isinstance(value, int | float):
                values.append(float(value))
    return values


def _joint_spans(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    return [
        max(row[index] for row in rows) - min(row[index] for row in rows)
        for index in range(len(rows[0]))
    ]


def _span(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _limit_hit_key(hit: Any) -> str:
    if isinstance(hit, dict):
        value = hit.get("limit") or hit.get("kind") or hit.get("reason") or "unknown"
        return str(value)
    return str(hit)


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


if __name__ == "__main__":
    main()
