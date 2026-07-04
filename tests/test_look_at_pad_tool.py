# ruff: noqa: D100,D101,D102,D103

import argparse
import json
import math
from pathlib import Path

import pytest

from reachy_mini.tools import look_at_pad
from reachy_mini.tools.look_at_pad import (
    CirclePlane,
    LookAtPadApp,
    build_replay_targets,
    named_screen_points,
    replay,
    screen_to_target,
    summarize_replay_records,
)


def test_named_pad_points_match_metric_y_z_plane() -> None:
    plane = CirclePlane(
        center_px=(360.0, 360.0),
        radius_px=288.0,
        radius_m=0.15,
        distance_m=0.5,
        eye_height_m=-0.004,
        height_offset_m=0.0,
    )
    points = named_screen_points(canvas_size=720)

    assert screen_to_target(*points["center"], plane) == (0.5, 0.0, -0.004)
    assert screen_to_target(*points["top"], plane) == (0.5, 0.0, 0.146)
    assert screen_to_target(*points["right"], plane) == (0.5, -0.15, -0.004)
    assert screen_to_target(*points["bottom"], plane) == (0.5, 0.0, -0.154)
    assert screen_to_target(*points["left"], plane) == (0.5, 0.15, -0.004)


def test_build_replay_targets_exercises_both_lateral_and_vertical_axes() -> None:
    plane = CirclePlane(
        center_px=(360.0, 360.0),
        radius_px=288.0,
        radius_m=0.15,
        distance_m=0.5,
        eye_height_m=0.01,
        height_offset_m=0.0,
    )

    targets = build_replay_targets(
        path_names=("center", "top", "right", "bottom", "left", "center"),
        plane=plane,
        canvas_size=720,
        fps=30.0,
        segment_duration=1.4,
        hold_duration=0.5,
        initial_hold=0.8,
        final_hold=1.0,
    )

    ys = [target.y for target in targets]
    zs = [target.z for target in targets]

    assert len(targets) == 339
    assert math.isclose(max(ys) - min(ys), 0.30, abs_tol=1e-12)
    assert math.isclose(max(zs) - min(zs), 0.30, abs_tol=1e-12)
    assert targets[0].frame_id == 0
    assert targets[-1].frame_id == len(targets) - 1


def test_pad_does_not_duplicate_daemon_tracking_defaults() -> None:
    assert not hasattr(look_at_pad, "DEFAULT_TRACKING_CONFIG")


def test_replay_summary_includes_command_smoothness() -> None:
    records = [
        {
            "timestamp": 1.0,
            "target_type": "look_at",
            "reason": "commanded",
            "input_target": {"y": 0.0, "z": 0.0},
            "final_command": [0.0, 0.0],
            "limit_hits": [],
            "ik_failed": False,
        },
        {
            "timestamp": 2.0,
            "target_type": "look_at",
            "reason": "commanded",
            "input_target": {"y": 0.1, "z": 0.0},
            "final_command": [1.0, 0.0],
            "limit_hits": [{"kind": "jerk", "limit": 8.0}],
            "ik_failed": False,
        },
        {
            "timestamp": 3.0,
            "target_type": "look_at",
            "reason": "commanded",
            "input_target": {"y": 0.1, "z": 0.1},
            "final_command": [1.0, 1.0],
            "limit_hits": [],
            "ik_failed": False,
        },
    ]

    summary = summarize_replay_records(records)

    assert summary["limit_hit_count"] == 1
    assert summary["command_smoothness"] == {
        "max_velocity": 1.0,
        "max_acceleration": 1.0,
        "max_jerk": None,
    }


def _profile_replay_records() -> list[dict[str, object]]:
    profile_commands = [
        [0.0] * 7,
        [0.1] * 7,
        [0.3] * 7,
        [0.4] * 7,
    ]
    final_commands = [command.copy() for command in profile_commands]
    final_commands[1][0] += 0.1
    return [
        {
            "timestamp": float(index),
            "target_type": "look_at",
            "reason": "commanded",
            "input_target": {"y": float(index), "z": float(index)},
            "ik_failed": False,
            "ik_joints": [0.5] * 7,
            "profiled_command": profiled,
            "final_command": final,
            "profile_limit_hits": [
                {
                    "joint_index": 1,
                    "kind": "jerk",
                    "source": "profile_jerk",
                    "value": 9.0,
                    "limit": 8.0,
                }
            ]
            if index == 2
            else [],
            "limit_hits": [{"joint_index": 1, "kind": "acceleration", "limit": 1.6}]
            if index == 3
            else [],
        }
        for index, (profiled, final) in enumerate(
            zip(profile_commands, final_commands), start=1
        )
    ]


def test_replay_summary_reports_profile_and_limiter_diagnostics() -> None:
    summary = summarize_replay_records(_profile_replay_records())

    assert summary["profile_limit_hits"] == {"jerk": 1}
    assert summary["profile_limit_hit_count"] == 1
    assert summary["limit_hits"] == {"acceleration": 1}
    assert summary["limiter_delta"] == {
        "max_abs_rad": pytest.approx(0.1),
        "mean_abs_rad": pytest.approx(0.1 / 28.0),
        "changed_tick_count": 1,
    }
    assert summary["limiter_changed_count"] == 1
    assert summary["limiter_compared_count"] == 4
    assert summary["command_smoothness"] == summary["final_command_smoothness"]
    for field in (
        "profiled_command_smoothness",
        "final_command_smoothness",
    ):
        assert set(summary[field]) == {
            "max_velocity",
            "max_acceleration",
            "max_jerk",
        }
        assert all(isinstance(value, float) for value in summary[field].values())


def test_replay_summary_skips_invalid_profile_pairs() -> None:
    valid = _profile_replay_records()[0]
    invalid_pairs = [
        {"final_command": [0.0] * 7},
        {"profiled_command": None, "final_command": [0.0] * 7},
        {"profiled_command": ["bad"] * 7, "final_command": [0.0] * 7},
        {"profiled_command": [float("nan")] * 7, "final_command": [0.0] * 7},
        {"profiled_command": [0.0] * 6, "final_command": [0.0] * 7},
    ]
    records = [valid]
    records.extend(
        {
            "timestamp": float(index + 2),
            "target_type": "look_at",
            "reason": "commanded",
            "profile_limit_hits": [],
            "limit_hits": [],
            **pair,
        }
        for index, pair in enumerate(invalid_pairs)
    )
    records.append(
        {
            "timestamp": 20.0,
            "target_type": "detection",
            "reason": "commanded",
            "profiled_command": [0.0] * 7,
            "final_command": [1.0] * 7,
            "profile_limit_hits": [],
            "limit_hits": [],
        }
    )

    summary = summarize_replay_records(records)

    assert summary["limiter_compared_count"] == 1
    assert summary["limiter_changed_count"] == 0
    assert summary["limiter_delta"] == {
        "max_abs_rad": 0.0,
        "mean_abs_rad": 0.0,
        "changed_tick_count": 0,
    }

    no_pairs = summarize_replay_records(records[1:])
    assert no_pairs["limiter_delta"] == {
        "max_abs_rad": None,
        "mean_abs_rad": None,
        "changed_tick_count": 0,
    }
    assert no_pairs["limiter_compared_count"] == 0


def test_replay_sends_config_and_retains_post_return_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    posted: list[tuple[str, object]] = []
    requested: list[str] = []
    post_return_state = {
        "head_pose": {
            "x": 0.01,
            "y": -0.02,
            "z": 0.0,
            "roll": 0.01,
            "pitch": -0.04,
            "yaw": 0.05,
        },
        "head_joints": [0.0] * 7,
        "body_yaw": -0.03,
    }

    def fake_post(
        _base_url: str,
        path: str,
        payload: object = None,
        timeout: float = 10.0,
    ) -> dict[str, object]:
        posted.append((path, payload))
        return {"path": path, "timeout": timeout}

    def fake_request(
        _method: str,
        _base_url: str,
        path: str,
        payload: object = None,
        timeout: float = 10.0,
    ) -> dict[str, object]:
        requested.append(path)
        if path.startswith("/tracking/telemetry"):
            return {
                "records": _profile_replay_records(),
                "dropped_records": 0,
                "oldest_sequence": 0,
                "newest_sequence": 3,
                "returned": 4,
                "limit": 5000,
            }
        if path == "/tracking/status":
            return {"running": True}
        if "with_head_pose=true" in path:
            return post_return_state
        return {"head_joints": [0.0] * 7, "body_yaw": 0.0}

    monkeypatch.setattr(look_at_pad, "_post_json", fake_post)
    monkeypatch.setattr(look_at_pad, "_request_json", fake_request)
    monkeypatch.setattr(
        look_at_pad,
        "_return_neutral",
        lambda *_args, **_kwargs: {"uuid": "neutral"},
    )
    monkeypatch.setattr(look_at_pad.time, "sleep", lambda _seconds: None)
    output_prefix = tmp_path / "replay"
    args = argparse.Namespace(
        base_url="http://robot/api",
        eye_height=0.0,
        canvas_size=720,
        height_offset=0.0,
        distance=0.5,
        radius_m=0.15,
        path="center",
        fps=30.0,
        segment_duration=0.0,
        hold_duration=0.0,
        initial_hold=0.0,
        final_hold=0.0,
        settle=0.0,
        timeout=1.0,
        telemetry_limit=5000,
        output_prefix=output_prefix,
        leave_running=False,
        return_neutral=True,
        return_duration=1.0,
        look_at_profile_response_hz=2.0,
    )

    result = replay(args)
    summary = json.loads(output_prefix.with_suffix(".json").read_text())

    start_payload = next(
        payload for path, payload in posted if path == "/tracking/start"
    )
    assert isinstance(start_payload, dict)
    assert start_payload["look_at_profile_response_hz"] == 2.0
    assert summary["tracking_config"] == start_payload
    assert result["tracking_config"] == start_payload
    assert summary["return_status"] == {"uuid": "neutral"}
    assert summary["post_return_state"] == post_return_state
    assert summary["neutral_return"]["within_tolerance"] is True
    assert requested[-1] == (
        "/state/full?with_head_pose=true&with_head_joints=true&with_body_yaw=true"
    )
    assert {path for path, _payload in posted} == {
        "/tracking/start",
        "/tracking/stop",
    }


def test_gui_plane_uses_current_slider_values() -> None:
    app = LookAtPadApp.__new__(LookAtPadApp)
    app.args = argparse.Namespace(
        canvas_size=720,
        height_offset=0.0,
        distance=0.5,
        radius_m=0.15,
    )
    app.eye_height_m = -0.01
    app.height_var = _ValueVar(0.055)
    app.distance_var = _ValueVar(0.7)
    app.radius_var = _ValueVar(0.22)

    plane = app._plane()

    assert plane.height_offset_m == 0.055
    assert plane.distance_m == 0.7
    assert plane.radius_m == 0.22


def test_gui_slider_redraws_without_selected_target() -> None:
    app = LookAtPadApp.__new__(LookAtPadApp)
    app.last_screen_point = None
    draw_count = 0

    def draw() -> None:
        nonlocal draw_count
        draw_count += 1

    app._draw = draw

    app._on_slider("0.10")

    assert draw_count == 1


class _ValueVar:
    def __init__(self, value: float) -> None:
        self.value = value

    def get(self) -> float:
        return self.value
