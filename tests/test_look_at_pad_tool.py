# ruff: noqa: D100,D101,D102,D103

import argparse
import math

from reachy_mini.tools.look_at_pad import (
    DEFAULT_TRACKING_CONFIG,
    CirclePlane,
    LookAtPadApp,
    build_replay_targets,
    named_screen_points,
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


def test_default_tracking_config_matches_interactive_pad_values() -> None:
    assert DEFAULT_TRACKING_CONFIG == {
        "smoothing_alpha": 0.20,
        "joint_safety_margin": 0.1745329252,
        "max_joint_velocity": 0.60,
        "max_joint_acceleration": 1.60,
        "max_joint_jerk": 8.0,
    }


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
