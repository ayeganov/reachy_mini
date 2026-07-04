# ruff: noqa: D100,D103

from __future__ import annotations

from pathlib import Path

import pytest

from reachy_mini.attention.detectors import ImageDetection
from reachy_mini.tools import attention
from reachy_mini.tools._robot_api import RobotApi


def test_detection_payload_contains_only_current_centroid() -> None:
    selected = ImageDetection(10.0, 20.0, 50.0, 80.0, 0.87, "face")

    payload = attention.detection_payload(selected, width=1280, height=720, frame_id=42)

    assert payload == {
        "u": 30.0,
        "v": 50.0,
        "width": 1280,
        "height": 720,
        "confidence": 0.87,
        "frame_id": 42,
    }
    assert "timestamp" not in payload


def test_parser_defaults_to_safe_preview() -> None:
    args = attention.build_parser().parse_args([])

    assert args.follow is False
    assert args.visualize is False
    assert args.return_neutral is True


def test_mode_banner_makes_motion_opt_in() -> None:
    preview = attention.build_parser().parse_args(
        ["--model", "rfdetr-nano", "--target", "person"]
    )
    follow = attention.build_parser().parse_args(
        ["--model", "rfdetr-nano", "--target", "person", "--follow"]
    )

    assert "PREVIEW ONLY" in attention.mode_banner(preview)
    assert "ZERO robot targets" in attention.mode_banner(preview)
    assert "FOLLOW ENABLED" in attention.mode_banner(follow)


def test_attention_uses_daemon_owned_tracking_defaults() -> None:
    assert not hasattr(attention, "TRACKING_CONFIG")


def test_finish_tracking_stops_and_returns_neutral_after_stop_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_post(path: str, payload: object = None) -> dict[str, object]:
        del payload
        calls.append(path)
        if path == "/tracking/stop":
            raise RuntimeError("stop failed")
        return {}

    class FakeApi:
        def post(self, path: str, payload: object = None) -> dict[str, object]:
            return fake_post(path, payload)

        def return_neutral(self) -> None:
            calls.append("neutral")

    error = attention.finish_tracking(
        FakeApi(),  # type: ignore[arg-type]
        return_neutral=True,
    )

    assert isinstance(error, RuntimeError)
    assert calls == ["/tracking/stop", "neutral"]


def test_console_entry_points_are_declared() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'reachy-mini-attend = "reachy_mini.tools.attention:main"' in pyproject
    assert (
        'reachy-mini-attend-viewer = "reachy_mini.tools.attention_viewer:main"'
        in pyproject
    )


def test_neutral_wait_accepts_move_list_response() -> None:
    class FakeApi(RobotApi):
        def __init__(self) -> None:
            super().__init__("http://robot/api", timeout=0.01)

        def post(self, path: str, payload: object = None) -> dict[str, object]:
            del payload
            assert path == "/move/goto"
            return {"uuid": "move-1"}

        def request_value(
            self, method: str, path: str, payload: object = None
        ) -> object:
            del payload
            assert (method, path) == ("GET", "/move/running")
            return []

    FakeApi().return_neutral(duration=0.0)
