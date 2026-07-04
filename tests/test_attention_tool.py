# ruff: noqa: D100,D103

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from reachy_mini.attention.detectors import ImageDetection
from reachy_mini.tools import attention
from reachy_mini.tools._robot_api import RobotApi


class _RunApi:
    base_url = "http://robot/api"

    def __init__(
        self,
        calls: list[str],
        start_error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.start_error = start_error

    def get(self, path: str) -> dict[str, object]:
        return {
            "/daemon/status": {"state": "running", "wlan_ip": "robot"},
            "/motors/status": {"mode": "enabled"},
            "/tracking/status": {"running": False},
        }[path]

    def post(self, path: str, payload: object = None) -> dict[str, object]:
        del payload
        self.calls.append(path)
        if path == "/tracking/start" and self.start_error is not None:
            raise self.start_error
        return {}

    def return_neutral(self) -> dict[str, object]:
        self.calls.append("neutral")
        return {"uuid": "neutral"}


class _RunCamera:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def start(self, wait_timeout: float) -> bool:
        del wait_timeout
        return True

    def get_frame_with_metadata(self) -> tuple[np.ndarray, dict[str, int]]:
        return np.zeros((10, 10, 3), dtype=np.uint8), {"ts": 1}

    def close(self) -> None:
        self.calls.append("camera_closed")


class _EmptyDetector:
    def detect(
        self,
        frame: np.ndarray,
        *,
        target: str,
        min_confidence: float,
    ) -> list[ImageDetection]:
        del frame, target, min_confidence
        return []


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


def test_control_client_entrypoint_is_declared_without_gui_viewer() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'reachy-mini-attend = "reachy_mini.tools.attention:main"' in pyproject
    assert "reachy-mini-attend-viewer" not in pyproject


def test_detector_dependencies_are_split_by_model_family() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    extras = pyproject["project"]["optional-dependencies"]

    assert "attention" not in extras
    assert extras["attention-yolo-face"] == ["ultralytics>=8.4,<9"]
    assert extras["attention-yoloe"] == [
        "clip @ git+https://github.com/ultralytics/CLIP.git@16be45c7062240d445cce764f2afd9454a91ef7e",
        "ultralytics>=8.4,<9",
    ]
    assert extras["attention-rfdetr"] == [
        "omegaconf>=2.3,<3",
        "rfdetr>=1.3,<2",
    ]
    help_text = attention.build_parser().format_help()
    for extra in (
        "attention-yolo-face",
        "attention-yoloe",
        "attention-rfdetr",
    ):
        assert extra in help_text


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

    assert FakeApi().return_neutral(duration=0.0) == {"uuid": "move-1"}


def test_attention_stops_tracking_when_start_response_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    api = _RunApi(calls, RuntimeError("response lost after start"))
    monkeypatch.setattr(attention, "RobotApi", lambda *args: api)
    monkeypatch.setattr(
        attention,
        "ZeroMQClient",
        lambda **kwargs: _RunCamera(calls),
    )
    args = attention.build_parser().parse_args(
        ["--model", "rfdetr-nano", "--target", "person", "--follow"]
    )

    with pytest.raises(RuntimeError, match="response lost after start"):
        attention.run(args, detector=_EmptyDetector())

    assert calls == [
        "neutral",
        "/tracking/start",
        "/tracking/stop",
        "neutral",
        "camera_closed",
    ]


def test_attention_stops_tracking_even_when_stream_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeStream:
        def close(self) -> None:
            calls.append("stream_closed")
            raise RuntimeError("stream close failed")

    api = _RunApi(calls)
    monkeypatch.setattr(attention, "RobotApi", lambda *args: api)
    monkeypatch.setattr(
        attention,
        "ZeroMQClient",
        lambda **kwargs: _RunCamera(calls),
    )
    monkeypatch.setattr(attention, "connect", lambda *args, **kwargs: FakeStream())
    args = attention.build_parser().parse_args(
        [
            "--model",
            "rfdetr-nano",
            "--target",
            "person",
            "--follow",
            "--duration",
            "0.000000001",
        ]
    )

    with pytest.raises(RuntimeError, match="stream close failed"):
        attention.run(args, detector=_EmptyDetector())

    assert calls == [
        "neutral",
        "/tracking/start",
        "stream_closed",
        "/tracking/stop",
        "neutral",
        "camera_closed",
    ]


def test_neutral_wait_rejects_missing_move_uuid() -> None:
    class FakeApi(RobotApi):
        def post(self, path: str, payload: object = None) -> dict[str, object]:
            del path, payload
            return {}

    with pytest.raises(ValueError, match="move UUID"):
        FakeApi("http://robot/api").return_neutral(duration=0.0)


@pytest.mark.parametrize(
    "running_moves",
    [
        {"uuid": "move-1"},
        [{"missing_uuid": "move-1"}],
    ],
)
def test_neutral_wait_rejects_malformed_running_moves(
    running_moves: object,
) -> None:
    class FakeApi(RobotApi):
        def post(self, path: str, payload: object = None) -> dict[str, object]:
            del path, payload
            return {"uuid": "move-1"}

        def request_value(
            self, method: str, path: str, payload: object = None
        ) -> object:
            del method, path, payload
            return running_moves

    with pytest.raises(ValueError, match="/move/running"):
        FakeApi("http://robot/api").return_neutral(duration=0.0)


def test_neutral_wait_raises_when_move_remains_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi(RobotApi):
        def post(self, path: str, payload: object = None) -> dict[str, object]:
            del path, payload
            return {"uuid": "move-1"}

        def request_value(
            self, method: str, path: str, payload: object = None
        ) -> object:
            del method, path, payload
            return [{"uuid": "move-1"}]

    monotonic_values = iter([0.0, 0.0, 6.0])
    monkeypatch.setattr(
        "reachy_mini.tools._robot_api.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr("reachy_mini.tools._robot_api.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="move-1"):
        FakeApi("http://robot/api", timeout=0.0).return_neutral(duration=0.0)
