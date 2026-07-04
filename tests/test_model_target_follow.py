# ruff: noqa: D100,D103

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from model_target_follow import (  # noqa: E402
    _establish_neutral_start,
    _finish_tracking,
    annotate_frame,
    build_parser,
    detection_payload,
    latency_summary,
    mode_banner,
)

from reachy_mini.attention.detectors import (
    ImageDetection,
    RfDetrDetector,
    YoloEDetector,
    YoloFaceDetector,
    available_models,
    create_detector,
    normalize_target,
    select_detection,
    supported_targets_for_model,
)


def _detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    confidence: float = 0.8,
    label: str = "face",
) -> ImageDetection:
    return ImageDetection(x1, y1, x2, y2, confidence, label)


def test_image_detection_reports_centroid_and_area() -> None:
    result = _detection(10.0, 20.0, 30.0, 60.0)

    assert result.centroid == (20.0, 40.0)
    assert result.area == 800.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x1": 2.0, "x2": 1.0},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"x1": float("nan")},
        {"label": ""},
    ],
)
def test_image_detection_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0,
        "confidence": 0.5,
        "label": "face",
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        ImageDetection(**values)  # type: ignore[arg-type]


def test_normalize_target_accepts_cli_spelling() -> None:
    assert normalize_target("  CELL_phone ") == "cell phone"


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("largest", (65.0, 65.0)),
        ("confidence", (10.0, 10.0)),
        ("center", (65.0, 65.0)),
    ],
)
def test_select_detection_strategies(
    strategy: str, expected: tuple[float, float]
) -> None:
    candidates = [
        _detection(0, 0, 20, 20, confidence=0.99),
        _detection(40, 40, 90, 90, confidence=0.6),
        _detection(140, 140, 170, 170, confidence=0.7),
    ]

    selected = select_detection(candidates, strategy=strategy, width=200, height=200)

    assert selected is not None
    assert selected.centroid == expected


class _FakeYolo:
    def __call__(
        self, frame: np.ndarray, *, verbose: bool, conf: float
    ) -> list[SimpleNamespace]:
        assert frame.shape == (100, 200, 3)
        assert verbose is False
        assert conf == 0.5
        boxes = SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [1.0, 2.0, 3.0, 4.0]]),
            conf=np.array([0.9, 0.4]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_face_adapter_normalizes_output_and_filters_confidence() -> None:
    detector = YoloFaceDetector(weights="unused.pt", model=_FakeYolo())

    results = detector.detect(
        np.zeros((100, 200, 3), dtype=np.uint8),
        target="face",
        min_confidence=0.5,
    )

    assert results == [_detection(10.0, 20.0, 50.0, 80.0, confidence=0.9)]


class _FakeYoloE:
    def __init__(self) -> None:
        self.configured_classes: list[list[str]] = []

    def set_classes(self, classes: list[str]) -> None:
        self.configured_classes.append(classes)

    def predict(
        self, frame: np.ndarray, *, verbose: bool, conf: float
    ) -> list[SimpleNamespace]:
        assert frame.shape == (100, 200, 3)
        assert verbose is False
        assert conf == 0.5
        boxes = SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [1.0, 2.0, 3.0, 4.0]]),
            conf=np.array([0.9, 0.4]),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_yoloe_adapter_accepts_arbitrary_target_and_configures_it_once() -> None:
    model = _FakeYoloE()
    detector = YoloEDetector(model=model)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    first = detector.detect(frame, target="  FACE ", min_confidence=0.5)
    second = detector.detect(frame, target="face", min_confidence=0.5)

    assert first == second == [_detection(10.0, 20.0, 50.0, 80.0, confidence=0.9)]
    assert model.configured_classes == [["face"]]


def test_yoloe_adapter_rejects_empty_target() -> None:
    detector = YoloEDetector(model=_FakeYoloE())

    with pytest.raises(ValueError, match="target must not be empty"):
        detector.detect(
            np.zeros((100, 200, 3), dtype=np.uint8),
            target="  ",
            min_confidence=0.5,
        )


class _FakeRfDetr:
    def predict(self, frame: np.ndarray, *, threshold: float) -> SimpleNamespace:
        assert frame.shape == (100, 200, 3)
        assert threshold == 0.5
        return SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [60.0, 10.0, 90.0, 40.0]]),
            confidence=np.array([0.91, 0.88]),
            class_id=np.array([1, 47]),
        )


def test_rfdetr_adapter_filters_requested_target() -> None:
    detector = RfDetrDetector(
        size="nano",
        model=_FakeRfDetr(),
        class_names={1: "person", 47: "cup"},
    )

    people = detector.detect(
        np.zeros((100, 200, 3), dtype=np.uint8),
        target="person",
        min_confidence=0.5,
    )

    assert people == [
        _detection(10.0, 20.0, 50.0, 80.0, confidence=0.91, label="person")
    ]


def test_registry_exposes_models_without_loading_weights() -> None:
    assert set(available_models()) == {
        "yolo-face",
        "rfdetr-nano",
        "rfdetr-large",
        "yoloe-26x",
    }
    assert supported_targets_for_model("yolo-face") == frozenset({"face"})
    assert supported_targets_for_model("yoloe-26x") is None


def test_registry_requires_face_weights() -> None:
    with pytest.raises(ValueError, match="--weights is required"):
        create_detector("yolo-face")


def test_explicit_missing_model_weights_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model weights do not exist"):
        create_detector("rfdetr-nano", weights=tmp_path / "missing.pth")


def test_detection_payload_contains_only_current_centroid() -> None:
    selected = _detection(10.0, 20.0, 50.0, 80.0, confidence=0.87)

    payload = detection_payload(selected, width=1280, height=720, frame_id=42)

    assert payload == {
        "u": 30.0,
        "v": 50.0,
        "width": 1280,
        "height": 720,
        "confidence": 0.87,
        "frame_id": 42,
    }
    assert "timestamp" not in payload


def test_annotation_does_not_modify_input_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    selected = _detection(10.0, 20.0, 50.0, 80.0, confidence=0.87)

    annotated = annotate_frame(
        frame,
        [selected],
        selected,
        model_name="yolo-face",
        target="face",
        reason="detected",
        follow=False,
    )

    assert not np.array_equal(annotated, frame)
    assert np.count_nonzero(frame) == 0


def test_latency_summary_uses_milliseconds() -> None:
    assert latency_summary([]) is None
    summary = latency_summary([0.001, 0.002, 0.003])
    assert summary is not None
    assert summary["median"] == pytest.approx(2.0)
    assert summary["maximum"] == pytest.approx(3.0)


def test_parser_defaults_to_safe_headless_preview() -> None:
    args = build_parser().parse_args([])

    assert args.model == "yolo-face"
    assert args.target == "face"
    assert args.selection == "largest"
    assert args.follow is False
    assert args.visualize is False
    assert args.visualization_endpoint == "ipc:///tmp/reachy-model-target-viewer"
    assert args.visualization_fps == 15.0
    assert args.return_neutral is True


def test_mode_banner_makes_preview_and_follow_behavior_explicit() -> None:
    preview = build_parser().parse_args(
        ["--model", "rfdetr-nano", "--target", "cell phone"]
    )
    follow = build_parser().parse_args(
        ["--model", "rfdetr-nano", "--target", "cell phone", "--follow"]
    )

    assert "PREVIEW ONLY" in mode_banner(preview)
    assert "ZERO robot targets" in mode_banner(preview)
    assert "--follow" in mode_banner(preview)
    assert "FOLLOW ENABLED" in mode_banner(follow)
    assert "moving the robot to neutral" in mode_banner(follow)


def test_follow_start_moves_to_neutral_before_reading_start_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    neutral_state = {
        "head_pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
        "body_yaw": 0.0,
    }

    def fake_return_neutral(base_url: str, timeout: float) -> None:
        calls.append(("move", base_url, timeout))

    def fake_request(
        method: str,
        base_url: str,
        path: str,
        payload: object = None,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        calls.append(("read", method, base_url, path, payload, timeout))
        return neutral_state

    monkeypatch.setattr("model_target_follow._return_neutral", fake_return_neutral)
    monkeypatch.setattr("model_target_follow._request_json", fake_request)

    result = _establish_neutral_start("http://robot/api", 3.0)

    assert result == neutral_state
    assert calls == [
        ("move", "http://robot/api", 3.0),
        (
            "read",
            "GET",
            "http://robot/api",
            "/state/full?with_head_pose=true&with_head_joints=true&with_body_yaw=true",
            None,
            3.0,
        ),
    ]


def test_tracking_stop_and_neutral_survive_telemetry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        method: str,
        base_url: str,
        path: str,
        payload: object = None,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        del base_url, payload, timeout
        calls.append((method, path))
        if path.startswith("/tracking/telemetry"):
            raise RuntimeError("telemetry unavailable")
        return {}

    def fake_neutral(base_url: str, timeout: float) -> None:
        del base_url, timeout
        calls.append(("POST", "neutral"))

    monkeypatch.setattr("model_target_follow._request_json", fake_request)
    monkeypatch.setattr("model_target_follow._return_neutral", fake_neutral)

    telemetry, error = _finish_tracking(
        "http://robot/api", timeout=1.0, return_neutral=True
    )

    assert telemetry == {}
    assert isinstance(error, RuntimeError)
    assert calls == [
        ("GET", "/tracking/telemetry?limit=3000"),
        ("POST", "/tracking/stop"),
        ("POST", "neutral"),
    ]
