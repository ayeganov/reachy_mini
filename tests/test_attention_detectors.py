# ruff: noqa: D100,D101,D102,D103,D107

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def detection(
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 50.0,
    y2: float = 80.0,
    *,
    confidence: float = 0.9,
    label: str = "face",
) -> ImageDetection:
    return ImageDetection(x1, y1, x2, y2, confidence, label)


def test_detection_contract_reports_centroid_and_area() -> None:
    result = detection()

    assert result.centroid == (30.0, 50.0)
    assert result.area == 2400.0


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
def test_detection_contract_rejects_invalid_values(kwargs: dict[str, object]) -> None:
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


def test_registry_contains_only_product_models() -> None:
    assert set(available_models()) == {
        "yolo-face",
        "rfdetr-nano",
        "rfdetr-large",
        "yoloe-26x",
    }
    assert supported_targets_for_model("yolo-face") == frozenset({"face"})
    assert supported_targets_for_model("yoloe-26x") is None
    with pytest.raises(ValueError, match="unknown model"):
        create_detector("yellow")


def test_target_normalization_and_selection() -> None:
    assert normalize_target("  CELL_phone ") == "cell phone"
    candidates = [
        detection(0, 0, 20, 20, confidence=0.99),
        detection(40, 40, 90, 90, confidence=0.6),
    ]

    assert (
        select_detection(candidates, strategy="largest", width=200, height=200)
        is candidates[1]
    )
    assert (
        select_detection(candidates, strategy="confidence", width=200, height=200)
        is candidates[0]
    )
    assert (
        select_detection(candidates, strategy="center", width=200, height=200)
        is candidates[1]
    )


class FakeYolo:
    def __call__(
        self, frame: np.ndarray, *, verbose: bool, conf: float
    ) -> list[SimpleNamespace]:
        assert frame.shape == (100, 200, 3)
        assert verbose is False
        assert conf == 0.5
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [1.0, 2.0, 3.0, 4.0]]),
                    conf=np.array([0.9, 0.4]),
                )
            )
        ]


def test_yolo_face_adapter_normalizes_model_output() -> None:
    detector = YoloFaceDetector(weights="unused.pt", model=FakeYolo())

    assert detector.detect(
        np.zeros((100, 200, 3), dtype=np.uint8),
        target="face",
        min_confidence=0.5,
    ) == [detection()]


class FakeYoloE:
    def __init__(self) -> None:
        self.configured_classes: list[list[str]] = []

    def set_classes(self, classes: list[str]) -> None:
        self.configured_classes.append(classes)

    def predict(
        self, frame: np.ndarray, *, verbose: bool, conf: float
    ) -> list[SimpleNamespace]:
        assert frame.shape == (100, 200, 3)
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=np.array([[10.0, 20.0, 50.0, 80.0]]),
                    conf=np.array([0.9]),
                )
            )
        ]


def test_yoloe_adapter_configures_arbitrary_target_once() -> None:
    model = FakeYoloE()
    detector = YoloEDetector(model=model)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    first = detector.detect(frame, target=" FACE ", min_confidence=0.5)
    second = detector.detect(frame, target="face", min_confidence=0.5)

    assert first == second == [detection()]
    assert model.configured_classes == [["face"]]


class FakeRfDetr:
    def predict(self, frame: np.ndarray, *, threshold: float) -> SimpleNamespace:
        assert frame.shape == (100, 200, 3)
        return SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [60.0, 10.0, 90.0, 40.0]]),
            confidence=np.array([0.9, 0.8]),
            class_id=np.array([1, 47]),
        )


def test_rfdetr_adapter_returns_only_requested_class() -> None:
    detector = RfDetrDetector(
        size="nano",
        model=FakeRfDetr(),
        class_names={1: "person", 47: "cup"},
    )

    assert detector.detect(
        np.zeros((100, 200, 3), dtype=np.uint8),
        target="person",
        min_confidence=0.5,
    ) == [detection(label="person")]


def test_factory_rejects_missing_weights_without_loading_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--weights is required"):
        create_detector("yolo-face")
    with pytest.raises(ValueError, match="model weights do not exist"):
        create_detector("rfdetr-nano", weights=tmp_path / "missing.pth")
