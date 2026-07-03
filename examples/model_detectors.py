"""Optional model adapters for the host-side target-following example."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt
from yellow_box_follow import YellowBoxDetectorConfig, detect_yellow_box

Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ImageDetection:
    """One labeled image-space bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    def __post_init__(self) -> None:
        """Validate the model-independent box contract."""
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detection values must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("detection bounds must be ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.label:
            raise ValueError("label must not be empty")

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the box center in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class Detector(Protocol):
    """Boundary implemented by every host-side detector adapter."""

    @property
    def supported_targets(self) -> frozenset[str]:
        """Return normalized target labels accepted by this detector."""

    def detect(
        self,
        frame_bgr: Frame,
        *,
        target: str,
        min_confidence: float,
    ) -> list[ImageDetection]:
        """Return matching current-frame detections."""


def normalize_target(target: str) -> str:
    """Normalize a user-facing target label."""
    return " ".join(target.strip().lower().replace("_", " ").split())


def _validate_request(
    target: str,
    min_confidence: float,
    supported_targets: frozenset[str],
) -> str:
    normalized = normalize_target(target)
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if normalized not in supported_targets:
        supported = ", ".join(sorted(supported_targets))
        raise ValueError(
            f"target {target!r} is not supported; supported targets: {supported}"
        )
    return normalized


def _validate_frame(frame_bgr: Frame) -> None:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        raise ValueError("frame must be a non-empty BGR image")


class YellowDetector:
    """Adapter around the approved yellow-box detector."""

    supported_targets = frozenset({"yellow"})

    def __init__(self) -> None:
        """Use the exact approved yellow detector configuration."""
        self.config = YellowBoxDetectorConfig()

    def detect(
        self,
        frame_bgr: Frame,
        *,
        target: str,
        min_confidence: float,
    ) -> list[ImageDetection]:
        """Convert the approved detector result to the common box contract."""
        _validate_request(target, min_confidence, self.supported_targets)
        detection, _mask = detect_yellow_box(frame_bgr, self.config)
        if detection is None:
            return []
        x, y, width, height = detection.bounding_box
        return [
            ImageDetection(
                x1=float(x),
                y1=float(y),
                x2=float(x + width),
                y2=float(y + height),
                confidence=1.0,
                label="yellow",
            )
        ]


class YoloFaceDetector:
    """Adapter for the local YOLO face checkpoint."""

    supported_targets = frozenset({"face"})

    def __init__(self, *, weights: Path | str, model: Any | None = None) -> None:
        """Load the checkpoint unless a test model was provided."""
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        self.model: Any = model

    def detect(
        self,
        frame_bgr: Frame,
        *,
        target: str,
        min_confidence: float,
    ) -> list[ImageDetection]:
        """Return face boxes above the requested confidence."""
        _validate_request(target, min_confidence, self.supported_targets)
        _validate_frame(frame_bgr)
        result = self.model(frame_bgr, verbose=False, conf=min_confidence)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _to_numpy(boxes.xyxy)
        confidence = _to_numpy(boxes.conf)
        return [
            ImageDetection(
                x1=float(bounds[0]),
                y1=float(bounds[1]),
                x2=float(bounds[2]),
                y2=float(bounds[3]),
                confidence=float(score),
                label="face",
            )
            for bounds, score in zip(xyxy, confidence, strict=True)
            if float(score) >= min_confidence
        ]


class RfDetrDetector:
    """Adapter for COCO-trained RF-DETR Nano or Large models."""

    def __init__(
        self,
        *,
        size: str,
        weights: Path | str | None = None,
        optimize: bool = False,
        model: Any | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        """Load one RF-DETR size and its COCO class mapping."""
        if size not in {"nano", "large"}:
            raise ValueError("RF-DETR size must be 'nano' or 'large'")
        if class_names is None:
            from rfdetr.util.coco_classes import COCO_CLASSES

            class_names = {
                int(key): normalize_target(value) for key, value in COCO_CLASSES.items()
            }
        else:
            class_names = {
                int(key): normalize_target(value) for key, value in class_names.items()
            }
        self.class_names = class_names
        self.supported_targets = frozenset(class_names.values())
        if model is None:
            from rfdetr import RFDETRLarge, RFDETRNano

            model_class = RFDETRNano if size == "nano" else RFDETRLarge
            kwargs = {} if weights is None else {"pretrain_weights": str(weights)}
            model = model_class(**kwargs)
            if optimize:
                model.optimize_for_inference()
        self.model: Any = model

    def detect(
        self,
        frame_bgr: Frame,
        *,
        target: str,
        min_confidence: float,
    ) -> list[ImageDetection]:
        """Return only boxes matching the requested COCO label."""
        normalized = _validate_request(target, min_confidence, self.supported_targets)
        _validate_frame(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictions: Any = self.model.predict(frame_rgb, threshold=min_confidence)
        xyxy = _to_numpy(predictions.xyxy)
        confidence = _to_numpy(predictions.confidence)
        class_id = _to_numpy(predictions.class_id)
        detections: list[ImageDetection] = []
        for bounds, score, raw_class_id in zip(xyxy, confidence, class_id, strict=True):
            label = self.class_names.get(int(raw_class_id))
            if label != normalized or float(score) < min_confidence:
                continue
            detections.append(
                ImageDetection(
                    x1=float(bounds[0]),
                    y1=float(bounds[1]),
                    x2=float(bounds[2]),
                    y2=float(bounds[3]),
                    confidence=float(score),
                    label=label,
                )
            )
        return detections


MODEL_DESCRIPTIONS = {
    "yellow": "approved HSV yellow reference detector",
    "yolo-face": "YOLO face detector",
    "rfdetr-nano": "RF-DETR Nano COCO detector",
    "rfdetr-large": "RF-DETR Large COCO detector",
}


def available_models() -> dict[str, str]:
    """Return registered model names and descriptions."""
    return dict(MODEL_DESCRIPTIONS)


def supported_targets_for_model(model_name: str) -> frozenset[str]:
    """Return labels without loading model weights."""
    if model_name == "yellow":
        return YellowDetector.supported_targets
    if model_name == "yolo-face":
        return YoloFaceDetector.supported_targets
    if model_name in {"rfdetr-nano", "rfdetr-large"}:
        from rfdetr.util.coco_classes import COCO_CLASSES

        return frozenset(normalize_target(value) for value in COCO_CLASSES.values())
    raise ValueError(f"unknown model {model_name!r}")


def create_detector(
    model_name: str,
    *,
    weights: Path | None = None,
    optimize: bool = False,
) -> Detector:
    """Create one detector from the built-in registry."""
    if model_name == "yellow":
        if weights is not None:
            raise ValueError("the yellow detector does not accept model weights")
        return YellowDetector()
    if model_name == "yolo-face":
        if weights is None:
            raise ValueError("--weights is required for yolo-face")
        return YoloFaceDetector(weights=weights)
    if model_name == "rfdetr-nano":
        return RfDetrDetector(size="nano", weights=weights, optimize=optimize)
    if model_name == "rfdetr-large":
        return RfDetrDetector(size="large", weights=weights, optimize=optimize)
    raise ValueError(f"unknown model {model_name!r}")


def select_detection(
    detections: list[ImageDetection],
    *,
    strategy: str,
    width: int,
    height: int,
) -> ImageDetection | None:
    """Select one current-frame detection without stale identity state."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not detections:
        return None
    if strategy == "largest":
        return max(detections, key=lambda detection: detection.area)
    if strategy == "confidence":
        return max(detections, key=lambda detection: detection.confidence)
    if strategy == "center":
        center_x = width / 2.0
        center_y = height / 2.0
        return min(
            detections,
            key=lambda detection: (
                (detection.centroid[0] - center_x) ** 2
                + (detection.centroid[1] - center_y) ** 2
            ),
        )
    raise ValueError(f"unknown selection strategy {strategy!r}")


def _to_numpy(value: Any) -> npt.NDArray[np.generic]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return cast(npt.NDArray[np.generic], np.asarray(value))
