"""Visual tracking API routes.

These routes accept 2D detections from an external perception process while the
robot-side daemon owns smoothing, safety constraints, and motor target updates.
"""

import json
import math
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field, FiniteFloat, NonNegativeInt

from ....daemon.backend.abstract import Backend
from ....daemon.tracking.telemetry import TelemetryQuery
from ....daemon.tracking.visual_servo import (
    TrackingDetection,
    TrackingLookAtTarget,
    VisualServoConfig,
    VisualServoController,
)
from ..dependencies import get_backend, ws_get_backend

router = APIRouter(prefix="/tracking")


class TrackingDetectionRequest(BaseModel):
    """2D detection produced by an external perception model."""

    u: FiniteFloat = Field(ge=0.0)
    v: FiniteFloat = Field(ge=0.0)
    timestamp: FiniteFloat | None = None
    confidence: FiniteFloat = Field(default=1.0, ge=0.0, le=1.0)
    frame_id: int | None = None
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)

    def to_detection(self) -> TrackingDetection:
        """Convert request into a TrackingDetection."""
        kwargs: dict[str, Any] = {
            "u": self.u,
            "v": self.v,
            "confidence": self.confidence,
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
        }
        if self.timestamp is not None:
            kwargs["timestamp"] = self.timestamp
        return TrackingDetection(**kwargs)


class TrackingLookAtRequest(BaseModel):
    """Metric 3D look-at target in the robot/world frame."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    timestamp: FiniteFloat | None = None
    confidence: FiniteFloat = Field(default=1.0, ge=0.0, le=1.0)
    frame_id: int | None = None

    def to_target(self) -> TrackingLookAtTarget:
        """Convert request into a TrackingLookAtTarget."""
        kwargs: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "frame_id": self.frame_id,
        }
        if self.timestamp is not None:
            kwargs["timestamp"] = self.timestamp
        return TrackingLookAtTarget(**kwargs)


class VisualServoConfigRequest(BaseModel):
    """Runtime configuration for the visual servo loop."""

    control_frequency: FiniteFloat = Field(default=50.0, gt=0.0)
    min_confidence: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    max_detection_age: FiniteFloat = Field(default=0.35, gt=0.0)
    smoothing_alpha: FiniteFloat = Field(default=0.35, gt=0.0, le=1.0)
    lookahead_distance: FiniteFloat = Field(default=0.5, gt=0.0)
    image_error_max_correction: FiniteFloat = Field(
        default=math.radians(8.0), gt=0.0, lt=math.pi / 2.0
    )
    image_error_elevation_limit: FiniteFloat = Field(
        default=math.atan2(0.2, 0.5), gt=0.0, lt=math.pi / 2.0
    )
    joint_safety_margin: FiniteFloat = Field(default=0.08726646259971647, ge=0.0)
    max_joint_velocity: FiniteFloat = Field(default=1.3962634015954636, gt=0.0)
    max_joint_acceleration: FiniteFloat = Field(default=5.235987755982989, gt=0.0)
    max_joint_jerk: FiniteFloat = Field(default=34.90658503988659, gt=0.0)
    look_at_profile_response_hz: FiniteFloat = Field(default=1.0, gt=0.0, le=5.0)
    automatic_body_yaw: bool = True
    telemetry_capacity: int = Field(default=3000, gt=0, le=5000)

    def to_config(self) -> VisualServoConfig:
        """Convert request into controller config."""
        return VisualServoConfig(**self.model_dump())


def _get_or_create_visual_servo(
    app_state: Any,
    backend: Backend,
) -> VisualServoController:
    controller = getattr(app_state, "visual_servo", None)
    if isinstance(controller, VisualServoController) and controller.backend is backend:
        return controller
    if controller is not None:
        stop_visual_servo(app_state)
    controller = getattr(app_state, "visual_servo", None)
    if controller is None:
        controller = VisualServoController(backend=backend)
        app_state.visual_servo = controller
    assert isinstance(controller, VisualServoController)
    return controller


def stop_visual_servo(app_state: Any) -> Any | None:
    """Stop and detach the app-owned visual servo controller."""
    controller = getattr(app_state, "visual_servo", None)
    if controller is None:
        return None

    stop = getattr(controller, "stop", None)
    if callable(stop):
        stop()

    if bool(getattr(controller, "running", False)):
        raise RuntimeError("Visual servo controller did not stop within timeout.")

    app_state.visual_servo = None
    return controller


def start_visual_servo(
    app_state: Any,
    backend: Backend,
    config: VisualServoConfig | None = None,
) -> VisualServoController:
    """Start a fresh visual servo controller for the app backend."""
    if getattr(app_state, "visual_servo", None) is not None:
        stop_visual_servo(app_state)

    controller = VisualServoController(
        backend=backend,
        config=config or VisualServoConfigRequest().to_config(),
    )
    app_state.visual_servo = controller
    controller.start()
    return controller


def get_visual_servo(
    request: Request,
    backend: Backend = Depends(get_backend),
) -> VisualServoController:
    """Get daemon-local visual servo controller."""
    return _get_or_create_visual_servo(request.app.state, backend)


@router.get("/status")
async def status(
    controller: VisualServoController = Depends(get_visual_servo),
) -> dict[str, Any]:
    """Return visual servo status."""
    return controller.status()


@router.get("/telemetry")
async def telemetry(
    request: Request,
    from_timestamp: FiniteFloat | None = Query(default=None, alias="from"),
    to_timestamp: FiniteFloat | None = Query(default=None, alias="to"),
    from_sequence: NonNegativeInt | None = None,
    to_sequence: NonNegativeInt | None = None,
    limit: NonNegativeInt = 1000,
) -> dict[str, Any]:
    """Return retained visual servo telemetry records."""
    try:
        query = TelemetryQuery(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            from_sequence=from_sequence,
            to_sequence=to_sequence,
            limit=limit,
        )
        query.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    controller = getattr(request.app.state, "visual_servo", None)
    if not isinstance(controller, VisualServoController):
        raise HTTPException(
            status_code=404,
            detail="Visual servo controller is not running; no telemetry buffer exists.",
        )
    return controller.telemetry.query(
        from_timestamp=query.from_timestamp,
        to_timestamp=query.to_timestamp,
        from_sequence=query.from_sequence,
        to_sequence=query.to_sequence,
        limit=query.limit,
    )


@router.post("/start")
async def start(
    request: Request,
    config: VisualServoConfigRequest = Body(default_factory=VisualServoConfigRequest),
    backend: Backend = Depends(get_backend),
) -> dict[str, Any]:
    """Start the robot-side visual servo controller."""
    controller = start_visual_servo(
        app_state=request.app.state,
        backend=backend,
        config=config.to_config(),
    )
    return controller.status()


@router.post("/stop")
async def stop(
    request: Request,
    controller: VisualServoController = Depends(get_visual_servo),
) -> dict[str, Any]:
    """Stop the robot-side visual servo controller."""
    stop_visual_servo(request.app.state)
    return controller.status()


@router.post("/detection")
async def detection(
    detection_req: TrackingDetectionRequest,
    controller: VisualServoController = Depends(get_visual_servo),
) -> dict[str, Any]:
    """Submit a 2D detection to the latest-only robot-side buffer."""
    controller.submit(detection_req.to_detection())
    return {"status": "accepted", **controller.status()}


@router.post("/look_at")
async def look_at(
    target_req: TrackingLookAtRequest,
    controller: VisualServoController = Depends(get_visual_servo),
) -> dict[str, Any]:
    """Submit a metric 3D look-at target to the latest-only robot-side buffer."""
    controller.submit_look_at(target_req.to_target())
    return {"status": "accepted", **controller.status()}


@router.websocket("/ws/detections")
async def ws_detections(
    websocket: WebSocket,
    backend: Backend = Depends(ws_get_backend),
) -> None:
    """Accept streamed 2D detections over a WebSocket."""
    await websocket.accept()
    controller = _get_or_create_visual_servo(websocket.app.state, backend)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                detection_req = TrackingDetectionRequest.model_validate_json(data)
                controller.submit(detection_req.to_detection())
                await websocket.send_text(json.dumps({"status": "accepted"}))
            except Exception as exc:
                await websocket.send_text(
                    json.dumps({"status": "error", "detail": str(exc)})
                )
    except WebSocketDisconnect:
        pass
