# ruff: noqa: D100,D103

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from reachy_mini.daemon.app import bg_job_register
from reachy_mini.daemon.app.dependencies import get_backend, ws_get_backend
from reachy_mini.daemon.app.main import Args, create_app
from reachy_mini.daemon.app.routers import tracking as tracking_router
from reachy_mini.daemon.app.routers.tracking import (
    _get_or_create_visual_servo,
    start_visual_servo,
)
from reachy_mini.daemon.tracking.visual_servo import VisualServoController


class _ApiKinematics:
    automatic_body_yaw = False

    def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
        self.automatic_body_yaw = automatic_body_yaw

    def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
        return np.zeros(7)


class _ApiBackend:
    is_move_running = False

    def __init__(self) -> None:
        self.head_kinematics = _ApiKinematics()
        self.command: np.ndarray | None = None

    def get_present_head_joint_positions(self) -> np.ndarray:
        return np.zeros(7)

    def get_present_head_pose(self) -> np.ndarray:
        return np.eye(4)

    def set_target_head_joint_positions(self, command: np.ndarray) -> None:
        self.command = command

    def _try_start_move(self) -> bool:
        if self.is_move_running:
            return False
        self.is_move_running = True
        return True

    def _end_move(self) -> None:
        self.is_move_running = False


def test_tracking_start_uses_centralized_defaults() -> None:
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: _ApiBackend()

    with TestClient(app) as client:
        response = client.post("/api/tracking/start", json={})
        assert app.state.visual_servo is not None
        config = app.state.visual_servo.config

    assert response.status_code == 200
    assert config.control_frequency == 50.0
    assert config.max_detection_age == 0.35
    assert config.joint_safety_margin == 0.1745329252
    assert config.max_joint_velocity == 0.60
    assert config.max_joint_acceleration == 2.40
    assert config.max_joint_jerk == 16.0
    assert config.look_at_profile_response_hz == 2.0
    assert config.image_horizontal_fov == math.radians(98.88965079926311)
    assert config.image_vertical_fov == math.radians(66.67916209122708)
    assert config.image_error_upward_elevation_limit == math.radians(30.0)
    assert config.image_error_downward_elevation_limit == math.radians(25.0)


def test_concurrent_tracking_starts_leave_one_running_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_start_entered = threading.Event()
    release_first_start = threading.Event()
    second_finished = threading.Event()
    controllers = []

    class FakeController:
        def __init__(self, **_kwargs: object) -> None:
            self.running = False
            controllers.append(self)

        def start(self) -> None:
            if len(controllers) == 1:
                first_start_entered.set()
                assert release_first_start.wait(timeout=1.0)
            self.running = True

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(tracking_router, "VisualServoController", FakeController)
    app_state = SimpleNamespace(visual_servo=None)
    backend = _ApiBackend()
    failures: list[BaseException] = []

    def start_controller(done: threading.Event | None = None) -> None:
        try:
            start_visual_servo(app_state, backend)  # type: ignore[arg-type]
        except BaseException as exc:  # pragma: no cover - exposed by assertion below
            failures.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=start_controller)
    second = threading.Thread(target=start_controller, args=(second_finished,))
    first.start()
    assert first_start_entered.wait(timeout=1.0)
    second.start()

    assert not second_finished.wait(timeout=0.05)
    release_first_start.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert failures == []
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(controllers) == 2
    assert sum(controller.running for controller in controllers) == 1
    assert app_state.visual_servo is controllers[-1]


def test_tracking_start_failure_cleans_candidate_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = []

    class FailingController:
        def __init__(self, **_kwargs: object) -> None:
            self.running = False
            self.stop_called = False
            candidates.append(self)

        def start(self) -> None:
            self.running = True
            raise RuntimeError("start failed")

        def stop(self) -> None:
            self.stop_called = True
            self.running = False

    monkeypatch.setattr(tracking_router, "VisualServoController", FailingController)
    app_state = SimpleNamespace(visual_servo=None)

    with pytest.raises(RuntimeError, match="start failed"):
        start_visual_servo(app_state, _ApiBackend())  # type: ignore[arg-type]

    assert len(candidates) == 1
    assert candidates[0].stop_called
    assert not candidates[0].running
    assert app_state.visual_servo is None


def test_tracking_status_route_is_registered() -> None:
    app = create_app(Args(autostart=False))

    with TestClient(app) as client:
        response = client.get("/api/tracking/status")

    assert response.status_code == 503
    assert response.json()["detail"] == "Backend not running"


def test_tracking_look_at_route_accepts_metric_target() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: backend

    with TestClient(app) as client:
        client.post("/api/tracking/start", json={})
        response = client.post(
            "/api/tracking/look_at",
            json={"x": 0.5, "y": 0.0, "z": 0.15, "confidence": 1.0},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["accepted_look_at_targets"] == 1


def test_tracking_detection_rejects_invalid_camera_dimensions() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: backend

    with TestClient(app) as client:
        client.post("/api/tracking/start", json={})
        response = client.post(
            "/api/tracking/detection",
            json={"u": 100.0, "v": 100.0, "width": 0, "height": 720},
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"u": 1280.01, "v": 360.0, "width": 1280, "height": 720},
        {"u": 640.0, "v": 720.01, "width": 1280, "height": 720},
        {"u": 640.0, "v": 360.0, "width": 8193, "height": 720},
        {"u": 640.0, "v": 360.0, "width": 1280, "height": 8193},
    ],
)
def test_tracking_detection_rejects_out_of_frame_or_oversized_geometry(
    payload: dict[str, float | int],
) -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: backend

    with TestClient(app) as client:
        response = client.post("/api/tracking/detection", json=payload)

    assert response.status_code == 422


def test_detection_websocket_rejects_out_of_frame_centroid_without_submission() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[ws_get_backend] = lambda: backend

    with TestClient(app) as client:
        with client.websocket_connect("/api/tracking/ws/detections") as websocket:
            websocket.send_json(
                {"u": 2560.0, "v": 1440.0, "width": 1280, "height": 720}
            )
            response = websocket.receive_json()
            accepted_detections = app.state.visual_servo.status()["accepted_detections"]

    assert response["status"] == "error"
    assert accepted_detections == 0


def test_tracking_api_rejects_non_finite_values() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: backend

    with TestClient(app) as client:
        detection_response = client.post(
            "/api/tracking/detection",
            content='{"u": Infinity, "v": 100.0, "width": 1280, "height": 720}',
            headers={"Content-Type": "application/json"},
        )
        look_at_response = client.post(
            "/api/tracking/look_at",
            content='{"x": 0.5, "y": Infinity, "z": 0.15}',
            headers={"Content-Type": "application/json"},
        )
        config_response = client.post(
            "/api/tracking/start",
            content='{"max_joint_velocity": Infinity}',
            headers={"Content-Type": "application/json"},
        )

    assert detection_response.status_code == 422
    assert look_at_response.status_code == 422
    assert config_response.status_code == 422


def test_tracking_start_rejects_oversized_telemetry_capacity() -> None:
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: _ApiBackend()

    with TestClient(app) as client:
        response = client.post(
            "/api/tracking/start",
            json={"telemetry_capacity": 5001},
        )

    if app.state.visual_servo is not None:
        app.state.visual_servo.stop()
    assert response.status_code == 422


def test_tracking_start_validates_look_at_profile_response_hz() -> None:
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: _ApiBackend()

    with TestClient(app) as client:
        accepted = client.post(
            "/api/tracking/start", json={"look_at_profile_response_hz": 1.0}
        )
        rejected = [
            client.post(
                "/api/tracking/start",
                json={"look_at_profile_response_hz": value},
            )
            for value in (0.0, -1.0, 5.1)
        ]
        rejected.extend(
            [
                client.post(
                    "/api/tracking/start",
                    content=f'{{"look_at_profile_response_hz": {value}}}',
                    headers={"Content-Type": "application/json"},
                )
                for value in ("NaN", "Infinity")
            ]
        )

    if app.state.visual_servo is not None:
        app.state.visual_servo.stop()
    assert accepted.status_code == 200
    assert all(response.status_code == 422 for response in rejected)


def test_tracking_start_validates_camera_and_elevation_limits() -> None:
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: _ApiBackend()

    with TestClient(app) as client:
        accepted = client.post(
            "/api/tracking/start",
            json={
                "image_horizontal_fov": 1.5,
                "image_vertical_fov": 1.0,
                "image_error_upward_elevation_limit": 0.3,
                "image_error_downward_elevation_limit": 0.1,
            },
        )
        rejected = [
            client.post(
                "/api/tracking/start",
                json={field: value},
            )
            for field in (
                "image_horizontal_fov",
                "image_vertical_fov",
                "image_error_upward_elevation_limit",
                "image_error_downward_elevation_limit",
            )
            for value in (0.0, -0.1, np.pi)
        ]

    if app.state.visual_servo is not None:
        app.state.visual_servo.stop()
    assert accepted.status_code == 200
    assert all(response.status_code == 422 for response in rejected)


def test_wireless_startup_starts_visual_tracking() -> None:
    class FakeDaemon:
        def __init__(self) -> None:
            self.backend = _ApiBackend()
            self.started = False
            self.stopped = False

        async def start(self, **kwargs: object) -> None:
            self.started = True

        async def stop(self, **kwargs: object) -> None:
            self.stopped = True

    class FakeAppManager:
        async def close(self) -> None:
            self.closed = True

    daemon = FakeDaemon()
    app = create_app(Args())
    app.state.args.wireless_version = True
    app.state.daemon = daemon
    app.state.app_manager = FakeAppManager()

    with TestClient(app):
        assert daemon.started
        assert app.state.visual_servo is not None
        assert app.state.visual_servo.running

    assert daemon.stopped


def test_daemon_stop_route_stops_visual_tracking_before_backend_stop(
    monkeypatch: object,
) -> None:
    class FakeVisualServo:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeDaemon:
        def __init__(self, visual_servo: FakeVisualServo) -> None:
            self.visual_servo = visual_servo
            self.stopped_after_visual_servo = False

        async def stop(self, **kwargs: object) -> None:
            self.stopped_after_visual_servo = self.visual_servo.stopped

    jobs = []

    def capture_job(command: str, coro_func: object, *args: object) -> str:
        jobs.append((coro_func, args))
        return "job-id"

    monkeypatch.setattr(bg_job_register, "run_command", capture_job)

    visual_servo = FakeVisualServo()
    daemon = FakeDaemon(visual_servo)
    app = create_app(Args(autostart=False))
    app.state.visual_servo = visual_servo
    app.dependency_overrides[
        __import__(
            "reachy_mini.daemon.app.dependencies",
            fromlist=["get_daemon"],
        ).get_daemon
    ] = lambda: daemon

    with TestClient(app) as client:
        response = client.post("/api/daemon/stop?goto_sleep=false")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-id"}
    assert len(jobs) == 1

    import asyncio
    import logging

    coro_func, args = jobs[0]
    asyncio.run(coro_func(*args, logger=logging.getLogger("test")))

    assert visual_servo.stopped
    assert app.state.visual_servo is None
    assert daemon.stopped_after_visual_servo


def test_daemon_restart_route_replaces_visual_tracking_after_backend_restart(
    monkeypatch: object,
) -> None:
    class FakeVisualServo:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeDaemon:
        def __init__(self, visual_servo: FakeVisualServo) -> None:
            self.backend = _ApiBackend()
            self.visual_servo = visual_servo
            self.restarted_after_visual_servo = False

        async def restart(self) -> None:
            self.restarted_after_visual_servo = self.visual_servo.stopped
            self.backend = _ApiBackend()

        async def stop(self, **kwargs: object) -> None:
            pass

    jobs = []

    def capture_job(command: str, coro_func: object, *args: object) -> str:
        jobs.append((coro_func, args))
        return "job-id"

    monkeypatch.setattr(bg_job_register, "run_command", capture_job)

    visual_servo = FakeVisualServo()
    daemon = FakeDaemon(visual_servo)
    app = create_app(Args(autostart=False, wireless_version=False))
    app.state.args.wireless_version = True
    app.state.visual_servo = visual_servo
    app.dependency_overrides[
        __import__(
            "reachy_mini.daemon.app.dependencies",
            fromlist=["get_daemon"],
        ).get_daemon
    ] = lambda: daemon

    with TestClient(app) as client:
        response = client.post("/api/daemon/restart")

    assert response.status_code == 200
    assert len(jobs) == 1

    import asyncio
    import logging

    coro_func, args = jobs[0]
    asyncio.run(coro_func(*args, logger=logging.getLogger("test")))

    assert visual_servo.stopped
    assert daemon.restarted_after_visual_servo
    assert isinstance(app.state.visual_servo, VisualServoController)
    assert app.state.visual_servo.backend is daemon.backend
    app.state.visual_servo.stop()


def test_replacing_visual_servo_backend_stops_old_controller() -> None:
    class AppState:
        pass

    old_backend = _ApiBackend()
    new_backend = _ApiBackend()
    app_state = AppState()
    old_controller = VisualServoController(backend=old_backend)  # type: ignore[arg-type]
    old_controller.start()
    app_state.visual_servo = old_controller

    new_controller = _get_or_create_visual_servo(app_state, new_backend)  # type: ignore[arg-type]

    assert old_controller.status()["last_reason"] == "stopped"
    assert not old_controller.running
    assert new_controller.backend is new_backend


def test_detection_websocket_closes_when_controller_is_replaced() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    app.dependency_overrides[get_backend] = lambda: backend
    app.dependency_overrides[ws_get_backend] = lambda: backend

    with TestClient(app) as client:
        with client.websocket_connect("/api/tracking/ws/detections") as websocket:
            websocket.send_json({"u": 10.0, "v": 10.0})
            assert websocket.receive_json() == {"status": "accepted"}
            old_controller = app.state.visual_servo

            response = client.post("/api/tracking/start", json={})
            assert response.status_code == 200
            new_controller = app.state.visual_servo
            assert new_controller is not old_controller

            websocket.send_json({"u": 20.0, "v": 20.0})
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_json()

    assert exc_info.value.code == 1012
    assert old_controller.status()["accepted_detections"] == 1
    assert new_controller.status()["accepted_detections"] == 0


def test_tracking_telemetry_route_does_not_create_controller() -> None:
    app = create_app(Args(autostart=False))

    with TestClient(app) as client:
        response = client.get("/api/tracking/telemetry")

    assert response.status_code == 404
    assert "Visual servo controller is not running" in response.json()["detail"]
    assert app.state.visual_servo is None


def test_tracking_telemetry_route_rejects_invalid_query_without_controller() -> None:
    app = create_app(Args(autostart=False))

    with TestClient(app) as client:
        range_response = client.get(
            "/api/tracking/telemetry",
            params={"from": 20.0, "to": 10.0},
        )
        limit_response = client.get(
            "/api/tracking/telemetry",
            params={"limit": 5001},
        )

    assert range_response.status_code == 422
    assert limit_response.status_code == 422
    assert app.state.visual_servo is None


def test_tracking_telemetry_route_filters_records() -> None:
    backend = _ApiBackend()
    app = create_app(Args(autostart=False))
    controller = VisualServoController(backend=backend)  # type: ignore[arg-type]
    controller.telemetry.append({"timestamp": 10.0, "sequence": 0, "reason": "a"})
    controller.telemetry.append({"timestamp": 11.0, "sequence": 1, "reason": "b"})
    controller.telemetry.append({"timestamp": 12.0, "sequence": 2, "reason": "c"})
    app.state.visual_servo = controller

    with TestClient(app) as client:
        response = client.get(
            "/api/tracking/telemetry",
            params={
                "from": 10.5,
                "to": 11.5,
                "from_sequence": 1,
                "to_sequence": 2,
                "limit": 5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["returned"] == 1
    assert body["records"][0]["sequence"] == 1
    assert body["oldest_sequence"] == 0
    assert body["newest_sequence"] == 2


def test_tracking_telemetry_route_rejects_invalid_query() -> None:
    app = create_app(Args(autostart=False))
    app.state.visual_servo = VisualServoController(backend=_ApiBackend())  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.get(
            "/api/tracking/telemetry",
            params={"from": 20.0, "to": 10.0},
        )

    assert response.status_code == 422


def test_tracking_telemetry_route_rejects_non_finite_query() -> None:
    app = create_app(Args(autostart=False))
    app.state.visual_servo = VisualServoController(backend=_ApiBackend())  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.get("/api/tracking/telemetry?from=Infinity")

    assert response.status_code == 422
