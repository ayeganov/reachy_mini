from fastapi.testclient import TestClient
import numpy as np

from reachy_mini.daemon.app.dependencies import get_backend
from reachy_mini.daemon.app.main import Args, create_app


def test_tracking_status_route_is_registered() -> None:
    app = create_app(Args(autostart=False))

    with TestClient(app) as client:
        response = client.get("/api/tracking/status")

    assert response.status_code == 503
    assert response.json()["detail"] == "Backend not running"


def test_tracking_look_at_route_accepts_metric_target() -> None:
    class FakeKinematics:
        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.full(7, 0.2)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()
            self.command: np.ndarray | None = None

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    backend = FakeBackend()
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


def test_wireless_startup_starts_visual_tracking() -> None:
    class FakeKinematics:
        def set_automatic_body_yaw(self, automatic_body_yaw: bool) -> None:
            self.automatic_body_yaw = automatic_body_yaw

        def ik(self, pose: np.ndarray, body_yaw: float = 0.0) -> np.ndarray:
            return np.zeros(7)

    class FakeBackend:
        is_move_running = False

        def __init__(self) -> None:
            self.head_kinematics = FakeKinematics()

        def get_present_head_joint_positions(self) -> np.ndarray:
            return np.zeros(7)

        def get_present_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def set_target_head_joint_positions(self, command: np.ndarray) -> None:
            self.command = command

    class FakeDaemon:
        def __init__(self) -> None:
            self.backend = FakeBackend()
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
