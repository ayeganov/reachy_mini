import pytest

from reachy_mini.daemon.backend.abstract import Backend, MotorControlMode


def test_wrapped_run_clears_ready_when_backend_crashes() -> None:
    class CrashingBackend(Backend):
        def __init__(self) -> None:
            super().__init__(use_audio=False)
            self.closed = False

        def run(self) -> None:
            self.ready.set()
            raise RuntimeError("channel closed")

        def close(self) -> None:
            self.closed = True

        def get_video_capture(self) -> object:
            raise NotImplementedError

        def get_audio_capture(self) -> object:
            raise NotImplementedError

        def get_motor_control_mode(self) -> MotorControlMode:
            return MotorControlMode.Enabled

        def set_motor_control_mode(self, mode: MotorControlMode) -> None:
            pass

        def set_motor_torque_ids(self, ids: list[str], on: bool) -> None:
            pass

    backend = CrashingBackend()

    with pytest.raises(RuntimeError, match="channel closed"):
        backend.wrapped_run()

    assert backend.closed
    assert backend.error == "channel closed"
    assert not backend.ready.is_set()
