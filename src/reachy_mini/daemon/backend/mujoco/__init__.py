"""MuJoCo Backend for Reachy Mini Daemon."""

from dataclasses import dataclass

from reachy_mini.daemon.backend.mujoco.audio_capture import NullAudioCapture

try:
    import mujoco  # noqa: F401

    from reachy_mini.daemon.backend.mujoco.backend import (
        MujocoBackend,
        MujocoBackendStatus,
    )
    from reachy_mini.daemon.backend.mujoco.video_capture import (
        MujocoCamera,
        MujocoVideoCapture,
    )

except ImportError as error:
    print(error)

    class MujocoMockupBackend:
        """Mockup class to avoid import errors when MuJoCo is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            """Raise ImportError when trying to instantiate the class."""
            raise ImportError(
                "MuJoCo is not installed. MuJoCo backend is not available."
                " To use MuJoCo backend, please install the 'mujoco' extra dependencies"
                " with 'pip install reachy_mini[mujoco]'."
            )

    MujocoBackend = MujocoMockupBackend  # type: ignore[assignment, misc]

    @dataclass
    class MujocoMockupBackendStatus:
        """Mockup class to avoid import errors when MuJoCo is not installed."""

        pass

    MujocoBackendStatus = MujocoMockupBackendStatus  # type: ignore[assignment, misc]
    MujocoVideoCapture = MujocoMockupBackend  # type: ignore[assignment, misc]
    MujocoCamera = MujocoMockupBackend  # type: ignore[assignment, misc]

__all__ = [
    "MujocoBackend",
    "MujocoBackendStatus",
    "MujocoVideoCapture",
    "MujocoCamera",
    "NullAudioCapture",
]
