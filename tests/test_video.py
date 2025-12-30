import numpy as np
import pytest

from reachy_mini.media.capture import CaptureConfig, MediaCapture
from reachy_mini.media.receivers.local_ipc_receiver import LocalIPCReceiver


@pytest.fixture
def media_capture():
    """Start MediaCapture for tests that need it."""
    config = CaptureConfig()
    capture = MediaCapture(config=config)
    if not capture.start():
        pytest.skip("Could not start MediaCapture (camera not available)")
    yield capture
    capture.stop()


@pytest.fixture
def ipc_receiver(media_capture):
    """Create LocalIPCReceiver that reads from MediaCapture's IPC bus."""
    receiver = LocalIPCReceiver()
    receiver.start()
    yield receiver
    receiver.close()


@pytest.mark.video
def test_get_frame_exists(ipc_receiver) -> None:
    """Test that a frame can be retrieved from the IPC bus and is not None."""
    # Give MediaCapture time to produce frames
    import time
    time.sleep(0.5)

    frame = ipc_receiver.get_frame()
    assert frame is not None, "No frame was retrieved from the IPC bus."
    assert isinstance(frame, np.ndarray), "Frame is not a numpy array."
    assert frame.size > 0, "Frame is empty."
    assert len(frame.shape) == 3, f"Frame should be 3D (H, W, C), got shape {frame.shape}"


@pytest.mark.video
def test_video_resolution_available(ipc_receiver) -> None:
    """Test that video resolution metadata is available from the receiver."""
    # Give MediaCapture time to produce frames
    import time
    time.sleep(0.5)

    resolution = ipc_receiver.video_resolution
    assert resolution is not None, "Video resolution should be available"
    assert len(resolution) == 2, "Resolution should be (width, height)"
    assert resolution[0] > 0 and resolution[1] > 0, "Resolution dimensions should be positive"


@pytest.mark.video
def test_multiple_frames(ipc_receiver) -> None:
    """Test that multiple frames can be retrieved sequentially."""
    import time
    time.sleep(0.5)

    frames = []
    for _ in range(5):
        frame = ipc_receiver.get_frame()
        if frame is not None:
            frames.append(frame)
        time.sleep(0.05)

    assert len(frames) >= 3, f"Should get at least 3 frames, got {len(frames)}"
