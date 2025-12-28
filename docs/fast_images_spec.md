# Fast Image Delivery via picamera2 Backend

## Overview
### Problem statement and goals
The current media path uses the default `MediaManager` backend (OpenCV + SoundDevice) when streaming is enabled. We need to deliver camera frames to developers with minimal overhead, close to the camera's theoretical FPS, and avoid buffering lag when `media.get_frame()` is called in a tight loop. The new feature will add a picamera2-based camera backend and make it the default path for the daemon while preserving the existing backend choices.

### Target users and use cases
- Developers running the daemon on Reachy Mini hardware (Raspberry Pi) who need high-FPS frame access.
- Developers calling `MediaManager.get_frame()` in tight loops for downstream processing and benchmarking.

### Success criteria
- `MediaManager.get_frame()` returns the latest frame and does not exhibit lag from internal buffering.
- Tight-loop `get_frame()` calls reach FPS close to the camera's max for the selected resolution.
- The daemon uses picamera2 by default while still allowing other backends via explicit configuration.
- Errors from picamera2 initialization fail fast and surface through existing error-reporting paths.

## Requirements
### Functional requirements
- Add a picamera2-based `CameraBase` implementation that returns frames as provided by picamera2 (no JPEG conversion).
- Make the daemon default `MediaManager` camera backend use picamera2.
- Preserve existing backend choices (OpenCV, GStreamer, WebRTC) via explicit selection.
- `MediaManager.get_frame()` returns the latest frame with minimal overhead; capture occurs on each `read()` call.

### Non-functional requirements
- Minimal overhead; close to theoretical FPS for the camera.
- No additional format conversion in the capture path.
- Work within existing abstractions (`CameraBase`, `MediaManager`), with no new transport layer.
- Cross-platform clients consuming frames; publisher is hardware-bound (Pi). 

### Dependencies and prerequisites
- picamera2 available on the Raspberry Pi environment where the daemon runs.
- Existing Python environment supports the current media stack and camera device access.
- Follow existing optional-dependency pattern: add a `picamera2` extra in `pyproject.toml` and install it only on the robot image; client installs should not include this extra.

## Architecture & Design
### High-level architecture
- Add a new `CameraBase` implementation: `Picamera2Camera`.
- Integrate it with `MediaManager` so the daemon defaults to picamera2 when `stream_media` is enabled.
- Leave `OpenCVCamera`, `GStreamerCamera`, and `GstWebRTCClient` unchanged and selectable.

### Key components and responsibilities
- `reachy_mini/media/camera_picamera2.py`
  - Owns picamera2 initialization and capture.
  - Implements `open/read/close` per `CameraBase`.
  - Returns frames as provided by picamera2 (likely `numpy` arrays), without conversion.

- `reachy_mini/media/media_manager.py`
  - Adds a new backend option (or updates default path) to instantiate `Picamera2Camera`.
  - Keeps explicit selection of other backends available.

- `reachy_mini/daemon/daemon.py`
  - Uses the default `MediaManager()` instantiation as-is, but the default backend now resolves to picamera2.

### Interfaces and contracts
- `CameraBase.read() -> Optional[npt.NDArray[np.uint8]]`
  - For picamera2, return the raw frame array without conversion.
  - Maintain the same return type as other backends to avoid downstream changes.

### Data structures and types
- Frame type: `numpy.ndarray` (dtype as provided by picamera2; no conversion to JPEG).
- Resolution and camera intrinsics: align with `CameraSpecs` and `CameraResolution` where applicable.

### Integration points with existing code
- `MediaManager._init_camera()` will instantiate `Picamera2Camera` for the default backend path.
- No changes required to `Daemon.start()` beyond the default backend behavior.

## Implementation Plan
1) Add picamera2 backend
- Create `reachy_mini/media/camera_picamera2.py` implementing `CameraBase`.
- Implement `open()` to initialize picamera2 and configure resolution.
- Implement `read()` to capture a frame per call.
- Implement `close()` to shut down picamera2 cleanly.

2) Wire into MediaManager
- Add backend selection for picamera2 as the new default path.
- Preserve explicit selection of existing backends.

3) Update configuration and docs (if any)
- Add a `picamera2` optional dependency group in `pyproject.toml` following the existing extras pattern (e.g., `gstreamer`, `wireless-version`).
- Document that robot installs include `picamera2` while client installs do not.

4) Validate integration
- Ensure `MediaManager.get_frame()` returns frames from picamera2.
- Verify error propagation on failed camera initialization.

### Parallelizable work
- picamera2 backend implementation can be done independently from wiring changes in `MediaManager`.

## Testing Strategy
### Test scenarios
- Initialize daemon with `stream_media` enabled and verify `MediaManager.get_frame()` returns frames.
- Tight-loop `get_frame()` to assess FPS and confirm no buffering lag.
- Force picamera2 initialization failure (e.g., missing device) and confirm error surfaces via current error paths.
- Explicitly select OpenCV/GStreamer backends and confirm existing behavior is unchanged.

### Acceptance criteria
- `MediaManager.get_frame()` returns latest frames at near-maximum FPS without added latency.
- The daemon defaults to picamera2 capture on Pi.
- Explicit backend selection still works for OpenCV/GStreamer/WebRTC.

### Verification
- Manual benchmark: measure FPS in a tight loop calling `media.get_frame()`.
- Runtime logs confirm picamera2 backend is in use.
- Error handling matches current patterns (daemon status/error state).
