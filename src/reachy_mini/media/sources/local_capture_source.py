"""Local audio capture source for client-side microphone input.

This source captures audio from the local microphone using SoundDevice
and provides it through the MediaSourceProtocol interface.
"""

from __future__ import annotations

import logging
import time
from queue import Empty, Full, Queue
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.capture import AudioMetadata
from reachy_mini.media.publishers.base import AudioData, MediaChunk


class LocalAudioCaptureSource:
    """Reads audio from local microphone using SoundDevice.

    Wraps callback-based capture into pull-based interface using internal queue.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        chunk_size: int = 1024,
        buffer_size: int = 100,
        device: Optional[int] = None,
        log_level: str = "INFO",
    ) -> None:
        """Initialize local audio capture source.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            chunk_size: Samples per audio chunk.
            buffer_size: Maximum number of chunks to buffer.
            device: Audio device index (None for default).
            log_level: Logging level string.

        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device = device
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)

        self._buffer: Queue[MediaChunk[AudioData, AudioMetadata]] = Queue(
            maxsize=buffer_size
        )
        self._stream: Optional[Any] = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if source is open and ready."""
        return self._is_open

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Get number of audio channels."""
        return self._channels

    def open(self) -> bool:
        """Initialize and start the audio capture stream."""
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self._device,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_open = True
            self._logger.info(
                "Local audio capture started: %dHz, %dch, device=%s",
                self._sample_rate,
                self._channels,
                self._device,
            )
            return True

        except ImportError:
            self._logger.error("SoundDevice not available")
            return False
        except Exception as e:
            self._logger.error("Failed to open local audio capture: %s", e)
            self.close()
            return False

    def _audio_callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """SoundDevice callback - pushes to internal buffer."""
        if status:
            self._logger.debug("Audio callback status: %s", status)

        chunk = MediaChunk(
            data=indata.copy().astype(np.float32),
            metadata=AudioMetadata(
                ts=time.monotonic(),
                sample_rate=self._sample_rate,
                channels=self._channels,
                samples=frames,
                dtype="float32",
            ),
        )
        try:
            self._buffer.put_nowait(chunk)
        except Full:
            # Drop oldest chunk if buffer is full
            try:
                self._buffer.get_nowait()
                self._buffer.put_nowait(chunk)
            except Empty:
                pass

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        if not self._is_open:
            return False
        # For callback-based source, we just check if buffer has data
        # Could implement actual waiting with timeout if needed
        return not self._buffer.empty()

    def read(self) -> Optional[MediaChunk[AudioData, AudioMetadata]]:
        """Read next audio chunk from buffer (non-blocking)."""
        try:
            return self._buffer.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        """Release resources."""
        self._is_open = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self._logger.warning("Error closing audio stream: %s", e)
            finally:
                self._stream = None

        # Clear buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break

        self._logger.info("Local audio capture closed")
