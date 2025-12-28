"""Base protocols and types for media receivers.

This module defines the MediaSource protocol that all receivers must implement,
enabling seamless swapping between local (direct IPC) and remote (ZMQ TCP) modes.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MediaClient(Protocol):
    """Protocol defining the unified interface for a high-level media client.

    This interface allows different receiver implementations (Local IPC,
    ZeroMQ TCP) to be swapped seamlessly in the client SDK.

    Example:
        def process_media(source: MediaClient) -> None:
            source.start(wait_timeout=2.0)

            frame = source.get_frame()
            if frame is not None:
                # Process video frame
                pass

            audio = source.get_audio_sample()
            if audio is not None:
                # Process audio sample
                pass

            source.close()

    """

    def start(self, wait_timeout: float = 0.0) -> bool:
        """Start receiving media.

        Args:
            wait_timeout: If > 0, wait up to this many seconds for
                the first data to arrive before returning.

        Returns:
            True if started successfully (and connected if wait_timeout > 0).

        """
        ...

    def get_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the latest available video frame.

        Returns:
            BGR numpy array (H, W, 3) or None if no frame is available.

        """
        ...

    def get_frame_with_metadata(
        self,
    ) -> Optional[tuple[npt.NDArray[np.uint8], dict[str, object]]]:
        """Return the latest frame with its metadata.

        Returns:
            Tuple of (frame, metadata_dict) or None if no frame available.
            Metadata dict contains: ts, width, height, format.

        """
        ...

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Return the latest audio chunk.

        Returns:
            Audio samples as numpy array (samples, channels) or None
            if no audio is available.

        """
        ...

    def get_audio_sample_with_metadata(
        self,
    ) -> Optional[tuple[npt.NDArray[np.float32], dict[str, object]]]:
        """Return the latest audio chunk with its metadata.

        Returns:
            Tuple of (audio, metadata_dict) or None if no audio available.
            Metadata dict contains: ts, sample_rate, channels, samples.

        """
        ...

    def play_sound(self, asset_file: str) -> None:
        """
        Play a sound file.

        Note: Not all receivers support audio playback. Remote receivers
        may log a warning instead.

        Args:
            asset_file: Name of the asset file to play

        """
        ...

    def close(self) -> None:
        """Release resources and close connections."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if receiver is connected and receiving data."""
        ...

    @property
    def video_resolution(self) -> Optional[tuple[int, int]]:
        """Get the current video resolution (width, height)."""
        ...

    @property
    def audio_sample_rate(self) -> Optional[int]:
        """Get the current audio sample rate in Hz."""
        ...
