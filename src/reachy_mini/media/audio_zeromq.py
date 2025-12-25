"""ZeroMQ audio implementation for remote media streaming.

Wraps ZeroMQReceiver to provide the AudioBase interface for seamless
integration with MediaManager.
"""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from reachy_mini.media.audio_base import AudioBase
from reachy_mini.media.receivers import ZeroMQReceiver
from reachy_mini.media.receivers.zeromq_receiver import ZeroMQReceiverConfig


class ZeroMQAudio(AudioBase):
    """Audio implementation using ZeroMQ TCP streaming.

    Connects to a remote ZeroMQ publisher (typically the daemon's
    ZeroMQPublisher) and provides audio through the AudioBase interface.

    Note: Some operations like play_sound require local hardware access
    and are not supported over ZeroMQ. Use the daemon's audio playback
    via Zenoh commands instead.

    Example:
        audio = ZeroMQAudio(host="192.168.1.100")
        audio.start_recording()

        sample = audio.get_audio_sample()
        if sample is not None:
            # Process audio samples
            pass

        audio.stop_recording()

    """

    def __init__(
        self,
        host: str = "localhost",
        audio_port: int = 5556,
        log_level: str = "INFO",
    ) -> None:
        """Initialize ZeroMQ audio.

        Args:
            host: Remote host address where ZeroMQPublisher is running.
            audio_port: TCP port for audio subscription.
            log_level: Logging level string.

        """
        self._host = host
        self._audio_port = audio_port
        self._log_level = log_level

        self._receiver: Optional[ZeroMQReceiver] = None
        self._is_recording = False
        self._is_playing = False

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        self._respeaker = None

    def start_recording(self) -> None:
        """Start receiving audio from ZeroMQ stream."""
        if self._is_recording:
            self.logger.warning("Already recording")
            return

        config = ZeroMQReceiverConfig(
            host=self._host,
            audio_port=self._audio_port,
            video_enabled=False,
            audio_enabled=True,
            log_level=self._log_level,
        )
        self._receiver = ZeroMQReceiver(config=config, log_level=self._log_level)

        if not self._receiver.start():
            raise RuntimeError(
                f"Failed to connect to ZeroMQ audio publisher at "
                f"{self._host}:{self._audio_port}"
            )

        self._is_recording = True
        self.logger.info(
            "ZeroMQ audio connected to %s:%s", self._host, self._audio_port
        )

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Get audio samples from the ZeroMQ stream.

        Returns:
            Audio samples as numpy array or None if no data available.

        """
        if self._receiver is None or not self._is_recording:
            return None

        return self._receiver.get_audio_sample()

    def stop_recording(self) -> None:
        """Stop receiving audio from ZeroMQ stream."""
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None

        self._is_recording = False
        self.logger.info("ZeroMQ audio recording stopped")

    def start_playing(self) -> None:
        """Start audio playback (not supported over ZeroMQ)."""
        self.logger.warning(
            "Audio playback is not supported over ZeroMQ. "
            "Use daemon commands for remote audio playback."
        )
        self._is_playing = True

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio data to remote (not supported over ZeroMQ).

        Args:
            data: Audio samples to push.

        """
        self.logger.warning(
            "Pushing audio samples is not supported over ZeroMQ. "
            "Use daemon commands for remote audio playback."
        )

    def stop_playing(self) -> None:
        """Stop audio playback."""
        self._is_playing = False

    def play_sound(self, sound_file: str) -> None:
        """Play a sound file (not supported over ZeroMQ).

        Args:
            sound_file: Path to the sound file.

        """
        self.logger.warning(
            "Cannot play sound '%s' over ZeroMQ. "
            "Use daemon commands for remote audio playback.",
            sound_file,
        )

    def get_input_audio_samplerate(self) -> int:
        """Get the input sample rate."""
        if self._receiver is not None and self._receiver.audio_sample_rate:
            return self._receiver.audio_sample_rate
        return self.SAMPLE_RATE

    @property
    def is_connected(self) -> bool:
        """Check if actively receiving audio from publisher."""
        if self._receiver is None:
            return False
        return self._receiver.is_connected
