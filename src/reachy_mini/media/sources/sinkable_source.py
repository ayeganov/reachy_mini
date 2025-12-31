"""A sink/source hybrid for in-memory queuing."""

from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from typing import Optional

from reachy_mini.media.capture import AudioMetadata
from reachy_mini.media.publishers.base import (
    AudioData,
    MediaChunk,
    MediaSinkProtocol,
    MediaSourceProtocol,
)


class SinkableSource(
    MediaSourceProtocol[AudioData, AudioMetadata],
    MediaSinkProtocol[AudioMetadata, AudioData],
):
    """Implements both Source and Sink protocols using an internal queue."""

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer: Queue[MediaChunk[AudioData, AudioMetadata]] = Queue(
            maxsize=buffer_size
        )
        self._is_open = False
        # Add a condition variable to signal when data arrives
        self._data_available = threading.Condition()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> bool:
        self._is_open = True
        return True

    def close(self) -> None:
        self._is_open = False
        with self._data_available:
            # Wake up anyone waiting in poll() so they can exit
            self._data_available.notify_all()

        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break

    def send(self, topic: bytes, metadata: AudioMetadata, data: AudioData) -> None:
        if not self._is_open:
            return

        chunk = MediaChunk(metadata=metadata, data=data)

        try:
            self._buffer.put_nowait(chunk)
        except Full:
            try:
                self._buffer.get_nowait()  # Drop oldest
                self._buffer.put_nowait(chunk)
            except (Empty, Full):
                pass

        with self._data_available:
            self._data_available.notify()

    def read(self) -> Optional[MediaChunk[AudioData, AudioMetadata]]:
        if not self._is_open:
            return None
        try:
            return self._buffer.get_nowait()
        except Empty:
            return None

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        if not self._buffer.empty():
            return True

        with self._data_available:
            if not self._buffer.empty():
                return True

            self._data_available.wait(timeout=timeout_ms / 1000.0)

            return not self._buffer.empty()
