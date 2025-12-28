"""A sink/source hybrid for in-memory queuing."""

from __future__ import annotations

from queue import Empty, Full, Queue
from typing import Optional

from reachy_mini.media.publishers.base import (
    AudioData,
    MediaChunk,
    MediaSinkProtocol,
    MediaSourceProtocol,
)

from reachy_mini.media.capture import AudioMetadata


class SinkableSource(
    MediaSourceProtocol[AudioData, AudioMetadata],
    MediaSinkProtocol[AudioMetadata, AudioData],
):
    """Implements both Source and Sink protocols using an internal queue.

    This allows it to be a target for writing data (as a sink) and a
    source for reading data for another component (like a publisher),
    effectively acting as an in-memory, thread-safe buffer.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        """Initialize the sinkable source.

        Args:
            buffer_size (int): Max number of chunks to buffer before dropping.
        """
        self._buffer: Queue[MediaChunk[AudioData, AudioMetadata]] = Queue(
            maxsize=buffer_size
        )
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if the source/sink is active."""
        return self._is_open

    def open(self) -> bool:
        """Open the source/sink."""
        self._is_open = True
        return True

    def close(self) -> None:
        """Close the source/sink and clear the buffer."""
        self._is_open = False
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break

    def send(self, topic: bytes, metadata: AudioMetadata, data: AudioData) -> None:
        """Create a MediaChunk and push it into the source's queue.
        The topic is ignored as it's not relevant for an in-memory queue.
        """
        if not self._is_open:
            return

        chunk = MediaChunk(metadata=metadata, data=data)
        try:
            self._buffer.put_nowait(chunk)
        except Full:
            try:
                # Make space by dropping the oldest item
                self._buffer.get_nowait()
                self._buffer.put_nowait(chunk)
            except (Empty, Full):
                pass

    def read(self) -> Optional[MediaChunk[AudioData, AudioMetadata]]:
        """Read a chunk from the queue (non-blocking)."""
        if not self._is_open:
            return None
        try:
            return self._buffer.get_nowait()
        except Empty:
            return None

    def poll(self, timeout_ms: int = 100) -> bool:
        """Check if data is available in the queue."""
        return not self._buffer.empty()
