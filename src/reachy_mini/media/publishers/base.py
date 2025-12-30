"""Base classes and protocols for media publishers.

Publishers subscribe to the IPC bus and republish media over network protocols.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import (
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

# Generic type variables for media protocols
DataType = TypeVar("DataType", bound=object, contravariant=True)
MetaType = TypeVar("MetaType", bound=object, contravariant=True)

# Concrete type aliases for convenience
VideoData = npt.NDArray[np.uint8]
AudioData = npt.NDArray[np.float32]


@dataclass
class MediaChunk(Generic[DataType, MetaType]):
    """Container for media data and its metadata."""

    data: DataType
    metadata: MetaType


class MediaSourceProtocol(Protocol[DataType, MetaType]):
    """Protocol for media data sources."""

    def open(self) -> bool:
        """Initialize the source."""
        ...

    def read(self) -> Optional[MediaChunk[DataType, MetaType]]:
        """Read next chunk. Returns None if no data available (non-blocking)."""
        ...

    def poll(self, timeout_ms: int = 100) -> bool:
        """Wait for data to be available."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    @property
    def is_open(self) -> bool:
        """Check if source is open and ready."""
        ...


class MediaSinkProtocol(Protocol[MetaType, DataType]):
    """Protocol for media transport sinks.

    Handlers bind vs connect internally based on implementation.
    """

    def open(self) -> bool:
        """Initialize the sink (bind or connect as appropriate)."""
        ...

    def send(self, topic: bytes, metadata: MetaType, data: DataType) -> None:
        """Send encoded data."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    @property
    def is_open(self) -> bool:
        """Check if sink is open and ready."""
        ...


@runtime_checkable
class MediaPublisherProtocol(Protocol):
    """Protocol defining the interface for media publishers."""

    @property
    def is_running(self) -> bool:
        """Check if publisher is running."""
        ...

    def start(self) -> bool:
        """Start the publisher.

        Returns:
            True if started successfully, False otherwise.
        """
        ...

    def stop(self) -> None:
        """Stop the publisher and release resources."""
        ...


@dataclass
class PublisherConfig:
    """Configuration for media publishers."""

    video_ipc_endpoint: str = "ipc:///tmp/reachy_video"
    audio_ipc_endpoint: str = "ipc:///tmp/reachy_audio"
    video_enabled: bool = True
    audio_enabled: bool = True
    log_level: str = "INFO"


class GenericMediaPublisher(Generic[DataType, MetaType]):
    """Generic publisher that moves data from a Source to a Sink."""

    def __init__(
        self,
        source: MediaSourceProtocol[DataType, MetaType],
        sink: MediaSinkProtocol[MetaType, DataType],
        topic: bytes,
        log_level: str = "INFO",
        name: str = "GenericMediaPublisher",
    ) -> None:
        """Initialize publisher.

        Args:
            source: Source protocol implementation.
            sink: Sink protocol implementation.
            topic: Topic bytes for the sink.
            log_level: Logging level.
            name: Name for the publisher thread.
        """
        self._source = source
        self._sink = sink
        self._topic = topic
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(log_level)
        self._name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._item_count = 0

    @property
    def is_running(self) -> bool:
        """Check if publisher is running."""
        return self._running

    def start(self) -> bool:
        """Start the publisher."""
        if self._running:
            self._logger.warning("Publisher already running")
            return True

        try:
            if not self._source.open():
                self._logger.error("Failed to open source")
                return False

            if not self._sink.open():
                self._logger.error("Failed to open sink")
                self._source.close()
                return False

            self._running = True
            self._item_count = 0
            self._thread = threading.Thread(
                target=self._loop,
                name=self._name,
                daemon=True,
            )
            self._thread.start()
            self._logger.info("%s started", self._name)
            return True

        except Exception as e:
            self._logger.error("Failed to start publisher: %s", e)
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the publisher."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._source.close()
        self._sink.close()
        self._logger.info("%s stopped (items: %d)", self._name, self._item_count)

    def _loop(self) -> None:
        """Run the main publish loop."""
        while self._running:
            if not self._source.poll(timeout_ms=100):
                continue

            chunk = self._source.read()
            if chunk is None:
                continue

            try:
                self._sink.send(
                    self._topic,
                    chunk.metadata,
                    chunk.data,
                )
                self._item_count += 1

            except Exception as e:
                self._logger.error("Error in publish loop: %s", e)
