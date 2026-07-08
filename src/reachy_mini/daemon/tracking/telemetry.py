"""Telemetry storage and replay helpers for visual servo tracking."""

from __future__ import annotations

import copy
import json
import math
import threading
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt

JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def finite_json_value(value: Any) -> JsonValue:
    """Convert nested Python/NumPy values to finite JSON-compatible values."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return finite_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [finite_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json_value(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class TelemetryQuery:
    """Filter and bound telemetry records."""

    from_timestamp: float | None = None
    to_timestamp: float | None = None
    from_sequence: int | None = None
    to_sequence: int | None = None
    limit: int = 1000

    def validate(self, max_limit: int = 5000) -> None:
        """Validate query bounds."""
        for value in (self.from_timestamp, self.to_timestamp):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("timestamp filters must be finite real numbers")
            if not math.isfinite(value):
                raise ValueError("timestamp filters must be finite real numbers")
        if self.from_timestamp is not None and self.to_timestamp is not None:
            from_timestamp = float(self.from_timestamp)
            to_timestamp = float(self.to_timestamp)
        else:
            from_timestamp = None
            to_timestamp = None
        if (
            from_timestamp is not None
            and to_timestamp is not None
            and from_timestamp > to_timestamp
        ):
            raise ValueError("from timestamp must be <= to timestamp")
        if self.from_sequence is not None and (
            isinstance(self.from_sequence, bool)
            or not isinstance(self.from_sequence, int)
        ):
            raise ValueError("from_sequence must be an integer")
        if self.to_sequence is not None and (
            isinstance(self.to_sequence, bool) or not isinstance(self.to_sequence, int)
        ):
            raise ValueError("to_sequence must be an integer")
        if self.from_sequence is not None and self.from_sequence < 0:
            raise ValueError("from_sequence must be non-negative")
        if self.to_sequence is not None and self.to_sequence < 0:
            raise ValueError("to_sequence must be non-negative")
        if (
            self.from_sequence is not None
            and self.to_sequence is not None
            and self.from_sequence > self.to_sequence
        ):
            raise ValueError("from_sequence must be <= to_sequence")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("limit must be an integer")
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.limit > max_limit:
            raise ValueError(f"limit must be <= {max_limit}")


class VisualServoTelemetryBuffer:
    """Thread-safe bounded telemetry ring buffer."""

    def __init__(self, capacity: int = 3000) -> None:
        """Initialize the buffer."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: deque[dict[str, JsonValue]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._dropped_records = 0

    def append(self, record: dict[str, Any]) -> None:
        """Append one record, copying it into JSON-compatible storage."""
        json_record = cast(dict[str, JsonValue], finite_json_value(record))
        with self._lock:
            if len(self._records) == self.capacity:
                self._dropped_records += 1
            self._records.append(json_record)

    def latest(self) -> dict[str, JsonValue] | None:
        """Return an isolated copy of only the newest record in constant time."""
        with self._lock:
            if not self._records:
                return None
            return copy.deepcopy(self._records[-1])

    def query(
        self,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        from_sequence: int | None = None,
        to_sequence: int | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Return a filtered telemetry snapshot."""
        query = TelemetryQuery(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            from_sequence=from_sequence,
            to_sequence=to_sequence,
            limit=limit,
        )
        query.validate()
        with self._lock:
            records = [copy.deepcopy(record) for record in self._records]
            dropped_records = self._dropped_records

        retained_timestamps: list[float] = []
        retained_sequences: list[int] = []
        for record in records:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, int | float):
                retained_timestamps.append(float(timestamp))
            sequence = record.get("sequence")
            if isinstance(sequence, int):
                retained_sequences.append(int(sequence))

        filtered = [record for record in records if _record_matches(record, query)][
            : query.limit
        ]

        return {
            "records": filtered,
            "dropped_records": dropped_records,
            "oldest_timestamp": min(retained_timestamps)
            if retained_timestamps
            else None,
            "newest_timestamp": max(retained_timestamps)
            if retained_timestamps
            else None,
            "oldest_sequence": min(retained_sequences) if retained_sequences else None,
            "newest_sequence": max(retained_sequences) if retained_sequences else None,
            "returned": len(filtered),
            "limit": query.limit,
        }


def _record_matches(record: dict[str, JsonValue], query: TelemetryQuery) -> bool:
    timestamp = record.get("timestamp")
    sequence = record.get("sequence")
    if not isinstance(timestamp, int | float):
        return False
    if not isinstance(sequence, int):
        return False
    if query.from_timestamp is not None and timestamp < query.from_timestamp:
        return False
    if query.to_timestamp is not None and timestamp > query.to_timestamp:
        return False
    if query.from_sequence is not None and sequence < query.from_sequence:
        return False
    if query.to_sequence is not None and sequence > query.to_sequence:
        return False
    return True


def dump_jsonl(records: Iterable[dict[str, JsonValue]], path: Path) -> None:
    """Write telemetry records as JSONL."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json_record = cast(dict[str, JsonValue], finite_json_value(record))
            stream.write(
                json.dumps(json_record, separators=(",", ":"), allow_nan=False)
            )
            stream.write("\n")


def load_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    """Load telemetry records from JSONL."""
    records: list[dict[str, JsonValue]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(
                    line,
                    parse_constant=_reject_non_finite_json_constant,
                )
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                records.append(cast(dict[str, JsonValue], finite_json_value(value)))
    return records


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def summarize_records(records: list[dict[str, JsonValue]]) -> dict[str, Any]:
    """Summarize telemetry records for offline inspection."""
    reason_counts = Counter(
        str(record.get("reason"))
        for record in records
        if record.get("reason") is not None
    )
    timestamps: list[float] = []
    sequences: list[int] = []
    for record in records:
        timestamp = record.get("timestamp")
        if isinstance(timestamp, int | float):
            timestamps.append(float(timestamp))
        sequence = record.get("sequence")
        if isinstance(sequence, int):
            sequences.append(int(sequence))
    latencies = []
    for record in records:
        latency = record.get("latency")
        if isinstance(latency, dict):
            duration = latency.get("processing_duration")
            if isinstance(duration, int | float):
                latencies.append(float(duration))

    profiled_command_smoothness = _summarize_command_smoothness(
        records, "profiled_command"
    )
    final_command_smoothness = _summarize_command_smoothness(records, "final_command")
    return {
        "total_records": len(records),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
        "first_sequence": min(sequences) if sequences else None,
        "last_sequence": max(sequences) if sequences else None,
        "reason_counts": dict(reason_counts),
        "command_count": sum(
            isinstance(record.get("final_command"), list) for record in records
        ),
        "ik_failure_count": sum(1 for record in records if record.get("ik_failed")),
        "limit_hit_count": _count_limit_hits(records),
        "latency": _summarize_numbers(latencies),
        "profiled_command_smoothness": profiled_command_smoothness,
        "final_command_smoothness": final_command_smoothness,
    }


def _count_limit_hits(records: list[dict[str, JsonValue]]) -> int:
    total = 0
    for record in records:
        limit_hits = record.get("limit_hits")
        if isinstance(limit_hits, list):
            total += len(limit_hits)
    return total


def _summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _summarize_command_smoothness(
    records: list[dict[str, JsonValue]],
    command_field: str,
) -> dict[str, float | None]:
    samples: list[tuple[float, npt.NDArray[np.float64]]] = []
    command_length: int | None = None
    for record in records:
        monotonic_timestamp = record.get("monotonic_timestamp")
        timestamp = (
            monotonic_timestamp
            if isinstance(monotonic_timestamp, int | float)
            and math.isfinite(monotonic_timestamp)
            else record.get("timestamp")
        )
        command = record.get(command_field)
        if isinstance(timestamp, int | float) and isinstance(command, list):
            numeric_command: list[float] = []
            for item in command:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, int | float)
                    or not math.isfinite(item)
                ):
                    break
                numeric_command.append(float(item))
            else:
                if not numeric_command:
                    continue
                if command_length is None:
                    command_length = len(numeric_command)
                if len(numeric_command) != command_length:
                    continue
                samples.append(
                    (float(timestamp), np.array(numeric_command, dtype=np.float64))
                )
    if len(samples) < 2:
        return {"max_velocity": None, "max_acceleration": None, "max_jerk": None}
    velocities = _differentiate(samples)
    accelerations = _differentiate(velocities)
    jerks = _differentiate(accelerations)
    return {
        "max_velocity": _max_norm(velocities),
        "max_acceleration": _max_norm(accelerations),
        "max_jerk": _max_norm(jerks),
    }


def _differentiate(
    samples: list[tuple[float, npt.NDArray[np.float64]]],
) -> list[tuple[float, npt.NDArray[np.float64]]]:
    result: list[tuple[float, npt.NDArray[np.float64]]] = []
    for (prev_time, prev_value), (now, value) in zip(samples, samples[1:]):
        dt = now - prev_time
        if dt > 0.0:
            result.append((now, (value - prev_value) / dt))
    return result


def _max_norm(samples: list[tuple[float, npt.NDArray[np.float64]]]) -> float | None:
    if not samples:
        return None
    return max(float(np.max(np.abs(value))) for _, value in samples)
