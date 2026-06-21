# ruff: noqa: D100,D101,D102,D103,D105,D107

import argparse
import json
import urllib.parse
from pathlib import Path

import pytest

from reachy_mini.daemon.tracking.telemetry import dump_jsonl, load_jsonl
from reachy_mini.tools import tracking_telemetry
from reachy_mini.tools.tracking_telemetry import format_summary


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.payload


def test_tracking_telemetry_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        {
            "sequence": 0,
            "timestamp": 10.0,
            "reason": "commanded",
            "ik_failed": False,
            "limit_hits": [],
            "latency": {"processing_duration": 0.01},
            "final_command": [0.0, 0.0],
        },
        {
            "sequence": 1,
            "timestamp": 10.1,
            "reason": "ik_failed",
            "ik_failed": True,
            "limit_hits": [{"kind": "upper_position"}],
            "latency": {"processing_duration": 0.02},
            "final_command": None,
        },
    ]
    path = tmp_path / "telemetry.jsonl"

    dump_jsonl(records, path)

    assert load_jsonl(path) == records


def test_tracking_telemetry_summary_output_contains_required_counts() -> None:
    text = format_summary(
        [
            {
                "sequence": 0,
                "timestamp": 10.0,
                "reason": "commanded",
                "ik_failed": False,
                "limit_hits": [],
                "latency": {"processing_duration": 0.01},
                "final_command": [0.0, 0.0],
            },
            {
                "sequence": 1,
                "timestamp": 10.1,
                "reason": "commanded",
                "ik_failed": False,
                "limit_hits": [{"kind": "velocity"}],
                "latency": {"processing_duration": 0.02},
                "final_command": [0.1, 0.0],
            },
        ]
    )

    assert "total_records: 2" in text
    assert "command_count: 2" in text
    assert "ik_failure_count: 0" in text
    assert "limit_hit_count: 1" in text
    assert "max_velocity:" in text


def test_tracking_telemetry_fetch_records_writes_filtered_dump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [{"sequence": 1, "timestamp": 10.0, "reason": "commanded"}]
    captured: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(json.dumps({"records": records}).encode())

    monkeypatch.setattr(
        tracking_telemetry.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    args = argparse.Namespace(
        base_url="http://localhost:8017/api/",
        from_timestamp=10.0,
        to_timestamp=20.0,
        from_sequence=1,
        to_sequence=3,
        limit=5,
        timeout=2.5,
        out=tmp_path / "dump.jsonl",
    )

    fetched = tracking_telemetry._fetch_records(args)
    dump_jsonl(fetched, args.out)

    parsed = urllib.parse.urlparse(str(captured["url"]))
    assert parsed.path == "/api/tracking/telemetry"
    assert urllib.parse.parse_qs(parsed.query) == {
        "from": ["10.0"],
        "to": ["20.0"],
        "from_sequence": ["1"],
        "to_sequence": ["3"],
        "limit": ["5"],
    }
    assert captured["timeout"] == 2.5
    assert fetched == records
    assert load_jsonl(args.out) == records


def test_tracking_telemetry_fetch_records_rejects_non_finite_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: str, timeout: float = 10.0) -> FakeResponse:
        return FakeResponse(b'{"records":[{"value":NaN}]}')

    monkeypatch.setattr(
        tracking_telemetry.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    args = argparse.Namespace(
        base_url="http://localhost:8017/api",
        from_timestamp=None,
        to_timestamp=None,
        from_sequence=None,
        to_sequence=None,
        limit=1000,
        timeout=10.0,
    )

    with pytest.raises(ValueError, match="non-finite"):
        tracking_telemetry._fetch_records(args)
