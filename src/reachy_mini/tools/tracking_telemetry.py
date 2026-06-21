"""Command-line tools for visual servo telemetry."""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn, cast

from reachy_mini.daemon.tracking.telemetry import (
    JsonValue,
    dump_jsonl,
    load_jsonl,
    summarize_records,
)


def format_summary(records: list[dict[str, JsonValue]]) -> str:
    """Format telemetry records as a line-oriented summary."""
    summary = summarize_records(records)
    lines = [
        f"total_records: {summary['total_records']}",
        f"first_timestamp: {summary['first_timestamp']}",
        f"last_timestamp: {summary['last_timestamp']}",
        f"first_sequence: {summary['first_sequence']}",
        f"last_sequence: {summary['last_sequence']}",
        "reason_counts: "
        + json.dumps(summary["reason_counts"], sort_keys=True, allow_nan=False),
        f"command_count: {summary['command_count']}",
        f"ik_failure_count: {summary['ik_failure_count']}",
        f"limit_hit_count: {summary['limit_hit_count']}",
        "latency: " + json.dumps(summary["latency"], sort_keys=True, allow_nan=False),
    ]
    smoothness = summary.get("command_smoothness")
    if isinstance(smoothness, dict):
        lines.extend(
            [
                f"max_velocity: {smoothness.get('max_velocity')}",
                f"max_acceleration: {smoothness.get('max_acceleration')}",
                f"max_jerk: {smoothness.get('max_jerk')}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """Run the tracking telemetry command-line interface."""
    parser = argparse.ArgumentParser(
        description="Inspect and export visual servo telemetry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump", help="Fetch telemetry and write JSONL.")
    dump_parser.add_argument(
        "--base-url",
        required=True,
        help=(
            "Tracking API base URL; the tool appends /tracking/telemetry "
            "(for example: http://host:8017/api)."
        ),
    )
    dump_parser.add_argument("--from", dest="from_timestamp", type=float)
    dump_parser.add_argument("--to", dest="to_timestamp", type=float)
    dump_parser.add_argument("--from-sequence", dest="from_sequence", type=int)
    dump_parser.add_argument("--to-sequence", dest="to_sequence", type=int)
    dump_parser.add_argument("--limit", type=int, default=1000)
    dump_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds.",
    )
    dump_parser.add_argument("--out", type=Path, required=True)

    summary_parser = subparsers.add_parser("summary", help="Summarize JSONL telemetry.")
    summary_parser.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "dump":
        records = _fetch_records(args)
        dump_jsonl(records, args.out)
    elif args.command == "summary":
        records = load_jsonl(args.path)
        print(format_summary(records))


def _fetch_records(args: argparse.Namespace) -> list[dict[str, JsonValue]]:
    params: dict[str, int | float] = {"limit": args.limit}
    if args.from_timestamp is not None:
        params["from"] = args.from_timestamp
    if args.to_timestamp is not None:
        params["to"] = args.to_timestamp
    if args.from_sequence is not None:
        params["from_sequence"] = args.from_sequence
    if args.to_sequence is not None:
        params["to_sequence"] = args.to_sequence

    url = args.base_url.rstrip("/") + "/tracking/telemetry"
    query = urllib.parse.urlencode(params)
    if query:
        url = f"{url}?{query}"

    with urllib.request.urlopen(url, timeout=args.timeout) as response:
        payload: Any = json.loads(
            response.read(),
            parse_constant=_reject_non_finite_json_constant,
        )

    if not isinstance(payload, dict):
        raise ValueError("telemetry response must be an object")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("telemetry response must include a records list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("telemetry records must be objects")
    return cast(list[dict[str, JsonValue]], records)


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")
