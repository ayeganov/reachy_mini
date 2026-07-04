"""Small HTTP client shared by Reachy Mini command-line tools."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Mapping
from typing import Any


class RobotApi:
    """Call the daemon HTTP API and perform common robot lifecycle operations."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        """Store the normalized API URL and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        """Return one JSON object from a GET endpoint."""
        return self.request("GET", path)

    def post(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one JSON object from a POST endpoint."""
        return self.request("POST", path, payload)

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request and require a JSON object response."""
        result = self.request_value(method, path, payload)
        if not isinstance(result, dict):
            raise ValueError(f"{path} did not return a JSON object")
        return result

    def request_value(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        """Send an HTTP request and return its decoded JSON value."""
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        if not body:
            return {}
        return json.loads(body)

    def return_neutral(self, duration: float = 2.0) -> None:
        """Move the head, body, and antennas to neutral and wait for completion."""
        move = self.post(
            "/move/goto",
            {
                "head_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
                "antennas": [0.0, 0.0],
                "body_yaw": 0.0,
                "duration": duration,
                "interpolation": "minjerk",
            },
        )
        uuid = move.get("uuid")
        deadline = time.monotonic() + max(self.timeout, duration + 5.0)
        while isinstance(uuid, str) and time.monotonic() < deadline:
            moves = self.request_value("GET", "/move/running")
            if not isinstance(moves, list):
                break
            if not any(
                item.get("uuid") == uuid for item in moves if isinstance(item, dict)
            ):
                break
            time.sleep(0.1)
