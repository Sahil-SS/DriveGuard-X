"""In-memory latest telemetry for the prototype dashboard."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


class LiveTelemetryStore:
    """Store the latest processed reading from each sensing subsystem.

    This is intentionally process-local. A production deployment should use a
    device gateway and a shared store/message broker instead.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: dict[str, Any] = {
            "battery": None,
            "imu": None,
            "road": None,
        }

    def update(self, source: str, reading: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot[source] = deepcopy(reading)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._snapshot)


live_telemetry = LiveTelemetryStore()
