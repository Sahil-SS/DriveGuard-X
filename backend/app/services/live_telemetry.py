"""In-memory latest telemetry for the prototype dashboard."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any
from datetime import datetime, timezone


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

    def update(self, source: str, reading: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self._snapshot[source] = deepcopy(reading)
            if all(self._snapshot[key] is not None for key in ("battery", "imu", "road")):
                from app.services.attribution_service import attribution_engine
                fused = attribution_engine.fuse_results(
                    self._snapshot["battery"], self._snapshot["imu"], self._snapshot["road"]
                )
                fused["timestamp"] = datetime.now(timezone.utc)
                fused["sequence"] = int(self._snapshot.get("sequence", 0)) + 1
                fused["source_timestamps"] = {
                    key: self._snapshot[key].get("timestamp") for key in ("battery", "imu", "road")
                }
                fused["available_modalities"] = [key for key in ("battery", "imu", "road") if self._snapshot[key]]
                self._snapshot["fused"] = fused
                self._snapshot["sequence"] = fused["sequence"]
                return deepcopy(fused)
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._snapshot)


live_telemetry = LiveTelemetryStore()
