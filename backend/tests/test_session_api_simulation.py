"""Session lifecycle simulation without requiring an external MongoDB server."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app


class FakeSessionStore:
    def __init__(self):
        self.doc = None

    async def start(self):
        if self.doc and self.doc["status"] == "recording":
            raise ValueError("An active session already exists: abc123")
        self.doc = {"session_id": "a" * 32, "started_at": datetime.now(timezone.utc), "ended_at": None, "status": "recording", "snapshots": [], "events": [], "snapshot_count": 0, "report_status": "not_requested", "report": None}
        return self.doc

    async def append_snapshot(self, snapshot):
        if self.doc and self.doc["status"] == "recording":
            self.doc["snapshots"].append(snapshot)
            self.doc["snapshot_count"] += 1

    async def stop(self, session_id):
        if not self.doc or self.doc["session_id"] != session_id or self.doc["status"] != "recording":
            return None
        self.doc["status"] = "finalized"
        self.doc["ended_at"] = datetime.now(timezone.utc)
        self.doc["report_status"] = "pending"
        return self.doc

    async def list(self, limit, offset):
        return ([self.doc] if self.doc else [], 1 if self.doc else 0)

    async def get(self, session_id):
        return self.doc if self.doc and self.doc["session_id"] == session_id else None

    async def set_report(self, session_id, status, report=None, error=None):
        self.doc["report_status"] = status
        if report is not None:
            self.doc["report"] = report
        if error:
            self.doc["report_error"] = error


def test_start_stop_history_and_report_lifecycle(monkeypatch):
    fake = FakeSessionStore()
    monkeypatch.setattr(routes, "session_store", fake)

    async def fake_report(_session):
        return {"session_overview": {"snapshot_count": 1}, "limitations": []}

    monkeypatch.setattr(routes, "generate_report", fake_report)
    with TestClient(app) as client:
        started = client.post("/api/v1/sessions")
        assert started.status_code == 200
        session_id = started.json()["session_id"]
        awaitable_snapshot = {"timestamp": datetime.now(timezone.utc), "battery_state": {}, "driving_behavior": {}, "road_condition": {}, "attribution": {}, "protection": {}, "sequence": 1, "source_timestamps": {}, "available_modalities": []}
        import asyncio
        asyncio.run(fake.append_snapshot(awaitable_snapshot))
        assert client.get("/api/v1/sessions").json()["items"][0]["snapshot_count"] == 1
        stopped = client.post(f"/api/v1/sessions/{session_id}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "finalized"
        detail = client.get(f"/api/v1/sessions/{session_id}")
        assert detail.status_code == 200
        assert len(detail.json()["snapshots"]) == 1
        assert client.post(f"/api/v1/sessions/{session_id}/stop").status_code == 404
