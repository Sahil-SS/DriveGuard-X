"""Durable MongoDB session storage for fused backend inference events."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.core.config import settings


class SessionStore:
    def __init__(self) -> None:
        self._client = None
        self._db = None

    def _database(self):
        if self._db is None:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError as exc:
                raise RuntimeError("MongoDB recording requires the motor package") from exc
            self._client = AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=2000)
            self._db = self._client[settings.mongodb_database]
        return self._db

    async def ensure_indexes(self) -> None:
        db = self._database()
        await db.sessions.create_index("started_at")
        await db.sessions.create_index("status")
        await db.sessions.create_index("session_id", unique=True)

    async def start(self) -> dict[str, Any]:
        db = self._database()
        await self.ensure_indexes()
        active = await db.sessions.find_one({"status": "recording"}, {"session_id": 1})
        if active:
            raise ValueError(f"An active session already exists: {active['session_id']}")
        doc = {
            "session_id": uuid4().hex,
            "started_at": datetime.now(timezone.utc),
            "ended_at": None,
            "status": "recording",
            "snapshots": [],
            "events": [],
            "snapshot_count": 0,
            "report_status": "not_requested",
            "report": None,
        }
        await db.sessions.insert_one(doc)
        return doc

    async def active(self) -> dict[str, Any] | None:
        return await self._database().sessions.find_one({"status": "recording"})

    async def append_snapshot(self, snapshot: dict[str, Any]) -> None:
        db = self._database()
        active = await db.sessions.find_one({"status": "recording"}, {"session_id": 1})
        if not active:
            return
        await db.sessions.update_one(
            {"session_id": active["session_id"], "status": "recording", "snapshot_count": {"$lt": settings.recording_max_snapshots}},
            {"$push": {"snapshots": snapshot}, "$inc": {"snapshot_count": 1}},
        )

    async def stop(self, session_id: str) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        result = await self._database().sessions.find_one_and_update(
            {"session_id": session_id, "status": "recording"},
            {"$set": {"status": "finalized", "ended_at": now, "report_status": "pending"}},
            return_document=1,
        )
        return result

    async def list(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        db = self._database()
        total = await db.sessions.count_documents({})
        cursor = db.sessions.find({}, {"snapshots": 0, "events": 0, "report": 0}).sort("started_at", -1).skip(offset).limit(limit)
        return await cursor.to_list(length=limit), total

    async def get(self, session_id: str) -> dict[str, Any] | None:
        return await self._database().sessions.find_one({"session_id": session_id})

    async def set_report(self, session_id: str, status: str, report: dict[str, Any] | None = None, error: str | None = None) -> None:
        fields: dict[str, Any] = {"report_status": status}
        if report is not None:
            fields.update({"report": report, "report_generated_at": datetime.now(timezone.utc)})
        if error:
            fields["report_error"] = error
        await self._database().sessions.update_one({"session_id": session_id}, {"$set": fields})


session_store = SessionStore()
