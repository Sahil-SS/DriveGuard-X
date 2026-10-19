from __future__ import annotations

import asyncio
import json
from typing import Any

from app.core.config import settings


def build_evidence(session: dict[str, Any]) -> dict[str, Any]:
    snapshots = session.get("snapshots", [])
    causes: dict[str, int] = {}
    roads: dict[str, int] = {}
    behaviors: dict[str, int] = {}
    protection_actions: dict[str, int] = {}
    battery_soh_values: list[float] = []
    pwm_limits: list[float] = []
    stressed = 0
    events = []
    previous = None
    for item in snapshots:
        cause = item.get("attribution", {}).get("likely_cause", "unknown")
        road = item.get("road_condition", {}).get("class", "unknown")
        behavior = item.get("driving_behavior", {}).get("class", "unknown")
        protection = item.get("protection", {})
        causes[cause] = causes.get(cause, 0) + 1
        roads[road] = roads.get(road, 0) + 1
        behaviors[behavior] = behaviors.get(behavior, 0) + 1
        action = protection.get("action", "unknown")
        protection_actions[action] = protection_actions.get(action, 0) + 1
        if item.get("battery_state", {}).get("soh_percent") is not None:
            battery_soh_values.append(float(item["battery_state"]["soh_percent"]))
        if protection.get("pwm_limit_pct") is not None:
            pwm_limits.append(float(protection["pwm_limit_pct"]))
        if item.get("battery_state", {}).get("stressed"):
            stressed += 1
        if previous and cause != previous:
            events.append({"timestamp": item.get("timestamp"), "type": "cause_change", "from": previous, "to": cause})
        previous = cause
    return {
        "snapshot_count": len(snapshots),
        "stressed_snapshot_count": stressed,
        "battery_soh_percent": {"min": min(battery_soh_values) if battery_soh_values else None, "max": max(battery_soh_values) if battery_soh_values else None, "latest": battery_soh_values[-1] if battery_soh_values else None},
        "cause_counts": causes,
        "road_counts": roads,
        "behavior_counts": behaviors,
        "protection_actions": protection_actions,
        "pwm_limit_percent": {"min": min(pwm_limits) if pwm_limits else None, "max": max(pwm_limits) if pwm_limits else None, "latest": pwm_limits[-1] if pwm_limits else None},
        "representative_snapshots": snapshots[-10:],
        "events": events[-50:],
    }


async def generate_report(session: dict[str, Any]) -> dict[str, Any]:
    evidence = build_evidence(session)
    if not settings.groq_api_key:
        return deterministic_report(session, evidence, "Groq is not configured; this report contains deterministic evidence only.")
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        prompt = "Return JSON only with keys session_overview,battery_analysis,driving_behavior_analysis,road_condition_analysis,cause_attribution_summary,significant_events,protection_summary,recommendations,limitations. Use only this evidence; never invent values. Separate observed facts from recommendations. Evidence: " + json.dumps(evidence, default=str)
        response = await asyncio.to_thread(client.chat.completions.create, model=settings.groq_model, messages=[{"role": "system", "content": "You are a careful vehicle telemetry analyst."}, {"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"}, timeout=settings.report_timeout_seconds)
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Groq returned a non-object report")
        return parsed
    except Exception as exc:
        return deterministic_report(session, evidence, f"Groq report unavailable: {type(exc).__name__}")


def deterministic_report(session: dict[str, Any], evidence: dict[str, Any], limitation: str) -> dict[str, Any]:
    return {
        "session_overview": {"session_id": session.get("session_id"), "started_at": session.get("started_at"), "ended_at": session.get("ended_at"), "snapshot_count": evidence["snapshot_count"]},
        "battery_analysis": {"stressed_snapshot_count": evidence["stressed_snapshot_count"]},
        "driving_behavior_analysis": evidence["behavior_counts"],
        "road_condition_analysis": evidence["road_counts"],
        "cause_attribution_summary": evidence["cause_counts"],
        "significant_events": evidence["events"],
        "protection_summary": "Use the recorded protection fields in the session snapshots.",
        "recommendations": [],
        "limitations": [limitation, "Only backend inference snapshots are included."]
    }
