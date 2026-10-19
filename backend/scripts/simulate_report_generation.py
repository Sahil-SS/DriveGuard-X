"""Generate a driver report from real model evidence.

This makes one Groq request when ``GROQ_API_KEY`` is configured in ``.env``.
It never prints the API key. Run from ``backend``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.schemas import BatteryFeatureVector, CombinedInferenceRequest, IMUPredictionRequest, IMUSample
from app.core.config import settings
from app.services.attribution_service import attribution_engine
from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu
from app.services.report_service import build_evidence, generate_report
from app.services.road_service import road_service


def main() -> int:
    battery = {"voltage_initial": 4.2, "voltage_final": 4.1, "voltage_min": 4.05, "voltage_max": 4.2, "voltage_mean": 4.15, "voltage_std": 0.02, "voltage_drop": 0.1, "current_mean": 0.1, "current_min": 0.05, "current_max": 0.2, "current_std": 0.03, "temperature_initial": 20.0, "temperature_final": 21.0, "temperature_max": 22.0, "temperature_mean": 20.5, "temperature_rise": 1.0, "load_current_abs_mean": 0.1, "load_current_abs_max": 0.2, "load_voltage_mean": 4.15, "load_voltage_min": 4.05, "discharge_duration": 5000.0}
    values = [(0.56, -0.02, 9.47, -0.20, -0.51, 0.14), (-0.24, 0.08, -0.21, 0.18, 0.04, -0.69), (0.10, -0.07, 10.17, 0.20, -0.19, -0.21), (-0.01, -0.09, 10.15, -0.04, -0.32, -0.22)]
    imu = [{"AccX": a, "AccY": b, "AccZ": c, "GyroX": d, "GyroY": e, "GyroZ": f} for a, b, c, d, e, f in values]
    audio = [0.1 * math.sin(2 * math.pi * 3000 * index / 16000) for index in range(1600)]
    battery_input = BatteryFeatureVector(**battery)
    imu_input = IMUPredictionRequest(samples=[IMUSample(**sample) for sample in imu])
    result = attribution_engine.process_inference(CombinedInferenceRequest(battery_features=battery_input, imu_samples=imu_input.samples, audio_samples=audio))
    timestamp = datetime.now(timezone.utc)
    session = {"session_id": "simulation", "started_at": timestamp, "ended_at": timestamp, "snapshots": [{**result, "timestamp": timestamp, "sequence": 1}]}
    print(f"Groq configured: {bool(settings.groq_api_key)}")
    print(f"Groq model: {settings.groq_model}")
    print("Evidence:")
    print(json.dumps(build_evidence(session), indent=2, default=str))
    report = asyncio.run(generate_report(session))
    print("Driver report:")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
