"""Run a visible, real-model DriveGuard-X backend simulation.

Run from ``backend``:

    .\\.venv\\Scripts\\python.exe scripts\\simulate_backend_pipeline.py

The script is intentionally independent of MongoDB and Groq. It validates the
sensor-to-model-to-fusion path and prints the exact outputs the dashboard sees.
It exits with status 1 if any model or response contract is invalid.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.schemas import BatteryFeatureVector, CombinedInferenceRequest, IMUPredictionRequest, IMUSample
from app.services.attribution_service import attribution_engine
from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu
from app.services.model_loader import get_battery_model_info, get_imu_model_info
from app.services.road_service import road_service


def battery() -> dict[str, float]:
    return {"voltage_initial": 4.2, "voltage_final": 4.1, "voltage_min": 4.05, "voltage_max": 4.2, "voltage_mean": 4.15, "voltage_std": 0.02, "voltage_drop": 0.1, "current_mean": 0.1, "current_min": 0.05, "current_max": 0.2, "current_std": 0.03, "temperature_initial": 20.0, "temperature_final": 21.0, "temperature_max": 22.0, "temperature_mean": 20.5, "temperature_rise": 1.0, "load_current_abs_mean": 0.1, "load_current_abs_max": 0.2, "load_voltage_mean": 4.15, "load_voltage_min": 4.05, "discharge_duration": 5000.0}


def imu() -> list[dict[str, float]]:
    values = [(0.56, -0.02, 9.47, -0.20, -0.51, 0.14), (-0.24, 0.08, -0.21, 0.18, 0.04, -0.69), (0.10, -0.07, 10.17, 0.20, -0.19, -0.21), (-0.01, -0.09, 10.15, -0.04, -0.32, -0.22)]
    return [{"AccX": a, "AccY": b, "AccZ": c, "GyroX": d, "GyroY": e, "GyroZ": f} for a, b, c, d, e, f in values]


def audio() -> list[float]:
    return [0.1 * math.sin(2 * math.pi * 3000 * index / 16000) for index in range(1600)]


def main() -> int:
    print("DriveGuard-X real model simulation")
    print("Battery model:", json.dumps(get_battery_model_info(), default=str))
    print("IMU model:", json.dumps(get_imu_model_info(), default=str))
    if not road_service.loaded:
        raise RuntimeError("Road model is not loaded")

    battery_input = BatteryFeatureVector(**battery())
    imu_input = IMUPredictionRequest(samples=[IMUSample(**sample) for sample in imu()])
    battery_result = predict_battery_soh(battery_input)
    imu_result = predict_imu(imu_input)
    road_result = road_service.predict(audio())
    combined = attribution_engine.process_inference(CombinedInferenceRequest(battery_features=battery_input, imu_samples=imu_input.samples, audio_samples=audio()))

    assert 0 <= battery_result["soh_percent"] <= 150
    assert imu_result["predicted_class"] in {1, 2, 3, 4}
    assert road_result["road_class"] in {"smooth", "moderately_rough", "rough", "impact"}
    assert math.isclose(sum(road_result["probabilities"].values()), 1.0, abs_tol=1e-5)
    assert combined["attribution"]["likely_cause"] in {"normal", "driver_induced", "road_induced", "battery_degraded", "mixed_or_high_stress"}
    assert 0 <= combined["protection"]["pwm_limit_pct"] <= 100
    print("Battery result:", json.dumps(battery_result, indent=2))
    print("IMU result:", json.dumps(imu_result, indent=2))
    print("Road result:", json.dumps(road_result, indent=2))
    print("Fused result:", json.dumps(combined, indent=2))
    print("SIMULATION PASS: all model and fusion contracts are valid")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SIMULATION FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
