"""Real-artifact integration tests for the complete inference pipeline.

These tests intentionally use the checked-in Battery, IMU, and Road model
artifacts. They do not require an ESP32, MongoDB, or Groq.
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.schemas import BatteryFeatureVector, IMUPredictionRequest, IMUSample
from app.main import app
from app.services.attribution_service import attribution_engine
from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu
from app.services.model_loader import get_battery_model_info, get_imu_model_info
from app.services.road_service import MODEL_PATH, road_service


def battery_payload() -> dict[str, float]:
    return {
        "voltage_initial": 4.2, "voltage_final": 4.1, "voltage_min": 4.05,
        "voltage_max": 4.2, "voltage_mean": 4.15, "voltage_std": 0.02,
        "voltage_drop": 0.1, "current_mean": 0.1, "current_min": 0.05,
        "current_max": 0.2, "current_std": 0.03, "temperature_initial": 20.0,
        "temperature_final": 21.0, "temperature_max": 22.0,
        "temperature_mean": 20.5, "temperature_rise": 1.0,
        "load_current_abs_mean": 0.1, "load_current_abs_max": 0.2,
        "load_voltage_mean": 4.15, "load_voltage_min": 4.05,
        "discharge_duration": 5000.0,
    }


def imu_payload() -> list[dict[str, float]]:
    return [
        {"AccX": 0.56, "AccY": -0.02, "AccZ": 9.47, "GyroX": -0.20, "GyroY": -0.51, "GyroZ": 0.14},
        {"AccX": -0.24, "AccY": 0.08, "AccZ": -0.21, "GyroX": 0.18, "GyroY": 0.04, "GyroZ": -0.69},
        {"AccX": 0.10, "AccY": -0.07, "AccZ": 10.17, "GyroX": 0.20, "GyroY": -0.19, "GyroZ": -0.21},
        {"AccX": -0.01, "AccY": -0.09, "AccZ": 10.15, "GyroX": -0.04, "GyroY": -0.32, "GyroZ": -0.22},
    ]


def audio_payload() -> list[float]:
    return [0.1 * math.sin(2 * math.pi * 3000 * index / 16000) for index in range(1600)]


def combined_payload() -> dict:
    return {"battery_features": battery_payload(), "imu_samples": imu_payload(), "audio_samples": audio_payload()}


def test_model_artifacts_are_present_and_loaded():
    battery_info = get_battery_model_info()
    imu_info = get_imu_model_info()
    assert battery_info["loaded"] is True
    assert battery_info["feature_count"] == 21
    assert imu_info["loaded"] is True
    assert imu_info["feature_count"] == 42
    assert Path(MODEL_PATH).is_file()
    assert road_service.loaded is True


def test_real_battery_model_output_is_finite_and_in_range():
    result = predict_battery_soh(BatteryFeatureVector(**battery_payload()))
    assert 0.0 <= result["soh"] <= 1.5
    assert 0.0 <= result["soh_percent"] <= 150.0
    assert result["input_feature_count"] == 21


def test_real_imu_model_output_has_expected_contract():
    request = IMUPredictionRequest(samples=[IMUSample(**sample) for sample in imu_payload()])
    result = predict_imu(request)
    assert result["predicted_class"] in {1, 2, 3, 4}
    assert result["feature_count"] == 42
    assert result["window_size"] == 4


def test_real_road_model_output_has_valid_probabilities():
    result = road_service.predict(audio_payload())
    assert result["road_class"] in {"smooth", "moderately_rough", "rough", "impact"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == set(road_service.get_model_info()["classes"])
    assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-5)


def test_combined_fusion_propagates_all_model_results():
    from app.api.schemas import CombinedInferenceRequest

    result = attribution_engine.process_inference(CombinedInferenceRequest(**combined_payload()))
    assert set(result) == {"battery_state", "driving_behavior", "road_condition", "attribution", "protection"}
    assert 0.0 <= result["battery_state"]["soh_percent"] <= 150.0
    assert result["driving_behavior"]["raw_class_id"] in {1, 2, 3, 4}
    assert result["road_condition"]["class"] in {"smooth", "moderately_rough", "rough", "impact"}
    assert result["attribution"]["likely_cause"] in {"normal", "driver_induced", "road_induced", "battery_degraded", "mixed_or_high_stress"}
    assert 0 <= result["protection"]["pwm_limit_pct"] <= 100


def test_http_endpoints_and_live_websocket_use_the_same_backend_results():
    with TestClient(app) as client:
        combined = client.post("/api/v1/inference", json=combined_payload())
        assert combined.status_code == 200, combined.text
        live_battery = client.post("/api/v1/live/battery", json=battery_payload())
        live_imu = client.post("/api/v1/live/imu", json={"samples": imu_payload()})
        live_road = client.post("/api/v1/live/road", json={"samples": audio_payload()})
        assert all(response.status_code == 200 for response in (live_battery, live_imu, live_road))
        snapshot = client.get("/api/v1/live/snapshot").json()
        fused = client.get("/api/v1/live/inference")
        assert fused.status_code == 200
        assert snapshot["fused"]["attribution"] == fused.json()["attribution"]
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket_snapshot = websocket.receive_json()
            assert websocket_snapshot["fused"]["protection"] == fused.json()["protection"]
